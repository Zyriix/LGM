import xformers

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.helpers import DropPath, drop_path

# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError: pass
# automatically import faster attention implementations
try: from xformers.ops import memory_efficient_attention
except ImportError: pass
try: from flash_attn import flash_attn_func              # qkv: BLHc, ret: BLHcq
except ImportError: pass
try: from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc
except ImportError:
    def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
        with torch.cuda.amp.autocast(enabled=False):
            attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
            if attn_mask is not None: attn.add_(attn_mask)
            return (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value

"""
Modules
"""
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., fused_if_available=True):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()
    
    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(
                x=x, weight1=self.fc1.weight, weight2=self.fc2.weight, bias1=self.fc1.bias, bias2=self.fc2.bias,
                activation='gelu_approx', save_pre_act=self.training, return_residual=False, checkpoint_lvl=0,
                heuristic=0, process_group=None,
            ))
        else:
            return self.drop(self.fc2( self.act(self.fc1(x)) ))
    
    def extra_repr(self) -> str:
        return f'fused_mlp_func={self.fused_mlp_func is not None}'


class Attention(nn.Module):
    def __init__(
        self, block_idx, embed_dim=768, num_heads=12,
        attn_drop=0., proj_drop=0., attn_l2_norm=False, flash_if_available=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads  # =64
        self.attn_l2_norm = attn_l2_norm
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)
        
        self.to_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_kv = nn.Linear(embed_dim, embed_dim * 2, bias=False)

        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop: float = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None
        
        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None
    
    def kv_caching(self, enable: bool): self.caching, self.cached_k, self.cached_v = enable, None, None
    
    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, context=None, attn_bias):
        B, L, C = x.shape
        
        q = self.to_q(x)
        kv = self.to_kv(x) if context is None else self.to_kv(context)
        k,v = kv.chunk(2, dim=-1)
        main_type = qkv.dtype
        
        using_flash = self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        if using_flash or using_xform:
            dim_cat = 1
        else:
            q=q.permute(0,2,1,3)
            k=k.permute(0,2,1,3)
            v=v.permute(0,2,1,3)
            dim_cat = 2

        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform: scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
        
        if self.caching:
            if self.cached_k is None: self.cached_k = k; self.cached_v = v
            else: k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat); v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
        
        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash:
            oup = flash_attn_func(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), dropout_p=dropout_p, softmax_scale=self.scale).view(B, L, C)
        elif self.using_xform:
            oup = memory_efficient_attention(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), attn_bias=None if attn_bias is None else attn_bias.to(dtype=main_type).expand(B, self.num_heads, -1, -1), p=dropout_p, scale=self.scale).view(B, L, C)
        else:
            oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)
        
        return self.proj_drop(self.proj(oup))

    def extra_repr(self) -> str:
        return f'using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}'

class AdaLN(nn.Module):
    def __init__(
        self,embed_dim, cond_dim, shared_aln: bool, norm_layer,
    ):
        super(AdaLN, self).__init__()        
        self.ln = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.const = nn.Parameter(torch.randn(1, 3, embed_dim) / embed_dim**0.5)
            self.affine = lambda x: x.add_(self.const)
        else:
            self.affine = nn.Linear(cond_dim, 3*embed_dim)        
    
    def forward(self, x, emb):   # C: embed_dim, D: cond_dim
        x = self.ln(x)

        b,d = emb.shape
        emb = emb.view(b, 1, d)
        gamma, scale, shift = self.affine(emb).view(b,1,3*d).chunk(3, dim=-1)
        
        x = x.mul(scale.add(1)).add_(shift).mul_(gamma)

        return x
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}'

class TransformerLayer(nn.Module):
    def __init__(
        self, block_idx, last_drop_p, embed_dim, cond_dim, shared_aln: bool, norm_layer,
        num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False,
        flash_if_available=False, fused_if_available=True, cross_attn=False,
    ):
        super(TransformerLayer, self).__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn1 = Attention(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available)
        if cross_attn:
            self.cross_attn = cross_attn
            self.attn2 = Attention(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available)
        self.ffn = FFN(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=drop, fused_if_available=fused_if_available)
        
        self.adaln_1 = AdaLN(embed_dim, cond_dim, shared_aln, norm_layer)
        self.adaln_2 = AdaLN(embed_dim, cond_dim, shared_aln, norm_layer)
        self.adaln_mlp = AdaLN(embed_dim, cond_dim, shared_aln, norm_layer)
    
    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, context=None, emb=None, attn_bias=None):   # C: embed_dim, D: cond_dim

        x = x + self.drop_path(self.attn1(self.adaln_1(x, emb), attn_bias=attn_bias))

        context = self.ada_context(context, emb)
        x = x + self.drop_path(self.attn2(self.adaln_2(x, emb), context, attn_bias))
            
        x = x + self.drop_path(self.ffn(self.adaln_mlp(x, emb)))

        return x
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}'


class Encoder(nn.Module):
    def __init__(
        self, num_tokens, max_seq_len, attn_layers, emb_dropout=0.0,
        sa_layer_num=6, sa_dim=512, sa_heads=12, sa_drop=0.0, sa_l2_norm=False,
        ca_layer_num=2, ca_dim=512, ca_heads=12, ca_drop=0.0, ca_l2_norm=False,
        flash_if_available=True, fused_if_available=True,
    ):
        super(Encoder, self).__init__()
        self.token_emb = nn.linear(patch_size, sa_dim)

        self.sa_layers = nn.ModuleList([
            TransformerLayer(block_idx=i, last_drop_p=emb_dropout, embed_dim=sa_dim, cond_dim=sa_dim, shared_aln=True, norm_layer=nn.LayerNorm, num_heads=sa_heads, mlp_ratio=4., drop=sa_drop, attn_drop=sa_drop, attn_l2_norm=sa_l2_norm, flash_if_available=flash_if_available, fused_if_available=fused_if_available, rope=True, cross_attn=False)
            for i in range(sa_layer_num)
        ])

        self.to_ca = nn.sequencial(norm_layer(sa_dim), nn.Linear(sa_dim, ca_dim))

        self.ca_layers = nn.ModuleList([
            TransformerLayer(block_idx=i, last_drop_p=emb_dropout, embed_dim=ca_dim, cond_dim=ca_dim, shared_aln=True, norm_layer=nn.LayerNorm, num_heads=ca_heads, mlp_ratio=4., drop=ca_drop, attn_drop=ca_drop, attn_l2_norm=ca_l2_norm, flash_if_available=flash_if_available, fused_if_available=fused_if_available, rope=False, cross_attn=True)
            for i in range(ca_layer_num)
        ])

        self.dummy_tokens = nn.Parameter(torch.randn(1, max_seq_len, ca_dim))

        self.out_norm = norm_layer(ca_dim)
        self.act = nn.GELU(approximate='tanh')

        self.out = nn.Linear(ca_dim, ca_dim)

    def forward(self, x, cond_emb=None, attn_bias=None):   # C: embed_dim, D: cond_dim
        x = self.token_emb(x)

        # encoding spatio information
        for layer in self.sa_layers:
            x = layer(x,attn_bias)

        x = self.to_ca(x)

        # remove the spatio bias using dummy tokens
        for layer in self.ca_layers:
            x = layer(self.dummy_tokens, x, attn_bias)
        x = self.out(self.act(self.out_norm(x)))

        return x


class Decoder(nn.Module):
    def __init__(
        self, num_tokens, max_seq_len, attn_layers, emb_dropout=0.0,
        sa_layer_num=6, sa_dim=512, sa_heads=12, sa_drop=0.0, sa_l2_norm=False,
        ca_layer_num=2, ca_dim=512, ca_heads=12, ca_drop=0.0, ca_l2_norm=False,
        flash_if_available=True, fused_if_available=True,
    ):
        super(Decoder, self).__init__()
        self.token_emb = nn.linear(patch_size, sa_dim)

        self.ca_layers = nn.ModuleList([
            TransformerLayer(block_idx=i, last_drop_p=emb_dropout, embed_dim=ca_dim, cond_dim=ca_dim, shared_aln=True, norm_layer=nn.LayerNorm, num_heads=ca_heads, mlp_ratio=4., drop=ca_drop, attn_drop=ca_drop, attn_l2_norm=ca_l2_norm, flash_if_available=flash_if_available, fused_if_available=fused_if_available, rope=False, cross_attn=True)
            for i in range(ca_layer_num)
        ])

        self.to_sa = nn.sequencial(norm_layer(ca_dim), nn.Linear(ca_dim, sa_dim))

        self.sa_layers = nn.ModuleList([
            TransformerLayer(block_idx=i, last_drop_p=emb_dropout, embed_dim=sa_dim, cond_dim=sa_dim, shared_aln=True, norm_layer=nn.LayerNorm, num_heads=sa_heads, mlp_ratio=4., drop=sa_drop, attn_drop=sa_drop, attn_l2_norm=sa_l2_norm, flash_if_available=flash_if_available, fused_if_available=fused_if_available, rope=True, cross_attn=False)
            for i in range(sa_layer_num)
        ])

        self.dummy_tokens = nn.Parameter(torch.randn(1, max_seq_len, ca_dim))

        self.out_norm = norm_layer(ca_dim)
        self.act = nn.GELU(approximate='tanh')

        self.out = nn.Linear(ca_dim, ca_dim)

    def forward(self, x, cond_emb=None, attn_bias=None):   # C: embed_dim, D: cond_dim
        x = self.token_emb(x)

        # remove the spatio bias using dummy tokens
        for layer in self.ca_layers:
            x = layer(self.dummy_tokens, x, attn_bias)

        x = self.to_sa(x)

        # encoding spatio information
        for layer in self.sa_layers:
            x = layer(x,attn_bias)

        x = self.out(self.act(self.out_norm(x)))

        return x
