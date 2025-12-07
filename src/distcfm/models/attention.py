import torch
import torch.nn as nn
from distcfm.utils.flash_attention_3_jvp import flash_attn_func as fa3_func


"""RMS Layer Normalization"""
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use in-place safe variant for less memory overhead
        norm_x = torch.rsqrt(x.pow(2).mean(-1, keepdim=True, dtype=torch.float32) + self.eps)
        return x * norm_x.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype)

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.attn = fa3_func
    
    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(
            B, L, 3, self.num_heads, self.head_dim
            ).permute(2, 0, 1, 3, 4)
        # Each is of shape (B, L, H, D)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = rope(q), rope(k)
        x = self.attn(q, k, v)
        x = x.reshape(B, L, C)
        x = self.proj(x)
        return x
    

class JointAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_cond = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_cond_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.attn = fa3_func
    
    def forward(self, x: torch.Tensor, x_cond: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(
            B, L, 3, self.num_heads, self.head_dim
            ).permute(2, 0, 1, 3, 4)
        # Each is of shape (B, L, H, D)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        kv_cond = self.kv_cond(x_cond).reshape(
            B, L, 2, self.num_heads, self.head_dim
            ).permute(2, 0, 1, 3, 4)    
        k_cond, v_cond = kv_cond.unbind(0)
        k_cond = self.k_cond_norm(k_cond)
        v_cond = v_cond * alphas.reshape(B, 1, self.num_heads, self.head_dim)
        k = torch.cat([k, k_cond], dim=1)
        v = torch.cat([v, v_cond], dim=1)

        x = self.attn(q, k, v)
        x = x.reshape(B, L, C)
        x = self.proj(x)
        return x