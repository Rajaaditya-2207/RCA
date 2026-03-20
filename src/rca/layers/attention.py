"""
Efficient Attention
===================

Flash-style efficient attention with RoPE support.

Author: Rajaaditya.R
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .positions import RotaryPositionEmbedding


class EfficientAttention(nn.Module):
    """
    Flash-style efficient attention.

    Uses PyTorch 2.0's scaled_dot_product_attention when available.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rotary: bool = True,
        use_mqa: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_mqa = use_mqa

        self.q_proj = nn.Linear(dim, dim, bias=False)

        # MQA: single K/V head shared across all Q heads
        kv_dim = self.head_dim if use_mqa else dim
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout_p = dropout
        self.use_rotary = use_rotary

        if use_rotary:
            self.rotary = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_mqa:
            # Single KV head, expand to match all Q heads
            k = self.k_proj(x).view(B, S, 1, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, S, 1, self.head_dim).transpose(1, 2)
            k = k.expand(-1, self.num_heads, -1, -1)
            v = v.expand(-1, self.num_heads, -1, -1)
        else:
            k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rotary:
            cos, sin = self.rotary(S, x.device)
            q, k = RotaryPositionEmbedding.apply_rotary(q, k, cos, sin)

        # Use PyTorch's flash attention if available
        if hasattr(F, "scaled_dot_product_attention"):
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout_p if self.training else 0,
                is_causal=is_causal,
            )
        else:
            # Fallback for older PyTorch
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if is_causal:
                mask = torch.triu(
                    torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
                )
                scores = scores.masked_fill(mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            attn = torch.matmul(attn, v)

        out = attn.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)
