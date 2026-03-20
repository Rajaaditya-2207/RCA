"""
Sliding Window Attention + Global Memory Tokens
================================================

Local windowed attention with global memory tokens for
deep reasoning. Acts as "Focus" in the Brain Analogy —
perfect vision for recent tokens + global context bookmarks.

Author: Rajaaditya.R
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .positions import RotaryPositionEmbedding


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention with Global Memory Tokens.

    - Local window attention: O(n·w) instead of O(n²)
    - Global memory tokens: M learned tokens that attend to everything
    - RoPE within windows for position awareness
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 512,
        num_memory_tokens: int = 32,
        dropout: float = 0.1,
        use_mqa: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.num_memory_tokens = num_memory_tokens
        self.use_mqa = use_mqa
        self.scale = self.head_dim ** -0.5

        # Q projection: always full heads
        self.q_proj = nn.Linear(dim, dim, bias=False)

        # K, V projections: single head if MQA, else full
        kv_dim = self.head_dim if use_mqa else dim
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Global memory tokens — learned embeddings
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(
                torch.randn(1, num_memory_tokens, dim) * 0.02
            )
            self.memory_k_proj = nn.Linear(dim, kv_dim, bias=False)
            self.memory_v_proj = nn.Linear(dim, kv_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        # RoPE for local positions
        self.rotary = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        """
        Forward pass with sliding window + memory tokens.

        Args:
            x: [B, S, D]
            is_causal: apply causal masking

        Returns:
            output: [B, S, D]
        """
        B, S, D = x.shape
        H = self.num_heads
        hd = self.head_dim
        W = self.window_size

        # Project Q, K, V
        q = self.q_proj(x)  # [B, S, D]

        if self.use_mqa:
            k = self.k_proj(x)  # [B, S, hd]
            v = self.v_proj(x)  # [B, S, hd]
        else:
            k = self.k_proj(x)  # [B, S, D]
            v = self.v_proj(x)  # [B, S, D]

        # Reshape Q: [B, H, S, hd]
        q = q.view(B, S, H, hd).transpose(1, 2)

        if self.use_mqa:
            # Single KV head, expand to all heads
            k = k.view(B, S, 1, hd).transpose(1, 2).expand(-1, H, -1, -1)
            v = v.view(B, S, 1, hd).transpose(1, 2).expand(-1, H, -1, -1)
        else:
            k = k.view(B, S, H, hd).transpose(1, 2)
            v = v.view(B, S, H, hd).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary(S, x.device)
        q, k = RotaryPositionEmbedding.apply_rotary(q, k, cos, sin)

        # Compute attention output
        if S <= W:
            # Short sequence: standard attention
            output = self._standard_attention(q, k, v, is_causal)
        else:
            # Long sequence: sliding window
            output = self._windowed_attention(q, k, v, is_causal)

        # Add global memory attention
        if self.num_memory_tokens > 0:
            mem_out = self._memory_attention(x, q)
            output = output + mem_out

        # [B, H, S, hd] -> [B, S, D]
        output = output.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(output)

    def _standard_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool
    ) -> torch.Tensor:
        """Standard scaled dot-product attention for short sequences."""
        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        S = q.shape[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if is_causal:
            mask = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, v)

    def _windowed_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool
    ) -> torch.Tensor:
        """
        Sliding window attention: each token attends to at most W previous tokens.
        """
        B, H, S, hd = q.shape
        W = self.window_size

        outputs = torch.zeros_like(q)

        for i in range(0, S, W):
            end = min(i + W, S)
            # Window start: look back up to W tokens
            start = max(0, end - W)

            q_win = q[:, :, i:end, :]
            k_win = k[:, :, start:end, :]
            v_win = v[:, :, start:end, :]

            win_len = end - start
            q_len = end - i

            scores = torch.matmul(q_win, k_win.transpose(-2, -1)) * self.scale

            if is_causal:
                # Build causal mask for this window
                q_positions = torch.arange(i, end, device=q.device)
                k_positions = torch.arange(start, end, device=q.device)
                mask = q_positions.unsqueeze(1) < k_positions.unsqueeze(0)
                scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            outputs[:, :, i:end, :] = torch.matmul(attn, v_win)

        return outputs

    def _memory_attention(
        self, x: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        """
        Attend to global memory tokens.

        Each sequence token can attend to M memory tokens
        that act as global context bookmarks.
        """
        B, H, S, hd = q.shape
        M = self.num_memory_tokens

        # Expand memory tokens for this batch
        mem = self.memory_tokens.expand(B, -1, -1)  # [B, M, D]

        if self.use_mqa:
            mk = self.memory_k_proj(mem).view(B, M, 1, hd).transpose(1, 2).expand(-1, H, -1, -1)
            mv = self.memory_v_proj(mem).view(B, M, 1, hd).transpose(1, 2).expand(-1, H, -1, -1)
        else:
            mk = self.memory_k_proj(mem).view(B, M, H, hd).transpose(1, 2)
            mv = self.memory_v_proj(mem).view(B, M, H, hd).transpose(1, 2)

        # q: [B, H, S, hd], mk: [B, H, M, hd]
        scores = torch.matmul(q, mk.transpose(-2, -1)) * self.scale  # [B, H, S, M]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        return torch.matmul(attn, mv)  # [B, H, S, hd]
