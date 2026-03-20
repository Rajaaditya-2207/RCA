"""
Position Encodings
==================

ALiBi and RoPE implementations.

Author: Rajaaditya.R
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class ALiBiPositionEmbedding(nn.Module):
    """
    ALiBi: Attention with Linear Biases.

    Better extrapolation than fixed positional encodings.
    """

    def __init__(
        self,
        num_heads: int,
        max_seq_len: int = 8192,
        learnable: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Compute ALiBi slopes (powers of 2)
        def get_slopes(n: int):
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            base = 2.0 ** (-(2.0 ** -(math.log2(closest_power_of_2) - 3)))
            slopes = [base * (base ** (-i)) for i in range(closest_power_of_2)]
            if closest_power_of_2 != n:
                extra_base = 2.0 ** (-(2.0 ** -(math.log2(2 * closest_power_of_2) - 3)))
                extra = [extra_base * (extra_base ** (-i)) for i in range(n - closest_power_of_2)]
                slopes = slopes + extra
            return slopes

        slopes = torch.tensor(get_slopes(num_heads), dtype=torch.float32)

        # Build distance matrix
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # [S, S]

        # alibi: [1, H, S, S]
        alibi = slopes.view(1, num_heads, 1, 1) * distance.unsqueeze(0).unsqueeze(0)
        self.register_buffer("alibi", alibi)

        if learnable:
            self.scale = nn.Parameter(torch.ones(num_heads))
        else:
            self.register_buffer("scale", torch.ones(num_heads))

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.alibi[:, :, :seq_len, :seq_len] * self.scale.view(1, -1, 1, 1)


class RotaryPositionEmbedding(nn.Module):
    """RoPE: Rotary Position Embedding."""

    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Duplicate to match full head_dim: [S, dim/2] -> [S, dim]
        # This matches rotate_half which splits the last dim in two halves
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def apply_rotary(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat([-x2, x1], dim=-1)

        return (
            q * cos + rotate_half(q) * sin,
            k * cos + rotate_half(k) * sin,
        )
