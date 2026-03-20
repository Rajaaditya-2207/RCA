"""
Normalization Layers
====================

RMSNorm and DeepNorm implementations.

Author: Rajaaditya.R
"""

import torch
import torch.nn as nn
import math


class RMSNorm(nn.Module):
    """
    RMSNorm: x / RMS(x) * gamma

    More stable and often better than LayerNorm for language models.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.size(-1))
        return x / (rms + self.eps) * self.weight


class DeepNorm(nn.Module):
    """DeepNorm for very deep networks (stabilizes training at 100+ layers)."""

    def __init__(self, dim: int, depth: int = 12):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.alpha = (2 * depth) ** 0.25

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.alpha * x + x)
