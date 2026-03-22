"""
Normalization Layers
====================

RMSNorm (fused) and DeepNorm implementations.

Author: Rajaaditya.R
"""

import torch
import torch.nn as nn
import math


class RMSNorm(nn.Module):
    """
    Fused RMSNorm: x * rsqrt(mean(x²) + eps) * gamma

    Uses rsqrt(mean(x²)) instead of norm(2)/sqrt(D) for better
    numerical performance and GPU kernel fusion.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused: rsqrt(mean(x²) + eps) is much faster than norm(2)/sqrt(D)
        # PyTorch can fuse this into a single kernel with torch.compile
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class DeepNorm(nn.Module):
    """DeepNorm for very deep networks (stabilizes training at 100+ layers)."""

    def __init__(self, dim: int, depth: int = 12):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.alpha = (2 * depth) ** 0.25

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.alpha * x + x)
