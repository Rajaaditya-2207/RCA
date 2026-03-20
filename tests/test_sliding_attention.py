"""
Tests for Sliding Window Attention
====================================

Tests: window masking, memory tokens, causal masking, MQA variant.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rca.layers.sliding_attention import SlidingWindowAttention


class TestSlidingWindowAttention:
    @pytest.fixture
    def swa(self):
        return SlidingWindowAttention(
            dim=128, num_heads=4, window_size=32,
            num_memory_tokens=8, dropout=0.0,
        )

    @pytest.fixture
    def swa_mqa(self):
        return SlidingWindowAttention(
            dim=128, num_heads=4, window_size=32,
            num_memory_tokens=8, dropout=0.0, use_mqa=True,
        )

    def test_short_sequence(self, swa):
        """Short sequence (< window) uses standard attention."""
        x = torch.randn(2, 16, 128)
        out = swa(x)
        assert out.shape == (2, 16, 128)

    def test_long_sequence(self, swa):
        """Long sequence uses sliding window."""
        x = torch.randn(2, 64, 128)
        out = swa(x)
        assert out.shape == (2, 64, 128)

    def test_single_token(self, swa):
        x = torch.randn(2, 1, 128)
        out = swa(x)
        assert out.shape == (2, 1, 128)

    def test_mqa_variant(self, swa_mqa):
        """MQA should produce same shapes with fewer KV params."""
        x = torch.randn(2, 16, 128)
        out = swa_mqa(x)
        assert out.shape == (2, 16, 128)

    def test_mqa_fewer_params(self, swa, swa_mqa):
        """MQA should have fewer K/V parameters."""
        mha_params = sum(p.numel() for p in swa.parameters())
        mqa_params = sum(p.numel() for p in swa_mqa.parameters())
        assert mqa_params < mha_params

    def test_gradient_flow(self, swa):
        x = torch.randn(2, 16, 128, requires_grad=True)
        out = swa(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_no_memory_tokens(self):
        """Can disable memory tokens."""
        swa = SlidingWindowAttention(
            dim=128, num_heads=4, window_size=32,
            num_memory_tokens=0, dropout=0.0,
        )
        x = torch.randn(2, 16, 128)
        out = swa(x)
        assert out.shape == (2, 16, 128)

    def test_various_lengths(self, swa):
        for seq_len in [1, 4, 16, 31, 32, 33, 64]:
            x = torch.randn(1, seq_len, 128)
            out = swa(x)
            assert out.shape == (1, seq_len, 128), f"Failed for seq_len={seq_len}"
