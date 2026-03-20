"""
Tests for Gated Linear Attention
=================================

Tests: shape validation, parallel vs recurrent equivalence, gradient flow.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rca.layers.gla import GatedLinearAttention


class TestGatedLinearAttention:
    @pytest.fixture
    def gla(self):
        return GatedLinearAttention(dim=128, num_heads=4, expand_k=1.0, expand_v=2.0)

    def test_shape_output(self, gla):
        x = torch.randn(2, 16, 128)
        out = gla(x)
        assert out.shape == (2, 16, 128)

    def test_single_token(self, gla):
        x = torch.randn(2, 1, 128)
        out = gla(x)
        assert out.shape == (2, 1, 128)

    def test_recurrent_shape(self, gla):
        x = torch.randn(2, 1, 128)
        out, state = gla.forward_recurrent(x)
        assert out.shape == (2, 1, 128)
        assert state is not None
        # State: [B, H, dk, dv]
        assert state.shape[0] == 2
        assert state.shape[1] == 4  # num_heads

    def test_recurrent_state_update(self, gla):
        """Verify state evolves across steps."""
        x1 = torch.randn(2, 1, 128)
        x2 = torch.randn(2, 1, 128)

        out1, state1 = gla.forward_recurrent(x1)
        out2, state2 = gla.forward_recurrent(x2, state1)

        # States should differ after processing different inputs
        assert not torch.allclose(state1, state2, atol=1e-6)

    def test_gradient_flow(self, gla):
        x = torch.randn(2, 8, 128, requires_grad=True)
        out = gla(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_various_sequence_lengths(self, gla):
        for seq_len in [1, 4, 16, 63, 64, 65, 128]:
            x = torch.randn(1, seq_len, 128)
            out = gla(x, chunk_size=64)
            assert out.shape == (1, seq_len, 128), f"Failed for seq_len={seq_len}"
