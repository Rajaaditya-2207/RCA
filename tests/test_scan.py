"""
Tests for Parallel Scan
========================

Tests correctness of parallel scan against sequential reference.
"""

import pytest
import torch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rca.layers.scan import parallel_scan_linear, compute_parallel_scan


def sequential_scan(gates, inputs, initial=None):
    """Reference sequential scan for validation."""
    B, S, D = gates.shape
    outputs = []
    h = initial if initial is not None else torch.zeros(B, D, device=gates.device)
    for s in range(S):
        h = gates[:, s] * h + inputs[:, s]
        outputs.append(h)
    return torch.stack(outputs, dim=1)


class TestParallelScan:
    def test_basic_correctness(self):
        B, S, D = 2, 8, 4
        gates = torch.rand(B, S, D) * 0.9
        inputs = torch.randn(B, S, D)

        expected = sequential_scan(gates, inputs)
        result = parallel_scan_linear(gates, inputs)

        assert torch.allclose(result, expected, atol=1e-4), (
            f"Max diff: {(result - expected).abs().max().item()}"
        )

    def test_with_initial_state(self):
        B, S, D = 2, 8, 4
        gates = torch.rand(B, S, D) * 0.9
        inputs = torch.randn(B, S, D)
        initial = torch.randn(B, D)

        expected = sequential_scan(gates, inputs, initial)
        result = parallel_scan_linear(gates, inputs, initial)

        assert torch.allclose(result, expected, atol=1e-4)

    def test_single_step(self):
        B, D = 2, 4
        gates = torch.rand(B, 1, D) * 0.9
        inputs = torch.randn(B, 1, D)

        expected = sequential_scan(gates, inputs)
        result = parallel_scan_linear(gates, inputs)

        assert torch.allclose(result, expected, atol=1e-4)

    def test_various_shapes(self):
        for B in [1, 4]:
            for S in [1, 2, 4, 8, 16, 32]:
                for D in [1, 4, 16]:
                    gates = torch.rand(B, S, D) * 0.9
                    inputs = torch.randn(B, S, D)

                    expected = sequential_scan(gates, inputs)
                    result = parallel_scan_linear(gates, inputs)

                    assert torch.allclose(result, expected, atol=1e-3), (
                        f"Failed for B={B}, S={S}, D={D}: "
                        f"max diff {(result - expected).abs().max().item()}"
                    )

    def test_dispatcher(self):
        B, S, D = 2, 8, 4
        gates = torch.rand(B, S, D) * 0.9
        inputs = torch.randn(B, S, D)

        expected = sequential_scan(gates, inputs)
        result = compute_parallel_scan(gates, inputs, use_cuda=False)

        assert torch.allclose(result, expected, atol=1e-4)

    def test_gradient_flow(self):
        B, S, D = 2, 8, 4
        gates = (torch.rand(B, S, D) * 0.9).requires_grad_(True)
        inputs = torch.randn(B, S, D).requires_grad_(True)

        result = parallel_scan_linear(gates, inputs)
        loss = result.sum()
        loss.backward()

        assert gates.grad is not None
        assert inputs.grad is not None
        assert not torch.isnan(gates.grad).any()
        assert not torch.isnan(inputs.grad).any()
