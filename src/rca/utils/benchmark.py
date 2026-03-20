"""
Benchmarking Utilities
======================

Speed, memory, and generation benchmarks.

Author: Rajaaditya.R
"""

import torch
import time
from typing import Dict, Optional


class RCABenchmark:
    """Benchmark suite for RCA models."""

    @staticmethod
    @torch.no_grad()
    def speed_test(
        model,
        seq_len: int = 512,
        batch_size: int = 4,
        n_warmup: int = 3,
        n_runs: int = 10,
    ) -> Dict[str, float]:
        """Measure forward pass throughput."""
        device = next(model.parameters()).device
        model.eval()

        input_ids = torch.randint(
            0, model.config.vocab_size, (batch_size, seq_len), device=device
        )

        # Warmup
        for _ in range(n_warmup):
            model(input_ids)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        t0 = time.time()
        for _ in range(n_runs):
            model(input_ids)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        total_tokens = batch_size * seq_len * n_runs
        tokens_per_sec = total_tokens / elapsed

        return {
            "total_time_sec": elapsed,
            "avg_time_sec": elapsed / n_runs,
            "tokens_per_sec": tokens_per_sec,
            "tokens_per_sec_millions": tokens_per_sec / 1e6,
            "batch_size": batch_size,
            "seq_len": seq_len,
        }

    @staticmethod
    @torch.no_grad()
    def memory_test(
        model,
        seq_len: int = 512,
        batch_size: int = 4,
    ) -> Dict[str, float]:
        """Measure GPU memory usage."""
        device = next(model.parameters()).device
        if device.type != "cuda":
            return {"error": "Memory test requires CUDA device"}

        model.eval()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        baseline_mem = torch.cuda.memory_allocated()

        input_ids = torch.randint(
            0, model.config.vocab_size, (batch_size, seq_len), device=device
        )
        model(input_ids)

        peak_mem = torch.cuda.max_memory_allocated()
        current_mem = torch.cuda.memory_allocated()

        return {
            "baseline_mb": baseline_mem / 1e6,
            "peak_mb": peak_mem / 1e6,
            "current_mb": current_mem / 1e6,
            "forward_pass_mb": (peak_mem - baseline_mem) / 1e6,
        }

    @staticmethod
    @torch.no_grad()
    def generation_test(
        model,
        prompt_len: int = 32,
        gen_tokens: int = 128,
    ) -> Dict[str, float]:
        """Measure generation speed (tokens/sec)."""
        device = next(model.parameters()).device
        model.eval()

        input_ids = torch.randint(
            0, model.config.vocab_size, (1, prompt_len), device=device
        )

        t0 = time.time()
        output = model.generate(input_ids, max_new_tokens=gen_tokens)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        new_tokens = output.shape[1] - prompt_len
        tokens_per_sec = new_tokens / elapsed

        return {
            "total_time_sec": elapsed,
            "new_tokens": new_tokens,
            "tokens_per_sec": tokens_per_sec,
            "time_per_token_ms": (elapsed / max(1, new_tokens)) * 1000,
        }

    @staticmethod
    def compare_models(models: dict, **kwargs) -> Dict[str, Dict]:
        """Compare multiple models side by side."""
        results = {}
        for name, model in models.items():
            results[name] = {
                "speed": RCABenchmark.speed_test(model, **kwargs),
                "params": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            }
        return results
