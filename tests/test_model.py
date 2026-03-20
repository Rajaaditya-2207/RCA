"""
Tests for RCA Model
===================

Tests config, model creation, forward pass, loss, generate, save/load.
"""

import pytest
import torch
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rca import RCAModel, RCAConfig


class TestRCAConfig:
    def test_default_config(self):
        config = RCAConfig()
        assert config.state_dim == 768
        assert config.n_layers == 12
        assert config.n_heads == 12

    def test_presets(self):
        for preset in ["rca_tiny", "rca_small", "rca_base", "rca_large", "rca_xl"]:
            config = getattr(RCAConfig, preset)()
            assert config.state_dim > 0
            assert config.n_layers > 0
            assert config.state_dim % config.n_heads == 0

    def test_validation_fails(self):
        with pytest.raises(AssertionError):
            RCAConfig(state_dim=100, n_heads=3)  # 100 % 3 != 0

    def test_json_roundtrip(self):
        config = RCAConfig.rca_tiny()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            config.to_json(path)
            loaded = RCAConfig.from_json(path)
            assert loaded.state_dim == config.state_dim
            assert loaded.n_layers == config.n_layers
        finally:
            os.unlink(path)


class TestRCAModel:
    @pytest.fixture
    def tiny_model(self):
        config = RCAConfig.rca_tiny()
        config.vocab_size = 1000
        return RCAModel(config)

    def test_model_creation(self, tiny_model):
        assert tiny_model is not None
        assert tiny_model.count_parameters() > 0

    def test_forward_pass(self, tiny_model):
        input_ids = torch.randint(0, 1000, (2, 16))
        out = tiny_model(input_ids)
        assert out.logits is not None
        assert out.logits.shape == (2, 16, 1000)
        assert out.loss is None

    def test_forward_with_labels(self, tiny_model):
        input_ids = torch.randint(0, 1000, (2, 16))
        labels = input_ids.clone()
        out = tiny_model(input_ids, labels=labels)
        assert out.loss is not None
        assert out.loss.dim() == 0  # scalar
        assert out.loss.item() > 0

    def test_generate(self, tiny_model):
        input_ids = torch.randint(0, 1000, (1, 8))
        with torch.no_grad():
            out = tiny_model.generate(input_ids, max_new_tokens=5)
        assert out.shape[0] == 1
        assert out.shape[1] >= 8  # at least prompt length

    def test_save_load_pretrained(self, tiny_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_model.save_pretrained(tmpdir)
            loaded = RCAModel.from_pretrained(tmpdir)
            assert loaded.count_parameters() == tiny_model.count_parameters()

            # Check outputs match
            input_ids = torch.randint(0, 1000, (1, 8))
            tiny_model.eval()
            loaded.eval()
            with torch.no_grad():
                out1 = tiny_model(input_ids)
                out2 = loaded(input_ids)
            assert torch.allclose(out1.logits, out2.logits, atol=1e-5)

    def test_different_sequence_lengths(self, tiny_model):
        for seq_len in [1, 4, 16, 64]:
            input_ids = torch.randint(0, 1000, (1, seq_len))
            out = tiny_model(input_ids)
            assert out.logits.shape == (1, seq_len, 1000)


class TestUltraReasoningModel:
    """Tests for the Ultra-Reasoning Architecture (Brain Analogy)."""

    @pytest.fixture
    def ultra_config(self):
        config = RCAConfig.rca_ultra()
        # Use small dims for fast tests
        config.vocab_size = 500
        config.state_dim = 128
        config.n_layers = 10
        config.n_heads = 4
        config.gla_heads = 4
        config.sliding_window_size = 64
        config.num_memory_tokens = 4
        return config

    @pytest.fixture
    def ultra_model(self, ultra_config):
        return RCAModel(ultra_config)

    def test_ultra_config_preset(self):
        config = RCAConfig.rca_ultra()
        assert config.use_ultra_reasoning is True
        assert config.use_glu_ffn is True
        assert 0.0 < config.ssm_zone_end < 1.0
        assert config.ssm_zone_end < config.gla_zone_end < 1.0

    def test_ultra_model_creation(self, ultra_model, ultra_config):
        assert ultra_model is not None
        assert ultra_model.count_parameters() > 0
        assert len(ultra_model.layers) == ultra_config.n_layers

    def test_ultra_layer_zones(self, ultra_model):
        """Verify SSM → GLA → Reasoning zone distribution."""
        zones = ultra_model.get_layer_zones()
        assert "ssm" in zones
        assert "gla" in zones
        assert "reasoning" in zones
        assert len(zones["ssm"]) > 0
        assert len(zones["gla"]) > 0
        assert len(zones["reasoning"]) > 0
        # SSM should be the largest zone
        assert len(zones["ssm"]) >= len(zones["gla"])
        # All layers accounted for
        total = len(zones["ssm"]) + len(zones["gla"]) + len(zones["reasoning"])
        assert total == len(ultra_model.layers)

    def test_ultra_forward_pass(self, ultra_model, ultra_config):
        input_ids = torch.randint(0, ultra_config.vocab_size, (2, 16))
        out = ultra_model(input_ids)
        assert out.logits.shape == (2, 16, ultra_config.vocab_size)
        assert out.loss is None

    def test_ultra_forward_with_labels(self, ultra_model, ultra_config):
        input_ids = torch.randint(0, ultra_config.vocab_size, (2, 16))
        labels = input_ids.clone()
        out = ultra_model(input_ids, labels=labels)
        assert out.loss is not None
        assert out.loss.dim() == 0
        assert out.loss.item() > 0

    def test_ultra_generate(self, ultra_model, ultra_config):
        input_ids = torch.randint(0, ultra_config.vocab_size, (1, 8))
        with torch.no_grad():
            out = ultra_model.generate(input_ids, max_new_tokens=5)
        assert out.shape[0] == 1
        assert out.shape[1] >= 8

    def test_ultra_save_load(self, ultra_model, ultra_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            ultra_model.save_pretrained(tmpdir)
            loaded = RCAModel.from_pretrained(tmpdir)
            assert loaded.count_parameters() == ultra_model.count_parameters()
            assert loaded.config.use_ultra_reasoning is True

            # Outputs should match
            input_ids = torch.randint(0, ultra_config.vocab_size, (1, 8))
            ultra_model.eval()
            loaded.eval()
            with torch.no_grad():
                out1 = ultra_model(input_ids)
                out2 = loaded(input_ids)
            assert torch.allclose(out1.logits, out2.logits, atol=1e-5)

    def test_ultra_gradient_flow(self, ultra_model, ultra_config):
        """Verify gradients flow through all three zone types."""
        input_ids = torch.randint(0, ultra_config.vocab_size, (1, 8))
        labels = input_ids.clone()
        out = ultra_model(input_ids, labels=labels)
        out.loss.backward()

        # Check at least one parameter in each zone has non-zero grad
        zones = ultra_model.get_layer_zones()
        for zone_name, layer_indices in zones.items():
            has_grad = False
            for idx in layer_indices:
                layer = ultra_model.layers[idx]
                for p in layer.parameters():
                    if p.grad is not None and p.grad.abs().sum() > 0:
                        has_grad = True
                        break
                if has_grad:
                    break
            assert has_grad, f"No gradient flow in {zone_name} zone"
