import pytest
import torch
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rca.layers.gla import GatedLinearAttention
from rca.layers.scan import parallel_scan_linear, compute_parallel_scan
from rca.layers.sliding_attention import SlidingWindowAttention
from rca import RCAModel, RCAConfig

# ------------- test_scan helpers -------------
def sequential_scan(gates, inputs, initial=None):
    """Reference sequential scan for validation."""
    B, S, D = gates.shape
    outputs = []
    h = initial if initial is not None else torch.zeros(B, D, device=gates.device)
    for s in range(S):
        h = gates[:, s] * h + inputs[:, s]
        outputs.append(h)
    return torch.stack(outputs, dim=1)

# ------------- test_gla.py -------------
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
        assert state.shape[0] == 2
        assert state.shape[1] == 4

    def test_recurrent_state_update(self, gla):
        x1 = torch.randn(2, 1, 128)
        x2 = torch.randn(2, 1, 128)
        out1, state1 = gla.forward_recurrent(x1)
        out2, state2 = gla.forward_recurrent(x2, state1)
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

# ------------- test_scan.py -------------
class TestParallelScan:
    def test_basic_correctness(self):
        B, S, D = 2, 8, 4
        gates = torch.rand(B, S, D) * 0.9
        inputs = torch.randn(B, S, D)
        expected = sequential_scan(gates, inputs)
        result = parallel_scan_linear(gates, inputs)
        assert torch.allclose(result, expected, atol=1e-4)

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
                    assert torch.allclose(result, expected, atol=1e-3)

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

# ------------- test_sliding_attention.py -------------
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
        x = torch.randn(2, 16, 128)
        out = swa(x)
        assert out.shape == (2, 16, 128)

    def test_long_sequence(self, swa):
        x = torch.randn(2, 64, 128)
        out = swa(x)
        assert out.shape == (2, 64, 128)

    def test_single_token(self, swa):
        x = torch.randn(2, 1, 128)
        out = swa(x)
        assert out.shape == (2, 1, 128)

    def test_mqa_variant(self, swa_mqa):
        x = torch.randn(2, 16, 128)
        out = swa_mqa(x)
        assert out.shape == (2, 16, 128)

    def test_mqa_fewer_params(self, swa, swa_mqa):
        mha_params = sum(p.numel() for p in swa.parameters())
        mqa_params = sum(p.numel() for p in swa_mqa.parameters())
        assert mqa_params < mha_params

    def test_gradient_flow(self, swa):
        x = torch.randn(2, 16, 128, requires_grad=True)
        out = swa(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_no_memory_tokens(self):
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
            assert out.shape == (1, seq_len, 128)

# ------------- test_model.py -------------
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

    def test_validation_fails(self):
        with pytest.raises(AssertionError):
            RCAConfig(state_dim=100, n_heads=3)

    def test_json_roundtrip(self):
        config = RCAConfig.rca_tiny()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            config.to_json(path)
            loaded = RCAConfig.from_json(path)
            assert loaded.state_dim == config.state_dim
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

    def test_forward_pass(self, tiny_model):
        input_ids = torch.randint(0, 1000, (2, 16))
        out = tiny_model(input_ids)
        assert out.logits.shape == (2, 16, 1000)

    def test_forward_with_labels(self, tiny_model):
        input_ids = torch.randint(0, 1000, (2, 16))
        labels = input_ids.clone()
        out = tiny_model(input_ids, labels=labels)
        assert out.loss is not None

    def test_generate(self, tiny_model):
        input_ids = torch.randint(0, 1000, (1, 8))
        with torch.no_grad():
            out = tiny_model.generate(input_ids, max_new_tokens=5)
        assert out.shape[0] == 1

    def test_save_load_pretrained(self, tiny_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_model.save_pretrained(tmpdir)
            loaded = RCAModel.from_pretrained(tmpdir)
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
    @pytest.fixture
    def ultra_config(self):
        config = RCAConfig.rca_ultra()
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

    def test_ultra_model_creation(self, ultra_model, ultra_config):
        assert ultra_model is not None

    def test_ultra_layer_zones(self, ultra_model):
        zones = ultra_model.get_layer_zones()
        assert "ssm" in zones
        assert "gla" in zones
        assert "reasoning" in zones

    def test_ultra_forward_pass(self, ultra_model, ultra_config):
        input_ids = torch.randint(0, ultra_config.vocab_size, (2, 16))
        out = ultra_model(input_ids)
        assert out.logits.shape == (2, 16, ultra_config.vocab_size)

    def test_ultra_forward_with_labels(self, ultra_model, ultra_config):
        input_ids = torch.randint(0, ultra_config.vocab_size, (2, 16))
        labels = input_ids.clone()
        out = ultra_model(input_ids, labels=labels)
        assert out.loss is not None

    def test_ultra_generate(self, ultra_model, ultra_config):
        input_ids = torch.randint(0, ultra_config.vocab_size, (1, 8))
        with torch.no_grad():
            out = ultra_model.generate(input_ids, max_new_tokens=5)
        assert out.shape[0] == 1

    def test_ultra_save_load(self, ultra_model, ultra_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            ultra_model.save_pretrained(tmpdir)
            loaded = RCAModel.from_pretrained(tmpdir)
            assert loaded.config.use_ultra_reasoning is True

    def test_ultra_gradient_flow(self, ultra_model, ultra_config):
        input_ids = torch.randint(0, ultra_config.vocab_size, (1, 8))
        labels = input_ids.clone()
        out = ultra_model(input_ids, labels=labels)
        out.loss.backward()
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
