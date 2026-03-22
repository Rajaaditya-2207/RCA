"""
RCA Configuration
=================

Model configuration with presets and validation.

Author: Rajaaditya.R
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class RCAConfig:
    """
    Configuration for RCA models.

    Supports preset sizes and full customization.
    """

    # Core dimensions
    vocab_size: int = 50257
    state_dim: int = 768
    n_layers: int = 12
    ssm_expand: int = 2

    # Attention
    n_heads: int = 12
    num_attention_layers: int = 4
    attention_every_n: int = 3
    use_hybrid_attention: bool = True

    # SSM options
    use_selective_scan: bool = True
    use_full_matrix: bool = False

    # Regularization
    dropout: float = 0.1

    # Positional encoding
    use_alibi: bool = True
    use_rotary: bool = True
    alibi_learnable: bool = True
    max_seq_len: int = 8192

    # Training
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False

    # Misc
    pad_token_id: Optional[int] = None
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Ultra-Reasoning Architecture
    use_ultra_reasoning: bool = False
    use_glu_ffn: bool = False
    gla_heads: int = 8
    gla_expand_k: float = 1.0
    gla_expand_v: float = 2.0
    sliding_window_size: int = 512
    num_memory_tokens: int = 32
    ssm_zone_end: float = 0.6
    gla_zone_end: float = 0.85
    use_mqa: bool = False

    # Performance / Training Optimization
    gradient_checkpointing: bool = False
    use_torch_compile: bool = False
    compile_mode: str = "reduce-overhead"

    def __post_init__(self):
        """Validate config after initialization."""
        assert self.state_dim > 0, "state_dim must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.state_dim % self.n_heads == 0, (
            f"state_dim ({self.state_dim}) must be divisible by n_heads ({self.n_heads})"
        )
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "RCAConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: str) -> "RCAConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    # =========================================================================
    # Legacy Presets (kept for backward compat)
    # =========================================================================

    @classmethod
    def rca_tiny(cls) -> "RCAConfig":
        """~10M params — for testing and debugging."""
        return cls(
            state_dim=256,
            n_layers=4,
            n_heads=4,
            ssm_expand=2,
            num_attention_layers=1,
            attention_every_n=4,
            dropout=0.1,
        )

    @classmethod
    def rca_small(cls) -> "RCAConfig":
        """~50M params — lightweight model."""
        return cls(
            state_dim=512,
            n_layers=8,
            n_heads=8,
            ssm_expand=2,
            num_attention_layers=2,
            attention_every_n=4,
            dropout=0.1,
        )

    @classmethod
    def rca_base(cls) -> "RCAConfig":
        """~125M params — standard model."""
        return cls(
            state_dim=768,
            n_layers=12,
            n_heads=12,
            ssm_expand=2,
            num_attention_layers=4,
            attention_every_n=3,
            dropout=0.1,
        )

    @classmethod
    def rca_large(cls) -> "RCAConfig":
        """~350M params — high capacity."""
        return cls(
            state_dim=1024,
            n_layers=24,
            n_heads=16,
            ssm_expand=2,
            num_attention_layers=8,
            attention_every_n=3,
            dropout=0.1,
        )

    @classmethod
    def rca_xl(cls) -> "RCAConfig":
        """~1B params — extra-large model."""
        return cls(
            state_dim=1280,
            n_layers=36,
            n_heads=20,
            ssm_expand=2,
            num_attention_layers=12,
            attention_every_n=3,
            dropout=0.05,
        )

    @classmethod
    def rca_ultra(cls) -> "RCAConfig":
        """~300M params — ultra-reasoning architecture.

        Brain Analogy:
        - Layers 1–19 (60%): SSM backbone (stream of consciousness)
        - Layers 20–27 (25%): GLA (working memory)
        - Layers 28–32 (15%): Sliding Window + Memory Tokens (focus)
        - All layers: GLU-FFN (active neurons / world knowledge)
        """
        return cls(
            state_dim=768,
            n_layers=32,
            n_heads=12,
            ssm_expand=2,
            num_attention_layers=0,
            attention_every_n=0,
            use_hybrid_attention=False,
            dropout=0.1,
            use_ultra_reasoning=True,
            use_glu_ffn=True,
            gla_heads=12,
            gla_expand_k=1.0,
            gla_expand_v=2.0,
            sliding_window_size=512,
            num_memory_tokens=32,
            ssm_zone_end=0.6,
            gla_zone_end=0.85,
            use_mqa=False,
        )

    # =========================================================================
    # Production Presets — Ultra-Reasoning with seq_len=4096
    # =========================================================================

    @classmethod
    def rca_100m(cls) -> "RCAConfig":
        """~100M params — Ultra-Reasoning, fits easily on T4/P100.

        Training budget (7 hrs):
          T4:   ~800M tokens   (batch=8,  grad_accum=4,  fp16)
          P100: ~1.2B tokens   (batch=8,  grad_accum=4,  fp16)
        """
        return cls(
            state_dim=512,
            n_layers=12,
            n_heads=8,
            ssm_expand=2,
            max_seq_len=4096,
            num_attention_layers=0,
            attention_every_n=0,
            use_hybrid_attention=False,
            dropout=0.1,
            use_ultra_reasoning=True,
            use_glu_ffn=True,
            gla_heads=8,
            gla_expand_k=1.0,
            gla_expand_v=2.0,
            sliding_window_size=512,
            num_memory_tokens=32,
            ssm_zone_end=0.6,
            gla_zone_end=0.85,
            use_mqa=False,
            gradient_checkpointing=False,
        )

    @classmethod
    def rca_500m(cls) -> "RCAConfig":
        """~500M params — Ultra-Reasoning, T4/P100 with checkpointing.

        Training budget (7 hrs):
          T4:   ~300M tokens   (batch=2, grad_accum=16, fp16, grad_ckpt)
          P100: ~500M tokens   (batch=2, grad_accum=16, fp16, grad_ckpt)
        """
        return cls(
            state_dim=1024,
            n_layers=20,
            n_heads=16,
            ssm_expand=2,
            max_seq_len=4096,
            num_attention_layers=0,
            attention_every_n=0,
            use_hybrid_attention=False,
            dropout=0.1,
            use_ultra_reasoning=True,
            use_glu_ffn=True,
            gla_heads=16,
            gla_expand_k=1.0,
            gla_expand_v=2.0,
            sliding_window_size=512,
            num_memory_tokens=32,
            ssm_zone_end=0.6,
            gla_zone_end=0.85,
            use_mqa=False,
            gradient_checkpointing=True,
        )

    @classmethod
    def rca_1b(cls) -> "RCAConfig":
        """~1B params — Ultra-Reasoning, T4/P100 with aggressive checkpointing.

        Training budget (7 hrs):
          T4:   ~150M tokens   (batch=1, grad_accum=32, fp16, grad_ckpt)
          P100: ~250M tokens   (batch=1, grad_accum=32, fp16, grad_ckpt)
        """
        return cls(
            state_dim=1280,
            n_layers=28,
            n_heads=20,
            ssm_expand=2,
            max_seq_len=4096,
            num_attention_layers=0,
            attention_every_n=0,
            use_hybrid_attention=False,
            dropout=0.05,
            use_ultra_reasoning=True,
            use_glu_ffn=True,
            gla_heads=20,
            gla_expand_k=1.0,
            gla_expand_v=2.0,
            sliding_window_size=512,
            num_memory_tokens=32,
            ssm_zone_end=0.6,
            gla_zone_end=0.85,
            use_mqa=True,
            gradient_checkpointing=True,
        )

    @classmethod
    def rca_5b(cls) -> "RCAConfig":
        """~5B params — Ultra-Reasoning, requires multi-GPU (FSDP).

        Training budget (1T tokens):
          4×A100 80GB: ~21 days  (FSDP, bf16)
        """
        return cls(
            state_dim=2048,
            n_layers=48,
            n_heads=32,
            ssm_expand=2,
            max_seq_len=4096,
            num_attention_layers=0,
            attention_every_n=0,
            use_hybrid_attention=False,
            dropout=0.05,
            use_ultra_reasoning=True,
            use_glu_ffn=True,
            gla_heads=32,
            gla_expand_k=1.0,
            gla_expand_v=2.0,
            sliding_window_size=512,
            num_memory_tokens=64,
            ssm_zone_end=0.6,
            gla_zone_end=0.85,
            use_mqa=True,
            gradient_checkpointing=True,
        )

    @classmethod
    def rca_10b(cls) -> "RCAConfig":
        """~10B params — Ultra-Reasoning, requires 8×A100 (FSDP).

        Training budget (1T tokens):
          8×A100 80GB: ~30 days  (FSDP, bf16)
        """
        return cls(
            state_dim=3072,
            n_layers=48,
            n_heads=32,
            ssm_expand=2,
            max_seq_len=4096,
            num_attention_layers=0,
            attention_every_n=0,
            use_hybrid_attention=False,
            dropout=0.05,
            use_ultra_reasoning=True,
            use_glu_ffn=True,
            gla_heads=32,
            gla_expand_k=1.0,
            gla_expand_v=2.0,
            sliding_window_size=512,
            num_memory_tokens=64,
            ssm_zone_end=0.6,
            gla_zone_end=0.85,
            use_mqa=True,
            gradient_checkpointing=True,
        )

    @classmethod
    def rca_100b(cls) -> "RCAConfig":
        """~100B params — Ultra-Reasoning, requires large cluster or TPU pod.

        Training budget (1T tokens):
          TPU v4-256 pod: ~45 days  (XLA FSDP)
          64×A100 80GB:  ~45 days  (FSDP, bf16)
        """
        return cls(
            state_dim=7680,
            n_layers=80,
            n_heads=64,
            ssm_expand=2,
            max_seq_len=4096,
            num_attention_layers=0,
            attention_every_n=0,
            use_hybrid_attention=False,
            dropout=0.05,
            use_ultra_reasoning=True,
            use_glu_ffn=True,
            gla_heads=64,
            gla_expand_k=1.0,
            gla_expand_v=2.0,
            sliding_window_size=512,
            num_memory_tokens=128,
            ssm_zone_end=0.6,
            gla_zone_end=0.85,
            use_mqa=True,
            gradient_checkpointing=True,
        )
