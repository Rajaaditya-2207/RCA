"""
RCA — Recursive Compression Architecture
=========================================

A production-ready, memory-efficient language model architecture
combining Selective State Space Models, Hybrid Attention, and
CUDA/Triton-accelerated parallel scan.

Author: Rajaaditya.R
Contact: rajaaditya.aadhi@gmail.com

Quick start::

    from rca import RCAModel, RCAConfig

    config = RCAConfig.rca_base()
    model = RCAModel(config)

    print(f"Parameters: {model.count_parameters():,}")
"""

__version__ = "1.0.0"
__author__ = "Rajaaditya.R"
__email__ = "rajaaditya.aadhi@gmail.com"

from .config import RCAConfig
from .modeling.rca_model import RCAModel, RCAForCausalLM
from .modeling.outputs import CausalLMOutput, ModelOutput, BaseModelOutput
from .trainer import RCATrainer, TrainingArguments
from .generator import RCAGenerator
from .utils.benchmark import RCABenchmark
from .utils.export import export_to_onnx, save_pretrained, load_pretrained
from .layers.scan import compute_parallel_scan, parallel_scan_linear, TRITON_AVAILABLE
from .layers.ssm import SelectiveStateSpaceModel, SimpleStateSpaceModel
from .layers.attention import EfficientAttention
from .layers.norm import RMSNorm, DeepNorm
from .layers.positions import ALiBiPositionEmbedding, RotaryPositionEmbedding

__all__ = [
    # Config
    "RCAConfig",
    # Models
    "RCAModel",
    "RCAForCausalLM",
    # Outputs
    "CausalLMOutput",
    "ModelOutput",
    "BaseModelOutput",
    # Training
    "RCATrainer",
    "TrainingArguments",
    # Generation
    "RCAGenerator",
    # Utilities
    "RCABenchmark",
    "export_to_onnx",
    "save_pretrained",
    "load_pretrained",
    # Layers (advanced)
    "compute_parallel_scan",
    "parallel_scan_linear",
    "TRITON_AVAILABLE",
    "SelectiveStateSpaceModel",
    "SimpleStateSpaceModel",
    "EfficientAttention",
    "RMSNorm",
    "DeepNorm",
    "ALiBiPositionEmbedding",
    "RotaryPositionEmbedding",
]
