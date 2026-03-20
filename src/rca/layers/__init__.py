"""RCA Layers."""

from .ssm import SelectiveStateSpaceModel, SimpleStateSpaceModel
from .attention import EfficientAttention
from .gla import GatedLinearAttention
from .sliding_attention import SlidingWindowAttention
from .scan import parallel_scan_linear, compute_parallel_scan, TRITON_AVAILABLE
from .norm import RMSNorm, DeepNorm
from .positions import ALiBiPositionEmbedding, RotaryPositionEmbedding

__all__ = [
    "SelectiveStateSpaceModel",
    "SimpleStateSpaceModel",
    "EfficientAttention",
    "GatedLinearAttention",
    "SlidingWindowAttention",
    "parallel_scan_linear",
    "compute_parallel_scan",
    "TRITON_AVAILABLE",
    "RMSNorm",
    "DeepNorm",
    "ALiBiPositionEmbedding",
    "RotaryPositionEmbedding",
]
