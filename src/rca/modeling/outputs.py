"""
Model Outputs
=============

Structured output types for RCA models.

Author: Rajaaditya.R
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class ModelOutput:
    """Base model output."""

    last_hidden_state: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    ssm_states: Optional[Tuple[torch.Tensor, ...]] = None


@dataclass
class CausalLMOutput:
    """Output for causal language modeling."""

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    ssm_states: Optional[Tuple[torch.Tensor, ...]] = None


@dataclass
class BaseModelOutput:
    """Alias compatible with HuggingFace conventions."""

    last_hidden_state: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
