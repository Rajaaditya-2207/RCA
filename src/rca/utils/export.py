"""
Export Utilities
================

ONNX export and pretrained model save/load helpers.

Author: Rajaaditya.R
"""

import torch
import os
import json
from typing import Optional


def export_to_onnx(
    model,
    output_path: str,
    seq_len: int = 128,
    batch_size: int = 1,
    opset_version: int = 17,
):
    """
    Export RCA model to ONNX format.

    Args:
        model: RCA model instance
        output_path: path for the .onnx file
        seq_len: example sequence length
        batch_size: example batch size
        opset_version: ONNX opset version
    """
    device = next(model.parameters()).device
    model.eval()

    dummy_input = torch.randint(
        0, model.config.vocab_size, (batch_size, seq_len), device=device
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=opset_version,
    )
    print(f"Exported ONNX model to {output_path}")


def save_pretrained(model, path: str):
    """Save model + config to directory."""
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(path)
    else:
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, "model.pt"))
        if hasattr(model, "config"):
            model.config.to_json(os.path.join(path, "config.json"))


def load_pretrained(model_class, path: str, **kwargs):
    """Load model from directory."""
    if hasattr(model_class, "from_pretrained"):
        return model_class.from_pretrained(path, **kwargs)
    else:
        from ..config import RCAConfig

        config = RCAConfig.from_json(os.path.join(path, "config.json"))
        model = model_class(config)
        state_dict = torch.load(
            os.path.join(path, "model.pt"),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        return model
