"""
Model Converter
================

Export RCA models to safetensors and GGUF formats.

Author: Rajaaditya.R
"""

import torch
import os
import json
import struct
from typing import Optional, Dict
from collections import OrderedDict


# =========================================================================
# Safetensors Export/Import
# =========================================================================

def export_safetensors(model, path: str, metadata: Optional[Dict[str, str]] = None):
    """
    Export model weights to safetensors format.

    Requires: pip install safetensors

    Args:
        model: RCA model instance
        path: directory to save to
        metadata: optional metadata dict
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("Install safetensors: pip install safetensors")

    os.makedirs(path, exist_ok=True)

    # Unwrap DDP/FSDP/compiled
    raw_model = model
    if hasattr(raw_model, "module"):
        raw_model = raw_model.module
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod

    # Build metadata
    meta = {"format": "rca", "version": "2.0"}
    if hasattr(raw_model, "config"):
        meta["model_type"] = "rca"
        meta["state_dim"] = str(raw_model.config.state_dim)
        meta["n_layers"] = str(raw_model.config.n_layers)
        meta["n_heads"] = str(raw_model.config.n_heads)
        meta["params"] = str(raw_model.count_parameters())
    if metadata:
        meta.update(metadata)

    # Save weights
    state_dict = raw_model.state_dict()
    # safetensors requires all tensors to be contiguous
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}
    save_file(state_dict, os.path.join(path, "model.safetensors"), metadata=meta)

    # Save config
    if hasattr(raw_model, "config"):
        raw_model.config.to_json(os.path.join(path, "config.json"))

    print(f"Exported safetensors to {path}")


def load_safetensors(model_class, path: str, device: str = "cpu"):
    """
    Load model from safetensors format.

    Args:
        model_class: RCA model class
        path: directory containing model.safetensors + config.json
        device: device to load to

    Returns:
        loaded model
    """
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("Install safetensors: pip install safetensors")

    from .config import RCAConfig

    config = RCAConfig.from_json(os.path.join(path, "config.json"))
    model = model_class(config)

    state_dict = load_file(os.path.join(path, "model.safetensors"), device=device)
    model.load_state_dict(state_dict)

    return model


# =========================================================================
# GGUF Export
# =========================================================================

# GGUF constants
GGUF_MAGIC = 0x46475547  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGUF value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_UINT64 = 10

# GGML tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_0 = 2


def _write_string(f, s: str):
    """Write a GGUF string (uint64 length + bytes)."""
    encoded = s.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)


def _write_kv(f, key: str, value, vtype: int):
    """Write a GGUF key-value pair."""
    _write_string(f, key)
    f.write(struct.pack("<I", vtype))
    if vtype == GGUF_TYPE_STRING:
        _write_string(f, str(value))
    elif vtype == GGUF_TYPE_UINT32:
        f.write(struct.pack("<I", int(value)))
    elif vtype == GGUF_TYPE_INT32:
        f.write(struct.pack("<i", int(value)))
    elif vtype == GGUF_TYPE_FLOAT32:
        f.write(struct.pack("<f", float(value)))
    elif vtype == GGUF_TYPE_UINT64:
        f.write(struct.pack("<Q", int(value)))


def _quantize_q8_0(tensor: torch.Tensor) -> bytes:
    """Quantize tensor to Q8_0 (block size 32, 1 scale per block)."""
    flat = tensor.float().flatten()
    # Pad to multiple of 32
    pad = (32 - len(flat) % 32) % 32
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad)])

    blocks = flat.view(-1, 32)
    scales = blocks.abs().max(dim=1).values / 127.0
    scales = scales.clamp(min=1e-10)

    quantized = torch.round(blocks / scales.unsqueeze(1)).clamp(-128, 127).to(torch.int8)

    result = bytearray()
    for i in range(len(blocks)):
        result.extend(struct.pack("<e", scales[i].item()))  # f16 scale
        result.extend(quantized[i].numpy().tobytes())

    return bytes(result)


def _quantize_q4_0(tensor: torch.Tensor) -> bytes:
    """Quantize tensor to Q4_0 (block size 32, 4-bit quantization)."""
    flat = tensor.float().flatten()
    # Pad to multiple of 32
    pad = (32 - len(flat) % 32) % 32
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad)])

    blocks = flat.view(-1, 32)
    scales = blocks.abs().max(dim=1).values / 7.0
    scales = scales.clamp(min=1e-10)

    quantized = torch.round(blocks / scales.unsqueeze(1)).clamp(-8, 7).to(torch.int8)

    result = bytearray()
    for i in range(len(blocks)):
        result.extend(struct.pack("<e", scales[i].item()))  # f16 scale
        # Pack two 4-bit values into one byte
        q = quantized[i].numpy()
        packed = bytearray(16)
        for j in range(16):
            lo = q[j] & 0x0F
            hi = q[j + 16] & 0x0F
            packed[j] = (hi << 4) | lo
        result.extend(packed)

    return bytes(result)


def export_gguf(
    model,
    output_path: str,
    quantization: str = "f16",
):
    """
    Export RCA model to GGUF format.

    Args:
        model: RCA model instance
        output_path: path for .gguf file
        quantization: "f32", "f16", "q8_0", or "q4_0"
    """
    quant_map = {"f32": GGML_TYPE_F32, "f16": GGML_TYPE_F16, "q8_0": GGML_TYPE_Q8_0, "q4_0": GGML_TYPE_Q4_0}
    if quantization not in quant_map:
        raise ValueError(f"Unsupported quantization: {quantization}. Use: {list(quant_map.keys())}")

    ggml_type = quant_map[quantization]

    # Unwrap
    raw_model = model
    if hasattr(raw_model, "module"):
        raw_model = raw_model.module
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod

    config = raw_model.config
    state_dict = raw_model.state_dict()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Metadata KVs
    kvs = OrderedDict()
    kvs[("general.architecture", GGUF_TYPE_STRING)] = "rca"
    kvs[("general.name", GGUF_TYPE_STRING)] = "RCA Ultra-Reasoning"
    kvs[("general.quantization", GGUF_TYPE_STRING)] = quantization
    kvs[("rca.context_length", GGUF_TYPE_UINT32)] = config.max_seq_len
    kvs[("rca.embedding_length", GGUF_TYPE_UINT32)] = config.state_dim
    kvs[("rca.block_count", GGUF_TYPE_UINT32)] = config.n_layers
    kvs[("rca.attention.head_count", GGUF_TYPE_UINT32)] = config.n_heads
    kvs[("rca.vocab_size", GGUF_TYPE_UINT32)] = config.vocab_size

    # Prepare tensor info
    tensor_data = []
    for name, param in state_dict.items():
        tensor_data.append((name, param))

    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensor_data)))  # n_tensors
        f.write(struct.pack("<Q", len(kvs)))  # n_kv

        # Write KVs
        for (key, vtype), value in kvs.items():
            _write_kv(f, key, value, vtype)

        # Tensor info (name, n_dims, dims, type, offset)
        # We need to calculate offsets, so first pass computes sizes
        tensor_bytes = []
        for name, param in tensor_data:
            t = param.contiguous()
            if quantization == "f32":
                data = t.float().numpy().tobytes()
            elif quantization == "f16":
                data = t.half().numpy().tobytes()
            elif quantization == "q8_0":
                data = _quantize_q8_0(t)
            elif quantization == "q4_0":
                data = _quantize_q4_0(t)
            tensor_bytes.append(data)

        # Write tensor infos
        offset = 0
        for i, (name, param) in enumerate(tensor_data):
            _write_string(f, name)
            ndim = len(param.shape)
            f.write(struct.pack("<I", ndim))
            for dim in param.shape:
                f.write(struct.pack("<Q", dim))
            f.write(struct.pack("<I", ggml_type))
            f.write(struct.pack("<Q", offset))
            offset += len(tensor_bytes[i])

        # Alignment padding
        alignment = 32
        pos = f.tell()
        pad = (alignment - pos % alignment) % alignment
        f.write(b"\x00" * pad)

        # Write tensor data
        for data in tensor_bytes:
            f.write(data)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported GGUF ({quantization}) to {output_path} ({size_mb:.1f} MB)")
