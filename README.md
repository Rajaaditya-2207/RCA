# RCA 1.0.0 — Recursive Compression Architecture

**Ultra-Reasoning Architecture** combining Mamba SSM, Gated Linear Attention (GLA), and Sliding Window Attention across specialized cognitive zones — with Triton/XLA-accelerated parallel scan.

> RCA replaces the quadratic attention bottleneck of Transformers with **O(1) memory, linear-time processing**, and delivers **transformer-level reasoning** at dramatically lower cost.

---

## Why RCA?

| Feature | Transformer | Mamba | **RCA 1.0.0** |
|---|---|---|---|
| Training complexity | O(N²) | O(N) | **O(N)** |
| Generation memory | O(N) KV cache | O(1) | **O(1)** |
| Long-range reasoning | ✅ (expensive) | ❌ (weak) | **✅ (3-zone architecture)** |
| Generation speed | Slows with context | Constant | **Constant** |
| Gradient checkpointing | Manual | Manual | **Built-in** |
| Distributed training | Needs wrapper | Needs wrapper | **Built-in DDP/FSDP/XLA** |

### Architecture: The Brain Analogy

RCA's Ultra-Reasoning architecture divides layers into three cognitive zones:

```
┌──────────────────────────────────────────────────────────────┐
│   SSM Zone (60%)          │ Stream of Consciousness         │
│   Mamba-style SSM blocks  │ Long-range O(1) sequence scan   │
├──────────────────────────────────────────────────────────────┤
│   GLA Zone (25%)          │ Working Memory                  │
│   Gated Linear Attention  │ Associative recall, O(N) linear │
├──────────────────────────────────────────────────────────────┤
│   Reasoning Zone (15%)    │ Focus                           │
│   Sliding Window + Memory │ Sharp local attention + global  │
│   Tokens                  │ context bookmarks               │
├──────────────────────────────────────────────────────────────┤
│   All Zones: GLU-FFN      │ Active Neurons / World Knowledge│
└──────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
# Core (CPU/GPU)
pip install rca-arch

# With GPU acceleration (Triton kernels)
pip install rca-arch[gpu]

# With export support (safetensors)
pip install rca-arch[export]

# With training utilities
pip install rca-arch[training]

# Everything
pip install rca-arch[all]
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0.0

---

## Quick Start

### 1. Create a Model from Presets

```python
from rca import RCAConfig, RCAModel

# Choose a preset — see "Model Presets" below
config = RCAConfig.rca_100m()
config.vocab_size = 32000  # match your tokenizer

model = RCAModel(config)
print(f"Parameters: {model.count_parameters():,}")
# → Parameters: ~100,000,000

# Inspect the architecture zones
print(model.get_layer_zones())
# → {'ssm': [0..6], 'gla': [7..9], 'reasoning': [10..11]}
```

### 2. Forward Pass

```python
import torch

x = torch.randint(0, 32000, (2, 4096))  # [batch, seq_len]
output = model(x)

print(output.logits.shape)  # [2, 4096, 32000]
print(output.loss)          # None (no labels provided)
```

### 3. Forward Pass with Loss

```python
x = torch.randint(0, 32000, (2, 4096))
labels = x.clone()

output = model(x, labels=labels)
print(f"Loss: {output.loss.item():.4f}")
```

### 4. Generate Text

```python
prompt = torch.randint(0, 32000, (1, 64))  # [1, prompt_len]
generated = model.generate(
    prompt,
    max_new_tokens=200,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
)
print(generated.shape)  # [1, 264]
```

---

## Model Presets

All presets use Ultra-Reasoning architecture with `max_seq_len=4096`.

| Preset | Params | Layers | Dim | Heads | Grad Ckpt | Target Hardware |
|---|---|---|---|---|---|---|
| `rca_100m()` | ~100M | 12 | 512 | 8 | ❌ | T4 / P100 (single GPU) |
| `rca_500m()` | ~500M | 20 | 1024 | 16 | ✅ | T4 / P100 (single GPU) |
| `rca_1b()` | ~1B | 28 | 1280 | 20 | ✅ | T4 / P100 (single GPU) |
| `rca_5b()` | ~5B | 48 | 2048 | 32 | ✅ | 4×A100 (FSDP) |
| `rca_10b()` | ~10B | 48 | 3072 | 32 | ✅ | 8×A100 (FSDP) |
| `rca_100b()` | ~100B | 80 | 7680 | 64 | ✅ | 64×A100 / TPU pod (FSDP) |

### Estimated Training Budget (7-hour window)

| Preset | T4 (16GB) | P100 (16GB) | Settings |
|---|---|---|---|
| `rca_100m` | ~800M tokens | ~1.2B tokens | batch=8, grad_accum=4, fp16 |
| `rca_500m` | ~300M tokens | ~500M tokens | batch=2, grad_accum=16, fp16, grad_ckpt |
| `rca_1b` | ~150M tokens | ~250M tokens | batch=1, grad_accum=32, fp16, grad_ckpt |

---

## Training

RCA includes a full-featured trainer with distributed training, mixed precision, and gradient checkpointing.

### Single GPU Training (T4 / P100)

```python
from rca import RCAConfig, RCAModel, RCATrainer, TrainingArguments

# 1. Config
config = RCAConfig.rca_100m()
config.vocab_size = 32000

# 2. Model
model = RCAModel(config)

# 3. Dataset (any PyTorch Dataset returning {"input_ids": tensor, "labels": tensor})
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data, seq_len=4096):
        self.data = data      # your tokenized data tensor
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}

# 4. Training args
args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    warmup_steps=200,
    fp16=True,                    # mixed precision
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
)

# 5. Train
trainer = RCATrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

### Multi-GPU Training (DDP)

```bash
torchrun --nproc_per_node=4 train.py
```

```python
args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    fp16=True,
    use_ddp=True,
    # ...
)
```

### Large Model Training (FSDP — 5B+)

```bash
torchrun --nproc_per_node=8 train.py
```

```python
config = RCAConfig.rca_5b()
config.vocab_size = 32000

args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=64,
    bf16=True,
    use_fsdp=True,
    # ...
)
```

### TPU Training (XLA)

```python
args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=8,
    use_xla=True,
    bf16=True,
    # ...
)
```

---

## Custom Architecture

Full control over every parameter:

```python
from rca import RCAConfig, RCAModel

config = RCAConfig(
    vocab_size=50257,
    state_dim=768,
    n_layers=24,
    n_heads=12,

    # Ultra-Reasoning zones
    use_ultra_reasoning=True,
    use_glu_ffn=True,
    ssm_zone_end=0.6,        # First 60% = SSM (stream of consciousness)
    gla_zone_end=0.85,       # Next 25%  = GLA (working memory)
    # Remaining 15%          = Reasoning (focus)

    # GLA settings
    gla_heads=12,
    gla_expand_k=1.0,
    gla_expand_v=2.0,

    # Reasoning settings
    sliding_window_size=512,
    num_memory_tokens=32,

    # Performance
    gradient_checkpointing=True,
    max_seq_len=4096,
    dropout=0.1,
)

model = RCAModel(config)
print(model.get_layer_zones())
```

### Key Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `use_ultra_reasoning` | `False` | Enable 3-zone architecture |
| `use_glu_ffn` | `False` | SwiGLU FFN instead of standard GELU |
| `ssm_zone_end` | `0.6` | Fraction of layers for SSM zone |
| `gla_zone_end` | `0.85` | Fraction of layers for SSM + GLA zones |
| `gradient_checkpointing` | `False` | Trade compute for memory savings |
| `sliding_window_size` | `512` | Local attention window in reasoning zone |
| `num_memory_tokens` | `32` | Global context bookmarks in reasoning zone |
| `use_mqa` | `False` | Multi-Query Attention for KV savings |

---

## Model Export

### Safetensors (recommended for fast loading)

```python
from rca import export_safetensors, load_safetensors, RCAModel

# Export
export_safetensors(model, "./my_model_safetensors/")

# Load
model = load_safetensors(RCAModel, "./my_model_safetensors/")
```

### GGUF (for llama.cpp / edge inference)

```python
from rca import export_gguf

# Full precision
export_gguf(model, "./my_model.gguf", quantization="f16")

# Quantized (smaller, faster on CPU)
export_gguf(model, "./my_model_q8.gguf", quantization="q8_0")
export_gguf(model, "./my_model_q4.gguf", quantization="q4_0")
```

**Quantization options:**

| Format | Size vs f32 | Quality | Use Case |
|---|---|---|---|
| `f32` | 1× | Lossless | Research / debugging |
| `f16` | 0.5× | Near-lossless | GPU inference |
| `q8_0` | 0.25× | Minimal loss | CPU / edge inference |
| `q4_0` | 0.125× | Some loss | Mobile / embedded |

### PyTorch Native Save/Load

```python
# Save
model.save_pretrained("./my_model/")

# Load
model = RCAModel.from_pretrained("./my_model/")
```

---

## Performance Features

### Gradient Checkpointing

Trades ~30% compute for ~60% memory savings. Enabled by default for 500M+ presets.

```python
config = RCAConfig.rca_500m()
# config.gradient_checkpointing is already True

# Or enable manually:
config.gradient_checkpointing = True
```

### Triton-Accelerated Parallel Scan

The SSM parallel scan automatically uses Triton kernels on NVIDIA GPUs:

```python
from rca import TRITON_AVAILABLE
print(f"Triton available: {TRITON_AVAILABLE}")
# Automatic — no code changes needed
```

### torch.compile

Fuses operations for additional speedup:

```python
args = TrainingArguments(
    use_torch_compile=True,
    compile_mode="reduce-overhead",  # or "max-autotune"
    # ...
)
```

### Fused RMSNorm

All normalization layers use an optimized `rsqrt(mean(x²))` implementation that is both faster and compatible with `torch.compile` kernel fusion.

---

## Kaggle / Colab Quick Training

Complete training script for free-tier GPUs:

```python
# Install
# !pip install rca-arch[gpu]

import torch
from rca import RCAConfig, RCAModel, RCATrainer, TrainingArguments

# Use 100M preset for T4
config = RCAConfig.rca_100m()
config.vocab_size = 32000

model = RCAModel(config)
print(f"Model: {model.count_parameters():,} params")
print(f"Zones: {model.get_layer_zones()}")

# Create a simple dataset (replace with your data)
from torch.utils.data import TensorDataset
data = torch.randint(0, 32000, (1000, 4097))
dataset = TensorDataset(data[:, :-1], data[:, 1:])

# Train
args = TrainingArguments(
    output_dir="/kaggle/working/rca_output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=3e-4,
    warmup_steps=100,
    fp16=True,
    logging_steps=5,
    save_steps=200,
)

trainer = RCATrainer(model=model, args=args, train_dataset=dataset)
trainer.train()

# Export
from rca import export_safetensors
export_safetensors(model, "/kaggle/working/rca_safetensors/")
```

---

## Running Tests

```bash
# Run the full test suite
python run_tests.py

# Run a standalone integration test
python run_single_test.py
```

---

## Project Structure

```
src/rca/
├── __init__.py          # Public API
├── config.py            # RCAConfig with presets
├── modeling/
│   ├── rca_model.py     # RCAModel (SSM/GLA/Reasoning blocks)
│   └── outputs.py       # Output dataclasses
├── layers/
│   ├── ssm.py           # Selective State Space Model
│   ├── gla.py           # Gated Linear Attention (vectorized)
│   ├── sliding_attention.py  # Sliding Window + Memory Tokens
│   ├── attention.py     # Efficient Attention (MQA/Rotary)
│   ├── scan.py          # Parallel scan (PyTorch/Triton/XLA)
│   ├── norm.py          # Fused RMSNorm, DeepNorm
│   └── positions.py     # ALiBi, Rotary embeddings
├── trainer.py           # RCATrainer (DDP/FSDP/XLA/compile)
├── converter.py         # Safetensors + GGUF export
├── generator.py         # Text generation utilities
└── utils/
    ├── benchmark.py     # Performance benchmarking
    └── export.py        # ONNX export, save/load
```

---

## Citation

```bibtex
@software{rca2024,
  title={RCA: Recursive Compression Architecture},
  author={Rajaaditya, R.},
  year={2024},
  url={https://github.com/rajaaditya/rca-arch}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
