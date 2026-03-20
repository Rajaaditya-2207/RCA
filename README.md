# RCA Architecture

RCA (Ultra-Reasoning Continuous Architecture) is a bleeding-edge, highly efficient hybrid sequence model framework. It is expressly designed to radically solve the quadratic attention bottleneck of traditional Transformers. 

By unifying linear-time State Space Models (SSM) and Gated Linear Attention with highly targeted local Sliding Window Attention, RCA provides an effectively infinite context window, flat constant-time generation speeds, and transformer-level reasoning capabilities.

## Key Capabilities
- **O(1) Memory Footprint:** The vast majority of the network runs natively in a compressed state matrix, allowing you to process millions of tokens using a flat, constant amount of memory.
- **Ultra-Fast Generation:** Because historical context does not continuously grow the Key-Value cache, text generation speed is significantly faster than standard Transformers at long contexts.
- **Long-Term Reasoning:** Highly targeted local attention modules maintain sharp, specific reasoning facts that pure linear models natively struggle with. It is perfectly tuned for multi-turn agentic workflows.

## Installation

You can install the official package directly from PyPI:
```bash
pip install rca-arch
```

## Quick Start (Presets)

Using the model in your PyTorch workflow is incredibly straightforward. If you want a standard model, you can initialize an architecture using one of our scaled presets:

```python
import torch
from rca import RCAConfig, RCAModel

# 1. Select your target scale
# Available presets: "rca_tiny", "rca_small", "rca_base", "rca_large", "rca_xl", "rca_ultra"
config = RCAConfig.rca_small()
config.vocab_size = 32000

# 2. Initialize the model
model = RCAModel(config)

# 3. Standard forward pass
x = torch.randint(0, 32000, (2, 1024)) # [batch_size, sequence_length]
logits, cache = model(x)

print("Output shape:", logits.shape)
```

## Advanced Configuration (Custom Models)

If you are a researcher or advanced user, you can bypass the preset initializers to design your own precise architectural layout. The `RCAConfig` dataclass grants you complete granular control over internal representation dimensions, feature flags, and hybrid zone boundaries.

```python
import torch
from rca import RCAConfig, RCAModel

# 1. Define your custom RCA Architecture layout
config = RCAConfig(
    vocab_size=50257,        # Standard tokenizer vocabulary size
    state_dim=768,           # Dimensionality (hidden_size) of the model
    n_layers=24,             # Total depth of the network
    n_heads=12,              # Number of attention/KV heads

    # -----------------------------
    # Cognitive Zone Boundaries
    # -----------------------------
    use_ultra_reasoning=True, # Activate the 3-zone architecture (SSM -> GLA -> Attention)
    use_glu_ffn=True,         # Use Gated Linear Units (SwiGLU) for MLPs 
    
    # Zone 1: The Context Engine (Long-Term Memory)
    # The first % of layers use O(1) State Space Models (Mamba-style)
    ssm_zone_end=0.6,         # The first 60% of layers will strictly use SSMs
    ssm_expand=2,             # State Space expansion factor

    # Zone 2: The Working Memory Engine
    # The next % of layers use Gated Linear Attention for data-dependent focus
    gla_zone_end=0.85,        # From 60% to 85% depth, the model switches to GLA
    gla_heads=8,              # Number of Gated Linear Attention heads
    gla_expand_k=1.0,         # KV State expansion ratios for GLA
    gla_expand_v=2.0,

    # Zone 3: The Reasoning Engine
    # The remaining layers (85%-100%) use strict Sliding Window Attention
    sliding_window_size=512,  # The local attention chunk window limit
    num_memory_tokens=32,     # Global summary bookmarks prepended for non-local context
    
    # Regularization
    dropout=0.1
)

# 2. Initialize the highly custom model
model = RCAModel(config)

# 3. Rapid Long-Context routing
x = torch.randint(0, 50257, (1, 8000))
logits, cache = model(x)
```

### Explaining Key Custom Attributes:
- **`use_ultra_reasoning`**: This must be set to `True` for the automatic triplet routing to function! If this is false, your model will behave structurally generically.
- **`ssm_zone_end` & `gla_zone_end`**: The internal layout is strictly gated by positional splits. A setting of `ssm_zone_end=0.5` mathematically allocates exactly the first block of $50\%$ of your `n_layers` to entirely rely on flat $O(1)$ linear State Space Models. The subsequent layers then check the next zone cut-off.
- **`sliding_window_size` & `num_memory_tokens`**: Any standard attention matrices allocated (i.e. those past the `gla_zone_end` marker) natively utilize a sliding window memory cut-off restricted precisely by `sliding_window_size`. This ensures extreme $O(N)$ linear memory scaling everywhere, however to avoid destroying complete global sequences, we inject specialized dummy placeholders regulated by `num_memory_tokens`.

## Running Tests & Verifications

If you clone the repository from GitHub, you can execute the comprehensive test suite to verify the custom kernel behavior, chunkwise parallel processing, and mathematical stability.

```bash
# Run the entire test suite across all architecture layers (SSM, GLA, Slidng Window)
python run_tests.py

# Run a localized standalone integration test
python run_single_test.py
```
