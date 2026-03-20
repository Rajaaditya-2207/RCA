"""
Gated Linear Attention (GLA)
============================

DeltaNet-style gated linear attention with data-dependent gating.
Acts as "Working Memory" — associative recall power of attention
while maintaining linear-time complexity.

Author: Rajaaditya.R
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class GatedLinearAttention(nn.Module):
    """
    Gated Linear Attention.

    Key equation:
        S_t = α_t * S_{t-1} + k_t^T v_t
        o_t = q_t * S_t

    Where α_t = σ(W_α · x_t) is a data-dependent gate controlling
    how much old memory to retain.

    - Training: chunkwise parallel computation
    - Inference: O(1) recurrent form
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        expand_k: float = 1.0,
        expand_v: float = 2.0,
        gate_logit_normalizer: int = 16,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim_k = int(dim * expand_k) // num_heads
        self.head_dim_v = int(dim * expand_v) // num_heads
        self.gate_logit_normalizer = gate_logit_normalizer

        self.q_proj = nn.Linear(dim, num_heads * self.head_dim_k, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim_k, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim_v, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim_v, dim, bias=False)

        # Data-dependent gate: controls memory retention
        self.gate_proj = nn.Linear(dim, num_heads, bias=True)

        # Output gate for highway connection
        self.output_gate = nn.Linear(dim, num_heads * self.head_dim_v, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        chunk_size: int = 64,
    ) -> torch.Tensor:
        """
        Parallel forward for training using chunkwise decomposition.

        Args:
            x: [B, S, D] input tensor
            chunk_size: size of chunks for parallel processing

        Returns:
            output: [B, S, D]
        """
        B, S, D = x.shape
        H = self.num_heads

        # Project to Q, K, V
        q = self.q_proj(x).view(B, S, H, self.head_dim_k)
        k = self.k_proj(x).view(B, S, H, self.head_dim_k)
        v = self.v_proj(x).view(B, S, H, self.head_dim_v)

        # Data-dependent forget gate
        gate_logits = self.gate_proj(x)  # [B, S, H]
        alpha = torch.sigmoid(gate_logits / self.gate_logit_normalizer)  # [B, S, H]

        # Normalize keys for stability
        k = k * (self.head_dim_k ** -0.5)

        # Pad sequence to be divisible by chunk_size
        pad = (chunk_size - S % chunk_size) % chunk_size
        if pad > 0:
            q = F.pad(q, (0, 0, 0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))
            alpha = F.pad(alpha, (0, 0, 0, pad))

        S_padded = S + pad
        num_chunks = S_padded // chunk_size

        # Reshape into chunks: [B, C, L, H, d]
        q_c = q.view(B, num_chunks, chunk_size, H, self.head_dim_k)
        k_c = k.view(B, num_chunks, chunk_size, H, self.head_dim_k)
        v_c = v.view(B, num_chunks, chunk_size, H, self.head_dim_v)
        alpha_c = alpha.view(B, num_chunks, chunk_size, H)

        # Process chunks with inter-chunk state propagation
        outputs = []
        state = torch.zeros(
            B, H, self.head_dim_k, self.head_dim_v,
            device=x.device, dtype=x.dtype,
        )

        for c in range(num_chunks):
            qc = q_c[:, c]  # [B, L, H, dk]
            kc = k_c[:, c]  # [B, L, H, dk]
            vc = v_c[:, c]  # [B, L, H, dv]
            ac = alpha_c[:, c]  # [B, L, H]

            chunk_out, state = self._process_chunk(qc, kc, vc, ac, state)
            outputs.append(chunk_out)

        output = torch.cat(outputs, dim=1)  # [B, S_padded, H*dv]

        # Remove padding
        if pad > 0:
            output = output[:, :S]

        # Output gate
        og = torch.sigmoid(self.output_gate(x))  # [B, S, H*dv]
        output = output * og

        return self.out_proj(output)

    def _process_chunk(
        self,
        q: torch.Tensor,   # [B, L, H, dk]
        k: torch.Tensor,   # [B, L, H, dk]
        v: torch.Tensor,   # [B, L, H, dv]
        alpha: torch.Tensor,  # [B, L, H]
        state: torch.Tensor,  # [B, H, dk, dv]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a single chunk with causal intra-chunk attention + state."""
        B, L, H, dk = q.shape
        dv = v.shape[-1]

        outputs = []
        local_state = state.clone()

        for t in range(L):
            qt = q[:, t]  # [B, H, dk]
            kt = k[:, t]  # [B, H, dk]
            vt = v[:, t]  # [B, H, dv]
            at = alpha[:, t].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

            # Decay old state and add new KV pair
            local_state = at * local_state + torch.einsum("bhk,bhv->bhkv", kt, vt)

            # Query the state
            out_t = torch.einsum("bhk,bhkv->bhv", qt, local_state)  # [B, H, dv]
            outputs.append(out_t)

        # Stack: [B, L, H, dv] -> reshape to [B, L, H*dv]
        stacked = torch.stack(outputs, dim=1)  # [B, L, H, dv]
        return stacked.reshape(B, L, H * dv), local_state

    def forward_recurrent(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recurrent forward for generation. O(1) per token.

        Args:
            x: [B, 1, D] single token
            state: [B, H, dk, dv] memory matrix

        Returns:
            output: [B, 1, D]
            new_state: [B, H, dk, dv]
        """
        B = x.shape[0]
        H = self.num_heads

        q = self.q_proj(x).view(B, 1, H, self.head_dim_k)
        k = self.k_proj(x).view(B, 1, H, self.head_dim_k)
        v = self.v_proj(x).view(B, 1, H, self.head_dim_v)

        gate_logits = self.gate_proj(x)  # [B, 1, H]
        alpha = torch.sigmoid(gate_logits / self.gate_logit_normalizer)

        k = k * (self.head_dim_k ** -0.5)

        if state is None:
            state = torch.zeros(
                B, H, self.head_dim_k, self.head_dim_v,
                device=x.device, dtype=x.dtype,
            )

        # Single step: S = α * S + k^T v
        qt = q[:, 0]  # [B, H, dk]
        kt = k[:, 0]  # [B, H, dk]
        vt = v[:, 0]  # [B, H, dv]
        at = alpha[:, 0].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

        new_state = at * state + torch.einsum("bhk,bhv->bhkv", kt, vt)

        # Query
        out = torch.einsum("bhk,bhkv->bhv", qt, new_state)  # [B, H, dv]
        out = out.reshape(B, 1, H * self.head_dim_v)

        og = torch.sigmoid(self.output_gate(x))  # [B, 1, H*dv]
        out = out * og

        return self.out_proj(out), new_state
