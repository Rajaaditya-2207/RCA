"""
State Space Models
==================

Selective SSM (Mamba-style) and Simple SSM implementations.

Author: Rajaaditya.R
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .scan import compute_parallel_scan


class SelectiveStateSpaceModel(nn.Module):
    """
    Selective SSM (like Mamba).

    Key innovation: A, B are INPUT-DEPENDENT.
    This allows the model to selectively keep/discard information.
    """

    def __init__(self, input_dim: int, state_dim: int, use_full_matrix: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.use_full_matrix = use_full_matrix

        # Selective parameters (input-dependent)
        self.dt_proj = nn.Linear(input_dim, state_dim, bias=True)
        self.B_proj = nn.Linear(
            input_dim,
            state_dim if not use_full_matrix else state_dim * state_dim,
            bias=False,
        )
        self.C_proj = nn.Linear(
            state_dim,
            input_dim if not use_full_matrix else state_dim,
            bias=False,
        )

        # A matrix
        if use_full_matrix:
            self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)
        else:
            self.A_log = nn.Parameter(torch.randn(state_dim) * 0.5 - 1)

        self.D = nn.Parameter(torch.zeros(input_dim))

        # Initialize
        with torch.no_grad():
            self.dt_proj.bias.fill_(1.0)

    def compute_params(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute discretized parameters from input."""
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        B, S, D_in = x.shape

        dt = F.softplus(self.dt_proj(x))
        B_proj = self.B_proj(x)

        if self.use_full_matrix:
            B_proj = B_proj.view(B, S, self.state_dim, self.state_dim)
            A = self.A.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
        else:
            A = -torch.exp(self.A_log.unsqueeze(0).unsqueeze(0))
            A = A.expand(B, S, -1)

        if squeeze:
            dt = dt.squeeze(1)
            if not self.use_full_matrix:
                A = A.squeeze(1)
            B_proj = B_proj.squeeze(1)

        return dt, A, B_proj

    def forward_sequential(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequential forward for generation. O(1) per token."""
        B = x.shape[0]
        dt, A, B_proj = self.compute_params(x)

        if state is None:
            state = torch.zeros(B, self.state_dim, device=x.device)

        if self.use_full_matrix:
            A_state = torch.einsum("bd,bdd->bd", state, A.squeeze(1))
            h_new = A_state + (B_proj.squeeze(1) * x)
        else:
            h_new = A * state + (B_proj * x)

        if self.use_full_matrix:
            output = torch.einsum("bd,cd->bc", h_new, self.C_proj.weight) + self.D * x
        else:
            output = F.linear(h_new, self.C_proj.weight) + self.D * x

        return output, h_new

    def forward_parallel(
        self,
        x: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        use_cuda: bool = True,
    ) -> torch.Tensor:
        """Parallel forward for training. O(log S) via scan."""
        B, S, _ = x.shape
        dt, A, B_proj = self.compute_params(x)

        # Discretize
        if self.use_full_matrix:
            A_bar = torch.exp(dt.unsqueeze(-1).unsqueeze(-1) * A)
            B_bar = dt.unsqueeze(-1) * B_proj
        else:
            A_bar = torch.exp(dt * A)
            B_bar = dt * B_proj

        # Input contribution
        if self.use_full_matrix:
            inputs = torch.einsum("bsdd,bsd->bsdd", B_bar, x.unsqueeze(-1))
        else:
            inputs = B_bar * x

        h0 = (
            initial_state
            if initial_state is not None
            else torch.zeros(B, self.state_dim, device=x.device)
        )

        # Use parallel scan
        if self.use_full_matrix:
            # Full matrix: fallback to sequential
            h_all = []
            h = h0
            for s in range(S):
                h = (
                    torch.matmul(h.unsqueeze(1), A_bar[:, s]).squeeze(1)
                    + inputs[:, s].diagonal(dim1=-2, dim2=-1)
                )
                h_all.append(h)
            h_all = torch.stack(h_all, dim=1)
        else:
            h_all = compute_parallel_scan(A_bar, inputs, h0, use_cuda=use_cuda)

        # Output projection
        if self.use_full_matrix:
            outputs = torch.einsum("bsd,cd->bsc", h_all, self.C_proj.weight)
            outputs = outputs + (self.D.unsqueeze(0).unsqueeze(0) * x)
        else:
            outputs = F.linear(h_all, self.C_proj.weight)
            outputs = outputs + (self.D.unsqueeze(0).unsqueeze(0) * x)

        return outputs


class SimpleStateSpaceModel(nn.Module):
    """Simple SSM without selective mechanism."""

    def __init__(self, input_dim: int, state_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim

        self.dt_proj = nn.Linear(input_dim, state_dim, bias=True)
        self.A_log = nn.Parameter(torch.randn(state_dim) * 0.5 - 1)
        self.B_proj = nn.Linear(input_dim, state_dim, bias=False)
        self.C_proj = nn.Linear(state_dim, input_dim, bias=False)
        self.D = nn.Parameter(torch.zeros(input_dim))

        with torch.no_grad():
            self.dt_proj.bias.fill_(1.0)

    def forward_sequential(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        dt = F.softplus(self.dt_proj(x))
        A = -torch.exp(self.A_log)
        B_proj = self.B_proj(x)

        gate = torch.exp(dt * A)

        if state is None:
            state = torch.zeros(B, self.state_dim, device=x.device)

        h_new = gate * state + (dt * B_proj) * x
        output = F.linear(h_new, self.C_proj.weight) + self.D * x

        return output, h_new

    def forward_parallel(
        self,
        x: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        use_cuda: bool = True,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        dt = F.softplus(self.dt_proj(x))
        A = -torch.exp(self.A_log)
        B_proj = self.B_proj(x)

        gate = torch.exp(dt * A)
        inputs = (dt * B_proj) * x

        h0 = (
            initial_state
            if initial_state is not None
            else torch.zeros(B, self.state_dim, device=x.device)
        )
        h_all = compute_parallel_scan(gate, inputs, h0, use_cuda=use_cuda)

        outputs = F.linear(h_all, self.C_proj.weight) + (
            self.D.unsqueeze(0).unsqueeze(0) * x
        )
        return outputs
