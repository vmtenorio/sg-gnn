"""
GRANDWrapper — standalone PyG wrapper for GRAND (Chamberlain et al., ICML 2021).

Reference: "GRAND: Graph Neural Diffusion" https://arxiv.org/abs/2106.10934
This is a from-scratch reimplementation of the method's core diffusion equation
(not a copy of any third-party code), written to match the paper's description
while keeping the same forward-pass interface as our other model wrappers.

Design:
  dX/dt = alpha * (A_hat X - X)   where A_hat is sym-normalised adjacency.
  Closed-form Euler discretisation over K steps:
      X(t+dt) = X(t) + dt * alpha * (A_hat X(t) - X(t))
  alpha is a learnable scalar (sigmoid-activated so it stays in (0,1)).

Interface (matches all other wrappers in this project):
  GRANDWrapper(in_dim, hid_dim, out_dim, num_layers, dropout, nonlin, last_act, **kwargs)
  forward(x, edge_index) -> [N, out_dim] last_act(logits)
"""

import torch
import torch.nn as nn
from torch_geometric.utils import add_self_loops, degree


def sym_norm_agg(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """One step of symmetric-normalised adjacency message passing: A_hat @ X."""
    N = x.size(0)
    # Add self-loops then normalise: D^{-1/2} A D^{-1/2}
    ei, _ = add_self_loops(edge_index, num_nodes=N)
    row, col = ei
    deg = degree(col, N, dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
    # A_hat x = sum_j (d_i^{-1/2} d_j^{-1/2}) x_j
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # [E]
    agg = torch.zeros_like(x)
    agg.scatter_add_(0, col.unsqueeze(-1).expand_as(x[row]), norm.unsqueeze(-1) * x[row])
    return agg


class GRANDWrapper(nn.Module):
    """GRAND diffusion model wrapped for node classification.

    Parameters
    ----------
    in_dim      : input feature dimension
    hid_dim     : hidden / diffusion dimension
    out_dim     : number of classes
    num_layers  : number of Euler diffusion steps (K)
    dropout     : dropout probability on the encoder
    nonlin      : activation after the encoder linear (ignored in diffusion phase)
    last_act    : activation applied to final logits (e.g. Softmax)
    t_init      : initial diffusion time (learnable)
    """

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        num_layers: int = 4,
        dropout: float = 0.5,
        nonlin=None,
        last_act=None,
        t_init: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.K = max(1, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.nonlin = nonlin if nonlin is not None else nn.ReLU()
        self.last_act = last_act if last_act is not None else nn.Softmax(dim=1)

        self.encoder = nn.Linear(in_dim, hid_dim)

        # Learnable diffusion time and alpha scalar
        self.log_t = nn.Parameter(torch.tensor(float(t_init)).log())
        self.alpha_logit = nn.Parameter(torch.zeros(1))  # sigmoid -> (0,1)

        self.decoder = nn.Linear(hid_dim, out_dim)

    def _diffuse(self, x0: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Euler discretisation of dX/dt = alpha*(A_hat X - X) for K steps."""
        t = self.log_t.exp()          # total time  > 0
        dt = t / self.K               # step size
        alpha = torch.sigmoid(self.alpha_logit)

        x = x0
        for _ in range(self.K):
            ax = sym_norm_agg(x, edge_index)
            x = x + dt * alpha * (ax - x)
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.nonlin(self.encoder(self.dropout(x)))
        h = self._diffuse(h, edge_index)
        out = self.decoder(self.dropout(h))
        return self.last_act(out)
