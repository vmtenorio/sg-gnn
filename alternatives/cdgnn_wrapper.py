"""
CDGNNWrapper — standalone PyG wrapper for CD-GNN (Zhao et al., 2023).

Reference: Zhao, K., Kang, Q., Song, Y., She, R., Wang, S., and Tay, W. P.,
"Graph Neural Convection-Diffusion with Heterophily," IJCAI 2023.
This is a from-scratch reimplementation of the method's convection-diffusion
equation (not a copy of the authors' code).

Design (from function_laplacian_convection.py):
  dX/dt = alpha * (diffusion_term + convection_term - X)

  Diffusion term  : A_sym @ X     (symmetric normalised adj, homophilic smoothing)
  Convection term : gate(Xi, Xj) * (Xi - Xj) @ W  scatter to j   (heterophily aware)

  Discretised with K Euler steps, step size = t/K.
  alpha is sigmoid-activated; the gate is a learned 2*hid -> 1 linear.

Interface:
  CDGNNWrapper(in_dim, hid_dim, out_dim, num_layers, dropout, nonlin, last_act, **kwargs)
  forward(x, edge_index) -> [N, out_dim] logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree


def _sym_agg(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """A_sym @ x  (with self-loops, symmetric normalisation)."""
    N = x.size(0)
    ei, _ = add_self_loops(edge_index, num_nodes=N)
    row, col = ei
    deg = degree(col, N, dtype=x.dtype)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[d_inv_sqrt == float("inf")] = 0.0
    norm = d_inv_sqrt[row] * d_inv_sqrt[col]
    out = torch.zeros_like(x)
    out.scatter_add_(0, col.unsqueeze(-1).expand_as(x[row]), norm.unsqueeze(-1) * x[row])
    return out


class CDGNNWrapper(nn.Module):
    """CD-GNN convection-diffusion model for node classification.

    Parameters
    ----------
    in_dim      : input feature dimension
    hid_dim     : hidden / diffusion dimension
    out_dim     : number of classes
    num_layers  : Euler diffusion steps K
    dropout     : dropout on encoder/decoder
    nonlin      : activation after encoder
    last_act    : activation on final logits
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
        self.decoder = nn.Linear(hid_dim, out_dim)

        # Diffusion time (learnable, positive)
        self.log_t = nn.Parameter(torch.tensor(float(t_init)).log())
        # Overall step weight (alpha)
        self.alpha_logit = nn.Parameter(torch.zeros(1))

        # Convection: gate computes per-edge scalar from [xi || xj], shape [E,1]
        self.gate = nn.Linear(2 * hid_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

        # Convection feature transform: (xi - xj) @ W  -> hid_dim
        self.conv_W = nn.Linear(hid_dim, hid_dim, bias=False)
        nn.init.xavier_normal_(self.conv_W.weight, gain=1.414)

        # Learnable mix between diffusion and convection outputs
        self.lamda = nn.Parameter(torch.tensor(0.0))

        # Batch norms for diffusion and convection branches (from original code)
        self.bn_diff = nn.BatchNorm1d(hid_dim)
        self.bn_conv = nn.BatchNorm1d(hid_dim)

    # ------------------------------------------------------------------
    def _step(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """One ODE step: diffusion + convection contribution."""
        N = x.size(0)
        row, col = edge_index  # src, dst

        # --- Diffusion branch: A_sym @ x  (already excludes self-loops here;
        #     _sym_agg adds them internally)
        diff = _sym_agg(x, edge_index)           # [N, H]

        # --- Convection branch -----------------------------------------
        xi = x[row]                               # [E, H]  source features
        xj = x[col]                               # [E, H]  dest features
        # Gate (attention scalar per edge)
        g = torch.tanh(self.gate(torch.cat([xi, xj], dim=-1))).squeeze(-1)  # [E]
        # Directional feature: ReLU( (xi - xj) @ W ) * xj   (from paper eq.)
        v_ij = F.relu(self.conv_W(xi - xj)) * xj  # [E, H]
        # Scatter-add weighted by gate -> convection agg at each dst node
        conv = torch.zeros(N, x.size(1), device=x.device)
        conv.scatter_add_(0, col.unsqueeze(-1).expand_as(v_ij), g.unsqueeze(-1) * v_ij)

        # Batch-normalise both branches and mix
        diff = self.bn_diff(diff)
        conv = self.bn_conv(conv)
        lam = torch.sigmoid(self.lamda)
        mixed = lam * conv + (1.0 - lam) * diff   # [N, H]

        return mixed  # f(x) before alpha * (f - x)

    # ------------------------------------------------------------------
    def _diffuse(self, x0: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        t = self.log_t.exp()
        dt = t / self.K
        alpha = torch.sigmoid(self.alpha_logit)

        x = x0
        for _ in range(self.K):
            fx = self._step(x, edge_index)
            x = x + dt * alpha * (fx - x)
        return x

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.nonlin(self.encoder(self.dropout(x)))
        h = self._diffuse(h, edge_index)
        out = self.decoder(self.dropout(h))
        return self.last_act(out)
