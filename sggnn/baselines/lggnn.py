"""
LGGNNWrapper — self-contained reimplementation of LG-GNN (Yu et al., IJCAI 2024).

Reference: Yu, Z., Feng, B., He, D., Wang, Z., Huang, Y., and Feng, Z.,
"LG-GNN: Local-Global Adaptive Graph Neural Network for Modeling Both
Homophily and Heterophily," IJCAI 2024.

NOTE: No public code exists for this paper. This is a reimplementation based on the
paper's description.

Design:
  Two parallel branches:
    Local branch  : 1-hop GCN applied to the graph's edge_index.
                    Captures local neighbourhood structure.
    Global branch : Multi-layer MLP on raw node features (no message passing).
                    Captures global feature distribution irrespective of graph structure.
                    Optionally replaced with multi-hop SGC for a graph-aware global view.

  Learned gating: output = sigmoid(alpha) * local + (1 - sigmoid(alpha)) * global
    - alpha is a learnable scalar (shared across nodes; could be extended per-node).

  Final decoder maps the fused hid_dim vector to logits.

Interface:
  LGGNNWrapper(in_dim, hid_dim, out_dim, num_layers, dropout, nonlin, last_act, **kwargs)
  forward(x, edge_index) -> [N, out_dim] last_act(logits)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SGConv


class LGGNNWrapper(nn.Module):
    """Local-Global GNN for heterophilic node classification.

    Parameters
    ----------
    in_dim      : input feature dimension
    hid_dim     : hidden dimension (both branches produce vectors of this size)
    out_dim     : number of classes
    num_layers  : depth of both branches (>= 1)
    dropout     : dropout probability
    nonlin      : activation between hidden layers
    last_act    : activation on final logits (e.g. Softmax)
    global_hops : hops for the global SGC branch (default 2; set 0 for MLP-only)
    """

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        nonlin=None,
        last_act=None,
        global_hops: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = max(1, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.nonlin = nonlin if nonlin is not None else nn.ReLU()
        self.last_act = last_act if last_act is not None else nn.Softmax(dim=1)

        # Local branch: stack of GCNConv layers
        self.local_convs = nn.ModuleList()
        self.local_convs.append(GCNConv(in_dim, hid_dim))
        for _ in range(self.num_layers - 1):
            self.local_convs.append(GCNConv(hid_dim, hid_dim))

        # Global branch: MLP (or SGC + MLP for graph-aware global)
        if global_hops > 0:
            # SGC pre-aggregates multi-hop info; one linear layer on top
            self.sgc = SGConv(in_dim, hid_dim, K=global_hops)
            global_in = hid_dim
        else:
            self.sgc = None
            global_in = in_dim

        # Remaining MLP layers for global branch
        global_layers = []
        for i in range(self.num_layers - 1):
            global_layers.extend([
                nn.Linear(global_in if i == 0 else hid_dim, hid_dim),
                type(self.nonlin)(),  # fresh instance per layer
                nn.Dropout(dropout),
            ])
        if self.num_layers == 1:
            global_layers.append(nn.Linear(global_in, hid_dim))
        self.global_mlp = nn.Sequential(*global_layers) if global_layers else nn.Identity()

        # Gating: learned scalar alpha, fused = sigmoid(alpha)*local + (1-sigmoid(alpha))*global
        self.alpha = nn.Parameter(torch.zeros(1))   # initialise at 0.5 post-sigmoid

        self.decoder = nn.Linear(hid_dim, out_dim)

    def _local(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.local_convs):
            h = conv(h, edge_index)
            if i < self.num_layers - 1:
                h = self.nonlin(h)
                h = self.dropout(h)
        return h   # [N, hid_dim]

    def _global(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.sgc is not None:
            h = self.nonlin(self.sgc(x, edge_index))
            h = self.dropout(h)
        else:
            h = x
        return self.global_mlp(h)   # [N, hid_dim]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_drop = self.dropout(x)
        local_out = self._local(x_drop, edge_index)   # [N, H]
        global_out = self._global(x_drop, edge_index)  # [N, H]

        gate = torch.sigmoid(self.alpha)               # scalar in (0,1)
        fused = gate * local_out + (1.0 - gate) * global_out

        out = self.decoder(self.dropout(fused))
        return self.last_act(out)
