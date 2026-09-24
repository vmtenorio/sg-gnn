import torch
import torch.nn as nn

from torch_geometric.nn import HeteroConv, GCNConv, SGConv, FAConv, Linear, MixHopConv, DirGNNConv
from torch_geometric.utils import to_torch_sparse_tensor


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout=0.,
                 nonlin=nn.Tanh(), last_act=nn.Softmax(dim=1),
                 gcnlayer=GCNConv, gcnlayer_kwargs={}):
        super().__init__()

        self.n_layers = n_layers
        self.nonlin = nonlin
        self.last_act = last_act

        self.dropout = nn.Dropout(dropout)

        self.gcn_layer = gcnlayer
        self.convs = nn.ModuleList()

        if n_layers > 1:
            self.convs.append(self.gcn_layer(in_dim, hid_dim, **gcnlayer_kwargs))
            for _ in range(n_layers - 2):
                self.convs.append(self.gcn_layer(hid_dim, hid_dim, **gcnlayer_kwargs))
            self.convs.append(self.gcn_layer(hid_dim, out_dim, **gcnlayer_kwargs))
        else:
            self.convs.append(self.gcn_layer(in_dim, out_dim, **gcnlayer_kwargs))


    def forward(self, x, edge_index):

        for i in range(self.n_layers - 1):
            x = self.nonlin(self.convs[i](x=x, edge_index=edge_index))
            x = self.dropout(x)
        x = self.convs[-1](x=x, edge_index=edge_index)
        x = self.last_act(x)

        return x


class FBGNNLayer(torch.nn.Module): # The same as MixHop but adding instead of concatenating
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.conv_layer = MixHopConv(in_channels, out_channels, add_self_loops=False)

    def forward(self, x, edge_index):
        x_out = self.conv_layer(x, edge_index)
        x_out = x_out.view(x_out.shape[0], -1, self.out_channels) # Second dimension is the number of powers
        x_out = x_out.sum(1)

        return x_out

class HeteroGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_channels, num_classes, num_layers, dropout, nonlin, last_act, graphs, aggr, orig_layer, struc_layer):
        super().__init__()
        self.aggr = aggr
        self.in_dim = in_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.graphs = graphs

        self.nonlin = nonlin
        self.last_act = last_act

        self.orig_layer = orig_layer
        self.struc_layer = struc_layer

        if self.aggr == 'cat':
            self.in_dim_layers = self.hidden_channels*len(self.graphs)
        else:
            self.in_dim_layers = self.hidden_channels

        self.build_convs()
        
        self.lin = Linear(self.in_dim_layers, num_classes)
        self.dropout = nn.Dropout(dropout)

    def compute_layer_dict(self, in_dim, out_dim):
        layer_dict = {}
        for gname in self.graphs:
            if 'KNN' in gname or 'EPS' in gname:
                layer_dict[('web', gname, 'web')] = self.struc_layer(in_dim, out_dim)
            else:
                layer_dict[('web', gname, 'web')] = self.orig_layer(in_dim, out_dim)
        return layer_dict

    def build_convs(self):
        self.convs = torch.nn.ModuleList()

        self.convs.append(HeteroConv(self.compute_layer_dict(self.in_dim, self.hidden_channels), aggr=self.aggr))
        for _ in range(self.num_layers-1):
            self.convs.append(HeteroConv(self.compute_layer_dict(self.in_dim_layers, self.hidden_channels), aggr=self.aggr))

    def forward(self, x_dict, edge_index):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index)
            x_dict = {key: self.nonlin(x) for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        return self.last_act(self.lin(x_dict['web']))

class AdaptiveAggGCN(HeteroGNN):
    def __init__(self, in_dim, hidden_channels, num_classes, num_layers, dropout, nonlin, last_act, graphs, orig_layer, struc_layer, per_node=-1):
        super().__init__(in_dim, hidden_channels, num_classes, num_layers, dropout, nonlin, last_act, graphs, 'cat', orig_layer, struc_layer)

        if per_node < 0:
            self.alphas = nn.Parameter(torch.ones(len(graphs)))
        else:
            self.alphas = nn.Parameter(torch.ones(len(graphs), per_node))

        self.softmax_alpha = nn.Softmax(dim=0)
    
    def build_convs(self):
        self.conv = HeteroConv(self.compute_layer_dict(self.in_dim, self.hidden_channels), aggr='cat')

    def forward(self, x_dict, edge_index):
        alphas = self.softmax_alpha(self.alphas)

        x_dict = self.conv(x_dict, edge_index)
        x_dict = {key: self.nonlin(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        x = x_dict['web']
        if alphas.ndim > 1:
            x = x*alphas.repeat_interleave(self.hidden_channels, dim=0).T
        else:
            x = x*alphas.repeat_interleave(self.hidden_channels)[None,:]
        
        return self.last_act(self.lin(x))

class gfGNN(nn.Module):
    def __init__(self, in_dim, hidden_channels, num_classes, num_layers, dropout, nonlin, last_act, K=3):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.nonlin = nonlin
        self.last_act = last_act

        self.sgc = SGConv(self.in_dim, self.hidden_channels, K=K)

        self.dropout = nn.Dropout(dropout)

        lin_modules = [nn.Linear(self.hidden_channels, self.hidden_channels), self.nonlin, self.dropout]*(self.num_layers-2) + \
            [nn.Linear(self.hidden_channels, num_classes)]
        self.mlp = nn.Sequential(*lin_modules)

    def forward(self, x, edge_index):
        h = self.sgc(x, edge_index)
        h = self.nonlin(h)
        h = self.dropout(h)
        return self.last_act(self.mlp(h))

class FAGCN(nn.Module):
    """Follows the official FAGCN (bdy9527/FAGCN, src/model.py):
        h = dropout(x); h = relu_or_nonlin(t1(h)); h = dropout(h); raw = h
        for each layer: h = FAConv(h, raw, edge_index)   # FAConv already adds eps*raw internally
        return t2(h)
    t1: in_dim -> hidden_dim, t2: hidden_dim -> num_classes. Propagation runs at `hidden_dim`
    width, so parameter matching scales hidden_dim like every other baseline. The activation
    after t1 is the shared `nonlin` (the official repo uses ReLU). `eps` defaults to the
    official repo's 0.3 (PyG's FAConv default is 0.1)."""

    def __init__(self, in_dim, hidden_dim, num_classes, num_layers, dropout, nonlin, last_act,
                 eps=0.3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.nonlin = nonlin
        self.last_act = last_act

        self.lin0 = nn.Linear(in_dim, hidden_dim, bias=False)  # t1

        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList([
            FAConv(hidden_dim, eps=eps, dropout=dropout) for _ in range(num_layers)
        ])

        self.linout = nn.Linear(hidden_dim, num_classes)  # t2

    def forward(self, x, x_0, edge_index):
        h = self.dropout(x)
        h = self.nonlin(self.lin0(h))
        h = self.dropout(h)
        raw = h
        for conv in self.convs:
            h = conv(x=h, x_0=raw, edge_index=edge_index)

        return self.last_act(self.linout(h))

class DirGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_channels, num_classes, num_layers, dropout, nonlin, last_act, gcnlayer=GCNConv):
        super().__init__()

        self.n_layers = num_layers
        self.hid_dim = hidden_channels
        self.nonlin = nonlin
        self.last_act = last_act

        self.dropout = nn.Dropout(dropout)
        
        self.gcn_layer = gcnlayer

        self.convs = nn.ModuleList()

        if self.n_layers > 1:
            self.convs.append(DirGNNConv(self.gcn_layer(in_dim, self.hid_dim)))
            for _ in range(self.n_layers - 2):
                self.convs.append(DirGNNConv(self.gcn_layer(self.hid_dim, self.hid_dim)))
            self.convs.append(DirGNNConv(self.gcn_layer(self.hid_dim, num_classes)))
        else:
            self.convs.append(DirGNNConv(self.gcn_layer(in_dim, num_classes)))


    def forward(self, x, edge_index):

        for i in range(self.n_layers - 1):
            x = self.nonlin(self.convs[i](x=x, edge_index=edge_index))
            x = self.dropout(x)
        x = self.convs[-1](x=x, edge_index=edge_index)
        x = self.last_act(x)

        return x

class H2GCN(nn.Module):
    def __init__(
            self,
            feat_dim: int,
            hidden_dim: int,
            class_dim: int,
            K: int = 2,
            dropout: float = 0.5,
            nonlin: nn.Module = nn.ReLU(),
            last_act: nn.Module = nn.Identity()
    ):
        super(H2GCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.k = K
        self.act = nonlin
        self.last_act = last_act
        self.w_embed = nn.Parameter(
            torch.zeros(size=(feat_dim, hidden_dim)),
            requires_grad=True
        )
        self.w_classify = nn.Parameter(
            torch.zeros(size=((2 ** (self.k + 1) - 1) * hidden_dim, class_dim)),
            requires_grad=True
        )
        self.initialized = False
        self.a1 = None
        self.a2 = None
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.w_embed)
        nn.init.xavier_uniform_(self.w_classify)

    @staticmethod
    def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
        csp = sp_tensor.coalesce()
        return torch.sparse_coo_tensor(
            indices=csp.indices(),
            values=torch.where(csp.values() > 0, 1, 0),
            size=csp.size(),
            dtype=torch.float
        )

    @staticmethod
    def _spspmm(sp1: torch.sparse.Tensor, sp2: torch.sparse.Tensor) -> torch.sparse.Tensor:
        assert sp1.shape[1] == sp2.shape[0], 'Cannot multiply size %s with %s' % (sp1.shape, sp2.shape)
        sp1, sp2 = sp1.coalesce(), sp2.coalesce()
        m, k = sp1.shape[0], sp2.shape[1]
        prod = (sp1 @ sp2).coalesce()
        indices, values = prod.indices(), prod.values()
        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(m, k),
            dtype=torch.float
        )

    @classmethod
    def _adj_norm(cls, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        n = adj.size(0)
        d_diag = torch.pow(torch.sparse.sum(adj, dim=1).values(), -0.5)
        d_diag = torch.where(torch.isinf(d_diag), torch.full_like(d_diag, 0), d_diag)
        d_tiled = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=d_diag,
            size=(n, n)
        )
        return cls._spspmm(cls._spspmm(d_tiled, adj), d_tiled)

    def _prepare_prop(self, adj):
        n = adj.size(0)
        device = adj.device
        self.initialized = True
        sp_eye = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=[1.0] * n,
            size=(n, n),
            dtype=torch.float
        ).to(device)
        a1 = self._indicator(adj - sp_eye)
        a2 = self._indicator(self._spspmm(adj, adj) - adj - sp_eye)
        self.a1 = self._adj_norm(a1)
        self.a2 = self._adj_norm(a2)

    def forward(self, x: torch.FloatTensor, edge_index: torch.Tensor) -> torch.FloatTensor:
        adj = to_torch_sparse_tensor(edge_index)
        if not self.initialized:
            self._prepare_prop(adj)
        rs = [self.act(torch.mm(x, self.w_embed))]
        for i in range(self.k):
            r_last = rs[-1]
            r1 = torch.spmm(self.a1, r_last)
            r2 = torch.spmm(self.a2, r_last)
            rs.append(self.act(torch.cat([r1, r2], dim=1)))
        r_final = torch.cat(rs, dim=1)
        r_final = self.dropout(r_final)
        return self.last_act(torch.mm(r_final, self.w_classify))
