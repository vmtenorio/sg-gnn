import torch
import torch.nn as nn

import dgl
import dgl.function as fn

import numpy as np
import networkx as nx

def node_homophily_pernode(graph, y):
    # Adapted from https://docs.dgl.ai/en/latest/_modules/dgl/homophily.html#node_homophilyç
    # Simply removing the mean in the return
    with graph.local_scope():
        # Handle the case where graph is of dtype int32.
        src, dst = graph.edges()
        src, dst = src.long(), dst.long()
        # Compute y_v = y_u for all edges.
        graph.edata["same_class"] = (y[src] == y[dst]).float()
        graph.update_all(
            fn.copy_e("same_class", "m"), fn.mean("m", "same_class_deg")
        )
        return graph.ndata["same_class_deg"]
    

def compute_features(A:np.ndarray, ftype:str = "role"):
    (num_nodes,num_nodes) = A.shape
    degs = np.sum(A,axis=0)
    egonet_inds = list(map(lambda i:np.concatenate(([i],np.where(A[i]==1)[0])),np.arange(num_nodes)))
    egonet = list(map(lambda inds:A[inds][:,inds],egonet_inds))

    G = nx.from_numpy_array(A)
    G.remove_edges_from(nx.selfloop_edges(G))

    if ftype == "local":
        raise NotImplementedError
    elif ftype == "global":
        f = [None]*7
        f[0] = list(nx.eccentricity(G).values())
        f[1] = list(nx.pagerank(G).values())
        f[2] = list(nx.eigenvector_centrality(G, max_iter=int(1e7)).values())
        f[3] = list(nx.betweenness_centrality(G).values())
        f[4] = list(nx.closeness_centrality(G).values())
        try:
            f[5] = list(nx.katz_centrality(G).values())
        except nx.PowerIterationFailedConvergence:
            f[5] = [1.]*num_nodes
        f[6] = list(nx.core_number(G).values())
    elif ftype == "role":
        f = [None]*7
        f[0] = degs # Degree
        f[1] = list(map(np.sum,egonet)) # Within Egonet Degrees
        f[2] = list(map(lambda inds:np.sum(degs[inds]),egonet_inds)) # Degree sum in egonet
        f[3] = [f[1][i]/f[2][i] if f[2][i]>0 else 0 for i in range(num_nodes)] # Ratio of within-egonet edges to egonet boundary edges
        f[4] = [1-f[3][i] if f[2][i]>0 else 0 for i in range(num_nodes)] # Ratio of non-egonet edges to egonet boundary edges
        f[5] = np.diag(np.linalg.matrix_power(A,3)) # 3-cliques (triangles)
        f[6] = [2*f[5][i]/(f[0][i]*(f[0][i]-1)) if f[0][i]>1 else 0 for i in range(num_nodes)] # Local Clustering coefficient
    else:
        raise NotImplementedError("Select an available feature type")
        
    Ft = np.array(f)
    # scale = np.max(Ft,axis=1)
    # scale[scale==0] = 1
    # Ft = Ft/scale[:,None]

    Ah = A + np.eye(num_nodes)
    Dh = np.diag(degs+1)
    F = np.concatenate([Ft.T, np.linalg.inv(Dh)@Ah@Ft.T, Ah@Ft.T],axis=1)
    scale = np.max(F,axis=1)
    scale[scale==0] = 1
    F = F/scale[:,None]

    return F


class GCNHLayer(nn.Module):
    def __init__(self, in_dim, out_dim, S, K):
        super().__init__()

        self.S = S.clone()
        self.N = self.S.shape[0]
        # self.S += torch.eye(self.N, device=self.S.device)
        self.d = self.S.sum(1)
        self.d = torch.where(self.d == 0, 1., self.d)
        self.D_inv = torch.diag(1 / torch.sqrt(self.d))
        self.S = self.D_inv @ self.S @ self.D_inv

        self.K = K
        self.Spow = torch.zeros((self.K, self.N, self.N), device=self.S.device)
        self.Spow[0,:,:] = torch.eye(self.N, device=self.S.device)
        for k in range(1, self.K):
            self.Spow[k,:,:] = self.Spow[k-1,:,:] @ self.S

        self.Spow = nn.Parameter(self.Spow, requires_grad=False)

        self.S = nn.Parameter(self.S, requires_grad=False)

        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.W = nn.Parameter(torch.empty(self.K, self.in_dim, self.out_dim))
        nn.init.kaiming_uniform_(self.W.data)

    def forward(self, _, x): # Graph kept for compatibility
        assert (self.N, self.in_dim) == x.shape
        out = torch.zeros((self.N, self.out_dim), device=x.device)
        for k in range(self.K):
            out += self.Spow[k,:,:] @ x @ self.W[k,:,:]
        return out


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout=0.,
                 nonlin=nn.Tanh(), last_act=nn.Softmax(dim=1),
                 gcnlayer=dgl.nn.GraphConv, gcnlayer_kwargs={}):
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


    def forward(self, graph, x):

        for i in range(self.n_layers - 1):
            x = self.nonlin(self.convs[i](graph, x))
            x = self.dropout(x)
        x = self.convs[-1](graph, x)
        x = self.last_act(x)

        return x


class AdaptiveAggGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_graphs=3, dropout=0.,
                 nonlin=nn.Tanh(), last_act=nn.Softmax(dim=1),
                 gcnlayer=dgl.nn.GraphConv, gcnlayer_kwargs={}, per_node=-1):
        super().__init__()

        self.n_graphs = n_graphs
        self.nonlin = nonlin
        self.last_act = last_act

        self.dropout = nn.Dropout(dropout)

        self.gcn_layer = gcnlayer
        if 'S' in gcnlayer_kwargs and type(gcnlayer_kwargs['S']) == list:
            graphs = gcnlayer_kwargs['S']
            self.convs = nn.ModuleList()
            for i in range(n_graphs):
                gcnlayer_kwargs_i = gcnlayer_kwargs.copy()
                gcnlayer_kwargs_i['S'] = graphs[i]
                self.convs.append(self.gcn_layer(in_dim, hid_dim, **gcnlayer_kwargs_i))
        else:
            self.convs = nn.ModuleList([self.gcn_layer(in_dim, hid_dim, **gcnlayer_kwargs) for _ in range(n_graphs)])
        self.linear = nn.Linear(hid_dim*n_graphs, out_dim)

        if per_node > 0:
            self.alphas = nn.Parameter(torch.ones(n_graphs, per_node))
            self.softmax_alpha = nn.Softmax(dim=0)
        else:
            self.alphas = nn.Parameter(torch.ones(n_graphs))
            self.softmax_alpha = nn.Softmax(dim=0)


    def forward(self, graphs, x):

        alphas = self.softmax_alpha(self.alphas)

        assert len(graphs) == self.n_graphs
        xs = []
        for i in range(self.n_graphs):
            h = self.convs[i](graphs[i], x.clone())
            if alphas.ndim > 1:
                h = alphas[i][:,None] * h
            else:
                h = alphas[i]*h
            h = self.nonlin(h)
            xs.append(h)

        x = torch.cat(xs, 1)
        x = self.dropout(x)

        return self.linear(x)