import numpy as np
import networkx as nx
import torch
import random
import datetime
from igraph import Graph

import logging
from scipy.sparse import csr_matrix, eye, diags
from scipy.sparse.linalg import inv

import torch_geometric.utils as tu

from torch_geometric.typing import SparseTensor
from torch_geometric.data import HeteroData
from torch_geometric.datasets import WebKB, Actor, WikipediaNetwork, Airports, Planetoid

def seed_everything(seed: int):    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_timestring():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")



def node_homophily_per_node(edge_index, y):
    y = y.squeeze(-1) if y.dim() > 1 else y

    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
    else:
        row, col = edge_index

    out = torch.zeros(row.size(0), device=row.device)
    out[y[row] == y[col]] = 1.
    out = tu.scatter(out, col, 0, dim_size=y.size(0), reduce='mean')
    return out


def compute_features_ig(A:np.ndarray, ftype:str = "role", compute_eigvec:bool = True):
    (num_nodes,num_nodes) = A.shape
    degs = np.sum(A,axis=0)
    A_sparse = csr_matrix(A)


    G = nx.from_numpy_array(A)
    G.remove_edges_from(nx.selfloop_edges(G))

    g_i = Graph.from_networkx(G)

    is_directed = G.is_directed()

    if ftype == "local":
        raise NotImplementedError
    elif ftype == "global":
        f = [None]*6
        f[0] = g_i.eccentricity()
        f[1] = g_i.personalized_pagerank(directed=is_directed)
        if compute_eigvec:
            f[2] = g_i.eigenvector_centrality(directed=is_directed)
            norm = np.linalg.norm(f[2])
            f[2] = [el/norm for el in f[2]]
        else:
            f[2] = [1.]*num_nodes

        f[3] = g_i.betweenness(directed=is_directed)
        if is_directed:
            f[3] = [el*(1/((num_nodes-1)*(num_nodes-2))) for el in f[3]]
        else:
            f[3] = [el*(2/((num_nodes-1)*(num_nodes-2))) for el in f[3]]

        f[4] = g_i.closeness()

        f[5] = g_i.coreness()
    elif ftype == "role":
        egonet_inds = list(map(lambda i:np.concatenate(([i],np.where(A[i]==1)[0])),np.arange(num_nodes)))
        egonet = list(map(lambda inds:A[inds][:,inds],egonet_inds))
        
        f = [None]*7
        f[0] = degs # Degree
        f[1] = list(map(np.sum,egonet)) # Within Egonet Degrees
        f[2] = list(map(lambda inds:np.sum(degs[inds]),egonet_inds)) # Degree sum in egonet
        f[3] = [f[1][i]/f[2][i] if f[2][i]>0 else 0 for i in range(num_nodes)] # Ratio of within-egonet edges to egonet boundary edges
        f[4] = [1-f[3][i] if f[2][i]>0 else 0 for i in range(num_nodes)] # Ratio of non-egonet edges to egonet boundary edges

        A3 = A_sparse @ A_sparse @ A_sparse
        f[5] = A3.diagonal() # 3-cliques (triangles)

        f[6] = [2*f[5][i]/(f[0][i]*(f[0][i]-1)) if f[0][i]>1 else 0 for i in range(num_nodes)] # Local Clustering coefficient
    else:
        raise NotImplementedError("Select an available feature type")
        
    Ft = np.array(f)
    Ft = np.nan_to_num(Ft)

    Ah = A_sparse + eye(num_nodes)
    Dh = diags(degs + 1)
    F = np.concatenate([Ft.T, inv(Dh) @ Ah @ Ft.T, Ah @ Ft.T], axis=1)
    scale = np.max(F,axis=1)
    scale[scale==0] = 1
    F = F/scale[:,None]

    return F

def compute_features(A:np.ndarray, ftype:str = "role"):
    (num_nodes,num_nodes) = A.shape
    degs = np.sum(A,axis=0)

    G = nx.from_numpy_array(A)
    G.remove_edges_from(nx.selfloop_edges(G))

    if ftype == "local":
        raise NotImplementedError
    elif ftype == "global":
        f = [None]*7
        try:
            f[0] = list(nx.eccentricity(G).values())
        except nx.NetworkXError:
            # Compute the eccentricity of each connected component separately
            eccs = {}
            for c in nx.connected_components(G):
                eccs.update(nx.eccentricity(G.subgraph(c)))
            f[0] = [eccs[k] for k in sorted(eccs.keys())]
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
        egonet_inds = list(map(lambda i:np.concatenate(([i],np.where(A[i]==1)[0])),np.arange(num_nodes)))
        egonet = list(map(lambda inds:A[inds][:,inds],egonet_inds))
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

    Ah = A + np.eye(num_nodes)
    Dh = np.diag(degs+1)
    F = np.concatenate([Ft.T, np.linalg.inv(Dh)@Ah@Ft.T, Ah@Ft.T],axis=1)
    scale = np.max(F,axis=1)
    scale[scale==0] = 1
    F = F/scale[:,None]

    return F


def get_data_dict(datasets, all_graphs, node_name, n_masks=10, N_train=0.8, N_val=0.1):
    data_dict = {}

    for d, dataset_name in enumerate(datasets):

        logging.info(f"Reading and processing data for {dataset_name}")

        if dataset_name in ['Texas', 'Wisconsin', 'Cornell']:
            data = WebKB('~/.datapyg', dataset_name)
        elif dataset_name == 'Actor':
            data = Actor('~/.datapyg')
        elif dataset_name in ['Chameleon', 'Squirrel']:
            data = WikipediaNetwork('~/.datapyg', dataset_name.lower())
        elif dataset_name in ['Cora', 'CiteSeer']:
            data = Planetoid('~/.datapyg', dataset_name)
        elif dataset_name in ['USA', 'Brazil', 'Europe']:
            data = Airports('~/.datapyg', dataset_name)

        graphs_dataset = all_graphs[dataset_name]
        graphs = list(graphs_dataset.keys())

        data_pyg = HeteroData()

        data_pyg[node_name].x = data[0].x

        data_pyg[node_name].N = data[0].x.size(0)
        data_pyg[node_name].num_feats = data[0].x.size(1)
        data_pyg[node_name].num_classes = data.num_classes

        data_pyg[node_name].y = data[0].y

        if hasattr(data_pyg[node_name], 'train_mask'):
            data_pyg[node_name].train_mask = data[0].train_mask
            data_pyg[node_name].val_mask = data[0].val_mask
            data_pyg[node_name].test_mask = data[0].test_mask
        else:
            train_mask, val_mask, test_mask = create_masks(data_pyg[node_name].N, n_masks, N_train, N_val)
            data_pyg[node_name].train_mask = train_mask
            data_pyg[node_name].val_mask = val_mask
            data_pyg[node_name].test_mask = test_mask

        for g, gname in enumerate(graphs):
            
            if dataset_name == "Chameleon" and gname == "EPS-GraphWave":
                # Skipping due to out of memory error
                continue

            graph = graphs_dataset[gname]
            if graph.ndim == 0:
                graph = graph.item()
            if type(graph) == csr_matrix:
                graph = graph.toarray()
            edge_idx = torch.nonzero(torch.from_numpy(graph)).t().contiguous()
            data_pyg[node_name, gname, node_name].edge_index = edge_idx

        data_pyg[node_name, 'EgoFeats', node_name].edge_index = torch.nonzero(torch.eye(data_pyg[node_name].N)).t().contiguous()

        data_dict[dataset_name] = data_pyg

    return data_dict



def create_masks(N, n_masks, train_frac=0.8, val_frac=0.1):
    # Calculate the number of samples for each split
    num_train = int(train_frac * N)
    num_val = int(val_frac * N)
    num_test = N - num_train - num_val  # Ensure all samples are used

    # Initialize masks with False
    train_mask = torch.zeros(N, n_masks, dtype=torch.bool)
    val_mask = torch.zeros(N, n_masks, dtype=torch.bool)
    test_mask = torch.zeros(N, n_masks, dtype=torch.bool)

    for col in range(n_masks):
        # Generate indices for each split
        indices = torch.randperm(N)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train + num_val]
        test_indices = indices[num_train + num_val:]

        # Set the corresponding mask values to True for each column
        train_mask[train_indices, col] = True
        val_mask[val_indices, col] = True
        test_mask[test_indices, col] = True

    return train_mask, val_mask, test_mask

