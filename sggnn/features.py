"""Structural node attributes: role-based and global centrality features
(Appendix A), plus the sparse variant and k-NN graph builder used for large
graphs (Roman-Empire)."""
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, eye, diags
import torch
from sklearn.neighbors import NearestNeighbors

def compute_features(A:np.ndarray, ftype:str = "role"):
    """Role-based or global structural attributes of Appendix A, concatenated
    with their one-hop mean and sum aggregations and row-scaled to max 1."""
    (num_nodes,num_nodes) = A.shape
    degs = np.sum(A,axis=0)

    G = nx.from_numpy_array(A)
    G.remove_edges_from(nx.selfloop_edges(G))

    if ftype == "global":
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


def compute_role_features_sparse(edge_index, num_nodes):
    """Sparse version of `compute_features(A, 'role')` for graphs too large
    for a dense N x N adjacency (Roman-Empire)."""
    row, col = edge_index.numpy()
    A = csr_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
    degs = np.array(A.sum(axis=1)).flatten()
    f = np.zeros((7, num_nodes), dtype=np.float32)
    f[0] = degs
    for i in range(num_nodes):
        inds = A[i].nonzero()[1]
        inds = np.concatenate([[i], inds])
        sub = A[inds][:, inds]
        f[1][i] = sub.sum()
        f[2][i] = degs[inds].sum()
    f[3] = np.where(f[2] > 0, f[1] / f[2], 0)
    f[4] = np.where(f[2] > 0, 1 - f[3], 0)
    A3 = A @ A @ A
    f[5] = np.array(A3.diagonal(), dtype=np.float32)
    f[6] = np.where(f[0] > 1, 2 * f[5] / (f[0] * (f[0] - 1)), 0)
    A_hat = A + eye(num_nodes)
    d_hat = np.array(A_hat.sum(axis=1)).flatten()
    D_inv = diags(np.where(d_hat > 0, 1.0 / d_hat, 0.0))
    F = np.concatenate([f.T, (D_inv @ A_hat @ f.T), (A_hat @ f.T)], axis=1)
    scale = np.abs(F).max(axis=1, keepdims=True)
    scale[scale == 0] = 1
    return (F / scale).astype(np.float32)


def knn_edge_index(feats, k, num_nodes):
    """Symmetrized k-NN graph as a deduplicated edge_index. The first returned
    neighbor is dropped as the query point itself; with duplicate feature rows
    that is not guaranteed, so a self loop can survive."""
    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1)
    nn_model.fit(feats)
    _, indices = nn_model.kneighbors(feats)
    src = np.repeat(np.arange(num_nodes), k)
    dst = indices[:, 1:].flatten()
    src_full = np.concatenate([src, dst])
    dst_full = np.concatenate([dst, src])
    unique = np.unique(np.stack([src_full, dst_full], axis=1), axis=0)
    return torch.from_numpy(unique.T).long()

