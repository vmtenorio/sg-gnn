"""
Attribute/embedding classes behind the cached k-NN and epsilon-ball graphs,
plus `build_cache`, which builds `embedding_graphs.npz` (Table IV/VI/VII/IX/X's
candidate-graph pool; see the "Graph cache" section of the README).

Each class computes one node representation (raw features, role-based or
global structural attributes, DeepWalk, Node2Vec, Struc2Vec, GraphWave) and
builds from it a k-NN graph and an epsilon-ball graph whose threshold keeps as
many pairs as the original graph has edges. DeepWalk/Node2Vec/Struc2Vec need
the optional GraphEmbedding package; GraphWave needs a graphwave
implementation exposing `graphwave.graphwave.graphwave_alg` -- both installed
separately, not vendored here (see README).
"""
import datetime

from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances

import torch
import numpy as np
import torch_geometric.utils as tu

from .data import load_dataset
from .features import compute_features


def _dense_adj(edge_index, N):
    A = torch.zeros((N, N))
    A[edge_index[0, :], edge_index[1, :]] = 1.
    return A.numpy()


class EmbeddingGraph:
    def __init__(self, data, nneigh: int = 3, th: float = None, verbose: bool = False):
        self.data = data
        self.N = self.data.x.shape[0]
        self.verbose = verbose

        self.embeddings = self.compute_embeddings()
        self.compute_knn_graph(nneigh)
        self.compute_eps_graph(th)

    def compute_embeddings(self):
        pass

    def compute_knn_graph(self, nneigh: int = 3):
        assert self.embeddings is not None, "Embeddings not yet computed."
        self.knn_graph = kneighbors_graph(self.embeddings, n_neighbors=nneigh, mode='connectivity', include_self=False)

    def compute_eps_graph(self, th: float = None):
        assert self.embeddings is not None, "Embeddings not yet computed."
        dists = pairwise_distances(self.embeddings)
        if th is None:
            nedges = self.data.edge_index.shape[1]
            dists_triu = dists[np.triu_indices(self.N, k=1)]
            order = np.argsort(dists_triu)
            th = dists_triu[order][int(nedges)]
        self.eps_graph = (dists <= th).astype(int)
        np.fill_diagonal(self.eps_graph, 0.)


class FeatEmbeddings(EmbeddingGraph):
    def compute_embeddings(self):
        return self.data.x.numpy()


class RoleFeatEmbeddings(EmbeddingGraph):
    def compute_embeddings(self):
        return compute_features(_dense_adj(self.data.edge_index, self.N), ftype='role')


class GlobalFeatEmbeddings(EmbeddingGraph):
    def compute_embeddings(self):
        return compute_features(_dense_adj(self.data.edge_index, self.N), ftype='global')


class Struc2VecEmbeddings(EmbeddingGraph):
    def compute_embeddings(self):
        from GraphEmbedding.ge import Struc2Vec
        g = tu.to_networkx(self.data)
        model = Struc2Vec(g, 10, 80, workers=4, verbose=0)
        model.train(window_size=5, iter=3)
        embeddings = model.get_embeddings()
        return np.stack([embeddings[i] for i in range(g.number_of_nodes())], 0)


class Node2VecEmbeddings(EmbeddingGraph):
    def compute_embeddings(self):
        from GraphEmbedding.ge import Node2Vec
        g = tu.to_networkx(self.data)
        model = Node2Vec(g, walk_length=10, num_walks=80, p=0.25, q=4, workers=4)
        model.train(window_size=5, iter=3)
        embeddings = model.get_embeddings()
        return np.stack([embeddings[i] for i in range(g.number_of_nodes())], 0)


class DeepWalkEmbeddings(EmbeddingGraph):
    def compute_embeddings(self):
        from GraphEmbedding.ge import DeepWalk
        g = tu.to_networkx(self.data)
        model = DeepWalk(g, walk_length=10, num_walks=80, workers=4)
        model.train(window_size=5, iter=3)
        embeddings = model.get_embeddings()
        return np.stack([embeddings[i] for i in range(g.number_of_nodes())], 0)


class GraphWaveEmbeddings(EmbeddingGraph):
    def compute_embeddings(self):
        from graphwave.graphwave.graphwave import graphwave_alg
        g = tu.to_networkx(self.data)
        # Size of chi = 2*d*J where d = 25 (third arg of linspace) and J = 2 (NB_FILTERS in graphwave.py)
        chi, _, _ = graphwave_alg(g, np.linspace(0, 100, 25), taus='auto', verbose=self.verbose)
        return chi


EMBEDDING_NAMES = ['Feat', 'RoleFeat', 'GlobalFeat', 'DeepWalk', 'Node2Vec', 'Struc2Vec', 'GraphWave']

DATASETS = ['Texas', 'Wisconsin', 'Cornell', 'Actor', 'Chameleon', 'Squirrel',
            'Cora', 'CiteSeer', 'USA', 'Brazil', 'Europe']


def build_cache(out_path, datasets=None, embedding_names=None, nneigh=3):
    """Build the k-NN (k=`nneigh`) and epsilon-ball graph cache. Writes a
    single `.npz` at `out_path` with '<dataset>_<graph>' keys, consumed by
    `sggnn.data.load_cached_graphs`.

    Regenerated graphs are algorithmically equivalent to the ones used for
    the paper but, for embedding methods with non-deterministic training
    (DeepWalk/Node2Vec/Struc2Vec's random walks, GraphWave's heat-kernel
    sampling), not bit-identical on every dataset.
    """
    datasets = datasets or DATASETS
    embedding_names = embedding_names or EMBEDDING_NAMES

    all_graphs = {}
    for dname in datasets:
        print(datetime.datetime.now(), "Starting dataset", dname, flush=True)
        data = load_dataset(dname)
        N = data[0].x.shape[0]
        all_graphs[dname] = {'Original': _dense_adj(data[0].edge_index, N)}

        for embname in embedding_names:
            print(datetime.datetime.now(), "Starting embedding", embname, flush=True)
            emb_model_cls = globals()[embname + 'Embeddings']
            emb = emb_model_cls(data[0], nneigh=nneigh, th=None, verbose=False)
            all_graphs[dname][f'EPS-{embname}'] = emb.eps_graph.copy()
            all_graphs[dname][f'KNN-{embname}'] = emb.knn_graph.copy()

    array_dict = {f'{dname}_{gname}': arr
                  for dname, graphs in all_graphs.items() for gname, arr in graphs.items()}
    np.savez_compressed(str(out_path), **array_dict)
    print(f"Saved cache to {out_path}")
    return out_path
