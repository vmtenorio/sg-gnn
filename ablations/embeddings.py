"""
Embedding-based candidate graphs used by the R sweep (Part C of
sensitivity_sweep.py / Table X): kNN and epsilon-ball graphs built from raw
features, our role/global structural attributes, and four learned node
embeddings (DeepWalk, Node2Vec, Struc2Vec, GraphWave), giving the paper's
14-view non-Original candidate pool (7 feature types x {kNN, eps-ball}).

External dependencies (not vendored here -- see the top-level README):
  - GraphEmbedding (`from GraphEmbedding.ge import Struc2Vec, Node2Vec, DeepWalk`),
    e.g. https://github.com/shenweichen/GraphEmbedding
  - a GraphWave implementation exposing `graphwave.graphwave.graphwave_alg`
    (Donnat et al., "Learning Structural Node Embeddings via Diffusion Wavelets", KDD 2018)
Both are optional: FeatEmbeddings/RoleFeatEmbeddings/GlobalFeatEmbeddings work
without them; only the four embedding-based graph types require them.
"""

from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances

import sys
from pathlib import Path

import torch
import numpy as np
import torch_geometric.utils as tu

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import compute_features_ig

# ------------------------------------------------
# Embedding models

class EmbeddingGraph:
    def __init__(self, data, nneigh: int = 3, th: float = None, verbose: bool = False):
        self.data = data
        self.N = self.data.x.shape[0]
        self.verbose = verbose

        self.embeddings = self.compute_embeddings()
        self.compute_knn_graph(nneigh)
        self.compute_eps_graph(th)

    def compute_embeddings(self):
        raise NotImplementedError

    def compute_knn_graph(self, nneigh: int = 3):
        assert self.embeddings is not None, "Embeddings not yet computed."
        self.knn_graph = kneighbors_graph(self.embeddings, n_neighbors=nneigh, mode='connectivity', include_self=False)

    def compute_eps_graph(self, th: float = None):
        assert self.embeddings is not None, "Embeddings not yet computed."
        dists = pairwise_distances(self.embeddings)
        if th is None:
            # Match the original graph's edge count by picking the corresponding rank in the sorted distances.
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
        A = torch.zeros((self.N, self.N))
        A[self.data.edge_index[0, :], self.data.edge_index[1, :]] = 1.
        return compute_features_ig(A.numpy(), ftype='role')


class GlobalFeatEmbeddings(EmbeddingGraph):
    def compute_embeddings(self):
        A = torch.zeros((self.N, self.N))
        A[self.data.edge_index[0, :], self.data.edge_index[1, :]] = 1.
        return compute_features_ig(A.numpy(), ftype='global')


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
        # len(chi) = 2*d*J, with d=25 (linspace below) and J=2 (NB_FILTERS in graphwave.py).
        chi, heat_print, taus = graphwave_alg(g, np.linspace(0, 100, 25), taus='auto', verbose=self.verbose)
        return chi


EMBEDDING_TYPES = ['Feat', 'RoleFeat', 'GlobalFeat', 'DeepWalk', 'Node2Vec', 'Struc2Vec', 'GraphWave']
