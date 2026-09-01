"""
Precomputes the 14-view (+ Original) candidate-graph pool used by the R sweep
(Part C of sensitivity_sweep.py / Table X): for each dataset, a kNN graph and
an epsilon-ball graph (k = 3 neighbors, matched-edge-count threshold) built
from each of 7 feature/embedding types (raw features, role/global structural
attributes, DeepWalk, Node2Vec, Struc2Vec, GraphWave).

This is a separate, one-off precomputation step (rather than being folded
into sensitivity_sweep.py directly) because the embedding methods are
expensive to compute repeatedly. Run this once; sensitivity_sweep.py's R
sweep then loads its output.

Requires the two external embedding libraries documented in embeddings.py's
module docstring (only for the DeepWalk/Node2Vec/Struc2Vec/GraphWave graphs;
comment out `EMBEDDING_TYPES` entries you don't need if you don't have them
installed).

Output:
  results/candidate_graphs/candidate_graphs.npz  -- {dataset}_{graph_name} -> dense adjacency
"""

import sys
import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from torch_geometric.datasets import WebKB, Actor, WikipediaNetwork, Airports, Planetoid

import embeddings

OUT_DIR = ROOT / 'results' / 'candidate_graphs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

datasets = ['Texas', 'Wisconsin', 'Cornell', 'Actor', 'Chameleon', 'Squirrel', 'Cora', 'CiteSeer', 'USA', 'Brazil', 'Europe']
nneigh = 3

all_graphs = {}
for dname in datasets:
    print(datetime.datetime.now(), "Starting dataset", dname, flush=True)

    if dname in ['Texas', 'Wisconsin', 'Cornell']:
        data = WebKB('~/.datapyg', dname)
    elif dname == 'Actor':
        data = Actor('~/.datapyg')
    elif dname in ['Chameleon', 'Squirrel']:
        data = WikipediaNetwork('~/.datapyg', dname.lower())
    elif dname in ['Cora', 'CiteSeer']:
        data = Planetoid('~/.datapyg', dname)
    elif dname in ['USA', 'Brazil', 'Europe']:
        data = Airports('~/.datapyg', dname)

    N = data[0].x.shape[0]
    A = torch.zeros((N, N))
    A[data[0].edge_index[0, :], data[0].edge_index[1, :]] = 1.

    all_graphs[dname] = {'Original': A.numpy()}

    for embname in embeddings.EMBEDDING_TYPES:
        print(datetime.datetime.now(), "  embedding:", embname, flush=True)
        emb_model = getattr(embeddings, embname + 'Embeddings')
        emb = emb_model(data[0], nneigh=nneigh, th=None, verbose=False)

        all_graphs[dname][f'EPS-{embname}'] = emb.eps_graph.copy()
        all_graphs[dname][f'KNN-{embname}'] = emb.knn_graph.toarray()

array_dict = {f'{dname}_{gname}': arr for dname, graphs in all_graphs.items() for gname, arr in graphs.items()}
np.savez_compressed(str(OUT_DIR / 'candidate_graphs.npz'), **array_dict)
print(f"Done. Saved to {OUT_DIR / 'candidate_graphs.npz'}")
