"""
Single alternative graphs with standard GNN layers (Table IV,
`tab:metrics_graphs_embeddings`), plus the single-layer adaptive models and
SG-GNN on the same graph set. `analysis/table4.py` builds the LaTeX table
(with the corrected std, see its docstring) from this module's saved output.

Needs the candidate-graph cache (`python main.py build-graphs`, or
GRAPH_CACHE env var pointing at an existing embedding_graphs.npz).

Output: results/<RESULTS_DIR>/table4/results.{pkl,json}
"""
import os
import sys
import json
import logging
import pickle
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

from sggnn.models import GCN, HeteroGNN, AdaptiveAggGCN, FBGNNLayer
from sggnn.data import get_data_dict, seed_everything, load_cached_graphs, sim_masks
from sggnn.train import train_model
from sggnn.paths import results_dir, to_json_safe

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

EXPERIMENT_ID = 'table4'
GRAPH_CACHE = os.environ.get('GRAPH_CACHE', str(ROOT / 'node_embeddings' / 'embedding_graphs.npz'))
DEVICE = torch.device(os.environ.get('GRAPH_EVAL_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu'))

SEED = 0
seed_everything(SEED)

all_graphs = load_cached_graphs(GRAPH_CACHE)
logging.info(f"Loaded graph cache from {GRAPH_CACHE}")

wd = 5e-4
epochs = int(os.environ.get('GRAPH_EVAL_EPOCHS', 2000))
patience = int(os.environ.get('GRAPH_EVAL_PATIENCE', 300))
n_layers = 2
hid_dim = 32
dropout = 0.5
aggr = 'cat'
last_act = nn.Softmax(dim=1)
nonlin = nn.Tanh()
lr = 0.005
node_name = 'web'
n_sims = int(os.environ.get('GRAPH_EVAL_NSIMS', 20))
SPLIT = os.environ.get('GRAPH_EVAL_SPLIT', 'random')

DATASETS = os.environ.get('GRAPH_EVAL_DATASETS',
                           'Texas,Wisconsin,Cornell,Actor,Chameleon,Squirrel,Cora,CiteSeer,USA,Europe,Brazil').split(',')

graphs = list(all_graphs[DATASETS[0]].keys())
del graphs[graphs.index('EPS-GraphWave')]

N_MASKS = int(os.environ.get('GRAPH_EVAL_NMASKS', 10))  # matches the original exp_1.py's hardcoded 10
data_dict = get_data_dict(DATASETS, all_graphs, node_name, N_MASKS, 0.8, 0.1, split=SPLIT)

EXPS = [{'leg': gname, 'graph': 'standard'} for gname in graphs]
EXPS += [
    {'leg': 'Adaptive-GNN-Node', 'graph': 'adaptive', 'per_node': True},
    {'leg': 'Adaptive-GNN', 'graph': 'adaptive', 'per_node': False},
    {'leg': 'SG-GNN', 'graph': 'heterogeneous'},
]

gcn_list = ['GCNConv', 'GATConv', 'SAGEConv', 'FBGNNLayer']
LAYERS = {'GCNConv': GCNConv, 'GATConv': GATConv, 'SAGEConv': SAGEConv, 'FBGNNLayer': FBGNNLayer}

OUT_DIR = results_dir(EXPERIMENT_ID)

best_accs_test = np.full((len(DATASETS), n_sims, len(gcn_list), len(EXPS)), np.nan)
learned_coefs = {}

for d, dataset_name in enumerate(DATASETS):
    logging.info(f"Starting dataset {dataset_name}")
    data_pyg = data_dict[dataset_name].to(DEVICE)
    N = data_pyg[node_name].N
    num_classes = data_pyg[node_name].num_classes
    in_dim = data_pyg[node_name].x.size(1)
    learned_coefs[dataset_name] = {}

    for e, exp in enumerate(EXPS):
        logging.info(f"Starting exp {exp['leg']}")
        for s in range(n_sims):
            train_idx_sim, val_idx_sim, test_idx_sim = sim_masks(data_pyg[node_name], s)

            learned_coefs[dataset_name][exp['leg']] = {}

            for g, gcnname in enumerate(gcn_list):
                gcn_class = LAYERS[gcnname]
                last_act_fn = last_act if gcnname == 'FBGNNLayer' else nn.Identity()

                if exp['graph'] == 'adaptive':
                    per_node_val = N if exp['per_node'] else -1
                    gcn_model = AdaptiveAggGCN(in_dim, hid_dim, num_classes, n_layers, dropout, nonlin,
                                                last_act_fn, graphs, gcn_class, gcn_class, per_node_val).to(DEVICE)
                    X = data_pyg.x_dict
                    edge_idx = data_pyg.edge_index_dict
                elif exp['graph'] == 'heterogeneous':
                    gcn_model = HeteroGNN(in_dim, hid_dim, num_classes, n_layers, dropout, nonlin,
                                           last_act_fn, graphs, aggr, gcn_class, gcn_class).to(DEVICE)
                    X = data_pyg.x_dict
                    edge_idx = data_pyg.edge_index_dict
                else:
                    gcn_model = GCN(in_dim, hid_dim, num_classes, n_layers, nonlin=nonlin,
                                     last_act=last_act_fn, dropout=dropout, gcnlayer=gcn_class).to(DEVICE)
                    X = data_pyg[node_name].x
                    edge_idx = data_pyg[node_name, exp['leg'], node_name].edge_index

                try:
                    model, _, _, _, val_accs, test_accs = train_model(
                        gcn_model, X, edge_idx, data_pyg[node_name].y,
                        train_idx_sim, val_idx_sim, test_idx_sim, {},
                        None, lr, wd, epochs, patience, verb=False)
                    best_epoch = int(np.argmax(val_accs))
                    best_accs_test[d, s, g, e] = test_accs[best_epoch]
                    if exp['graph'] == 'adaptive':
                        learned_coefs[dataset_name][exp['leg']][gcnname] = gcn_model.alphas.data.clone().cpu()
                except torch.cuda.OutOfMemoryError:
                    logging.warning(f"CUDA OOM: {dataset_name}, {exp['leg']}, {gcnname}, sim {s}")
                    torch.cuda.empty_cache()
                except Exception:
                    logging.exception(f"Error: {dataset_name}, {exp['leg']}, {gcnname}, sim {s}")

        np.save(OUT_DIR / 'best_accs_test.npy', best_accs_test)

payload = {
    'datasets': DATASETS, 'gcnlist': gcn_list, 'exps': EXPS,
    'learned_coefs': learned_coefs, 'best_accs_test': best_accs_test,
    'metadata': {
        'experiment_id': EXPERIMENT_ID, 'seed': SEED, 'split': SPLIT,
        'timestamp': datetime.datetime.now().isoformat(),
        'hid_dim': hid_dim, 'dropout': dropout, 'lr': lr, 'wd': wd,
        'epochs': epochs, 'patience': patience, 'n_sims': n_sims, 'n_layers': n_layers,
        'graph_cache': GRAPH_CACHE,
    },
}

with open(OUT_DIR / 'results.pkl', 'wb') as f:
    pickle.dump(payload, f)


with open(OUT_DIR / 'results.json', 'w') as f:
    json.dump(to_json_safe(payload), f, indent=2)

logging.info(f"Done. Results saved to {OUT_DIR}")
