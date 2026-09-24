"""
Learned adaptive coefficients alpha_r (Fig. 4, `fig:learned_coefs`): the
single-layer, globally adaptive SG-GNN with GCN and FBGNN base layers on
every cached graph except EPS-GraphWave, 20 runs per dataset. Keeps the raw
(pre-softmax) alphas of the last run per dataset/config.

Needs the candidate-graph cache (`python main.py build-graphs`).

Output: results/<RESULTS_DIR>/fig4/results.{pkl,json}
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
from torch_geometric.nn import GCNConv

from sggnn.models import AdaptiveAggGCN, FBGNNLayer
from sggnn.data import get_data_dict, seed_everything, load_cached_graphs, sim_masks
from sggnn.train import train_model
from sggnn.paths import results_dir, to_json_safe

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

EXPERIMENT_ID = 'fig4'
GRAPH_CACHE = os.environ.get('GRAPH_CACHE', str(ROOT / 'node_embeddings' / 'embedding_graphs.npz'))
DEVICE = torch.device(os.environ.get('COEFFICIENTS_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu'))
SEED = 1
seed_everything(SEED)

all_graphs = load_cached_graphs(GRAPH_CACHE)

wd, epochs, patience = 5e-4, int(os.environ.get('COEFFICIENTS_EPOCHS', 2000)), int(os.environ.get('COEFFICIENTS_PATIENCE', 300))
n_layers, hid_dim, dropout = 2, 32, 0.5
last_act = nn.Softmax(dim=1)
nonlin = nn.Tanh()
lr = 0.005
node_name = 'web'
n_sims = int(os.environ.get('COEFFICIENTS_NSIMS', 20))
SPLIT = os.environ.get('COEFFICIENTS_SPLIT', 'random')

DATASETS = os.environ.get('COEFFICIENTS_DATASETS',
                           'Texas,Wisconsin,Cornell,Actor,Chameleon,Squirrel,Cora,CiteSeer,USA,Europe,Brazil').split(',')

graphs = list(all_graphs[DATASETS[0]].keys())
del graphs[graphs.index('EPS-GraphWave')]

N_MASKS = int(os.environ.get('COEFFICIENTS_NMASKS', 10))  # matches the original exp_2.py's hardcoded 10
data_dict = get_data_dict(DATASETS, all_graphs, node_name, N_MASKS, 0.8, 0.1, split=SPLIT)

EXPS = [
    {'leg': 'Adaptive-GNN-GCN', 'layer': 'GCNConv', 'per_node': False},
    {'leg': 'Adaptive-GNN-FBGNN', 'layer': 'FBGNNLayer', 'per_node': False},
]
LAYERS = {'GCNConv': GCNConv, 'FBGNNLayer': FBGNNLayer}

OUT_DIR = results_dir(EXPERIMENT_ID)
best_accs_test = np.full((len(DATASETS), n_sims, len(EXPS)), np.nan)
learned_coefs, mlp_weights = {}, {}

for d, dataset_name in enumerate(DATASETS):
    logging.info(f"Starting dataset {dataset_name}")
    data_pyg = data_dict[dataset_name].to(DEVICE)
    N = data_pyg[node_name].N
    num_classes = data_pyg[node_name].num_classes
    in_dim = data_pyg[node_name].x.size(1)
    learned_coefs[dataset_name], mlp_weights[dataset_name] = {}, {}

    for e, exp in enumerate(EXPS):
        logging.info(f"Starting exp {exp['leg']}")
        for s in range(n_sims):
            tr, vl, te = sim_masks(data_pyg[node_name], s)

            gcn_class = LAYERS[exp['layer']]
            last_act_fn = last_act if exp['layer'] == 'FBGNNLayer' else nn.Identity()
            per_node_val = N if exp['per_node'] else -1
            gcn_model = AdaptiveAggGCN(in_dim, hid_dim, num_classes, n_layers, dropout, nonlin,
                                        last_act_fn, graphs, gcn_class, gcn_class, per_node_val).to(DEVICE)
            X, edge_idx = data_pyg.x_dict, data_pyg.edge_index_dict

            try:
                model, _, _, _, val_accs, test_accs = train_model(
                    gcn_model, X, edge_idx, data_pyg[node_name].y, tr, vl, te, {},
                    None, lr, wd, epochs, patience, verb=False)
                best_accs_test[d, s, e] = test_accs[int(np.argmax(val_accs))]
                mlp_weights[dataset_name][exp['leg']] = gcn_model.lin.weight.data.clone().cpu()
                learned_coefs[dataset_name][exp['leg']] = gcn_model.alphas.data.clone().cpu()
            except torch.cuda.OutOfMemoryError:
                logging.warning(f"CUDA OOM: {dataset_name}, {exp['leg']}, sim {s}")
                torch.cuda.empty_cache()
            except Exception:
                logging.exception(f"Error: {dataset_name}, {exp['leg']}, sim {s}")

        np.save(OUT_DIR / 'best_accs_test.npy', best_accs_test)

payload = {
    'datasets': DATASETS, 'exps': EXPS, 'learned_coefs': learned_coefs,
    'mlp_weights': mlp_weights, 'best_accs_test': best_accs_test,
    'metadata': {'experiment_id': EXPERIMENT_ID, 'seed': SEED, 'split': SPLIT,
                 'timestamp': datetime.datetime.now().isoformat(),
                 'hid_dim': hid_dim, 'dropout': dropout, 'lr': lr, 'wd': wd,
                 'epochs': epochs, 'patience': patience, 'n_sims': n_sims, 'n_layers': n_layers},
}
with open(OUT_DIR / 'results.pkl', 'wb') as f:
    pickle.dump(payload, f)


with open(OUT_DIR / 'results.json', 'w') as f:
    json.dump(to_json_safe(payload), f, indent=2)

logging.info(f"Done. Results saved to {OUT_DIR}")
