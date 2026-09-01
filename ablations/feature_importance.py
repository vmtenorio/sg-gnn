"""
Ablation: Leave-one-group-out feature importance.

For each condition (set of graph types), run AdaptiveAggGCN (FBGNNLayer)
and report mean +/- std test accuracy.

Output:
  results/ablations/importance/importance_results.npy  -- float32 [n_datasets, n_sims, n_conditions]
  results/ablations/importance/importance_results.pkl  -- full results dict
"""

import sys
import logging
import pickle
import time
import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from arch import AdaptiveAggGCN, FBGNNLayer
from utils import get_data_dict, seed_everything
from train import train_model

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# ── reproducibility ──────────────────────────────────────────────────────────
seed_everything(42)

# ── device ───────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# ── output dir ───────────────────────────────────────────────────────────────
OUT_DIR = ROOT / 'results' / 'ablations' / 'importance'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── standard hyperparams ─────────────────────────────────────────────────────
hid_dim   = 32
dropout   = 0.5
nonlin    = nn.Tanh()
last_act  = nn.Softmax(dim=1)
lr        = 5e-3
wd        = 5e-4
epochs    = 2000
patience  = 300
n_sims    = 10
n_layers  = 1
node_name = 'web'

datasets = [
    'Texas', 'Wisconsin', 'Cornell', 'Actor', 'Chameleon',
    'Squirrel', 'Cora', 'CiteSeer', 'USA', 'Europe', 'Brazil'
]

# ── load graphs ───────────────────────────────────────────────────────────────
logging.info("Loading precomputed graphs from cache...")
data_np = np.load(str(ROOT / 'node_embeddings' / 'embedding_graphs.npz'), allow_pickle=True)
all_graphs = {}
for filename in data_np.files:
    dataset = filename.split('_')[0]
    key     = filename.split('_')[1]
    if dataset not in all_graphs:
        all_graphs[dataset] = {}
    all_graphs[dataset][key] = data_np[filename]

# Print actual keys so we can verify grouping at runtime
available_keys = list(all_graphs[datasets[0]].keys())
logging.info(f"Available graph keys: {available_keys}")

# ── group definitions (mapped to actual keys) ─────────────────────────────────
# Role-based:   graphs built from structural-role features
# Global:       graphs built from global position features
# Feats:        graphs built from raw node features (Feat)
# Original:     original graph topology

ROLE_KEYS   = [k for k in available_keys if 'Role' in k]     # KNN-RoleFeat, EPS-RoleFeat
GLOBAL_KEYS = [k for k in available_keys if 'Global' in k]   # KNN-GlobalFeat, EPS-GlobalFeat
FEAT_KEYS   = [k for k in available_keys if k in ('KNN-Feat', 'EPS-Feat')]

logging.info(f"Role keys:   {ROLE_KEYS}")
logging.info(f"Global keys: {GLOBAL_KEYS}")
logging.info(f"Feat keys:   {FEAT_KEYS}")

CONDITIONS = {
    'all':             ['Original'] + ROLE_KEYS + GLOBAL_KEYS + FEAT_KEYS,
    'no_role':         ['Original'] + GLOBAL_KEYS + FEAT_KEYS,
    'no_global':       ['Original'] + ROLE_KEYS + FEAT_KEYS,
    'no_feat':         ['Original'] + ROLE_KEYS + GLOBAL_KEYS,
    'role_only':       ['Original'] + ROLE_KEYS,
    'global_only':     ['Original'] + GLOBAL_KEYS,
    'structural_only': ROLE_KEYS + GLOBAL_KEYS,
}

condition_names = list(CONDITIONS.keys())
logging.info(f"Conditions: {condition_names}")
for cname, ckeys in CONDITIONS.items():
    logging.info(f"  {cname}: {ckeys}")

# ── build per-condition filtered all_graphs dicts ────────────────────────────
def filter_all_graphs(all_graphs_full, keep_keys):
    """Return a copy of all_graphs with only keys in keep_keys present."""
    filtered = {}
    for ds, gdict in all_graphs_full.items():
        filtered[ds] = {k: v for k, v in gdict.items() if k in keep_keys}
    return filtered

# ── main loop ─────────────────────────────────────────────────────────────────
n_datasets   = len(datasets)
n_conditions = len(condition_names)
best_accs_test = np.zeros((n_datasets, n_sims, n_conditions), dtype=np.float32)

for c, cname in enumerate(condition_names):
    ckeys = CONDITIONS[cname]
    logging.info(f"\n=== Condition {c+1}/{n_conditions}: {cname} (graphs: {ckeys}) ===")

    # Build data_dict for this condition's graph subset
    cond_all_graphs = filter_all_graphs(all_graphs, ckeys)

    try:
        data_dict = get_data_dict(datasets, cond_all_graphs, node_name, n_sims, 0.8, 0.1)
    except Exception as exc:
        logging.warning(f"  get_data_dict failed for condition {cname}: {exc}")
        continue

    for d, dataset_name in enumerate(datasets):
        logging.info(f"  Dataset {d+1}/{n_datasets}: {dataset_name}")

        try:
            data_pyg = data_dict[dataset_name].to(device)
            N          = data_pyg[node_name].N
            num_classes = data_pyg[node_name].num_classes
            in_dim     = data_pyg[node_name].x.size(1)

            graphs_for_model = ckeys  # keys present for this condition

            for s in range(n_sims):
                if data_pyg[node_name].train_mask.ndim > 1:
                    idx = s % data_pyg[node_name].train_mask.shape[1]
                    train_mask = data_pyg[node_name].train_mask[:, idx]
                    val_mask   = data_pyg[node_name].val_mask[:, idx]
                    test_mask  = data_pyg[node_name].test_mask[:, idx]
                else:
                    train_mask = data_pyg[node_name].train_mask
                    val_mask   = data_pyg[node_name].val_mask
                    test_mask  = data_pyg[node_name].test_mask

                model = AdaptiveAggGCN(
                    in_dim, hid_dim, num_classes, n_layers,
                    dropout, nonlin, last_act,
                    graphs_for_model, FBGNNLayer, FBGNNLayer, -1
                ).to(device)

                X        = data_pyg.x_dict
                edge_idx = data_pyg.edge_index_dict

                model, _, _, _, val_accs, test_accs = train_model(
                    model, X, edge_idx, data_pyg[node_name].y,
                    train_mask, val_mask, test_mask, {},
                    None, lr, wd, epochs, patience, verb=False
                )

                best_epoch = int(np.argmax(val_accs))
                best_accs_test[d, s, c] = test_accs[best_epoch]

            mean_acc = best_accs_test[d, :, c].mean()
            std_acc  = best_accs_test[d, :, c].std()
            logging.info(f"    {dataset_name}: {mean_acc:.4f} +/- {std_acc:.4f}")

        except torch.cuda.OutOfMemoryError:
            logging.warning(f"    CUDA OOM on {dataset_name}, condition {cname} -- skipping")
            torch.cuda.empty_cache()
        except Exception as exc:
            logging.warning(f"    Error on {dataset_name}, condition {cname}: {exc}")

    # Save checkpoint after each condition
    np.save(str(OUT_DIR / 'importance_results.npy'), best_accs_test)
    logging.info(f"  Saved checkpoint after condition {cname}")

# ── final save ────────────────────────────────────────────────────────────────
np.save(str(OUT_DIR / 'importance_results.npy'), best_accs_test)

results_dict = {
    'results': best_accs_test,
    'condition_names': condition_names,
    'conditions': CONDITIONS,
    'metadata': {
        'seed':       42,
        'datasets':   datasets,
        'timestamp':  datetime.datetime.now().isoformat(),
        'hid_dim':    hid_dim,
        'dropout':    dropout,
        'lr':         lr,
        'wd':         wd,
        'epochs':     epochs,
        'patience':   patience,
        'n_sims':     n_sims,
        'n_layers':   n_layers,
    }
}

with open(str(OUT_DIR / 'importance_results.pkl'), 'wb') as f:
    pickle.dump(results_dict, f)

logging.info(f"\nDone. Results saved to {OUT_DIR}")
logging.info("Condition means (per dataset):")
for c, cname in enumerate(condition_names):
    means = best_accs_test[:, :, c].mean(axis=1)
    logging.info(f"  {cname}: {np.round(means, 4)}")
