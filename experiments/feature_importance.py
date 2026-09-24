"""
Leave-one-group-out feature-importance ablation (Table VII, `tab:feat_ablation`).

Model: single-layer SG-FBGNN (AdaptiveAggGCN with FBGNN layers), hidden
dimension 32, on eleven datasets x 10 splits. Conditions differ only in which
graph views are passed to the model:
  all              Original + role (EPS/KNN) + global (EPS/KNN) + feat (EPS/KNN)
  no_role / no_global / no_feat   `all` minus one family
  role_only / global_only         Original + one family
  structural_only                 role + global, without Original or feat
  full             every cached view except EPS-GraphWave (R=14), the graph
                   set of the main table; not reported in Table VII

The splits are drawn once, before the condition loop, and written over each
condition's masks (with an assertion that the replacement took). create_masks
draws from the global torch RNG, so without this every condition would be
evaluated on different random splits; with it, every condition is a paired
observation on identical data and differences can be tested per split
(analysis/plot_feature_importance.py).

The run checkpoints after every condition and resumes from
completed_conditions.json if restarted. A resumed run reuses the same splits
but not the same RNG stream for model initialisation and dropout, so it is
not bitwise identical to an uninterrupted one.

Needs the candidate-graph cache (`python main.py build-graphs`).

Output: results/<RESULTS_DIR>/table7/{checkpoint.npy,results.pkl,results.json}
"""

import os
import sys
import logging
import pickle
import json
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from sggnn.models import AdaptiveAggGCN, FBGNNLayer
from sggnn.data import get_data_dict, seed_everything, load_cached_graphs, sim_masks
from sggnn.train import train_model
from sggnn.paths import results_dir, git_commit, to_json_safe

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# ── reproducibility ──────────────────────────────────────────────────────────
seed = 42
seed_everything(seed)

# ── device ───────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

commit_hash = git_commit(short=True) or 'unknown'

# ── output dir ───────────────────────────────────────────────────────────────
OUT_DIR = results_dir('table7')

# ── hyperparams ──────────────────────────────────────────────────────────────
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
GRAPH_CACHE = os.environ.get('GRAPH_CACHE', str(ROOT / 'node_embeddings' / 'embedding_graphs.npz'))
all_graphs = load_cached_graphs(GRAPH_CACHE)

available_keys = list(all_graphs[datasets[0]].keys())
logging.info(f"Available graph keys: {available_keys}")

# ── view groups ──────────────────────────────────────────────────────────────
ROLE_KEYS   = [k for k in available_keys if 'Role' in k]     # KNN-RoleFeat, EPS-RoleFeat
GLOBAL_KEYS = [k for k in available_keys if 'Global' in k]   # KNN-GlobalFeat, EPS-GlobalFeat
FEAT_KEYS   = [k for k in available_keys if k in ('KNN-Feat', 'EPS-Feat')]

logging.info(f"Role keys:   {ROLE_KEYS}")
logging.info(f"Global keys: {GLOBAL_KEYS}")
logging.info(f"Feat keys:   {FEAT_KEYS}")

# Same graph list as exp_main_matched.py.
FULL_KEYS = [k for k in available_keys if k != 'EPS-GraphWave']
logging.info(f"Full keys ({len(FULL_KEYS)}):  {FULL_KEYS}")

CONDITIONS = {
    'all':             ['Original'] + ROLE_KEYS + GLOBAL_KEYS + FEAT_KEYS,
    'no_role':         ['Original'] + GLOBAL_KEYS + FEAT_KEYS,
    'no_global':       ['Original'] + ROLE_KEYS + FEAT_KEYS,
    'no_feat':         ['Original'] + ROLE_KEYS + GLOBAL_KEYS,
    'role_only':       ['Original'] + ROLE_KEYS,
    'global_only':     ['Original'] + GLOBAL_KEYS,
    'structural_only': ROLE_KEYS + GLOBAL_KEYS,
    'full':            FULL_KEYS,
}

condition_names = list(CONDITIONS.keys())
logging.info(f"Conditions: {condition_names}")
for cname, ckeys in CONDITIONS.items():
    logging.info(f"  {cname}: {ckeys}")

def filter_all_graphs(all_graphs_full, keep_keys):
    """Return a copy of all_graphs with only keys in keep_keys present."""
    filtered = {}
    for ds, gdict in all_graphs_full.items():
        filtered[ds] = {k: v for k, v in gdict.items() if k in keep_keys}
    return filtered

# ── shared splits across conditions (see module docstring) ───────────────────
# Two successive get_data_dict calls agree on only ~68% of mask entries.
logging.info("Drawing the shared split set (used by every condition)...")
seed_everything(seed)
_ref_dict = get_data_dict(datasets, all_graphs, node_name, n_sims, 0.8, 0.1)
SHARED_MASKS = {
    ds: (_ref_dict[ds][node_name].train_mask.clone(),
         _ref_dict[ds][node_name].val_mask.clone(),
         _ref_dict[ds][node_name].test_mask.clone())
    for ds in datasets
}
del _ref_dict
for ds in datasets:
    tr, va, te = SHARED_MASKS[ds]
    logging.info(f"  {ds}: masks {tuple(tr.shape)}, "
                 f"train {int(tr[:, 0].sum())} / val {int(va[:, 0].sum())} / "
                 f"test {int(te[:, 0].sum())} on split 0")


def apply_shared_masks(data_dict):
    """Overwrite each dataset's masks with the shared split set."""
    for ds in datasets:
        tr, va, te = SHARED_MASKS[ds]
        data_dict[ds][node_name].train_mask = tr.clone()
        data_dict[ds][node_name].val_mask   = va.clone()
        data_dict[ds][node_name].test_mask  = te.clone()
    return data_dict


# ── resume support ────────────────────────────────────────────────────────────
# SHARED_MASKS is recomputed identically on every start (nothing consumes RNG
# state before it), so completed conditions can be skipped and reloaded.
COMPLETED_PATH = OUT_DIR / 'completed_conditions.json'
CHECKPOINT_PATH = OUT_DIR / 'checkpoint.npy'

n_datasets   = len(datasets)
n_conditions = len(condition_names)

if COMPLETED_PATH.exists() and CHECKPOINT_PATH.exists():
    with open(COMPLETED_PATH) as f:
        completed_conditions = json.load(f)
    loaded = np.load(str(CHECKPOINT_PATH))
    if loaded.shape == (n_datasets, n_sims, n_conditions):
        best_accs_test = loaded
        logging.info(f"Resuming: found completed conditions {completed_conditions}")
    else:
        logging.warning(f"Checkpoint shape {loaded.shape} != expected "
                         f"{(n_datasets, n_sims, n_conditions)} -- starting fresh")
        best_accs_test = np.full((n_datasets, n_sims, n_conditions), np.nan, dtype=np.float32)
        completed_conditions = []
else:
    best_accs_test = np.full((n_datasets, n_sims, n_conditions), np.nan, dtype=np.float32)
    completed_conditions = []

for c, cname in enumerate(condition_names):
    if cname in completed_conditions:
        logging.info(f"\n=== Condition {c+1}/{n_conditions}: {cname} -- already completed, skipping ===")
        continue

    ckeys = CONDITIONS[cname]
    logging.info(f"\n=== Condition {c+1}/{n_conditions}: {cname} (graphs: {ckeys}) ===")

    cond_all_graphs = filter_all_graphs(all_graphs, ckeys)

    try:
        data_dict = get_data_dict(datasets, cond_all_graphs, node_name, n_sims, 0.8, 0.1)
    except Exception:
        logging.exception(f"  get_data_dict failed for condition {cname} -- not marked completed")
        continue

    # Replace the freshly drawn (and therefore condition-specific) masks with
    # the shared ones, then assert the replacement actually took.
    data_dict = apply_shared_masks(data_dict)
    for ds in datasets:
        assert torch.equal(data_dict[ds][node_name].train_mask, SHARED_MASKS[ds][0]), \
            f"shared-split assertion failed for {ds} in condition {cname}"
    logging.info(f"  shared splits applied and verified for all {len(datasets)} datasets")

    condition_had_failure = False
    for d, dataset_name in enumerate(datasets):
        logging.info(f"  Dataset {d+1}/{n_datasets}: {dataset_name}")

        try:
            data_pyg = data_dict[dataset_name].to(device)
            num_classes = data_pyg[node_name].num_classes
            in_dim     = data_pyg[node_name].x.size(1)

            for s in range(n_sims):
                train_mask, val_mask, test_mask = sim_masks(data_pyg[node_name], s)

                model = AdaptiveAggGCN(
                    in_dim, hid_dim, num_classes, n_layers,
                    dropout, nonlin, last_act,
                    ckeys, FBGNNLayer, FBGNNLayer, -1
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
            logging.warning(f"    CUDA OOM on {dataset_name}, condition {cname} -- "
                             f"skipping, recorded as NaN")
            best_accs_test[d, :, c] = np.nan
            condition_had_failure = True
            torch.cuda.empty_cache()
        except Exception:
            logging.exception(f"    Error on {dataset_name}, condition {cname} -- "
                               f"recorded as NaN, not marking condition completed")
            best_accs_test[d, :, c] = np.nan
            condition_had_failure = True

    if not condition_had_failure:
        completed_conditions.append(cname)
    else:
        logging.warning(f"  Condition {cname} had at least one failed dataset -- "
                         f"NOT marked completed, will be retried on resume")
    np.save(str(CHECKPOINT_PATH), best_accs_test)
    with open(COMPLETED_PATH, 'w') as f:
        json.dump(completed_conditions, f)
    logging.info(f"  Saved checkpoint after condition {cname} "
                 f"({len(completed_conditions)}/{n_conditions} conditions done)")

# ── final save ────────────────────────────────────────────────────────────────
if len(completed_conditions) < n_conditions:
    missing = [c for c in condition_names if c not in completed_conditions]
    logging.warning(f"\nNot all conditions completed this run (missing: {missing}). "
                     f"Re-run this script to resume -- completed conditions are "
                     f"cached in {COMPLETED_PATH} and {CHECKPOINT_PATH}. "
                     f"Skipping final .pkl/.json write.")
    sys.exit(0)

np.save(str(CHECKPOINT_PATH), best_accs_test)

results_dict = {
    'results': best_accs_test,
    'condition_names': condition_names,
    'conditions': CONDITIONS,
    'metadata': {
        'experiment_id': 'table7',
        'seed':       42,
        'datasets':   datasets,
        'timestamp':  datetime.datetime.now().isoformat(),
        'commit_hash': commit_hash,
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

with open(str(OUT_DIR / 'results.pkl'), 'wb') as f:
    pickle.dump(results_dict, f)


with open(str(OUT_DIR / 'results.json'), 'w') as f:
    json.dump(to_json_safe(results_dict), f, indent=2)

logging.info(f"\nDone. Results saved to {OUT_DIR}")
logging.info("Condition means (per dataset):")
for c, cname in enumerate(condition_names):
    means = best_accs_test[:, :, c].mean(axis=1)
    logging.info(f"  {cname}: {np.round(means, 4)}")
