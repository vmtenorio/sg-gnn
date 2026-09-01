"""
Adds GRAND, CD-GNN, LG-GNN (R2-C4) as additional rows in the SAME matched-
parameter main comparison table produced by exp_main_matched.py, rather than
as a standalone experiment (results/baselines/). Reuses the exact per-dataset
target parameter budget already computed and stored by exp_main_matched.py
(the largest SG-GNN variant's parameter count for that dataset), so these
three baselines are held to the identical fairness standard as the original
8 baselines.

Output:
  results/main_matched/<date>-newbaselines<suffix>.{npy,pkl,json}
"""

import sys
import os
import json
import logging
import pickle
import datetime
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'alternatives'))

import numpy as np
import torch
import torch.nn as nn

from utils import get_data_dict, seed_everything
from train import train_model
from grand_wrapper import GRANDWrapper
from cdgnn_wrapper import CDGNNWrapper
from lggnn_wrapper import LGGNNWrapper

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

DATASET_SUBSET = os.environ.get('MATCHED_DATASETS', '').split(',') if os.environ.get('MATCHED_DATASETS') else None
DEVICE_STR = os.environ.get('MATCHED_DEVICE', 'cuda:0')
# Path(s) to already-completed exp_main_matched.py pkl(s) to pull target budgets from
SOURCE_PKLS = os.environ['MATCHED_SOURCE_PKLS'].split(',')

assert torch.cuda.is_available(), "CUDA required"
device = torch.device(DEVICE_STR)
seed_everything(42)

try:
    commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT).decode().strip()
except Exception:
    commit_hash = 'unknown'

n_layers  = 2
dropout   = 0.5
nonlin    = nn.Tanh()
last_act_softmax = nn.Softmax(dim=1)
lr        = 5e-3
wd        = 5e-4
epochs    = int(os.environ.get('MATCHED_EPOCHS', 2000))
patience  = int(os.environ.get('MATCHED_PATIENCE', 300))
n_sims    = int(os.environ.get('MATCHED_NSIMS', 20))
node_name = 'web'

ALL_DATASETS = ['Texas', 'Wisconsin', 'Cornell', 'Actor', 'Chameleon',
                'Squirrel', 'Cora', 'CiteSeer', 'USA', 'Europe', 'Brazil']
datasets = DATASET_SUBSET if DATASET_SUBSET else ALL_DATASETS
logging.info(f"Running on datasets: {datasets} (device={DEVICE_STR})")

# ── load target budgets from already-completed matched-table runs ───────────
target_budget = {}
for pkl_path in SOURCE_PKLS:
    with open(pkl_path, 'rb') as f:
        src = pickle.load(f)
    for ds in src['metadata']['datasets']:
        # every baseline's matched_hparams entry stores the same 'target' value
        any_baseline = src['baseline_names'][0]
        target_budget[ds] = src['matched_hparams'][ds][any_baseline]['target']

missing = [d for d in datasets if d not in target_budget]
assert not missing, f"No target budget found for datasets: {missing} -- check MATCHED_SOURCE_PKLS"
logging.info(f"Target budgets loaded: { {d: target_budget[d] for d in datasets} }")

# ── load cached graphs (only need Original edge_index; single-graph models) ──
data_np = np.load(str(ROOT / 'node_embeddings' / 'embedding_graphs.npz'), allow_pickle=True)
all_graphs = {}
for filename in data_np.files:
    ds  = filename.split('_')[0]
    key = filename.split('_')[1]
    if ds not in all_graphs:
        all_graphs[ds] = {}
    all_graphs[ds][key] = data_np[filename]

data_dict = get_data_dict(datasets, all_graphs, node_name, n_sims, 0.8, 0.1)

OUT_DIR = ROOT / 'results' / 'main_matched'
OUT_DIR.mkdir(parents=True, exist_ok=True)
datestamp = datetime.datetime.now().strftime('%Y%m%d')
suffix = f"-{DATASET_SUBSET[0]}_to_{DATASET_SUBSET[-1]}" if DATASET_SUBSET else ""
fname = str(OUT_DIR / f'{datestamp}-newbaselines{suffix}')

CONFIG_NAMES = ['GRAND', 'CD-GNN', 'LG-GNN']
N_CONFIGS = len(CONFIG_NAMES)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def find_matched_hid(build_fn, target, lo=32, hi=1024, hard_cap=32768):
    if count_params(build_fn(lo)) >= target:
        return lo, count_params(build_fn(lo))
    while count_params(build_fn(hi)) < target and hi < hard_cap:
        hi *= 2
    hi = min(hi, hard_cap)
    a, b = lo, hi
    while a < b:
        mid = (a + b) // 2
        if count_params(build_fn(mid)) >= target:
            b = mid
        else:
            a = mid + 1
    return a, count_params(build_fn(a))


best_accs_test = np.zeros((len(datasets), n_sims, N_CONFIGS), dtype=np.float32)
param_counts = {}
matched_hparams = {}

for d, dataset_name in enumerate(datasets):
    logging.info(f"\n=== Dataset {d+1}/{len(datasets)}: {dataset_name} ===")
    data_pyg = data_dict[dataset_name].to(device)
    num_classes = data_pyg[node_name].num_classes
    in_dim = data_pyg[node_name].x.size(1)
    X_hom = data_pyg[node_name].x
    ei_orig = data_pyg[node_name, 'Original', node_name].edge_index

    target = target_budget[dataset_name]
    param_counts[dataset_name] = {}
    matched_hparams[dataset_name] = {}

    builders = {
        'GRAND': lambda hid: GRANDWrapper(in_dim, hid, num_classes, n_layers, dropout, nonlin, last_act_softmax),
        'CD-GNN': lambda hid: CDGNNWrapper(in_dim, hid, num_classes, n_layers, dropout, nonlin, last_act_softmax),
        'LG-GNN': lambda hid: LGGNNWrapper(in_dim, hid, num_classes, n_layers, dropout, nonlin, last_act_softmax),
    }

    matched_val = {}
    for bname, builder in builders.items():
        val, achieved = find_matched_hid(builder, target)
        matched_val[bname] = val
        param_counts[dataset_name][bname] = achieved
        matched_hparams[dataset_name][bname] = {'hid': val, 'achieved_params': achieved, 'target': target}
        shortfall = "" if achieved >= target else "  ** SHORT of target (arch limitation) **"
        logging.info(f"    {bname}: hid={val}, params={achieved}{shortfall}")

    for s in range(n_sims):
        if data_pyg[node_name].train_mask.ndim > 1:
            idx = s % data_pyg[node_name].train_mask.shape[1]
            tr = data_pyg[node_name].train_mask[:, idx]
            vl = data_pyg[node_name].val_mask[:, idx]
            te = data_pyg[node_name].test_mask[:, idx]
        else:
            tr = data_pyg[node_name].train_mask
            vl = data_pyg[node_name].val_mask
            te = data_pyg[node_name].test_mask

        for ci, bname in enumerate(CONFIG_NAMES):
            model = builders[bname](matched_val[bname]).to(device)
            try:
                _, _, _, _, val_accs, test_accs = train_model(
                    model, X_hom, ei_orig, data_pyg[node_name].y,
                    tr, vl, te, {}, None, lr, wd, epochs, patience, verb=False)
                best_accs_test[d, s, ci] = test_accs[int(np.argmax(val_accs))]
            except torch.cuda.OutOfMemoryError:
                logging.warning(f"    CUDA OOM: {bname}, sim {s} -- skipping")
                torch.cuda.empty_cache()
            except Exception as exc:
                logging.warning(f"    Error in {bname}, sim {s}: {exc}")

        if (s + 1) % 5 == 0:
            logging.info(f"  sim {s+1}/{n_sims} done")

    for ci, cname in enumerate(CONFIG_NAMES):
        m_ = best_accs_test[d, :, ci].mean()
        s_ = best_accs_test[d, :, ci].std()
        logging.info(f"  {cname}: {m_:.4f} +/- {s_:.4f}")

    np.save(fname + '.npy', best_accs_test)

np.save(fname + '.npy', best_accs_test)

results_dict = {
    'results': best_accs_test,
    'config_names': CONFIG_NAMES,
    'param_counts': param_counts,
    'matched_hparams': matched_hparams,
    'metadata': {
        'seed': 42, 'datasets': datasets, 'timestamp': datetime.datetime.now().isoformat(),
        'commit_hash': commit_hash, 'dropout': dropout, 'lr': lr, 'wd': wd,
        'epochs': epochs, 'patience': patience, 'n_sims': n_sims, 'n_layers': n_layers,
        'source_pkls': SOURCE_PKLS,
    },
}

with open(fname + '.pkl', 'wb') as f:
    pickle.dump(results_dict, f)


def json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


with open(fname + '.json', 'w') as f:
    json.dump(json_safe(results_dict), f, indent=2)

logging.info(f"\nDone. Results saved to {fname}{{.npy,.pkl,.json}}")
