"""
Ablation: Training time per epoch and peak GPU memory.

Models benchmarked:
  - GCN (single graph, Original)
  - FBGNN (single graph, Original, via GCN class with FBGNNLayer)
  - AdaptiveAggGCN-GCN   (all structural graphs, GCNConv)
  - AdaptiveAggGCN-FBGNN (all structural graphs, FBGNNLayer)

Datasets: ['Actor', 'Chameleon', 'Squirrel']
Timing: 5 warmup epochs, then timed over 20 epochs, repeated n_sims=3 times.

Output:
  results/ablations/timing/timing_results.pkl
  results/ablations/timing/timing_results.json
"""

import sys
import logging
import pickle
import datetime
import json
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from arch import AdaptiveAggGCN, GCN, FBGNNLayer
from utils import get_data_dict, seed_everything
from train import train_model

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

seed_everything(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")
cuda_available = device.type == 'cuda'

OUT_DIR = ROOT / 'results' / 'ablations' / 'timing'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── hyperparams ───────────────────────────────────────────────────────────────
hid_dim   = 32
dropout   = 0.5
nonlin    = nn.Tanh()
last_act  = nn.Softmax(dim=1)
lr        = 5e-3
wd        = 5e-4
n_layers  = 1
node_name = 'web'

WARMUP_EPOCHS = 5
TIMED_EPOCHS  = 20
n_sims        = 3

datasets = ['Actor', 'Chameleon', 'Squirrel']

# ── load graphs ───────────────────────────────────────────────────────────────
logging.info("Loading precomputed graphs...")
data_np = np.load(str(ROOT / 'node_embeddings' / 'embedding_graphs.npz'), allow_pickle=True)
all_graphs = {}
for filename in data_np.files:
    ds  = filename.split('_')[0]
    key = filename.split('_')[1]
    if ds not in all_graphs:
        all_graphs[ds] = {}
    all_graphs[ds][key] = data_np[filename]

available_keys = list(all_graphs[datasets[0]].keys())
# Exclude EPS-GraphWave (OOM on Chameleon)
all_graph_keys = [k for k in available_keys if k != 'EPS-GraphWave']
logging.info(f"Graph keys used: {all_graph_keys}")

data_dict = get_data_dict(datasets, all_graphs, node_name, n_sims, 0.8, 0.1)

# ── helper: single-step timing ───────────────────────────────────────────────
def time_epoch(model, X, edge_idx, y, train_mask, optimizer, criterion, fw_kwargs):
    """Run one training step and return wall-clock seconds."""
    if cuda_available:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.train()
    optimizer.zero_grad()
    out = model(X, edge_index=edge_idx, **fw_kwargs)
    loss = criterion(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    if cuda_available:
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ── result structure ──────────────────────────────────────────────────────────
timing_results = {}

for dataset_name in datasets:
    logging.info(f"\n=== Dataset: {dataset_name} ===")
    timing_results[dataset_name] = {}

    try:
        data_pyg    = data_dict[dataset_name].to(device)
        N           = data_pyg[node_name].N
        num_classes = data_pyg[node_name].num_classes
        in_dim      = data_pyg[node_name].x.size(1)
        y           = data_pyg[node_name].y

        # Use first mask column for timing
        tm = data_pyg[node_name].train_mask
        if tm.ndim > 1:
            train_mask = tm[:, 0]
        else:
            train_mask = tm

        criterion = nn.CrossEntropyLoss()

        # Build model specs
        model_specs = [
            {
                'name': 'GCN',
                'build': lambda: GCN(
                    in_dim, hid_dim, num_classes, n_layers,
                    nonlin=nonlin, last_act=nn.Identity(),
                    dropout=dropout, gcnlayer=GCNConv
                ).to(device),
                'X':        data_pyg[node_name].x,
                'ei':       data_pyg[node_name, 'Original', node_name].edge_index,
                'fw_kwargs': {},
            },
            {
                'name': 'FBGNN',
                'build': lambda: GCN(
                    in_dim, hid_dim, num_classes, n_layers,
                    nonlin=nonlin, last_act=last_act,
                    dropout=dropout, gcnlayer=FBGNNLayer
                ).to(device),
                'X':        data_pyg[node_name].x,
                'ei':       data_pyg[node_name, 'Original', node_name].edge_index,
                'fw_kwargs': {},
            },
            {
                'name': 'AdaptiveAggGCN-GCN',
                'build': lambda: AdaptiveAggGCN(
                    in_dim, hid_dim, num_classes, n_layers,
                    dropout, nonlin, nn.Identity(),
                    all_graph_keys, GCNConv, GCNConv, -1
                ).to(device),
                'X':        data_pyg.x_dict,
                'ei':       data_pyg.edge_index_dict,
                'fw_kwargs': {},
            },
            {
                'name': 'AdaptiveAggGCN-FBGNN',
                'build': lambda: AdaptiveAggGCN(
                    in_dim, hid_dim, num_classes, n_layers,
                    dropout, nonlin, last_act,
                    all_graph_keys, FBGNNLayer, FBGNNLayer, -1
                ).to(device),
                'X':        data_pyg.x_dict,
                'ei':       data_pyg.edge_index_dict,
                'fw_kwargs': {},
            },
        ]

        for spec in model_specs:
            mname = spec['name']
            logging.info(f"  Benchmarking {mname}...")
            epoch_times_all_sims = []
            peak_mem_all = []

            try:
                for sim in range(n_sims):
                    model     = spec['build']()
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                    X         = spec['X']
                    ei        = spec['ei']
                    fw_kw     = spec['fw_kwargs']

                    # Reset memory stats
                    if cuda_available:
                        torch.cuda.reset_peak_memory_stats(device)

                    # Warmup
                    for _ in range(WARMUP_EPOCHS):
                        time_epoch(model, X, ei, y, train_mask, optimizer, criterion, fw_kw)

                    # Reset again after warmup
                    if cuda_available:
                        torch.cuda.reset_peak_memory_stats(device)

                    # Timed epochs
                    epoch_times = []
                    for _ in range(TIMED_EPOCHS):
                        t = time_epoch(model, X, ei, y, train_mask, optimizer, criterion, fw_kw)
                        epoch_times.append(t)

                    if cuda_available:
                        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
                    else:
                        peak_mem = float('nan')

                    epoch_times_all_sims.extend(epoch_times)
                    peak_mem_all.append(peak_mem)

                n_params = count_params(model)
                mean_t   = float(np.mean(epoch_times_all_sims))
                std_t    = float(np.std(epoch_times_all_sims))
                mean_mem = float(np.mean(peak_mem_all))

                timing_results[dataset_name][mname] = {
                    'time_per_epoch_mean': mean_t,
                    'time_per_epoch_std':  std_t,
                    'peak_memory_mb':      mean_mem,
                    'n_params':            n_params,
                }
                logging.info(f"    {mname}: {mean_t*1000:.2f} ms/epoch (+/- {std_t*1000:.2f}), "
                             f"peak mem={mean_mem:.1f} MB, params={n_params}")

            except torch.cuda.OutOfMemoryError:
                logging.warning(f"    CUDA OOM on {mname}, {dataset_name} -- skipping")
                torch.cuda.empty_cache()
                timing_results[dataset_name][mname] = {
                    'time_per_epoch_mean': float('nan'),
                    'time_per_epoch_std':  float('nan'),
                    'peak_memory_mb':      float('nan'),
                    'n_params':            -1,
                    'error': 'CUDA OOM',
                }
            except Exception as exc:
                logging.warning(f"    Error on {mname}, {dataset_name}: {exc}")
                timing_results[dataset_name][mname] = {
                    'time_per_epoch_mean': float('nan'),
                    'time_per_epoch_std':  float('nan'),
                    'peak_memory_mb':      float('nan'),
                    'n_params':            -1,
                    'error': str(exc),
                }

    except torch.cuda.OutOfMemoryError:
        logging.warning(f"  CUDA OOM setting up {dataset_name} -- skipping")
        torch.cuda.empty_cache()
    except Exception as exc:
        logging.warning(f"  Error setting up {dataset_name}: {exc}")

# ── save ─────────────────────────────────────────────────────────────────────
full_results = {
    'results': timing_results,
    'metadata': {
        'seed':          42,
        'datasets':      datasets,
        'timestamp':     datetime.datetime.now().isoformat(),
        'hid_dim':       hid_dim,
        'dropout':       dropout,
        'lr':            lr,
        'wd':            wd,
        'warmup_epochs': WARMUP_EPOCHS,
        'timed_epochs':  TIMED_EPOCHS,
        'n_sims':        n_sims,
        'n_layers':      n_layers,
        'all_graph_keys': all_graph_keys,
    }
}

with open(str(OUT_DIR / 'timing_results.pkl'), 'wb') as f:
    pickle.dump(full_results, f)

# JSON: replace NaN with None for valid JSON
def to_json_safe(obj):
    if isinstance(obj, float) and (obj != obj):  # isnan
        return None
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    return obj

with open(str(OUT_DIR / 'timing_results.json'), 'w') as f:
    json.dump(to_json_safe(full_results), f, indent=2)

logging.info(f"\nDone. Results saved to {OUT_DIR}")
