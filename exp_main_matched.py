"""
Main SG-GNN vs. baselines comparison table (Table VI in the paper), run under
a parameter-fairness protocol: since SG-GNN processes R graphs through R
parallel branches, it may use more parameters than a single-graph baseline of
the same hidden dimension, so accuracy gains could in principle come from
extra capacity rather than a better graph structure.

Strategy: do NOT shrink SG-GNN's hidden dimension. Instead, INCREASE each
baseline's capacity (hidden dim, or num_layers where hidden dim has no
effect) so its parameter count is >= the largest SG-GNN variant's parameter
count, for that dataset. This replaces the standalone parameter-budget
ablation entirely -- the main table itself is now already
parameter-fair.

SG-GNN variants (unchanged, hid_dim=32, as in the original paper):
  SG-GCN_N / SG-FBGNN_N : AdaptiveAggGCN(per_node=N)       -- single layer, node-specific alphas
  SG-GCN   / SG-FBGNN    : AdaptiveAggGCN(per_node=-1)     -- single layer, global alphas
  SG-GCN_L / SG-FBGNN_L  : HeteroGNN(num_layers=2)         -- multi-layer, implicit importance (Ss:full_sg-gnn)

Baselines (capacity increased to match):
  GCN, GAT, gfNN, FAGCN, DirGNN, MixHop (FBGNN base), SSGC, H2GCN

Special case: FAGCN's hidden_channels argument has NO effect on its parameter
count (its internal FAConv layers operate directly on the input dimension).
Its capacity is instead increased via num_layers. For low-dimensional
datasets this may still fall short of the target -- this is reported
transparently rather than hidden.

Output:
  results/main_matched/<date>-main_matched.npy
  results/main_matched/<date>-main_matched.pkl
  results/main_matched/<date>-main_matched.json
  results/main_matched/<date>-table.tex   (regenerated LaTeX table)
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

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SSGConv

from arch import (
    AdaptiveAggGCN, HeteroGNN, GCN, FBGNNLayer,
    gfGNN, FAGCN, DirGNN, H2GCN,
)
from utils import get_data_dict, seed_everything
from train import train_model

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# ── CLI: optional dataset subset + device, for splitting across 2 GPUs ───────
DATASET_SUBSET = os.environ.get('MATCHED_DATASETS', '').split(',') if os.environ.get('MATCHED_DATASETS') else None
DEVICE_STR = os.environ.get('MATCHED_DEVICE', 'cuda:0')

assert torch.cuda.is_available(), "CUDA required"
device = torch.device(DEVICE_STR)

seed_everything(42)

try:
    commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT).decode().strip()
except Exception:
    commit_hash = 'unknown'

# ── hyperparams (mirror exp_2.py's original full config) ────────────────────
n_layers  = 2
hid_dim   = 32          # SG-GNN hidden dim -- UNCHANGED
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

# ── load cached graphs ────────────────────────────────────────────────────────
data_np = np.load(str(ROOT / 'node_embeddings' / 'embedding_graphs.npz'), allow_pickle=True)
all_graphs = {}
for filename in data_np.files:
    ds  = filename.split('_')[0]
    key = filename.split('_')[1]
    if ds not in all_graphs:
        all_graphs[ds] = {}
    all_graphs[ds][key] = data_np[filename]

graphs = list(all_graphs[ALL_DATASETS[0]].keys())
del graphs[graphs.index('EPS-GraphWave')]
R = len(graphs)
logging.info(f"Using {R} graph views: {graphs}")

data_dict = get_data_dict(datasets, all_graphs, node_name, n_sims, 0.8, 0.1)

# ── output ────────────────────────────────────────────────────────────────────
OUT_DIR = ROOT / 'results' / 'main_matched'
OUT_DIR.mkdir(parents=True, exist_ok=True)
datestamp = datetime.datetime.now().strftime('%Y%m%d')
suffix = f"-{DATASET_SUBSET[0]}_to_{DATASET_SUBSET[-1]}" if DATASET_SUBSET else ""
fname = str(OUT_DIR / f'{datestamp}-main_matched{suffix}')

# ── config names (match table rows exactly) ──────────────────────────────────
BASELINE_NAMES = ['GCN', 'GAT', 'gfNN', 'FAGCN', 'DirGNN', 'MixHop', 'SSGC', 'H2GCN']
SGGNN_NAMES    = ['SG-GCN_N', 'SG-GCN', 'SG-GCN_L', 'SG-FBGNN_N', 'SG-FBGNN', 'SG-FBGNN_L']
CONFIG_NAMES   = BASELINE_NAMES + SGGNN_NAMES
N_CONFIGS      = len(CONFIG_NAMES)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def find_matched_hid(build_fn, target, lo=32, hi=1024, hard_cap=32768):
    """Binary-search the smallest hid_dim >= lo such that build_fn(hid) has
    >= target params. Assumes monotonic non-decreasing param count in hid."""
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


def find_matched_num_layers(build_fn, target, lo=2, hi=50):
    """Linear-search the smallest num_layers >= lo such that build_fn(nl) has
    >= target params. Used for FAGCN, whose hidden_dim has no effect."""
    best_nl, best_params = lo, count_params(build_fn(lo))
    if best_params >= target:
        return best_nl, best_params
    for nl in range(lo + 1, hi + 1):
        p = count_params(build_fn(nl))
        if p >= target:
            return nl, p
        best_nl, best_params = nl, p
    return best_nl, best_params  # best effort, still short of target


best_accs_test = np.zeros((len(datasets), n_sims, N_CONFIGS), dtype=np.float32)
param_counts   = {}     # {dataset: {config_name: n_params}}
matched_hparams = {}    # {dataset: {baseline_name: {'hid'|'n_layers': value}}}

for d, dataset_name in enumerate(datasets):
    logging.info(f"\n=== Dataset {d+1}/{len(datasets)}: {dataset_name} ===")

    data_pyg    = data_dict[dataset_name].to(device)
    N           = data_pyg[node_name].N
    num_classes = data_pyg[node_name].num_classes
    in_dim      = data_pyg[node_name].x.size(1)
    out_dim     = num_classes

    X_het   = data_pyg.x_dict
    ei_het  = data_pyg.edge_index_dict
    X_hom   = data_pyg[node_name].x
    ei_orig = data_pyg[node_name, 'Original', node_name].edge_index

    # ── 1. compute SG-GNN variant param counts -> target budget ─────────────
    sggnn_builders = {
        'SG-GCN_N': lambda: AdaptiveAggGCN(in_dim, hid_dim, out_dim, n_layers, dropout, nonlin,
                                           nn.Identity(), graphs, GCNConv, GCNConv, N),
        'SG-GCN':   lambda: AdaptiveAggGCN(in_dim, hid_dim, out_dim, n_layers, dropout, nonlin,
                                           nn.Identity(), graphs, GCNConv, GCNConv, -1),
        'SG-GCN_L': lambda: HeteroGNN(in_dim, hid_dim, out_dim, n_layers, dropout, nonlin,
                                      nn.Identity(), graphs, 'cat', GCNConv, GCNConv),
        'SG-FBGNN_N': lambda: AdaptiveAggGCN(in_dim, hid_dim, out_dim, n_layers, dropout, nonlin,
                                             last_act_softmax, graphs, FBGNNLayer, FBGNNLayer, N),
        'SG-FBGNN':   lambda: AdaptiveAggGCN(in_dim, hid_dim, out_dim, n_layers, dropout, nonlin,
                                             last_act_softmax, graphs, FBGNNLayer, FBGNNLayer, -1),
        'SG-FBGNN_L': lambda: HeteroGNN(in_dim, hid_dim, out_dim, n_layers, dropout, nonlin,
                                        last_act_softmax, graphs, 'cat', FBGNNLayer, FBGNNLayer),
    }
    sggnn_params = {name: count_params(fn()) for name, fn in sggnn_builders.items()}
    target = max(sggnn_params.values())
    logging.info(f"  SG-GNN param counts: {sggnn_params}")
    logging.info(f"  Target budget for baselines: {target}")

    # ── 2. find matched capacity for each baseline ──────────────────────────
    param_counts[dataset_name]    = dict(sggnn_params)
    matched_hparams[dataset_name] = {}

    def gcn_builder(layer_cls, layer_kwargs=None):
        layer_kwargs = layer_kwargs or {}
        return lambda hid: GCN(in_dim, hid, out_dim, n_layers, nonlin=nonlin,
                                last_act=nn.Identity(), dropout=dropout,
                                gcnlayer=layer_cls, gcnlayer_kwargs=layer_kwargs)

    baseline_specs = {
        'GCN':    ('hid', gcn_builder(GCNConv)),
        'GAT':    ('hid', gcn_builder(GATConv)),
        'MixHop': ('hid', gcn_builder(FBGNNLayer)),
        'SSGC':   ('hid', gcn_builder(SSGConv, {'alpha': 0.05})),
        'gfNN':   ('hid', lambda hid: gfGNN(in_dim, hid, out_dim, n_layers, dropout, nonlin, nn.Identity())),
        'DirGNN': ('hid', lambda hid: DirGNN(in_dim, hid, out_dim, n_layers, dropout, nonlin, nn.Identity())),
        'H2GCN':  ('hid', lambda hid: H2GCN(in_dim, hid, out_dim, n_layers, dropout, nonlin, nn.Identity())),
        'FAGCN':  ('n_layers', lambda nl: FAGCN(in_dim, hid_dim, out_dim, nl, dropout, nonlin, nn.Identity())),
    }

    matched_val = {}
    for bname, (knob, builder) in baseline_specs.items():
        if knob == 'hid':
            val, achieved = find_matched_hid(builder, target)
        else:
            val, achieved = find_matched_num_layers(builder, target)
        matched_val[bname] = val
        param_counts[dataset_name][bname] = achieved
        matched_hparams[dataset_name][bname] = {knob: val, 'achieved_params': achieved, 'target': target}
        shortfall = "" if achieved >= target else f"  ** SHORT of target (arch limitation) **"
        logging.info(f"    {bname}: {knob}={val}, params={achieved}{shortfall}")

    del sggnn_builders

    # ── 3. build the 14 model configs for this dataset's sim loop ───────────
    def build_configs():
        cfgs = []
        # -- baselines (matched capacity) --
        cfgs.append(('GCN', GCN(in_dim, matched_val['GCN'], out_dim, n_layers, nonlin=nonlin,
                                 last_act=nn.Identity(), dropout=dropout, gcnlayer=GCNConv).to(device),
                     X_hom, ei_orig, {}))
        cfgs.append(('GAT', GCN(in_dim, matched_val['GAT'], out_dim, n_layers, nonlin=nonlin,
                                 last_act=nn.Identity(), dropout=dropout, gcnlayer=GATConv).to(device),
                     X_hom, ei_orig, {}))
        cfgs.append(('gfNN', gfGNN(in_dim, matched_val['gfNN'], out_dim, n_layers, dropout, nonlin,
                                    nn.Identity()).to(device), X_hom, ei_orig, {}))
        fagcn_model = FAGCN(in_dim, hid_dim, out_dim, matched_val['FAGCN'], dropout, nonlin,
                             nn.Identity()).to(device)
        cfgs.append(('FAGCN', fagcn_model, X_hom, ei_orig, {'x_0': X_hom.clone()}))
        cfgs.append(('DirGNN', DirGNN(in_dim, matched_val['DirGNN'], out_dim, n_layers, dropout, nonlin,
                                       nn.Identity()).to(device), X_hom, ei_orig, {}))
        cfgs.append(('MixHop', GCN(in_dim, matched_val['MixHop'], out_dim, n_layers, nonlin=nonlin,
                                    last_act=last_act_softmax, dropout=dropout,
                                    gcnlayer=FBGNNLayer).to(device), X_hom, ei_orig, {}))
        cfgs.append(('SSGC', GCN(in_dim, matched_val['SSGC'], out_dim, n_layers, nonlin=nonlin,
                                  last_act=nn.Identity(), dropout=dropout, gcnlayer=SSGConv,
                                  gcnlayer_kwargs={'alpha': 0.05}).to(device), X_hom, ei_orig, {}))
        cfgs.append(('H2GCN', H2GCN(in_dim, matched_val['H2GCN'], out_dim, n_layers, dropout, nonlin,
                                     nn.Identity()).to(device), X_hom, ei_orig, {}))
        # -- SG-GNN variants (unchanged, hid_dim=32) --
        cfgs.append(('SG-GCN_N', AdaptiveAggGCN(in_dim, hid_dim, out_dim, n_layers, dropout, nonlin,
                                                 nn.Identity(), graphs, GCNConv, GCNConv, N).to(device),
                     X_het, ei_het, {}))
        cfgs.append(('SG-GCN', AdaptiveAggGCN(in_dim, hid_dim, out_dim, n_layers, dropout, nonlin,
                                               nn.Identity(), graphs, GCNConv, GCNConv, -1).to(device),
                     X_het, ei_het, {}))
        cfgs.append(('SG-GCN_L', HeteroGNN(in_dim, hid_dim, out_dim, n_layers, dropout, nonlin,
                                            nn.Identity(), graphs, 'cat', GCNConv, GCNConv).to(device),
                     X_het, ei_het, {}))
        cfgs.append(('SG-FBGNN_N', AdaptiveAggGCN(in_dim, hid_dim, out_dim, n_layers, dropout, nonlin,
                                                   last_act_softmax, graphs, FBGNNLayer, FBGNNLayer, N).to(device),
                     X_het, ei_het, {}))
        cfgs.append(('SG-FBGNN', AdaptiveAggGCN(in_dim, hid_dim, out_dim, n_layers, dropout, nonlin,
                                                 last_act_softmax, graphs, FBGNNLayer, FBGNNLayer, -1).to(device),
                     X_het, ei_het, {}))
        cfgs.append(('SG-FBGNN_L', HeteroGNN(in_dim, hid_dim, out_dim, n_layers, dropout, nonlin,
                                              last_act_softmax, graphs, 'cat', FBGNNLayer, FBGNNLayer).to(device),
                     X_het, ei_het, {}))
        return cfgs

    # ── 4. simulation loop ────────────────────────────────────────────────────
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

        for ci, (cname, model, X, ei, fw_kw) in enumerate(build_configs()):
            try:
                _, _, _, _, val_accs, test_accs = train_model(
                    model, X, ei, data_pyg[node_name].y,
                    tr, vl, te, fw_kw, None, lr, wd, epochs, patience, verb=False)
                best_accs_test[d, s, ci] = test_accs[int(np.argmax(val_accs))]
            except torch.cuda.OutOfMemoryError:
                logging.warning(f"    CUDA OOM: {cname}, sim {s} -- skipping")
                torch.cuda.empty_cache()
            except Exception as exc:
                logging.warning(f"    Error in {cname}, sim {s}: {exc}")

        if (s + 1) % 5 == 0:
            logging.info(f"  sim {s+1}/{n_sims} done")

    for ci, cname in enumerate(CONFIG_NAMES):
        m_ = best_accs_test[d, :, ci].mean()
        s_ = best_accs_test[d, :, ci].std()
        logging.info(f"  {cname}: {m_:.4f} +/- {s_:.4f}")

    np.save(fname + '.npy', best_accs_test)  # checkpoint

# ── final save ────────────────────────────────────────────────────────────────
np.save(fname + '.npy', best_accs_test)

means = best_accs_test.mean(axis=1)
stds  = best_accs_test.std(axis=1)
max_indices = np.argmax(means, axis=1)

results_dict = {
    'results':        best_accs_test,
    'config_names':   CONFIG_NAMES,
    'baseline_names':  BASELINE_NAMES,
    'sggnn_names':    SGGNN_NAMES,
    'param_counts':   param_counts,
    'matched_hparams': matched_hparams,
    'metadata': {
        'seed': 42, 'datasets': datasets, 'timestamp': datetime.datetime.now().isoformat(),
        'commit_hash': commit_hash,
        'hid_dim_sggnn': hid_dim, 'dropout': dropout, 'lr': lr, 'wd': wd,
        'epochs': epochs, 'patience': patience, 'n_sims': n_sims, 'n_layers': n_layers,
        'graphs': graphs,
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

# ── regenerate LaTeX table ────────────────────────────────────────────────────
lines = []
lines.append(r"\begin{table*}[t]")
lines.append(r"    \centering")
lines.append(r"    \setlength{\tabcolsep}{2pt}")
lines.append(r"    \begin{tabular}{l|" + "c" * len(datasets) + "}")
lines.append(r"    \toprule")
lines.append("     & " + " & ".join(datasets) + r" \\")
lines.append(r"    \midrule")

for ci, cname in enumerate(CONFIG_NAMES):
    if cname == 'H2GCN':
        after = r"    \midrule"
    elif cname == 'SG-GCN_L':
        after = ""  # keep GCN-block/FBGNN-block adjacent; no midrule requested here
    else:
        after = ""
    row = [cname.replace('_N', '$_N$').replace('_L', '$_L$')]
    for d in range(len(datasets)):
        mean = means[d, ci] * 100
        std  = stds[d, ci] * 100
        cell = f"{mean:.2f}{{\\scriptsize\\! $\\pm\\!\\!$ {std:.2f}}}"
        if max_indices[d] == ci:
            cell = r"\textbf{" + cell + "}"
        row.append(cell)
    lines.append(" & ".join(row) + r" \\")
    if after:
        lines.append(after)

lines.append(r"    \bottomrule")
lines.append(r"    \end{tabular}")
lines.append(r"    \caption{Node classification accuracy (\%) comparing our SG-GNN architectures with "
             r"state-of-the-art baselines across all datasets, with baseline capacity increased so each "
             r"has parameter count $\geq$ the largest SG-GNN variant for that dataset (see "
             r"Appendix/Sec.~\ref{S:sg-gnn} for the matched parameter counts). "
             r"SG-GNN$_N$ refers to the node-specific adaptive single-layer model, SG-GNN is the global "
             r"adaptive single-layer model, and SG-GNN$_L$ is the multi-layer ($L=2$) adaptive model. "
             r"Each is implemented with both GCN and FBGNN base layers. Best results per dataset are in "
             r"\textbf{bold}.}")
lines.append(r"        \label{tab:metrics_sggnn}")
lines.append(r"\end{table*}")

table_fname = str(OUT_DIR / f'{datestamp}-table{suffix}.tex')
with open(table_fname, 'w') as f:
    f.write('\n'.join(lines))

logging.info(f"\nDone. Results saved to {fname}{{.npy,.pkl,.json}}")
logging.info(f"LaTeX table saved to {table_fname}")
