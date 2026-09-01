"""
Full parameter-fair comparison on Roman-Empire (22,662 nodes), extending
exp_largegraph.py to the same standard as the main 11-dataset table
(exp_main_matched.py): all 11 baselines given capacity >= the largest SG-GNN
variant's parameter count, and all 6 SG-GNN variants evaluated (the original
exp_largegraph.py only ran SG-GCN and SG-FBGNN, and none of the baselines
had matched capacity -- confirmed via results/largegraph/*_timing.pkl that
SG-GCN had ~4x the parameters of GCN/GAT/GRAND there, unlike the fixed
11-dataset table).

Graph views: same 4 as exp_largegraph.py (Original, KNN-Feat, KNN-Role,
EgoFeats) -- R=4, not R=14, since the full embedding-based candidate pool
does not scale to this graph size. This keeps the SG-GNN parameter budget
comparatively small, so matched baselines only need modest hidden-dim
increases (order of hundreds, not thousands).

Output:
  results/largegraph/<date>-largegraph_matched.{npy,pkl,json}
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
from scipy.sparse import csr_matrix, eye, diags
from sklearn.neighbors import NearestNeighbors
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GATConv, SSGConv

from arch import AdaptiveAggGCN, HeteroGNN, GCN, FBGNNLayer, gfGNN, FAGCN, DirGNN, H2GCN
from utils import seed_everything
from train import train_model
from grand_wrapper import GRANDWrapper
from cdgnn_wrapper import CDGNNWrapper
from lggnn_wrapper import LGGNNWrapper

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

DEVICE_STR = os.environ.get('MATCHED_DEVICE', 'cuda:0')
assert torch.cuda.is_available(), "CUDA required"
device = torch.device(DEVICE_STR)
seed_everything(42)

try:
    commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT).decode().strip()
except Exception:
    commit_hash = 'unknown'

n_layers  = 2
hid_dim   = 32  # SG-GNN hidden dim -- unchanged
dropout   = 0.5
nonlin    = nn.Tanh()
last_act_softmax = nn.Softmax(dim=1)
lr        = 5e-3
wd        = 5e-4
epochs    = int(os.environ.get('MATCHED_EPOCHS', 2000))
patience  = int(os.environ.get('MATCHED_PATIENCE', 300))
n_sims    = int(os.environ.get('MATCHED_NSIMS', 10))
K_knn     = 10
node_name = 'web'

OUT_DIR = ROOT / 'results' / 'largegraph'
OUT_DIR.mkdir(parents=True, exist_ok=True)
datestamp = datetime.datetime.now().strftime('%Y%m%d')
fname = str(OUT_DIR / f'{datestamp}-largegraph_matched')

# ── load Roman-Empire + build the same 4 graphs as exp_largegraph.py ────────
logging.info("Loading Roman-Empire ...")
dataset = HeterophilousGraphDataset(root='~/.datapyg', name='Roman-empire')
g = dataset[0]
N = g.num_nodes
num_classes = int(g.y.max().item()) + 1
in_dim = g.x.size(1)
logging.info(f"Roman-Empire: {N} nodes, {g.edge_index.shape[1]} edges, {num_classes} classes")


def compute_role_features_sparse(edge_index, num_nodes):
    row, col = edge_index.numpy()
    A = csr_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
    degs = np.array(A.sum(axis=1)).flatten()
    f = np.zeros((7, num_nodes), dtype=np.float32)
    f[0] = degs
    for i in range(num_nodes):
        inds = A[i].nonzero()[1]
        inds = np.concatenate([[i], inds])
        sub = A[inds][:, inds]
        f[1][i] = sub.sum()
        f[2][i] = degs[inds].sum()
    f[3] = np.where(f[2] > 0, f[1] / f[2], 0)
    f[4] = np.where(f[2] > 0, 1 - f[3], 0)
    A3 = A @ A @ A
    f[5] = np.array(A3.diagonal(), dtype=np.float32)
    f[6] = np.where(f[0] > 1, 2 * f[5] / (f[0] * (f[0] - 1)), 0)
    A_hat = A + eye(num_nodes)
    d_hat = np.array(A_hat.sum(axis=1)).flatten()
    D_inv = diags(np.where(d_hat > 0, 1.0 / d_hat, 0.0))
    F = np.concatenate([f.T, (D_inv @ A_hat @ f.T), (A_hat @ f.T)], axis=1)
    scale = np.abs(F).max(axis=1, keepdims=True)
    scale[scale == 0] = 1
    return (F / scale).astype(np.float32)


def knn_edge_index(feats, k, num_nodes):
    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1)
    nn_model.fit(feats)
    _, indices = nn_model.kneighbors(feats)
    src = np.repeat(np.arange(num_nodes), k)
    dst = indices[:, 1:].flatten()
    src_full = np.concatenate([src, dst])
    dst_full = np.concatenate([dst, src])
    unique = np.unique(np.stack([src_full, dst_full], axis=1), axis=0)
    return torch.from_numpy(unique.T).long()


logging.info("Computing kNN-Feat graph ...")
ei_feat = knn_edge_index(g.x.numpy(), K_knn, N)
logging.info("Computing role features + kNN-Role graph ...")
role_feats = compute_role_features_sparse(g.edge_index, N)
ei_role = knn_edge_index(role_feats, K_knn, N)

data_pyg = HeteroData()
data_pyg[node_name].x = g.x
data_pyg[node_name].y = g.y
data_pyg[node_name].train_mask = g.train_mask
data_pyg[node_name].val_mask = g.val_mask
data_pyg[node_name].test_mask = g.test_mask
data_pyg[node_name, 'Original', node_name].edge_index = g.edge_index
data_pyg[node_name, 'KNN-Feat', node_name].edge_index = ei_feat
data_pyg[node_name, 'KNN-Role', node_name].edge_index = ei_role
data_pyg[node_name, 'EgoFeats', node_name].edge_index = torch.arange(N).unsqueeze(0).repeat(2, 1)
graphs_hetero = ['Original', 'KNN-Feat', 'KNN-Role', 'EgoFeats']
R = len(graphs_hetero)
data_pyg = data_pyg.to(device)

tm = g.train_mask
n_splits = tm.shape[1] if tm.ndim > 1 else 1
X_hom = data_pyg[node_name].x
ei_orig = data_pyg[node_name, 'Original', node_name].edge_index
X_het = data_pyg.x_dict
ei_het = data_pyg.edge_index_dict
y = data_pyg[node_name].y

# ── config names ──────────────────────────────────────────────────────────────
BASELINE_NAMES = ['GCN', 'GAT', 'gfNN', 'FAGCN', 'DirGNN', 'MixHop', 'SSGC', 'H2GCN',
                  'GRAND', 'CD-GNN', 'LG-GNN']
SGGNN_NAMES    = ['SG-GCN_N', 'SG-GCN', 'SG-GCN_L', 'SG-FBGNN_N', 'SG-FBGNN', 'SG-FBGNN_L']
CONFIG_NAMES   = BASELINE_NAMES + SGGNN_NAMES


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


def find_matched_num_layers(build_fn, target, lo=2, hi=50):
    best_nl, best_params = lo, count_params(build_fn(lo))
    if best_params >= target:
        return best_nl, best_params
    for nl in range(lo + 1, hi + 1):
        p = count_params(build_fn(nl))
        if p >= target:
            return nl, p
        best_nl, best_params = nl, p
    return best_nl, best_params

# ── 1. compute SG-GNN variant param counts -> target budget ─────────────────
sggnn_builders = {
    'SG-GCN_N': lambda: AdaptiveAggGCN(in_dim, hid_dim, num_classes, n_layers, dropout, nonlin,
                                       nn.Identity(), graphs_hetero, GCNConv, GCNConv, N),
    'SG-GCN':   lambda: AdaptiveAggGCN(in_dim, hid_dim, num_classes, n_layers, dropout, nonlin,
                                       nn.Identity(), graphs_hetero, GCNConv, GCNConv, -1),
    'SG-GCN_L': lambda: HeteroGNN(in_dim, hid_dim, num_classes, n_layers, dropout, nonlin,
                                  nn.Identity(), graphs_hetero, 'cat', GCNConv, GCNConv),
    'SG-FBGNN_N': lambda: AdaptiveAggGCN(in_dim, hid_dim, num_classes, n_layers, dropout, nonlin,
                                         last_act_softmax, graphs_hetero, FBGNNLayer, FBGNNLayer, N),
    'SG-FBGNN':   lambda: AdaptiveAggGCN(in_dim, hid_dim, num_classes, n_layers, dropout, nonlin,
                                         last_act_softmax, graphs_hetero, FBGNNLayer, FBGNNLayer, -1),
    'SG-FBGNN_L': lambda: HeteroGNN(in_dim, hid_dim, num_classes, n_layers, dropout, nonlin,
                                    last_act_softmax, graphs_hetero, 'cat', FBGNNLayer, FBGNNLayer),
}
sggnn_params = {name: count_params(fn()) for name, fn in sggnn_builders.items()}
target = max(sggnn_params.values())
logging.info(f"SG-GNN param counts: {sggnn_params}")
logging.info(f"Target budget for baselines: {target}")
del sggnn_builders

# ── 2. find matched capacity for each baseline ──────────────────────────────
def gcn_builder(layer_cls, layer_kwargs=None):
    layer_kwargs = layer_kwargs or {}
    return lambda hid: GCN(in_dim, hid, num_classes, n_layers, nonlin=nonlin,
                            last_act=nn.Identity(), dropout=dropout,
                            gcnlayer=layer_cls, gcnlayer_kwargs=layer_kwargs)


baseline_specs = {
    'GCN':    ('hid', gcn_builder(GCNConv)),
    'GAT':    ('hid', gcn_builder(GATConv)),
    'MixHop': ('hid', gcn_builder(FBGNNLayer)),
    'SSGC':   ('hid', gcn_builder(SSGConv, {'alpha': 0.05})),
    'gfNN':   ('hid', lambda hid: gfGNN(in_dim, hid, num_classes, n_layers, dropout, nonlin, nn.Identity())),
    'DirGNN': ('hid', lambda hid: DirGNN(in_dim, hid, num_classes, n_layers, dropout, nonlin, nn.Identity())),
    'H2GCN':  ('hid', lambda hid: H2GCN(in_dim, hid, num_classes, n_layers, dropout, nonlin, nn.Identity())),
    'GRAND':  ('hid', lambda hid: GRANDWrapper(in_dim, hid, num_classes, n_layers, dropout, nonlin, last_act_softmax)),
    'CD-GNN': ('hid', lambda hid: CDGNNWrapper(in_dim, hid, num_classes, n_layers, dropout, nonlin, last_act_softmax)),
    'LG-GNN': ('hid', lambda hid: LGGNNWrapper(in_dim, hid, num_classes, n_layers, dropout, nonlin, last_act_softmax)),
    'FAGCN':  ('n_layers', lambda nl: FAGCN(in_dim, hid_dim, num_classes, nl, dropout, nonlin, nn.Identity())),
}

matched_val = {}
param_counts = dict(sggnn_params)
matched_hparams = {}
for bname, (knob, builder) in baseline_specs.items():
    if knob == 'hid':
        val, achieved = find_matched_hid(builder, target)
    else:
        val, achieved = find_matched_num_layers(builder, target)
    matched_val[bname] = val
    param_counts[bname] = achieved
    matched_hparams[bname] = {knob: val, 'achieved_params': achieved, 'target': target}
    shortfall = "" if achieved >= target else "  ** SHORT of target (arch limitation) **"
    logging.info(f"  {bname}: {knob}={val}, params={achieved}{shortfall}")

# ── 3. build the 17 model configs ────────────────────────────────────────────
def build_configs():
    cfgs = []
    cfgs.append(('GCN', GCN(in_dim, matched_val['GCN'], num_classes, n_layers, nonlin=nonlin,
                             last_act=nn.Identity(), dropout=dropout, gcnlayer=GCNConv).to(device),
                 X_hom, ei_orig, {}))
    cfgs.append(('GAT', GCN(in_dim, matched_val['GAT'], num_classes, n_layers, nonlin=nonlin,
                             last_act=nn.Identity(), dropout=dropout, gcnlayer=GATConv).to(device),
                 X_hom, ei_orig, {}))
    cfgs.append(('gfNN', gfGNN(in_dim, matched_val['gfNN'], num_classes, n_layers, dropout, nonlin,
                                nn.Identity()).to(device), X_hom, ei_orig, {}))
    fagcn_model = FAGCN(in_dim, hid_dim, num_classes, matched_val['FAGCN'], dropout, nonlin,
                         nn.Identity()).to(device)
    cfgs.append(('FAGCN', fagcn_model, X_hom, ei_orig, {'x_0': X_hom.clone()}))
    cfgs.append(('DirGNN', DirGNN(in_dim, matched_val['DirGNN'], num_classes, n_layers, dropout, nonlin,
                                   nn.Identity()).to(device), X_hom, ei_orig, {}))
    cfgs.append(('MixHop', GCN(in_dim, matched_val['MixHop'], num_classes, n_layers, nonlin=nonlin,
                                last_act=last_act_softmax, dropout=dropout,
                                gcnlayer=FBGNNLayer).to(device), X_hom, ei_orig, {}))
    cfgs.append(('SSGC', GCN(in_dim, matched_val['SSGC'], num_classes, n_layers, nonlin=nonlin,
                              last_act=nn.Identity(), dropout=dropout, gcnlayer=SSGConv,
                              gcnlayer_kwargs={'alpha': 0.05}).to(device), X_hom, ei_orig, {}))
    cfgs.append(('H2GCN', H2GCN(in_dim, matched_val['H2GCN'], num_classes, n_layers, dropout, nonlin,
                                 nn.Identity()).to(device), X_hom, ei_orig, {}))
    cfgs.append(('GRAND', GRANDWrapper(in_dim, matched_val['GRAND'], num_classes, n_layers, dropout,
                                        nonlin, last_act_softmax).to(device), X_hom, ei_orig, {}))
    cfgs.append(('CD-GNN', CDGNNWrapper(in_dim, matched_val['CD-GNN'], num_classes, n_layers, dropout,
                                         nonlin, last_act_softmax).to(device), X_hom, ei_orig, {}))
    cfgs.append(('LG-GNN', LGGNNWrapper(in_dim, matched_val['LG-GNN'], num_classes, n_layers, dropout,
                                         nonlin, last_act_softmax).to(device), X_hom, ei_orig, {}))
    # SG-GNN variants (unchanged, hid_dim=32)
    cfgs.append(('SG-GCN_N', AdaptiveAggGCN(in_dim, hid_dim, num_classes, n_layers, dropout, nonlin,
                                             nn.Identity(), graphs_hetero, GCNConv, GCNConv, N).to(device),
                 X_het, ei_het, {}))
    cfgs.append(('SG-GCN', AdaptiveAggGCN(in_dim, hid_dim, num_classes, n_layers, dropout, nonlin,
                                           nn.Identity(), graphs_hetero, GCNConv, GCNConv, -1).to(device),
                 X_het, ei_het, {}))
    cfgs.append(('SG-GCN_L', HeteroGNN(in_dim, hid_dim, num_classes, n_layers, dropout, nonlin,
                                        nn.Identity(), graphs_hetero, 'cat', GCNConv, GCNConv).to(device),
                 X_het, ei_het, {}))
    cfgs.append(('SG-FBGNN_N', AdaptiveAggGCN(in_dim, hid_dim, num_classes, n_layers, dropout, nonlin,
                                               last_act_softmax, graphs_hetero, FBGNNLayer, FBGNNLayer, N).to(device),
                 X_het, ei_het, {}))
    cfgs.append(('SG-FBGNN', AdaptiveAggGCN(in_dim, hid_dim, num_classes, n_layers, dropout, nonlin,
                                             last_act_softmax, graphs_hetero, FBGNNLayer, FBGNNLayer, -1).to(device),
                 X_het, ei_het, {}))
    cfgs.append(('SG-FBGNN_L', HeteroGNN(in_dim, hid_dim, num_classes, n_layers, dropout, nonlin,
                                          last_act_softmax, graphs_hetero, 'cat', FBGNNLayer, FBGNNLayer).to(device),
                 X_het, ei_het, {}))
    return cfgs


# ── 4. simulation loop ────────────────────────────────────────────────────────
best_accs_test = np.zeros((n_sims, len(CONFIG_NAMES)), dtype=np.float32)

for s in range(n_sims):
    idx = s % n_splits
    tr = data_pyg[node_name].train_mask[:, idx] if n_splits > 1 else data_pyg[node_name].train_mask
    vl = data_pyg[node_name].val_mask[:, idx] if n_splits > 1 else data_pyg[node_name].val_mask
    te = data_pyg[node_name].test_mask[:, idx] if n_splits > 1 else data_pyg[node_name].test_mask

    logging.info(f"Sim {s+1}/{n_sims}")

    for ci, (cname, model, X, ei, fw_kw) in enumerate(build_configs()):
        try:
            _, _, _, _, val_accs, test_accs = train_model(
                model, X, ei, y, tr, vl, te, fw_kw, None, lr, wd, epochs, patience, verb=False)
            best_accs_test[s, ci] = test_accs[int(np.argmax(val_accs))]
        except torch.cuda.OutOfMemoryError:
            logging.warning(f"  CUDA OOM: {cname}, sim {s} -- skipping")
            torch.cuda.empty_cache()
        except Exception as exc:
            logging.warning(f"  Error in {cname}, sim {s}: {exc}")
        logging.info(f"  {cname}: {best_accs_test[s, ci]:.4f}")

    np.save(fname + '.npy', best_accs_test)  # checkpoint every sim

means = best_accs_test.mean(axis=0)
stds  = best_accs_test.std(axis=0)
logging.info("\n=== Roman-Empire full parameter-fair comparison ===")
for ci, cname in enumerate(CONFIG_NAMES):
    logging.info(f"  {cname:12s}  {means[ci]:.4f} +/- {stds[ci]:.4f}")

# ── save ──────────────────────────────────────────────────────────────────────
np.save(fname + '.npy', best_accs_test)

meta = {
    'dataset': 'Roman-empire', 'n_nodes': N, 'n_classes': num_classes,
    'config_names': CONFIG_NAMES, 'baseline_names': BASELINE_NAMES, 'sggnn_names': SGGNN_NAMES,
    'best_accs_test': best_accs_test.tolist(), 'means': means.tolist(), 'stds': stds.tolist(),
    'param_counts': param_counts, 'matched_hparams': matched_hparams,
    'graphs': graphs_hetero,
    'hyperparams': {
        'n_layers': n_layers, 'hid_dim_sggnn': hid_dim, 'dropout': dropout, 'lr': lr, 'wd': wd,
        'epochs': epochs, 'patience': patience, 'K_knn': K_knn, 'n_sims': n_sims, 'seed': 42,
    },
    'commit_hash': commit_hash, 'timestamp': datetime.datetime.now().isoformat(),
}

with open(fname + '.pkl', 'wb') as f:
    pickle.dump({**meta, 'best_accs_test': best_accs_test}, f)
with open(fname + '.json', 'w') as f:
    json.dump(meta, f, indent=2)

logging.info(f"Saved to {fname}{{.npy,.pkl,.json}}")
