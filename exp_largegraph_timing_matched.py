"""
Training time and peak GPU memory on Roman-Empire at MATCHED capacity --
i.e., using the same hidden dimensions (or depth, for FAGCN) that produced
the parameter-fair accuracy comparison in exp_largegraph_matched.py, rather
than the flat hidden dimension of 32 used in exp_largegraph_timing.py.

This answers a different, complementary question from the original timing
table: not "what does it cost to add R parallel branches at equal per-branch
width" (the original table, kept as-is), but "what does the actual fair
comparison setup cost in wall-clock time and memory."

Mirrors exp_largegraph_timing.py's methodology exactly (5 warmup epochs, 20
timed epochs, 3 repeats) but reads matched hidden dims from the saved
exp_largegraph_matched.py results instead of using hid_dim=32 for baselines.

Output:
  results/largegraph/<date>-largegraph_timing_matched.{pkl,json}
"""

import sys
import json
import logging
import pickle
import time
import datetime
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

from arch import AdaptiveAggGCN, GCN, FBGNNLayer, gfGNN, FAGCN, DirGNN, H2GCN
from utils import seed_everything
from grand_wrapper import GRANDWrapper
from cdgnn_wrapper import CDGNNWrapper
from lggnn_wrapper import LGGNNWrapper

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

assert torch.cuda.is_available(), "CUDA required"
device = torch.device('cuda:0')
seed_everything(42)

OUT_DIR = ROOT / 'results' / 'largegraph'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load matched hyperparameters from the accuracy run ──────────────────────
MATCHED_PKL = ROOT / 'results' / 'largegraph' / '20260725-largegraph_matched.pkl'
with open(MATCHED_PKL, 'rb') as f:
    matched_results = pickle.load(f)
matched_hparams = matched_results['matched_hparams']
sggnn_hid = 32  # SG-GNN's own hidden dim -- unchanged throughout this project

# ── hyperparams (match exp_largegraph_matched.py) ────────────────────────────
n_layers = 2
dropout  = 0.5
lr       = 5e-3
wd       = 5e-4
K_knn    = 10
node_name = 'web'
nonlin   = nn.Tanh()
last_act_softmax = nn.Softmax(dim=1)

WARMUP_EPOCHS = 5
TIMED_EPOCHS  = 20
n_sims        = 3

# ── load Roman-Empire + build graphs (same as exp_largegraph_matched.py) ────
logging.info("Loading Roman-Empire ...")
dataset = HeterophilousGraphDataset(root='~/.datapyg', name='Roman-empire')
g = dataset[0]
N = g.num_nodes
num_classes = int(g.y.max().item()) + 1
in_dim = g.x.size(1)


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
data_pyg[node_name, 'Original', node_name].edge_index = g.edge_index
data_pyg[node_name, 'KNN-Feat', node_name].edge_index = ei_feat
data_pyg[node_name, 'KNN-Role', node_name].edge_index = ei_role
data_pyg[node_name, 'EgoFeats', node_name].edge_index = torch.arange(N).unsqueeze(0).repeat(2, 1)
graphs_hetero = ['Original', 'KNN-Feat', 'KNN-Role', 'EgoFeats']
data_pyg = data_pyg.to(device)

X_hom = data_pyg[node_name].x
ei_orig = data_pyg[node_name, 'Original', node_name].edge_index
X_het = data_pyg.x_dict
ei_het = data_pyg.edge_index_dict
y = data_pyg[node_name].y

tm = g.train_mask
train_mask = (tm[:, 0] if tm.ndim > 1 else tm).to(device)
criterion = nn.CrossEntropyLoss()


def time_epoch(model, X, edge_idx, y, train_mask, optimizer, criterion, fw_kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.train()
    optimizer.zero_grad()
    out = model(X, edge_index=edge_idx, **fw_kwargs)
    loss = criterion(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ── model specs at MATCHED capacity ──────────────────────────────────────────
def h(name):
    return matched_hparams[name]['hid']


model_specs = [
    {'name': 'GCN', 'build': lambda: GCN(in_dim, h('GCN'), num_classes, n_layers,
                                          nonlin=nonlin, last_act=nn.Identity(),
                                          dropout=dropout, gcnlayer=GCNConv).to(device),
     'X': X_hom, 'ei': ei_orig, 'fw_kwargs': {}},
    {'name': 'GAT', 'build': lambda: GCN(in_dim, h('GAT'), num_classes, n_layers,
                                          nonlin=nonlin, last_act=nn.Identity(),
                                          dropout=dropout, gcnlayer=GATConv).to(device),
     'X': X_hom, 'ei': ei_orig, 'fw_kwargs': {}},
    {'name': 'gfNN', 'build': lambda: gfGNN(in_dim, h('gfNN'), num_classes, n_layers,
                                             dropout, nonlin, nn.Identity()).to(device),
     'X': X_hom, 'ei': ei_orig, 'fw_kwargs': {}},
    {'name': 'FAGCN', 'build': lambda: FAGCN(in_dim, sggnn_hid, num_classes,
                                              matched_hparams['FAGCN']['n_layers'],
                                              dropout, nonlin, nn.Identity()).to(device),
     'X': X_hom, 'ei': ei_orig, 'fw_kwargs': {'x_0': X_hom.clone()}},
    {'name': 'DirGNN', 'build': lambda: DirGNN(in_dim, h('DirGNN'), num_classes, n_layers,
                                                dropout, nonlin, nn.Identity()).to(device),
     'X': X_hom, 'ei': ei_orig, 'fw_kwargs': {}},
    {'name': 'MixHop', 'build': lambda: GCN(in_dim, h('MixHop'), num_classes, n_layers,
                                             nonlin=nonlin, last_act=last_act_softmax,
                                             dropout=dropout, gcnlayer=FBGNNLayer).to(device),
     'X': X_hom, 'ei': ei_orig, 'fw_kwargs': {}},
    {'name': 'SSGC', 'build': lambda: GCN(in_dim, h('SSGC'), num_classes, n_layers,
                                           nonlin=nonlin, last_act=nn.Identity(), dropout=dropout,
                                           gcnlayer=SSGConv, gcnlayer_kwargs={'alpha': 0.05}).to(device),
     'X': X_hom, 'ei': ei_orig, 'fw_kwargs': {}},
    {'name': 'H2GCN', 'build': lambda: H2GCN(in_dim, h('H2GCN'), num_classes, n_layers,
                                              dropout, nonlin, nn.Identity()).to(device),
     'X': X_hom, 'ei': ei_orig, 'fw_kwargs': {}},
    {'name': 'GRAND', 'build': lambda: GRANDWrapper(in_dim, h('GRAND'), num_classes, n_layers,
                                                     dropout, nonlin, last_act_softmax).to(device),
     'X': X_hom, 'ei': ei_orig, 'fw_kwargs': {}},
    {'name': 'CD-GNN', 'build': lambda: CDGNNWrapper(in_dim, h('CD-GNN'), num_classes, n_layers,
                                                      dropout, nonlin, last_act_softmax).to(device),
     'X': X_hom, 'ei': ei_orig, 'fw_kwargs': {}},
    {'name': 'LG-GNN', 'build': lambda: LGGNNWrapper(in_dim, h('LG-GNN'), num_classes, n_layers,
                                                      dropout, nonlin, last_act_softmax).to(device),
     'X': X_hom, 'ei': ei_orig, 'fw_kwargs': {}},
    {'name': 'SG-GCN', 'build': lambda: AdaptiveAggGCN(
        in_dim, sggnn_hid, num_classes, n_layers, dropout, nonlin, nn.Identity(),
        graphs_hetero, GCNConv, GCNConv, -1).to(device),
     'X': X_het, 'ei': ei_het, 'fw_kwargs': {}},
    {'name': 'SG-FBGNN', 'build': lambda: AdaptiveAggGCN(
        in_dim, sggnn_hid, num_classes, n_layers, dropout, nonlin, last_act_softmax,
        graphs_hetero, FBGNNLayer, FBGNNLayer, -1).to(device),
     'X': X_het, 'ei': ei_het, 'fw_kwargs': {}},
]

timing_results = {}
for spec in model_specs:
    mname = spec['name']
    logging.info(f"Benchmarking {mname} (matched capacity) ...")
    epoch_times_all_sims = []
    peak_mem_all = []
    try:
        for sim in range(n_sims):
            model = spec['build']()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            torch.cuda.reset_peak_memory_stats(device)
            for _ in range(WARMUP_EPOCHS):
                time_epoch(model, spec['X'], spec['ei'], y, train_mask, optimizer, criterion, spec['fw_kwargs'])
            torch.cuda.reset_peak_memory_stats(device)
            epoch_times = [time_epoch(model, spec['X'], spec['ei'], y, train_mask, optimizer, criterion, spec['fw_kwargs'])
                           for _ in range(TIMED_EPOCHS)]
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            epoch_times_all_sims.extend(epoch_times)
            peak_mem_all.append(peak_mem)

        n_params = count_params(model)
        mean_t = float(np.mean(epoch_times_all_sims))
        std_t = float(np.std(epoch_times_all_sims))
        mean_mem = float(np.mean(peak_mem_all))
        timing_results[mname] = {
            'time_per_epoch_mean': mean_t, 'time_per_epoch_std': std_t,
            'peak_memory_mb': mean_mem, 'n_params': n_params,
            'matched_hid_or_layers': matched_hparams.get(mname, {'hid': sggnn_hid}),
        }
        logging.info(f"  {mname}: {mean_t*1000:.2f} ms/epoch (+/- {std_t*1000:.2f}), "
                     f"peak mem={mean_mem:.1f} MB, params={n_params}")
    except torch.cuda.OutOfMemoryError:
        logging.warning(f"  CUDA OOM on {mname} -- skipping")
        torch.cuda.empty_cache()
        timing_results[mname] = {'error': 'CUDA OOM'}
    except Exception as exc:
        logging.warning(f"  Error on {mname}: {exc}")
        timing_results[mname] = {'error': str(exc)}

datestamp = datetime.datetime.now().strftime('%Y%m%d')
fname = str(OUT_DIR / f'{datestamp}-largegraph_timing_matched')

full_results = {
    'results': timing_results,
    'metadata': {
        'dataset': 'Roman-empire', 'n_nodes': N, 'n_classes': num_classes,
        'seed': 42, 'timestamp': datetime.datetime.now().isoformat(),
        'sggnn_hid_dim': sggnn_hid, 'dropout': dropout, 'lr': lr, 'wd': wd,
        'warmup_epochs': WARMUP_EPOCHS, 'timed_epochs': TIMED_EPOCHS,
        'n_sims': n_sims, 'n_layers': n_layers, 'graphs': graphs_hetero,
        'source_matched_pkl': str(MATCHED_PKL),
    }
}

with open(fname + '.pkl', 'wb') as f:
    pickle.dump(full_results, f)


def to_json_safe(obj):
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    return obj


with open(fname + '.json', 'w') as f:
    json.dump(to_json_safe(full_results), f, indent=2)

logging.info(f"\nDone. Saved to {fname}{{.pkl,.json}}")
