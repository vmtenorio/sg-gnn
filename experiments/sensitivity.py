"""
Sensitivity sweeps: k (Table VIII, `tab:k_sweep`), epsilon-ball average degree
(Table IX, `tab:eps_sweep`), and R -- number of candidate graphs, random
subsets (Table X, `tab:r_sweep`). Merges the former sensitivity_sweep.py
(k sweep) and sensitivity_sweep_v2.py (eps sweep, R sweep) into one module;
the old sensitivity_sweep.py's superseded fixed-sequence R ablation (Part B)
is dropped, not ported.

Model: single-layer SG-GCN, hidden dimension 32, on Texas, Squirrel, Cora and
Actor. Runs all three sweeps by default; select a subset with
SENSITIVITY_PARTS=k,eps,r (comma-separated). Needs the candidate-graph cache
for the R sweep's 14-view pool (`python main.py build-graphs`); the k/eps
sweeps rebuild their own graphs on the fly and do not need it.

k and eps sweeps use SENS_SPLIT='random' by default (matching every other
table); pass SENS_SPLIT=public to use each dataset's own masks instead. The R
sweep is unaffected by this flag -- it always draws random splits via
sggnn.data.get_data_dict (default split='random') -- and reproduces its
subsets deterministically via zlib.crc32 (stable_seed below), with a
SENS_RELOAD_SUBSETS_FROM option to replay a prior run's exact subsets.

`analysis/sensitivity_tables.py` builds Tables VIII/IX/X from this module's
saved output.

Output: results/<RESULTS_DIR>/{table8,table9,table10}/results.{pkl,json,npy}
"""
import os
import sys
import zlib
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
from torch_geometric.nn import GCNConv
from torch_geometric.data import HeteroData
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances

from sggnn.models import AdaptiveAggGCN
from sggnn.features import compute_features
from sggnn.data import get_data_dict, create_masks, seed_everything, load_cached_graphs, load_dataset, sim_masks
from sggnn.train import train_model
from sggnn.paths import results_dir, to_json_safe

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

GRAPH_CACHE = os.environ.get('GRAPH_CACHE', str(ROOT / 'node_embeddings' / 'embedding_graphs.npz'))

hid_dim, dropout, n_layers, node_name = 32, 0.5, 1, 'web'
nonlin, last_act = nn.Tanh(), nn.Softmax(dim=1)
lr, wd = 5e-3, 5e-4
epochs = int(os.environ.get('SENS_EPOCHS', 2000))
patience = int(os.environ.get('SENS_PATIENCE', 300))
n_sims = int(os.environ.get('SENS_NSIMS', 10))
datasets = os.environ.get('SENS_DATASETS', 'Texas,Squirrel,Cora,Actor').split(',')
n_datasets = len(datasets)
PARTS = os.environ.get('SENSITIVITY_PARTS', 'k,eps,r').split(',')
# Table VIII/IX now use random splits by default (matching every other table);
# pass SENS_SPLIT=public to use each dataset's own masks instead (geom-gcn
# splits for Texas/Squirrel/Actor, the single public Planetoid split for
# Cora). Nothing in the paper depends on which one is used, so this stays a
# flag rather than a hardcoded choice, consistent with sggnn.data.get_data_dict.
SPLIT = os.environ.get('SENS_SPLIT', 'random')
assert SPLIT in ('random', 'public'), f"Unknown SENS_SPLIT: {SPLIT}"


def get_adjacency_matrix(dataset_name):
    data = load_dataset(dataset_name)[0]
    N = data.x.shape[0]
    A = np.zeros((N, N))
    A[data.edge_index[0].numpy(), data.edge_index[1].numpy()] = 1.0
    return A, data


def _masks(N, n_sims_masks, pyg_data_for_public):
    if SPLIT == 'public' and pyg_data_for_public.train_mask is not None:
        return (pyg_data_for_public.train_mask, pyg_data_for_public.val_mask, pyg_data_for_public.test_mask)
    return create_masks(N, n_sims_masks, 0.8, 0.1)


def _train_and_eval(model, data_pyg, tmask, vmask, tsmask):
    _, _, _, _, val_accs, test_accs = train_model(
        model, data_pyg.x_dict, data_pyg.edge_index_dict, data_pyg[node_name].y,
        tmask, vmask, tsmask, {}, None, lr, wd, epochs, patience, verb=False)
    return test_accs[int(np.argmax(val_accs))]


def _save(experiment_id, payload):
    out = results_dir(experiment_id)
    with open(out / 'results.pkl', 'wb') as f:
        pickle.dump(payload, f)
    with open(out / 'results.json', 'w') as f:
        json.dump(to_json_safe(payload), f, indent=2)
    logging.info(f"Saved {experiment_id} to {out}")


# ── k sweep (Table VIII) ──────────────────────────────────────────────────────
def build_knn_graph(features, k):
    A_knn = kneighbors_graph(features, k, mode='connectivity', include_self=False).toarray()
    return np.maximum(A_knn, A_knn.T)


_FEATURE_CACHE = {}


def _get_cached_features(dname):
    if dname not in _FEATURE_CACHE:
        A, pyg_data = get_adjacency_matrix(dname)
        F_role = compute_features(A, ftype='role')
        F_global = compute_features(A, ftype='global')
        _FEATURE_CACHE[dname] = (A, pyg_data, F_role, F_global)
    return _FEATURE_CACHE[dname]


def _base_hetero_data(A, pyg_data, n_sims_masks):
    """HeteroData with the node attributes, splits and Original graph; callers add their views."""
    data_pyg = HeteroData()
    data_pyg[node_name].x = pyg_data.x
    N = pyg_data.x.size(0)
    data_pyg[node_name].N = N
    data_pyg[node_name].num_feats = pyg_data.x.size(1)
    data_pyg[node_name].y = pyg_data.y
    data_pyg[node_name].num_classes = int(pyg_data.y.max().item()) + 1

    tm, vm, tsm = _masks(N, n_sims_masks, pyg_data)
    data_pyg[node_name].train_mask, data_pyg[node_name].val_mask, data_pyg[node_name].test_mask = tm, vm, tsm

    data_pyg[node_name, 'Original', node_name].edge_index = torch.tensor(np.array(np.nonzero(A)), dtype=torch.long)
    return data_pyg, N


def make_hetero_data_with_knn(dataset_name, k, n_sims_masks):
    A, pyg_data, F_role, F_global = _get_cached_features(dataset_name)
    data_pyg, _ = _base_hetero_data(A, pyg_data, n_sims_masks)
    ei_knn_role = torch.tensor(np.array(np.nonzero(build_knn_graph(F_role, k))), dtype=torch.long)
    data_pyg[node_name, f'KNN-Role-{k}', node_name].edge_index = ei_knn_role
    ei_knn_global = torch.tensor(np.array(np.nonzero(build_knn_graph(F_global, k))), dtype=torch.long)
    data_pyg[node_name, f'KNN-Global-{k}', node_name].edge_index = ei_knn_global
    return data_pyg, ['Original', f'KNN-Role-{k}', f'KNN-Global-{k}']


def run_k_sweep():
    k_values = [int(x) for x in os.environ.get('SENS_K_VALUES', '1,3,5,7,10').split(',')]
    results = np.full((n_datasets, n_sims, len(k_values)), np.nan, dtype=np.float32)
    logging.info("\n===== k sweep (Table VIII) =====")
    for ki, k in enumerate(k_values):
        for d, dname in enumerate(datasets):
            logging.info(f"  k={k} dataset {d+1}/{n_datasets}: {dname}")
            try:
                data_pyg, graph_keys = make_hetero_data_with_knn(dname, k, n_sims)
                data_pyg = data_pyg.to(device)
                in_dim, num_classes = data_pyg[node_name].x.size(1), data_pyg[node_name].num_classes
                for s in range(n_sims):
                    tmask, vmask, tsmask = sim_masks(data_pyg[node_name], s)
                    model = AdaptiveAggGCN(in_dim, hid_dim, num_classes, n_layers, dropout, nonlin,
                                            last_act, graph_keys, GCNConv, GCNConv, -1).to(device)
                    results[d, s, ki] = _train_and_eval(model, data_pyg, tmask, vmask, tsmask)
            except torch.cuda.OutOfMemoryError:
                logging.warning(f"CUDA OOM: {dname}, k={k}")
                torch.cuda.empty_cache()
            except Exception:
                logging.exception(f"Error: {dname}, k={k}")
    _save('table8', {'results': results, 'k_values': k_values, 'datasets': datasets,
                      'metadata': {'n_sims': n_sims, 'timestamp': datetime.datetime.now().isoformat()}})
    return results


# ── eps sweep (Table IX) ──────────────────────────────────────────────────────
def build_eps_graph(features, target_avg_degree):
    """Top-K closest pairs by rank (stable sort), not a distance threshold --
    see the module history in KNNHeterophilic for why thresholding degenerates
    under the heavy exact ties in role-based features."""
    dists = pairwise_distances(features)
    N = dists.shape[0]
    triu_idx = np.triu_indices(N, k=1)
    dists_triu = dists[triu_idx]
    target_edges = min(max(int(target_avg_degree * N / 2), 1), len(dists_triu))
    selected = np.argsort(dists_triu, kind='stable')[:target_edges]
    rows, cols = triu_idx[0][selected], triu_idx[1][selected]
    A_eps = np.zeros((N, N), dtype=int)
    A_eps[rows, cols] = 1
    A_eps[cols, rows] = 1
    return A_eps


def make_hetero_data_with_eps(dname, target_deg, n_sims_masks):
    A, pyg_data, F_role, F_global = _get_cached_features(dname)
    data_pyg, N = _base_hetero_data(A, pyg_data, n_sims_masks)
    A_eps_role = build_eps_graph(F_role, target_deg)
    ei_eps_role = torch.tensor(np.array(np.nonzero(A_eps_role)), dtype=torch.long)
    data_pyg[node_name, f'EPS-Role-{target_deg}', node_name].edge_index = ei_eps_role
    A_eps_global = build_eps_graph(F_global, target_deg)
    ei_eps_global = torch.tensor(np.array(np.nonzero(A_eps_global)), dtype=torch.long)
    data_pyg[node_name, f'EPS-Global-{target_deg}', node_name].edge_index = ei_eps_global
    avg_degs = (ei_eps_role.shape[1] / N, ei_eps_global.shape[1] / N)
    return data_pyg, ['Original', f'EPS-Role-{target_deg}', f'EPS-Global-{target_deg}'], avg_degs


def run_eps_sweep():
    eps_target_degs = [int(x) for x in os.environ.get('SENS_EPS_DEGS', '1,3,5,7,10').split(',')]
    results = np.full((n_datasets, n_sims, len(eps_target_degs)), np.nan, dtype=np.float32)
    avg_degrees = {}
    logging.info("\n===== eps sweep (Table IX) =====")
    for ei, p in enumerate(eps_target_degs):
        for d, dname in enumerate(datasets):
            logging.info(f"  deg={p} dataset {d+1}/{n_datasets}: {dname}")
            try:
                data_pyg, graph_keys, avg_degs = make_hetero_data_with_eps(dname, p, n_sims)
                data_pyg = data_pyg.to(device)
                avg_degrees.setdefault(dname, {})[p] = avg_degs
                in_dim, num_classes = data_pyg[node_name].x.size(1), data_pyg[node_name].num_classes
                for s in range(n_sims):
                    tmask, vmask, tsmask = sim_masks(data_pyg[node_name], s)
                    model = AdaptiveAggGCN(in_dim, hid_dim, num_classes, n_layers, dropout, nonlin,
                                            last_act, graph_keys, GCNConv, GCNConv, -1).to(device)
                    results[d, s, ei] = _train_and_eval(model, data_pyg, tmask, vmask, tsmask)
            except torch.cuda.OutOfMemoryError:
                logging.warning(f"CUDA OOM: {dname}, deg={p}")
                torch.cuda.empty_cache()
            except Exception:
                logging.exception(f"Error: {dname}, deg={p}")
    _save('table9', {'results': results, 'eps_target_degs': eps_target_degs, 'avg_degrees': avg_degrees,
                      'datasets': datasets,
                      'metadata': {'n_sims': n_sims, 'timestamp': datetime.datetime.now().isoformat()}})
    return results


# ── R sweep (Table X) ─────────────────────────────────────────────────────────
def stable_seed(R, draw, dataset_name):
    """zlib.crc32, not Python's salted hash(), so subsets are reproducible
    across processes/runs."""
    return 10_000 * R + 100 * draw + zlib.crc32(dataset_name.encode()) % 97


def run_r_sweep():
    all_graphs_cache = load_cached_graphs(GRAPH_CACHE)
    available_keys = [k for k in all_graphs_cache[datasets[0]].keys() if k != 'EPS-GraphWave']
    non_original_keys = [k for k in available_keys if k != 'Original']

    R_values = [int(x) for x in os.environ.get('SENS_R_VALUES', '2,3,4,6,8').split(',')]
    N_DRAWS = int(os.environ.get('SENS_N_DRAWS', 5))
    n_sims_per_draw = int(os.environ.get('SENS_NSIMS_PER_DRAW', 4))
    n_obs_per_R = N_DRAWS * n_sims_per_draw

    reload_from = os.environ.get('SENS_RELOAD_SUBSETS_FROM', '')
    reloaded = None
    if reload_from:
        with open(reload_from, 'rb') as f:
            reloaded = pickle.load(f)['chosen_subsets']

    results = np.full((n_datasets, n_obs_per_R, len(R_values)), np.nan, dtype=np.float32)
    chosen_subsets = {}
    logging.info("\n===== R sweep (Table X) =====")

    for Ri, R in enumerate(R_values):
        for d, dname in enumerate(datasets):
            logging.info(f"  R={R} dataset {d+1}/{n_datasets}: {dname}")
            chosen_subsets.setdefault(dname, {})[R] = []
            try:
                obs_idx = 0
                for draw in range(N_DRAWS):
                    if reloaded is not None:
                        gkeys = list(reloaded[dname][R][draw])
                    else:
                        rng = np.random.RandomState(seed=stable_seed(R, draw, dname))
                        gkeys = ['Original'] + list(rng.choice(non_original_keys, size=R - 1, replace=False))
                    chosen_subsets[dname][R].append(gkeys)

                    subset_graphs = {dname: {k: v for k, v in all_graphs_cache[dname].items() if k in gkeys}}
                    data_pyg = get_data_dict([dname], subset_graphs, node_name, n_sims_per_draw, 0.8, 0.1)[dname].to(device)
                    in_dim, num_classes = data_pyg[node_name].x.size(1), data_pyg[node_name].num_classes

                    for s in range(n_sims_per_draw):
                        tmask, vmask, tsmask = sim_masks(data_pyg[node_name], s)
                        model = AdaptiveAggGCN(in_dim, hid_dim, num_classes, n_layers, dropout, nonlin,
                                                last_act, gkeys, GCNConv, GCNConv, -1).to(device)
                        results[d, obs_idx, Ri] = _train_and_eval(model, data_pyg, tmask, vmask, tsmask)
                        obs_idx += 1
            except torch.cuda.OutOfMemoryError:
                logging.warning(f"CUDA OOM: {dname}, R={R}")
                torch.cuda.empty_cache()
            except Exception:
                logging.exception(f"Error: {dname}, R={R}")
    _save('table10', {'results': results, 'R_values': R_values, 'N_DRAWS': N_DRAWS,
                       'n_sims_per_draw': n_sims_per_draw, 'chosen_subsets': chosen_subsets,
                       'datasets': datasets,
                       'metadata': {'timestamp': datetime.datetime.now().isoformat(),
                                    'r_sweep_subset_seed': 'zlib.crc32(dataset_name) % 97'}})
    return results


if __name__ == '__main__':
    if 'k' in PARTS:
        run_k_sweep()
    if 'eps' in PARTS:
        run_eps_sweep()
    if 'r' in PARTS:
        run_r_sweep()
