"""
Sensitivity/ablation sweeps over the three graph-construction hyperparameters
discussed in the paper: the number of neighbors k, the epsilon-ball target
average degree, and the number of candidate graphs R (Tables VIII, IX, X).

Part A -- k sweep: recompute kNN graphs on-the-fly for k in [1, 3, 5, 7, 10].

Part B -- epsilon sweep: builds epsilon-ball graphs from role- and
global-based structural features via rank-based thresholding to a target
average degree (robust to the heavy ties role-based features produce at
short distances -- raw percentile thresholding is degenerate here, since
>5% of pairwise distances are exactly zero for several datasets). Target
average degrees in [1,3,5,7,10] mirror the k sweep's k values directly for
comparability.

Part C -- R sweep: for each R, draws N_DRAWS independent random subsets of
size R-1 from the full 14-view non-Original candidate pool (Original is
always included), and averages over both random draws and data splits. This
directly operationalizes Propositions 1-2's claim about R independent/random
graph views, rather than a fixed, curated, cumulative sequence of graph
sets (which would confound "number of views" with "which specific views are
added"). REQUIRES running `python precompute_candidate_graphs.py` first (see
that script) to build the candidate pool -- Part C raises a clear error if
its output is missing.

All three parts use the same subset of datasets for speed: ['Texas', 'Squirrel', 'Cora', 'Actor'].
Parts A/B compute role/global structural-feature graphs on-the-fly, cached in
memory per dataset (see `_FEATURE_CACHE` below) so they are not recomputed at
every sweep point; Part C instead loads its (more expensive, embedding-based)
candidate pool from disk.

Output:
  results/ablations/sensitivity/sens_k_results.npy     -- [n_datasets, n_sims, n_k_values]      (Table VIII)
  results/ablations/sensitivity/sens_eps_results.npy    -- [n_datasets, n_sims, n_eps_values]     (Table IX)
  results/ablations/sensitivity/sens_R_results.npy      -- [n_datasets, n_draws*n_sims_per_draw, n_R_values]  (Table X)
  results/ablations/sensitivity/sens_results.pkl        -- combined dict with metadata
"""

import sys
import os
import logging
import pickle
import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import HeteroData
from torch_geometric.datasets import WebKB, Actor, WikipediaNetwork, Airports, Planetoid
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances

from arch import AdaptiveAggGCN
from utils import compute_features, create_masks, seed_everything
from train import train_model

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

seed_everything(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

OUT_DIR = ROOT / 'results' / 'ablations' / 'sensitivity'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── hyperparams ───────────────────────────────────────────────────────────────
hid_dim   = 32
dropout   = 0.5
nonlin    = nn.Tanh()
last_act  = nn.Softmax(dim=1)
lr        = 5e-3
wd        = 5e-4
epochs    = int(os.environ.get('SENS_EPOCHS', 2000))
patience  = int(os.environ.get('SENS_PATIENCE', 300))
n_sims    = int(os.environ.get('SENS_NSIMS', 10))
n_layers  = 1
node_name = 'web'

datasets = os.environ.get('SENS_DATASETS', 'Texas,Squirrel,Cora,Actor').split(',')
n_datasets = len(datasets)

# ── helper: load raw adjacency matrix ────────────────────────────────────────
def get_adjacency_matrix(dataset_name):
    if dataset_name in ['Texas', 'Wisconsin', 'Cornell']:
        data = WebKB('~/.datapyg', dataset_name)[0]
    elif dataset_name == 'Actor':
        data = Actor('~/.datapyg')[0]
    elif dataset_name in ['Chameleon', 'Squirrel']:
        data = WikipediaNetwork('~/.datapyg', dataset_name.lower())[0]
    elif dataset_name in ['Cora', 'CiteSeer']:
        data = Planetoid('~/.datapyg', dataset_name)[0]
    elif dataset_name in ['USA', 'Brazil', 'Europe']:
        data = Airports('~/.datapyg', dataset_name)[0]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    N = data.x.shape[0]
    A = np.zeros((N, N))
    A[data.edge_index[0].numpy(), data.edge_index[1].numpy()] = 1.0
    return A, data


_FEATURE_CACHE = {}  # dataset_name -> (A, pyg_data, F_role, F_global)
                     # role/global structural features depend only on the dataset's
                     # adjacency, not on k or the target degree -- computed once per
                     # dataset and reused across every sweep point.


def _get_cached_features(dataset_name):
    if dataset_name not in _FEATURE_CACHE:
        A, pyg_data = get_adjacency_matrix(dataset_name)
        logging.info(f"    Computing role features for {dataset_name}...")
        F_role = compute_features(A, ftype='role')
        logging.info(f"    Computing global features for {dataset_name}...")
        F_global = compute_features(A, ftype='global')
        _FEATURE_CACHE[dataset_name] = (A, pyg_data, F_role, F_global)
    return _FEATURE_CACHE[dataset_name]


def _base_hetero_data(dataset_name, n_sims_masks):
    """Build a HeteroData object with just node features/labels/masks and the
    original graph filled in; caller adds the k-NN / epsilon-ball views."""
    A, pyg_data, _, _ = _get_cached_features(dataset_name)
    data_pyg = HeteroData()
    data_pyg[node_name].x = pyg_data.x
    N = pyg_data.x.size(0)
    data_pyg[node_name].N = N
    data_pyg[node_name].num_feats = pyg_data.x.size(1)
    data_pyg[node_name].y = pyg_data.y
    data_pyg[node_name].num_classes = int(pyg_data.y.max().item()) + 1

    if pyg_data.train_mask is not None and pyg_data.train_mask.ndim >= 1:
        data_pyg[node_name].train_mask = pyg_data.train_mask
        data_pyg[node_name].val_mask   = pyg_data.val_mask
        data_pyg[node_name].test_mask  = pyg_data.test_mask
    else:
        tm, vm, tsm = create_masks(N, n_sims_masks, 0.8, 0.1)
        data_pyg[node_name].train_mask = tm
        data_pyg[node_name].val_mask   = vm
        data_pyg[node_name].test_mask  = tsm

    edge_idx_orig = torch.tensor(np.array(np.nonzero(A)), dtype=torch.long)
    data_pyg[node_name, 'Original', node_name].edge_index = edge_idx_orig
    return data_pyg


def build_knn_graph(features, k):
    """Build a symmetric kNN graph from a feature matrix. Returns dense adjacency."""
    A_knn_sp = kneighbors_graph(features, k, mode='connectivity', include_self=False)
    A_knn = A_knn_sp.toarray()
    A_knn = np.maximum(A_knn, A_knn.T)  # symmetrize
    return A_knn


def build_eps_graph(features, target_avg_degree):
    """Epsilon-ball graph selecting exactly the top-K closest pairs by rank,
    where K is chosen to hit a target average degree.

    Role-based structural features often have heavy ties at short distances
    (>5% of pairwise distances exactly zero, from structurally-identical
    nodes such as degree-1 leaves with matching egonet stats). A threshold
    comparison (dists <= th) is degenerate under such ties: if th lands
    inside the tied block, ALL tied pairs are included regardless of how
    many were actually requested, making average degree insensitive to the
    target. Selecting the top-K ranked pairs directly (stable sort, so ties
    are broken deterministically) guarantees the requested average degree
    exactly, regardless of tie structure.
    """
    dists = pairwise_distances(features)
    N = dists.shape[0]
    triu_idx = np.triu_indices(N, k=1)
    dists_triu = dists[triu_idx]
    target_edges = int(target_avg_degree * N / 2)
    target_edges = min(max(target_edges, 1), len(dists_triu))
    order = np.argsort(dists_triu, kind='stable')
    selected = order[:target_edges]
    rows, cols = triu_idx[0][selected], triu_idx[1][selected]
    A_eps = np.zeros((N, N), dtype=int)
    A_eps[rows, cols] = 1
    A_eps[cols, rows] = 1
    return A_eps


def make_hetero_data_with_knn(dataset_name, k, n_sims_masks):
    """HeteroData with Original + KNN-Role-k + KNN-Global-k graphs."""
    data_pyg = _base_hetero_data(dataset_name, n_sims_masks)
    _, _, F_role, F_global = _get_cached_features(dataset_name)

    A_knn_role = build_knn_graph(F_role, k)
    ei_knn_role = torch.tensor(np.array(np.nonzero(A_knn_role)), dtype=torch.long)
    data_pyg[node_name, f'KNN-Role-{k}', node_name].edge_index = ei_knn_role

    A_knn_global = build_knn_graph(F_global, k)
    ei_knn_global = torch.tensor(np.array(np.nonzero(A_knn_global)), dtype=torch.long)
    data_pyg[node_name, f'KNN-Global-{k}', node_name].edge_index = ei_knn_global

    return data_pyg, ['Original', f'KNN-Role-{k}', f'KNN-Global-{k}']


def make_hetero_data_with_eps(dataset_name, target_deg, n_sims_masks):
    """HeteroData with Original + EPS-Role-d + EPS-Global-d graphs, where d is
    the target average degree of the constructed epsilon-ball graphs."""
    data_pyg = _base_hetero_data(dataset_name, n_sims_masks)
    _, _, F_role, F_global = _get_cached_features(dataset_name)
    N = data_pyg[node_name].N

    A_eps_role = build_eps_graph(F_role, target_deg)
    ei_eps_role = torch.tensor(np.array(np.nonzero(A_eps_role)), dtype=torch.long)
    data_pyg[node_name, f'EPS-Role-{target_deg}', node_name].edge_index = ei_eps_role

    A_eps_global = build_eps_graph(F_global, target_deg)
    ei_eps_global = torch.tensor(np.array(np.nonzero(A_eps_global)), dtype=torch.long)
    data_pyg[node_name, f'EPS-Global-{target_deg}', node_name].edge_index = ei_eps_global

    avg_deg_role = ei_eps_role.shape[1] / N
    avg_deg_global = ei_eps_global.shape[1] / N
    return data_pyg, ['Original', f'EPS-Role-{target_deg}', f'EPS-Global-{target_deg}'], (avg_deg_role, avg_deg_global)


def _run_sims(model_graph_keys, x_dict, edge_index_dict, y, train_mask, val_mask, test_mask, in_dim, num_classes, n_reps):
    """Train n_reps models on the given (possibly split-varying) masks and
    return the array of best-validation-epoch test accuracies."""
    accs = np.zeros(n_reps, dtype=np.float32)
    for s in range(n_reps):
        if train_mask.ndim > 1:
            idx = s % train_mask.shape[1]
            tmask, vmask, tsmask = train_mask[:, idx], val_mask[:, idx], test_mask[:, idx]
        else:
            tmask, vmask, tsmask = train_mask, val_mask, test_mask

        model = AdaptiveAggGCN(
            in_dim, hid_dim, num_classes, n_layers,
            dropout, nonlin, last_act,
            model_graph_keys, GCNConv, GCNConv, -1
        ).to(device)

        model, _, _, _, val_accs, test_accs = train_model(
            model, x_dict, edge_index_dict, y,
            tmask, vmask, tsmask, {},
            None, lr, wd, epochs, patience, verb=False
        )
        best_epoch = int(np.argmax(val_accs))
        accs[s] = test_accs[best_epoch]
    return accs


# ── Part A: k sweep ───────────────────────────────────────────────────────────
k_values = [int(x) for x in os.environ.get('SENS_K_VALUES', '1,3,5,7,10').split(',')]
n_k = len(k_values)

sens_k_results = np.zeros((n_datasets, n_sims, n_k), dtype=np.float32)

logging.info("\n===== Part A: k sweep =====")

for ki, k in enumerate(k_values):
    logging.info(f"\n--- k={k} ({ki+1}/{n_k}) ---")
    for d, dataset_name in enumerate(datasets):
        logging.info(f"  Dataset {d+1}/{n_datasets}: {dataset_name}")
        try:
            data_pyg, graph_keys = make_hetero_data_with_knn(dataset_name, k, n_sims)
            data_pyg = data_pyg.to(device)
            in_dim = data_pyg[node_name].x.size(1)
            num_classes = data_pyg[node_name].num_classes

            sens_k_results[d, :, ki] = _run_sims(
                graph_keys, data_pyg.x_dict, data_pyg.edge_index_dict, data_pyg[node_name].y,
                data_pyg[node_name].train_mask, data_pyg[node_name].val_mask, data_pyg[node_name].test_mask,
                in_dim, num_classes, n_sims
            )
            logging.info(f"    {dataset_name} k={k}: {sens_k_results[d, :, ki].mean():.4f}")
        except torch.cuda.OutOfMemoryError:
            logging.warning(f"    CUDA OOM: {dataset_name}, k={k} -- skipping")
            torch.cuda.empty_cache()
        except Exception as exc:
            logging.warning(f"    Error: {dataset_name}, k={k}: {exc}")

    np.save(str(OUT_DIR / 'sens_k_results.npy'), sens_k_results)
    logging.info(f"  Checkpoint saved after k={k}")

# ── Part B: epsilon sweep ─────────────────────────────────────────────────────
eps_target_degs = [int(x) for x in os.environ.get('SENS_EPS_DEGS', '1,3,5,7,10').split(',')]
n_eps = len(eps_target_degs)

sens_eps_results = np.zeros((n_datasets, n_sims, n_eps), dtype=np.float32)
eps_avg_degrees = {}

logging.info("\n===== Part B: epsilon sweep =====")

for ei, p in enumerate(eps_target_degs):
    logging.info(f"\n--- target_avg_degree={p} ({ei+1}/{n_eps}) ---")
    for d, dataset_name in enumerate(datasets):
        logging.info(f"  Dataset {d+1}/{n_datasets}: {dataset_name}")
        try:
            data_pyg, graph_keys, avg_degs = make_hetero_data_with_eps(dataset_name, p, n_sims)
            data_pyg = data_pyg.to(device)
            eps_avg_degrees.setdefault(dataset_name, {})[p] = avg_degs
            in_dim = data_pyg[node_name].x.size(1)
            num_classes = data_pyg[node_name].num_classes

            sens_eps_results[d, :, ei] = _run_sims(
                graph_keys, data_pyg.x_dict, data_pyg.edge_index_dict, data_pyg[node_name].y,
                data_pyg[node_name].train_mask, data_pyg[node_name].val_mask, data_pyg[node_name].test_mask,
                in_dim, num_classes, n_sims
            )
            logging.info(f"    {dataset_name} p={p}: {sens_eps_results[d, :, ei].mean():.4f} "
                         f"(avg deg role={avg_degs[0]:.2f}, global={avg_degs[1]:.2f})")
        except torch.cuda.OutOfMemoryError:
            logging.warning(f"    CUDA OOM: {dataset_name}, p={p} -- skipping")
            torch.cuda.empty_cache()
        except Exception as exc:
            logging.warning(f"    Error: {dataset_name}, p={p}: {exc}")

    np.save(str(OUT_DIR / 'sens_eps_results.npy'), sens_eps_results)
    logging.info(f"  Checkpoint saved after p={p}")

# ── Part C: R sweep (random subset sampling) ──────────────────────────────────
# For each R, draw N_DRAWS independent random subsets of size R-1 from the full
# 14-view non-Original candidate pool (Original is always included), and
# evaluate each subset over n_sims_per_draw data splits. The candidate pool
# (kNN/eps-ball graphs from 7 feature/embedding types: raw features, our
# role/global structural attributes, and DeepWalk/Node2Vec/Struc2Vec/GraphWave
# embeddings) must be precomputed first by running precompute_candidate_graphs.py,
# since the embedding methods are too expensive to recompute at every sweep point.
CANDIDATE_GRAPHS_PATH = ROOT / 'results' / 'candidate_graphs' / 'candidate_graphs.npz'

R_values        = [int(x) for x in os.environ.get('SENS_R_VALUES', '2,3,4,6,8').split(',')]
N_DRAWS         = int(os.environ.get('SENS_N_DRAWS', 5))
n_sims_per_draw = int(os.environ.get('SENS_NSIMS_PER_DRAW', 4))
n_R             = len(R_values)
n_obs_per_R     = N_DRAWS * n_sims_per_draw

sens_R_results = np.zeros((n_datasets, n_obs_per_R, n_R), dtype=np.float32)
R_chosen_subsets = {}  # {dataset: {R: [[graphs drawn per draw], ...]}}

logging.info("\n===== Part C: R sweep (random subset sampling) =====")

if not CANDIDATE_GRAPHS_PATH.exists():
    raise FileNotFoundError(
        f"{CANDIDATE_GRAPHS_PATH} not found. Run `python precompute_candidate_graphs.py` "
        "from the ablations/ directory first to build the 14-view candidate pool."
    )
_candidate_npz = np.load(str(CANDIDATE_GRAPHS_PATH), allow_pickle=True)
_candidate_cache = {}  # dataset -> {graph_name: dense adjacency}
for _fname in _candidate_npz.files:
    _ds, _key = _fname.split('_', 1)
    _candidate_cache.setdefault(_ds, {})[_key] = _candidate_npz[_fname]


def _build_candidate_pool(dataset_name, n_sims_masks):
    """Build a HeteroData object from the precomputed 14-view candidate pool
    (+ Original) used to draw random R-subsets from."""
    data_pyg = _base_hetero_data(dataset_name, n_sims_masks)
    graph_keys = ['Original']
    for key, A_g in _candidate_cache[dataset_name].items():
        if key == 'Original':
            continue
        ei = torch.tensor(np.array(np.nonzero(A_g)), dtype=torch.long)
        data_pyg[node_name, key, node_name].edge_index = ei
        graph_keys.append(key)
    return data_pyg, graph_keys


for Ri, R in enumerate(R_values):
    logging.info(f"\n--- R={R} ({Ri+1}/{n_R}) ---")

    for d, dataset_name in enumerate(datasets):
        logging.info(f"  Dataset {d+1}/{n_datasets}: {dataset_name}")
        R_chosen_subsets.setdefault(dataset_name, {})[R] = []

        try:
            data_pyg, full_pool_keys = _build_candidate_pool(dataset_name, n_sims_per_draw)
            data_pyg = data_pyg.to(device)
            non_original_keys = [k for k in full_pool_keys if k != 'Original']
            in_dim = data_pyg[node_name].x.size(1)
            num_classes = data_pyg[node_name].num_classes

            obs_idx = 0
            for draw in range(N_DRAWS):
                rng = np.random.RandomState(seed=10_000 * R + 100 * draw + hash(dataset_name) % 97)
                chosen_others = list(rng.choice(non_original_keys, size=R - 1, replace=False))
                gkeys = ['Original'] + chosen_others
                R_chosen_subsets[dataset_name][R].append(gkeys)

                ei_dict = {(node_name, gk, node_name): data_pyg[node_name, gk, node_name].edge_index for gk in gkeys}
                accs = _run_sims(
                    gkeys, {node_name: data_pyg[node_name].x}, ei_dict, data_pyg[node_name].y,
                    data_pyg[node_name].train_mask, data_pyg[node_name].val_mask, data_pyg[node_name].test_mask,
                    in_dim, num_classes, n_sims_per_draw
                )
                sens_R_results[d, obs_idx:obs_idx + n_sims_per_draw, Ri] = accs
                obs_idx += n_sims_per_draw
                logging.info(f"    draw {draw+1}/{N_DRAWS}, graphs={gkeys}: {accs.mean():.4f}")

            logging.info(f"    {dataset_name} R={R} overall: {sens_R_results[d, :, Ri].mean():.4f} "
                         f"+/- {sens_R_results[d, :, Ri].std():.4f} "
                         f"(over {N_DRAWS} random subsets x {n_sims_per_draw} splits)")

        except torch.cuda.OutOfMemoryError:
            logging.warning(f"    CUDA OOM: {dataset_name}, R={R} -- skipping")
            torch.cuda.empty_cache()
        except Exception as exc:
            logging.warning(f"    Error: {dataset_name}, R={R}: {exc}")

    np.save(str(OUT_DIR / 'sens_R_results.npy'), sens_R_results)
    logging.info(f"  Checkpoint saved after R={R}")

# ── save combined results ─────────────────────────────────────────────────────
full_results = {
    'sens_k_results':   sens_k_results,
    'k_values':         k_values,
    'sens_eps_results': sens_eps_results,
    'eps_target_degs':  eps_target_degs,
    'eps_avg_degrees':  eps_avg_degrees,
    'sens_R_results':   sens_R_results,
    'R_values':         R_values,
    'N_DRAWS':          N_DRAWS,
    'n_sims_per_draw':  n_sims_per_draw,
    'R_chosen_subsets': R_chosen_subsets,
    'datasets':         datasets,
    'metadata': {
        'seed': 42, 'timestamp': datetime.datetime.now().isoformat(),
        'hid_dim': hid_dim, 'dropout': dropout, 'lr': lr, 'wd': wd,
        'epochs': epochs, 'patience': patience, 'n_sims': n_sims, 'n_layers': n_layers,
    }
}

with open(str(OUT_DIR / 'sens_results.pkl'), 'wb') as f:
    pickle.dump(full_results, f)

logging.info(f"\nDone. Results saved to {OUT_DIR}")
