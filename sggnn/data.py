"""Dataset loading, train/val/test splits (random or public), the candidate
graph cache reader, and parameter-matching helpers for capacity-fair
baselines."""
import random
import logging

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch_geometric.data import HeteroData
from torch_geometric.datasets import WebKB, Actor, WikipediaNetwork, Airports, Planetoid, HeterophilousGraphDataset

from .features import compute_role_features_sparse, knn_edge_index


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_dataset(name):
    """Return the torch_geometric dataset object for one of the paper's datasets."""
    if name in ['Texas', 'Wisconsin', 'Cornell']:
        return WebKB('~/.datapyg', name)
    if name == 'Actor':
        return Actor('~/.datapyg')
    if name in ['Chameleon', 'Squirrel']:
        return WikipediaNetwork('~/.datapyg', name.lower())
    if name in ['Cora', 'CiteSeer']:
        return Planetoid('~/.datapyg', name)
    if name in ['USA', 'Brazil', 'Europe']:
        return Airports('~/.datapyg', name)
    raise ValueError(f"Unknown dataset: {name}")


def get_data_dict(datasets, all_graphs, node_name, n_masks=10, N_train=0.8, N_val=0.1, split='random'):
    """Build the per-dataset HeteroData objects used across the experiment scripts.

    `split` controls which train/val/test masks are attached:
      - 'random' (default, matches every published table): always draws
        `n_masks` fresh random N_train/N_val/rest splits via `create_masks`,
        regardless of whether the underlying torch_geometric dataset ships
        public splits.
      - 'public': uses the dataset's own masks (`data[0].train_mask`, etc.)
        when present -- WebKB, Actor and WikipediaNetwork ship 10 geom-gcn
        splits, Planetoid ships a single split -- and falls back to `n_masks`
        random splits (with a warning) for datasets that have none (e.g. the
        Airports datasets: USA, Europe, Brazil).
    """
    assert split in ('random', 'public'), f"Unknown split mode: {split}"
    data_dict = {}

    for dataset_name in datasets:

        logging.info(f"Reading and processing data for {dataset_name}")
        data = load_dataset(dataset_name)

        graphs_dataset = all_graphs[dataset_name]
        graphs = list(graphs_dataset.keys())

        data_pyg = HeteroData()

        data_pyg[node_name].x = data[0].x

        data_pyg[node_name].N = data[0].x.size(0)
        data_pyg[node_name].num_feats = data[0].x.size(1)
        data_pyg[node_name].num_classes = data.num_classes

        data_pyg[node_name].y = data[0].y

        has_public_split = (getattr(data[0], 'train_mask', None) is not None)
        if split == 'public' and has_public_split:
            data_pyg[node_name].train_mask = data[0].train_mask
            data_pyg[node_name].val_mask = data[0].val_mask
            data_pyg[node_name].test_mask = data[0].test_mask
        else:
            if split == 'public' and not has_public_split:
                logging.warning(f"  {dataset_name} has no public split; falling back to "
                                 f"{n_masks} random {N_train:.0%}/{N_val:.0%} splits")
            train_mask, val_mask, test_mask = create_masks(data_pyg[node_name].N, n_masks, N_train, N_val)
            data_pyg[node_name].train_mask = train_mask
            data_pyg[node_name].val_mask = val_mask
            data_pyg[node_name].test_mask = test_mask

        for gname in graphs:
            if dataset_name == "Chameleon" and gname == "EPS-GraphWave":
                # Skipping due to out of memory error
                continue

            graph = graphs_dataset[gname]
            if graph.ndim == 0:
                graph = graph.item()
            if type(graph) == csr_matrix:
                graph = graph.toarray()
            edge_idx = torch.nonzero(torch.from_numpy(graph)).t().contiguous()
            data_pyg[node_name, gname, node_name].edge_index = edge_idx

        data_pyg[node_name, 'EgoFeats', node_name].edge_index = torch.arange(data_pyg[node_name].N).repeat(2, 1)

        data_dict[dataset_name] = data_pyg

    return data_dict


def create_masks(N, n_masks, train_frac=0.8, val_frac=0.1):
    """Draw `n_masks` random train/val/test splits from the global torch RNG."""
    num_train = int(train_frac * N)
    num_val = int(val_frac * N)

    train_mask = torch.zeros(N, n_masks, dtype=torch.bool)
    val_mask = torch.zeros(N, n_masks, dtype=torch.bool)
    test_mask = torch.zeros(N, n_masks, dtype=torch.bool)

    for col in range(n_masks):
        indices = torch.randperm(N)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train + num_val]
        test_indices = indices[num_train + num_val:]

        train_mask[train_indices, col] = True
        val_mask[val_indices, col] = True
        test_mask[test_indices, col] = True

    return train_mask, val_mask, test_mask


def load_roman_empire(node_name, K_knn=10):
    """Roman-Empire as a HeteroData with its 10 public splits and four views:
    Original, KNN-Feat, KNN-Role (k=`K_knn`, sparse role features) and EgoFeats."""
    logging.info("Loading Roman-Empire ...")
    g = HeterophilousGraphDataset(root='~/.datapyg', name='Roman-empire')[0]
    N = g.num_nodes
    num_classes = int(g.y.max().item()) + 1
    logging.info(f"Roman-Empire: {N} nodes, {g.edge_index.shape[1]} edges, {num_classes} classes")

    logging.info("Computing kNN-Feat graph ...")
    ei_feat = knn_edge_index(g.x.numpy(), K_knn, N)
    logging.info("Computing role features + kNN-Role graph ...")
    role_feats = compute_role_features_sparse(g.edge_index, N)
    ei_role = knn_edge_index(role_feats, K_knn, N)

    data_pyg = HeteroData()
    data_pyg[node_name].x = g.x
    data_pyg[node_name].y = g.y
    data_pyg[node_name].N = N
    data_pyg[node_name].num_classes = num_classes
    data_pyg[node_name].train_mask = g.train_mask
    data_pyg[node_name].val_mask = g.val_mask
    data_pyg[node_name].test_mask = g.test_mask
    data_pyg[node_name, 'Original', node_name].edge_index = g.edge_index
    data_pyg[node_name, 'KNN-Feat', node_name].edge_index = ei_feat
    data_pyg[node_name, 'KNN-Role', node_name].edge_index = ei_role
    data_pyg[node_name, 'EgoFeats', node_name].edge_index = torch.arange(N).unsqueeze(0).repeat(2, 1)
    return data_pyg, ['Original', 'KNN-Feat', 'KNN-Role', 'EgoFeats']


def sim_masks(store, s):
    """Train/val/test masks for simulation `s` of a node store, cycling over its
    split columns when it holds several."""
    tm, vm, tsm = store.train_mask, store.val_mask, store.test_mask
    if tm.ndim > 1:
        idx = s % tm.shape[1]
        return tm[:, idx], vm[:, idx], tsm[:, idx]
    return tm, vm, tsm


def load_cached_graphs(path):
    """Read the graph cache written by `sggnn.embeddings.build_cache`.

    Keys are `<dataset>_<graph>`; returns {dataset: {graph: array}}.
    """
    data_np = np.load(str(path), allow_pickle=True)
    all_graphs = {}
    for filename in data_np.files:
        dataset = filename.split('_')[0]
        key = filename.split('_')[1]
        all_graphs.setdefault(dataset, {})[key] = data_np[filename]
    return all_graphs


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def find_matched_hid(build_fn, target, lo=32, hi=1024, hard_cap=32768):
    """Binary-search the smallest hid_dim >= lo such that build_fn(hid) has
    >= target params. Assumes monotonic non-decreasing param count in hid.

    Every probe instantiates a model and so consumes torch RNG state."""
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

