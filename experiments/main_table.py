"""
Table VI (`tab:metrics_sggnn`): main SG-GNN vs. baselines comparison. Merges
the former exp_main_matched.py (8 baselines + 6 SG-GNN variants, 11 small/
medium datasets), exp_newbaselines_matched.py (+ GRAND/CD-GNN/LG-GNN rows),
exp_largegraph_matched.py (+ Roman-Empire column), and make_table6_merge.py
(merging the three into one table).

SG-GNN keeps hidden dimension 32. Each baseline's capacity is increased until
its parameter count is >= the largest SG-GNN variant's, on that dataset.

Run one part at a time with MAIN_TABLE_PART={main,newbaselines,largegraph,merge}
(default: main). newbaselines needs main's output (MATCHED_SOURCE_PKLS,
defaults to table6_main's results.pkl) for the parameter target; merge needs
all three. main/newbaselines need the candidate-graph cache
(`python main.py build-graphs`); largegraph builds its own (Roman-Empire).

`analysis/table6.py` builds the final LaTeX from the `merge` output.

Output: results/<RESULTS_DIR>/table6_{main,newbaselines,largegraph}/results.{pkl,json}
        results/<RESULTS_DIR>/table6/results.{pkl,json}  (merge)
"""
import sys
import os
import json
import logging
import pickle
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SSGConv

from sggnn.models import (
    AdaptiveAggGCN, HeteroGNN, GCN, FBGNNLayer,
    gfGNN, FAGCN, DirGNN, H2GCN,
)
from sggnn.data import (get_data_dict, seed_everything, load_cached_graphs, load_roman_empire,
                         sim_masks, count_params, find_matched_hid)
from sggnn.train import train_model
from sggnn.baselines import GRANDWrapper, CDGNNWrapper, LGGNNWrapper
from sggnn.paths import results_dir, git_commit, to_json_safe

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

GRAPH_CACHE = os.environ.get('GRAPH_CACHE', str(ROOT / 'node_embeddings' / 'embedding_graphs.npz'))
PART = os.environ.get('MAIN_TABLE_PART', 'main')

# Hyperparameters shared by every part (same as exp_2.py).
n_layers  = 2
hid_dim   = 32          # SG-GNN hidden dim
dropout   = 0.5
nonlin    = nn.Tanh()
last_act_softmax = nn.Softmax(dim=1)
lr        = 5e-3
wd        = 5e-4
epochs    = int(os.environ.get('MATCHED_EPOCHS', 2000))
patience  = int(os.environ.get('MATCHED_PATIENCE', 300))
node_name = 'web'

ALL_DATASETS = ['Texas', 'Wisconsin', 'Cornell', 'Actor', 'Chameleon',
                'Squirrel', 'Cora', 'CiteSeer', 'USA', 'Europe', 'Brazil']
BASELINE_NAMES  = ['GCN', 'GAT', 'gfNN', 'FAGCN', 'DirGNN', 'MixHop', 'SSGC', 'H2GCN']
DIFFUSION_NAMES = ['GRAND', 'CD-GNN', 'LG-GNN']
SGGNN_NAMES     = ['SG-GCN_N', 'SG-GCN', 'SG-GCN_L', 'SG-FBGNN_N', 'SG-FBGNN', 'SG-FBGNN_L']


def sggnn_builders(in_dim, out_dim, graphs, N):
    """Zero-argument constructors of the six SG-GNN variants (hidden dim fixed)."""
    return {
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


def diffusion_builders(in_dim, out_dim):
    """hid -> model constructors for GRAND, CD-GNN and LG-GNN."""
    return {
        'GRAND':  lambda hid: GRANDWrapper(in_dim, hid, out_dim, n_layers, dropout, nonlin, last_act_softmax),
        'CD-GNN': lambda hid: CDGNNWrapper(in_dim, hid, out_dim, n_layers, dropout, nonlin, last_act_softmax),
        'LG-GNN': lambda hid: LGGNNWrapper(in_dim, hid, out_dim, n_layers, dropout, nonlin, last_act_softmax),
    }


def baseline_builders(in_dim, out_dim, with_diffusion=False):
    """hid -> model constructors, in capacity-matching order. Every probe of
    find_matched_hid consumes torch RNG state, so this order is part of the
    experiment: the diffusion baselines go right before FAGCN."""
    def gcn_builder(layer_cls, last_act=nn.Identity(), layer_kwargs=None):
        layer_kwargs = layer_kwargs or {}
        return lambda hid: GCN(in_dim, hid, out_dim, n_layers, nonlin=nonlin,
                                last_act=last_act, dropout=dropout,
                                gcnlayer=layer_cls, gcnlayer_kwargs=layer_kwargs)

    builders = {
        'GCN':    gcn_builder(GCNConv),
        'GAT':    gcn_builder(GATConv),
        'MixHop': gcn_builder(FBGNNLayer, last_act=last_act_softmax),
        'SSGC':   gcn_builder(SSGConv, layer_kwargs={'alpha': 0.05}),
        'gfNN':   lambda hid: gfGNN(in_dim, hid, out_dim, n_layers, dropout, nonlin, nn.Identity()),
        'DirGNN': lambda hid: DirGNN(in_dim, hid, out_dim, n_layers, dropout, nonlin, nn.Identity()),
        'H2GCN':  lambda hid: H2GCN(in_dim, hid, out_dim, n_layers, dropout, nonlin, nn.Identity()),
    }
    if with_diffusion:
        builders.update(diffusion_builders(in_dim, out_dim))
    builders['FAGCN'] = lambda hid: FAGCN(in_dim, hid, out_dim, n_layers, dropout, nonlin, nn.Identity())
    return builders


def match_capacity(builders, target, indent='    '):
    """Smallest hid per baseline whose parameter count reaches `target`."""
    matched_val, achieved_params, matched_hparams = {}, {}, {}
    for bname, builder in builders.items():
        val, achieved = find_matched_hid(builder, target)
        matched_val[bname] = val
        achieved_params[bname] = achieved
        matched_hparams[bname] = {'hid': val, 'achieved_params': achieved, 'target': target}
        shortfall = "" if achieved >= target else "  ** SHORT of target (arch limitation) **"
        logging.info(f"{indent}{bname}: hid={val}, params={achieved}{shortfall}")
    return matched_val, achieved_params, matched_hparams


def build_configs(config_names, builders, sg_builders, matched_val, device, X_hom, ei_orig, X_het, ei_het):
    """Instantiate one model per config, in `config_names` order, as
    (name, model, X, edge_index, forward kwargs)."""
    cfgs = []
    for cname in config_names:
        if cname in sg_builders:
            cfgs.append((cname, sg_builders[cname]().to(device), X_het, ei_het, {}))
        else:
            model = builders[cname](matched_val[cname]).to(device)
            fw_kw = {'x_0': X_hom.clone()} if cname == 'FAGCN' else {}
            cfgs.append((cname, model, X_hom, ei_orig, fw_kw))
    return cfgs


def best_test_acc(model, X, ei, y, masks, fw_kw, label, indent='    '):
    """Test accuracy at the best-validation epoch, or NaN if training fails."""
    try:
        _, _, _, _, val_accs, test_accs = train_model(
            model, X, ei, y, *masks, fw_kw, None, lr, wd, epochs, patience, verb=False)
        return test_accs[int(np.argmax(val_accs))]
    except torch.cuda.OutOfMemoryError:
        logging.warning(f"{indent}CUDA OOM: {label} -- skipping, recorded as NaN")
        torch.cuda.empty_cache()
        return np.nan
    except Exception:
        # Loud on purpose: a silent 0.0 here is indistinguishable from a
        # legitimately bad accuracy. NaN marks the run as not completed.
        logging.exception(f"{indent}Error in {label} -- recorded as NaN, not completed")
        return np.nan


def _dataset_subset():
    """Optional MATCHED_DATASETS subset, e.g. for splitting the run across GPUs."""
    subset = os.environ.get('MATCHED_DATASETS')
    return subset.split(',') if subset else ALL_DATASETS


def _save(fname, results_dict):
    with open(fname + '.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
    with open(fname + '.json', 'w') as f:
        json.dump(to_json_safe(results_dict), f, indent=2)
    logging.info(f"\nDone. Results saved to {fname}{{.npy,.pkl,.json}}")


def run_main():
    DEVICE_STR = os.environ.get('MATCHED_DEVICE', 'cuda:0')

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device(DEVICE_STR)

    seed_everything(42)

    commit_hash = git_commit(short=True) or 'unknown'
    n_sims = int(os.environ.get('MATCHED_NSIMS', 20))

    datasets = _dataset_subset()
    logging.info(f"Running on datasets: {datasets} (device={DEVICE_STR})")

    all_graphs = load_cached_graphs(GRAPH_CACHE)

    graphs = [k for k in all_graphs[ALL_DATASETS[0]].keys() if k != 'EPS-GraphWave']
    logging.info(f"Using {len(graphs)} graph views: {graphs}")

    data_dict = get_data_dict(datasets, all_graphs, node_name, n_sims, 0.8, 0.1)

    OUT_DIR = results_dir('table6_main')
    fname = str(OUT_DIR / 'results')

    CONFIG_NAMES = BASELINE_NAMES + SGGNN_NAMES

    best_accs_test = np.full((len(datasets), n_sims, len(CONFIG_NAMES)), np.nan, dtype=np.float32)
    param_counts   = {}     # {dataset: {config_name: n_params}}
    matched_hparams = {}    # {dataset: {baseline_name: {'hid': value, ...}}}

    for d, dataset_name in enumerate(datasets):
        logging.info(f"\n=== Dataset {d+1}/{len(datasets)}: {dataset_name} ===")

        data_pyg    = data_dict[dataset_name].to(device)
        N           = data_pyg[node_name].N
        out_dim     = data_pyg[node_name].num_classes
        in_dim      = data_pyg[node_name].x.size(1)

        X_het   = data_pyg.x_dict
        ei_het  = data_pyg.edge_index_dict
        X_hom   = data_pyg[node_name].x
        ei_orig = data_pyg[node_name, 'Original', node_name].edge_index

        # SG-GNN param counts set the target budget for every baseline.
        sg_builders = sggnn_builders(in_dim, out_dim, graphs, N)
        sggnn_params = {name: count_params(fn()) for name, fn in sg_builders.items()}
        target = max(sggnn_params.values())
        logging.info(f"  SG-GNN param counts: {sggnn_params}")
        logging.info(f"  Target budget for baselines: {target}")

        builders = baseline_builders(in_dim, out_dim)
        matched_val, achieved, matched_hparams[dataset_name] = match_capacity(builders, target)
        param_counts[dataset_name] = {**sggnn_params, **achieved}

        for s in range(n_sims):
            masks = sim_masks(data_pyg[node_name], s)
            cfgs = build_configs(CONFIG_NAMES, builders, sg_builders, matched_val, device,
                                 X_hom, ei_orig, X_het, ei_het)
            for ci, (cname, model, X, ei, fw_kw) in enumerate(cfgs):
                best_accs_test[d, s, ci] = best_test_acc(
                    model, X, ei, data_pyg[node_name].y, masks, fw_kw,
                    f"{cname}, sim {s} (dataset {dataset_name})")

            if (s + 1) % 5 == 0:
                logging.info(f"  sim {s+1}/{n_sims} done")

        for ci, cname in enumerate(CONFIG_NAMES):
            m_ = best_accs_test[d, :, ci].mean()
            s_ = best_accs_test[d, :, ci].std()
            logging.info(f"  {cname}: {m_:.4f} +/- {s_:.4f}")

        np.save(fname + '.npy', best_accs_test)  # checkpoint

    np.save(fname + '.npy', best_accs_test)

    _save(fname, {
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
    })


def run_newbaselines():
    DEVICE_STR = os.environ.get('MATCHED_DEVICE', 'cuda:0')
    SOURCE_PKLS = os.environ.get('MATCHED_SOURCE_PKLS', str(results_dir('table6_main') / 'results.pkl')).split(',')

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device(DEVICE_STR)
    seed_everything(42)

    commit_hash = git_commit(short=True) or 'unknown'
    n_sims = int(os.environ.get('MATCHED_NSIMS', 20))

    datasets = _dataset_subset()
    logging.info(f"Running on datasets: {datasets} (device={DEVICE_STR})")

    # Target budgets come from already-completed table6_main runs.
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

    # Only the Original edge_index of the cache is used.
    all_graphs = load_cached_graphs(GRAPH_CACHE)

    data_dict = get_data_dict(datasets, all_graphs, node_name, n_sims, 0.8, 0.1)

    OUT_DIR = results_dir('table6_newbaselines')
    fname = str(OUT_DIR / 'results')

    CONFIG_NAMES = DIFFUSION_NAMES

    best_accs_test = np.full((len(datasets), n_sims, len(CONFIG_NAMES)), np.nan, dtype=np.float32)
    param_counts = {}
    matched_hparams = {}

    for d, dataset_name in enumerate(datasets):
        logging.info(f"\n=== Dataset {d+1}/{len(datasets)}: {dataset_name} ===")
        data_pyg = data_dict[dataset_name].to(device)
        num_classes = data_pyg[node_name].num_classes
        in_dim = data_pyg[node_name].x.size(1)
        X_hom = data_pyg[node_name].x
        ei_orig = data_pyg[node_name, 'Original', node_name].edge_index

        builders = diffusion_builders(in_dim, num_classes)
        matched_val, param_counts[dataset_name], matched_hparams[dataset_name] = \
            match_capacity(builders, target_budget[dataset_name])

        for s in range(n_sims):
            masks = sim_masks(data_pyg[node_name], s)
            # Each model is built right before it trains, so RNG draws interleave.
            for ci, bname in enumerate(CONFIG_NAMES):
                model = builders[bname](matched_val[bname]).to(device)
                best_accs_test[d, s, ci] = best_test_acc(
                    model, X_hom, ei_orig, data_pyg[node_name].y, masks, {},
                    f"{bname}, sim {s} (dataset {dataset_name})")

            if (s + 1) % 5 == 0:
                logging.info(f"  sim {s+1}/{n_sims} done")

        for ci, cname in enumerate(CONFIG_NAMES):
            m_ = best_accs_test[d, :, ci].mean()
            s_ = best_accs_test[d, :, ci].std()
            logging.info(f"  {cname}: {m_:.4f} +/- {s_:.4f}")

        np.save(fname + '.npy', best_accs_test)

    np.save(fname + '.npy', best_accs_test)

    _save(fname, {
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
    })


def run_largegraph():
    DEVICE_STR = os.environ.get('MATCHED_DEVICE', 'cuda:0')
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device(DEVICE_STR)
    seed_everything(42)

    commit_hash = git_commit(short=True) or 'unknown'
    n_sims = int(os.environ.get('MATCHED_NSIMS', 10))
    K_knn  = 10

    OUT_DIR = results_dir('table6_largegraph')
    fname = str(OUT_DIR / 'results')

    data_pyg, graphs_hetero = load_roman_empire(node_name, K_knn)
    N = data_pyg[node_name].N
    num_classes = data_pyg[node_name].num_classes
    in_dim = data_pyg[node_name].x.size(1)
    data_pyg = data_pyg.to(device)

    X_hom = data_pyg[node_name].x
    ei_orig = data_pyg[node_name, 'Original', node_name].edge_index
    X_het = data_pyg.x_dict
    ei_het = data_pyg.edge_index_dict
    y = data_pyg[node_name].y

    baseline_names = BASELINE_NAMES + DIFFUSION_NAMES
    CONFIG_NAMES   = baseline_names + SGGNN_NAMES

    sg_builders = sggnn_builders(in_dim, num_classes, graphs_hetero, N)
    sggnn_params = {name: count_params(fn()) for name, fn in sg_builders.items()}
    target = max(sggnn_params.values())
    logging.info(f"SG-GNN param counts: {sggnn_params}")
    logging.info(f"Target budget for baselines: {target}")

    builders = baseline_builders(in_dim, num_classes, with_diffusion=True)
    matched_val, achieved, matched_hparams = match_capacity(builders, target, indent='  ')
    param_counts = {**sggnn_params, **achieved}

    best_accs_test = np.full((n_sims, len(CONFIG_NAMES)), np.nan, dtype=np.float32)

    for s in range(n_sims):
        masks = sim_masks(data_pyg[node_name], s)

        logging.info(f"Sim {s+1}/{n_sims}")

        cfgs = build_configs(CONFIG_NAMES, builders, sg_builders, matched_val, device,
                             X_hom, ei_orig, X_het, ei_het)
        for ci, (cname, model, X, ei, fw_kw) in enumerate(cfgs):
            best_accs_test[s, ci] = best_test_acc(model, X, ei, y, masks, fw_kw,
                                                  f"{cname}, sim {s}", indent='  ')
            logging.info(f"  {cname}: {best_accs_test[s, ci]:.4f}")

        np.save(fname + '.npy', best_accs_test)  # checkpoint every sim

    means = best_accs_test.mean(axis=0)
    stds  = best_accs_test.std(axis=0)
    logging.info("\n=== Roman-Empire full parameter-fair comparison ===")
    for ci, cname in enumerate(CONFIG_NAMES):
        logging.info(f"  {cname:12s}  {means[ci]:.4f} +/- {stds[ci]:.4f}")

    np.save(fname + '.npy', best_accs_test)

    meta = {
        'dataset': 'Roman-empire', 'n_nodes': N, 'n_classes': num_classes,
        'config_names': CONFIG_NAMES, 'baseline_names': baseline_names, 'sggnn_names': SGGNN_NAMES,
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


def run_merge():
    """Merge table6_main + table6_newbaselines + table6_largegraph into the
    published 17-config x 12-dataset table (table6/results.pkl), following
    KNNHeterophilic's make_table6_merge.py reconstruction: table6_newbaselines
    supplies GRAND/CD-GNN/LG-GNN, spliced in after the 8 original baselines
    and before the 6 SG-GNN variants."""
    with open(results_dir('table6_main') / 'results.pkl', 'rb') as f:
        main = pickle.load(f)
    with open(results_dir('table6_newbaselines') / 'results.pkl', 'rb') as f:
        newb = pickle.load(f)
    with open(results_dir('table6_largegraph') / 'results.pkl', 'rb') as f:
        lg = pickle.load(f)

    n_orig_baselines = 8
    small_datasets = main['metadata']['datasets']
    all_datasets = small_datasets + [lg['dataset']]

    def pad_largegraph_row(arr_2d, n_sims_small):
        # lg['best_accs_test'] is [n_sims_lg, n_configs]; align to the small
        # tables' [n_sims_small] axis by cycling/truncating if needed.
        n_sims_lg = arr_2d.shape[0]
        if n_sims_lg == n_sims_small:
            return arr_2d
        idx = np.arange(n_sims_small) % n_sims_lg
        return arr_2d[idx]

    n_sims_small = main['results'].shape[1]
    lg_results = np.asarray(lg['best_accs_test'])
    lg_results = pad_largegraph_row(lg_results, n_sims_small)[None, :, :]  # [1, n_sims, n_configs]

    # Config order in `lg`: 8 baselines, GRAND, CD-GNN, LG-GNN, 6 SG-GNN (17 total).
    lg_config_names = lg['config_names']
    main_config_names = main['config_names']
    newb_config_names = newb['config_names']
    assert lg_config_names == main_config_names[:n_orig_baselines] + newb_config_names + main_config_names[n_orig_baselines:]

    # Small-dataset results: splice newbaselines in after the 8 baselines.
    main_results = np.concatenate([
        main['results'][:, :, :n_orig_baselines],
        newb['results'],
        main['results'][:, :, n_orig_baselines:],
    ], axis=2)  # [n_small_datasets, n_sims, 17]

    merged_config_names = main_config_names[:n_orig_baselines] + newb_config_names + main_config_names[n_orig_baselines:]
    merged_results = np.concatenate([main_results, lg_results], axis=0)  # [n_all_datasets, n_sims, 17]

    means = merged_results.mean(axis=1)
    stds = merged_results.std(axis=1)

    payload = {
        'results': merged_results, 'config_names': merged_config_names,
        'baseline_names': merged_config_names[:n_orig_baselines + 3],
        'sggnn_names': merged_config_names[n_orig_baselines + 3:],
        'datasets': all_datasets, 'means': means, 'stds': stds,
        'metadata': {'experiment_id': 'table6', 'timestamp': datetime.datetime.now().isoformat()},
    }
    out = results_dir('table6')
    with open(out / 'results.pkl', 'wb') as f:
        pickle.dump(payload, f)

    with open(out / 'results.json', 'w') as f:
        json.dump(to_json_safe(payload), f, indent=2)
    logging.info(f"Merged table saved to {out}")
    return payload


if __name__ == '__main__':
    if PART == 'main':
        run_main()
    elif PART == 'newbaselines':
        run_newbaselines()
    elif PART == 'largegraph':
        run_largegraph()
    elif PART == 'merge':
        run_merge()
    else:
        raise ValueError(f"Unknown MAIN_TABLE_PART: {PART}")
