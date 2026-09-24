"""
Total wall-clock training time to convergence: Table XII (`tab:profiling_small`,
Actor/Chameleon/Squirrel) and Table XIII (`tab:roman_empire_time`, Roman-Empire).
Merges the former timing_total.py and timing_total_largegraph.py.

Each model is trained with the project's standard early stopping (patience=300,
epochs<=2000), n_sims repetitions, each on a different random split. Recorded
per (dataset, model): total wall-clock time to convergence, epochs run, epoch
of best validation accuracy, time/epoch, peak GPU memory, best test accuracy.

Select which table with TIMING_PARTS=table12,table13 (default: both). Needs
the candidate-graph cache for table12 (`python main.py build-graphs`); table13
builds its own graphs (Roman-Empire, downloaded by PyTorch Geometric) on the fly.

`analysis/timing_table.py` builds Table XII's LaTeX; `analysis/plot_timing_largegraph.py`
builds Table XIII's.

Output: results/<RESULTS_DIR>/{table12,table13}/results.{pkl,json}
"""
import os
import sys
import logging
import pickle
import datetime
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv

from sggnn.models import AdaptiveAggGCN, GCN, FBGNNLayer
from sggnn.data import (get_data_dict, seed_everything, load_cached_graphs, load_roman_empire,
                         count_params, sim_masks)
from sggnn.train import train_model
from sggnn.baselines import GRANDWrapper, CDGNNWrapper, LGGNNWrapper
from sggnn.paths import results_dir, git_commit, nan_to_none

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

GRAPH_CACHE = os.environ.get('GRAPH_CACHE', str(ROOT / 'node_embeddings' / 'embedding_graphs.npz'))
PARTS = os.environ.get('TIMING_PARTS', 'table12,table13').split(',')

def time_model(build, X, ei, fw_kw, y, store, n_sims, device, lr, wd, epochs, patience, indent=''):
    """Train a fresh `build()` model n_sims times to convergence, one split per
    sim, and summarize wall-clock time, epochs, peak GPU memory and accuracy."""
    cuda_available = device.type == 'cuda'
    total_times, epochs_runs, best_epochs = [], [], []
    peak_mems, best_test_accs = [], []
    n_params = -1

    for sim in range(n_sims):
        train_mask, val_mask, test_mask = sim_masks(store, sim)

        model = build()

        if cuda_available:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        model, _, _, _, val_accs, test_accs, epoch_info = train_model(
            model, X, ei, y, train_mask, val_mask, test_mask, fw_kw,
            None, lr, wd, epochs, patience, verb=False,
            return_epoch_info=True,
        )

        if cuda_available:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_time = t1 - t0
        epochs_run = epoch_info['epochs_run']
        best_epoch = int(np.argmax(val_accs[:epochs_run]))
        best_test_acc = float(test_accs[best_epoch])

        if cuda_available:
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
        else:
            peak_mem = float('nan')

        n_params = count_params(model)

        total_times.append(total_time)
        epochs_runs.append(epochs_run)
        best_epochs.append(best_epoch)
        peak_mems.append(peak_mem)
        best_test_accs.append(best_test_acc)

        logging.info(
            f"{indent}  sim {sim}: {total_time:.2f}s total, {epochs_run} epochs "
            f"({total_time / epochs_run * 1000:.2f} ms/epoch), "
            f"best_epoch={best_epoch}, test_acc={best_test_acc:.4f}, "
            f"peak_mem={peak_mem:.1f} MB"
        )

    time_per_epoch = [t / e for t, e in zip(total_times, epochs_runs)]

    return {
        'total_time_sec_mean':   float(np.mean(total_times)),
        'total_time_sec_std':    float(np.std(total_times)),
        'total_time_sec_all':    total_times,
        'epochs_run_mean':       float(np.mean(epochs_runs)),
        'epochs_run_std':        float(np.std(epochs_runs)),
        'epochs_run_all':        epochs_runs,
        'best_epoch_mean':       float(np.mean(best_epochs)),
        'best_epoch_std':        float(np.std(best_epochs)),
        'best_epoch_all':        best_epochs,
        'time_per_epoch_mean':   float(np.mean(time_per_epoch)),
        'time_per_epoch_std':    float(np.std(time_per_epoch)),
        'peak_memory_mb_mean':   float(np.mean(peak_mems)),
        'peak_memory_mb_std':    float(np.std(peak_mems)),
        'n_params':              n_params,
        'best_test_acc_mean':    float(np.mean(best_test_accs)),
        'best_test_acc_std':     float(np.std(best_test_accs)),
        'best_test_acc_all':     best_test_accs,
    }


def log_summary(mname, r, indent=''):
    logging.info(
        f"{indent}{mname}: total={r['total_time_sec_mean']:.2f}s "
        f"(+/- {r['total_time_sec_std']:.2f}), "
        f"epochs={r['epochs_run_mean']:.1f}, "
        f"{r['time_per_epoch_mean']*1000:.2f} ms/epoch, "
        f"peak mem={r['peak_memory_mb_mean']:.1f} MB, params={r['n_params']}, "
        f"test_acc={r['best_test_acc_mean']:.4f}"
    )


def _save(out_dir, full_results):
    with open(str(out_dir / 'results.pkl'), 'wb') as f:
        pickle.dump(full_results, f)
    with open(str(out_dir / 'results.json'), 'w') as f:
        json.dump(nan_to_none(full_results), f, indent=2)
    logging.info(f"\nDone. Results saved to {out_dir}")


def run_table12():
    SEED = 42
    seed_everything(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    cuda_available = device.type == 'cuda'

    commit_hash = git_commit(short=True) or 'unknown'

    OUT_DIR = results_dir('table12')

    # ── hyperparams (project standard) ───────────────────────────────────────────
    hid_dim   = 32
    dropout   = 0.5
    nonlin    = nn.Tanh()
    last_act  = nn.Softmax(dim=1)
    lr        = 5e-3
    wd        = 5e-4
    n_layers  = 1
    node_name = 'web'
    epochs    = 2000
    patience  = 300

    n_sims = 3  # repeats, each on a distinct train/val/test split column

    datasets = ['Actor', 'Chameleon', 'Squirrel']

    logging.info("Loading precomputed graphs...")
    all_graphs = load_cached_graphs(GRAPH_CACHE)

    available_keys = list(all_graphs[datasets[0]].keys())
    # EPS-GraphWave runs out of memory on Chameleon and is excluded everywhere.
    all_graph_keys = [k for k in available_keys if k != 'EPS-GraphWave']
    logging.info(f"Graph keys used: {all_graph_keys}")

    data_dict = get_data_dict(datasets, all_graphs, node_name, n_sims, 0.8, 0.1)


    # ── result structure ──────────────────────────────────────────────────────────
    timing_total_results = {}

    for dataset_name in datasets:
        logging.info(f"\n=== Dataset: {dataset_name} ===")
        timing_total_results[dataset_name] = {}

        try:
            data_pyg    = data_dict[dataset_name].to(device)
            num_classes = data_pyg[node_name].num_classes
            in_dim      = data_pyg[node_name].x.size(1)
            y           = data_pyg[node_name].y


            model_specs = [
                {
                    'name': 'GCN',
                    'build': lambda: GCN(
                        in_dim, hid_dim, num_classes, n_layers,
                        nonlin=nonlin, last_act=nn.Identity(),
                        dropout=dropout, gcnlayer=GCNConv
                    ).to(device),
                    'X':  data_pyg[node_name].x,
                    'ei': data_pyg[node_name, 'Original', node_name].edge_index,
                    'fw_kwargs': {},
                },
                {
                    'name': 'FBGNN',
                    'build': lambda: GCN(
                        in_dim, hid_dim, num_classes, n_layers,
                        nonlin=nonlin, last_act=last_act,
                        dropout=dropout, gcnlayer=FBGNNLayer
                    ).to(device),
                    'X':  data_pyg[node_name].x,
                    'ei': data_pyg[node_name, 'Original', node_name].edge_index,
                    'fw_kwargs': {},
                },
                {
                    'name': 'AdaptiveAggGCN-GCN',
                    'build': lambda: AdaptiveAggGCN(
                        in_dim, hid_dim, num_classes, n_layers,
                        dropout, nonlin, nn.Identity(),
                        all_graph_keys, GCNConv, GCNConv, -1
                    ).to(device),
                    'X':  data_pyg.x_dict,
                    'ei': data_pyg.edge_index_dict,
                    'fw_kwargs': {},
                },
                {
                    'name': 'AdaptiveAggGCN-FBGNN',
                    'build': lambda: AdaptiveAggGCN(
                        in_dim, hid_dim, num_classes, n_layers,
                        dropout, nonlin, last_act,
                        all_graph_keys, FBGNNLayer, FBGNNLayer, -1
                    ).to(device),
                    'X':  data_pyg.x_dict,
                    'ei': data_pyg.edge_index_dict,
                    'fw_kwargs': {},
                },
            ]

            for spec in model_specs:
                mname = spec['name']
                logging.info(f"  Training {mname} to convergence ({n_sims} repeats)...")

                try:
                    r = time_model(spec['build'], spec['X'], spec['ei'], spec['fw_kwargs'], y,
                                   data_pyg[node_name], n_sims, device, lr, wd, epochs, patience,
                                   indent='  ')
                    timing_total_results[dataset_name][mname] = r
                    log_summary(mname, r, indent='  ')

                except torch.cuda.OutOfMemoryError:
                    logging.warning(f"    CUDA OOM on {mname}, {dataset_name} -- skipping")
                    torch.cuda.empty_cache()
                    timing_total_results[dataset_name][mname] = {'error': 'CUDA OOM'}
                except Exception as exc:
                    logging.warning(f"    Error on {mname}, {dataset_name}: {exc}")
                    timing_total_results[dataset_name][mname] = {'error': str(exc)}

        except torch.cuda.OutOfMemoryError:
            logging.warning(f"  CUDA OOM setting up {dataset_name} -- skipping")
            torch.cuda.empty_cache()
        except Exception as exc:
            logging.warning(f"  Error setting up {dataset_name}: {exc}")

    # ── save ─────────────────────────────────────────────────────────────────────
    full_results = {
        'results': timing_total_results,
        'metadata': {
            'experiment_id': 'timing_total',
            'seed':          SEED,
            'datasets':      datasets,
            'timestamp':     datetime.datetime.now().isoformat(),
            'commit_hash':   commit_hash,
            'torch_version': torch.__version__,
            'device_name':   torch.cuda.get_device_name(device) if cuda_available else 'cpu',
            'hid_dim':       hid_dim,
            'dropout':       dropout,
            'lr':            lr,
            'wd':            wd,
            'epochs':        epochs,
            'patience':      patience,
            'n_sims':        n_sims,
            'n_layers':      n_layers,
            'all_graph_keys': all_graph_keys,
        }
    }

    _save(OUT_DIR, full_results)


def run_table13():
    SEED = 42
    seed_everything(SEED)

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device('cuda:0')  # select the physical GPU with CUDA_VISIBLE_DEVICES

    commit_hash = git_commit(short=True) or 'unknown'

    OUT_DIR = results_dir('table13')

    # ── hyperparams (fixed-width comparison) ─────────────────────────────────────
    n_layers = 2
    hid_dim  = 32
    dropout  = 0.5
    lr       = 5e-3
    wd       = 5e-4
    K_knn    = 10
    node_name = 'web'
    nonlin   = nn.Tanh()
    last_act_softmax = nn.Softmax(dim=1)
    epochs   = 2000
    patience = 300

    n_sims = 3  # repeats, each on a distinct pre-defined split column

    data_pyg, graphs_hetero = load_roman_empire(node_name, K_knn)
    N = data_pyg[node_name].N
    num_classes = data_pyg[node_name].num_classes
    in_dim = data_pyg[node_name].x.size(1)
    data_pyg = data_pyg.to(device)

    y = data_pyg[node_name].y

    X_hom = data_pyg[node_name].x
    ei_orig = data_pyg[node_name, 'Original', node_name].edge_index
    X_het = data_pyg.x_dict
    ei_het = data_pyg.edge_index_dict


    # ── model specs ───────────────────────────────────────────────────────────────
    model_specs = [
        {
            'name': 'GCN',
            'build': lambda: GCN(in_dim, hid_dim, num_classes, n_layers,
                                  nonlin=nonlin, last_act=nn.Identity(),
                                  dropout=dropout, gcnlayer=GCNConv).to(device),
            'X': X_hom, 'ei': ei_orig, 'fw_kwargs': {},
        },
        {
            'name': 'GAT',
            'build': lambda: GCN(in_dim, hid_dim, num_classes, n_layers,
                                  nonlin=nonlin, last_act=nn.Identity(),
                                  dropout=dropout, gcnlayer=GATConv).to(device),
            'X': X_hom, 'ei': ei_orig, 'fw_kwargs': {},
        },
        {
            'name': 'GRAND',
            'build': lambda: GRANDWrapper(in_dim, hid_dim, num_classes, n_layers,
                                           dropout, nonlin, last_act_softmax).to(device),
            'X': X_hom, 'ei': ei_orig, 'fw_kwargs': {},
        },
        {
            'name': 'CD-GNN',
            'build': lambda: CDGNNWrapper(in_dim, hid_dim, num_classes, n_layers,
                                           dropout, nonlin, last_act_softmax).to(device),
            'X': X_hom, 'ei': ei_orig, 'fw_kwargs': {},
        },
        {
            'name': 'LG-GNN',
            'build': lambda: LGGNNWrapper(in_dim, hid_dim, num_classes, n_layers,
                                           dropout, nonlin, last_act_softmax).to(device),
            'X': X_hom, 'ei': ei_orig, 'fw_kwargs': {},
        },
        {
            'name': 'SG-GCN',
            'build': lambda: AdaptiveAggGCN(
                in_dim, hid_dim, num_classes, n_layers, dropout, nonlin, nn.Identity(),
                graphs_hetero, GCNConv, GCNConv, -1).to(device),
            'X': X_het, 'ei': ei_het, 'fw_kwargs': {},
        },
        {
            'name': 'SG-FBGNN',
            'build': lambda: AdaptiveAggGCN(
                in_dim, hid_dim, num_classes, n_layers, dropout, nonlin, last_act_softmax,
                graphs_hetero, FBGNNLayer, FBGNNLayer, -1).to(device),
            'X': X_het, 'ei': ei_het, 'fw_kwargs': {},
        },
    ]

    # ── result structure ──────────────────────────────────────────────────────
    timing_total_results = {}

    for spec in model_specs:
        mname = spec['name']
        logging.info(f"Training {mname} to convergence on Roman-Empire ({n_sims} repeats)...")

        try:
            r = time_model(spec['build'], spec['X'], spec['ei'], spec['fw_kwargs'], y,
                           data_pyg[node_name], n_sims, device, lr, wd, epochs, patience)
            timing_total_results[mname] = r
            log_summary(mname, r)

        except torch.cuda.OutOfMemoryError:
            logging.warning(f"CUDA OOM on {mname} -- skipping")
            torch.cuda.empty_cache()
            timing_total_results[mname] = {'error': 'CUDA OOM'}
        except Exception as exc:
            logging.warning(f"Error on {mname}: {exc}")
            timing_total_results[mname] = {'error': str(exc)}

    # ── save ─────────────────────────────────────────────────────────────────
    full_results = {
        'results': timing_total_results,
        'metadata': {
            'experiment_id': 'timing_total_largegraph',
            'dataset':       'Roman-empire',
            'n_nodes':       N,
            'n_classes':     num_classes,
            'seed':          SEED,
            'timestamp':     datetime.datetime.now().isoformat(),
            'commit_hash':   commit_hash,
            'torch_version': torch.__version__,
            'device_name':   torch.cuda.get_device_name(device),
            'hid_dim':       hid_dim,
            'dropout':       dropout,
            'lr':            lr,
            'wd':            wd,
            'epochs':        epochs,
            'patience':      patience,
            'n_sims':        n_sims,
            'n_layers':      n_layers,
            'K_knn':         K_knn,
            'graphs_hetero': graphs_hetero,
            'model_names':   [spec['name'] for spec in model_specs],
        }
    }

    _save(OUT_DIR, full_results)


if __name__ == '__main__':
    if 'table12' in PARTS:
        run_table12()
    if 'table13' in PARTS:
        run_table13()
