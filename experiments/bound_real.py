"""
Empirical tightness of Theorem 1 on the 11 real datasets, using the theorem's
EXACT model.

Experiment ID: bound_theorem_real

Produces the real-data numbers quoted in Section VI-E (ratios between 1e-7
and 1e-4, alpha term below 0.4% of the bound). The model is the theorem's
plain 2-layer GCN on the Original graph only, taken from
`experiments/bound_synth.py`, and the bound (eq:err_bnd) is used verbatim:

    ||Z* - Zhat||_F  <=  rho1 * rho2 * ( alpha*sqrt(N) + 2*(1+sqrt(N))*||Delta||_F*||X||_F )

Model (Theorem 1, verbatim, imported unmodified from bound_synth.TheoremGCN)
-----------------------------------------------------------------------------
    Phi(X; A, Theta) = sigma2( Arw sigma1( Arw X Theta1 ) Theta2 )

with Arw = Dhat^{-1}(A + I) row-normalized, sigma1 = tanh, sigma2 = identity
(headline) or softmax (saturation-artifact comparison), no biases. A is the
'Original' graph ONLY -- no multi-branch HeteroData, no other graph views.

Data
----
11 real datasets loaded via sggnn.data.get_data_dict, restricted to the 'Original'
edge_index. Per dataset, 10 splits from the mask columns produced by
get_data_dict/create_masks.
    A*  = 'Original' edges with y_i == y_j only. Arw* = row_normalize(A*+I).
    X*  = per-class mean features.
    alpha = max_i ||X*_i - X_i||_2.
||Delta||_F has a closed form here: Delta = A - A* is exactly the cross-label
entries of the Original graph (each weight 1), so
    ||Delta||_F = sqrt(#cross-label edges in Original).
This is asserted against a dense computation on Texas (183 nodes) as a
correctness check; no other dataset ever builds a dense N x N matrix (Actor:
7600 nodes, Squirrel: 5201 nodes -- Arw is kept as a sparse COO tensor
throughout).

Outputs
-------
  results/<RESULTS_DIR>/bound_real/bound_real.pkl
  results/<RESULTS_DIR>/bound_real/bound_real.json
Plots/tables are produced separately by `analysis/plot_bound_real.py`,
which consumes only these artifacts.
"""

import os
import sys
import json
import pickle
import logging
import argparse
import datetime
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn

from sggnn.data import get_data_dict, seed_everything, load_cached_graphs
from bound_synth import TheoremGCN

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

from sggnn.paths import results_dir, git_commit, nan_to_none
OUT_DIR = Path(os.environ.get('BOUND_REAL_OUTDIR', str(results_dir('bound_real'))))
OUT_DIR.mkdir(parents=True, exist_ok=True)

NODE_NAME = 'web'
DATASETS = [
    'Texas', 'Wisconsin', 'Cornell', 'Actor', 'Chameleon',
    'Squirrel', 'Cora', 'CiteSeer', 'USA', 'Europe', 'Brazil',
]

HID_DIM = 32
LR = 5e-3
WD = 5e-4
EPOCHS = 2000
PATIENCE = 300
N_SPLITS = 10
BASE_SEED = 100


# ── model: Theorem 1 verbatim, sigma2 extended with 'softmax' ───────────────
class TheoremGCNReal(TheoremGCN):
    """Same as bound_synth.TheoremGCN; adds sigma2='softmax' without touching
    the original class (which only supports 'identity'/'tanh')."""

    def __init__(self, in_dim, hid_dim, out_dim, sigma1='tanh', sigma2='identity'):
        base = sigma2 if sigma2 in ('identity', 'tanh') else 'identity'
        super().__init__(in_dim, hid_dim, out_dim, sigma1=sigma1, sigma2=base)
        self.sigma2_name = sigma2
        if sigma2 == 'softmax':
            self.sigma2 = lambda t: torch.softmax(t, dim=1)


# ── sparse row-normalization (no dense N x N matrices) ───────────────────────
def sparse_row_normalize(edge_index, N, device, dtype=torch.float32):
    """Arw = Dhat^{-1}(A + I), sparse. `edge_index` must NOT contain self
    loops (they are added here, matching bound_synth.row_normalize's dense
    convention where self loops enter only through the '+I' term)."""
    self_loops = torch.arange(N, device=device)
    row = torch.cat([edge_index[0], self_loops])
    col = torch.cat([edge_index[1], self_loops])
    val = torch.ones(row.shape[0], device=device, dtype=dtype)

    deg = torch.zeros(N, device=device, dtype=dtype)
    deg.scatter_add_(0, row, val)
    inv_deg = 1.0 / deg.clamp(min=1e-12)
    norm_val = val * inv_deg[row]

    indices = torch.stack([row, col], dim=0)
    Arw = torch.sparse_coo_tensor(indices, norm_val, size=(N, N)).coalesce()
    return Arw




def cross_label_delta(edge_index, y):
    """Delta = A - A* is exactly the cross-label entries of A (each weight 1).
    ||Delta||_F = sqrt(#cross-label directed edges). Closed form -- no matrix
    is ever built. `edge_index` must have self loops already removed."""
    src, dst = edge_index
    cross = (y[src] != y[dst])
    n_cross = int(cross.sum().item())
    n_same = int((~cross).sum().item())
    delta_normF = float(np.sqrt(n_cross))
    return delta_normF, n_cross, n_same


def dense_delta_check(edge_index, y, N, device):
    """Dense-matrix correctness check for the closed-form ||Delta||_F above.
    Only ever called on Texas (N=183)."""
    A = torch.zeros(N, N, device=device)
    A[edge_index[0], edge_index[1]] = 1.0
    same = (y[:, None] == y[None, :]).float()
    A_star = A * same
    Delta = A - A_star
    return torch.linalg.matrix_norm(Delta, ord='fro').item()


def build_oracle_features(X, y, num_classes):
    X_star = torch.zeros_like(X)
    for c in range(num_classes):
        mask = (y == c)
        if mask.any():
            X_star[mask] = X[mask].mean(dim=0)
    return X_star


def edge_homophily(n_cross, n_same):
    total = n_cross + n_same
    return float(n_same) / total if total > 0 else float('nan')


def remove_self_loops(edge_index):
    # Kept local: torch_geometric.utils.remove_self_loops returns the same edges
    # but changes the softmax-config numbers in the last digits.
    src, dst = edge_index
    mask = src != dst
    return torch.stack([src[mask], dst[mask]])


# ── one (dataset, split, sigma2) run ─────────────────────────────────────────
def run_once(X, y, edge_index, star_edge_index, N, num_classes,
             train_mask, val_mask, test_mask, sigma2, seed, device):
    seed_everything(seed)

    Arw = sparse_row_normalize(edge_index, N, device)
    Arw_star = sparse_row_normalize(star_edge_index, N, device)
    X_star = build_oracle_features(X, y, num_classes)

    model = TheoremGCNReal(X.shape[1], HID_DIM, num_classes, sigma1='tanh', sigma2=sigma2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.CrossEntropyLoss()

    best_val, best_state, best_test, wait = -1.0, None, 0.0, 0
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        out = model(X, Arw)
        lossf(out[train_mask], y[train_mask]).backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(X, Arw).argmax(1)
            v = (pred[val_mask] == y[val_mask]).float().mean().item()
        if v > best_val:
            best_val, wait = v, 0
            best_test = (pred[test_mask] == y[test_mask]).float().mean().item()
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        Z_hat = model(X, Arw)
        Z_star = model(X_star, Arw_star)
        observed = torch.linalg.matrix_norm(Z_star - Z_hat, ord='fro').item()

        rho1, rho2 = model.rhos()
        alpha = (X_star - X).norm(dim=1).max().item()
        X_normF = torch.linalg.matrix_norm(X, ord='fro').item()
        delta_normF, n_cross, n_same = cross_label_delta(edge_index, y)
        sqrtN = float(N) ** 0.5

        # Eq. (12), exactly as stated -- no step tightened
        term_alpha = alpha * sqrtN
        term_delta = 2.0 * (1.0 + sqrtN) * delta_normF * X_normF
        bound = rho1 * rho2 * (term_alpha + term_delta)
        ratio = observed / bound if bound > 0 else float('nan')
        term_alpha_full = rho1 * rho2 * term_alpha
        term_delta_full = rho1 * rho2 * term_delta
        alpha_share_pct = 100.0 * term_alpha_full / bound if bound > 0 else float('nan')

    return {
        'observed_error': observed,
        'bound': bound,
        'ratio': ratio,
        'term_alpha': term_alpha_full,
        'term_delta': term_delta_full,
        'alpha_share_pct': alpha_share_pct,
        'rho1': rho1, 'rho2': rho2, 'rho_prod': rho1 * rho2,
        'alpha': alpha,
        'X_normF': X_normF,
        'delta_normF': delta_normF,
        'n_cross_edges': n_cross,
        'n_same_edges': n_same,
        'edge_homophily': edge_homophily(n_cross, n_same),
        'test_acc': best_test,
        'val_acc': best_val,
        'N': N,
        'sigma2': sigma2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_splits', type=int, default=N_SPLITS)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    logging.info(f"device={device}  n_splits={args.n_splits}")

    # ── load graphs, keep ONLY the 'Original' view to avoid building the
    #    13 other dense N x N HeteroData branches ─────────────────────────
    # Seed before the very first RNG use (the random splits drawn by
    # get_data_dict below), not just inside run_once's per-split
    # seed_everything(seed). Without this, the original script draws a
    # different split every process run even at fixed seed/args -- confirmed
    # by running the unmodified original twice and getting different
    # results. Fixed here only; the manuscript quotes qualitative ranges
    # (e.g. "between 1e-7 and 1e-4", "never more than 0.4%"), not exact
    # per-dataset digits, so this does not change any number in the paper.
    seed_everything(BASE_SEED)

    logging.info("Loading precomputed graphs (Original view only)...")
    GRAPH_CACHE = os.environ.get('GRAPH_CACHE', str(ROOT / 'node_embeddings' / 'embedding_graphs.npz'))
    all_graphs = {ds: {'Original': g['Original']}
                  for ds, g in load_cached_graphs(GRAPH_CACHE).items() if ds in DATASETS}

    data_dict = get_data_dict(DATASETS, all_graphs, NODE_NAME, args.n_splits, 0.8, 0.1)

    results = {}
    errors = {}

    for dataset_name in DATASETS:
        logging.info(f"\n=== Dataset: {dataset_name} ===")
        try:
            data_pyg = data_dict[dataset_name].to(device)
            N = data_pyg[NODE_NAME].N
            num_classes = data_pyg[NODE_NAME].num_classes
            X = data_pyg[NODE_NAME].x
            y = data_pyg[NODE_NAME].y

            ei_orig_raw = data_pyg[NODE_NAME, 'Original', NODE_NAME].edge_index
            edge_index = remove_self_loops(ei_orig_raw)

            src, dst = edge_index
            same_mask = (y[src] == y[dst])
            star_edge_index = torch.stack([src[same_mask], dst[same_mask]])

            if dataset_name == 'Texas':
                closed_form, _, _ = cross_label_delta(edge_index, y)
                dense = dense_delta_check(edge_index, y, N, device)
                logging.info(f"  [check] ||Delta||_F closed-form={closed_form:.6f} "
                             f"dense={dense:.6f} diff={abs(closed_form - dense):.2e}")
                assert abs(closed_form - dense) < 1e-3, \
                    f"closed-form/dense ||Delta||_F mismatch on Texas: {closed_form} vs {dense}"

            tm = data_pyg[NODE_NAME].train_mask
            vm = data_pyg[NODE_NAME].val_mask
            tsm = data_pyg[NODE_NAME].test_mask

            dataset_results = {'identity': [], 'softmax': []}
            for split in range(args.n_splits):
                train_mask = tm[:, split] if tm.ndim > 1 else tm
                val_mask = vm[:, split] if vm.ndim > 1 else vm
                test_mask = tsm[:, split] if tsm.ndim > 1 else tsm
                seed = BASE_SEED + split

                for sigma2 in ('identity', 'softmax'):
                    r = run_once(X, y, edge_index, star_edge_index, N, num_classes,
                                 train_mask, val_mask, test_mask, sigma2, seed, device)
                    dataset_results[sigma2].append(r)

                logging.info(
                    f"  split {split}: "
                    f"id: obs={dataset_results['identity'][-1]['observed_error']:.3f} "
                    f"ratio={dataset_results['identity'][-1]['ratio']:.3e} "
                    f"acc={dataset_results['identity'][-1]['test_acc']:.3f} | "
                    f"sm: obs={dataset_results['softmax'][-1]['observed_error']:.3f} "
                    f"ratio={dataset_results['softmax'][-1]['ratio']:.3e}"
                )

            agg = {}
            for sigma2 in ('identity', 'softmax'):
                runs = dataset_results[sigma2]
                a = {'runs': runs}
                for k in runs[0]:
                    if k == 'sigma2':
                        continue
                    vals = [rr[k] for rr in runs]
                    a[f'{k}_mean'] = float(np.mean(vals))
                    a[f'{k}_std'] = float(np.std(vals))
                agg[sigma2] = a

            results[dataset_name] = agg
            logging.info(
                f"  [{dataset_name}] identity: obs={agg['identity']['observed_error_mean']:.4f} "
                f"bound={agg['identity']['bound_mean']:.4g} ratio={agg['identity']['ratio_mean']:.4g} | "
                f"softmax: ratio={agg['softmax']['ratio_mean']:.4g} | "
                f"N={N} homophily={agg['identity']['edge_homophily_mean']:.3f}"
            )

        except torch.cuda.OutOfMemoryError:
            logging.warning(f"  CUDA OOM on {dataset_name} -- skipping")
            torch.cuda.empty_cache()
            errors[dataset_name] = 'CUDA OutOfMemoryError'
        except Exception as exc:
            logging.warning(f"  Error on {dataset_name}: {exc}", exc_info=True)
            errors[dataset_name] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    payload = {
        'results': results,
        'errors': errors,
        'metadata': {
            'experiment_id': 'bound_theorem_real',
            'datasets': DATASETS,
            'node_name': NODE_NAME,
            'n_splits': args.n_splits,
            'seeds': [BASE_SEED + s for s in range(args.n_splits)],
            'hid_dim': HID_DIM,
            'lr': LR,
            'wd': WD,
            'epochs': EPOCHS,
            'patience': PATIENCE,
            'device': str(device),
            'timestamp': datetime.datetime.now().isoformat(),
            'git_commit': git_commit(),
            'torch': torch.__version__,
            'bound': ('rho1*rho2*(alpha*sqrt(N) + 2*(1+sqrt(N))*||Delta||_F*||X||_F), '
                      'Eq. (12) exactly as stated; no proof step tightened.'),
            'model': ('sigma2(Arw sigma1(Arw X Theta1) Theta2), Arw = Dhat^{-1}(A+I) '
                      '(sparse, row-normalized), sigma1 = tanh, sigma2 in {identity, softmax} '
                      'per config, no biases. Plain 2-layer GCN on the single Original graph '
                      '-- NOT the deployed multi-branch AdaptiveAggGCN.'),
            'oracle': ("A* = Original edges with y_i==y_j only; Arw* = row_normalize(A*+I). "
                       "X* = per-class mean features. alpha = max_i ||X*_i - X_i||_2. "
                       "||Delta||_F = sqrt(#cross-label directed edges in Original), closed "
                       "form, verified against a dense computation on Texas."),
            'note': ('Replaces the flawed measurement in ablations/bound_tightness.py, which '
                      'profiled AdaptiveAggGCN (R=14 branches), used a softmax output, took '
                      'rho_combined from two unrelated weight matrices, and swapped A* into '
                      'only one of 14 branches.'),
        },
    }

    with open(OUT_DIR / 'bound_real.pkl', 'wb') as f:
        pickle.dump(payload, f)

    with open(OUT_DIR / 'bound_real.json', 'w') as f:
        json.dump(nan_to_none(payload), f, indent=2, default=float)

    logging.info(f"\nDone. Saved -> {OUT_DIR}")
    if errors:
        logging.warning(f"Datasets with errors: {list(errors.keys())}")


if __name__ == '__main__':
    main()
