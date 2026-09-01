"""
Ablation: Empirical verification of Theorem 1 (bound tightness).

For each dataset:
  1. Train AdaptiveAggGCN (FBGNNLayer, single layer) on the full data.
  2. Construct:
       A*  = oracle graph (only edges where y_i == y_j in the Original graph)
       X*  = class-mean features (replace each node's features with its class mean)
       Z_hat  = model output with actual A, X
       Z_star = model output with A*, X* (same weights, different inputs)
  3. Compute:
       observed_error     = ||Z_hat - Z_star||_F
       Delta              = A - A* (difference matrices)
       theoretical_bound  = rho1 * rho2 * (alpha * sqrt(N) + 2*(1+sqrt(N)) * ||Delta||_F * ||X||_F)
         where rho1, rho2 = spectral norms of the first-layer weight matrices
               alpha      = max_i ||X*_i - X_i||_2  (max row-wise L2 distance)

The analysis uses ALL nodes (no train/test masking) and focuses on
the Original graph branch of the first GNN layer.

Output:
  results/ablations/theory/theory_results.pkl
  results/ablations/theory/theory_results.json
"""

import sys
import logging
import pickle
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix

from arch import AdaptiveAggGCN, FBGNNLayer
from utils import get_data_dict, create_masks, seed_everything
from train import train_model

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

seed_everything(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

OUT_DIR = ROOT / 'results' / 'ablations' / 'theory'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── hyperparams ───────────────────────────────────────────────────────────────
hid_dim   = 32
dropout   = 0.5
nonlin    = nn.Tanh()
last_act  = nn.Softmax(dim=1)
lr        = 5e-3
wd        = 5e-4
epochs    = 2000
patience  = 300
n_sims    = 3       # fewer sims -- we only need a trained model, not mean/std
n_layers  = 1
node_name = 'web'

datasets = [
    'Texas', 'Wisconsin', 'Cornell', 'Actor', 'Chameleon',
    'Squirrel', 'Cora', 'CiteSeer', 'USA', 'Europe', 'Brazil'
]

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
all_graph_keys = [k for k in available_keys if k != 'EPS-GraphWave']
logging.info(f"Graph keys: {all_graph_keys}")

data_dict = get_data_dict(datasets, all_graphs, node_name, n_sims, 0.8, 0.1)

# ── helper: extract first-layer weight matrices ───────────────────────────────
def get_first_layer_weights(model, graph_key, node_name):
    """
    Extract the weight matrix from the first-layer GCNConv/FBGNNLayer
    for the given graph key. Returns a 2-D tensor (out_dim x in_dim).

    PyG's HeteroConv stores sub-convolutions in self.convs as a ModuleDict
    whose keys are tuple edge-types: (src_node, edge_type, dst_node).
    """
    hetero_key = (node_name, graph_key, node_name)
    conv_dict = model.conv.convs  # HeteroConv.convs is an nn.ModuleDict

    sub_conv = None
    if hetero_key in conv_dict:
        sub_conv = conv_dict[hetero_key]
    else:
        # Fall back: search by graph_key substring in any key representation
        for k, v in conv_dict.items():
            k_str = str(k)
            if graph_key in k_str:
                sub_conv = v
                break

    if sub_conv is None:
        return None

    # FBGNNLayer wraps MixHopConv; GCNConv has .lin.weight
    if hasattr(sub_conv, 'conv_layer'):
        # FBGNNLayer -> MixHopConv
        inner = sub_conv.conv_layer
        if hasattr(inner, 'lins'):
            # MixHopConv has a list of linears (one per hop power)
            W = inner.lins[0].weight  # shape: (out, in)
        elif hasattr(inner, 'lin'):
            W = inner.lin.weight
        else:
            return None
    elif hasattr(sub_conv, 'lin'):
        W = sub_conv.lin.weight
    else:
        return None
    return W


def spectral_norm_matrix(W):
    """Compute spectral norm (largest singular value) of a 2-D tensor."""
    return torch.linalg.matrix_norm(W, ord=2).item()


# ── main loop ─────────────────────────────────────────────────────────────────
theory_results = {}

for dataset_name in datasets:
    logging.info(f"\n=== Dataset: {dataset_name} ===")

    try:
        data_pyg    = data_dict[dataset_name].to(device)
        N           = data_pyg[node_name].N
        num_classes = data_pyg[node_name].num_classes
        in_dim      = data_pyg[node_name].x.size(1)
        y           = data_pyg[node_name].y

        # ── train model (use first mask split) ───────────────────────────────
        tm  = data_pyg[node_name].train_mask
        vm  = data_pyg[node_name].val_mask
        tsm = data_pyg[node_name].test_mask
        if tm.ndim > 1:
            train_mask = tm[:, 0]
            val_mask   = vm[:, 0]
            test_mask  = tsm[:, 0]
        else:
            train_mask = tm
            val_mask   = vm
            test_mask  = tsm

        model = AdaptiveAggGCN(
            in_dim, hid_dim, num_classes, n_layers,
            dropout, nonlin, last_act,
            all_graph_keys, FBGNNLayer, FBGNNLayer, -1
        ).to(device)

        model, _, _, _, val_accs, _ = train_model(
            model, data_pyg.x_dict, data_pyg.edge_index_dict,
            y, train_mask, val_mask, test_mask, {},
            None, lr, wd, epochs, patience, verb=False
        )
        logging.info(f"  Training done, best val acc: {max(val_accs):.4f}")

        model.eval()

        # ── build oracle adjacency A* from the Original graph ────────────────
        ei_orig = data_pyg[node_name, 'Original', node_name].edge_index  # [2, E]
        src     = ei_orig[0].cpu().numpy()
        dst     = ei_orig[1].cpu().numpy()
        y_cpu   = y.cpu().numpy()

        # A* keeps only edges where y_i == y_j
        same_class_mask = (y_cpu[src] == y_cpu[dst])
        src_star = src[same_class_mask]
        dst_star = dst[same_class_mask]

        ei_star = torch.tensor(
            np.stack([src_star, dst_star], axis=0), dtype=torch.long, device=device
        )

        # ── class-mean features X* ────────────────────────────────────────────
        X = data_pyg[node_name].x  # [N, F]
        X_star = torch.zeros_like(X)
        for c in range(num_classes):
            mask_c = (y == c)
            if mask_c.sum() > 0:
                X_star[mask_c] = X[mask_c].mean(dim=0)

        # ── Z_hat: model output with actual A, X ─────────────────────────────
        with torch.no_grad():
            Z_hat = model(data_pyg.x_dict, edge_index=data_pyg.edge_index_dict)

        # ── Z_star: model output with A*, X* ─────────────────────────────────
        # Build a modified HeteroData with A* replacing Original and X*
        x_dict_star  = {node_name: X_star}
        ei_dict_star = dict(data_pyg.edge_index_dict)  # shallow copy
        ei_dict_star[(node_name, 'Original', node_name)] = ei_star

        with torch.no_grad():
            Z_star = model(x_dict_star, edge_index=ei_dict_star)

        # ── observed error ────────────────────────────────────────────────────
        observed_error = torch.linalg.matrix_norm(Z_hat - Z_star, ord='fro').item()

        # ── Delta = A - A* (difference of adjacency matrices) ─────────────────
        A_dense = torch.zeros((N, N), device=device)
        A_dense[ei_orig[0], ei_orig[1]] = 1.0

        A_star_dense = torch.zeros((N, N), device=device)
        A_star_dense[ei_star[0], ei_star[1]] = 1.0

        Delta = A_dense - A_star_dense
        delta_norm = torch.linalg.matrix_norm(Delta, ord='fro').item()

        # ── spectral norms of first-layer weight matrices ─────────────────────
        # Use 'Original' graph branch (first key)
        W1 = get_first_layer_weights(model, 'Original', node_name)
        # Use a second graph key as second weight factor
        W2 = get_first_layer_weights(model, all_graph_keys[1] if len(all_graph_keys) > 1 else 'Original', node_name)

        if W1 is None:
            logging.warning(f"  Could not extract W1 for {dataset_name}, using identity approx")
            rho1 = 1.0
        else:
            rho1 = spectral_norm_matrix(W1)

        if W2 is None:
            rho2 = rho1
        else:
            rho2 = spectral_norm_matrix(W2)

        # Also get final linear layer spectral norm
        W_lin = model.lin.weight  # [out, n_graphs*hid]
        rho_lin = spectral_norm_matrix(W_lin)

        # For a two-factor bound: rho1 * rho_lin
        rho_combined = rho1 * rho_lin

        # ── alpha = max_i ||X*_i - X_i||_2 ───────────────────────────────────
        row_diffs = (X_star - X).norm(dim=1)  # [N]
        alpha     = row_diffs.max().item()

        # ── ||X||_F ──────────────────────────────────────────────────────────
        X_norm_F = torch.linalg.matrix_norm(X, ord='fro').item()

        # ── theoretical bound ─────────────────────────────────────────────────
        sqrt_N = float(N) ** 0.5
        theoretical_bound = rho_combined * (
            alpha * sqrt_N + 2.0 * (1.0 + sqrt_N) * delta_norm * X_norm_F
        )

        ratio = observed_error / theoretical_bound if theoretical_bound > 0 else float('nan')

        theory_results[dataset_name] = {
            'observed_error':    float(observed_error),
            'theoretical_bound': float(theoretical_bound),
            'ratio':             float(ratio),
            'delta_norm':        float(delta_norm),
            'alpha':             float(alpha),
            'X_norm_F':          float(X_norm_F),
            'rho1':              float(rho1),
            'rho2':              float(rho2),
            'rho_lin':           float(rho_lin),
            'rho_combined':      float(rho_combined),
            'N':                 int(N),
            'n_classes':         int(num_classes),
        }

        logging.info(
            f"  {dataset_name}: observed={observed_error:.4f}, "
            f"bound={theoretical_bound:.4f}, ratio={ratio:.4f}, "
            f"delta_norm={delta_norm:.4f}"
        )

    except torch.cuda.OutOfMemoryError:
        logging.warning(f"  CUDA OOM on {dataset_name} -- skipping")
        torch.cuda.empty_cache()
    except Exception as exc:
        logging.warning(f"  Error on {dataset_name}: {exc}", exc_info=True)

# ── save ─────────────────────────────────────────────────────────────────────
full_results = {
    'results': theory_results,
    'metadata': {
        'seed':      42,
        'datasets':  datasets,
        'timestamp': datetime.datetime.now().isoformat(),
        'hid_dim':   hid_dim,
        'dropout':   dropout,
        'lr':        lr,
        'wd':        wd,
        'epochs':    epochs,
        'patience':  patience,
        'n_sims':    n_sims,
        'n_layers':  n_layers,
        'note': (
            'rho_combined = spectral_norm(W_orig_layer1) * spectral_norm(W_lin). '
            'Bound: rho_combined*(alpha*sqrt(N) + 2*(1+sqrt(N))*||Delta||_F*||X||_F). '
            'A* uses edges from Original graph where y_i==y_j only.'
        ),
    }
}

with open(str(OUT_DIR / 'theory_results.pkl'), 'wb') as f:
    pickle.dump(full_results, f)


def to_json_safe(obj):
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    return obj


with open(str(OUT_DIR / 'theory_results.json'), 'w') as f:
    json.dump(to_json_safe(full_results), f, indent=2)

logging.info(f"\nDone. Results saved to {OUT_DIR}")

# Print summary table
logging.info("\nSummary: observed error vs theoretical bound")
logging.info(f"{'Dataset':<15} {'Observed':>12} {'Bound':>12} {'Ratio':>10}")
for ds, r in theory_results.items():
    logging.info(
        f"{ds:<15} {r['observed_error']:>12.4f} {r['theoretical_bound']:>12.4f} {r['ratio']:>10.4f}"
    )
