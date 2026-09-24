"""
Analysis for experiments/feature_importance.py (Table VII).

Consumes only results/<RESULTS_DIR>/table7/results.json -- no training.
Produces, in the same directory: summary.txt (accuracy tables, paired tests
against `all`, Holm-Bonferroni survivors) and table7.tex (LaTeX fragment,
same column layout as Table VII, tab:feat_ablation).
"""

import sys
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import stats

logging.basicConfig(format='%(message)s', level=logging.INFO)

from sggnn.paths import results_dir
from sggnn.report import Summary
RES_DIR = results_dir('table7')
with open(RES_DIR / 'results.json') as f:
    payload = json.load(f)

results = np.array(payload['results'])  # [n_datasets, n_sims, n_conditions]
datasets = payload['metadata']['datasets']
condition_names = payload['condition_names']
meta = payload['metadata']

means = results.mean(axis=1) * 100  # [n_datasets, n_conditions]
stds  = results.std(axis=1) * 100

DATASET_ABBR = {
    'Texas': 'Texas', 'Wisconsin': 'Wisc.', 'Cornell': 'Cornell',
    'Actor': 'Actor', 'Chameleon': 'Cham.', 'Squirrel': 'Squirrel',
    'Cora': 'Cora', 'CiteSeer': 'CiteSeer', 'USA': 'USA',
    'Europe': 'Europe', 'Brazil': 'Brazil',
}
CONDITION_LABEL = {
    'all': 'all', 'no_role': 'no-role', 'no_global': 'no-global',
    'no_feat': 'no-feat', 'role_only': 'role-only', 'global_only': 'global-only',
    'structural_only': 'structural-only', 'full': 'full',
}

# SG-GCN and SG-FBGNN rows of Table VI (tab:metrics_sggnn, 20 splits), for
# comparison with the `full` condition. The ablation model is SG-FBGNN.
SGGNN_MAIN_TABLE = {
    'SG-GCN': {
        'Texas': 83.42, 'Wisconsin': 83.85, 'Cornell': 81.32, 'Actor': 36.74,
        'Chameleon': 66.42, 'Squirrel': 58.48, 'Cora': 87.04, 'CiteSeer': 77.56,
        'USA': 63.45, 'Europe': 45.73, 'Brazil': 61.79,
    },
    'SG-FBGNN': {
        'Texas': 82.63, 'Wisconsin': 83.27, 'Cornell': 78.16, 'Actor': 35.68,
        'Chameleon': 67.49, 'Squirrel': 51.64, 'Cora': 87.15, 'CiteSeer': 76.59,
        'USA': 51.18, 'Europe': 47.68, 'Brazil': 58.21,
    },
}

out = Summary()
emit = out.emit

emit("=" * 100)
emit(f"importance_v2  |  n_sims={meta['n_sims']}  |  {meta['timestamp'][:19]}  |  commit={meta.get('commit_hash', 'unknown')}")
emit(f"datasets: {datasets}")
emit(f"conditions: {condition_names}")
emit("=" * 100)
emit()

# ── mean-only table ───────────────────────────────────────────────────────────
header = f"{'Condition':<18}" + "".join(f"{DATASET_ABBR[d]:>10}" for d in datasets)
emit("Mean accuracy (%)")
emit(header)
for c, cname in enumerate(condition_names):
    row = f"{CONDITION_LABEL[cname]:<18}" + "".join(f"{means[d, c]:>10.2f}" for d in range(len(datasets)))
    emit(row)
emit()

# ── mean +/- std table ────────────────────────────────────────────────────────
emit("Mean +/- std accuracy (%)")
emit(header)
for c, cname in enumerate(condition_names):
    row = f"{CONDITION_LABEL[cname]:<18}"
    for d in range(len(datasets)):
        row += f"{means[d, c]:>7.2f}+/-{stds[d, c]:<5.2f}"
    emit(row)
emit()

# ── comparison: full vs all, and full vs SG-GCN / SG-FBGNN main-table rows ───
full_idx = condition_names.index('full')
all_idx  = condition_names.index('all')

emit("=" * 100)
emit("full vs. all (this ablation, 10 splits)  |  full vs. main-table SG-GCN / SG-FBGNN rows (20 splits)")
emit("=" * 100)
emit(f"{'Dataset':<12}{'full':>8}{'all':>8}{'full-all':>10}{'SG-GCN(20s)':>13}{'full-SGGCN':>12}{'SG-FBGNN(20s)':>15}{'full-SGFB':>11}")
for d, dname in enumerate(datasets):
    full_v = means[d, full_idx]
    all_v  = means[d, all_idx]
    sggcn  = SGGNN_MAIN_TABLE['SG-GCN'][dname]
    sgfb   = SGGNN_MAIN_TABLE['SG-FBGNN'][dname]
    emit(
        f"{dname:<12}{full_v:>8.2f}{all_v:>8.2f}{full_v - all_v:>10.2f}"
        f"{sggcn:>13.2f}{full_v - sggcn:>12.2f}{sgfb:>15.2f}{full_v - sgfb:>11.2f}"
    )

# ── paired comparison of every condition against `all` ───────────────────────
# All conditions share one split set, so each pair is compared per split. This
# removes the split-to-split difficulty variation common to both conditions,
# the dominant source of spread here (per-split std reaches 24 points on Brazil).
def _pval(t, df):
    return float(2 * stats.t.sf(abs(t), df))

n_splits = results.shape[1]
df = n_splits - 1
comparisons = []

emit()
emit("=" * 104)
emit(f"Paired comparison against `all`, per dataset ({n_splits} shared splits, paired t on per-split differences)")
emit("Delta = condition - all, in accuracy points. 'unpaired t' is what an independent-means test would have given.")
emit("=" * 104)
emit(f"{'Dataset':<12}{'Condition':<18}{'Delta':>8}{'sd(Delta)':>11}{'paired t':>10}{'p':>10}{'unpaired t':>12}")
for d, dname in enumerate(datasets):
    for c, cname in enumerate(condition_names):
        if cname == 'all':
            continue
        diff = (results[d, :, c] - results[d, :, all_idx]) * 100
        md, sd = float(diff.mean()), float(diff.std(ddof=1))
        t_paired = md / (sd / np.sqrt(n_splits)) if sd > 0 else float('nan')
        a, b = results[d, :, c] * 100, results[d, :, all_idx] * 100
        se_unpaired = np.sqrt(a.var(ddof=1) / n_splits + b.var(ddof=1) / n_splits)
        t_unpaired = md / se_unpaired if se_unpaired > 0 else float('nan')
        p = _pval(t_paired, df) if sd > 0 else float('nan')
        comparisons.append((dname, cname, md, p))
        emit(f"{dname:<12}{CONDITION_LABEL[cname]:<18}{md:>8.2f}{sd:>11.2f}"
             f"{t_paired:>10.2f}{p:>10.4f}{t_unpaired:>12.2f}")

# ── multiple-comparison control ──────────────────────────────────────────────
# One test per (dataset, condition) pair. Reporting the raw minimum p-value over
# this many tests would overstate the evidence, so we apply Holm-Bonferroni and
# report which effects survive it.
ordered = sorted((c for c in comparisons if c[3] == c[3]), key=lambda r: r[3])
m = len(ordered)
n_skipped = len(comparisons) - m
emit()
emit("=" * 104)
emit(f"Holm-Bonferroni over {m} paired tests (family-wise alpha = 0.05)"
     + (f"; {n_skipped} of {len(comparisons)} skipped as degenerate (zero variance)" if n_skipped else ""))
emit("=" * 104)
survivors, rejected = [], False
for i, (dname, cname, md, p) in enumerate(ordered):
    thresh = 0.05 / (m - i)
    if not rejected and p <= thresh:
        survivors.append((dname, cname, md, p, thresh))
    else:
        rejected = True
if survivors:
    emit(f"{'Dataset':<12}{'Condition':<18}{'Delta':>8}{'p':>12}{'Holm thresh':>14}")
    for dname, cname, md, p, thresh in survivors:
        emit(f"{dname:<12}{CONDITION_LABEL[cname]:<18}{md:>8.2f}{p:>12.2e}{thresh:>14.2e}")
else:
    emit("No comparison survives Holm-Bonferroni correction.")
emit()
emit(f"{len(survivors)} of {m} comparisons survive. Claims in the manuscript should be limited to these.")

with open(RES_DIR / 'summary.txt', 'w') as f:
    f.write('\n'.join(out) + '\n')
logging.info(f"\nWrote {RES_DIR / 'summary.txt'}")

# ── LaTeX table fragment (same layout as tab:feat_ablation) ──────────────────
tex = []
tex.append(r"\begin{table}[h]")
tex.append(r"    \centering")
tex.append(r"    \small")
tex.append(r"    \resizebox{\columnwidth}{!}{%")
tex.append(r"    \begin{tabular}{l" + "c" * len(datasets) + "}")
tex.append(r"    \toprule")
tex.append(
    r"    Condition & " + " & ".join(DATASET_ABBR[d] for d in datasets) + r" \\"
)
tex.append(r"    \midrule")
for c, cname in enumerate(condition_names):
    label = CONDITION_LABEL[cname]
    row = [label] + [f"{means[d, c]:.2f}" for d in range(len(datasets))]
    tex.append("    " + " & ".join(row) + r" \\")
tex.append(r"    \bottomrule")
tex.append(r"    \end{tabular}}")
tex.append(
    r"\caption{Leave-one-group-out feature ablation: node classification accuracy (\%), "
    r"10 splits, SG-FBGNN (single-layer, hidden dimension 32). \emph{full} uses all "
    fr"$R={len(payload['conditions']['full'])}$ graph views, matching the graph set used for "
    r"SG-GNN in the main results table (\Cref{tab:metrics_sggnn}).} \label{tab:feat_ablation_v2}"
)
tex.append(r"\end{table}")

with open(RES_DIR / 'table7.tex', 'w') as f:
    f.write('\n'.join(tex) + '\n')
logging.info(f"Wrote {RES_DIR / 'table7.tex'}")
