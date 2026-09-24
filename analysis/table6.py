"""
Table VI (`tab:metrics_sggnn`) LaTeX, built only from
experiments/main_table.py's `run_merge()` output
(results/<RESULTS_DIR>/table6/results.pkl; no training).
"""
import sys
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sggnn.paths import results_dir

with open(results_dir('table6') / 'results.pkl', 'rb') as f:
    payload = pickle.load(f)

datasets = payload['datasets']
config_names = payload['config_names']
means, stds = payload['means'], payload['stds']
# Best (bold) and second-best (underline) per dataset, over all configs.
# Stable sort on -means (not argsort(...)[::-1], which flips tie order) so
# exact ties keep the earlier config's row as the winner, matching the
# manuscript's convention (e.g. SG-GCN_L before SG-FBGNN_L on Brazil, both
# 64.29 exactly).
ranked = np.argsort(-means, axis=1, kind='stable')
max_indices = ranked[:, 0]
second_indices = ranked[:, 1]

lines = [
    r"\begin{table*}[t]", r"    \centering", r"    \setlength{\tabcolsep}{2pt}",
    r"    \begin{tabular}{l|" + "c" * len(datasets) + "}", r"    \toprule",
    "     & " + " & ".join(datasets) + r" \\", r"    \midrule",
]
for ci, cname in enumerate(config_names):
    after = r"    \midrule" if cname in ('H2GCN', 'LG-GNN') else ""
    row = [cname.replace('_N', '$_N$').replace('_L', '$_L$')]
    for d in range(len(datasets)):
        mean, std = means[d, ci] * 100, stds[d, ci] * 100
        cell = f"{mean:.2f}{{\\scriptsize\\! $\\pm\\!\\!$ {std:.2f}}}"
        if max_indices[d] == ci:
            cell = r"\textbf{" + cell + "}"
        elif second_indices[d] == ci:
            cell = r"\underline{" + cell + "}"
        row.append(cell)
    lines.append(" & ".join(row) + r" \\")
    if after:
        lines.append(after)
lines += [r"    \bottomrule", r"    \end{tabular}", r"\end{table*}"]

table6_tex = "\n".join(lines)
(results_dir('table6') / 'table6.tex').write_text(table6_tex)
print(table6_tex)
print(f"\nSaved to {results_dir('table6') / 'table6.tex'}")
