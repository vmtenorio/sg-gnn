"""
LaTeX for Tables VIII (`tab:k_sweep`), IX (`tab:eps_sweep`), X (`tab:r_sweep`),
built only from experiments/sensitivity.py's saved artifacts (no training).

Reads results/<RESULTS_DIR>/{table8,table9,table10}/results.pkl; writes
{table8,table9,table10}.tex into those same directories.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pickle
from sggnn.paths import results_dir


def make_sweep_table(datasets, values, results, col_label_fmt, caption, label):
    """results: [n_datasets, n_sims, n_values]."""
    means = results.mean(axis=1) * 100
    stds = results.std(axis=1) * 100
    lines = [
        r"\begin{table}[h]", r"    \centering", r"    \small",
        r"    \resizebox{\columnwidth}{!}{%",
        r"    \begin{tabular}{l" + "c" * len(values) + "}",
        r"    \toprule",
        "    Dataset & " + " & ".join(col_label_fmt(v) for v in values) + r" \\",
        r"    \midrule",
    ]
    for d, dname in enumerate(datasets):
        row = [dname] + [f"{means[d, vi]:.2f}$\\pm${stds[d, vi]:.2f}" for vi in range(len(values))]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"    \bottomrule", r"    \end{tabular}}",
              r"    \caption{" + caption + "}", r"    \label{" + label + "}", r"\end{table}"]
    return "\n".join(lines)


table8_pkl = results_dir('table8') / 'results.pkl'
if table8_pkl.exists():
    with open(table8_pkl, 'rb') as f:
        table8 = pickle.load(f)
    n_splits_k = table8['metadata']['n_sims']
    tex8 = make_sweep_table(
        table8['datasets'], table8['k_values'], table8['results'],
        col_label_fmt=lambda k: f"$k={k}$",
        caption=f"$k$-sensitivity sweep: node classification accuracy (\\%) $\\pm$ std, {n_splits_k} splits.",
        label="tab:k_sweep")
    (results_dir('table8') / 'table8.tex').write_text(tex8)
    print(tex8, '\n')
else:
    print(f"Skipping table8: {table8_pkl} not found")

table9_pkl = results_dir('table9') / 'results.pkl'
if table9_pkl.exists():
    with open(table9_pkl, 'rb') as f:
        table9 = pickle.load(f)
    n_splits_eps = table9['metadata']['n_sims']
    tex9 = make_sweep_table(
        table9['datasets'], table9['eps_target_degs'], table9['results'],
        col_label_fmt=lambda p: f"deg$={p}$",
        caption=f"$\\epsilon$-ball average-degree sensitivity sweep: node classification accuracy "
                f"(\\%) $\\pm$ std, {n_splits_eps} splits.",
        label="tab:eps_sweep")
    (results_dir('table9') / 'table9.tex').write_text(tex9)
    print(tex9, '\n')
else:
    print(f"Skipping table9: {table9_pkl} not found")

table10_pkl = results_dir('table10') / 'results.pkl'
if table10_pkl.exists():
    with open(table10_pkl, 'rb') as f:
        table10 = pickle.load(f)
    n_draws, n_sims_per_draw = table10['N_DRAWS'], table10['n_sims_per_draw']
    tex10 = make_sweep_table(
        table10['datasets'], table10['R_values'], table10['results'],
        col_label_fmt=lambda R: f"$R={R}$",
        caption=(f"Ablation over the number of candidate graphs $R$, drawn as random subsets of a "
                 f"14-view pool: node classification accuracy (\\%) $\\pm$ std, "
                 f"{n_draws * n_sims_per_draw} random-draw $\\times$ split observations per $R$ value."),
        label="tab:r_sweep")
    (results_dir('table10') / 'table10.tex').write_text(tex10)
    print(tex10)
else:
    print(f"Skipping table10: {table10_pkl} not found")

print("\nDone (each table built if its artifact was present).")
