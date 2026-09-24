"""
Table XII (`tab:profiling_small`) LaTeX, built only from experiments/timing.py's
saved output (results/<RESULTS_DIR>/table12/results.pkl; no training).

Output: results/<RESULTS_DIR>/table12/table12.tex
"""
import pickle
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sggnn.paths import results_dir
from sggnn.report import latex_thousands
IN_PKL = results_dir('table12') / 'results.pkl'
OUT_DIR = IN_PKL.parent

with open(IN_PKL, 'rb') as f:
    payload = pickle.load(f)

results = payload['results']
row_order = ['GCN', 'FBGNN', 'AdaptiveAggGCN-GCN', 'AdaptiveAggGCN-FBGNN']
display_name = {'GCN': 'GCN', 'FBGNN': 'FBGNN', 'AdaptiveAggGCN-GCN': 'SG-GCN',
                 'AdaptiveAggGCN-FBGNN': 'SG-FBGNN'}

lines = [
    r"\begin{table}[h]",
    r"    \centering",
    r"    \small",
    r"    \color{blue}",
    r"    \resizebox{\columnwidth}{!}{%",
    r"    \begin{tabular}{llrrrrr}",
    r"    \toprule",
    r"    Dataset & Model & Total (s) & Epochs & ms/epoch & Peak mem (MB) & Acc.\ (\%) \\",
    r"    \midrule",
]
for dataset in results:
    for model in row_order:
        r = results[dataset][model]
        total_s = r['total_time_sec_mean']
        epochs = int(round(r['epochs_run_mean']))
        ms_epoch = r['time_per_epoch_mean'] * 1000
        mem = r['peak_memory_mb_mean']
        mem_str = latex_thousands(mem, 1)
        acc = r['best_test_acc_mean'] * 100
        lines.append(f"    {dataset} & {display_name[model]} & {total_s:.2f} & {epochs} & "
                      f"{ms_epoch:.2f} & {mem_str} & {acc:.2f} \\\\")
lines += [
    r"    \bottomrule",
    r"    \end{tabular}}",
    r"    \caption{\blue{End-to-end training time to convergence, epochs run, derived per-epoch "
    r"time, peak memory and test accuracy on three representative datasets.}}",
    r"    \label{tab:profiling_small}",
    r"\end{table}",
]

tex = "\n".join(lines)
(OUT_DIR / 'table12.tex').write_text(tex)
print(tex)
print(f"\nSaved to {OUT_DIR / 'table12.tex'}")
