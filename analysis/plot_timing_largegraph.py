"""
Analysis for experiments/timing.py's Table XIII part (Roman-Empire).

Consumes only the saved artifacts (results/<RESULTS_DIR>/table13/results.pkl)
-- no training code here, so this can be re-run any time to regenerate the
summary and LaTeX fragment without re-timing anything.

Writes, into the same results directory:
  summary.txt                 -- human-readable table + ordering comparison
  table_timing_largegraph.tex -- LaTeX fragment for Table XIII
                                 (tab:roman_empire_time)
"""

import sys
import pickle
import logging
from pathlib import Path

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sggnn.paths import results_dir
from sggnn.report import latex_thousands
RESULTS_DIR = results_dir('table13')
PKL_PATH = RESULTS_DIR / 'results.pkl'

MODEL_ORDER = ['GCN', 'GAT', 'GRAND', 'CD-GNN', 'LG-GNN', 'SG-GCN', 'SG-FBGNN']


def main():
    with open(PKL_PATH, 'rb') as f:
        full_results = pickle.load(f)

    results = full_results['results']
    meta = full_results['metadata']

    rows = []  # (name, total_s_mean, total_s_std, epochs_mean, ms_epoch_mean,
               #  ms_epoch_std, mem_mean, params, acc_mean, acc_std)
    errors = []

    for name in MODEL_ORDER:
        r = results.get(name)
        if r is None:
            errors.append((name, 'missing from results'))
            continue
        if 'error' in r:
            errors.append((name, r['error']))
            continue
        rows.append((
            name,
            r['total_time_sec_mean'], r['total_time_sec_std'],
            r['epochs_run_mean'],
            r['time_per_epoch_mean'] * 1000, r['time_per_epoch_std'] * 1000,
            r['peak_memory_mb_mean'], r['n_params'],
            r['best_test_acc_mean'] * 100, r['best_test_acc_std'] * 100,
        ))

    # ── orderings ────────────────────────────────────────────────────────
    by_total_time = sorted(rows, key=lambda x: x[1])
    by_ms_epoch = sorted(rows, key=lambda x: x[4])
    order_total = [x[0] for x in by_total_time]
    order_ms = [x[0] for x in by_ms_epoch]
    orderings_match = order_total == order_ms

    # ── SG-GCN vs GCN / CD-GNN / LG-GNN, total time ─────────────────────
    lookup = {x[0]: x for x in rows}
    sg_gcn_comparisons = []
    if 'SG-GCN' in lookup:
        sg_total = lookup['SG-GCN'][1]
        for other in ['GCN', 'CD-GNN', 'LG-GNN']:
            if other in lookup:
                other_total = lookup[other][1]
                ratio = sg_total / other_total
                cheaper = 'cheaper than' if sg_total < other_total else 'more expensive than'
                sg_gcn_comparisons.append(
                    f"SG-GCN total time = {sg_total:.2f}s is {ratio:.2f}x {other}'s "
                    f"({other_total:.2f}s) -> SG-GCN is {cheaper} {other} in total wall-clock time."
                )

    # ── write summary.txt ────────────────────────────────────────────────
    lines = []
    lines.append("Roman-Empire total wall-clock training time (to convergence)")
    lines.append("=" * 70)
    lines.append(
        f"hid_dim={meta['hid_dim']}, n_layers={meta['n_layers']}, "
        f"dropout={meta['dropout']}, lr={meta['lr']}, wd={meta['wd']}, "
        f"epochs<={meta['epochs']}, patience={meta['patience']}, "
        f"n_sims={meta['n_sims']}, seed={meta['seed']}"
    )
    lines.append(f"commit={meta['commit_hash']}, torch={meta['torch_version']}, "
                 f"device={meta['device_name']}, timestamp={meta['timestamp']}")
    lines.append("")
    header = (f"{'Model':12s} {'Total(s)':>12s} {'Epochs':>8s} {'ms/epoch':>10s} "
              f"{'PeakMem(MB)':>12s} {'Params':>10s} {'Acc(%)':>10s}")
    lines.append(header)
    lines.append("-" * len(header))
    for (name, tot_m, tot_s, ep_m, ms_m, ms_s, mem_m, params, acc_m, acc_s) in rows:
        lines.append(
            f"{name:12s} {tot_m:9.2f}+-{tot_s:<5.2f} {ep_m:8.1f} "
            f"{ms_m:7.2f}+-{ms_s:<4.2f} {mem_m:12.1f} {params:10d} "
            f"{acc_m:6.2f}+-{acc_s:<3.2f}"
        )
    lines.append("")

    if errors:
        lines.append("Models that errored / OOM'd (excluded above):")
        for name, err in errors:
            lines.append(f"  {name}: {err}")
        lines.append("")

    lines.append("Ordering by total training time: " + " < ".join(order_total))
    lines.append("Ordering by ms/epoch:            " + " < ".join(order_ms))
    lines.append(f"Orderings match: {orderings_match}")
    if not orderings_match:
        lines.append(
            "-> The two orderings disagree: at least one model converges in a "
            "different number of epochs than its ms/epoch rank would suggest, "
            "so ranking models by ms/epoch is not a safe proxy for ranking them "
            "by total training cost."
        )
    lines.append("")

    lines.append("SG-GCN vs. GCN / CD-GNN / LG-GNN (total wall-clock time):")
    for line in sg_gcn_comparisons:
        lines.append("  " + line)

    summary_path = RESULTS_DIR / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    logging.info(f"Wrote {summary_path}")

    # ── write LaTeX table fragment ───────────────────────────────────────
    tex_lines = []
    tex_lines.append(r"\begin{table}[h]")
    tex_lines.append(r"    \centering")
    tex_lines.append(r"    \small")
    tex_lines.append(r"    \begin{tabular}{lrrrrrr}")
    tex_lines.append(r"    \toprule")
    tex_lines.append(
        r"    Model & Total time (s) & Epochs & ms/epoch & Peak mem (MB) & Params & Acc.\ (\%) \\"
    )
    tex_lines.append(r"    \midrule")
    for (name, tot_m, tot_s, ep_m, ms_m, ms_s, mem_m, params, acc_m, acc_s) in rows:
        tex_lines.append(
            f"    {name} & {tot_m:.2f} & {ep_m:.0f} & {ms_m:.2f} & {mem_m:.1f} & "
            f"{latex_thousands(params)} & {acc_m:.2f} \\\\"
        )
    tex_lines.append(r"    \bottomrule")
    tex_lines.append(r"    \end{tabular}")
    tex_lines.append(
        r"\caption{Total wall-clock training time to convergence on Roman-Empire "
        r"(project-standard early stopping, patience 300, epochs $\leq$ 2000, "
        r"3 repeats, hidden dimension 32), with derived ms/epoch, peak memory, "
        r"parameter count, and best test accuracy.} \label{tab:roman_empire_time_total}"
    )
    tex_lines.append(r"\end{table}")

    tex_path = RESULTS_DIR / 'table_timing_largegraph.tex'
    with open(tex_path, 'w') as f:
        f.write("\n".join(tex_lines) + "\n")
    logging.info(f"Wrote {tex_path}")


if __name__ == '__main__':
    main()
