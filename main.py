#!/usr/bin/env python3
"""
Single entry point for reproducing the paper's tables and figures.

Usage:
    python main.py <target> [--quick]
    python main.py build-graphs

Each target (other than build-graphs) runs the experiment script(s) that
produce the raw results, then the analysis script that builds the LaTeX
table/figure from those saved artifacts (`analysis/` scripts never retrain
anything). Every step is a subprocess of the same interpreter running the
corresponding module under experiments/ or analysis/, so `python main.py X`
behaves exactly like running those scripts by hand in sequence -- this file
only sequences them and forwards environment-variable configuration
(RESULTS_DIR, GRAPH_CACHE, per-experiment *_DEVICE/*_NSIMS/etc., all
documented in each module's docstring).

--quick sets a handful of environment variables (fewer epochs/sims/datasets)
for a fast smoke test; it is only wired up for targets cheap enough to smoke
test without a GPU cluster (table4, fig4, table7, table8/9/10, table5).

Targets and what they need precomputed (see README "Graph cache"):
    table4      Table IV     needs embedding_graphs.npz
    table5      Table V      no cache needed (times its own graph construction)
    table6      Table VI     needs embedding_graphs.npz (main/newbaselines parts)
    fig4        Fig. 4       needs embedding_graphs.npz
    table7      Table VII    needs embedding_graphs.npz
    table8      Table VIII   no cache needed (k sweep rebuilds its own graphs)
    table9      Table IX     no cache needed (eps sweep rebuilds its own graphs)
    table10     Table X      needs embedding_graphs.npz (R sweep's 14-view pool)
    bound-real  Sec. VI-E    needs embedding_graphs.npz
    table11     Sec. VI-E    builds from bound-real's saved output (no cache)
    table12     Table XII    needs embedding_graphs.npz
    table13     tab:roman_empire_time   no cache needed (builds Roman-Empire graphs on the fly)
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# target -> (list of (module_path, extra_env) to run in order)
TARGETS = {
    'table4': [
        ('experiments/graph_eval.py', {}),
        ('analysis/table4.py', {}),
    ],
    'table5': [
        ('experiments/graph_timing.py', {}),
        ('analysis/table5.py', {}),
    ],
    'table6': [
        ('experiments/main_table.py', {'MAIN_TABLE_PART': 'main'}),
        ('experiments/main_table.py', {'MAIN_TABLE_PART': 'newbaselines'}),
        ('experiments/main_table.py', {'MAIN_TABLE_PART': 'largegraph'}),
        ('experiments/main_table.py', {'MAIN_TABLE_PART': 'merge'}),
        ('analysis/table6.py', {}),
    ],
    'fig4': [
        ('experiments/coefficients.py', {}),
    ],
    'table7': [
        ('experiments/feature_importance.py', {}),
        ('analysis/plot_feature_importance.py', {}),
    ],
    'table8': [
        ('experiments/sensitivity.py', {'SENSITIVITY_PARTS': 'k'}),
        ('analysis/sensitivity_tables.py', {}),
    ],
    'table9': [
        ('experiments/sensitivity.py', {'SENSITIVITY_PARTS': 'eps'}),
        ('analysis/sensitivity_tables.py', {}),
    ],
    'table10': [
        ('experiments/sensitivity.py', {'SENSITIVITY_PARTS': 'r'}),
        ('analysis/sensitivity_tables.py', {}),
    ],
    'bound-real': [
        ('experiments/bound_real.py', {}),
    ],
    'table11': [
        ('analysis/plot_bound_real.py', {}),
    ],
    'table12': [
        ('experiments/timing.py', {'TIMING_PARTS': 'table12'}),
        ('analysis/timing_table.py', {}),
    ],
    'table13': [
        ('experiments/timing.py', {'TIMING_PARTS': 'table13'}),
        ('analysis/plot_timing_largegraph.py', {}),
    ],
}

# Cheap enough to smoke-test with reduced settings. Values are merged into the
# environment for every step of the target.
QUICK_ENV = {
    # Wisconsin, not Texas: analysis/table4.py's published subset is
    # {Wisconsin, Cornell, Actor, Cora} -- Texas alone would make it skip everything.
    'table4': {'GRAPH_EVAL_EPOCHS': '20', 'GRAPH_EVAL_NSIMS': '1', 'GRAPH_EVAL_DATASETS': 'Wisconsin'},
    'fig4': {'COEFFICIENTS_EPOCHS': '20', 'COEFFICIENTS_NSIMS': '1', 'COEFFICIENTS_DATASETS': 'Texas'},
    'table7': {'SENS_NSIMS': '1'},
    'table8': {'SENS_EPOCHS': '20', 'SENS_NSIMS': '1', 'SENS_DATASETS': 'Texas'},
    'table9': {'SENS_EPOCHS': '20', 'SENS_NSIMS': '1', 'SENS_DATASETS': 'Texas'},
    'table10': {'SENS_EPOCHS': '20', 'SENS_N_DRAWS': '1', 'SENS_NSIMS_PER_DRAW': '1', 'SENS_DATASETS': 'Texas'},
    'table5': {'TABLE5_DATASETS': 'Texas', 'TABLE5_EMBEDDINGS': 'Feat,RoleFeat,GlobalFeat'},
}


def run_step(module_path, extra_env):
    env = dict(os.environ)
    env.update(extra_env)
    print(f"\n$ python {module_path}" + (f"  (env: {extra_env})" if extra_env else ""))
    result = subprocess.run([sys.executable, str(ROOT / module_path)], cwd=str(ROOT), env=env)
    if result.returncode != 0:
        raise SystemExit(f"'{module_path}' exited with code {result.returncode}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('target', choices=list(TARGETS.keys()) + ['build-graphs'])
    ap.add_argument('--quick', action='store_true', help='Reduced settings for a fast smoke test (see module docstring).')
    args = ap.parse_args()

    if args.target == 'build-graphs':
        sys.path.insert(0, str(ROOT))
        from sggnn.embeddings import build_cache
        out_path = os.environ.get('GRAPH_CACHE', str(ROOT / 'node_embeddings' / 'embedding_graphs.npz'))
        build_cache(out_path)
        return

    quick_env = QUICK_ENV.get(args.target, {}) if args.quick else {}
    if args.quick and args.target not in QUICK_ENV:
        print(f"Note: no --quick settings defined for '{args.target}'; running at default settings.")

    for module_path, extra_env in TARGETS[args.target]:
        merged_env = {**quick_env, **extra_env}
        run_step(module_path, merged_env)


if __name__ == '__main__':
    main()
