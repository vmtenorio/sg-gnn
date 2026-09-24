"""
Table IV (`tab:metrics_graphs_embeddings`) LaTeX, built only from
experiments/graph_eval.py's saved artifact (results/<id>/table4/results.pkl
by default, or TABLE4_SOURCE env var for a legacy `results-Exp1` pickle).
Scales std to accuracy points with the same factor (100) as the mean --
KNNHeterophilic's AnalysisResults.ipynb notebook used factor 10 for std,
which under-reported every error bar by 10x; see make_table4_fixed.py in
that repo for the original bug report.
"""
import os
import sys
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sggnn.paths import results_dir

SOURCE = os.environ.get('TABLE4_SOURCE', str(results_dir('table4') / 'results.pkl'))
OUT_DIR = results_dir('table4')

with open(SOURCE, 'rb') as f:
    payload = pickle.load(f)

best_accs_test = payload['best_accs_test']  # [n_datasets, n_sims, n_gcn, n_exps]
exps = payload['exps']
gcnlist = payload['gcnlist']
datasets = payload['datasets']

exps_no_ours = [exp['leg'] for exp in exps if ('Adaptive' not in exp['leg'] and exp['leg'] != 'SG-GNN')]

g_mapper = {
    'Original': r'$\mathcal{G}$',
    'EPS-Feat': r'$\ccalG^{\rm ball}_{\epsilon, {\rm feat}}$',
    'EPS-RoleFeat': r'$\ccalG^{\rm ball}_{\epsilon, {\rm role}}$',
    'EPS-GlobalFeat': r'$\ccalG^{\rm ball}_{\epsilon, {\rm glob}}$',
    'EPS-DeepWalk': r'$\ccalG^{\rm ball}_{\epsilon, {\rm DW}}$',
    'EPS-Node2Vec': r'$\ccalG^{\rm ball}_{\epsilon, {\rm N2V}}$',
    'EPS-Struc2Vec': r'$\ccalG^{\rm ball}_{\epsilon, {\rm S2V}}$',
    'KNN-Feat': r'$\ccalG^{\rm nn}_{k, {\rm feat}}$',
    'KNN-RoleFeat': r'$\ccalG^{\rm nn}_{k, {\rm role}}$',
    'KNN-GlobalFeat': r'$\ccalG^{\rm nn}_{k, {\rm glob}}$',
    'KNN-DeepWalk': r'$\ccalG^{\rm nn}_{k, {\rm DW}}$',
    'KNN-Node2Vec': r'$\ccalG^{\rm nn}_{k, {\rm N2V}}$',
    'KNN-Struc2Vec': r'$\ccalG^{\rm nn}_{k, {\rm S2V}}$',
    'KNN-GraphWave': r'$\ccalG^{\rm nn}_{k, {\rm GW}}$',
}

datasets_include = [d for d in ['Wisconsin', 'Cornell', 'Actor', 'Cora'] if d in datasets]
if len(datasets_include) < 4:
    print(f"Note: source only has {datasets}; restricting Table IV's usual "
          f"{{Wisconsin, Cornell, Actor, Cora}} subset to {datasets_include} (e.g. --quick runs).")
idxs = [datasets.index(d) for d in datasets_include]
g_gcn = gcnlist.index('GCNConv')
g_fbgnn = gcnlist.index('FBGNNLayer')


def highlight_best_second_best(row):
    sorted_indices = row.argsort().values[::-1]
    factor = 100.
    best_idx = sorted_indices[0]
    second_best_idx = sorted_indices[1]
    row = row * factor
    formatted_row = row.copy().round(2).astype(str)
    formatted_row.iloc[best_idx] = f"\\textbf{{{row.iloc[best_idx]:.2f}}}"
    formatted_row.iloc[second_best_idx] = f"\\underline{{{row.iloc[second_best_idx]:.2f}}}"
    return formatted_row


def _table(stat):
    return pd.concat([
        pd.DataFrame(getattr(np.asarray(best_accs_test)[idxs, :, g_gcn, :-3], stat)(1).T,
                     index=[g_mapper[name] for name in exps_no_ours],
                     columns=pd.MultiIndex.from_product([['GCN'], datasets_include])),
        pd.DataFrame(getattr(np.asarray(best_accs_test)[idxs, :, g_fbgnn, :-3], stat)(1).T,
                     index=[g_mapper[name] for name in exps_no_ours],
                     columns=pd.MultiIndex.from_product([['FBGNN'], datasets_include])),
    ], axis=1)


df = _table('mean')
df_std = (_table('std') * 100).round(2).astype(str)

df_formatted = df.apply(highlight_best_second_best, axis=0)
cells = df_formatted + ' $\\pm$ ' + df_std


def rewrap(cell):
    # Wraps the whole "mean $\pm$ std" cell in \textbf{}/\underline{}, not
    # just the mean (pandas' raw to_latex() only bolds the mean).
    for tag in ('\\textbf{', '\\underline{'):
        if cell.startswith(tag):
            inner = cell[len(tag):]
            mean_str, rest = inner.split('}', 1)
            return tag + mean_str + rest + '}'
    return cell


cells = cells.map(rewrap)
latex_table4 = cells.to_latex()

(OUT_DIR / 'table4.tex').write_text(latex_table4)
(OUT_DIR / 'table4_summary.json').write_text(json.dumps({
    'source': SOURCE, 'datasets_include': datasets_include,
    'means_points': json.loads((df * 100).round(2).to_json()),
    'std_points': json.loads(df_std.to_json()),
}, indent=2))

print(latex_table4)
print(f"\nSaved to {OUT_DIR}")
