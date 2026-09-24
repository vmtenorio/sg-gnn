"""
Table V (`tab:comp_times_embs`) LaTeX, built only from
experiments/graph_timing.py's saved output (no training/timing here).
"""
import sys
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sggnn.paths import results_dir

OUT_DIR = results_dir('table5')

with open(OUT_DIR / 'results.pkl', 'rb') as f:
    payload = pickle.load(f)

datasets = payload['datasets']
embedding_names = payload['embedding_names']
comp_times = payload['comp_times']
short_names = {'Feat': 'Feat', 'RoleFeat': 'Role', 'GlobalFeat': 'Global', 'DeepWalk': 'DW',
               'Node2Vec': 'N2V', 'Struc2Vec': 'S2V', 'GraphWave': 'GW'}

lines = [
    r"\begin{table}[]", r"    \centering", r"    \setlength{\tabcolsep}{5pt}",
    r"    \begin{tabular}{l" + "c" * len(embedding_names) + "}", r"    \toprule",
    " & " + " & ".join(short_names.get(n, n) for n in embedding_names) + r" \\", r"    \midrule",
]
for d, dname in enumerate(datasets):
    row = [dname] + [f"{comp_times[d, e]:.2f}" for e in range(len(embedding_names))]
    lines.append(" & ".join(row) + r" \\")
lines += [r"    \bottomrule", r"    \end{tabular}",
          r"    \caption{Computational time (in seconds) for node embedding generation "
          r"(where applicable) and subsequent $k$-NN and $\epsilon$-ball graph construction.}",
          r"    \label{tab:comp_times_embs}", r"\end{table}"]

table5_tex = "\n".join(lines)
(OUT_DIR / 'table5.tex').write_text(table5_tex)
print(table5_tex)
print(f"\nSaved to {OUT_DIR / 'table5.tex'}")
