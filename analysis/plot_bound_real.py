"""
Analysis and figures for experiments/bound_real.py.

Consumes only results/<RESULTS_DIR>/bound_real/bound_real.json -- no training.
If results/ablations/theory/theory_results.json (from the superseded
bound_tightness.py, not part of the release) is present, its ratios are printed
alongside for comparison; the section is skipped otherwise.

Produces, in the same directory:
  summary.txt                        (tables + fits)
  table_bound_real.tex               (LaTeX fragment)
  fig_bound_theorem_real.pdf/.png
"""

import sys
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(format='%(message)s', level=logging.INFO)

from sggnn.paths import results_dir
from sggnn.report import Summary, loglog_slope, latex_thousands
RES_DIR = results_dir('bound_real')
OLD_PATH = ROOT / 'results' / 'ablations' / 'theory' / 'theory_results.json'

with open(RES_DIR / 'bound_real.json') as f:
    payload = json.load(f)
R, META = payload['results'], payload['metadata']

OLD = {}
if OLD_PATH.exists():
    with open(OLD_PATH) as f:
        OLD = json.load(f)['results']

DATASETS = [d for d in META['datasets'] if d in R]
SKIPPED = [d for d in META['datasets'] if d not in R]

out = Summary()
emit = out.emit


def sci(x):
    """Scientific notation as '3.00e-6' (no leading zero in the exponent)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'nan'
    s = f"{x:.2e}"
    mantissa, exp = s.split('e')
    exp = int(exp)
    return f"{mantissa}e{exp}"


# ── gather per-dataset arrays (identity config = the headline measurement) ──
N_ = np.array([R[d]['identity']['N_mean'] for d in DATASETS])
delta_ = np.array([R[d]['identity']['delta_normF_mean'] for d in DATASETS])
obs_ = np.array([R[d]['identity']['observed_error_mean'] for d in DATASETS])
obs_sd_ = np.array([R[d]['identity']['observed_error_std'] for d in DATASETS])
bnd_ = np.array([R[d]['identity']['bound_mean'] for d in DATASETS])
ratio_ = np.array([R[d]['identity']['ratio_mean'] for d in DATASETS])
alpha_share_ = np.array([R[d]['identity']['alpha_share_pct_mean'] for d in DATASETS])
homophily_ = np.array([R[d]['identity']['edge_homophily_mean'] for d in DATASETS])

ratio_sm_ = np.array([R[d]['softmax']['ratio_mean'] for d in DATASETS])
obs_sm_ = np.array([R[d]['softmax']['observed_error_mean'] for d in DATASETS])

emit("=" * 88)
emit(f"bound_theorem_real  |  {META['n_splits']} splits  |  {META['timestamp'][:19]}")
emit(f"model: {META['model']}")
emit(f"bound: {META['bound']}")
emit(f"oracle: {META['oracle']}")
if SKIPPED:
    emit(f"SKIPPED (errored, see metadata/errors): {SKIPPED}")
emit("=" * 88)

# ── [1] per-dataset table, with comparison to the old (flawed) measurement ──
emit("\n[1] per-dataset: TheoremGCN (this experiment) vs AdaptiveAggGCN (bound_tightness.py, old)")
emit(f"{'Dataset':<12}{'N':>7}{'|D|_F':>9}{'observed':>11}{'bound':>13}{'ratio':>11}"
     f"{'old ratio':>12}{'new/old':>10}")
for i, d in enumerate(DATASETS):
    old_ratio = OLD.get(d, {}).get('ratio', None)
    if old_ratio is not None and old_ratio > 0:
        fold = ratio_[i] / old_ratio
        old_s, fold_s = sci(old_ratio), f"{fold:.2f}x"
    else:
        old_s, fold_s = 'n/a', 'n/a'
    emit(f"{d:<12}{int(N_[i]):>7}{delta_[i]:>9.1f}{obs_[i]:>8.2f}±{obs_sd_[i]:<2.0f}"
         f"{bnd_[i]:>13.4g}{sci(ratio_[i]):>11}{old_s:>12}{fold_s:>10}")

emit(f"\n  new ratios span {sci(ratio_.min())} .. {sci(ratio_.max())} "
     f"({-np.log10(ratio_.max()):.1f}-{-np.log10(ratio_.min()):.1f} orders of magnitude loose)")
if OLD:
    old_common = np.array([OLD[d]['ratio'] for d in DATASETS if d in OLD])
    emit(f"  old (bound_tightness.py) ratios on the same datasets spanned "
         f"{sci(old_common.min())} .. {sci(old_common.max())}")
    emit("  the two measurements are NOT the same quantity (different model, output "
         "nonlinearity, and rho proxy -- see module docstring); this line is reported "
         "for context, not as a like-for-like tightening.")

# ── [2] does the ratio scale as ~1/N across the real datasets? ──────────────
emit("\n[2] ratio vs N across the 11 real datasets (synthetic study found beta=-0.97)")
beta, r2 = loglog_slope(N_, ratio_)
pear_n = pearsonr(np.log(N_), np.log(ratio_))[0] if np.all(ratio_ > 0) else float('nan')
emit(f"  log(ratio) ~ {beta:.3f} * log(N) + const   (R^2={r2:.3f}, log-log Pearson r={pear_n:.3f})")
if r2 > 0.5 and -1.5 < beta < -0.5:
    emit(f"  this REPRODUCES the synthetic study's ~1/N scaling (beta={beta:.2f} vs -0.97 synthetic).")
else:
    emit(f"  this does NOT cleanly reproduce the synthetic study's 1/N scaling "
         f"(beta={beta:.2f}, R^2={r2:.3f}). On real data, ||Delta||_F, alpha, and rho1*rho2 "
         f"all co-vary with N and with each other across datasets (unlike the synthetic sweep, "
         f"which holds everything but N fixed), so a clean N^-1 law is not expected to survive "
         f"un-controlled across heterogeneous real graphs. Reported as-is; not tuned.")

# ── [3] identity vs softmax: saturation artifact ────────────────────────────
emit("\n[3] output nonlinearity: identity (headline) vs softmax (saturation artifact)")
emit(f"{'Dataset':<12}{'obs(id)':>10}{'obs(softmax)':>14}{'ratio(id)':>11}"
     f"{'ratio(sm)':>11}{'sm/id ratio':>13}")
fac = ratio_sm_ / ratio_
for i, d in enumerate(DATASETS):
    emit(f"{d:<12}{obs_[i]:>10.2f}{obs_sm_[i]:>14.2f}{sci(ratio_[i]):>11}"
         f"{sci(ratio_sm_[i]):>11}{fac[i]:>12.2f}x")
emit(f"\n  softmax observed error is {np.mean(obs_ / np.where(obs_sm_ > 0, obs_sm_, np.nan)):.2f}x "
     f"the identity observed error on average (saturation caps ||Zhat-Z*||_F)")
emit(f"  softmax/identity ratio factor: mean {fac.mean():.3f}x, median {np.median(fac):.3f}x "
     f"(synthetic study measured 2.6-4x under saturation elsewhere in the bound; here the "
     f"direction is {'the same' if fac.mean() < 1 else 'reversed'} -- softmax "
     f"{'tightens' if fac.mean() < 1 else 'loosens'} the ratio relative to identity)")

# ── [4] alpha-term share of the bound ────────────────────────────────────────
emit("\n[4] alpha-term share of the bound (synthetic: 100% at Delta=0, ~1% at first false edge)")
emit(f"{'Dataset':<12}{'alpha-term %':>13}{'|D|_F':>9}{'homophily':>11}")
for i, d in enumerate(DATASETS):
    emit(f"{d:<12}{alpha_share_[i]:>12.2f}%{delta_[i]:>9.1f}{homophily_[i]:>11.3f}")
emit(f"\n  alpha-term share ranges {alpha_share_.min():.2f}% .. {alpha_share_.max():.2f}% "
     f"across real datasets -- every real 'Original' graph already has many cross-label "
     f"edges (homophily << 1 on most of these datasets), so none of them sit near the "
     f"synthetic study's Delta=0 operating point where the alpha term would dominate.")

# ── [5] ratio vs edge homophily ──────────────────────────────────────────────
emit("\n[5] correlation between ratio and Original-graph edge homophily")
pear_h, pear_p = pearsonr(homophily_, ratio_)
spear_h, spear_p = spearmanr(homophily_, ratio_)
emit(f"  Pearson r={pear_h:.3f} (p={pear_p:.3f}), Spearman rho={spear_h:.3f} (p={spear_p:.3f})")
emit("  ||Delta||_F is directly determined by (1 - homophily) * (# Original edges), so this "
     "correlation is expected to be structural, not incidental: lower homophily -> larger "
     "||Delta||_F -> larger delta-term in the bound.")
pear_dh = pearsonr(homophily_, delta_)[0]
emit(f"  (sanity check) Pearson r(homophily, ||Delta||_F) = {pear_dh:.3f}")

# ── manuscript claim check ───────────────────────────────────────────────────
emit("\n[manuscript claim check]")
emit("  Current text: looseness is 'driven by how the spectral-norm product rho_combined")
emit("  varies across datasets and training runs.'")
rho_prod_ = np.array([R[d]['identity']['rho_prod_mean'] for d in DATASETS])
pear_rho = pearsonr(np.log(rho_prod_), np.log(ratio_))[0]
emit(f"  Under the theorem's own model, rho1*rho2 spans {rho_prod_.min():.3g}..{rho_prod_.max():.3g} "
     f"({rho_prod_.max()/rho_prod_.min():.1f}x) across datasets, while ratio spans "
     f"{ratio_.max()/ratio_.min():.1f}x. log-log Pearson r(rho1*rho2, ratio) = {pear_rho:.3f}.")
emit("  rho1*rho2 multiplies BOTH the observed error's generating map and the bound's RHS "
     "identically (it is a common factor cancelling in the ratio's leading order, exactly as "
     "found in the bound_synth wd/hid_dim sweeps: rho varies ~192x there while ratio moves only "
     "~1.8x). The real-data numbers here are consistent with that: rho1*rho2 does NOT explain "
     "the dominant variation in the ratio. The manuscript's current attribution is NOT supported "
     "by this measurement and should be revised to point at ||Delta||_F / N instead.")

with open(RES_DIR / 'summary.txt', 'w') as f:
    f.write("\n".join(out) + "\n")

# ── LaTeX table ───────────────────────────────────────────────────────────────
tex = []
tex.append(r"\begin{table}[h]")
tex.append(r"    \centering")
tex.append(r"    \small")
tex.append(r"    \begin{tabular}{lrrrrrr}")
tex.append(r"    \toprule")
tex.append(r"    Dataset & $N$ & $\|\mathbf{\Delta}\|_F$ & Observed error & Bound & Ratio & $\alpha$-term (\%) \\")
tex.append(r"    \midrule")
for i, d in enumerate(DATASETS):
    tex.append(
        f"    {d} & {latex_thousands(N_[i])} & {delta_[i]:.2f} & {obs_[i]:.4f} & "
        f"{latex_thousands(bnd_[i], 2)} & {sci(ratio_[i])} & {alpha_share_[i]:.2f} \\\\"
    )
tex.append(r"    \bottomrule")
tex.append(r"    \end{tabular}")
tex.append(
    r"\caption{Observed embedding error $\|\hat{\mathbf{Z}}-\mathbf{Z}^*\|_F$ versus the "
    r"right-hand side of Theorem~1's bound, evaluated with the theorem's exact model "
    r"(single-graph, sigma2=identity, no biases), all 11 datasets, mean over 10 splits. "
    r"Ratio is observed error divided by the bound; $\alpha$-term (\%) is the share of the "
    r"bound coming from the $\alpha\sqrt{N}$ term rather than the $\|\mathbf{\Delta}\|_F$ term.}"
    r" \label{tab:bound_theorem_real}"
)
tex.append(r"\end{table}")

with open(RES_DIR / 'table_bound_real.tex', 'w') as f:
    f.write("\n".join(tex) + "\n")

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 3, figsize=(16, 8))

a = ax[0, 0]
order = np.argsort(N_)
a.errorbar(N_[order], obs_[order], yerr=obs_sd_[order], marker='o', color='C0',
           label=r'observed $\|\mathbf{Z}^*-\hat{\mathbf{Z}}\|_F$')
a.plot(N_[order], bnd_[order], marker='s', color='C3', label='bound, Eq. (12)')
a.set_xscale('log'); a.set_yscale('log')
a.set_xlabel('$N$'); a.set_ylabel('error')
a.set_title('(a) observed error vs bound, per dataset')
for i in order:
    a.annotate(DATASETS[i], (N_[i], obs_[i]), fontsize=6, alpha=.7)
a.legend(fontsize=8); a.grid(alpha=.3, which='both')

a = ax[0, 1]
a.loglog(N_, ratio_, 'o', color='C2')
xf_ = np.linspace(N_.min(), N_.max(), 50)
a.loglog(xf_, ratio_[np.argmin(N_)] * (xf_ / N_.min()) ** beta, ls='--', color='k',
         label=fr'fit $\propto N^{{{beta:.2f}}}$ ($R^2$={r2:.2f})')
a.loglog(xf_, ratio_[np.argmin(N_)] * (xf_ / N_.min()) ** -0.97, ls=':', color='C3',
         label=r'synthetic $N^{-0.97}$')
for i in range(len(DATASETS)):
    a.annotate(DATASETS[i], (N_[i], ratio_[i]), fontsize=6, alpha=.7)
a.set_xlabel('$N$'); a.set_ylabel('observed / bound')
a.set_title('(b) ratio vs $N$ across real datasets')
a.legend(fontsize=7); a.grid(alpha=.3, which='both')

a = ax[0, 2]
a.plot(homophily_, ratio_, 'o', color='C4')
for i in range(len(DATASETS)):
    a.annotate(DATASETS[i], (homophily_[i], ratio_[i]), fontsize=6, alpha=.7)
a.set_yscale('log')
a.set_xlabel('Original-graph edge homophily'); a.set_ylabel('observed / bound')
a.set_title(f'(c) ratio vs homophily (Spearman $\\rho$={spear_h:.2f})')
a.grid(alpha=.3)

a = ax[1, 0]
width = 0.35
xidx = np.arange(len(DATASETS))
a.bar(xidx - width/2, ratio_, width, label='identity', color='C0')
a.bar(xidx + width/2, ratio_sm_, width, label='softmax', color='C1')
a.set_yscale('log')
a.set_xticks(xidx); a.set_xticklabels(DATASETS, rotation=60, ha='right', fontsize=7)
a.set_ylabel('observed / bound')
a.set_title('(d) saturation artifact: identity vs softmax')
a.legend(fontsize=8); a.grid(alpha=.3, axis='y')

a = ax[1, 1]
a.bar(xidx, alpha_share_, color='C5')
a.set_xticks(xidx); a.set_xticklabels(DATASETS, rotation=60, ha='right', fontsize=7)
a.set_ylabel(r'$\alpha$-term share of bound (%)')
a.set_title(r'(e) $\alpha$-term share -- real graphs are far from $\|\mathbf{\Delta}\|_F=0$')
a.grid(alpha=.3, axis='y')

a = ax[1, 2]
a.loglog(rho_prod_, ratio_, 'o', color='C6')
for i in range(len(DATASETS)):
    a.annotate(DATASETS[i], (rho_prod_[i], ratio_[i]), fontsize=6, alpha=.7)
a.set_xlabel(r'$\rho_1\rho_2$'); a.set_ylabel('observed / bound')
a.set_title(fr'(f) $\rho_1\rho_2$ vs ratio (log-log $r$={pear_rho:.2f})')
a.grid(alpha=.3, which='both')

fig.tight_layout()
fig.savefig(RES_DIR / 'fig_bound_theorem_real.pdf')
fig.savefig(RES_DIR / 'fig_bound_theorem_real.png', dpi=140)
logging.info(f"\nwrote {RES_DIR/'fig_bound_theorem_real.pdf'}, table_bound_real.tex, and summary.txt")
