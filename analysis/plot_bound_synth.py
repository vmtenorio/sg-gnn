"""
Analysis and figures for the `bound_synth` experiment.

Consumes only results/<RESULTS_DIR>/bound_synth/bound_synth.json -- no training.
Produces, in the same directory:
  fig_bound_synth.pdf   (4-panel summary figure)
  summary.txt           (tables + fitted trends)
  table_bound_csbm.tex  (LaTeX fragment)
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
from sggnn.report import Summary, loglog_slope
RES_DIR = results_dir('bound_synth')
with open(RES_DIR / 'bound_synth.json') as f:
    payload = json.load(f)
R, META = payload['results'], payload['metadata']

out = Summary()
emit = out.emit


def col(sweep, key):
    return np.array([e[f'{key}_mean'] for e in R[sweep]])


def cfg(sweep, key):
    return np.array([e['config'][key] for e in R[sweep]])


emit("=" * 78)
emit(f"bound_synth  |  {META['n_seeds']} seeds  |  {META['timestamp'][:19]}")
emit(f"base: {META['base_config']}")
emit("=" * 78)

# ── Sweep A: ||Delta||_F ─────────────────────────────────────────────────────
d = R['delta']
dF, obs, bnd = col('delta', 'delta_normF'), col('delta', 'observed_error'), col('delta', 'bound')
ratio, acc = col('delta', 'ratio'), col('delta', 'test_acc')
ta, td = col('delta', 'term_alpha'), col('delta', 'term_delta')
obs_sd = np.array([e['observed_error_std'] for e in d])

emit(f"\n[A] ||Delta||_F sweep  (N={META['base_config']['N']}, all else fixed)")
emit(f"{'m_fp':>6}{'|D|_F':>8}{'observed':>12}{'bound':>12}{'ratio':>11}"
     f"{'alpha-term %':>14}{'test acc':>10}")
for i, e in enumerate(d):
    emit(f"{e['config']['m_fp']:>6}{dF[i]:>8.1f}{obs[i]:>9.2f}±{obs_sd[i]:<2.0f}"
         f"{bnd[i]:>12.3g}{ratio[i]:>11.2e}{100*ta[i]/bnd[i]:>13.1f}%{acc[i]:>10.3f}")

pos = dF > 0
emit(f"\n  observed vs ||Delta||_F : Pearson r={pearsonr(dF, obs)[0]:.4f}, "
     f"Spearman rho={spearmanr(dF, obs)[0]:.4f}")
b, r2 = loglog_slope(dF[pos], obs[pos])
emit(f"  observed ~ ||Delta||_F^{b:.2f}  (log-log R^2={r2:.4f})")
b2, r22 = loglog_slope(dF[pos], bnd[pos])
emit(f"  bound    ~ ||Delta||_F^{b2:.2f}  (log-log R^2={r22:.4f})")
emit(f"  ratio at ||Delta||_F=0 : {ratio[0]:.3e}  ({-np.log10(ratio[0]):.1f} orders loose)")
emit(f"  ratio at ||Delta||_F>0 : {ratio[pos].min():.2e} .. {ratio[pos].max():.2e}"
     f"  ({-np.log10(ratio[pos].max()):.1f}-{-np.log10(ratio[pos].min()):.1f} orders)")
emit(f"  a single false-positive edge (m=1) moves the alpha-term share of the "
     f"bound from {100*ta[0]/bnd[0]:.0f}% to {100*ta[1]/bnd[1]:.1f}%")

# ── Slack decomposition: which inequality loses what ─────────────────────────
STEPS = [('s7_rowmax',  '||E||_F <= sqrt(N) alpha'),
         ('s6_arw_in',  '||Arw E||_F <= ||E||_F'),
         ('s5_rho1',    '||Arw E T1||_F <= rho1 ||Arw E||_F'),
         ('s4_sigma1',  'sigma1 nonexpansive'),
         ('s3_arw_out', '||Arw dS||_F <= ||dS||_F'),
         ('s2_rho2',    '||Arw dS T2||_F <= rho2 ||Arw dS||_F'),
         ('s1_sigma2',  'sigma2 nonexpansive')]

emit("\n[A2] per-step slack at ||Delta||_F = 0 is retained as an internal")
emit("     diagnostic only; the manuscript and response report the bound as a")
emit("     whole. The product below is a completeness check on the accounting.")
prod = 1.0
for key, _ in STEPS:
    prod *= col('delta', key)[0]
emit(f"     product of the seven step fractions = {prod:.4f}  vs  observed/bound "
     f"= {ratio[0]:.4f}   (rel. error {abs(prod-ratio[0])/ratio[0]:.1e})")

# ── Sweep B2: average degree, the largest single slack at Delta = 0 ──────────
emit("\n[B2] average intra-class degree  (m=0; the bound only uses ||Arw||_1=1,")
emit("     but Arw averages the noise over d_hat neighbours at both layers)")
emit(f"{'deg':>7}{'s6_arw_in':>12}{'s3_arw_out':>13}{'ratio':>11}{'x loose':>10}{'acc':>8}")
dg, dr = cfg('degree_nodelta', 'deg').astype(float), col('degree_nodelta', 'ratio')
for i, e in enumerate(R['degree_nodelta']):
    emit(f"{e['config']['deg']:>7.1f}{e['s6_arw_in_mean']:>12.3f}"
         f"{e['s3_arw_out_mean']:>13.3f}{dr[i]:>11.3f}{1/dr[i]:>9.1f}x"
         f"{e['test_acc_mean']:>8.3f}")
bd, r2d = loglog_slope(dg, dr)
emit(f"  ratio ~ deg^{bd:.2f}  (log-log R^2={r2d:.3f});  ratio spread "
     f"{dr.max()/dr.min():.1f}x over a {dg.max()/dg.min():.0f}x degree range")

# ── Sweep B: alpha ───────────────────────────────────────────────────────────
for name, tag in (('alpha_nodelta', 'm=0, bound = rho1*rho2*alpha*sqrt(N)'),
                  ('alpha', f"m={META['base_config']['m_fp']}, Delta term dominates")):
    a_, o_, bn_, r_ = (col(name, 'alpha'), col(name, 'observed_error'),
                       col(name, 'bound'), col(name, 'ratio'))
    emit(f"\n[B] alpha sweep  ({tag})")
    emit(f"{'sigma':>7}{'alpha':>9}{'observed':>11}{'bound':>12}{'ratio':>11}")
    for i, e in enumerate(R[name]):
        emit(f"{e['config']['sigma']:>7.2f}{a_[i]:>9.2f}{o_[i]:>11.2f}{bn_[i]:>12.3g}{r_[i]:>11.2e}")
    b, r2 = loglog_slope(a_, o_)
    emit(f"  observed ~ alpha^{b:.2f} (R^2={r2:.3f});  ratio spread "
         f"{r_.max()/r_.min():.2f}x")

# ── Sweep C/D: N and M ───────────────────────────────────────────────────────
for name, xk, lab in (('N_nodelta', 'N', 'N at Delta = 0 (alpha term alone)'),
                      ('N_fixed_delta', 'N', 'N at fixed ||Delta||_F'),
                      ('N_fixed_density', 'N', 'N at fixed FP density'),
                      ('M', 'M', 'feature dimension M (layers kept square)')):
    x, o_, bn_, r_ = cfg(name, xk), col(name, 'observed_error'), col(name, 'bound'), col(name, 'ratio')
    xf = col(name, 'X_normF')
    emit(f"\n[C] {lab}")
    emit(f"{xk:>7}{'||X||_F':>10}{'observed':>11}{'bound':>12}{'ratio':>11}")
    for i in range(len(x)):
        emit(f"{x[i]:>7.0f}{xf[i]:>10.1f}{o_[i]:>11.2f}{bn_[i]:>12.3g}{r_[i]:>11.2e}")
    bo, _ = loglog_slope(x.astype(float), o_)
    bb, _ = loglog_slope(x.astype(float), bn_)
    emit(f"  observed ~ {xk}^{bo:.2f} ;  bound ~ {xk}^{bb:.2f} ;  "
         f"ratio ~ {xk}^{loglog_slope(x.astype(float), r_)[0]:.2f}")

# ── Sweep E: rho control (Samu's hypothesis) ─────────────────────────────────
emit("\n[E] does controlling rho1*rho2 tighten the ratio?")
for name, xk in (('rho_wd', 'wd'), ('rho_hid', 'hid_dim')):
    x, rp, o_, bn_, r_, ac = (cfg(name, xk), col(name, 'rho_prod'),
                              col(name, 'observed_error'), col(name, 'bound'),
                              col(name, 'ratio'), col(name, 'test_acc'))
    emit(f"\n  {xk} sweep")
    emit(f"{xk:>10}{'rho1*rho2':>12}{'observed':>11}{'bound':>12}{'ratio':>11}{'acc':>8}")
    for i in range(len(x)):
        emit(f"{x[i]:>10g}{rp[i]:>12.3f}{o_[i]:>11.2f}{bn_[i]:>12.3g}{r_[i]:>11.2e}{ac[i]:>8.3f}")
    emit(f"  rho1*rho2 varies {rp.max()/rp.min():.1f}x  ->  ratio varies "
         f"{r_.max()/r_.min():.2f}x   (rho is a common factor on both sides)")

emit("\n[E2] layer shape -- rho = sigma_max is tight only for a near-isometry")
emit(f"{'M':>5}{'hid':>6}{'C':>4}{'s5_rho1':>10}{'s2_rho2':>10}{'ratio':>11}{'x loose':>10}")
for e in R['shape']:
    c = e['config']
    emit(f"{c['M']:>5}{c['hid_dim']:>6}{c['C']:>4}{e['s5_rho1_mean']:>10.3f}"
         f"{e['s2_rho2_mean']:>10.3f}{e['ratio_mean']:>11.3f}"
         f"{1/e['ratio_mean']:>9.1f}x")

# ── Sweep G: sigma1 and the noise model ─────────────────────────────────────
emit("\n[G] the two per-step slacks the operating point removes")
emit(f"{'sigma1':>10}{'m_fp':>7}{'s4_sigma1':>12}{'ratio':>11}{'acc':>8}")
for e in R['sigma1']:
    emit(f"{e['config']['sigma1']:>10}{e['config']['m_fp']:>7}"
         f"{e['s4_sigma1_mean']:>12.3f}{e['ratio_mean']:>11.3e}"
         f"{e['test_acc_mean']:>8.3f}")
emit(f"\n{'noise':>10}{'m_fp':>7}{'s7_rowmax':>12}{'ratio':>11}{'acc':>8}")
for e in R['noise']:
    emit(f"{e['config']['noise']:>10}{e['config']['m_fp']:>7}"
         f"{e['s7_rowmax_mean']:>12.3f}{e['ratio_mean']:>11.3e}"
         f"{e['test_acc_mean']:>8.3f}")

# ── Sweep F: sigma2 ──────────────────────────────────────────────────────────
emit("\n[F] output nonlinearity (saturation cap artifact)")
emit(f"{'sigma2':>10}{'m_fp':>7}{'observed':>11}{'bound':>12}{'ratio':>11}")
for e in R['sigma2']:
    emit(f"{e['config']['sigma2']:>10}{e['config']['m_fp']:>7}"
         f"{e['observed_error_mean']:>11.2f}{e['bound_mean']:>12.3g}{e['ratio_mean']:>11.2e}")

# ── how does the Delta-attributable error actually grow? ────────────────────
# The observed error at ||Delta||_F = 0 is a floor set entirely by the feature
# noise alpha. The part of the error attributable to false-positive edges is the
# excess over that floor. The bound is linear in ||Delta||_F; this asks what the
# excess error actually does.
emit("\n[G] excess error over the alpha-floor,  obs(m) - obs(0)")
emit(f"{'|D|_F':>8}{'excess':>10}{'bound':>12}{'excess/bound':>14}")
exc = obs - obs[0]
for i in range(1, len(dF)):
    emit(f"{dF[i]:>8.1f}{exc[i]:>10.2f}{bnd[i]:>12.3g}{exc[i]/bnd[i]:>14.2e}")
be, r2e = loglog_slope(dF[1:], exc[1:])
emit(f"\n  excess ~ ||Delta||_F^{be:.2f}   (log-log R^2={r2e:.4f})")
emit("  the bound grows as ||Delta||_F^1 (linear, by construction), so the "
     "observed excess")
emit("  grows FASTER than the bound in shape: the gap is widest at small "
     "||Delta||_F and")
emit("  narrows as false positives accumulate. Crossover -- where the excess "
     "first exceeds")
cross = np.where(exc > obs[0])[0]
if len(cross):
    emit(f"  the alpha-floor -- is at ||Delta||_F = {dF[cross[0]]:.1f} "
         f"(m = {R['delta'][cross[0]]['config']['m_fp']} false-positive edges).")

with open(RES_DIR / 'summary.txt', 'w') as f:
    f.write("\n".join(out) + "\n")

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 3, figsize=(15, 8))

a = ax[0, 0]
a.errorbar(dF, obs, yerr=obs_sd, marker='o', color='C0',
           label=r'observed $\|\mathbf{Z}^*-\hat{\mathbf{Z}}\|_F$')
a.plot(dF, bnd, marker='s', color='C3', label='bound, Eq. (12)')
a.set_yscale('log'); a.set_xlabel(r'$\|\mathbf{\Delta}\|_F$'); a.set_ylabel('error')
a.set_title(r'(a) observed error vs bound'); a.legend(fontsize=8); a.grid(alpha=.3)

a = ax[0, 1]
a.plot(dF, ratio, marker='o', color='C2')
a.set_yscale('log'); a.set_xlabel(r'$\|\mathbf{\Delta}\|_F$')
a.set_ylabel('observed / bound')
a.set_title(f'(b) tight at $\\Delta=0$ ({1/ratio[0]:.1f}$\\times$), collapses at the\nfirst false positive')
a.annotate(f'{ratio[0]:.1e}\n($\\Delta=0$)', (dF[0], ratio[0]),
           textcoords='offset points', xytext=(10, -14), fontsize=8)
a.grid(alpha=.3)

a = ax[0, 2]
# the excess is below the seed-to-seed noise floor for the first few m, so it can
# be negative; anchor the reference lines on the first positive point instead.
posx = np.where(exc[1:] > 0)[0] + 1
a.loglog(dF[posx], exc[posx], marker='o', color='C0', label='observed excess')
x0, y0 = dF[posx[0]], exc[posx[0]]
xf_ = np.linspace(x0, dF[-1], 50)
a.loglog(xf_, y0 * (xf_ / x0) ** be, ls='--', color='k',
         label=fr'fit $\propto\|\mathbf{{\Delta}}\|_F^{{{be:.2f}}}$ ($R^2$={r2e:.2f})')
a.loglog(xf_, y0 * (xf_ / x0), ls=':', color='C3',
         label=r'bound shape $\propto\|\mathbf{\Delta}\|_F^{1}$')
a.set_xlabel(r'$\|\mathbf{\Delta}\|_F$'); a.set_ylabel(r'excess over $\alpha$-floor')
a.set_title('(c) true excess error is quadratic, bound is linear')
a.legend(fontsize=8); a.grid(alpha=.3, which='both')

a = ax[1, 0]
a.plot(dF, obs, marker='o', color='C0')
a2 = a.twinx(); a2.plot(dF, 100 * acc, marker='^', color='C1', ls='--')
a2.set_ylabel('test accuracy (%)', color='C1'); a2.tick_params(axis='y', colors='C1')
a.set_xlabel(r'$\|\mathbf{\Delta}\|_F$'); a.set_ylabel('observed error', color='C0')
a.set_title("(d) the theorem's qualitative claim holds"
            f"\n(Spearman $\\rho$={spearmanr(dF, obs)[0]:.3f})")
a.grid(alpha=.3)

a = ax[1, 1]
Nx0, Nr0 = cfg('N_nodelta', 'N').astype(float), col('N_nodelta', 'ratio')
a.loglog(Nx0, Nr0, marker='^', color='C2', label=r'$\mathbf{\Delta}=\mathbf{0}$')
Nx, Nr = cfg('N_fixed_delta', 'N').astype(float), col('N_fixed_delta', 'ratio')
a.loglog(Nx, Nr, marker='o', color='C4', label=r'fixed $\|\mathbf{\Delta}\|_F$')
Nx2, Nr2 = cfg('N_fixed_density', 'N').astype(float), col('N_fixed_density', 'ratio')
a.loglog(Nx2, Nr2, marker='s', color='C5', label='fixed FP density')
a.loglog(Nx, Nr[0] * (Nx / Nx[0]) ** -1.0, ls=':', color='k', label=r'$N^{-1}$')
a.set_xlabel('$N$'); a.set_ylabel('observed / bound')
a.set_title(r'(e) the $\alpha$ term is $N$-invariant;' '\n'
            r'the $\|\mathbf{\Delta}\|_F$ term loses $\Theta(N)$')
a.legend(fontsize=8); a.grid(alpha=.3, which='both')

a = ax[1, 2]
LBL = [r'$\|E\|_F\!\leq\!\sqrt{N}\alpha$', r'$\|\hat{A}E\|_F$',
       r'$\rho_1$', r'$\sigma_1$', r'$\|\hat{A}\delta S\|_F$',
       r'$\rho_2$', r'$\sigma_2$']
vals = [col('delta', k)[0] for k, _ in STEPS]
ypos = np.arange(len(vals))
a.barh(ypos, vals, color=['C2' if v > .9 else 'C3' for v in vals])
a.axvline(1.0, ls=':', color='k')
a.set_yticks(ypos); a.set_yticklabels(LBL, fontsize=8); a.invert_yaxis()
a.set_xlim(0, 1.05); a.set_xlabel('fraction of the error retained')
a.set_title(f'(f) per-step slack at $\\Delta=0$\n'
            f'(product = {np.prod(vals):.2f} = obs/bound)')
a.grid(alpha=.3, axis='x')

fig.tight_layout()
fig.savefig(RES_DIR / 'fig_bound_synth.pdf')
fig.savefig(RES_DIR / 'fig_bound_synth.png', dpi=140)

# ── Table (`tab:bound_csbm`): m in {0,1,10,100,400,800} subset of sweep A ────
TABLE_MS = [0, 1, 10, 100, 400, 800]
m_all = cfg('delta', 'm_fp')
delta_all = col('delta', 'delta_normF')
obs_all = col('delta', 'observed_error')
bound_all = col('delta', 'bound')
ratio_all = col('delta', 'ratio')
alpha_term_all = col('delta', 'term_alpha')


def sci(x):
    mant, exp = f"{x:.2e}".split('e')
    exp = int(exp)
    return f"${mant}\\!\\times\\!10^{{{exp}}}$"


tex_lines = [
    r"\begin{table}[h]", r"    \centering", r"    \small",
    r"    \resizebox{\columnwidth}{!}{%",
    r"    \begin{tabular}{rrrrrr}", r"    \toprule",
    r"    $m$ & $\|\bbDelta\|_F$ & $\|\bbZ^* - \hbZ\|_F$ & $B$ & "
    r"$\|\bbZ^* - \hbZ\|_F / B$ & $\rho_1\rho_2\alpha\sqrt{N}/B$ \\",
    r"    \midrule",
]
for m in TABLE_MS:
    idx = int(np.where(m_all == m)[0][0])
    alpha_share = alpha_term_all[idx] / bound_all[idx]
    tex_lines.append(
        f"    {m} & {delta_all[idx]:.1f} & {obs_all[idx]:.2f} & {sci(bound_all[idx])} & "
        f"{sci(ratio_all[idx])} & ${alpha_share:.3f}$ \\\\"
    )
tex_lines += [
    r"    \bottomrule", r"    \end{tabular}}",
    r"    \caption{Embedding error $\|\bbZ^* - \hbZ\|_F$ against $B$, the right-hand side "
    r"of~\eqref{eq:err_bnd}, on a contextual stochastic block model in which $m$ "
    r"false-positive edges are injected into an initially label-consistent graph, so "
    r"that $\|\bbDelta\|_F = \sqrt{2m}$ exactly.}",
    r"    \label{tab:bound_csbm}", r"\end{table}",
]
table_csbm_tex = "\n".join(tex_lines)
(RES_DIR / 'table_bound_csbm.tex').write_text(table_csbm_tex)
print(table_csbm_tex)

logging.info(f"\nwrote {RES_DIR/'fig_bound_synth.pdf'}, table_bound_csbm.tex, and summary.txt")
