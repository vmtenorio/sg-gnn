"""
Empirical tightness of Theorem 1 in a controlled synthetic (CSBM) setting.

Experiment ID: bound_synth

Produces Table XI (`tab:bound_csbm`, sweep A) and the alpha, degree and N
trends quoted in Section VI-E. The theorem's model is instantiated exactly and
the quantities the bound depends on are controlled, so the observed error can
be compared against the bound as a *function* of ||Delta||_F and alpha. The
real-data counterpart is experiments/bound_real.py.

The bound is used exactly as stated in the manuscript (eq:err_bnd):

    ||Z* - Zhat||_F  <=  rho1 * rho2 * ( alpha*sqrt(N) + 2*(1+sqrt(N))*||Delta||_F*||X||_F )

No tightening of any proof step is applied.

Model (Theorem 1, verbatim)
---------------------------
    Phi(X; A, Theta) = sigma2( Arw sigma1( Arw X Theta1 ) Theta2 )

with Arw = Dhat^{-1} (A + I) row-normalized (so ||Arw||_1 = 1, as the proof
requires), sigma1 = sigma2 = identity by default, both nonexpansive. No biases:
the theorem's Phi has none. sigma2 = identity is deliberate -- a saturating
output (tanh/softmax) caps ||Zhat - Z*||_F at O(sqrt(N)) by construction, which
would manufacture an N-dependent ratio unrelated to the bound. Both choices are
recorded; sweep F below reproduces the capped variant for comparison.

Data (contextual SBM)
---------------------
    y_i uniform over C classes; x_i = mu_{y_i} + sigma * eps_i, ||eps_i||_2 = sqrt(M)
    G*  : intra-class Erdos-Renyi edges only  ->  Delta = 0
    G   : G* plus m uniformly sampled inter-class ("false positive") edges
So Delta = A - A* is exactly the set of injected false-positive edges, and
||Delta||_F = sqrt(2m) is known in closed form rather than estimated.
X* = class means (recovers y by Definition 2); alpha = max_i ||X*_i - X_i||_2.

Operating point
---------------
Every inequality of the proof is instrumented by `step_slacks` as the fraction
of the error it retains, so the looseness can be attributed to named steps
rather than reported as one opaque constant. The default configuration puts
every step except the two ||Delta||_F ones close to equality:

  square layers  M = hid = C = 16   rho = sigma_max is near-exact for a Theta
                                    close to a scaled isometry; squeezing
                                    50 -> 32 -> 3 is what costs an order
  average degree 1                  the proof only uses ||Arw||_1 = 1, but Arw
                                    *averages* the noise over d_hat neighbours
                                    at both layers, contracting it by ~1/d_hat
  fixed-norm noise                  makes ||X* - X||_F = sqrt(N) alpha exact
  sigma1 = identity                 nonexpansive with constant exactly 1
  isometry penalty iso = 1.0        pushes Theta1, Theta2 toward scaled
                                    isometries, so the two rho steps are
                                    near equality (see `isometry_penalty`)

At this operating point the ratio at Delta = 0 is about 0.7. The
||Delta||_F term stays Theta(N) loose because two independent steps each pay
sqrt(N) -- ||Delta 1||_2 <= sqrt(N)||Delta||_F and the Cauchy-Schwarz step
||(Arws - Arw) X||_F <= ||Arws - Arw||_F ||X||_F, where Arws - Arw has only
~2m nonzero rows. Neither is removable by any choice of data: see sweeps C/G.

Sweeps
------
  A  ||Delta||_F   : m in {0, 1, ..., 800}, everything else fixed  (headline)
  B  alpha, degree : sigma in {0.25, ..., 4.0}; average degree in {0.5, ..., 16}
  C  N             : N in {50, ..., 2000}, at Delta = 0, fixed m, and fixed m/N
  D  M             : feature dimension, layers kept square
  E  rho control   : weight decay, hidden dimension, layer shape, isometry penalty
  F  sigma2        : output nonlinearity (saturation cap artifact)
  G  sigma1, noise : the two per-step slacks that the operating point removes

Outputs
-------
  results/<RESULTS_DIR>/bound_synth/bound_synth.pkl
  results/<RESULTS_DIR>/bound_synth/bound_synth.json
Plots are produced separately by `analysis/plot_bound_synth.py`, which consumes
only these artifacts.
"""

import sys
import json
import pickle
import logging
import argparse
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from sggnn.data import seed_everything

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

from sggnn.paths import results_dir, git_commit
OUT_DIR = results_dir('bound_synth')


# ── model: Theorem 1 verbatim ────────────────────────────────────────────────
ACTS = {'identity': (lambda t: t), 'tanh': torch.tanh, 'relu': torch.relu}


class TheoremGCN(nn.Module):
    """Phi(X;A,Theta) = sigma2( Arw sigma1( Arw X Theta1 ) Theta2 ), no biases."""

    def __init__(self, in_dim, hid_dim, out_dim, sigma1='identity', sigma2='identity'):
        super().__init__()
        self.Theta1 = nn.Parameter(torch.empty(in_dim, hid_dim))
        self.Theta2 = nn.Parameter(torch.empty(hid_dim, out_dim))
        nn.init.xavier_uniform_(self.Theta1)
        nn.init.xavier_uniform_(self.Theta2)
        self.sigma1 = ACTS[sigma1]
        self.sigma2 = ACTS[sigma2]

    def forward(self, X, Arw):
        H = self.sigma1(Arw @ (X @ self.Theta1))
        return self.sigma2(Arw @ (H @ self.Theta2))

    def rhos(self):
        r1 = torch.linalg.matrix_norm(self.Theta1, ord=2).item()
        r2 = torch.linalg.matrix_norm(self.Theta2, ord=2).item()
        return r1, r2


def isometry_penalty(T):
    """||T^T T - (tr(T^T T)/k) I||_F^2 -- zero iff T is a scaled isometry.

    rho = sigma_max(T) is an exact proxy for ||Z T||_F / ||Z||_F only when T acts
    isometrically; with freely trained weights the spectrum decays and the two
    rho steps of the proof lose about a factor of three between them. Penalising
    the departure from an isometry during training removes that slack without
    touching Eq. (12), which holds for any Theta. It does not cost accuracy at
    the operating point (see the `iso` sweep).
    """
    G = T.T @ T
    k = G.shape[0]
    return ((G - torch.eye(k, device=T.device) * (G.trace() / k)) ** 2).sum()


def row_normalize(A):
    """Arw = Dhat^{-1} (A + I) with Dhat = diag((A + I) 1)."""
    Ahat = A + torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    return Ahat / Ahat.sum(dim=1, keepdim=True)


# ── data: contextual SBM ─────────────────────────────────────────────────────
def make_csbm(N, C, M, deg, m_fp, sigma, mu_scale, noise, device, gen):
    """Return X, X*, A (with false positives), A* (oracle), y, Delta.

    A* holds intra-class Erdos-Renyi edges only, with the wiring probability set
    so that the expected intra-class degree is `deg`; A = A* + Delta where Delta
    is m_fp uniformly sampled inter-class edges. Both symmetric, binary, no self
    loops (self loops enter through Arw).

    `noise='sphere'` gives every node a feature perturbation of the *same* norm
    sigma*sqrt(M), so that alpha = max_i ||X*_i - X_i||_2 = sigma*sqrt(M) and
    ||X* - X||_F = sqrt(N)*alpha holds with equality. This makes the proof's
    row-max step exact rather than loose by the max/mean gap of N chi variates;
    `noise='gauss'` restores the isotropic Gaussian used before.
    """
    y = torch.arange(N, device=device) % C
    y = y[torch.randperm(N, generator=gen, device=device)]

    mu = mu_scale * torch.randn(C, M, generator=gen, device=device)
    X_star = mu[y]
    eps = torch.randn(N, M, generator=gen, device=device)
    if noise == 'sphere':
        eps = eps / eps.norm(dim=1, keepdim=True) * (M ** 0.5)
    X = X_star + sigma * eps

    p_in = deg / (N / C - 1.0)

    same = (y[:, None] == y[None, :])
    triu = torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)

    # oracle graph: intra-class ER
    coin = torch.rand(N, N, generator=gen, device=device)
    keep = same & triu & (coin < p_in)
    A_star = torch.zeros(N, N, device=device)
    A_star[keep] = 1.0
    A_star = A_star + A_star.T

    # false-positive edges: sample m_fp inter-class pairs without replacement
    cross = torch.nonzero(triu & ~same, as_tuple=False)
    m_fp = min(m_fp, cross.shape[0])
    Delta = torch.zeros(N, N, device=device)
    if m_fp > 0:
        pick = torch.randperm(cross.shape[0], generator=gen, device=device)[:m_fp]
        sel = cross[pick]
        Delta[sel[:, 0], sel[:, 1]] = 1.0
        Delta = Delta + Delta.T

    return X, X_star, A_star + Delta, A_star, y, Delta, int(m_fp)


def split_masks(N, device, gen, train_frac=0.8, val_frac=0.1):
    idx = torch.randperm(N, generator=gen, device=device)
    n_tr, n_va = int(train_frac * N), int(val_frac * N)
    masks = []
    for lo, hi in ((0, n_tr), (n_tr, n_tr + n_va), (n_tr + n_va, N)):
        m = torch.zeros(N, dtype=torch.bool, device=device)
        m[idx[lo:hi]] = True
        masks.append(m)
    return masks


def step_slacks(model, X, X_star, Arw, Arw_star, Delta, alpha, sqrtN,
                rho1, rho2, delta_normF, X_normF):
    """Fraction of the error each inequality of the Eq. (12) proof retains.

    At Delta = 0 the proof chain is, writing E = X* - X and S = sigma1(Arw X T1),

        ||Z* - Zhat||_F
          <= [s1] ||Arw (S* - S) T2||_F        sigma2 nonexpansive
          <= [s2] rho2 ||Arw (S* - S)||_F      definition of rho2
          <= [s3] rho2 ||S* - S||_F            ||Arw||_1 = 1
          <= [s4] rho2 ||Arw E T1||_F          sigma1 nonexpansive
          <= [s5] rho1 rho2 ||Arw E||_F        definition of rho1
          <= [s6] rho1 rho2 ||E||_F            ||Arw||_1 = 1
          <= [s7] rho1 rho2 alpha sqrt(N)      row-max bound on ||E||_F

    so each s_k lies in (0, 1] and their product is exactly observed/bound when
    Delta = 0. For the ||Delta||_F term we record instead the two steps that each
    cost a factor sqrt(N): `d_ones` for ||Delta 1||_2 <= sqrt(N)||Delta||_F, and
    `d_cs` for the Cauchy-Schwarz ||(Arws - Arw) X||_F <= ||Arws - Arw||_F ||X||_F,
    together with their composite `d_pert` against the (1+sqrt(N))||Delta||_F
    prefactor actually used in the bound. These are diagnostics only: the bound
    reported above is still Eq. (12) verbatim.
    """
    fro = lambda t: torch.linalg.matrix_norm(t, ord='fro').item()
    safe = lambda a, b: (a / b) if b > 0 else float('nan')

    E = X_star - X
    v7 = fro(E)
    v6 = fro(Arw_star @ E)
    v5 = fro((Arw_star @ E) @ model.Theta1)
    S_star = model.sigma1(Arw_star @ (X_star @ model.Theta1))
    S_hat = model.sigma1(Arw_star @ (X @ model.Theta1))
    v4 = fro(S_star - S_hat)
    v3 = fro(Arw_star @ (S_star - S_hat))
    v2 = fro((Arw_star @ (S_star - S_hat)) @ model.Theta2)
    v1 = fro(model(X_star, Arw_star) - model(X, Arw_star))

    P = Arw_star - Arw
    return {
        's7_rowmax': safe(v7, alpha * sqrtN),
        's6_arw_in': safe(v6, v7),
        's5_rho1': safe(v5, rho1 * v6),
        's4_sigma1': safe(v4, v5),
        's3_arw_out': safe(v3, v4),
        's2_rho2': safe(v2, rho2 * v3),
        's1_sigma2': safe(v1, v2),
        'd_ones': safe(Delta.sum(dim=1).norm().item(), sqrtN * delta_normF),
        'd_cs': safe(fro(P @ X), fro(P) * X_normF),
        'd_pert': safe(fro(P), (1.0 + sqrtN) * delta_normF),
    }


# ── one (config, seed) run ───────────────────────────────────────────────────
def run_once(cfg, seed, device):
    seed_everything(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    X, X_star, A, A_star, y, Delta, m_fp = make_csbm(
        cfg['N'], cfg['C'], cfg['M'], cfg['deg'], cfg['m_fp'],
        cfg['sigma'], cfg['mu_scale'], cfg['noise'], device, gen)
    Arw, Arw_star = row_normalize(A), row_normalize(A_star)
    tr, va, te = split_masks(cfg['N'], device, gen)

    model = TheoremGCN(cfg['M'], cfg['hid_dim'], cfg['C'],
                       cfg['sigma1'], cfg['sigma2']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    lossf = nn.CrossEntropyLoss()

    best_val, best_state, best_test, wait = -1.0, None, 0.0, 0
    for _ in range(cfg['epochs']):
        model.train(); opt.zero_grad()
        out = model(X, Arw)
        loss = lossf(out[tr], y[tr])
        if cfg['iso']:
            loss = loss + cfg['iso'] * (isometry_penalty(model.Theta1)
                                        + isometry_penalty(model.Theta2))
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(X, Arw).argmax(1)
            v = (pred[va] == y[va]).float().mean().item()
        if v > best_val:
            best_val, wait = v, 0
            best_test = (pred[te] == y[te]).float().mean().item()
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg['patience']:
                break
    model.load_state_dict(best_state)
    model.eval()

    # ── evaluate Theorem 1 at the trained weights ────────────────────────────
    with torch.no_grad():
        Z_hat = model(X, Arw)                 # Phi(X;  A,  Theta)
        Z_star = model(X_star, Arw_star)      # Phi(X*; A*, Theta)
        observed = torch.linalg.matrix_norm(Z_star - Z_hat, ord='fro').item()

        rho1, rho2 = model.rhos()
        alpha = (X_star - X).norm(dim=1).max().item()
        X_normF = torch.linalg.matrix_norm(X, ord='fro').item()
        delta_normF = torch.linalg.matrix_norm(Delta, ord='fro').item()
        sqrtN = float(cfg['N']) ** 0.5

        # Eq. (12), exactly as stated -- no step tightened
        term_alpha = alpha * sqrtN
        term_delta = 2.0 * (1.0 + sqrtN) * delta_normF * X_normF
        bound = rho1 * rho2 * (term_alpha + term_delta)

        slack = step_slacks(model, X, X_star, Arw, Arw_star, Delta, alpha,
                            sqrtN, rho1, rho2, delta_normF, X_normF)

    return {
        **slack,
        'observed_error': observed,
        'bound': bound,
        'ratio': observed / bound if bound > 0 else float('nan'),
        'term_alpha': rho1 * rho2 * term_alpha,
        'term_delta': rho1 * rho2 * term_delta,
        'rho1': rho1, 'rho2': rho2, 'rho_prod': rho1 * rho2,
        'alpha': alpha,
        'X_normF': X_normF,
        'delta_normF': delta_normF,
        'm_fp': m_fp,
        'test_acc': best_test,
        'val_acc': best_val,
        'n_params': cfg['M'] * cfg['hid_dim'] + cfg['hid_dim'] * cfg['C'],
    }


# ── sweep definitions ────────────────────────────────────────────────────────
# Operating point chosen so that every inequality of the proof other than the
# two ||Delta||_F steps is close to equality (see `step_slacks`): square layers
# make rho1, rho2 near-exact; average degree 1 removes the noise averaging that
# ||Arw||_1 = 1 cannot see; fixed-norm noise makes the row-max step exact; and
# sigma1 = identity is nonexpansive with constant exactly 1.
BASE = dict(N=200, C=16, M=16, deg=1.0, m_fp=100, sigma=1.0, mu_scale=1.5,
            hid_dim=16, lr=1e-2, wd=5e-4, epochs=500, patience=120, iso=1.0,
            noise='sphere', sigma1='identity', sigma2='identity')


def build_sweeps():
    sweeps = {}

    # A: ||Delta||_F -- the headline. Log-spaced and dense near Delta = 0, where
    #    the bound reduces to rho1*rho2*alpha*sqrt(N) and no ||X||_F factor enters.
    sweeps['delta'] = [{**BASE, 'm_fp': m} for m in
                       (0, 1, 2, 5, 10, 25, 50, 100, 200, 400, 800)]

    # B: alpha (feature noise). Run at m=0, where the alpha term IS the whole
    #    bound, and at m=800, where the Delta term dominates.
    sweeps['alpha_nodelta'] = [{**BASE, 'sigma': s_, 'm_fp': 0}
                               for s_ in (0.25, 0.5, 1.0, 2.0, 4.0)]
    sweeps['alpha'] = [{**BASE, 'sigma': s_} for s_ in (0.25, 0.5, 1.0, 2.0, 4.0)]

    # B2: average intra-class degree -- the single largest looseness source at
    #     Delta = 0. The proof only uses ||Arw||_1 = 1, but Arw *averages* the
    #     feature noise over d_hat neighbours at each of the two layers, so the
    #     realised contraction is ~1/d_hat and the bound misses it entirely.
    sweeps['degree_nodelta'] = [{**BASE, 'deg': d, 'm_fp': 0}
                                for d in (0.5, 1, 2, 4, 8, 16)]

    # C1: N at fixed ||Delta||_F and fixed average intra-class degree -- isolates
    #     the explicit sqrt(N) factors and the sqrt(N) growth of ||X||_F.
    sweeps['N_fixed_delta'] = [{**BASE, 'N': n} for n in (50, 100, 200, 500, 1000, 2000)]
    # C0: N at Delta = 0 -- the alpha term alone, where the ratio is N-invariant.
    sweeps['N_nodelta'] = [{**BASE, 'N': n, 'm_fp': 0}
                           for n in (50, 100, 200, 500, 1000, 2000)]
    # C2: N at fixed false-positive density (m_fp ~ N), the realistic scaling.
    sweeps['N_fixed_density'] = [{**BASE, 'N': n,
                                  'm_fp': max(1, int(BASE['m_fp'] * n / BASE['N']))}
                                 for n in (50, 100, 200, 500, 1000, 2000)]

    # D: feature dimension. ||X||_F grows as sqrt(M) and enters the bound, but
    #    the classification problem is essentially unchanged -- a clean isolation
    #    of one looseness source.
    sweeps['M'] = [{**BASE, 'M': m_, 'hid_dim': m_} for m_ in (8, 16, 32, 64, 128)]

    # E: does controlling rho shrink the constant? (l2 strength and capacity)
    sweeps['rho_wd'] = [{**BASE, 'wd': w} for w in (5e-4, 5e-3, 5e-2, 5e-1, 5e0)]
    sweeps['rho_hid'] = [{**BASE, 'hid_dim': h} for h in (8, 16, 32, 64, 128)]

    # E2: layer shape. rho = sigma_max is a tight proxy for ||Z Theta||_F/||Z||_F
    #     only when Theta is close to a scaled isometry; squeezing M -> hid -> C
    #     is what makes the two rho steps lose an order between them.
    sweeps['shape'] = [{**BASE, 'M': m_, 'hid_dim': h, 'm_fp': 0}
                       for m_, h in ((16, 16), (16, 32), (32, 16), (50, 32), (50, 8))]

    # E3: isometry penalty -- the other half of the rho slack, and the only knob
    #     that moves the two rho steps toward equality at fixed layer shape.
    sweeps['iso'] = [{**BASE, 'iso': i_, 'm_fp': m}
                     for i_ in (0.0, 1e-2, 1e-1, 1.0) for m in (0, 1, 100)]

    # F: output nonlinearity -- shows the saturation cap artifact explicitly.
    sweeps['sigma2'] = [{**BASE, 'sigma2': s_, 'm_fp': m}
                        for s_ in ('identity', 'tanh') for m in (0, 100, 800)]

    # G: sigma1 and the noise model -- the two remaining per-step slacks, each
    #    exactly 1 at the chosen operating point.
    sweeps['sigma1'] = [{**BASE, 'sigma1': s_, 'm_fp': m}
                        for s_ in ('identity', 'tanh', 'relu') for m in (0, 100)]
    sweeps['noise'] = [{**BASE, 'noise': n_, 'm_fp': m}
                       for n_ in ('sphere', 'gauss') for m in (0, 100)]
    return sweeps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    logging.info(f"device={device}  seeds={args.seeds}")

    sweeps = build_sweeps()
    results = {}
    for name, cfgs in sweeps.items():
        results[name] = []
        for cfg in cfgs:
            runs = [run_once(cfg, 100 + s, device) for s in range(args.seeds)]
            agg = {'config': cfg, 'runs': runs}
            for k in runs[0]:
                vals = [r[k] for r in runs]
                agg[f'{k}_mean'] = float(np.mean(vals))
                agg[f'{k}_std'] = float(np.std(vals))
            results[name].append(agg)
            logging.info(
                f"[{name}] N={cfg['N']} deg={cfg['deg']} m={cfg['m_fp']} "
                f"sig={cfg['sigma']} wd={cfg['wd']:g} M={cfg['M']} "
                f"hid={cfg['hid_dim']} iso={cfg['iso']:g} "
                f"s1={cfg['sigma1']} s2={cfg['sigma2']} | "
                f"|D|_F={agg['delta_normF_mean']:.1f} obs={agg['observed_error_mean']:.3f} "
                f"bound={agg['bound_mean']:.3g} ratio={agg['ratio_mean']:.3g} "
                f"acc={agg['test_acc_mean']:.3f}")

    payload = {
        'results': results,
        'metadata': {
            'experiment_id': 'bound_synth',
            'seeds': [100 + s for s in range(args.seeds)],
            'n_seeds': args.seeds,
            'base_config': BASE,
            'device': str(device),
            'timestamp': datetime.datetime.now().isoformat(),
            'git_commit': git_commit(),
            'torch': torch.__version__,
            'bound': ('rho1*rho2*(alpha*sqrt(N) + 2*(1+sqrt(N))*||Delta||_F*||X||_F), '
                      'Eq. (12) exactly as stated; no proof step tightened.'),
            'model': ('sigma2(Arw sigma1(Arw X Theta1) Theta2), Arw = Dhat^{-1}(A+I), '
                      'sigma1 and sigma2 per config, both nonexpansive, no biases.'),
            'slacks': ('s1..s7 are the fraction of the error retained by each '
                       'inequality of the proof; at Delta = 0 their product is '
                       'observed/bound. d_ones, d_cs, d_pert are the analogous '
                       'diagnostics for the two sqrt(N) steps of the Delta term. '
                       'Diagnostics only -- the reported bound is Eq. (12) verbatim.'),
        },
    }

    with open(OUT_DIR / 'bound_synth.pkl', 'wb') as f:
        pickle.dump(payload, f)
    with open(OUT_DIR / 'bound_synth.json', 'w') as f:
        json.dump(payload, f, indent=2, default=float)
    logging.info(f"saved -> {OUT_DIR}")


if __name__ == '__main__':
    main()
