#!/usr/bin/env python3
"""
run_boundary_benchmarks.py -- the boundary test on standard forecasting data (P0-7).

The controlled sweep places synthetic distributions on the win/tie/lose map. This
script asks the same question of real benchmarks, and it asks it *inside the model
class of Theorem 1* rather than by swapping a normalization inside a deep network,
so that a disagreement is informative about the boundary rather than about a
backbone.

Model class (Definition 1 with d = 1). A context window of L steps and C channels
is mean-pooled into P patches, giving G = C * P input groups -- "a group can be a
lag, a patch, or a variable". The predictor is

    y_hat = u * ( p(s)^T x ) + b,

where p is alpha-entmax over the G group scores, s is bounded to [-2, 2], and
(u, b) is the scalar readout that Definition 1 calls the sensitivity of the
downstream layers. Only alpha changes between arms; the data, the groups, the
optimizer (L-BFGS-B from zero) and the split are identical.

Because the fit is deterministic, variability comes from rolling-origin folds
rather than from seeds.

For each dataset the script also reads off the phase-map coordinates from the
TRAINING split only -- the effective support fraction k/G and the sample size n --
and records what the controlled map predicts, so the real-data outcome can be
compared against a prediction instead of being described after the fact.

Usage:
  python run_boundary_benchmarks.py --ett-dir DIR [--ettm-dir DIR] [--metr FILE]
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SCORE_BOUND = 2.0      # overridden by --score-bound
BISECT_ITERS = 60
ALPHAS = [1.0, 1.25, 1.5, 1.75, 2.0]
FLOOR_REL = 1e-3          # practical margin, relative to the softmax arm


# --------------------------------------------------------------------------- #
# alpha-entmax (identical to run_entmax_sweep.py)
# --------------------------------------------------------------------------- #
def softmax(s):
    e = np.exp(s - s.max()); return e / e.sum()


def entmax(s, alpha):
    if alpha == 1.0:
        return softmax(s)
    a1 = alpha - 1.0; z = a1 * s; inv = 1.0 / a1
    hi = z.max(); lo = hi - 1.0
    for _ in range(BISECT_ITERS):
        mid = .5 * (lo + hi)
        if np.power(np.maximum(z - mid, 0.), inv).sum() > 1.0: lo = mid
        else: hi = mid
    p = np.power(np.maximum(z - .5 * (lo + hi), 0.), inv)
    t = p.sum()
    return p / t if t > 0 else np.full(len(s), 1.0 / len(s))


def entmax_vjp(p, g, alpha):
    if alpha == 1.0:
        return p * (g - p @ g)
    u = np.power(p, 2.0 - alpha, where=p > 0, out=np.zeros_like(p))
    us = u.sum()
    return np.zeros_like(p) if us <= 0 else u * g - (u @ g) / us * u


# --------------------------------------------------------------------------- #
# Data -> groups
# --------------------------------------------------------------------------- #
def ett_frame(path):
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in df.columns if c.lower() == 'date']).astype(np.float64).values


def metr_frame(path, n_sensors=None):
    """METR-LA. n_sensors=None uses the whole 207-sensor network."""
    df = pd.read_hdf(path)
    V = np.asarray(df.values, dtype=np.float64)
    return V if n_sensors is None else V[:, :n_sensors]


def make_groups(V, L, P, horizon, target_col):
    """Mean-pool an L-step window into P patches per channel -> G = C*P groups."""
    T, C = V.shape
    step = L // P
    n = T - L - horizon + 1
    if n <= 0: raise ValueError('series too short')
    X = np.empty((n, C * P)); y = np.empty(n)
    for i in range(n):
        w = V[i:i + L]                              # L x C
        X[i] = w.reshape(P, step, C).mean(axis=1).T.reshape(-1)
        y[i] = V[i + L + horizon - 1, target_col]
    return X, y


# --------------------------------------------------------------------------- #
# Fit: y = u * (p(s)^T x) + b
# --------------------------------------------------------------------------- #
def fit(alpha, X, y):
    n, G = X.shape
    XtX = X.T @ X; Xty = X.T @ y; ybar = y.mean(); yy = y @ y

    def loss_and_grad(th):
        s = th[:G]; u = th[G]; b = th[G + 1]
        p = entmax(s, alpha)
        m1 = XtX @ p                       # X^T X p
        q = p @ Xty                        # sum_i (p^T x_i) y_i
        ss = p @ m1                        # sum_i (p^T x_i)^2
        sm = p @ X.sum(0)                  # sum_i p^T x_i
        loss = (u * u * ss + n * b * b + yy + 2 * u * b * sm - 2 * u * q - 2 * b * n * ybar) / n
        g_p = (2.0 / n) * (u * u * m1 + u * b * X.sum(0) - u * Xty)
        g_s = entmax_vjp(p, g_p, alpha)
        g_u = (2.0 / n) * (u * ss + b * sm - q)
        g_b = (2.0 / n) * (n * b + u * sm - n * ybar)
        return loss, np.concatenate([g_s, [g_u, g_b]])

    th0 = np.zeros(G + 2); th0[G] = 1.0; th0[G + 1] = float(ybar)
    bounds = [(-SCORE_BOUND, SCORE_BOUND)] * G + [(None, None), (None, None)]
    # alpha near 1 gives the stiff exponent 1/(alpha-1), so small alpha needs a
    # larger iteration budget; 800 left 22 of 95 alpha=1.25 fits at maxiter.
    res = minimize(loss_and_grad, th0, jac=True, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 20000, 'maxfun': 40000, 'ftol': 1e-14, 'gtol': 1e-10})
    return (entmax(res.x[:G], alpha), float(res.x[G]), float(res.x[G + 1]),
            bool(res.success), int(res.nit), float(res.fun))


def nmse(p, u, b, X, y):
    pred = u * (X @ p) + b
    return float(np.mean((pred - y) ** 2) / np.var(y))


def phase_coordinates(Xraw, X, y, ridge=1.0, share=0.90):
    """Read the phase-map coordinates (k/G, sigma_z) off the TRAINING split.

    k/G  -- the effective support: the fraction of groups carrying `share` of the
            total ridge coefficient mass.
    sigma_z -- the nuisance scale of the controlled study, estimated on the raw
            (unstandardized) groups as the spread of the off-support groups
            relative to the on-support ones, which is what sigma_z means when the
            relevant inputs are unit-variance.
    """
    G = X.shape[1]
    A = X.T @ X + ridge * np.eye(G)
    beta = np.linalg.solve(A, X.T @ y)
    order = np.argsort(np.abs(beta))[::-1]
    w = np.abs(beta)[order]
    c = np.cumsum(w) / max(w.sum(), 1e-12)
    k = int(np.searchsorted(c, share) + 1)
    sup = order[:k]; nui = order[k:]
    sd = Xraw.std(axis=0)
    s_sup = np.median(sd[sup]) if len(sup) else 1.0
    s_nui = np.median(sd[nui]) if len(nui) else 0.0
    return k / G, float(s_nui / max(s_sup, 1e-12))


def map_prediction(kfrac, n):
    """What the controlled phase map says for these coordinates."""
    if kfrac <= 0.25 and n >= 512: return 'win'
    if kfrac <= 0.125: return 'win'
    if kfrac >= 0.95: return 'tie'
    if n < 512: return 'lose'
    return 'tie'


# --------------------------------------------------------------------------- #
def run_dataset(name, V, target_col, L, P, horizons, folds):
    out = []
    for h in horizons:
        X, y = make_groups(V, L, P, h, target_col)
        n_all = len(y)
        for f in range(folds):
            # rolling origin: expanding train window, disjoint test block
            hi = int(n_all * (0.5 + 0.1 * f)); te = int(n_all * (0.5 + 0.1 * (f + 1)))
            Xtr, ytr, Xte, yte = X[:hi], y[:hi], X[hi:te], y[hi:te]
            if len(yte) < 50: continue
            Xraw = Xtr.copy()
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
            Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
            ym, ys = ytr.mean(), ytr.std() + 1e-12
            ytr = (ytr - ym) / ys; yte = (yte - ym) / ys
            kfrac, sigma_z = phase_coordinates(Xraw, Xtr, ytr)
            row = dict(dataset=name, horizon=h, fold=f, n_train=len(ytr), G=X.shape[1],
                       support_fraction=kfrac, sigma_z_hat=sigma_z,
                       map_prediction=map_prediction(kfrac, len(ytr)))
            for a in ALPHAS:
                p, u, b, ok, nit, tr_loss = fit(a, Xtr, ytr)
                del tr_loss
                row[f'nmse_{a}'] = nmse(p, u, b, Xte, yte)
                row[f'train_nmse_{a}'] = nmse(p, u, b, Xtr, ytr)
                row[f'zeros_{a}'] = int((p <= 1e-9).sum())
                row[f'converged_{a}'] = ok
                row[f'iters_{a}'] = nit
                row[f'maxw_{a}'] = float(p.max())
            out.append(row)
            print(f"  {name} h={h} fold={f} n={len(ytr)} k/G={kfrac:.3f} "
                  f"pred={row['map_prediction']} "
                  + " ".join(f"a{a}={row[f'nmse_{a}']:.4f}" for a in ALPHAS), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ett-dir', default=None, help='directory with ETTh1.csv / ETTh2.csv')
    ap.add_argument('--ettm-dir', default=None, help='directory with ETTm1.csv / ETTm2.csv')
    ap.add_argument('--metr', default=None, help='path to metr-la.h5')
    ap.add_argument('--metr-sensors', type=int, default=None,
                    help='use only the first N sensors; default is the full 207')
    ap.add_argument('--window', type=int, default=96)
    ap.add_argument('--patches', type=int, default=12)
    ap.add_argument('--horizons', nargs='*', type=int, default=[96, 192, 336, 720])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--out', default=None)
    ap.add_argument('--score-bound', type=float, default=None,
                    help='clip scores to [-B, B]; large B lets softmax concentrate freely')
    args = ap.parse_args()

    global SCORE_BOUND
    if args.score_bound is not None: SCORE_BOUND = args.score_bound
    rows = []; t0 = time.time()
    for d, names in ((args.ett_dir, ['ETTh1', 'ETTh2']), (args.ettm_dir, ['ETTm1', 'ETTm2'])):
        if not d: continue
        for nm in names:
            f = Path(d) / f'{nm}.csv'
            if not f.exists(): print('skip (missing)', f); continue
            V = ett_frame(f)
            print(f'{nm}: {V.shape[0]} steps x {V.shape[1]} channels', flush=True)
            rows += run_dataset(nm, V, V.shape[1] - 1, args.window, args.patches,
                                args.horizons, args.folds)
    if args.metr and Path(args.metr).exists():
        V = metr_frame(args.metr, args.metr_sensors)
        print(f'METR-LA: {V.shape[0]} steps x {V.shape[1]} sensors', flush=True)
        rows += run_dataset('METR-LA', V, 0, 24, 12, [3, 6, 12], args.folds)

    if not rows: raise SystemExit('no datasets were run')
    df = pd.DataFrame(rows)
    out = args.out or str(Path(__file__).resolve().parents[2] / 'results' / 'boundary_benchmarks.csv')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    json.dump(rows, open(out.replace('.csv', '.json'), 'w'), indent=2)
    print(f'\nwrote {out}  ({len(df)} rows, {time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
