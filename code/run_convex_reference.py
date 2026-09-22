#!/usr/bin/env python3
"""Convex references for the controlled linear cells (statistical vs optimization failure).

Model of the controlled study: y_hat = p^T x with p on the simplex. The training
MSE is a convex quadratic in p, so two exact references exist that need no
knowledge of the true support:
  SIMPLEX  : argmin_p ||Xp - y||^2 / n  s.t. p >= 0, 1^T p = 1        (what sparsemax can reach)
  RATIO(B) : the same with p_i >= e^{-2B} max_j p_j                   (what softmax with |s| <= B can reach)
Both are solved with SLSQP.  Compared against the L-BFGS-B fits of the sweep
(softmax and sparsemax parametrizations, scores clipped to [-B, B]) on the SAME
training sample per seed: training objective gap, test excess MSE, support F1.
Writes results/convex_reference.json.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'package' / 'misa_sparsity'))
import run_entmax_sweep as sw

CELLS = [(2, 1.0, 512, 'win'), (2, 2.0, 128, 'win'), (4, 1.0, 128, 'win'), (8, 1.0, 512, 'boundary'),
         (8, 1.0, 128, 'lose'), (16, 2.0, 128, 'lose'), (16, 1.0, 128, 'lose'), (32, 1.0, 128, 'dense')]
BS = (2.0, 8.0); SEEDS = range(20); G = sw.G

def qp(X, y, ratio_bound=None):
    n = len(y); XtX = X.T @ X / n; Xty = X.T @ y / n; yy = y @ y / n
    if ratio_bound is None:
        f = lambda p: (p @ XtX @ p - 2 * p @ Xty + yy, 2 * (XtX @ p - Xty))
        cons = [{'type': 'eq', 'fun': lambda p: p.sum() - 1, 'jac': lambda p: np.ones(G)}]
        res = minimize(f, np.full(G, 1 / G), jac=True, method='SLSQP', bounds=[(0, 1)] * G, constraints=cons,
                       options={'maxiter': 2000, 'ftol': 1e-15})
        return res.x, float(res.fun)
    r = np.exp(-2 * ratio_bound)                       # p_i >= r * t,  p_i <= t,  t = max_j p_j
    def f(z):
        p = z[:G]; return (p @ XtX @ p - 2 * p @ Xty + yy, np.concatenate([2 * (XtX @ p - Xty), [0.0]]))
    cons = [{'type': 'eq', 'fun': lambda z: z[:G].sum() - 1, 'jac': lambda z: np.concatenate([np.ones(G), [0.0]])},
            {'type': 'ineq', 'fun': lambda z: z[G] - z[:G], 'jac': lambda z: np.hstack([-np.eye(G), np.ones((G, 1))])},
            {'type': 'ineq', 'fun': lambda z: z[:G] - r * z[G], 'jac': lambda z: np.hstack([np.eye(G), -r * np.ones((G, 1))])}]
    z0 = np.concatenate([np.full(G, 1 / G), [1 / G]])
    res = minimize(f, z0, jac=True, method='SLSQP', bounds=[(0, 1)] * (G + 1), constraints=cons,
                   options={'maxiter': 2000, 'ftol': 1e-15})
    return res.x[:G], float(res.fun)

def fit_long(alpha, X, y, B):
    sw.SCORE_BOUND = B; n = X.shape[0]; XtX = X.T @ X; Xty = X.T @ y
    def lg(s):
        p = sw.entmax(s, alpha); return (p @ XtX @ p - 2 * p @ Xty + y @ y) / n, sw.entmax_vjp(p, (2.0 / n) * (XtX @ p - Xty), alpha)
    res = minimize(lg, np.zeros(G), jac=True, method='L-BFGS-B', bounds=[(-B, B)] * G,
                   options={'maxiter': 20000, 'maxfun': 40000, 'ftol': 1e-15, 'gtol': 1e-12})
    return sw.entmax(res.x, alpha)

def train_mse(p, X, y): return float(np.mean((X @ p - y) ** 2))

def main():
    out = {}; t0 = time.time()
    hdr = f"{'cell':<16}{'B':>3} | {'train gap: sp-fit vs SIMPLEX':>30} {'soft-fit vs RATIO':>18} | {'test excess  soft   sparse  RATIO  SIMPLEX':>44} | {'F1 sp':>6}{'F1 SIMPLEX':>11} | {'regime fit / ref':>17}"
    print(hdr)
    for k, sig, n, lab in CELLS:
        for B in BS:
            rec = {m: [] for m in ['tr_soft', 'tr_sp', 'tr_sp_long', 'tr_simplex', 'tr_ratio', 'te_soft', 'te_sp', 'te_simplex', 'te_ratio', 'f1_sp', 'f1_simplex']}
            for seed in SEEDS:
                X, y = sw.sample_data(np.random.default_rng(seed), n, k, sig)
                sw.SCORE_BOUND = B
                p_soft, _ = sw.fit(1.0, X, y); p_sp, _ = sw.fit(2.0, X, y); p_sp_long = fit_long(2.0, X, y, B)
                p_qp, _ = qp(X, y); p_rb, _ = qp(X, y, ratio_bound=B)
                for name, p in [('soft', p_soft), ('sp', p_sp), ('simplex', p_qp), ('ratio', p_rb)]:
                    rec['tr_' + name].append(train_mse(p, X, y))
                    ex, irr, f1, nz = sw.evaluate(p, k, sig, np.random.default_rng(10_000 + seed)); rec['te_' + name].append(ex)
                    if name in ('sp', 'simplex'): rec['f1_' + name].append(f1)
                rec['tr_sp_long'].append(train_mse(p_sp_long, X, y))
            R = {m: np.array(v) for m, v in rec.items()}
            gap_sp = R['tr_sp'] - R['tr_simplex']; gap_sp_long = R['tr_sp_long'] - R['tr_simplex']; gap_soft = R['tr_soft'] - R['tr_ratio']
            d_fit = R['te_soft'] - R['te_sp']; d_ref = R['te_ratio'] - R['te_simplex']
            reg = lambda d: sw.regime(float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))))
            key = f'k{k}_s{sig}_n{n}_B{int(B)}'
            out[key] = dict(k=k, sigma_z=sig, n=n, B=B, label=lab,
                            train_gap_sparsemax_fit=float(gap_sp.mean()), train_gap_sparsemax_longfit=float(gap_sp_long.mean()),
                            train_gap_softmax_fit_vs_ratio=float(gap_soft.mean()),
                            test_excess={m: float(R['te_' + m].mean()) for m in ['soft', 'sp', 'simplex', 'ratio']},
                            f1_sparsemax=float(R['f1_sp'].mean()), f1_simplex=float(R['f1_simplex'].mean()),
                            delta_fit=float(d_fit.mean()), delta_fit_se=float(d_fit.std(ddof=1) / np.sqrt(len(d_fit))), regime_fit=reg(d_fit),
                            delta_ref=float(d_ref.mean()), delta_ref_se=float(d_ref.std(ddof=1) / np.sqrt(len(d_ref))), regime_ref=reg(d_ref))
            o = out[key]
            print(f"{lab+f' k{k} s{sig} n{n}':<16}{int(B):>3} | {gap_sp.mean():>14.2e} (long {gap_sp_long.mean():>9.1e}) {gap_soft.mean():>18.2e} | "
                  f"{R['te_soft'].mean():>10.5f} {R['te_sp'].mean():>8.5f} {R['te_ratio'].mean():>7.5f} {R['te_simplex'].mean():>8.5f} | "
                  f"{o['f1_sparsemax']:>6.2f}{o['f1_simplex']:>11.2f} | {o['regime_fit']:>7} / {o['regime_ref']:<7}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open(ROOT / 'results' / 'convex_reference.json', 'w'), indent=1); print('wrote results/convex_reference.json')

if __name__ == '__main__': main()
