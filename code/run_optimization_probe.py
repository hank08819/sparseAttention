#!/usr/bin/env python3
"""Is the gap to the convex optimum an expressiveness limit or an optimization path?

For any p on the simplex, sparsemax(p) = p, and every entry of p lies in [0,1],
so with |s| <= 2 the sparsemax parametrization already spans the whole simplex:
the bound cannot be what keeps a fit away from the SIMPLEX optimum. This script
fits the same cells from three initializations and compares training MSE:

  zeros     the initialization used throughout the paper
  simplex   s = p*, the SIMPLEX optimum's own score vector
  best-of-N N random Dirichlet starts

Writes results/optimization_probe.json.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'package' / 'misa_sparsity'))
import run_entmax_sweep as sw
import importlib.util
_c = importlib.util.spec_from_file_location('cr', ROOT / 'scripts' / 'run_convex_reference.py')
cr = importlib.util.module_from_spec(_c); _c.loader.exec_module(cr)
G = sw.G; SEEDS = range(20); NSTART = 8
CELLS = [(2, 1.0, 512), (4, 1.0, 128), (8, 1.0, 128), (16, 1.0, 128), (16, 2.0, 128), (32, 1.0, 128)]

def fit_from(X, y, s0, B):
    n = X.shape[0]; XtX = X.T @ X; Xty = X.T @ y
    def lg(s):
        p = sw.entmax(s, 2.0)
        return (p @ XtX @ p - 2 * p @ Xty + y @ y) / n, sw.entmax_vjp(p, (2.0 / n) * (XtX @ p - Xty), 2.0)
    r = minimize(lg, s0, jac=True, method='L-BFGS-B', bounds=[(-B, B)] * G,
                 options={'maxiter': 20000, 'maxfun': 40000, 'ftol': 1e-15, 'gtol': 1e-12})
    return sw.entmax(r.x, 2.0), float(r.fun)

def main():
    out = {}
    print(f"{'cell':<18}{'B':>2} | training MSE gap to SIMPLEX: {'zeros':>10}{'from p*':>11}{'best-of-8':>11} | {'test excess: zeros':>19}{'p*-init':>10}{'SIMPLEX':>9}")
    for k, sig, n in CELLS:
        for B in (2.0, 8.0):
            g0, gp, gr, t0, tp, ts, reach = [], [], [], [], [], [], 0
            for seed in SEEDS:
                X, y = sw.sample_data(np.random.default_rng(seed), n, k, sig)
                sw.SCORE_BOUND = B
                p_qp, f_qp = cr.qp(X, y)
                tm = lambda p: float(np.mean((X @ p - y) ** 2))
                pz, fz = fit_from(X, y, np.zeros(G), B)
                pp, fp = fit_from(X, y, np.clip(p_qp, -B, B), B)     # s = p*, exactly representable
                best = None
                for j in range(NSTART):
                    r = np.random.default_rng(500 + 13 * seed + j).dirichlet(np.ones(G))
                    _, f = fit_from(X, y, np.clip(r, -B, B), B)
                    best = f if best is None else min(best, f)
                g0.append(fz - f_qp); gp.append(fp - f_qp); gr.append(best - f_qp)
                ev = lambda p: sw.evaluate(p, k, sig, np.random.default_rng(10_000 + seed))[0]
                t0.append(ev(pz)); tp.append(ev(pp)); ts.append(ev(p_qp))
                reach += abs(fp - f_qp) < 1e-10
            key = f'k{k}_s{sig}_n{n}_B{int(B)}'
            out[key] = dict(k=k, sigma_z=sig, n=n, B=B,
                            gap_zeros=float(np.mean(g0)), gap_from_pstar=float(np.mean(gp)),
                            gap_best_of_8=float(np.mean(gr)), reached_optimum_from_pstar=int(reach),
                            test_zeros=float(np.mean(t0)), test_pstar=float(np.mean(tp)), test_simplex=float(np.mean(ts)))
            print(f"k{k} s{sig} n{n:<6}{int(B):>2} | {np.mean(g0):>29.2e}{np.mean(gp):>11.2e}{np.mean(gr):>11.2e} | "
                  f"{np.mean(t0):>19.6f}{np.mean(tp):>10.6f}{np.mean(ts):>9.6f}   reached p* {reach}/20", flush=True)
    json.dump(out, open(ROOT / 'results' / 'optimization_probe.json', 'w'), indent=1)
    print('wrote results/optimization_probe.json')

if __name__ == '__main__': main()
