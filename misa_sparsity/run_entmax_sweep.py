#!/usr/bin/env python3
"""
run_entmax_sweep.py -- alpha-entmax sweep over the controlled grid (P0-8).

The manuscript compares only the two endpoints of the entmax family, softmax
(alpha = 1) and sparsemax (alpha = 2), and states in Related Work that equality
of the endpoints does not determine the interior alpha path. This script tests
the interior directly.

It reuses the exact controlled design of run_controlled.py -- the same data
generator, the same predictor y_hat = p(s)^T x, the same L-BFGS-B fit from zero
initialization with scores clipped to [-2, 2], the same 20 seeds, and the same
20,000 fresh evaluation samples -- and changes nothing but the normalization:

    alpha in {1.0, 1.25, 1.5, 1.75, 2.0}

Within a seed all alphas see the identical training sample, so the comparison is
matched fit for fit.

alpha-entmax (Peters et al., 2019) is
    p(s) = [ (alpha - 1) s - tau ]_+ ^ (1 / (alpha - 1)),
with tau set so that the entries sum to one; alpha = 1 recovers softmax and
alpha = 2 recovers sparsemax. Its Jacobian is
    J = diag(u) - u u^T / sum(u),      u_i = p_i ^ (2 - alpha),
which reduces to the softmax and sparsemax Jacobians at the endpoints, so all
five settings are optimized with analytic gradients on equal footing.

Outputs:
  results/entmax_sweep.json   per-cell, per-alpha statistics and regime labels

Usage:
  python run_entmax_sweep.py            # full 60-cell x 20-seed x 5-alpha sweep
  python run_entmax_sweep.py --quick    # smoke test
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.optimize import minimize

G = 32
IRREDUCIBLE = 0.05 ** 2
SCORE_BOUND = 2.0     # overridden by --score-bound
EVAL_SAMPLES = 20_000
TOL = 1e-6
FLOOR = 1e-4          # practical-significance margin, as in the phase map
BISECT_ITERS = 60

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")

ALPHAS = [1.0, 1.25, 1.5, 1.75, 2.0]
# The original grid stopped at n = 2048. The standard benchmarks of
# run_boundary_benchmarks.py supply 8,302 to 62,540 training windows, so the
# sample-size axis is extended to cover them and a real-data setting can be
# looked up on the map rather than extrapolated from its edge.
NS = [128, 512, 2048, 8192, 32768]


# --------------------------------------------------------------------------- #
# alpha-entmax and its vector-Jacobian product
# --------------------------------------------------------------------------- #
def softmax(s):
    e = np.exp(s - s.max())
    return e / e.sum()


def entmax(s, alpha):
    """alpha-entmax by bisection on the threshold tau."""
    if alpha == 1.0:
        return softmax(s)
    a1 = alpha - 1.0
    z = a1 * s
    inv = 1.0 / a1
    hi = z.max()              # every entry clipped to zero -> sum 0 < 1
    lo = hi - 1.0             # largest entry equals 1      -> sum >= 1
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        total = np.power(np.maximum(z - mid, 0.0), inv).sum()
        if total > 1.0:
            lo = mid
        else:
            hi = mid
    p = np.power(np.maximum(z - 0.5 * (lo + hi), 0.0), inv)
    ssum = p.sum()
    return p / ssum if ssum > 0 else np.full(G, 1.0 / G)


def entmax_vjp(p, g, alpha):
    """J^T g for J = diag(u) - u u^T / sum(u) with u_i = p_i^(2 - alpha)."""
    if alpha == 1.0:
        return p * (g - p @ g)
    u = np.power(p, 2.0 - alpha, where=p > 0, out=np.zeros_like(p))
    usum = u.sum()
    if usum <= 0:
        return np.zeros_like(p)
    return u * g - (u @ g) / usum * u


# --------------------------------------------------------------------------- #
# Data and fitting (identical to run_controlled.py apart from the projection)
# --------------------------------------------------------------------------- #
def sample_data(rng, n, k, sigma_z):
    X = rng.standard_normal((n, G))
    X[:, k:] *= sigma_z
    y = X[:, :k].mean(axis=1) + rng.normal(0.0, 0.05, size=n)
    return X, y


def fit(alpha, X, y):
    n = X.shape[0]
    XtX = X.T @ X
    Xty = X.T @ y

    def loss_and_grad(s):
        p = entmax(s, alpha)
        mse = (p @ XtX @ p - 2 * p @ Xty + y @ y) / n
        g_p = (2.0 / n) * (XtX @ p - Xty)
        return mse, entmax_vjp(p, g_p, alpha)

    res = minimize(
        loss_and_grad, np.zeros(G), jac=True, method="L-BFGS-B",
        bounds=[(-SCORE_BOUND, SCORE_BOUND)] * G,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    return entmax(res.x, alpha), bool(res.success)


def support_f1(p, k):
    pred = p > TOL
    true = np.zeros(G, dtype=bool)
    true[:k] = True
    if pred.sum() == 0:
        return 0.0
    tp = np.logical_and(pred, true).sum()
    precision = tp / pred.sum()
    recall = tp / true.sum()
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def evaluate(p, k, sigma_z, rng):
    X, y = sample_data(rng, EVAL_SAMPLES, k, sigma_z)
    mse = float(np.mean((X @ p - y) ** 2))
    irr_weight = float(p[k:].sum()) if k < G else 0.0
    n_zero = int((p <= TOL).sum())
    return mse - IRREDUCIBLE, irr_weight, support_f1(p, k), n_zero


def regime(diff, se):
    """Win/tie/lose rule of the phase map, applied against the alpha = 1 baseline."""
    if diff > 2 * se and diff > FLOOR:
        return "win"
    if diff < -2 * se and diff < -FLOOR:
        return "lose"
    return "tie"


# --------------------------------------------------------------------------- #
def run_sweep(ks, sigmas, ns, seeds, alphas):
    results = {}
    n_fits = 0
    t0 = time.time()
    for k in ks:
        for sigma_z in sigmas:
            for n in ns:
                acc = {a: {"excess": [], "irr": [], "f1": [], "zeros": []} for a in alphas}
                fails = 0
                for seed in seeds:
                    train_rng = np.random.default_rng(seed)
                    X, y = sample_data(train_rng, n, k, sigma_z)
                    for a in alphas:
                        # eval stream fixed per (seed, cell) so alphas are matched
                        eval_rng = np.random.default_rng(10_000 + seed)
                        p, ok = fit(a, X, y)
                        fails += (not ok)
                        ex, irr, f1, nz = evaluate(p, k, sigma_z, eval_rng)
                        acc[a]["excess"].append(ex)
                        acc[a]["irr"].append(irr)
                        acc[a]["f1"].append(f1)
                        acc[a]["zeros"].append(nz)
                        n_fits += 1
                key = f"k{k}_s{sigma_z}_n{n}"
                cell = {"k": k, "sigma_z": sigma_z, "n": n, "nonconverged": fails}
                base = np.array(acc[alphas[0]]["excess"])   # alpha = 1 (softmax)
                for a in alphas:
                    e = np.array(acc[a]["excess"])
                    d = base - e                            # positive favors alpha
                    # per-seed values are kept: at n = 128 the excess-MSE
                    # distribution across seeds is heavy-tailed (a few seeds pick
                    # a badly wrong support), so cell means alone misdescribe the
                    # scarce-data corner.
                    cell[str(a)] = {
                        "excess_per_seed": [float(v) for v in e],
                        "f1_per_seed": [float(v) for v in acc[a]["f1"]],
                        "excess_mse_mean": float(e.mean()),
                        "excess_mse_median": float(np.median(e)),
                        "excess_mse_std": float(e.std()),
                        "irr_weight_mean": float(np.mean(acc[a]["irr"])),
                        "f1_mean": float(np.mean(acc[a]["f1"])),
                        "exact_zeros_mean": float(np.mean(acc[a]["zeros"])),
                        "diff_vs_softmax_mean": float(d.mean()),
                        "diff_vs_softmax_se": float(d.std(ddof=1) / np.sqrt(len(d))),
                        "regime_vs_softmax": regime(
                            float(d.mean()),
                            float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0),
                    }
                results[key] = cell
                msg = "  ".join(f"a={a}:{cell[str(a)]['excess_mse_mean']:.5f}" for a in alphas)
                print(f"  {key}  {msg}   ({time.time()-t0:.0f}s)", flush=True)
    print(f"Completed {n_fits} fits in {time.time()-t0:.0f}s.")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ns", nargs="*", type=int, default=None,
                    help="sample sizes; default is the extended grid")
    ap.add_argument("--out", default=None)
    ap.add_argument("--score-bound", type=float, default=None,
                    help="clip scores to [-B, B]; the reported grid uses B = 2")
    args = ap.parse_args()
    if args.quick:
        ks, sigmas, ns, seeds = [2, 8], [1], [512], range(3)
    else:
        ks, sigmas, ns, seeds = [2, 4, 8, 16, 32], [0.25, 0.5, 1, 2], NS, range(20)
    print(f"alpha-entmax sweep: {len(ks)}k x {len(sigmas)}sigma x {len(ns)}n "
          f"x {len(list(seeds))}seeds x {len(ALPHAS)}alpha = "
          f"{len(ks)*len(sigmas)*len(ns)*len(list(seeds))*len(ALPHAS)} fits", flush=True)
    global SCORE_BOUND
    if args.score_bound is not None: SCORE_BOUND = args.score_bound
    if args.ns: ns = args.ns
    results = run_sweep(ks, sigmas, ns, list(seeds), ALPHAS)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = args.out or os.path.join(RESULTS_DIR, "entmax_sweep.json")
    with open(out, "w") as f:
        json.dump({"alphas": ALPHAS, "cells": results}, f, indent=2)
    print("Wrote", out)


if __name__ == "__main__":
    main()
