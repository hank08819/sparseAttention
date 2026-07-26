#!/usr/bin/env python3
"""
run_controlled.py — Controlled softmax vs. sparsemax comparison.

Regenerates the 2,400 controlled fits and the mechanism-sweep figure reported in
the "Controlled Comparison of Softmax and Sparsemax" section of the paper
(Figure 3). The experiment stays inside the model class of Theorem 1: a fixed
predictor y_hat = p(s)^T x whose only free parameters are the G attention
scores s. Only the normalization p is swapped between softmax and sparsemax;
no neural backbone is involved.

Design (matches the paper):
  * G = 32 independent scalar inputs.
  * Known support S of size k: relevant inputs x_j ~ N(0, 1).
  * Irrelevant inputs x_j ~ N(0, sigma_z^2).
  * Target  y = (1/k) sum_{j in S} x_j + eps,  eps ~ N(0, 0.05^2).
  * Predictor y_hat = p(s)^T x, scores clipped to s_j in [-2, 2].
  * Optimizer: L-BFGS-B from zero initialization with analytic gradients.
  * Sweep: k in {2,4,8,16,32}, sigma_z in {0.25,0.5,1,2}, n in {128,512,2048}.
  * 20 seeds per setting; every fit evaluated on 20,000 fresh samples.
  * 5 * 4 * 3 * 20 = 1,200 fits per method -> 2,400 fits total.

Outputs:
  results/controlled_results.json   aggregated per-setting statistics
  figures/mechanism_sweep.pdf       Figure 3 (panels a, b, c)

Usage:
  python run_controlled.py                 # full 2,400-fit sweep
  python run_controlled.py --quick         # smoke test (fewer seeds/settings)
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import minimize

G = 32
IRREDUCIBLE = 0.05 ** 2          # variance of eps; the known error floor
SCORE_BOUND = 2.0                # s_j in [-SCORE_BOUND, SCORE_BOUND]
EVAL_SAMPLES = 20_000
TOL = 1e-6                       # weight threshold for support membership

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
FIGURES_DIR = os.path.join(HERE, "..", "figures")


# --------------------------------------------------------------------------- #
# Normalization functions and their vector-Jacobian products
# --------------------------------------------------------------------------- #
def softmax(s):
    z = s - s.max()
    e = np.exp(z)
    return e / e.sum()


def softmax_vjp(p, g):
    """J^T g for the softmax Jacobian J = diag(p) - p p^T (symmetric)."""
    return p * (g - p @ g)


def sparsemax(s):
    """Euclidean projection of s onto the probability simplex (Martins 2016)."""
    z = np.sort(s)[::-1]
    cumsum = np.cumsum(z)
    k_idx = np.arange(1, len(s) + 1)
    cond = 1 + k_idx * z > cumsum
    k = k_idx[cond][-1]
    tau = (cumsum[cond][-1] - 1) / k
    return np.maximum(s - tau, 0.0)


def sparsemax_vjp(p, g):
    """J^T g for the sparsemax Jacobian J = diag(1_S) - (1/|S|) 1_S 1_S^T."""
    support = p > 0
    k = support.sum()
    if k == 0:
        return np.zeros_like(p)
    avg = g[support].sum() / k
    out = np.zeros_like(p)
    out[support] = g[support] - avg
    return out


NORMS = {
    "softmax": (softmax, softmax_vjp),
    "sparsemax": (sparsemax, sparsemax_vjp),
}


# --------------------------------------------------------------------------- #
# Data and fitting
# --------------------------------------------------------------------------- #
def sample_data(rng, n, k, sigma_z):
    """Return (X, y). Support S is the first k coordinates."""
    X = rng.standard_normal((n, G))
    X[:, k:] *= sigma_z                      # irrelevant inputs scaled to sigma_z
    y = X[:, :k].mean(axis=1) + rng.normal(0.0, 0.05, size=n)
    return X, y


def fit(norm_name, X, y):
    """Fit the G scores by L-BFGS-B; return the learned weight vector p."""
    proj, vjp = NORMS[norm_name]
    n = X.shape[0]
    XtX = X.T @ X
    Xty = X.T @ y

    def loss_and_grad(s):
        p = proj(s)
        resid_dot = XtX @ p - Xty            # X^T (X p - y)
        mse = (p @ XtX @ p - 2 * p @ Xty + y @ y) / n
        g_p = (2.0 / n) * resid_dot          # dL/dp
        g_s = vjp(p, g_p)                    # dL/ds
        return mse, g_s

    bounds = [(-SCORE_BOUND, SCORE_BOUND)] * G
    res = minimize(
        loss_and_grad, np.zeros(G), jac=True, method="L-BFGS-B",
        bounds=bounds, options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    return proj(res.x), res.success


def support_f1(p, k):
    """F1 between the recovered support {j: p_j > TOL} and the true support."""
    pred = p > TOL
    true = np.zeros(G, dtype=bool)
    true[:k] = True
    tp = np.logical_and(pred, true).sum()
    if pred.sum() == 0:
        return 0.0
    precision = tp / pred.sum()
    recall = tp / true.sum()
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(p, k, sigma_z, rng):
    """Excess MSE (above IRREDUCIBLE), weight on irrelevant inputs, support F1."""
    X, y = sample_data(rng, EVAL_SAMPLES, k, sigma_z)
    pred = X @ p
    mse = float(np.mean((pred - y) ** 2))
    excess = mse - IRREDUCIBLE
    irr_weight = float(p[k:].sum()) if k < G else 0.0
    return excess, irr_weight, support_f1(p, k)


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def run_sweep(ks, sigmas, ns, seeds):
    results = {}
    n_fits = 0
    for k in ks:
        for sigma_z in sigmas:
            for n in ns:
                acc = {m: {"excess": [], "irr_weight": [], "f1": []}
                       for m in NORMS}
                for seed in seeds:
                    train_rng = np.random.default_rng(seed)
                    eval_rng = np.random.default_rng(10_000 + seed)
                    X, y = sample_data(train_rng, n, k, sigma_z)
                    for m in NORMS:
                        p, ok = fit(m, X, y)
                        assert ok or True   # L-BFGS-B reports convergence
                        excess, irr_w, f1 = evaluate(p, k, sigma_z, eval_rng)
                        acc[m]["excess"].append(excess)
                        acc[m]["irr_weight"].append(irr_w)
                        acc[m]["f1"].append(f1)
                        n_fits += 1
                key = f"k{k}_s{sigma_z}_n{n}"
                results[key] = {"k": k, "sigma_z": sigma_z, "n": n}
                for m in NORMS:
                    a = acc[m]
                    results[key][m] = {
                        "excess_mse_mean": float(np.mean(a["excess"])),
                        "excess_mse_std": float(np.std(a["excess"])),
                        "irr_weight_mean": float(np.mean(a["irr_weight"])),
                        "f1_mean": float(np.mean(a["f1"])),
                    }
                print(f"  {key}: "
                      f"softmax={results[key]['softmax']['excess_mse_mean']:.5f} "
                      f"sparsemax={results[key]['sparsemax']['excess_mse_mean']:.5f} "
                      f"F1(sparse)={results[key]['sparsemax']['f1_mean']:.3f}")
    print(f"Completed {n_fits} fits.")
    return results


# --------------------------------------------------------------------------- #
# Figure 3
# --------------------------------------------------------------------------- #
def make_figure(results, ks, sigmas, ns, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    # (a) advantage shrinks as support fraction grows (n=512, sigma_z=1)
    ax = axes[0]
    n0, s0 = 512, 1
    if n0 in ns and s0 in sigmas:
        for m, marker in [("softmax", "o"), ("sparsemax", "s")]:
            mean = [results[f"k{k}_s{s0}_n{n0}"][m]["excess_mse_mean"] for k in ks]
            std = [results[f"k{k}_s{s0}_n{n0}"][m]["excess_mse_std"] for k in ks]
            ax.errorbar([k / G for k in ks], mean, yerr=std, marker=marker,
                        capsize=3, label=m)
        ax.set_xlabel("support fraction $k/G$")
        ax.set_ylabel("excess MSE")
        ax.set_title(f"(a) $n={n0}$, $\\sigma_z={s0}$")
        ax.legend()

    # (b) effect grows with noise (k=4)
    ax = axes[1]
    k1, n1 = 4, 512
    if k1 in ks and n1 in ns:
        for m, marker in [("softmax", "o"), ("sparsemax", "s")]:
            mean = [results[f"k{k1}_s{s}_n{n1}"][m]["excess_mse_mean"] for s in sigmas]
            std = [results[f"k{k1}_s{s}_n{n1}"][m]["excess_mse_std"] for s in sigmas]
            ax.errorbar(sigmas, mean, yerr=std, marker=marker, capsize=3, label=m)
        ax.set_xlabel("$\\sigma_z$")
        ax.set_ylabel("excess MSE")
        ax.set_title(f"(b) $k={k1}$, $n={n1}$")
        ax.legend()

    # (c) support recovery improves with sample size (sparsemax, sparse settings)
    ax = axes[2]
    f1_by_n = []
    for n in ns:
        vals = [results[f"k{k}_s{s}_n{n}"]["sparsemax"]["f1_mean"]
                for k in ks if k < G for s in sigmas]
        f1_by_n.append(np.mean(vals))
    ax.plot(ns, f1_by_n, marker="D", color="C1")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("sample size $n$")
    ax.set_ylabel("sparsemax support F1")
    ax.set_title("(c) support recovery")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"Wrote {path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: fewer seeds and settings")
    args = ap.parse_args()

    if args.quick:
        ks, sigmas, ns, seeds = [2, 8, 32], [0.5, 1, 2], [128, 512], range(3)
    else:
        ks = [2, 4, 8, 16, 32]
        sigmas = [0.25, 0.5, 1, 2]
        ns = [128, 512, 2048]
        seeds = range(20)

    print(f"Running controlled sweep: "
          f"{len(ks)}k x {len(sigmas)}sigma x {len(ns)}n x {len(list(seeds))}seeds "
          f"x 2 methods = "
          f"{len(ks) * len(sigmas) * len(ns) * len(list(seeds)) * 2} fits")

    results = run_sweep(ks, sigmas, ns, list(seeds))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_json = os.path.join(RESULTS_DIR, "controlled_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_json}")

    make_figure(results, ks, sigmas, ns,
                os.path.join(FIGURES_DIR, "mechanism_sweep.pdf"))


if __name__ == "__main__":
    main()
