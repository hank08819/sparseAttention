"""
run_controlled_temporal.py — Does the WIN/TIE/LOSE phase map survive
time-series realism?

The controlled study draws i.i.d. rows, which a reviewer can reasonably say is
not representative of multivariate time series. This extension re-runs the
matched softmax-vs-sparsemax comparison with the two most salient time-series
violations, holding the rest of the protocol identical (same predictor class,
optimizer, bounds, 20 seeds, 20k-sample evaluation):

  AR(1) inputs   x_{j,t} = phi * x_{j,t-1} + sqrt(1-phi^2) e_{j,t}
                 with phi in {0.5, 0.9}; training rows are CONSECUTIVE time
                 steps, so they are serially dependent (effective n < n).
  Heavy tails    e ~ t(3)/sqrt(3) (unit variance, crypto-like tails),
                 with phi = 0.5.

Target: y_t = mean_{j in S} x_{j,t} + eps_t, eps ~ N(0, 0.05^2), as before.
Grid: k in {2, 8, 32}, sigma_z in {0.5, 2}, n = 512.

Output: results/controlled_temporal.json and a printed WIN/TIE/LOSE table to
compare against the i.i.d. phase map.
"""
import json
import os

import numpy as np

from msca_sparsity.run_controlled import (G, IRREDUCIBLE, fit, support_f1,
                                          EVAL_SAMPLES)

KS = [2, 8, 32]
SIGMAS = [0.5, 2]
N = 512
SEEDS = range(20)
VARIANTS = [("ar05", 0.5, "gauss"), ("ar09", 0.9, "gauss"),
            ("ar05_t3", 0.5, "t3")]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "controlled_temporal.json")


def sample_series(rng, n, k, sigma_z, phi, tails):
    """AR(1) multivariate series; rows are consecutive time steps."""
    def innov(size):
        if tails == "t3":
            return rng.standard_t(3, size=size) / np.sqrt(3.0)
        return rng.standard_normal(size)

    burn = 100
    T = n + burn
    e = innov((T, G))
    X = np.zeros((T, G))
    X[0] = e[0]
    c = np.sqrt(1.0 - phi ** 2)
    for t in range(1, T):
        X[t] = phi * X[t - 1] + c * e[t]
    X = X[burn:]
    X[:, k:] *= sigma_z
    y = X[:, :k].mean(axis=1) + rng.normal(0.0, 0.05, size=n)
    return X, y


def evaluate_series(p, k, sigma_z, phi, tails, rng):
    X, y = sample_series(rng, EVAL_SAMPLES, k, sigma_z, phi, tails)
    mse = float(np.mean((X @ p - y) ** 2))
    return mse - IRREDUCIBLE


def main():
    res = {}
    for vname, phi, tails in VARIANTS:
        for k in KS:
            for s in SIGMAS:
                accs = {"softmax": [], "sparsemax": []}
                f1s = []
                for seed in SEEDS:
                    tr_rng = np.random.default_rng(seed)
                    ev_rng = np.random.default_rng(10_000 + seed)
                    X, y = sample_series(tr_rng, N, k, s, phi, tails)
                    for m in accs:
                        p, _ = fit(m, X, y)
                        accs[m].append(evaluate_series(p, k, s, phi, tails,
                                                       ev_rng))
                        if m == "sparsemax":
                            f1s.append(support_f1(p, k))
                key = f"{vname}_k{k}_s{s}"
                soft = np.array(accs["softmax"]); spar = np.array(accs["sparsemax"])
                diff = soft.mean() - spar.mean()
                se = np.sqrt((soft.std() ** 2 + spar.std() ** 2) / len(SEEDS))
                if diff > 2 * se and diff > 1e-4:
                    reg = "WIN"
                elif diff < -2 * se and diff < -1e-4:
                    reg = "LOSE"
                else:
                    reg = "TIE"
                res[key] = {"variant": vname, "phi": phi, "tails": tails,
                            "k": k, "sigma_z": s, "n": N,
                            "diff_mean": float(diff), "diff_2se": float(2 * se),
                            "sparse_f1": float(np.mean(f1s)), "regime": reg}
                print(f"{vname:8s} k={k:2d} s={s:4} | diff={diff:+.5f} "
                      f"(2SE {2*se:.5f}) F1={np.mean(f1s):.3f} -> {reg}",
                      flush=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
