"""
run_neural_phase_map.py — Does the win/tie/lose phase map hold for a trained
neural attention model?

The controlled phase map uses the linear predictor y_hat = p(s)^T x. Reviewers
can object that the boundary may not transfer to trained attention networks.
This experiment answers that directly. We place the actual MISA
temporal-attention core (BiGRU + sparsemax/softmax over time steps, single
scale, no cross-signal branch) on synthetic sequences with a KNOWN support.

Data. Each sample is a window of T=32 scalar time steps. The first k steps are
relevant: x_t ~ N(0,1). The remaining steps are irrelevant: x_t ~ N(0,
sigma_z^2). The target is y = mean of the k relevant steps + eps,
eps ~ N(0, 0.05^2). The support is the same in every sample, so the network
can learn it.

Grid. k in {2, 8, 32}, sigma_z in {0.5, 2}, n in {512, 2048}, 5 seeds,
sparsemax vs softmax: 120 trained networks.

Metrics per run: excess test MSE (above 0.05^2) on 20,000 fresh samples, and
total attention on the irrelevant steps. Cells are classified win/tie/lose by
the 5-seed mean difference against a two-standard-error threshold, as in the
linear phase map.

Usage (as the misa_sparsity package): python run_neural_phase_map.py
"""
import json
import os
import time

import numpy as np
import torch

import misa_sparsity.run_ablation as ra
from misa_sparsity import MISABiGRU, Trainer

torch.set_num_threads(8)

T = 32
KS = [2, 8, 32]
SIGMAS = [0.5, 2]
NS = [512, 2048]
SEEDS = range(5)
EVAL_N = 20000
IRREDUCIBLE = 0.05 ** 2
OUT = "results/neural_phase_map.json"


def gen(rng, n, k, sigma_z):
    X = rng.standard_normal((n, T, 1)).astype(np.float32)
    if k < T:
        X[:, k:, :] *= sigma_z
    y = X[:, :k, 0].mean(axis=1) + rng.normal(0, 0.05, n).astype(np.float32)
    return X, y.astype(np.float32)


def one_run(k, sigma_z, n, seed, use_sparsemax):
    rng = np.random.default_rng(seed)
    Xtr, ytr = gen(rng, n, k, sigma_z)
    Xv, yv = gen(rng, max(n // 5, 64), k, sigma_z)
    ev_rng = np.random.default_rng(10_000 + seed)
    Xte, yte = gen(ev_rng, EVAL_N, k, sigma_z)
    Z = lambda A: np.zeros_like(A)

    torch.manual_seed(seed)
    model = MISABiGRU(feat_dim=1, hidden=64, n_heads=4, dropout=0.25,
                      lambda_gate=0.05, use_sparsemax=use_sparsemax,
                      use_multiscale=False, use_cross_asset=False)
    tr = Trainer(model, lr=ra.CFG["lr"], epochs=ra.CFG["epochs"],
                 patience=ra.CFG["patience"], warmup=ra.CFG["warmup"],
                 batch_size=ra.CFG["batch_size"], device="cpu", verbose=False)
    tr.fit(Xtr, Z(Xtr), ytr, Xv, Z(Xv), yv)

    model.eval()
    with torch.no_grad():
        xt = torch.tensor(Xte)
        pred, a1, _ = model(xt, torch.zeros_like(xt))
        pred = pred.numpy()
        irr = float(a1[:, k:].sum(dim=1).mean()) if k < T else 0.0
    excess = float(np.mean((pred - yte) ** 2)) - IRREDUCIBLE
    return excess, irr


def main():
    os.makedirs("results", exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for n in NS:
        for k in KS:
            for s in SIGMAS:
                key = f"k{k}_s{s}_n{n}"
                if key in res and "regime" in res[key]:
                    continue
                acc = {"sparse": {"ex": [], "irr": []},
                       "soft": {"ex": [], "irr": []}}
                t0 = time.time()
                for seed in SEEDS:
                    for label, sp in [("sparse", True), ("soft", False)]:
                        ex, irr = one_run(k, s, n, seed, sp)
                        acc[label]["ex"].append(ex)
                        acc[label]["irr"].append(irr)
                sp_ex = np.array(acc["sparse"]["ex"])
                so_ex = np.array(acc["soft"]["ex"])
                diff = so_ex.mean() - sp_ex.mean()
                se = np.sqrt((so_ex.std() ** 2 + sp_ex.std() ** 2) / len(SEEDS))
                if diff > 2 * se and diff > 1e-4:
                    reg = "WIN"
                elif diff < -2 * se and diff < -1e-4:
                    reg = "LOSE"
                else:
                    reg = "TIE"
                res[key] = {
                    "k": k, "sigma_z": s, "n": n,
                    "sparse_excess": float(sp_ex.mean()),
                    "soft_excess": float(so_ex.mean()),
                    "diff_mean": float(diff), "diff_2se": float(2 * se),
                    "sparse_irr_attn": float(np.mean(acc["sparse"]["irr"])),
                    "soft_irr_attn": float(np.mean(acc["soft"]["irr"])),
                    "regime": reg,
                }
                print(f"{key}: diff={diff:+.5f} (2SE {2*se:.5f}) "
                      f"irr_attn sp={res[key]['sparse_irr_attn']:.3f}"
                      f"/so={res[key]['soft_irr_attn']:.3f} -> {reg}"
                      f"  ({time.time()-t0:.0f}s)", flush=True)
                json.dump(res, open(OUT, "w"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
