"""
run_multiseed_matched.py — 10-seed matched sparsemax-vs-softmax contrast.

Upgrades the matched contrast from one training seed to ten on the five
representative assets (BTC, ETH, XRP, LTC, DOGE) across the three periods
(15 settings). Both cells keep multi-scale and the cross-signal gate on; only
the normalization differs. Seeds are 42-51.

Per setting we report the 10-seed mean NMSE for each cell and the relative
difference (soft - sparse) / soft. The summary applies a one-sided exact
Wilcoxon test to the 15 per-setting relative differences.

Results are saved incrementally and the script is safe to resume.
Usage (as the msca_sparsity package): python run_multiseed_matched.py
"""
import json
import os
import time

import numpy as np
import torch

import msca_sparsity.run_ablation as ra

torch.set_num_threads(8)

ASSETS = ["BTC", "ETH", "XRP", "LTC", "DOGE"]
PERIODS = ["P1", "P2", "P3"]
SEEDS = list(range(42, 52))
SP = dict(use_sparsemax=True,  use_multiscale=True, use_cross_asset=True)
SF = dict(use_sparsemax=False, use_multiscale=True, use_cross_asset=True)
OUT = "results/multiseed_matched.json"


def run():
    os.makedirs("results", exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for asset in ASSETS:
        for period in PERIODS:
            key = f"{asset}_{period}"
            row = res.get(key, {"SP": {}, "SF": {}})
            if (len(row["SP"]) == len(SEEDS) and len(row["SF"]) == len(SEEDS)):
                continue
            try:
                splits = ra.prepare_splits(asset, period)
            except Exception as e:
                print(f"{key}: data ERR {e}", flush=True)
                continue
            for seed in SEEDS:
                for label, kw in [("SP", SP), ("SF", SF)]:
                    sk = str(seed)
                    if sk in row[label]:
                        continue
                    t = time.time()
                    nmse, _ = ra.train_variant(splits, kw, seed=seed)
                    row[label][sk] = float(nmse)
                    print(f"{key} {label} seed{seed}: NMSE={nmse:.4f} "
                          f"({time.time()-t:.0f}s)", flush=True)
                    res[key] = row
                    json.dump(res, open(OUT, "w"), indent=2)
    return res


def analyze(res):
    from scipy.stats import wilcoxon
    rows = []
    print("\n" + "=" * 70)
    print("10-SEED MATCHED CONTRAST (sparse vs soft, 15 settings)")
    print(f"{'setting':10s} {'sparse mean±std':>18} {'soft mean±std':>18} "
          f"{'rel diff':>9}")
    for key, v in sorted(res.items()):
        sp = np.array(list(v["SP"].values()))
        sf = np.array(list(v["SF"].values()))
        if len(sp) < len(SEEDS) or len(sf) < len(SEEDS):
            continue
        rel = (sf.mean() - sp.mean()) / sf.mean()
        rows.append(rel)
        print(f"{key:10s} {sp.mean():>10.4f}±{sp.std():.4f} "
              f"{sf.mean():>10.4f}±{sf.std():.4f} {rel:>+9.4f}")
    if rows:
        r = np.array(rows)
        print("-" * 70)
        print(f"mean rel diff {r.mean():+.4f}  median {np.median(r):+.4f}  "
              f"sparse better in {int((r>0).sum())}/{len(r)}")
        if np.any(r != 0):
            w, p = wilcoxon(r, alternative="greater")
            print(f"one-sided Wilcoxon (sparse < soft): W={w:.0f} p={p:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    analyze(run())
