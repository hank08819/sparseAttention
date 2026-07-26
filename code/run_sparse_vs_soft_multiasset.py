"""
run_sparse_vs_soft_multiasset.py — Matched sparsemax-vs-softmax across many assets.

The paper's matched ablation covers only 8 asset-period settings and finds no
sparsemax main effect. This script runs the SAME matched contrast at scale, as
the paper's future-work section proposes ("extend the matched softmax-sparsemax
comparison to all assets"), to (a) gain power to detect a small true effect and
(b) test the theory-driven prediction that sparsemax's benefit grows with the
residual noise of an asset (blue-chip C1 -> speculative C5).

Two matched cells only (everything else held fixed: multi-scale ON, cross-signal
gate ON, same params/optimizer/schedule, seed 42):

    SP   sparsemax + multi-scale + gate   (= full MISA)
    SF   softmax   + multi-scale + gate

For each setting we report NMSE(softmax) - NMSE(sparsemax); positive = sparsemax
lower error. Completed cells from ablation_2x2_results.json are reused.

Usage (import as the msca_sparsity package):
    python run_sparse_vs_soft_multiasset.py
"""
import json
import os
import time

import numpy as np
import torch

import msca_sparsity.run_ablation as ra

torch.set_num_threads(8)

# ALL 47 assets, grouped into the five noise/liquidity categories (C1 blue-chip
# -> C5 speculative). Built from the full category map in run_ablation so the
# comparison is not limited to any subset.
ASSETS = {cat.split("_")[0]: assets for cat, assets in ra.CATEGORIES.items()}
PERIODS = ["P1", "P2", "P3"]

SP = dict(use_sparsemax=True,  use_multiscale=True, use_cross_asset=True)
SF = dict(use_sparsemax=False, use_multiscale=True, use_cross_asset=True)

OUT = "results/sparse_vs_soft_multiasset.json"
REUSE = "results/ablation_2x2_results.json"


def load_reuse():
    """Pull already-trained SP/SF cells from the 2x2 run (identical protocol)."""
    reused = {}
    if os.path.exists(REUSE):
        d = json.load(open(REUSE))
        for k, v in d.items():
            if "C11_sparse_multi" in v and "C01_soft_multi" in v:
                reused[k] = {"SP": v["C11_sparse_multi"]["NMSE"],
                             "SF": v["C01_soft_multi"]["NMSE"]}
    return reused


def run():
    os.makedirs("results", exist_ok=True)
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for k, v in load_reuse().items():
        results.setdefault(k, {}).update(v)
        results[k]["category"] = results[k].get("category", cat_of(k.rsplit("_", 1)[0]))
    json.dump(results, open(OUT, "w"), indent=2)

    for cat, assets in ASSETS.items():
        for asset in assets:
            for period in PERIODS:
                key = f"{asset}_{period}"
                row = results.get(key, {})
                if "SP" in row and "SF" in row:
                    continue
                if not os.path.exists(ra.csv_path(asset, period)):
                    continue
                try:
                    splits = ra.prepare_splits(asset, period)
                except Exception as e:
                    print(f"{key}: data ERR {e}", flush=True)
                    continue
                for label, kw in [("SP", SP), ("SF", SF)]:
                    if label in row:
                        continue
                    t = time.time()
                    nmse, _ = ra.train_variant(splits, kw, seed=42)
                    row[label] = float(nmse)
                    print(f"{key} [{cat}] {label}: NMSE={nmse:.4f} ({time.time()-t:.0f}s)",
                          flush=True)
                    results[key] = row
                    results[key]["category"] = cat
                    json.dump(results, open(OUT, "w"), indent=2)
    return results


def cat_of(asset):
    for c, aa in ASSETS.items():
        if asset in aa:
            return c
    return "?"


def analyze(results):
    from scipy.stats import wilcoxon
    rows = [(k, v) for k, v in results.items() if "SP" in v and "SF" in v]
    if not rows:
        print("nothing complete"); return
    diffs = np.array([v["SF"] - v["SP"] for _, v in rows])          # >0 = sparsemax better
    print("\n" + "=" * 66)
    print(f"MATCHED sparsemax vs softmax — {len(rows)} settings (seed 42)")
    print(f"  mean (softmax-sparsemax) NMSE diff : {diffs.mean():+.5f}")
    print(f"  sparsemax lower in                 : {int((diffs>0).sum())}/{len(diffs)}")
    med = np.median(diffs)
    print(f"  median diff                        : {med:+.5f}")
    if len(diffs) >= 6 and np.any(diffs != 0):
        w, p = wilcoxon(diffs, alternative="greater")
        print(f"  one-sided Wilcoxon (sparsemax<softmax): W={w:.0f}, p={p:.4f}")
    print("-" * 66)
    print("  by category (mean softmax-sparsemax diff, +=sparsemax better):")
    for c in ["C1", "C2", "C3", "C4", "C5"]:
        dd = [v["SF"] - v["SP"] for _, v in rows if v.get("category") == c]
        if dd:
            print(f"    {c}: mean={np.mean(dd):+.5f}  n={len(dd)}  "
                  f"sparse-better={int((np.array(dd)>0).sum())}/{len(dd)}")
    print("=" * 66)


if __name__ == "__main__":
    analyze(run())
