"""
run_ablation_2x2.py — 2x2 matched factorial ablation of MISA (internal: MSCA-SBiGRU).

Crosses the model's two exact-selection mechanisms on the SAME model — identical
parameter budget, optimizer, data split, schedule, and seed (42). The
cross-signal gate is held ON in every cell, so the only things that vary are the
two factors:

    Factor A  attention normalization : sparsemax  vs  softmax
    Factor B  temporal representation : multi-scale vs single-scale

Four matched cells:

    C11  sparsemax + multi-scale    (= full MISA)
    C01  softmax   + multi-scale
    C10  sparsemax + single-scale
    C00  softmax   + single-scale

For each of the 8 matched asset-period settings, all four cells are trained
fresh under identical conditions. We report, per setting and averaged:

    main effect of sparsemax  = mean_B[ NMSE(softmax) - NMSE(sparsemax) ]
    main effect of multiscale = mean_A[ NMSE(single)  - NMSE(multi)     ]
    interaction               = (C00 - C10) - (C01 - C11)

A positive main effect of sparsemax means sparse selection lowers error; a
positive interaction means sparsemax helps more when only a single scale is
available. Results are saved incrementally (safe to resume).

Usage (import as the msca_sparsity package):
    python run_ablation_2x2.py
"""
import json
import os
import time

import numpy as np
import torch

from msca_sparsity.train import dm_test
import msca_sparsity.run_ablation as ra

torch.set_num_threads(8)

SETTINGS = ["BTC_P1", "BTC_P2", "BTC_P3", "LTC_P1", "LTC_P2", "LTC_P3",
            "BCH_P1", "BCH_P2"]

CELLS = {
    "C11_sparse_multi":  dict(use_sparsemax=True,  use_multiscale=True,  use_cross_asset=True),
    "C01_soft_multi":    dict(use_sparsemax=False, use_multiscale=True,  use_cross_asset=True),
    "C10_sparse_single": dict(use_sparsemax=True,  use_multiscale=False, use_cross_asset=True),
    "C00_soft_single":   dict(use_sparsemax=False, use_multiscale=False, use_cross_asset=True),
}

OUT = "results/ablation_2x2_results.json"


def run():
    os.makedirs("results", exist_ok=True)
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for key in SETTINGS:
        asset, period = key.rsplit("_", 1)
        row = results.get(key, {})
        if all(c in row for c in CELLS):
            continue
        try:
            splits = ra.prepare_splits(asset, period)
        except Exception as e:
            print(f"{key}: data ERR {e}", flush=True)
            continue
        preds = {}
        for cname, kw in CELLS.items():
            if cname in row:
                continue
            t = time.time()
            nmse, pred = ra.train_variant(splits, kw, seed=42)
            preds[cname] = pred
            row[cname] = {"NMSE": float(nmse)}
            print(f"{key} {cname}: NMSE={nmse:.4f} ({time.time()-t:.0f}s)", flush=True)
            results[key] = row
            json.dump(results, open(OUT, "w"), indent=2)
        # DM tests vs full MISA on the fresh predictions we have this run
        if "C11_sparse_multi" in preds:
            y_te = splits["te"][2]
            e_full = y_te - preds["C11_sparse_multi"]
            for cname in ["C01_soft_multi", "C10_sparse_single", "C00_soft_single"]:
                if cname in preds:
                    stat, pval = dm_test(e_full, y_te - preds[cname], horizon=5)
                    row[cname]["DM_stat_vs_full"] = float(stat)
                    row[cname]["DM_pval_vs_full"] = float(pval)
        results[key] = row
        json.dump(results, open(OUT, "w"), indent=2)
    return results


def analyze(results):
    done = {k: v for k, v in results.items() if all(c in v for c in CELLS)}
    if not done:
        print("No completed settings yet.")
        return
    eff_sparse, eff_multi, inter = [], [], []
    print("\n" + "=" * 74)
    print("2x2 MATCHED FACTORIAL — test NMSE per cell (gate ON, seed 42)")
    print(f"{'Setting':8s} {'C11 spМ':>9} {'C01 sfМ':>9} {'C10 spS':>9} {'C00 sfS':>9}"
          f" {'ΔsparseA':>9} {'inter':>8}")
    print("-" * 74)
    for k, v in done.items():
        c11 = v["C11_sparse_multi"]["NMSE"]
        c01 = v["C01_soft_multi"]["NMSE"]
        c10 = v["C10_sparse_single"]["NMSE"]
        c00 = v["C00_soft_single"]["NMSE"]
        es = 0.5 * ((c01 - c11) + (c00 - c10))   # softmax minus sparsemax
        em = 0.5 * ((c10 - c11) + (c00 - c01))   # single minus multi
        it = (c00 - c10) - (c01 - c11)           # interaction
        eff_sparse.append(es); eff_multi.append(em); inter.append(it)
        print(f"{k:8s} {c11:9.4f} {c01:9.4f} {c10:9.4f} {c00:9.4f} {es:+9.4f} {it:+8.4f}")
    print("-" * 74)
    n = len(done)

    def summarize(name, arr):
        a = np.array(arr)
        pos = int((a > 0).sum())
        print(f"  {name:26s} mean={a.mean():+.4f}  std={a.std():.4f}  "
              f"(favorable in {pos}/{n} settings)")

    print(f"Averaged over {n} matched settings "
          f"(positive = the exact-selection mechanism lowers NMSE):")
    summarize("main effect sparsemax", eff_sparse)
    summarize("main effect multi-scale", eff_multi)
    summarize("interaction (sparse×scale)", inter)
    print("=" * 74)


if __name__ == "__main__":
    res = run()
    analyze(res)
