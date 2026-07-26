"""
run_norm_ablation.py — Does the sparsemax effect depend on input normalization?

Re-runs the matched sparsemax-vs-softmax contrast on the 8 ablation settings
under z-score (StandardScaler) input normalization, and compares the sparsemax
main effect to the MinMax result already computed in
results/sparse_vs_soft_multiasset.json.

Both cells share the identical normalization inside each run, so the per-setting
relative contrast (SF - SP) / SF is scale-free; we report the mean over the 8
settings under each normalization.

Only INPUT scaling changes (MinMax [0,1]  ->  z-score). The target and the NMSE
metric are unchanged, so the two normalizations are directly comparable.

Usage (import as the msca_sparsity package):
    python run_norm_ablation.py
"""
import json
import os

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import msca_sparsity.run_ablation as ra

torch.set_num_threads(6)

SETTINGS = ["BTC_P1", "BTC_P2", "BTC_P3", "LTC_P1", "LTC_P2", "LTC_P3",
            "BCH_P1", "BCH_P2"]
SP = dict(use_sparsemax=True,  use_multiscale=True, use_cross_asset=True)
SF = dict(use_sparsemax=False, use_multiscale=True, use_cross_asset=True)
OUT = "results/norm_ablation_zscore.json"
MINMAX = "results/sparse_vs_soft_multiasset.json"


def prepare_splits_std(asset, period):
    """Identical to ra.prepare_splits but with StandardScaler (z-score) inputs."""
    from sklearn.preprocessing import StandardScaler
    ca = ra.CA_MAP[asset]
    close, feat_tgt = ra.build_features(ra.csv_path(asset, period), ra.CFG["n_features"])
    _, feat_ca = ra.build_features(ra.csv_path(ca, period), ra.CFG["n_features"])
    price_range = float(close.max() - close.min())
    X, Xca, y, pc = ra.make_windows(close, feat_tgt, feat_ca,
                                    ra.CFG["window"], ra.CFG["horizon"])
    n = len(y)
    nt = int(n * ra.CFG["train_ratio"])
    nv = int(n * ra.CFG["val_frac"])
    F = X.shape[-1]
    sc, sca = StandardScaler(), StandardScaler()
    sc.fit(X[:nt].reshape(-1, F)); sca.fit(Xca[:nt].reshape(-1, F))
    X = sc.transform(X.reshape(-1, F)).reshape(X.shape)
    Xca = sca.transform(Xca.reshape(-1, F)).reshape(Xca.shape)
    return dict(tr=(X[:nt], Xca[:nt], y[:nt]),
                val=(X[nt:nt+nv], Xca[nt:nt+nv], y[nt:nt+nv]),
                te=(X[nt+nv:], Xca[nt+nv:], y[nt+nv:]),
                pte=pc[nt+nv:], price_range=price_range, F=F)


def run():
    os.makedirs("results", exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for key in SETTINGS:
        asset, period = key.rsplit("_", 1)
        row = res.get(key, {})
        if "SP" in row and "SF" in row:
            continue
        splits = prepare_splits_std(asset, period)
        for label, kw in [("SP", SP), ("SF", SF)]:
            if label in row:
                continue
            nmse, _ = ra.train_variant(splits, kw, seed=42)
            row[label] = float(nmse)
            print(f"[zscore] {key} {label}: NMSE={nmse:.4f}", flush=True)
            res[key] = row
            json.dump(res, open(OUT, "w"), indent=2)
    return res


def compare(zres):
    mm = json.load(open(MINMAX)) if os.path.exists(MINMAX) else {}

    def main_effect(store):
        rel, raw = [], []
        for k in SETTINGS:
            v = store.get(k)
            if v and "SP" in v and "SF" in v:
                rel.append((v["SF"] - v["SP"]) / v["SF"])   # >0 = sparsemax better
                raw.append(v["SF"] - v["SP"])
        return np.array(rel), np.array(raw)

    print("\n" + "=" * 60)
    print("SPARSEMAX MAIN EFFECT vs INPUT NORMALIZATION (8 settings)")
    print(f"{'normalization':14s} {'rel mean':>10} {'raw mean':>10} {'sparse>soft':>12}")
    for name, store in [("MinMax [0,1]", mm), ("z-score", zres)]:
        rel, raw = main_effect(store)
        if len(rel):
            print(f"{name:14s} {rel.mean():+10.4f} {raw.mean():+10.5f} "
                  f"{int((raw>0).sum())}/{len(raw):<11}")
    print("(rel = mean (SF-SP)/SF; >0 favors sparsemax)")
    print("=" * 60)


if __name__ == "__main__":
    compare(run())
