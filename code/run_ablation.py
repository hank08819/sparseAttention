"""
run_ablation.py — Ablation study for MSCA-SBiGRU
==================================================
Trains 4 model variants per asset-period pair (seed 42 only for speed):

  FULL  : full MSCA-SBiGRU (sparsemax + multi-scale + cross-asset gate)
  ABL_A : softmax attention  (sparsemax → softmax; scale fusion also softmax)
  ABL_B : single-scale only  (use_multiscale=False)
  ABL_C : no cross-asset     (use_cross_asset=False)

Computes DM(FULL vs variant) for each variant on every available pair.
Saves to results/ablation_results.json  (safe to resume).

Usage:
    cd code-data-pr+/code
    python examples/run_ablation.py
"""

import json, os, sys, time
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msca_sparsity import MSCABiGRU, Trainer, build_features
from msca_sparsity.train import dm_test

DATA    = "../data"
DEVICE  = "cpu"
SEED    = 42

CFG = dict(
    window=30, horizon=5, hidden=64, n_heads=4, dropout=0.25,
    lr=5e-4, epochs=150, patience=20, warmup=20,
    batch_size=64, train_ratio=0.70, val_frac=0.10, n_features=7,
)

DATE_TAG = {"P1": "20221101", "P2": "20231001", "P3": "20240301"}

CATEGORIES = {
    "C1_BTC_family":    ["BTC", "LTC", "BCH", "DOGE", "SHIB", "XRP"],
    "C2_ETH_ecosystem": ["ETH", "MATIC", "ARB", "OP", "LRC"],
    "C3_Alt_L1":        ["SOL", "ADA", "AVAX", "DOT", "ATOM",
                         "NEAR", "ALGO", "ICP", "HBAR", "VET", "APT"],
    "C4_DeFi":          ["LINK", "AAVE", "UNI", "MKR", "CRV",
                         "COMP", "SNX", "GRT", "SUSHI", "YFI", "BAL", "1INCH"],
    "C5_Speculative":   ["EOS", "XTZ", "MANA", "SAND", "AXS", "APE",
                         "ENJ", "CHZ", "BAT", "ZRX", "ANKR", "OXT", "NMR"],
}

CA_MAP = {a: "BTC" for cat in CATEGORIES.values() for a in cat if a != "BTC"}
CA_MAP["BTC"] = "ETH"

VARIANTS = {
    "FULL":  dict(use_sparsemax=True,  use_multiscale=True,  use_cross_asset=True),
    "ABL_A": dict(use_sparsemax=False, use_multiscale=True,  use_cross_asset=True),
    "ABL_B": dict(use_sparsemax=True,  use_multiscale=False, use_cross_asset=True),
    "ABL_C": dict(use_sparsemax=True,  use_multiscale=True,  use_cross_asset=False),
}


def csv_path(asset, period):
    return os.path.join(DATA, f"{asset}_5m_{period}_{DATE_TAG[period]}.csv")


def make_windows(close, feat_tgt, feat_ca, window=30, horizon=5):
    log_ret = np.diff(np.log(np.maximum(close, 1e-10)))
    n = min(len(feat_tgt), len(feat_ca))
    X, Xca, y, pc = [], [], [], []
    for i in range(n - window - horizon + 1):
        X.append(feat_tgt[i:i + window])
        Xca.append(feat_ca[i:i + window])
        y.append(log_ret[i + window - 1: i + window + horizon - 1].sum())
        pc.append(close[i + window - 1])
    return (np.array(X, np.float32), np.array(Xca, np.float32),
            np.array(y, np.float32), np.array(pc, np.float32))


def prepare_splits(asset, period):
    ca = CA_MAP[asset]
    close, feat_tgt = build_features(csv_path(asset, period), CFG["n_features"])
    _, feat_ca      = build_features(csv_path(ca, period),    CFG["n_features"])
    price_range = float(close.max() - close.min())
    X, Xca, y, pc = make_windows(close, feat_tgt, feat_ca, CFG["window"], CFG["horizon"])
    n  = len(y)
    nt = int(n * CFG["train_ratio"])
    nv = int(n * CFG["val_frac"])
    F  = X.shape[-1]
    sc = MinMaxScaler(); sca = MinMaxScaler()
    sc.fit(X[:nt].reshape(-1, F)); sca.fit(Xca[:nt].reshape(-1, F))
    X   = sc.transform(X.reshape(-1, F)).reshape(X.shape)
    Xca = sca.transform(Xca.reshape(-1, F)).reshape(Xca.shape)
    return dict(
        tr=(X[:nt], Xca[:nt], y[:nt]),
        val=(X[nt:nt+nv], Xca[nt:nt+nv], y[nt:nt+nv]),
        te=(X[nt+nv:], Xca[nt+nv:], y[nt+nv:]),
        pte=pc[nt+nv:], price_range=price_range, F=F,
    )


def train_variant(splits, kwargs, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MSCABiGRU(feat_dim=splits["F"], hidden=64, n_heads=4,
                      dropout=0.25, lambda_gate=0.05, **kwargs)
    trainer = Trainer(model, lr=CFG["lr"], epochs=CFG["epochs"],
                      patience=CFG["patience"], warmup=CFG["warmup"],
                      batch_size=CFG["batch_size"], device=DEVICE, verbose=False)
    trainer.fit(*splits["tr"], *splits["val"])
    pred = trainer.predict(*splits["te"][:2])
    nmse = trainer.evaluate(*splits["te"], splits["pte"],
                            price_range=splits["price_range"])["NMSE"]
    return float(nmse), pred


def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def main():
    os.makedirs("results", exist_ok=True)
    out_path = "results/ablation_results.json"
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing pairs", flush=True)

    total_pairs = wins = losses = 0

    for cat, assets in CATEGORIES.items():
        for asset in assets:
            for period in DATE_TAG:
                key = f"{asset}_{period}"
                if key in results:
                    for v in ["ABL_A", "ABL_B", "ABL_C"]:
                        if v in results[key]:
                            p = results[key][v]["DM_pval"]
                            s = results[key][v]["DM_stat"]
                            if p < 0.05:
                                if s < 0: wins += 1
                                else: losses += 1
                    total_pairs += 1
                    continue

                tgt_csv = csv_path(asset, period)
                ca_csv  = csv_path(CA_MAP.get(asset, "BTC"), period)
                if not os.path.exists(tgt_csv) or not os.path.exists(ca_csv):
                    continue

                print(f"\n[{cat}]  {asset} {period}", flush=True)
                t0 = time.time()
                try:
                    splits = prepare_splits(asset, period)
                except Exception as e:
                    print(f"  ERR data: {e}", flush=True); continue

                # Train all variants
                preds = {}
                nmses = {}
                for vname, vkw in VARIANTS.items():
                    try:
                        nmse, pred = train_variant(splits, vkw)
                        preds[vname] = pred
                        nmses[vname] = nmse
                    except Exception as e:
                        print(f"  ERR {vname}: {e}", flush=True)

                if "FULL" not in preds:
                    continue

                y_te   = splits["te"][2]
                e_full = y_te - preds["FULL"]
                row    = {"FULL_NMSE": nmses["FULL"]}

                for vname in ["ABL_A", "ABL_B", "ABL_C"]:
                    if vname not in preds:
                        continue
                    e_v = y_te - preds[vname]
                    dm_stat, dm_pval = dm_test(e_full, e_v, horizon=5)
                    row[vname] = {
                        "NMSE": nmses[vname],
                        "DM_stat": float(dm_stat),
                        "DM_pval": float(dm_pval),
                        "NMSE_delta": float(nmses[vname] - nmses["FULL"]),
                    }
                    flag = ""
                    if dm_pval < 0.05:
                        if dm_stat < 0: wins += 1; flag = "FULL_WINS"
                        else: losses += 1; flag = "FULL_LOSES"
                    label = {"ABL_A": "Softmax", "ABL_B": "NoScale", "ABL_C": "NoGate"}[vname]
                    print(f"  vs {label:8s}: NMSE={nmses[vname]:.4f} (Δ{nmses[vname]-nmses['FULL']:+.4f}) "
                          f"DM={dm_stat:+.3f}{sig(dm_pval):3s} {flag}", flush=True)

                results[key] = row
                total_pairs += 1
                print(f"  FULL NMSE={nmses['FULL']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print(f"{'Variant':<12} {'Mean NMSE':>10} {'Mean Δ':>10} {'DM wins':>8} {'DM losses':>10}")
    print("-" * 70)

    for vname in ["ABL_A", "ABL_B", "ABL_C"]:
        label = {"ABL_A": "Softmax", "ABL_B": "NoScale", "ABL_C": "NoGate"}[vname]
        deltas, pair_wins, pair_losses = [], 0, 0
        nmses_v, nmses_f = [], []
        for key, row in results.items():
            if vname in row and "FULL_NMSE" in row:
                deltas.append(row[vname]["NMSE_delta"])
                nmses_v.append(row[vname]["NMSE"])
                nmses_f.append(row["FULL_NMSE"])
                p = row[vname]["DM_pval"]
                s = row[vname]["DM_stat"]
                if p < 0.05:
                    if s < 0: pair_wins += 1
                    else: pair_losses += 1
        mean_nmse = float(np.mean(nmses_v)) if nmses_v else float("nan")
        mean_delta = float(np.mean(deltas)) if deltas else float("nan")
        print(f"  {label:<10} {mean_nmse:>10.4f} {mean_delta:>+10.4f} {pair_wins:>8} {pair_losses:>10}")

    full_mean = float(np.mean([r["FULL_NMSE"] for r in results.values() if "FULL_NMSE" in r]))
    print(f"  {'FULL':10} {full_mean:>10.4f} {'—':>10} {'—':>8} {'—':>10}")
    print(f"\nTotal pairs: {total_pairs}")


if __name__ == "__main__":
    main()
