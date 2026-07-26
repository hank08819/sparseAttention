"""
run_dcc_garch_52assets.py  —  DCC-GARCH BTC correlation for all 52 assets
==========================================================================
Computes bivariate DCC(1,1) mean conditional correlation ρ(BTC, asset)
for each asset-period pair available on disk.

Used for:
  1. Category-level DiD: treated = assets with large Δρ(P2→P3)
  2. Verification of theorem assumptions (B3: intermittent cross-asset signal)

Output: results/dcc_garch_52assets.json

Usage:
    cd code-data-pr+/code
    python examples/run_dcc_garch_52assets.py
"""

import json, os, sys, warnings
import numpy as np
import pandas as pd
from arch import arch_model

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA     = "../data"
DATE_TAG = {"P1": "20221101", "P2": "20231001", "P3": "20240301"}

CATEGORIES = {
    "C1_BTC_family":    ["LTC", "BCH", "DOGE", "SHIB", "XRP"],
    "C2_ETH_ecosystem": ["ETH", "MATIC", "ARB", "OP", "LRC"],
    "C3_Alt_L1":        ["SOL", "ADA", "AVAX", "DOT", "ATOM",
                         "NEAR", "ALGO", "ICP", "HBAR", "VET", "APT"],
    "C4_DeFi":          ["LINK", "AAVE", "UNI", "MKR", "CRV",
                         "COMP", "SNX", "GRT", "SUSHI", "YFI", "BAL", "1INCH"],
    "C5_Speculative":   ["EOS", "XTZ", "MANA", "SAND", "AXS", "APE",
                         "ENJ", "CHZ", "BAT", "ZRX", "ANKR", "OXT", "NMR"],
}
# BTC is excluded from the loop (it IS the cross-asset reference)


def csv_path(asset, period):
    return os.path.join(DATA, f"{asset}_5m_{period}_{DATE_TAG[period]}.csv")


def load_log_ret(asset, period, min_rows=500):
    path = csv_path(asset, period)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    close = df["close"].astype(float).values
    close = np.maximum(close, 1e-10)
    ret = np.diff(np.log(close))
    if len(ret) < min_rows:
        return None
    return ret


def fit_dcc(r_btc, r_alt, max_rows=4000):
    """
    Simplified DCC(1,1): fit univariate GARCH(1,1) to each series,
    extract standardised residuals, then compute rolling Pearson
    correlation as a DCC proxy.
    Returns mean conditional correlation over the test window.
    """
    n = min(len(r_btc), len(r_alt), max_rows)
    rb = r_btc[-n:] * 100
    ra = r_alt[-n:] * 100

    def garch_std_resid(r):
        try:
            m = arch_model(r, vol="Garch", p=1, q=1, dist="normal", rescale=False)
            res = m.fit(disp="off", options={"maxiter": 200})
            return res.std_resid
        except Exception:
            return (r - r.mean()) / (r.std() + 1e-12)

    z_b = garch_std_resid(rb)
    z_a = garch_std_resid(ra)

    # rolling 100-bar Pearson as DCC proxy
    win = 100
    corrs = []
    for i in range(win, len(z_b)):
        zb_w = z_b[i-win:i]; za_w = z_a[i-win:i]
        c = np.corrcoef(zb_w, za_w)[0, 1]
        if np.isfinite(c):
            corrs.append(c)

    return float(np.mean(corrs)) if corrs else float(np.corrcoef(rb, ra)[0, 1])


def main():
    os.makedirs("results", exist_ok=True)
    out_path = "results/dcc_garch_52assets.json"

    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing entries.", flush=True)

    periods = ["P1", "P2", "P3"]

    for cat, assets in CATEGORIES.items():
        for asset in assets:
            for period in periods:
                key = f"{asset}_{period}"
                if key in results:
                    print(f"  SKIP {key}", flush=True)
                    continue

                r_btc = load_log_ret("BTC", period)
                r_alt = load_log_ret(asset, period)

                if r_btc is None or r_alt is None:
                    print(f"  SKIP {key}: missing data", flush=True)
                    continue

                print(f"  [{cat}] {key} ... ", end="", flush=True)
                rho = fit_dcc(r_btc, r_alt)
                results[key] = {"rho_BTC": round(rho, 6), "category": cat}
                print(f"ρ(BTC,{asset})={rho:+.4f}", flush=True)

                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)

    # ── summary: mean ρ by category × period ─────────────────────────────────
    print("\n" + "=" * 60)
    print("Mean ρ(BTC, asset)  by category × period")
    print(f"{'Category':<25} {'P1':>8} {'P2':>8} {'P3':>8} {'Δ(P3-P2)':>10}")
    print("-" * 60)

    cat_rho = {}
    for cat, assets in CATEGORIES.items():
        cat_rho[cat] = {}
        row = f"{cat:<25}"
        p3_rho = []; p2_rho = []
        for period in periods:
            vals = [results[f"{a}_{period}"]["rho_BTC"]
                    for a in assets if f"{a}_{period}" in results]
            m = np.mean(vals) if vals else float("nan")
            cat_rho[cat][period] = round(float(m), 4)
            row += f" {m:>+8.4f}"
            if period == "P2": p2_rho = vals
            if period == "P3": p3_rho = vals
        delta = (np.mean(p3_rho) - np.mean(p2_rho)
                 if p2_rho and p3_rho else float("nan"))
        row += f" {delta:>+10.4f}"
        print(row)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # save category summary
    with open("results/dcc_category_summary.json", "w") as f:
        json.dump(cat_rho, f, indent=2)

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
