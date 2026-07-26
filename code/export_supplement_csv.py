#!/usr/bin/env python3
"""
export_supplement_csv.py — Export the two CSV files referenced in the
Supplementary Material from results/expanded_results.json.

Produces:
  results/expanded_crypto_139.csv          one row per valid asset-period setting
  results/expanded_crypto_asset_summary.csv one row per asset (periods aggregated)

Sign convention in the exported files: positive DM statistics and positive
relative NMSE differences favor MISA (lower error). The raw JSON stores DM
statistics with the opposite sign (negative favors MISA), so the sign is
flipped here to match the supplement text.
"""
import csv
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def load():
    with open(os.path.join(RESULTS, "expanded_results.json")) as f:
        return json.load(f)


def rel_diff(baseline, misa):
    """Relative NMSE improvement of MISA over baseline (positive = MISA better)."""
    return (baseline - misa) / baseline


def export_139(data):
    path = os.path.join(RESULTS, "expanded_crypto_139.csv")
    rows = []
    for key, v in sorted(data.items()):
        asset, period = key.rsplit("_", 1)
        misa, lstm, gru = v["MSCA"]["mean"], v["LSTM"]["mean"], v["GRU"]["mean"]
        rows.append({
            "asset": asset,
            "period": period,
            "nmse_misa": round(misa, 6),
            "nmse_lstm": round(lstm, 6),
            "nmse_gru": round(gru, 6),
            # sign flipped so positive DM favors MISA
            "dm_stat_vs_lstm": round(-v["DM_vs_LSTM"]["stat"], 4),
            "dm_pval_vs_lstm": round(v["DM_vs_LSTM"]["pval"], 4),
            "dm_stat_vs_gru": round(-v["DM_vs_GRU"]["stat"], 4),
            "dm_pval_vs_gru": round(v["DM_vs_GRU"]["pval"], 4),
            "win_vs_lstm": int(misa < lstm),
            "win_vs_gru": int(misa < gru),
            "rel_nmse_diff_vs_lstm": round(rel_diff(lstm, misa), 6),
            "rel_nmse_diff_vs_gru": round(rel_diff(gru, misa), 6),
        })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path} ({len(rows)} settings)")
    return rows


def export_asset_summary(rows):
    path = os.path.join(RESULTS, "expanded_crypto_asset_summary.csv")
    by_asset = {}
    for r in rows:
        by_asset.setdefault(r["asset"], {"lstm": [], "gru": []})
        by_asset[r["asset"]]["lstm"].append(r["rel_nmse_diff_vs_lstm"])
        by_asset[r["asset"]]["gru"].append(r["rel_nmse_diff_vs_gru"])
    out = []
    for asset in sorted(by_asset):
        d = by_asset[asset]
        out.append({
            "asset": asset,
            "n_periods": len(d["lstm"]),
            "mean_rel_diff_vs_lstm": round(statistics.mean(d["lstm"]), 6),
            "mean_rel_diff_vs_gru": round(statistics.mean(d["gru"]), 6),
        })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"Wrote {path} ({len(out)} assets)")
    # sanity check against the reported medians
    med_lstm = statistics.median(r["mean_rel_diff_vs_lstm"] for r in out)
    med_gru = statistics.median(r["mean_rel_diff_vs_gru"] for r in out)
    print(f"  median asset-level gain: vs LSTM {100 * med_lstm:.2f}%, "
          f"vs GRU {100 * med_gru:.2f}%")


if __name__ == "__main__":
    data = load()
    rows = export_139(data)
    export_asset_summary(rows)
