"""
examples/generate_figures.py — Regenerate all paper figures from results JSON.

Usage:
    python examples/generate_figures.py --results path/to/msca_results.json \
                                         --out figures/
"""

import argparse, json, os
import numpy as np

from msca_sparsity.figures import (
    plot_nmse_comparison,
    plot_nmse_all_pairs,
    plot_llm_comparison,
    plot_dm_statistics,
    plot_dm_all_pairs,
    plot_scale_weights,
    plot_gate_values,
    plot_sparsity_rates,
)

ASSETS  = ["BTC", "ETH", "XRP", "LTC", "DOGE"]
PERIODS = ["P1", "P2", "P3"]

# ── Hardcoded paper values (all 15 asset-period pairs) ────────────────────────
# Structure: PAPER_NMSE[asset][period][model]
PAPER_NMSE = {
    "BTC": {
        "P1": {"MSCA-SBiGRU": 0.0473, "LSTM": 0.0489, "GRU": 0.0481},
        "P2": {"MSCA-SBiGRU": 0.0885, "LSTM": 0.0871, "GRU": 0.0912},
        "P3": {"MSCA-SBiGRU": 0.0521, "LSTM": 0.0598, "GRU": 0.0634},
    },
    "ETH": {
        "P1": {"MSCA-SBiGRU": 0.0657, "LSTM": 0.0671, "GRU": 0.0663},
        "P2": {"MSCA-SBiGRU": 0.2401, "LSTM": 0.2388, "GRU": 0.2512},
        "P3": {"MSCA-SBiGRU": 0.1124, "LSTM": 0.1198, "GRU": 0.1187},
    },
    "XRP": {
        "P1": {"MSCA-SBiGRU": 0.1210, "LSTM": 0.1230, "GRU": 0.1221},
        "P2": {"MSCA-SBiGRU": 0.3506, "LSTM": 0.3481, "GRU": 0.3524},
        "P3": {"MSCA-SBiGRU": 0.1843, "LSTM": 0.1921, "GRU": 0.1897},
    },
    "LTC": {
        "P1": {"MSCA-SBiGRU": 0.1845, "LSTM": 0.1862, "GRU": 0.1858},
        "P2": {"MSCA-SBiGRU": 0.2143, "LSTM": 0.2210, "GRU": 0.2291},
        "P3": {"MSCA-SBiGRU": 0.1576, "LSTM": 0.1698, "GRU": 0.1634},
    },
    "DOGE": {
        "P1": {"MSCA-SBiGRU": 0.4821, "LSTM": 0.4870, "GRU": 0.4845},
        "P2": {"MSCA-SBiGRU": 0.5143, "LSTM": 0.5121, "GRU": 0.5234},
        "P3": {"MSCA-SBiGRU": 0.3892, "LSTM": 0.3975, "GRU": 0.3961},
    },
}

# All 30 DM statistics (negative = MSCA-SBiGRU wins)
PAPER_DM = {
    "BTC_P1_vs_LSTM":  -0.873,  "BTC_P1_vs_GRU":  -0.612,
    "ETH_P1_vs_LSTM":  -0.541,  "ETH_P1_vs_GRU":  -0.318,
    "XRP_P1_vs_LSTM":  -1.124,  "XRP_P1_vs_GRU":  -0.897,
    "LTC_P1_vs_LSTM":  -0.844,  "LTC_P1_vs_GRU":  -0.572,
    "DOGE_P1_vs_LSTM": -0.483,  "DOGE_P1_vs_GRU": -0.391,
    "BTC_P2_vs_LSTM":  +6.799,  "BTC_P2_vs_GRU":  -2.505,
    "ETH_P2_vs_LSTM":  -0.724,  "ETH_P2_vs_GRU":  -2.183,
    "XRP_P2_vs_LSTM":  -1.312,  "XRP_P2_vs_GRU":  -1.087,
    "LTC_P2_vs_LSTM":  -1.742,  "LTC_P2_vs_GRU":  -2.318,
    "DOGE_P2_vs_LSTM": +0.312,  "DOGE_P2_vs_GRU": -1.724,
    "BTC_P3_vs_LSTM":  -2.073,  "BTC_P3_vs_GRU":  -3.231,
    "ETH_P3_vs_LSTM":  -2.541,  "ETH_P3_vs_GRU":  -1.483,
    "XRP_P3_vs_LSTM":  -2.881,  "XRP_P3_vs_GRU":  -2.024,
    "LTC_P3_vs_LSTM":  -2.456,  "LTC_P3_vs_GRU":  -1.653,
    "DOGE_P3_vs_LSTM": -2.152,  "DOGE_P3_vs_GRU": -2.073,
}

PAPER_GATE = {
    "BTC":  [0.48, 0.18, 0.79],
    "ETH":  [0.52, 0.21, 0.81],
    "XRP":  [0.44, 0.15, 0.77],
    "LTC":  [0.46, 0.17, 0.74],
    "DOGE": [0.49, 0.23, 0.67],
}

PAPER_SCALE_WEIGHTS = {
    "P1": [0.73, 0.18, 0.09],
    "P2": [0.31, 0.42, 0.27],
    "P3": [0.19, 0.20, 0.61],
}

PAPER_SPARSITY = {"P1": 12.0, "P2": 35.0, "P3": 19.0}


def _nmse_by_period(nmse_by_asset: dict) -> dict:
    """Reshape PAPER_NMSE[asset][period][model] → results[period][asset][model]."""
    out = {p: {} for p in PERIODS}
    for asset, periods in nmse_by_asset.items():
        for period, models in periods.items():
            out[period][asset] = models
    return out


def main(results_path: str | None, out_dir: str = "figures"):
    os.makedirs(out_dir, exist_ok=True)

    # Optionally load from JSON; fall back to hardcoded paper values
    if results_path and os.path.exists(results_path):
        R = json.load(open(results_path))
        # Expect R[f"{asset}_{period}"][model][metric]
        nmse_by_asset = {}
        for asset in ASSETS:
            nmse_by_asset[asset] = {}
            for period in PERIODS:
                key = f"{asset}_{period}"
                nmse_by_asset[asset][period] = {
                    "MSCA-SBiGRU": R[key]["MSCA-BiGRU"]["NMSE"],
                    "LSTM":        R[key]["LSTM"]["NMSE"],
                    "GRU":         R[key]["Base GRU"]["NMSE"],
                }
        gate_data     = PAPER_GATE        # gate values not in results JSON
        scale_weights = PAPER_SCALE_WEIGHTS
        sparsity      = PAPER_SPARSITY
        dm_data       = PAPER_DM
    else:
        print("No results JSON found — using hardcoded paper values.")
        nmse_by_asset = PAPER_NMSE
        gate_data     = PAPER_GATE
        scale_weights = PAPER_SCALE_WEIGHTS
        sparsity      = PAPER_SPARSITY
        dm_data       = PAPER_DM

    nmse_by_period = _nmse_by_period(nmse_by_asset)

    # ── NMSE per-regime (Fig 5) ───────────────────────────────────────────────
    plot_nmse_comparison(
        nmse_by_period,
        os.path.join(out_dir, "fig5_nmse_comparison.pdf"),
    )

    # ── NMSE all 15 pairs flat chart ─────────────────────────────────────────
    plot_nmse_all_pairs(
        nmse_by_asset,
        os.path.join(out_dir, "fig_nmse_all_pairs.pdf"),
    )

    # ── LLM comparison (Fig 6) ────────────────────────────────────────────────
    llm_nmse = {
        "MSCA-SBiGRU": nmse_by_asset["BTC"]["P1"]["MSCA-SBiGRU"],
        "PatchTST": 1.62, "Informer": 2.48, "Autoformer": 3.10,
        "FinBERT": 4.51, "FLANG": 5.08, "LLMTime": 6.52,
    }
    plot_llm_comparison(llm_nmse, os.path.join(out_dir, "fig6_llm_comparison.pdf"))

    # ── DM statistics (subset: BTC/ETH/XRP for compact chart) ────────────────
    dm_subset = {k: v for k, v in dm_data.items()
                 if any(a in k for a in ["BTC", "ETH", "XRP"])}
    plot_dm_statistics(dm_subset, os.path.join(out_dir, "fig_dm_statistics.pdf"))

    # ── All 30 DM statistics ──────────────────────────────────────────────────
    plot_dm_all_pairs(dm_data, os.path.join(out_dir, "fig_dm_all_pairs.pdf"))

    # ── Scale weights ─────────────────────────────────────────────────────────
    plot_scale_weights(scale_weights, os.path.join(out_dir, "fig_scale_weights.pdf"))

    # ── Gate values (all 5 assets) ────────────────────────────────────────────
    plot_gate_values(gate_data, os.path.join(out_dir, "fig_gate_values.pdf"))

    # ── Sparsity rates ────────────────────────────────────────────────────────
    plot_sparsity_rates(sparsity, os.path.join(out_dir, "fig_sparsity_rates.pdf"))

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=None, help="Path to msca_results.json (optional)")
    p.add_argument("--out", default="figures", help="Output directory for PDFs")
    args = p.parse_args()
    main(args.results, args.out)
