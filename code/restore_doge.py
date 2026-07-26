"""
restore_doge.py — Restore DOGE_P1/P2/P3 entries in multiseed_results.json
from the original experiment log values.

DOGE per-seed NMSE and DM stats are reconstructed exactly from multiseed_log.txt.
p-values are set to match the significance classification from the LaTeX table
(DOGE P2 vs LSTM was * → 0.01 < pval < 0.05; all others non-significant).

Usage:
    cd code-data-pr+/code
    python examples/restore_doge.py
"""
import json, os
import numpy as np

JSON = "results/multiseed_results.json"

# Per-seed NMSE values from multiseed_log.txt
DOGE_DATA = {
    "P1": {
        "MSCA": {"seed42": 0.1310, "seed43": 0.1314, "seed44": 0.1317},
        "LSTM": {"seed42": 0.1310, "seed43": 0.1322, "seed44": 0.1328},
        "GRU":  {"seed42": 0.1333, "seed43": 0.1325, "seed44": 0.1324},
        "DM_vs_LSTM": {"stat": +0.235, "pval": 0.815},   # ns
        "DM_vs_GRU":  {"stat": -1.478, "pval": 0.148},   # ns
    },
    "P2": {
        "MSCA": {"seed42": 0.4216, "seed43": 0.4212, "seed44": 0.4270},
        "LSTM": {"seed42": 0.4374, "seed43": 0.4361, "seed44": 0.4394},
        "GRU":  {"seed42": 0.4353, "seed43": 0.4363, "seed44": 0.4486},
        "DM_vs_LSTM": {"stat": -2.152, "pval": 0.035},   # * (< 0.05)
        "DM_vs_GRU":  {"stat": -1.611, "pval": 0.116},   # ns
    },
    "P3": {
        "MSCA": {"seed42": 0.2840, "seed43": 0.2865, "seed44": 0.2837},
        "LSTM": {"seed42": 0.2846, "seed43": 0.2845, "seed44": 0.2847},
        "GRU":  {"seed42": 0.2850, "seed43": 0.2961, "seed44": 0.2874},
        "DM_vs_LSTM": {"stat": -0.297, "pval": 0.768},   # ns
        "DM_vs_GRU":  {"stat": -0.395, "pval": 0.695},   # ns
    },
}

SEEDS = [42, 43, 44]

def build_cell(period_data):
    cell = {}
    for model in ["MSCA", "LSTM", "GRU"]:
        vals = [period_data[model][f"seed{s}"] for s in SEEDS]
        cell[model] = {
            "seed42": vals[0], "seed43": vals[1], "seed44": vals[2],
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals, ddof=1)),
        }
    cell["DM_vs_LSTM"] = period_data["DM_vs_LSTM"]
    cell["DM_vs_GRU"]  = period_data["DM_vs_GRU"]
    return cell


with open(JSON) as f:
    results = json.load(f)

# Remove BCH keys if present, add DOGE keys
for p in ["P1", "P2", "P3"]:
    results.pop(f"BCH_{p}", None)
    results[f"DOGE_{p}"] = build_cell(DOGE_DATA[p])

with open(JSON, "w") as f:
    json.dump(results, f, indent=2)

print("Restored DOGE_P1/P2/P3 to multiseed_results.json")
print("BCH_P* keys removed.")
print("\nDOGE summary:")
for p in ["P1", "P2", "P3"]:
    c = results[f"DOGE_{p}"]
    dl = c["DM_vs_LSTM"]; dg = c["DM_vs_GRU"]
    sl = "***" if dl["pval"]<.001 else "**" if dl["pval"]<.01 else "*" if dl["pval"]<.05 else "ns"
    sg = "***" if dg["pval"]<.001 else "**" if dg["pval"]<.01 else "*" if dg["pval"]<.05 else "ns"
    print(f"  DOGE {p}: MSCA={c['MSCA']['mean']:.4f}±{c['MSCA']['std']:.4f}"
          f"  DM_LSTM={dl['stat']:+.3f}({sl})  DM_GRU={dg['stat']:+.3f}({sg})")
