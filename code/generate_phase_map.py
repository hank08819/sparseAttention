"""
generate_phase_map.py — WIN/TIE/LOSE phase map for sparsemax vs softmax.

Classifies each of the 60 controlled settings (k, sigma_z, n) from
results/controlled_results.json into
  WIN  : sparsemax significantly better (diff > 2 SE and > 1e-4)
  LOSE : softmax significantly better  (diff < -2 SE and < -1e-4)
  TIE  : otherwise
where diff = excess_MSE(softmax) - excess_MSE(sparsemax) over 20 seeds.

Writes figures/fig_phase_map.pdf (one panel per sample size).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "controlled_results.json")
FIG = os.path.join(HERE, "..", "figures", "fig_phase_map.pdf")

KS = [2, 4, 8, 16, 32]
SIGMAS = [0.25, 0.5, 1, 2]
NS = [128, 512, 2048]
COLORS = {"WIN": "#4c9ce8", "TIE": "#d9d9d9", "LOSE": "#e8734c"}


def classify(v):
    diff = v["softmax"]["excess_mse_mean"] - v["sparsemax"]["excess_mse_mean"]
    se = np.sqrt((v["softmax"]["excess_mse_std"] ** 2
                  + v["sparsemax"]["excess_mse_std"] ** 2) / 20)
    if diff > 2 * se and diff > 1e-4:
        return "WIN", diff
    if diff < -2 * se and diff < -1e-4:
        return "LOSE", diff
    return "TIE", diff


def main():
    d = json.load(open(RES))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=True)
    for ax, n in zip(axes, NS):
        for yi, k in enumerate(KS):
            for xi, s in enumerate(SIGMAS):
                v = d[f"k{k}_s{s}_n{n}"]
                reg, diff = classify(v)
                ax.add_patch(plt.Rectangle((xi, yi), 1, 1,
                                           color=COLORS[reg], ec="white", lw=2))
                label = f"{diff:+.4f}".replace("+0.0000", "0")
                ax.text(xi + 0.5, yi + 0.5, label, ha="center", va="center",
                        fontsize=7,
                        color="black" if reg != "TIE" else "#555555")
        ax.set_xlim(0, 4); ax.set_ylim(0, 5)
        ax.set_xticks(np.arange(4) + 0.5); ax.set_xticklabels(SIGMAS)
        ax.set_yticks(np.arange(5) + 0.5)
        ax.set_yticklabels([f"{k}/32" for k in KS])
        ax.set_xlabel("irrelevant-input noise $\\sigma_z$")
        ax.set_title(f"$n={n}$")
        ax.invert_yaxis()
    axes[0].set_ylabel("support fraction $k/G$")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[r]) for r in
               ["WIN", "TIE", "LOSE"]]
    fig.legend(handles, ["sparsemax WINS", "TIE", "sparsemax LOSES"],
               ncol=3, loc="upper center", frameon=False)
    fig.suptitle("")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG, bbox_inches="tight")
    print("wrote", FIG)


if __name__ == "__main__":
    main()
