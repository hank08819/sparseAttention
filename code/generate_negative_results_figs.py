"""
generate_negative_results_figs.py — Figures for the boundary / negative-results
supplement. Reads the saved JSON results and writes two PDFs into figures/.

  fig_injection_probe.pdf   attention on injected noise steps + NMSE gap vs r
  fig_category_gradient.pdf  139-setting sparsemax-softmax relative diff by C1-C5
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

R_VALUES = [0, 8, 16, 32]


def injection_fig():
    d = json.load(open(os.path.join(RES, "noise_injection.json")))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    frac = [r / (30 + r) for r in R_VALUES]  # injected share of the window

    for key, mk in zip(d, ["o", "s"]):
        rr = d[key]
        sp_attn = [rr[f"r{r}"]["sparse"]["inj_attn"] for r in R_VALUES]
        so_attn = [rr[f"r{r}"]["soft"]["inj_attn"] for r in R_VALUES]
        ax1.plot(R_VALUES, sp_attn, mk + "-", color="C0", label=f"{key} sparsemax")
        ax1.plot(R_VALUES, so_attn, mk + "--", color="C1", label=f"{key} softmax")
        gap = [rr[f"r{r}"]["soft"]["NMSE"] - rr[f"r{r}"]["sparse"]["NMSE"]
               for r in R_VALUES]
        ax2.plot(R_VALUES, gap, mk + "-", label=key)

    ax1.plot(R_VALUES, frac, "k:", lw=1, label="uniform share")
    ax1.set_xlabel("injected noise steps $r$")
    ax1.set_ylabel("attention weight on injected steps")
    ax1.set_title("(a) Sparsemax does not exclude injected noise\n(after full training)")
    ax1.legend(fontsize=7)

    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_xlabel("injected noise steps $r$")
    ax2.set_ylabel("NMSE(softmax) $-$ NMSE(sparsemax)")
    ax2.set_title("(b) No accuracy gap opens up with $r$")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIG, "fig_injection_probe.pdf")
    fig.savefig(p, bbox_inches="tight"); print("wrote", p)


def category_fig():
    d = json.load(open(os.path.join(RES, "sparse_vs_soft_multiasset.json")))
    rows = [v for v in d.values() if "SP" in v and "SF" in v]
    cats = ["C1", "C2", "C3", "C4", "C5"]
    means, sems, ns = [], [], []
    for c in cats:
        dd = np.array([(v["SF"] - v["SP"]) / v["SF"] * 100
                       for v in rows if v.get("category") == c])
        means.append(dd.mean()); sems.append(dd.std() / np.sqrt(len(dd))); ns.append(len(dd))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axhline(0, color="gray", lw=0.8)
    ax.errorbar(range(5), means, yerr=sems, fmt="o", capsize=4, color="C3")
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cats, ns)])
    ax.set_ylabel("relative NMSE diff (%)\n$+$ favors sparsemax")
    ax.set_title("Sparsemax vs softmax by asset category (139 settings)\n"
                 "No blue-chip $\\to$ speculative noise gradient")
    fig.tight_layout()
    p = os.path.join(FIG, "fig_category_gradient.pdf")
    fig.savefig(p, bbox_inches="tight"); print("wrote", p)


if __name__ == "__main__":
    injection_fig()
    category_fig()
