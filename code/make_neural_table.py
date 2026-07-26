"""
make_neural_table.py — Emit the LaTeX rows of the neural phase-map table from
results/neural_phase_map.json (single source of truth, no hand transcription).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "neural_phase_map.json")

ORDER = [(k, s, n) for n in (512, 2048) for k in (2, 8, 32) for s in (0.5, 2)]
NAME = {"WIN": r"\textbf{win}", "TIE": "tie", "LOSE": "lose"}


def fmt(x, nd=4):
    return f"{x:+.{nd}f}".replace("+0.0000", "0").replace("-0.0000", "0")


def main():
    d = json.load(open(RES))
    print(f"% generated from neural_phase_map.json ({len(d)} cells)")
    prev_n = None
    for k, s, n in ORDER:
        key = f"k{k}_s{s}_n{n}"
        if key not in d:
            print(f"% MISSING {key}")
            continue
        v = d[key]
        if prev_n is not None and n != prev_n:
            print(r"\midrule")
        prev_n = n
        sig = "" if s != 0.5 else ""
        print(f"{n} & {k}/32 & {s} & {fmt(v['diff_mean'])} & "
              f"{v['sparse_irr_attn']:.3f} & {v['soft_irr_attn']:.3f} & "
              f"{NAME[v['regime']]} \\\\")


if __name__ == "__main__":
    main()
