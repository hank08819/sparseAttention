"""
run_score_separation.py — Can a score-separation regularizer restore support
recovery for sparsemax?

The injected-noise probe (run_noise_injection.py) showed WHY sparsemax fails on
real data: after full training, the learned temporal-attention scores do not
separate known-irrelevant steps, so exact-zero selection has nothing to remove.
Theorem 3 (support recovery) requires a positive score margin gamma; this
experiment tests whether adding a sharpening penalty that pushes the attention
distribution away from uniform (thereby widening score gaps) restores the
mechanism.

Regularizer (Gini / Tsallis-2 sharpening), added to the training loss:

    L_sep = lambda_a * mean_batch( 1 - sum_j a_j^2 )

which is ~1 - 1/T for uniform attention and 0 for a one-hot distribution.
It is differentiable through sparsemax and softmax alike.

Pre-registered success criteria (all three must hold, judged on the injected
probe of run_noise_injection.py, same settings and seed):
  S1  inj_attn(sparse+sep) falls well below the sparse baseline (toward 0).
  S2  NMSE(sparse+sep) <= NMSE(softmax baseline), with the gap growing in r.
  S3  No harm at r=0 (NMSE within noise of the sparse baseline).

Failure on any criterion is reported as such. Usage (as msca_sparsity pkg):
    python run_score_separation.py
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import msca_sparsity.run_ablation as ra
import msca_sparsity.run_noise_injection as ni
from msca_sparsity import MSCABiGRU, Trainer

torch.set_num_threads(8)

ASSETS = ["BTC_P2", "SAND_P2"]
R_VALUES = [0, 8, 16, 32]
LAMBDAS = [0.02, 0.1]
OUT = "results/score_separation.json"
BASE = "results/noise_injection.json"


class SepSharp(nn.Module):
    """Wraps the temporal-attention core; adds the sharpening penalty through
    the gate-loss channel so the stock Trainer needs no changes."""

    def __init__(self, feat_dim, lam):
        super().__init__()
        self.base = MSCABiGRU(feat_dim=feat_dim, hidden=64, n_heads=4,
                              dropout=0.25, lambda_gate=0.05,
                              use_sparsemax=True, use_multiscale=False,
                              use_cross_asset=False)
        self.lam = lam

    def forward(self, x_tgt, x_ca):
        pred, a1, gl = self.base(x_tgt, x_ca)
        sep = self.lam * (1.0 - (a1 ** 2).sum(dim=-1)).mean()
        return pred, a1, gl + sep


def train_and_probe(splits, lam, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SepSharp(splits["F"], lam)
    tr = Trainer(model, lr=ra.CFG["lr"], epochs=ra.CFG["epochs"],
                 patience=ra.CFG["patience"], warmup=ra.CFG["warmup"],
                 batch_size=ra.CFG["batch_size"], device="cpu", verbose=False)
    tr.fit(*splits["tr"], *splits["val"])
    nmse = tr.evaluate(*splits["te"], splits["pte"],
                       price_range=splits["price_range"])["NMSE"]
    r = splits["r"]
    inj_w = 0.0
    if r > 0:
        model.eval()
        with torch.no_grad():
            Xte = torch.tensor(splits["te"][0], dtype=torch.float32)
            Xca = torch.tensor(splits["te"][1], dtype=torch.float32)
            _, a1, _ = model(Xte, Xca)
            inj_w = float(a1[:, 30:].sum(dim=1).mean())
    return float(nmse), inj_w


def run():
    os.makedirs("results", exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for key in ASSETS:
        asset, period = key.rsplit("_", 1)
        res.setdefault(key, {})
        for r in R_VALUES:
            for lam in LAMBDAS:
                ck = f"r{r}_lam{lam}"
                if ck in res[key]:
                    continue
                splits = ni.prepare_inject(asset, period, r, ni.SIGMA, seed=0)
                t = time.time()
                nmse, inj = train_and_probe(splits, lam)
                res[key][ck] = {"NMSE": nmse, "inj_attn": inj}
                print(f"{key} r={r} lam={lam}: NMSE={nmse:.4f} "
                      f"inj_attn={inj:.4f} ({time.time()-t:.0f}s)", flush=True)
                json.dump(res, open(OUT, "w"), indent=2)
    return res


def report(res):
    base = json.load(open(BASE)) if os.path.exists(BASE) else {}
    print("\n" + "=" * 78)
    print("SCORE-SEPARATION FIX vs BASELINES (temporal core, seed 42)")
    for key in ASSETS:
        print(f"\n{key}:")
        print(f"{'r':>3} | {'base sparse':>16} {'base soft':>16} | "
              + "  ".join(f"sep lam={l}" for l in LAMBDAS))
        for r in R_VALUES:
            b = base.get(key, {}).get(f"r{r}", {})
            bs = b.get("sparse", {})
            bo = b.get("soft", {})
            cells = []
            for lam in LAMBDAS:
                c = res.get(key, {}).get(f"r{r}_lam{lam}")
                cells.append(f"{c['NMSE']:.4f}/{c['inj_attn']:.3f}"
                             if c else "--")
            print(f"{r:>3} | {bs.get('NMSE', float('nan')):>7.4f}"
                  f"/{bs.get('inj_attn', float('nan')):.3f}"
                  f" {bo.get('NMSE', float('nan')):>7.4f}"
                  f"/{bo.get('inj_attn', float('nan')):.3f} | "
                  + "  ".join(f"{c:>13}" for c in cells))
    print("\ncells: NMSE/inj_attn.  S1: sep inj_attn << base sparse.")
    print("S2: sep NMSE <= base soft, gap grows with r.  S3: no harm at r=0.")
    print("=" * 78)


if __name__ == "__main__":
    report(run())
