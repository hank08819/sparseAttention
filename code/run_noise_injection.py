"""
run_noise_injection.py — Theorem 1 on real crypto data, with ground-truth noise.

MISA's sparse selection acts over TIME STEPS (alpha = sparsemax over the window).
Theorem 1 says: when irrelevant groups (here, time steps) carry residual noise,
bounded softmax must pass some of it to the prediction (weight >= e^{-2B}/G on
every step), while an exact-zero selector removes it entirely.

We instantiate this directly. Into each real 30-step window we append r extra
"time steps" of PURE NOISE that carry no information about the target. These
injected steps are known-irrelevant groups. The model is reduced to its
temporal-attention core (use_multiscale=False, use_cross_asset=False) so the
only mechanism in play is the temporal sparsemax/softmax selection.

Pre-registered predictions (from Theorem 1):
  P1  The sparsemax advantage over softmax grows with r (softmax leaks more).
  P2  Sparsemax puts ~0 total attention on the r injected steps; softmax puts
      > 0 (about r * e^{-2B}/G), leaking their noise into the prediction.

Usage (import as the msca_sparsity package):
    python run_noise_injection.py
"""
import json
import os

import numpy as np
import torch

import msca_sparsity.run_ablation as ra
from msca_sparsity import MSCABiGRU, Trainer

torch.set_num_threads(6)

ASSETS = ["BTC_P2", "SAND_P2"]      # one liquid (C1), one speculative (C5)
R_VALUES = [0, 8, 16, 32]           # injected noise steps, at sigma_z = 1
SIGMA = 1.0
CORE = dict(use_multiscale=False, use_cross_asset=False)   # temporal-attn core
OUT = "results/noise_injection.json"


def prepare_inject(asset, period, r, sigma_z, seed=0):
    """Real 30-step windows + r appended pure-noise steps (known irrelevant)."""
    from sklearn.preprocessing import MinMaxScaler
    ca = ra.CA_MAP[asset]
    close, feat_tgt = ra.build_features(ra.csv_path(asset, period), ra.CFG["n_features"])
    _, feat_ca = ra.build_features(ra.csv_path(ca, period), ra.CFG["n_features"])
    price_range = float(close.max() - close.min())
    X, Xca, y, pc = ra.make_windows(close, feat_tgt, feat_ca,
                                    ra.CFG["window"], ra.CFG["horizon"])
    n = len(y); nt = int(n * ra.CFG["train_ratio"]); nv = int(n * ra.CFG["val_frac"])
    F = X.shape[-1]
    sc, sca = MinMaxScaler(), MinMaxScaler()
    sc.fit(X[:nt].reshape(-1, F)); sca.fit(Xca[:nt].reshape(-1, F))
    X = sc.transform(X.reshape(-1, F)).reshape(X.shape)
    Xca = sca.transform(Xca.reshape(-1, F)).reshape(Xca.shape)

    if r > 0:
        rng = np.random.default_rng(seed)
        # noise per feature column: N(train mean, (sigma_z * train std)^2)
        mu = X[:nt].reshape(-1, F).mean(0)
        sd = X[:nt].reshape(-1, F).std(0) + 1e-8
        def inject(A):
            noise = rng.normal(mu, sigma_z * sd, size=(A.shape[0], r, F)).astype(np.float32)
            return np.concatenate([A, noise], axis=1)
        X, Xca = inject(X), inject(Xca)

    return dict(tr=(X[:nt], Xca[:nt], y[:nt]),
                val=(X[nt:nt+nv], Xca[nt:nt+nv], y[nt:nt+nv]),
                te=(X[nt+nv:], Xca[nt+nv:], y[nt+nv:]),
                pte=pc[nt+nv:], price_range=price_range, F=F, r=r)


def train_and_probe(splits, use_sparsemax, seed=42):
    """Train the temporal-attn core; return NMSE and mean attention on injected steps."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = MSCABiGRU(feat_dim=splits["F"], hidden=64, n_heads=4, dropout=0.25,
                      lambda_gate=0.05, use_sparsemax=use_sparsemax, **CORE)
    tr = Trainer(model, lr=ra.CFG["lr"], epochs=ra.CFG["epochs"],
                 patience=ra.CFG["patience"], warmup=ra.CFG["warmup"],
                 batch_size=ra.CFG["batch_size"], device="cpu", verbose=False)
    tr.fit(*splits["tr"], *splits["val"])
    nmse = tr.evaluate(*splits["te"], splits["pte"],
                       price_range=splits["price_range"])["NMSE"]
    # attention on injected steps (test set)
    r = splits["r"]
    inj_w = 0.0
    if r > 0:
        model.eval()
        with torch.no_grad():
            Xte = torch.tensor(splits["te"][0], dtype=torch.float32)
            Xca = torch.tensor(splits["te"][1], dtype=torch.float32)
            _, a1, _ = model(Xte, Xca)          # a1: (n, 30+r)
            inj_w = float(a1[:, 30:].sum(dim=1).mean())   # total weight on noise steps
    return float(nmse), inj_w


def run():
    os.makedirs("results", exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for key in ASSETS:
        asset, period = key.rsplit("_", 1)
        res.setdefault(key, {})
        for r in R_VALUES:
            rk = f"r{r}"
            if rk in res[key] and "sparse" in res[key][rk] and "soft" in res[key][rk]:
                continue
            splits = prepare_inject(asset, period, r, SIGMA, seed=0)
            cell = res[key].get(rk, {})
            for label, sp in [("sparse", True), ("soft", False)]:
                if label in cell:
                    continue
                nmse, inj = train_and_probe(splits, sp)
                cell[label] = {"NMSE": nmse, "inj_attn": inj}
                print(f"{key} r={r} {label}: NMSE={nmse:.4f}  inj_attn={inj:.4f}",
                      flush=True)
                res[key][rk] = cell
                json.dump(res, open(OUT, "w"), indent=2)
    return res


def report(res):
    print("\n" + "=" * 70)
    print("NOISE-INJECTION TEST (temporal-attn core, sigma_z=1)")
    for key, rr in res.items():
        print(f"\n{key}:  r  | NMSE sparse  NMSE soft  gap(soft-sparse) | inj_attn sp/soft")
        for r in R_VALUES:
            c = rr.get(f"r{r}")
            if c and "sparse" in c and "soft" in c:
                sp, so = c["sparse"], c["soft"]
                gap = so["NMSE"] - sp["NMSE"]
                print(f"       {r:3d} | {sp['NMSE']:10.4f}  {so['NMSE']:9.4f}  {gap:+15.4f} "
                      f"| {sp['inj_attn']:.4f} / {so['inj_attn']:.4f}")
    print("\nP1: gap should rise with r.  P2: sparse inj_attn ~0 < soft inj_attn.")
    print("=" * 70)


if __name__ == "__main__":
    report(run())
