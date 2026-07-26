"""
examples/run_experiment.py — End-to-end example for MSCA-SBiGRU.

Usage:
    python examples/run_experiment.py --tgt data/BTC_5m_P1_20221101.csv \
                                       --ca  data/ETH_5m_P1_20221101.csv

Requires: msca_sparsity installed (`pip install -e ..` from this directory)
or the package root on PYTHONPATH.
"""

import argparse, json, os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from msca_sparsity import MSCABiGRU, Trainer, build_features
from msca_sparsity.train import dm_test

CFG = dict(window=30, horizon=5, hidden=64, n_heads=4, dropout=0.25,
           lr=5e-4, epochs=200, patience=25, warmup=30,
           batch_size=64, train_ratio=0.70, val_frac=0.10,
           n_features=7, device="cpu", seed=42)


def make_windows(close, feat_tgt, feat_ca, window=30, horizon=5):
    log_ret = np.diff(np.log(close))
    n = min(len(feat_tgt), len(feat_ca))
    X, Xca, y, pc = [], [], [], []
    for i in range(n - window - horizon + 1):
        X.append(feat_tgt[i:i + window])
        Xca.append(feat_ca[i:i + window])
        y.append(log_ret[i + window - 1: i + window + horizon - 1].sum())
        pc.append(close[i + window - 1])
    return (np.array(X, dtype=np.float32), np.array(Xca, dtype=np.float32),
            np.array(y, dtype=np.float32), np.array(pc, dtype=np.float32))


def main(tgt_path: str, ca_path: str, out_dir: str = "results"):
    np.random.seed(CFG["seed"])
    import torch; torch.manual_seed(CFG["seed"])

    # ── Features ──────────────────────────────────────────────────────────────
    close, feat_tgt = build_features(tgt_path, CFG["n_features"])
    _,     feat_ca  = build_features(ca_path,  CFG["n_features"])
    price_range = float(close.max() - close.min())

    X, Xca, y, pc = make_windows(close, feat_tgt, feat_ca,
                                  CFG["window"], CFG["horizon"])
    n  = len(y)
    nt = int(n * CFG["train_ratio"])
    nv = int(n * CFG["val_frac"])
    F  = X.shape[-1]

    sc  = MinMaxScaler(); sca = MinMaxScaler()
    sc.fit(X[:nt].reshape(-1, F));   sca.fit(Xca[:nt].reshape(-1, F))
    X   = sc.transform(X.reshape(-1, F)).reshape(X.shape)
    Xca = sca.transform(Xca.reshape(-1, F)).reshape(Xca.shape)

    splits = dict(
        tr=(X[:nt], Xca[:nt], y[:nt]),
        val=(X[nt:nt+nv], Xca[nt:nt+nv], y[nt:nt+nv]),
        te=(X[nt+nv:], Xca[nt+nv:], y[nt+nv:]),
        pte=pc[nt+nv:],
    )

    # ── Train full model ──────────────────────────────────────────────────────
    model   = MSCABiGRU(feat_dim=F, hidden=CFG["hidden"],
                        n_heads=CFG["n_heads"], dropout=CFG["dropout"])
    trainer = Trainer(model, lr=CFG["lr"], epochs=CFG["epochs"],
                      patience=CFG["patience"], warmup=CFG["warmup"],
                      batch_size=CFG["batch_size"], device=CFG["device"])
    trainer.fit(*splits["tr"], *splits["val"])

    metrics = trainer.evaluate(*splits["te"], splits["pte"], price_range)
    pred    = trainer.predict(*splits["te"][:2])
    print(f"\nFull model  NMSE={metrics['NMSE']:.4f}  R²={metrics['R2']:.4f}")

    # ── Quick ablation ─────────────────────────────────────────────────────────
    results = {"full": metrics}
    for variant, kwargs in [
        ("no_cross", dict(use_cross_asset=False)),
        ("no_multi",  dict(use_multiscale=False)),
    ]:
        m2 = MSCABiGRU(feat_dim=F, **{**dict(hidden=CFG["hidden"],
                        n_heads=CFG["n_heads"], dropout=CFG["dropout"]), **kwargs})
        t2 = Trainer(m2, **{k: CFG[k] for k in
                     ["lr","epochs","patience","warmup","batch_size","device"]})
        t2.fit(*splits["tr"], *splits["val"])
        results[variant] = t2.evaluate(*splits["te"], splits["pte"], price_range)
        pred2 = t2.predict(*splits["te"][:2])
        stat, pval = dm_test(splits["te"][2] - pred, splits["te"][2] - pred2,
                             horizon=CFG["horizon"])
        results[variant]["DM_vs_full"] = stat
        results[variant]["p_vs_full"]  = pval
        print(f"  {variant:12s}  NMSE={results[variant]['NMSE']:.4f}"
              f"  DM_vs_full={stat:+.3f}  p={pval:.3f}")

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "experiment_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tgt", required=True, help="Target asset CSV path")
    p.add_argument("--ca",  required=True, help="Cross-asset CSV path")
    p.add_argument("--out", default="results", help="Output directory")
    args = p.parse_args()
    main(args.tgt, args.ca, args.out)
