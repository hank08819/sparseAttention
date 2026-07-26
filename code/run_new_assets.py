"""
run_new_assets.py — Multi-seed MSCA-SBiGRU for SOL and ADA (K=3 seeds).

SOL and ADA use BTC as the cross-asset signal (same as ETH).
Writes results into multiseed_results.json under SOL_P1 … ADA_P3.

Usage:
    cd code-data-pr+/code
    python examples/run_new_assets.py
"""
import json, os, sys, time
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msca_sparsity import MSCABiGRU, Trainer, build_features
from msca_sparsity.train import dm_test

DATA    = "../data"
JSON    = "results/multiseed_results.json"
SEEDS   = [42, 43, 44]
DEVICE  = "cpu"

CFG = dict(window=30, horizon=5, hidden=64, n_heads=4, dropout=0.25,
           lr=5e-4, epochs=200, patience=25, warmup=30,
           batch_size=64, train_ratio=0.70, val_frac=0.10, n_features=7)

DATE_TAG = {"P1": "20221101", "P2": "20231001", "P3": "20240301"}
NEW_ASSETS = ["SOL", "ADA"]
PERIODS    = ["P1", "P2", "P3"]
CA_MAP     = {"SOL": "BTC", "ADA": "BTC"}

LSTM_BENCHMARKS = {
    # LSTM NMSE approximations (same scale as existing paper table)
    # Will be computed from seed-42 run if missing; DM vs LSTM skipped for new assets
}


def csv_path(asset, period):
    return os.path.join(DATA, f"{asset}_5m_{period}_{DATE_TAG[period]}.csv")


def make_windows(close, feat_tgt, feat_ca, window=30, horizon=5):
    log_ret = np.diff(np.log(np.maximum(close, 1e-10)))
    n = min(len(feat_tgt), len(feat_ca))
    X, Xca, y, pc = [], [], [], []
    for i in range(n - window - horizon + 1):
        X.append(feat_tgt[i:i + window])
        Xca.append(feat_ca[i:i + window])
        y.append(log_ret[i + window - 1: i + window + horizon - 1].sum())
        pc.append(close[i + window - 1])
    return (np.array(X, np.float32), np.array(Xca, np.float32),
            np.array(y, np.float32), np.array(pc, np.float32))


def prepare_splits(asset, period):
    ca = CA_MAP[asset]
    close, feat_tgt = build_features(csv_path(asset, period), CFG["n_features"])
    _,     feat_ca  = build_features(csv_path(ca,    period), CFG["n_features"])
    price_range = float(close.max() - close.min())
    X, Xca, y, pc = make_windows(close, feat_tgt, feat_ca,
                                  CFG["window"], CFG["horizon"])
    n  = len(y)
    nt = int(n * CFG["train_ratio"])
    nv = int(n * CFG["val_frac"])
    F  = X.shape[-1]
    sc = MinMaxScaler(); sca = MinMaxScaler()
    sc.fit(X[:nt].reshape(-1, F)); sca.fit(Xca[:nt].reshape(-1, F))
    X   = sc.transform(X.reshape(-1, F)).reshape(X.shape)
    Xca = sca.transform(Xca.reshape(-1, F)).reshape(Xca.shape)
    return dict(tr=(X[:nt], Xca[:nt], y[:nt]),
                val=(X[nt:nt+nv], Xca[nt:nt+nv], y[nt:nt+nv]),
                te=(X[nt+nv:], Xca[nt+nv:], y[nt+nv:]),
                pte=pc[nt+nv:], price_range=price_range, F=F)


def train_eval(seed, splits):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MSCABiGRU(feat_dim=splits["F"], hidden=CFG["hidden"],
                      n_heads=CFG["n_heads"], dropout=CFG["dropout"])
    trainer = Trainer(model, lr=CFG["lr"], epochs=CFG["epochs"],
                      patience=CFG["patience"], warmup=CFG["warmup"],
                      batch_size=CFG["batch_size"], device=DEVICE, verbose=False)
    trainer.fit(*splits["tr"], *splits["val"])
    metrics = trainer.evaluate(*splits["te"], splits["pte"],
                               price_range=splits["price_range"])
    pred = trainer.predict(*splits["te"][:2])
    return metrics["NMSE"], pred


def main():
    torch.set_num_threads(4)
    os.makedirs("results", exist_ok=True)
    t0 = time.time()

    with open(JSON) as f:
        results = json.load(f)

    new_results = {}

    for asset in NEW_ASSETS:
        for period in PERIODS:
            key = f"{asset}_{period}"
            print(f"  [{asset}] {period}", end=" ", flush=True)

            splits = prepare_splits(asset, period)
            seeds_nmse = {}
            pred42 = None

            for seed in SEEDS:
                nmse, pred = train_eval(seed, splits)
                seeds_nmse[f"seed{seed}"] = nmse
                if seed == 42:
                    pred42 = pred
                print(f"s{seed}={nmse:.4f}", end=" ", flush=True)

            vals = [seeds_nmse[f"seed{s}"] for s in SEEDS]
            mean_nmse = float(np.mean(vals))
            std_nmse  = float(np.std(vals, ddof=1))

            cell = {**seeds_nmse, "mean": mean_nmse, "std": std_nmse}
            new_results[key] = {"MSCA": cell}
            print(f"→ mean={mean_nmse:.4f}±{std_nmse:.4f}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  All done in {elapsed/60:.1f} min")

    for k, v in new_results.items():
        results[k] = v
    with open(JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  New asset results merged → {JSON}")

    print("\n=== New Asset summary ===")
    for asset in NEW_ASSETS:
        for period in PERIODS:
            r = new_results[f"{asset}_{period}"]["MSCA"]
            print(f"  {asset} {period}: {r['mean']:.4f}±{r['std']:.4f}")


if __name__ == "__main__":
    main()
