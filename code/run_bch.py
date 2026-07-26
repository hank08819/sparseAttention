"""
run_bch.py — Run the K=3 seed experiment for BCH only.

Trains MSCA-SBiGRU, LSTM, and GRU on BCH × {P1,P2,P3} × {seed 42,43,44}.
Replaces DOGE_P1/P2/P3 keys in multiseed_results.json with BCH_P1/P2/P3.

Usage:
    cd code-data-pr+/code
    python examples/run_bch.py
"""

import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msca_sparsity import MSCABiGRU, Trainer, build_features
from msca_sparsity.train import dm_test

DATA   = "../data"
JSON   = "results/multiseed_results.json"
SEEDS  = [42, 43, 44]
DEVICE = "cpu"

CFG = dict(window=30, horizon=5, hidden=64, n_heads=4, dropout=0.25,
           lr=5e-4, epochs=200, patience=25, warmup=30,
           batch_size=64, train_ratio=0.70, val_frac=0.10, n_features=7)

DATE_TAG = {"P1": "20221101", "P2": "20231001", "P3": "20240301"}
PERIODS  = ["P1", "P2", "P3"]
CA       = "BTC"   # cross-asset leader for BCH


class LSTMBaseline(nn.Module):
    def __init__(self, feat_dim=7, hidden=64, dropout=0.20):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x_tgt, x_ca=None):
        out, _ = self.lstm(x_tgt)
        return self.fc(out[:, -1, :]).squeeze(-1), None, torch.tensor(0.0)


class GRUBaseline(nn.Module):
    def __init__(self, feat_dim=7, hidden=64, dropout=0.20):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden, num_layers=2,
                          batch_first=True, dropout=dropout)
        self.fc  = nn.Linear(hidden, 1)

    def forward(self, x_tgt, x_ca=None):
        out, _ = self.gru(x_tgt)
        return self.fc(out[:, -1, :]).squeeze(-1), None, torch.tensor(0.0)


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


def prepare_splits(period):
    close, feat_tgt = build_features(csv_path("BCH", period), CFG["n_features"])
    _,     feat_ca  = build_features(csv_path(CA,    period), CFG["n_features"])
    price_range = float(close.max() - close.min())

    X, Xca, y, pc = make_windows(close, feat_tgt, feat_ca,
                                  CFG["window"], CFG["horizon"])
    n  = len(y)
    nt = int(n * CFG["train_ratio"])
    nv = int(n * CFG["val_frac"])
    F  = X.shape[-1]

    sc = MinMaxScaler(); sca = MinMaxScaler()
    sc.fit(X[:nt].reshape(-1, F));  sca.fit(Xca[:nt].reshape(-1, F))
    X   = sc.transform(X.reshape(-1, F)).reshape(X.shape)
    Xca = sca.transform(Xca.reshape(-1, F)).reshape(Xca.shape)

    return dict(tr=(X[:nt], Xca[:nt], y[:nt]),
                val=(X[nt:nt+nv], Xca[nt:nt+nv], y[nt:nt+nv]),
                te=(X[nt+nv:], Xca[nt+nv:], y[nt+nv:]),
                pte=pc[nt+nv:], price_range=price_range, F=F)


def train_eval(model_cls, splits, seed, kw=None):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_cls(feat_dim=splits["F"], **(kw or {}))
    trainer = Trainer(model, lr=CFG["lr"], epochs=CFG["epochs"],
                      patience=CFG["patience"], warmup=CFG["warmup"],
                      batch_size=CFG["batch_size"], device=DEVICE, verbose=False)
    trainer.fit(*splits["tr"], *splits["val"])
    metrics = trainer.evaluate(*splits["te"], splits["pte"],
                               price_range=splits["price_range"])
    pred = trainer.predict(*splits["te"][:2])
    return metrics["NMSE"], pred


def main():
    torch.set_num_threads(6)
    os.makedirs("results", exist_ok=True)
    t0 = time.time()

    bch_results = {}
    for period in PERIODS:
        key = f"BCH_{period}"
        splits = prepare_splits(period)
        cell = {"MSCA": {}, "LSTM": {}, "GRU": {}}
        pred42 = {}

        for seed in SEEDS:
            lbl = f"seed{seed}"
            nm, pm = train_eval(MSCABiGRU, splits, seed,
                                dict(hidden=64, n_heads=4, dropout=0.25, lambda_gate=0.05))
            nl, pl = train_eval(LSTMBaseline, splits, seed, dict(hidden=64, dropout=0.20))
            ng, pg = train_eval(GRUBaseline,  splits, seed, dict(hidden=64, dropout=0.20))

            cell["MSCA"][lbl] = nm
            cell["LSTM"][lbl] = nl
            cell["GRU"][lbl]  = ng
            if seed == 42:
                pred42 = {"MSCA": pm, "LSTM": pl, "GRU": pg}

            print(f"  [BCH] {period} seed {seed} | MSCA={nm:.4f}  LSTM={nl:.4f}  GRU={ng:.4f}", flush=True)

        for mk in ["MSCA", "LSTM", "GRU"]:
            vals = [cell[mk][f"seed{s}"] for s in SEEDS]
            cell[mk]["mean"] = float(np.mean(vals))
            cell[mk]["std"]  = float(np.std(vals, ddof=1))

        y_te   = splits["te"][2]
        e_msca = y_te - pred42["MSCA"]
        e_lstm = y_te - pred42["LSTM"]
        e_gru  = y_te - pred42["GRU"]
        dm_l, p_l = dm_test(e_msca, e_lstm, horizon=5)
        dm_g, p_g = dm_test(e_msca, e_gru,  horizon=5)
        cell["DM_vs_LSTM"] = {"stat": dm_l, "pval": p_l}
        cell["DM_vs_GRU"]  = {"stat": dm_g, "pval": p_g}
        bch_results[key] = cell
        print(f"  [BCH] {period} done | DM_LSTM={dm_l:+.3f}  DM_GRU={dm_g:+.3f}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  [BCH] all periods done in {elapsed/60:.1f} min", flush=True)

    # --- merge into main results JSON (replace DOGE keys) ---
    if os.path.exists(JSON):
        with open(JSON) as f:
            results = json.load(f)
    else:
        results = {}

    for p in PERIODS:
        results.pop(f"DOGE_{p}", None)
        results[f"BCH_{p}"] = bch_results[f"BCH_{p}"]

    with open(JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results updated → {JSON}")
    print("  (DOGE keys removed, BCH keys added)")

    print("\n=== BCH summary ===")
    for period in PERIODS:
        key = f"BCH_{period}"
        c = bch_results[key]
        dl = c["DM_vs_LSTM"]; dg = c["DM_vs_GRU"]
        sl = "***" if dl["pval"]<.001 else "**" if dl["pval"]<.01 else "*" if dl["pval"]<.05 else "ns"
        sg = "***" if dg["pval"]<.001 else "**" if dg["pval"]<.01 else "*" if dg["pval"]<.05 else "ns"
        print(f"  BCH {period}: MSCA={c['MSCA']['mean']:.4f}±{c['MSCA']['std']:.4f}"
              f"  LSTM={c['LSTM']['mean']:.4f}  GRU={c['GRU']['mean']:.4f}"
              f"  DM_LSTM={dl['stat']:+.3f}({sl})  DM_GRU={dg['stat']:+.3f}({sg})")


if __name__ == "__main__":
    main()
