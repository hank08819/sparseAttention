"""
run_lnn.py — Liquid Neural Network (CfC) baseline for all 15 asset-period pairs.

Implements a Closed-form Continuous-time (CfC) recurrent cell following
Hasani et al. (2022, Nature Machine Intelligence). The CfC cell computes:

    h(t+Δ) = σ(A(x,h)) ⊙ h(t) + (1 − σ(A(x,h))) ⊙ tanh(W·[x,h] + b)

where A(x,h) = W_A·[x,h] + b_A encodes the adaptive time constant.
This closed-form solution avoids ODE integration while preserving the
continuous-time inductive bias.

Runs K=3 seeds (42,43,44) across all 15 asset-period pairs; updates
multiseed_results.json with LNN_<asset>_<period> entries.

Usage:
    cd code-data-pr+/code
    python examples/run_lnn.py
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msca_sparsity import Trainer, build_features
from msca_sparsity.train import dm_test

DATA    = "../data"
JSON    = "results/multiseed_results.json"
SEEDS   = [42, 43, 44]
DEVICE  = "cpu"

CFG = dict(window=30, horizon=5, hidden=64, dropout=0.20,
           lr=5e-4, epochs=200, patience=25, warmup=30,
           batch_size=64, train_ratio=0.70, val_frac=0.10, n_features=7)

DATE_TAG = {"P1": "20221101", "P2": "20231001", "P3": "20240301"}
ASSETS   = ["BTC", "ETH", "XRP", "LTC", "DOGE"]
PERIODS  = ["P1", "P2", "P3"]
CA_MAP   = {"BTC": "ETH", "ETH": "BTC", "XRP": "BTC",
            "LTC": "BTC", "DOGE": "BTC"}


# ── CfC cell ──────────────────────────────────────────────────────────────────

class CfCCell(nn.Module):
    """Closed-form Continuous-time cell (Hasani et al. 2022)."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        in_h = input_size + hidden_size
        self.W_h  = nn.Linear(in_h, hidden_size)
        self.W_A  = nn.Linear(in_h, hidden_size)
        self.drop = nn.Dropout(CFG["dropout"])

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        xh  = torch.cat([x, h], dim=-1)
        A   = torch.sigmoid(self.W_A(xh))          # adaptive time gate
        h_new = A * h + (1.0 - A) * torch.tanh(self.W_h(xh))
        return self.drop(h_new)


class LNNBaseline(nn.Module):
    """Two-layer CfC network with a linear output head."""
    def __init__(self, feat_dim: int = 7, hidden: int = 64, dropout: float = 0.20):
        super().__init__()
        self.cell1 = CfCCell(feat_dim, hidden)
        self.cell2 = CfCCell(hidden, hidden)
        self.fc    = nn.Linear(hidden, 1)

    def forward(self, x_tgt, x_ca=None):
        B, T, F = x_tgt.shape
        h1 = torch.zeros(B, self.cell1.hidden_size, device=x_tgt.device)
        h2 = torch.zeros(B, self.cell2.hidden_size, device=x_tgt.device)
        for t in range(T):
            h1 = self.cell1(x_tgt[:, t, :], h1)
            h2 = self.cell2(h1, h2)
        return self.fc(h2).squeeze(-1), None, torch.tensor(0.0)


# ── Data helpers (same as run_bch.py) ─────────────────────────────────────────

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
    sc.fit(X[:nt].reshape(-1, F));  sca.fit(Xca[:nt].reshape(-1, F))
    X   = sc.transform(X.reshape(-1, F)).reshape(X.shape)
    Xca = sca.transform(Xca.reshape(-1, F)).reshape(Xca.shape)

    return dict(tr=(X[:nt], Xca[:nt], y[:nt]),
                val=(X[nt:nt+nv], Xca[nt:nt+nv], y[nt:nt+nv]),
                te=(X[nt+nv:], Xca[nt+nv:], y[nt+nv:]),
                pte=pc[nt+nv:], price_range=price_range, F=F)


def train_eval(seed, splits):
    torch.manual_seed(seed); np.random.seed(seed)
    model = LNNBaseline(feat_dim=splits["F"])
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

    # Load existing results to get MSCA predictions for DM test
    with open(JSON) as f:
        main_results = json.load(f)

    lnn_results = {}

    for asset in ASSETS:
        for period in PERIODS:
            key_main = f"{asset}_{period}"
            key_lnn  = f"LNN_{asset}_{period}"
            print(f"  [{asset}] {period}", end=" ", flush=True)

            splits = prepare_splits(asset, period)
            cell   = {}
            pred42 = None

            for seed in SEEDS:
                lbl = f"seed{seed}"
                nmse, pred = train_eval(seed, splits)
                cell[lbl] = nmse
                if seed == 42:
                    pred42 = pred
                print(f"s{seed}={nmse:.4f}", end=" ", flush=True)

            vals = [cell[f"seed{s}"] for s in SEEDS]
            cell["mean"] = float(np.mean(vals))
            cell["std"]  = float(np.std(vals, ddof=1))

            # DM test: need MSCA seed-42 predictions — compute from stored NMSE
            # We don't have MSCA pred vector stored, so we compute DM using
            # test log-return errors directly (MSCA NMSE from JSON is sufficient
            # for NMSE table; DM requires re-running MSCA, done separately in
            # run_garch.py). Here we record LNN NMSE only.
            lnn_results[key_lnn] = cell
            print(f"→ mean={cell['mean']:.4f}±{cell['std']:.4f}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  All done in {elapsed/60:.1f} min")

    # Merge into main results JSON
    with open(JSON) as f:
        results = json.load(f)
    for k, v in lnn_results.items():
        results[k] = v
    with open(JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  LNN results merged → {JSON}")

    # Also save separately
    with open("results/lnn_results.json", "w") as f:
        json.dump(lnn_results, f, indent=2)
    print("  Saved → results/lnn_results.json")

    print("\n=== LNN summary ===")
    for asset in ASSETS:
        for period in PERIODS:
            r = lnn_results[f"LNN_{asset}_{period}"]
            print(f"  LNN {asset} {period}: {r['mean']:.4f}±{r['std']:.4f}  "
                  f"vs MSCA {main_results[f'{asset}_{period}']['MSCA']['mean']:.4f}")


if __name__ == "__main__":
    main()
