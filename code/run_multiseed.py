"""
run_multiseed.py — K=3 seed experiment for all 15 asset-period pairs.

Trains MSCA-SBiGRU, LSTM, and GRU on every combination with seeds 42/43/44.
Reports mean ± std NMSE (K=3) and DM statistics from seed 42.

Usage:
    cd code-data-pr+/code
    python examples/run_multiseed.py

Output:
    results/multiseed_results.json   — full numeric results
    results/multiseed_table.txt      — copy-paste LaTeX table fragment
"""

import json, os, sys, time
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msca_sparsity import MSCABiGRU, Trainer, build_features
from msca_sparsity.train import dm_test

# ─── Configuration ────────────────────────────────────────────────────────────
DATA = "../data"
SEEDS = [42, 43, 44]
DEVICE = "cpu"  # MPS 4× slower than CPU for this small-batch workload

CFG = dict(window=30, horizon=5, hidden=64, n_heads=4, dropout=0.25,
           lr=5e-4, epochs=200, patience=25, warmup=30,
           batch_size=64, train_ratio=0.70, val_frac=0.10, n_features=7)

# cross-asset leader for each target
CA_MAP = {"BTC": "ETH", "ETH": "BTC", "XRP": "BTC", "LTC": "BTC", "DOGE": "BTC"}
DATE_TAG = {"P1": "20221101", "P2": "20231001", "P3": "20240301"}

ASSETS  = ["BTC", "ETH", "XRP", "LTC", "DOGE"]
PERIODS = ["P1", "P2", "P3"]



# ─── Baseline models (LSTM / GRU) ─────────────────────────────────────────────
class LSTMBaseline(nn.Module):
    def __init__(self, feat_dim=7, hidden=64, dropout=0.20):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x_tgt, x_ca=None):
        out, _ = self.lstm(x_tgt)
        pred   = self.fc(out[:, -1, :]).squeeze(-1)
        return pred, None, torch.tensor(0.0, device=x_tgt.device)


class GRUBaseline(nn.Module):
    def __init__(self, feat_dim=7, hidden=64, dropout=0.20):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden, num_layers=2,
                          batch_first=True, dropout=dropout)
        self.fc  = nn.Linear(hidden, 1)

    def forward(self, x_tgt, x_ca=None):
        out, _ = self.gru(x_tgt)
        pred   = self.fc(out[:, -1, :]).squeeze(-1)
        return pred, None, torch.tensor(0.0, device=x_tgt.device)


# ─── Data helpers ─────────────────────────────────────────────────────────────
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
    _,     feat_ca  = build_features(csv_path(ca, period),    CFG["n_features"])
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

    return dict(
        tr=(X[:nt], Xca[:nt], y[:nt]),
        val=(X[nt:nt+nv], Xca[nt:nt+nv], y[nt:nt+nv]),
        te=(X[nt+nv:], Xca[nt+nv:], y[nt+nv:]),
        pte=pc[nt+nv:], price_range=price_range, F=F,
    )


def train_and_eval(model_class, splits, seed, model_kwargs=None):
    """Train one model with one seed; return (nmse, predictions)."""
    torch.manual_seed(seed); np.random.seed(seed)
    kw = model_kwargs or {}
    model = model_class(feat_dim=splits["F"], **kw)
    trainer = Trainer(model, lr=CFG["lr"], epochs=CFG["epochs"],
                      patience=CFG["patience"], warmup=CFG["warmup"],
                      batch_size=CFG["batch_size"], device=DEVICE, verbose=False)
    trainer.fit(*splits["tr"], *splits["val"])
    metrics = trainer.evaluate(*splits["te"], splits["pte"],
                               price_range=splits["price_range"])
    pred = trainer.predict(*splits["te"][:2])
    return metrics["NMSE"], pred


# ─── Per-asset worker (runs in a child process) ───────────────────────────────
def run_asset(asset):
    """Train all 3 periods × 3 seeds × 3 models for one asset; return results dict."""
    torch.set_num_threads(3)  # 5 workers × 3 threads ≈ 16 cores total
    # Redirect stdout to a per-asset log so output survives the spawn boundary
    import sys as _sys
    _log = open(f"results/worker_{asset}.log", "w", buffering=1)
    _sys.stdout = _log
    t0 = time.time()
    asset_results = {}
    for period in PERIODS:
        key = f"{asset}_{period}"
        splits = prepare_splits(asset, period)
        cell = {"MSCA": {}, "LSTM": {}, "GRU": {}}
        pred42 = {}

        for seed in SEEDS:
            label = f"seed{seed}"
            nmse_m, pred_m = train_and_eval(
                MSCABiGRU, splits, seed,
                dict(hidden=64, n_heads=4, dropout=0.25, lambda_gate=0.05))
            cell["MSCA"][label] = nmse_m
            if seed == 42:
                pred42["MSCA"] = pred_m

            nmse_l, pred_l = train_and_eval(
                LSTMBaseline, splits, seed, dict(hidden=64, dropout=0.20))
            cell["LSTM"][label] = nmse_l
            if seed == 42:
                pred42["LSTM"] = pred_l

            nmse_g, pred_g = train_and_eval(
                GRUBaseline, splits, seed, dict(hidden=64, dropout=0.20))
            cell["GRU"][label] = nmse_g
            if seed == 42:
                pred42["GRU"] = pred_g

            print(f"  [{asset}] {period} seed {seed} | MSCA={nmse_m:.4f}"
                  f"  LSTM={nmse_l:.4f}  GRU={nmse_g:.4f}", flush=True)

        for model_key in ["MSCA", "LSTM", "GRU"]:
            vals = [cell[model_key][f"seed{s}"] for s in SEEDS]
            cell[model_key]["mean"] = float(np.mean(vals))
            cell[model_key]["std"]  = float(np.std(vals, ddof=1))

        y_te   = splits["te"][2]
        e_msca = y_te - pred42["MSCA"]
        e_lstm = y_te - pred42["LSTM"]
        e_gru  = y_te - pred42["GRU"]
        dm_vs_lstm, p_vs_lstm = dm_test(e_msca, e_lstm, horizon=5)
        dm_vs_gru,  p_vs_gru  = dm_test(e_msca, e_gru,  horizon=5)
        cell["DM_vs_LSTM"] = {"stat": dm_vs_lstm, "pval": p_vs_lstm}
        cell["DM_vs_GRU"]  = {"stat": dm_vs_gru,  "pval": p_vs_gru}
        asset_results[key] = cell

    elapsed = time.time() - t0
    print(f"  [{asset}] done in {elapsed/60:.1f} min", flush=True)
    _log.close()
    return asset_results


# ─── Main loop ────────────────────────────────────────────────────────────────
def main():
    os.makedirs("results", exist_ok=True)
    t_start = time.time()
    print(f"Device: {DEVICE}  batch_size: {CFG['batch_size']}  "
          f"Seeds: {SEEDS}  Workers: {len(ASSETS)}", flush=True)

    with mp.Pool(processes=len(ASSETS)) as pool:
        asset_dicts = pool.map(run_asset, ASSETS)

    # Merge per-asset worker logs into the main log
    for asset in ASSETS:
        wlog = f"results/worker_{asset}.log"
        if os.path.exists(wlog):
            with open(wlog) as f:
                sys.stdout.write(f.read())
            os.remove(wlog)

    results = {}
    for d in asset_dicts:
        results.update(d)

    with open("results/multiseed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → results/multiseed_results.json", flush=True)

    print_latex_table(results)
    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time/60:.1f} min")


def sig_str(pval, stat):
    s = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    return f"${stat:+.3f}^{{{s}}}$" if s else f"${stat:+.3f}$"


def print_latex_table(results):
    print("\n" + "="*70)
    print("LATEX TABLE FRAGMENT")
    print("="*70)
    sig_wins = 0; sig_losses = 0
    for period in PERIODS:
        print(f"\\midrule")
        for asset in ASSETS:
            key = f"{asset}_{period}"
            c = results[key]
            msca_mean = c["MSCA"]["mean"]; msca_std = c["MSCA"]["std"]
            lstm_mean = c["LSTM"]["mean"]; gru_mean  = c["GRU"]["mean"]
            dm_l = c["DM_vs_LSTM"]; dm_g = c["DM_vs_GRU"]
            if dm_l["pval"] < 0.05 and dm_l["stat"] < 0:
                sig_wins += 1
            if dm_g["pval"] < 0.05 and dm_g["stat"] < 0:
                sig_wins += 1
            if dm_l["pval"] < 0.05 and dm_l["stat"] > 0:
                sig_losses += 1
            if dm_g["pval"] < 0.05 and dm_g["stat"] > 0:
                sig_losses += 1
            row = (f" & {asset} & "
                   f"${msca_mean:.4f}\\pm{msca_std:.4f}$ & "
                   f"${lstm_mean:.4f}$ & "
                   f"${gru_mean:.4f}$ & "
                   f"{sig_str(dm_l['pval'], dm_l['stat'])} & "
                   f"{sig_str(dm_g['pval'], dm_g['stat'])} \\\\")
            print(row)
    print(f"\nTotal: {sig_wins} significant DM wins, {sig_losses} significant DM losses")


if __name__ == "__main__":
    main()
