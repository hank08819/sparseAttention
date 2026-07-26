"""
run_gradient_efficiency.py — Gradient Efficiency Analysis
=========================================================
Measures convergence quality (NMSE) across batch sizes {64, 128, 256, 512}
for MSCA-SBiGRU, LSTM, GRU, iTransformer, TFT on 3 representative pairs.

Tests the hypothesis: sparsemax regularisation makes MSCA gradient-efficient
(converges with fewer gradient updates), while dense-parameterised baselines
degrade at large batch sizes.

Gradient updates per training run = floor(N_train / batch_size) × epochs_stopped

Output: results/gradient_efficiency.json + printed table

Usage:
    cd code-data-pr+/code
    python examples/run_gradient_efficiency.py
"""

import json, os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msca_sparsity import MSCABiGRU, Trainer, build_features

DATA   = "../data"
DEVICE = "cpu"
SEEDS  = [42, 43, 44]

# Representative pairs: one per regime, diverse assets
TEST_PAIRS = [
    ("BTC",   "ETH",  "P1"),   # BTC crisis — high volatility
    ("MATIC", "BTC",  "P2"),   # ETH-ecosystem bull run
    ("LINK",  "BTC",  "P3"),   # DeFi post-ETF
]

DATE_TAG = {"P1": "20221101", "P2": "20231001", "P3": "20240301"}
BATCH_SIZES = [64, 128, 256, 512]

BASE_CFG = dict(
    window=30, horizon=5, dropout=0.25, hidden=64, n_heads=4,
    lr=5e-4, epochs=150, patience=20, warmup=20,
    train_ratio=0.70, val_frac=0.10, n_features=7,
)


# ── Baseline models ───────────────────────────────────────────────────────────

class LSTMBaseline(nn.Module):
    def __init__(self, feat_dim=7, hidden=64, dropout=0.20):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x_tgt, x_ca=None):
        out, _ = self.lstm(x_tgt)
        return self.fc(out[:, -1, :]).squeeze(-1), None, \
               torch.tensor(0., device=x_tgt.device)


class GRUBaseline(nn.Module):
    def __init__(self, feat_dim=7, hidden=64, dropout=0.20):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden, num_layers=2,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x_tgt, x_ca=None):
        out, _ = self.gru(x_tgt)
        return self.fc(out[:, -1, :]).squeeze(-1), None, \
               torch.tensor(0., device=x_tgt.device)


class iTransformerBaseline(nn.Module):
    def __init__(self, feat_dim=7, seq_len=30, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.proj_in  = nn.Linear(seq_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*2, dropout=dropout, batch_first=True)
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.proj_out = nn.Linear(d_model * feat_dim, 1)
    def forward(self, x_tgt, x_ca=None):
        x = self.proj_in(x_tgt.transpose(1, 2))
        x = self.encoder(x).reshape(x_tgt.size(0), -1)
        return self.proj_out(x).squeeze(-1), None, \
               torch.tensor(0., device=x_tgt.device)


class _GRN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1=nn.Linear(d,d); self.fc2=nn.Linear(d,d)
        self.gate=nn.Linear(d,d); self.ln=nn.LayerNorm(d)
    def forward(self, x):
        h = self.fc2(F.elu(self.fc1(x)))
        return self.ln(x + torch.sigmoid(self.gate(x)) * h)

class TFTBaseline(nn.Module):
    def __init__(self, feat_dim=7, seq_len=30, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model)
        self.pos  = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.grn  = _GRN(d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.attn = nn.TransformerEncoder(enc, num_layers=2)
        self.out  = nn.Linear(d_model, 1)
    def forward(self, x_tgt, x_ca=None):
        x = self.grn(self.proj(x_tgt) + self.pos)
        return self.out(self.attn(x)[:, -1, :]).squeeze(-1), None, \
               torch.tensor(0., device=x_tgt.device)


MODELS = {
    "MSCA":         lambda f: MSCABiGRU(feat_dim=f, hidden=64, n_heads=4,
                                         dropout=0.25, lambda_gate=0.05),
    "LSTM":         lambda f: LSTMBaseline(feat_dim=f),
    "GRU":          lambda f: GRUBaseline(feat_dim=f),
    "iTransformer": lambda f: iTransformerBaseline(feat_dim=f),
    "TFT":          lambda f: TFTBaseline(feat_dim=f),
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def csv_path(asset, period):
    return os.path.join(DATA, f"{asset}_5m_{period}_{DATE_TAG[period]}.csv")


def prepare_splits(asset, ca, period):
    close, feat_tgt = build_features(csv_path(asset, period), BASE_CFG["n_features"])
    _, feat_ca      = build_features(csv_path(ca,    period), BASE_CFG["n_features"])
    price_range = float(close.max() - close.min())
    log_ret = np.diff(np.log(np.maximum(close, 1e-10)))
    n = min(len(feat_tgt), len(feat_ca))
    w, h = BASE_CFG["window"], BASE_CFG["horizon"]
    X, Xca, y, pc = [], [], [], []
    for i in range(n - w - h + 1):
        X.append(feat_tgt[i:i+w]); Xca.append(feat_ca[i:i+w])
        y.append(log_ret[i+w-1:i+w+h-1].sum()); pc.append(close[i+w-1])
    X  = np.array(X,  np.float32)
    Xca= np.array(Xca,np.float32)
    y  = np.array(y,  np.float32)
    pc = np.array(pc, np.float32)
    nt = int(len(y) * BASE_CFG["train_ratio"])
    nv = int(len(y) * BASE_CFG["val_frac"])
    F  = X.shape[-1]
    sc = MinMaxScaler(); sca = MinMaxScaler()
    sc.fit(X[:nt].reshape(-1,F)); sca.fit(Xca[:nt].reshape(-1,F))
    X   = sc.transform(X.reshape(-1,F)).reshape(X.shape)
    Xca = sca.transform(Xca.reshape(-1,F)).reshape(Xca.shape)
    return dict(
        tr=(X[:nt], Xca[:nt], y[:nt]),
        val=(X[nt:nt+nv], Xca[nt:nt+nv], y[nt:nt+nv]),
        te=(X[nt+nv:], Xca[nt+nv:], y[nt+nv:]),
        pte=pc[nt+nv:], price_range=price_range, F=F, N_train=nt,
    )


def run_one(model_factory, splits, seed, batch_size):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_factory(splits["F"])
    trainer = Trainer(
        model, lr=BASE_CFG["lr"], epochs=BASE_CFG["epochs"],
        patience=BASE_CFG["patience"], warmup=BASE_CFG["warmup"],
        batch_size=batch_size, device=DEVICE, verbose=False,
    )
    trainer.fit(*splits["tr"], *splits["val"])
    pred = trainer.predict(*splits["te"][:2])
    y_te = splits["te"][2]
    pc   = splits["pte"]
    pr   = splits["price_range"]
    true_p = pc * np.exp(y_te)
    pred_p = pc * np.exp(pred)
    nmse = float(np.mean((true_p - pred_p)**2) / (pr**2 + 1e-12) * 1e3)
    mse  = float(np.mean((y_te - pred)**2))
    # approximate gradient updates used
    batches_per_epoch = max(1, splits["N_train"] // batch_size)
    return nmse, mse, batches_per_epoch


def main():
    os.makedirs("results", exist_ok=True)
    out_path = "results/gradient_efficiency.json"
    results = {}

    print("\n" + "=" * 80)
    print("GRADIENT EFFICIENCY ANALYSIS")
    print("Hypothesis: sparsemax regularisation → convergence with fewer gradient steps")
    print("=" * 80)

    for asset, ca, period in TEST_PAIRS:
        pair_key = f"{asset}_{period}"
        print(f"\n── Pair: {asset} {period}  (cross-asset: {ca}) ──")
        try:
            splits = prepare_splits(asset, ca, period)
        except Exception as e:
            print(f"  ERR: {e}"); continue

        N_train = splits["N_train"]
        results[pair_key] = {}

        for mname, mfactory in MODELS.items():
            results[pair_key][mname] = {}
            row = f"  {mname:<14}"
            ref_nmse = None   # batch=64 baseline

            for bs in BATCH_SIZES:
                nmses, mses = [], []
                for seed in SEEDS:
                    nmse, mse, bpe = run_one(mfactory, splits, seed, bs)
                    nmses.append(nmse); mses.append(mse)

                mean_nmse = float(np.mean(nmses))
                mean_mse  = float(np.mean(mses))
                grad_updates = bpe * BASE_CFG["epochs"]  # upper bound

                if ref_nmse is None:
                    ref_nmse = mean_nmse
                    ratio = 1.0
                    flag  = ""
                else:
                    ratio = mean_nmse / ref_nmse
                    flag  = " ⚠" if ratio > 1.5 else (" ✓" if ratio < 1.05 else "")

                results[pair_key][mname][str(bs)] = {
                    "NMSE_mean": mean_nmse,
                    "NMSE_std":  float(np.std(nmses)),
                    "MSE_mean":  mean_mse,
                    "ratio_vs_bs64": ratio,
                    "approx_grad_updates": grad_updates,
                }
                row += f"  bs={bs}: {mean_nmse:.4f}({ratio:.2f}×){flag}"

            print(row)

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY: Mean NMSE ratio (bs=256 / bs=64) — lower = more gradient-efficient")
    print(f"{'Model':<14}", end="")
    for asset, ca, period in TEST_PAIRS:
        print(f"  {asset}_{period}", end="")
    print("  MEAN")
    print("-" * 80)

    for mname in MODELS:
        print(f"{mname:<14}", end="")
        ratios = []
        for asset, ca, period in TEST_PAIRS:
            pk = f"{asset}_{period}"
            if pk in results and mname in results[pk]:
                r = results[pk][mname].get("256", {}).get("ratio_vs_bs64", float("nan"))
            else:
                r = float("nan")
            ratios.append(r)
            print(f"  {r:6.2f}×    ", end="")
        mean_r = float(np.nanmean(ratios))
        flag = " ← ROBUST" if mean_r < 1.1 else (" ← DEGRADED" if mean_r > 1.5 else "")
        print(f"  {mean_r:.2f}×{flag}")

    print("\nGradient updates at bs=64  (epochs=150, N_train≈6000):",
          6000 // 64 * 150, "steps")
    print("Gradient updates at bs=256 (epochs=150, N_train≈6000):",
          6000 // 256 * 150, "steps")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
