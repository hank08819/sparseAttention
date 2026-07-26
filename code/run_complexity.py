"""
run_complexity.py — Computational complexity analysis
======================================================
Reports parameter counts and training time per epoch for all models:
  MSCA-SBiGRU, LSTM, GRU, iTransformer, PatchTST, Informer, TFT, N-BEATS, TimesNet
  + three MSCA ablation variants

Usage:
    cd code-data-pr+/code
    python examples/run_complexity.py
"""

import sys, os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msca_sparsity import MSCABiGRU
from msca_sparsity.train import Trainer

# Import baselines from extended script (reuse implementations)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_extended_baselines import (
    iTransformerBaseline, PatchTSTBaseline, InformerBaseline,
    TFTBaseline, NBEATSBaseline, TimesNetBaseline,
)

DEVICE     = "cpu"
FEAT_DIM   = 7
SEQ_LEN    = 30
BATCH_SIZE = 64
N_WARMUP   = 3    # warmup epochs before timing
N_TIME     = 10   # epochs to time

# Synthetic dataset matching real data size
N_TRAIN = 6000
N_VAL   = 800


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_fake_data(n, feat_dim=FEAT_DIM, seq_len=SEQ_LEN):
    X   = np.random.randn(n, seq_len, feat_dim).astype(np.float32)
    Xca = np.random.randn(n, seq_len, feat_dim).astype(np.float32)
    y   = np.random.randn(n).astype(np.float32)
    return X, Xca, y


def time_model(model, n_warmup=N_WARMUP, n_time=N_TIME):
    """Return seconds per epoch on synthetic data."""
    trainer = Trainer(model, lr=5e-4, epochs=n_warmup + n_time, patience=9999,
                      warmup=9999, batch_size=BATCH_SIZE, device=DEVICE, verbose=False)
    X_tr, Xca_tr, y_tr   = make_fake_data(N_TRAIN)
    X_val, Xca_val, y_val = make_fake_data(N_VAL)

    # warmup (JIT, cache, etc.)
    trainer.epochs = n_warmup
    trainer.fit(X_tr, Xca_tr, y_tr, X_val, Xca_val, y_val)

    # timed run
    t0 = time.time()
    trainer.epochs = n_warmup + n_time
    trainer.warmup = n_warmup + n_time   # disable early stopping during timing
    trainer.fit(X_tr, Xca_tr, y_tr, X_val, Xca_val, y_val)
    elapsed = time.time() - t0
    return elapsed / n_time


class LSTMBaseline(nn.Module):
    def __init__(self, feat_dim=7, hidden=64, dropout=0.20):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x_tgt, x_ca=None):
        out, _ = self.lstm(x_tgt)
        return self.fc(out[:, -1, :]).squeeze(-1), None, torch.tensor(0., device=x_tgt.device)


class GRUBaseline(nn.Module):
    def __init__(self, feat_dim=7, hidden=64, dropout=0.20):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden, num_layers=2,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x_tgt, x_ca=None):
        out, _ = self.gru(x_tgt)
        return self.fc(out[:, -1, :]).squeeze(-1), None, torch.tensor(0., device=x_tgt.device)


MODELS = {
    # ── Proposed ──────────────────────────────────────────────────────────────
    "MSCA-SBiGRU (Full)":
        lambda: MSCABiGRU(feat_dim=FEAT_DIM, hidden=64, n_heads=4,
                          use_sparsemax=True, use_multiscale=True, use_cross_asset=True),
    "  ABL-A (Softmax)":
        lambda: MSCABiGRU(feat_dim=FEAT_DIM, hidden=64, n_heads=4,
                          use_sparsemax=False, use_multiscale=True, use_cross_asset=True),
    "  ABL-B (SingleScale)":
        lambda: MSCABiGRU(feat_dim=FEAT_DIM, hidden=64, n_heads=4,
                          use_sparsemax=True, use_multiscale=False, use_cross_asset=True),
    "  ABL-C (NoGate)":
        lambda: MSCABiGRU(feat_dim=FEAT_DIM, hidden=64, n_heads=4,
                          use_sparsemax=True, use_multiscale=True, use_cross_asset=False),
    # ── RNN baselines ─────────────────────────────────────────────────────────
    "LSTM":
        lambda: LSTMBaseline(feat_dim=FEAT_DIM),
    "GRU":
        lambda: GRUBaseline(feat_dim=FEAT_DIM),
    # ── Transformer baselines ─────────────────────────────────────────────────
    "iTransformer":
        lambda: iTransformerBaseline(feat_dim=FEAT_DIM),
    "PatchTST":
        lambda: PatchTSTBaseline(feat_dim=FEAT_DIM),
    "Informer":
        lambda: InformerBaseline(feat_dim=FEAT_DIM),
    "TFT":
        lambda: TFTBaseline(feat_dim=FEAT_DIM),
    "N-BEATS":
        lambda: NBEATSBaseline(feat_dim=FEAT_DIM),
    "TimesNet":
        lambda: TimesNetBaseline(feat_dim=FEAT_DIM),
}


def main():
    os.makedirs("results", exist_ok=True)
    print(f"\nComplexity analysis  (device={DEVICE}, batch={BATCH_SIZE})\n")
    print(f"{'Model':<26} {'Params':>10}  {'s/epoch':>8}  {'Ratio':>7}")
    print("-" * 58)

    msca_time = None
    rows = {}

    for name, factory in MODELS.items():
        model = factory()
        n_params = count_params(model)
        t_epoch  = time_model(model)
        rows[name] = {"params": n_params, "s_per_epoch": round(t_epoch, 4)}

        if "MSCA-SBiGRU" in name and "ABL" not in name:
            msca_time = t_epoch

        ratio = t_epoch / msca_time if msca_time else float("nan")
        print(f"{name:<26} {n_params:>10,}  {t_epoch:>8.3f}s  {ratio:>6.2f}×")

    print("-" * 58)
    with open("results/complexity_analysis.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nSaved → results/complexity_analysis.json")


if __name__ == "__main__":
    main()
