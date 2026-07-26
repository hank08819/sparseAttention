"""
run_transformer_dm.py
=====================
Train iTransformer and PatchTST on all 15 asset-period pairs (seeds 42/43/44),
compute DM tests against MSCA-SBiGRU (seed 42), and save results.

Runtime: ~25-35 min on CPU.

Usage:
    cd code-data-pr+/code
    python examples/run_transformer_dm.py
"""

import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msca_sparsity import MSCABiGRU, Trainer, build_features
from msca_sparsity.train import dm_test

# ── Config ────────────────────────────────────────────────────────────────────
DATA    = "../data"
SEEDS   = [42, 43, 44]
DEVICE  = "cpu"
CFG     = dict(window=30, horizon=5, hidden=64, n_heads=4, dropout=0.25,
               lr=5e-4, epochs=200, patience=25, warmup=30,
               batch_size=64, train_ratio=0.70, val_frac=0.10, n_features=7)
CA_MAP  = {"BTC":"ETH","ETH":"BTC","XRP":"BTC","LTC":"BTC","DOGE":"BTC"}
DATE_TAG= {"P1":"20221101","P2":"20231001","P3":"20240301"}
ASSETS  = ["BTC","ETH","XRP","LTC","DOGE"]
PERIODS = ["P1","P2","P3"]

# ── iTransformer ──────────────────────────────────────────────────────────────
class iTransformer(nn.Module):
    """
    Liu et al. (2024) — inverted attention across variates.
    Each of the F variates (features) is embedded from its T-length time series
    into d_model, then multi-head self-attention is applied across the F tokens.
    """
    def __init__(self, feat_dim=7, seq_len=30, d_model=64, n_heads=4,
                 n_layers=2, d_ff=128, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        # project each variate's time series (length seq_len) into d_model
        self.variate_embed = nn.Linear(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm    = nn.LayerNorm(d_model)
        # aggregate across variates and project to scalar prediction
        self.head    = nn.Linear(feat_dim * d_model, 1)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x_tgt, x_ca=None):
        # x_tgt: (B, T, F)
        B, T, F = x_tgt.shape
        # transpose → (B, F, T); embed each variate
        x = x_tgt.permute(0, 2, 1)            # (B, F, T)
        x = self.variate_embed(x)             # (B, F, d_model)
        x = self.encoder(x)                   # (B, F, d_model)
        x = self.norm(x)
        x = self.drop(x.reshape(B, -1))       # (B, F*d_model)
        pred = self.head(x).squeeze(-1)        # (B,)
        return pred, None, torch.tensor(0.0, device=x_tgt.device)


# ── PatchTST ──────────────────────────────────────────────────────────────────
class PatchTST(nn.Module):
    """
    Nie et al. (2023) — channel-independent patch transformer.
    Each variate is segmented into overlapping patches; patches are embedded
    and processed by a standard transformer encoder.
    """
    def __init__(self, feat_dim=7, seq_len=30, patch_len=8, stride=4,
                 d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.1):
        super().__init__()
        self.feat_dim  = feat_dim
        self.patch_len = patch_len
        self.stride    = stride
        self.num_patches = (seq_len - patch_len) // stride + 1  # 6 for T=30

        self.patch_embed = nn.Linear(patch_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm     = nn.LayerNorm(d_model)
        self.drop     = nn.Dropout(dropout)
        # aggregate across variates and patches
        self.head = nn.Linear(feat_dim * self.num_patches * d_model, 1)

    def forward(self, x_tgt, x_ca=None):
        # x_tgt: (B, T, F)
        B, T, F = x_tgt.shape
        x = x_tgt.permute(0, 2, 1)  # (B, F, T)

        # patching: (B, F, T) → (B*F, num_patches, patch_len)
        patches = []
        for i in range(self.num_patches):
            start = i * self.stride
            patches.append(x[:, :, start:start + self.patch_len])
        x_p = torch.stack(patches, dim=2)       # (B, F, num_patches, patch_len)
        x_p = x_p.reshape(B * F, self.num_patches, self.patch_len)

        x_p = self.patch_embed(x_p)            # (B*F, num_patches, d_model)
        x_p = self.encoder(x_p)               # (B*F, num_patches, d_model)
        x_p = self.norm(x_p)
        x_p = x_p.reshape(B, F * self.num_patches * x_p.shape[-1])
        x_p = self.drop(x_p)
        pred = self.head(x_p).squeeze(-1)      # (B,)
        return pred, None, torch.tensor(0.0, device=x_tgt.device)


# ── Data helpers (identical to run_multiseed.py) ──────────────────────────────
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
    return dict(tr=(X[:nt], Xca[:nt], y[:nt]),
                val=(X[nt:nt+nv], Xca[nt:nt+nv], y[nt:nt+nv]),
                te=(X[nt+nv:], Xca[nt+nv:], y[nt+nv:]),
                pte=pc[nt+nv:], price_range=price_range, F=F)


def train_and_predict(model_class, splits, seed, model_kwargs=None):
    """Train one model, one seed; return (nmse, predictions_on_test)."""
    torch.manual_seed(seed); np.random.seed(seed)
    kw = model_kwargs or {}
    model = model_class(**kw)
    trainer = Trainer(model, lr=CFG["lr"], epochs=CFG["epochs"],
                      patience=CFG["patience"], warmup=CFG["warmup"],
                      batch_size=CFG["batch_size"], device=DEVICE, verbose=False)
    trainer.fit(*splits["tr"], *splits["val"])
    metrics = trainer.evaluate(*splits["te"], splits["pte"],
                               price_range=splits["price_range"])
    pred = trainer.predict(*splits["te"][:2])
    return metrics["NMSE"], pred


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("results", exist_ok=True)
    t0 = time.time()

    results = {}

    for asset in ASSETS:
        for period in PERIODS:
            key = f"{asset}_{period}"
            print(f"\n[{asset} {period}]", flush=True)
            splits = prepare_splits(asset, period)
            F = splits["F"]

            # ── MSCA seed-42 predictions (reference) ──────────────────────
            nmse_msca, pred_msca = train_and_predict(
                MSCABiGRU, splits, 42,
                dict(feat_dim=F, hidden=64, n_heads=4,
                     dropout=0.25, lambda_gate=0.05))
            y_te = splits["te"][2]
            e_msca = y_te - pred_msca
            print(f"  MSCA  s42 NMSE={nmse_msca:.4f}", flush=True)

            # ── iTransformer (3 seeds for NMSE mean±std; DM from seed 42) ──
            itrans_nmse = []
            for seed in SEEDS:
                nmse_it, pred_it = train_and_predict(
                    iTransformer, splits, seed,
                    dict(feat_dim=F, seq_len=CFG["window"],
                         d_model=64, n_heads=4, n_layers=2,
                         d_ff=128, dropout=0.1))
                itrans_nmse.append(nmse_it)
                if seed == 42:
                    pred_it42 = pred_it
                print(f"  iTransformer s{seed} NMSE={nmse_it:.4f}", flush=True)

            e_it = y_te - pred_it42
            dm_it, p_it = dm_test(e_msca, e_it, horizon=5)

            # ── PatchTST (3 seeds; DM from seed 42) ───────────────────────
            ptst_nmse = []
            for seed in SEEDS:
                nmse_pt, pred_pt = train_and_predict(
                    PatchTST, splits, seed,
                    dict(feat_dim=F, seq_len=CFG["window"],
                         patch_len=8, stride=4,
                         d_model=64, n_heads=4, n_layers=2,
                         d_ff=128, dropout=0.1))
                ptst_nmse.append(nmse_pt)
                if seed == 42:
                    pred_pt42 = pred_pt
                print(f"  PatchTST  s{seed} NMSE={nmse_pt:.4f}", flush=True)

            e_pt = y_te - pred_pt42
            dm_pt, p_pt = dm_test(e_msca, e_pt, horizon=5)

            def sig(p):
                if p < 0.001: return "***"
                if p < 0.01:  return "**"
                if p < 0.05:  return "*"
                return "ns"

            print(f"  DM(MSCA vs iTransformer)={dm_it:+.3f}{sig(p_it)}  "
                  f"DM(MSCA vs PatchTST)={dm_pt:+.3f}{sig(p_pt)}", flush=True)

            results[key] = {
                "MSCA_NMSE_s42": float(nmse_msca),
                "iTransformer": {
                    "NMSE_mean": float(np.mean(itrans_nmse)),
                    "NMSE_std":  float(np.std(itrans_nmse, ddof=1)),
                    "NMSE_seeds": [float(v) for v in itrans_nmse],
                    "DM_stat": float(dm_it),
                    "DM_pval": float(p_it),
                },
                "PatchTST": {
                    "NMSE_mean": float(np.mean(ptst_nmse)),
                    "NMSE_std":  float(np.std(ptst_nmse, ddof=1)),
                    "NMSE_seeds": [float(v) for v in ptst_nmse],
                    "DM_stat": float(dm_pt),
                    "DM_pval": float(p_pt),
                },
            }

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = "results/transformer_dm_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)

    elapsed = time.time() - t0
    print(f"Total time: {elapsed/60:.1f} min", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    def sig(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    print("\n=== DM Summary (MSCA vs Transformer baselines, seed-42) ===")
    print(f"{'Pair':<12}  {'iTransformer':>18}  {'PatchTST':>18}")
    print("-" * 55)
    wins_it = wins_pt = 0
    for asset in ASSETS:
        for period in PERIODS:
            key = f"{asset}_{period}"
            r   = results[key]
            dm_it = r["iTransformer"]["DM_stat"]
            p_it  = r["iTransformer"]["DM_pval"]
            dm_pt = r["PatchTST"]["DM_stat"]
            p_pt  = r["PatchTST"]["DM_pval"]
            if p_it < 0.05 and dm_it < 0: wins_it += 1
            if p_pt < 0.05 and dm_pt < 0: wins_pt += 1
            print(f"{key:<12}  {dm_it:+7.3f}{sig(p_it):>5}       "
                  f"{dm_pt:+7.3f}{sig(p_pt):>5}")
    print("-" * 55)
    print(f"MSCA significant wins: iTransformer={wins_it}/15  PatchTST={wins_pt}/15")


if __name__ == "__main__":
    main()
