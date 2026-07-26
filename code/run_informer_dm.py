"""
run_informer_dm.py
==================
Train Informer (Zhou et al. 2021) on all 15 asset-period pairs (seeds 42/43/44),
compute DM tests against MSCA-SBiGRU (seed 42), and merge into
results/transformer_dm_results.json alongside iTransformer and PatchTST.

Can run concurrently with or after run_transformer_dm.py.
If transformer_dm_results.json already exists, Informer results are merged in.

Runtime: ~8-12 min on CPU.

Usage:
    cd code-data-pr+/code
    python examples/run_informer_dm.py
"""

import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msca_sparsity import MSCABiGRU, Trainer, build_features
from msca_sparsity.train import dm_test

# ── Config (identical to run_transformer_dm.py) ───────────────────────────────
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


# ── Informer ──────────────────────────────────────────────────────────────────
class Informer(nn.Module):
    """
    Zhou et al. (2021) — Transformer with distilling encoder layers.
    Uses standard self-attention + MaxPool1d distilling (halves sequence length
    between layers), which is the key structural innovation of Informer.
    """
    def __init__(self, feat_dim=7, seq_len=30, d_model=64, n_heads=4,
                 n_layers=2, d_ff=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        # one distilling conv per inter-layer gap
        self.distill = nn.ModuleList([
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            for _ in range(n_layers - 1)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # compute output length after n_layers-1 distilling ops
        out_len = seq_len
        for _ in range(n_layers - 1):
            # MaxPool1d: floor((L + 2*pad - kernel) / stride) + 1
            out_len = (out_len + 2 * 1 - 3) // 2 + 1
        self.head = nn.Linear(out_len * d_model, 1)

    def forward(self, x_tgt, x_ca=None):
        # x_tgt: (B, T, F)
        x = self.input_proj(x_tgt)          # (B, T, d_model)
        for i, layer in enumerate(self.layers):
            x = layer(x)                    # (B, T', d_model)
            if i < len(self.distill):
                x = x.permute(0, 2, 1)     # (B, d_model, T')
                x = self.distill[i](x)     # (B, d_model, T'//2)
                x = x.permute(0, 2, 1)     # (B, T'//2, d_model)
        x = self.norm(x)
        x = self.drop(x.reshape(x.shape[0], -1))  # (B, T_out*d_model)
        pred = self.head(x).squeeze(-1)     # (B,)
        return pred, None, torch.tensor(0.0, device=x_tgt.device)


# ── Data helpers ──────────────────────────────────────────────────────────────
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
    out_path = "results/transformer_dm_results.json"
    t0 = time.time()

    # load existing results if present (from run_transformer_dm.py)
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"Loaded existing {out_path} — will add Informer results.", flush=True)
    else:
        results = {}
        print("No existing transformer_dm_results.json — creating fresh.", flush=True)

    def sig(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    for asset in ASSETS:
        for period in PERIODS:
            key = f"{asset}_{period}"
            print(f"\n[{asset} {period}]", flush=True)
            splits = prepare_splits(asset, period)
            F = splits["F"]

            # ── MSCA seed-42 predictions (reference for DM test) ──────────
            nmse_msca, pred_msca = train_and_predict(
                MSCABiGRU, splits, 42,
                dict(feat_dim=F, hidden=64, n_heads=4,
                     dropout=0.25, lambda_gate=0.05))
            y_te   = splits["te"][2]
            e_msca = y_te - pred_msca
            print(f"  MSCA  s42 NMSE={nmse_msca:.4f}", flush=True)

            # ── Informer (3 seeds for NMSE mean±std; DM from seed 42) ─────
            inf_nmse = []
            for seed in SEEDS:
                nmse_inf, pred_inf = train_and_predict(
                    Informer, splits, seed,
                    dict(feat_dim=F, seq_len=CFG["window"],
                         d_model=64, n_heads=4, n_layers=2,
                         d_ff=128, dropout=0.1))
                inf_nmse.append(nmse_inf)
                if seed == 42:
                    pred_inf42 = pred_inf
                print(f"  Informer  s{seed} NMSE={nmse_inf:.4f}", flush=True)

            e_inf = y_te - pred_inf42
            dm_inf, p_inf = dm_test(e_msca, e_inf, horizon=5)
            print(f"  DM(MSCA vs Informer)={dm_inf:+.3f}{sig(p_inf)}", flush=True)

            # ── Merge into results dict ────────────────────────────────────
            if key not in results:
                results[key] = {"MSCA_NMSE_s42": float(nmse_msca)}

            results[key]["Informer"] = {
                "NMSE_mean":  float(np.mean(inf_nmse)),
                "NMSE_std":   float(np.std(inf_nmse, ddof=1)),
                "NMSE_seeds": [float(v) for v in inf_nmse],
                "DM_stat":    float(dm_inf),
                "DM_pval":    float(p_inf),
            }

            # save incrementally so progress is not lost
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nInformer results merged → {out_path}", flush=True)
    print(f"Total time: {elapsed/60:.1f} min", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== DM Summary: MSCA vs Informer (seed-42) ===")
    print(f"{'Pair':<12}  {'Informer NMSE':>16}  {'DM stat':>10}")
    print("-" * 45)
    wins = 0
    for asset in ASSETS:
        for period in PERIODS:
            key = f"{asset}_{period}"
            r = results[key].get("Informer", {})
            if not r:
                print(f"{key:<12}  (missing)", flush=True)
                continue
            dm  = r["DM_stat"]
            p   = r["DM_pval"]
            mn  = r["NMSE_mean"]
            std = r["NMSE_std"]
            if p < 0.05 and dm < 0:
                wins += 1
            print(f"{key:<12}  {mn:.4f}±{std:.4f}      {dm:+7.3f}{sig(p):>5}")
    print("-" * 45)
    print(f"MSCA significant wins vs Informer: {wins}/15")


if __name__ == "__main__":
    main()
