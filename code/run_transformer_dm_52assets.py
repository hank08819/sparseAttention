"""
run_transformer_dm_52assets.py  —  Transformer DM tests for all available pairs
================================================================================
Re-trains iTransformer, PatchTST, and Informer on every asset-period pair
that has data, computes Harvey-corrected DM statistics vs MSCA-SBiGRU.

Merges results into results/transformer_dm_52assets.json.
Existing entries in that file are skipped (safe to resume).

MSCA predictions are loaded from results/expanded_results.json  (via retraining
seed 42 for the same pair). DM is computed against the Transformer's seed-42
prediction on identical test windows.

Usage:
    cd code-data-pr+/code
    python examples/run_transformer_dm_52assets.py
"""

import json, os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msca_sparsity import MSCABiGRU, Trainer, build_features
from msca_sparsity.train import dm_test

DATA     = "../data"
DEVICE   = "cpu"
DATE_TAG = {"P1": "20221101", "P2": "20231001", "P3": "20240301"}

CATEGORIES = {
    "C1_BTC_family":    ["BTC", "LTC", "BCH", "DOGE", "SHIB", "XRP"],
    "C2_ETH_ecosystem": ["ETH", "MATIC", "ARB", "OP", "LRC"],
    "C3_Alt_L1":        ["SOL", "ADA", "AVAX", "DOT", "ATOM",
                         "NEAR", "ALGO", "ICP", "HBAR", "VET", "APT"],
    "C4_DeFi":          ["LINK", "AAVE", "UNI", "MKR", "CRV",
                         "COMP", "SNX", "GRT", "SUSHI", "YFI", "BAL", "1INCH"],
    "C5_Speculative":   ["EOS", "XTZ", "MANA", "SAND", "AXS", "APE",
                         "ENJ", "CHZ", "BAT", "ZRX", "ANKR", "OXT", "NMR"],
}

CA_MAP = {a: "BTC" for cat in CATEGORIES.values() for a in cat if a != "BTC"}
CA_MAP["BTC"] = "ETH"
ALL_ASSETS = [a for assets in CATEGORIES.values() for a in assets]

CFG = dict(
    window=30, horizon=5, hidden=64, n_heads=4, dropout=0.25,
    lr=5e-4, epochs=150, patience=20, warmup=20,
    batch_size=64, train_ratio=0.70, val_frac=0.10, n_features=7,
)

SEED = 42


# ── iTransformer (variate-level attention) ─────────────────────────────────────
class iTransformerBaseline(nn.Module):
    def __init__(self, feat_dim=7, seq_len=30, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.proj_in  = nn.Linear(seq_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2,
            dropout=dropout, batch_first=True)
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.proj_out = nn.Linear(d_model * feat_dim, 1)

    def forward(self, x_tgt, x_ca=None):
        # x_tgt: (B, T, F) → transpose to (B, F, T)
        x = x_tgt.transpose(1, 2)                   # (B, F, T)
        x = self.proj_in(x)                          # (B, F, d_model)
        x = self.encoder(x)                          # (B, F, d_model)
        x = x.reshape(x.size(0), -1)                # (B, F*d_model)
        pred = self.proj_out(x).squeeze(-1)
        return pred, None, torch.tensor(0., device=x_tgt.device)


# ── PatchTST (channel-independent patches) ────────────────────────────────────
class PatchTSTBaseline(nn.Module):
    def __init__(self, feat_dim=7, seq_len=30, patch_len=6, d_model=64,
                 n_heads=4, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        n_patches = seq_len // patch_len
        self.proj_in  = nn.Linear(patch_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2,
            dropout=dropout, batch_first=True)
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.proj_out = nn.Linear(d_model * n_patches * feat_dim, 1)
        self.n_patches = n_patches

    def forward(self, x_tgt, x_ca=None):
        B, T, F = x_tgt.shape
        # split into patches per channel
        x = x_tgt.reshape(B, self.n_patches, self.patch_len, F)  # (B,Np,PL,F)
        x = x.permute(0, 3, 1, 2).reshape(B * F, self.n_patches, self.patch_len)
        x = self.proj_in(x)             # (B*F, Np, d_model)
        x = self.encoder(x)             # (B*F, Np, d_model)
        x = x.reshape(B, F * self.n_patches * x.size(-1))
        pred = self.proj_out(x).squeeze(-1)
        return pred, None, torch.tensor(0., device=x_tgt.device)


# ── Informer (ProbSparse approximation via top-k attention) ───────────────────
class InformerBaseline(nn.Module):
    def __init__(self, feat_dim=7, seq_len=30, d_model=64, n_heads=4,
                 dropout=0.1, factor=3):
        super().__init__()
        self.proj_in   = nn.Linear(feat_dim, d_model)
        self.pos_emb   = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2,
            dropout=dropout, batch_first=True)
        self.encoder   = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.proj_out  = nn.Linear(d_model, 1)
        self.factor    = factor

    def forward(self, x_tgt, x_ca=None):
        x = self.proj_in(x_tgt) + self.pos_emb   # (B, T, d_model)
        # ProbSparse: keep top-k queries, zero out the rest
        B, T, D = x.shape
        k = max(1, int(self.factor * math.log(T)))
        scores = x.norm(dim=-1)                   # (B, T)
        topk_idx = scores.topk(k, dim=1).indices  # (B, k)
        mask = torch.zeros(B, T, device=x.device)
        mask.scatter_(1, topk_idx, 1.0)
        x = x * mask.unsqueeze(-1)
        x = self.encoder(x)
        pred = self.proj_out(x[:, -1, :]).squeeze(-1)
        return pred, None, torch.tensor(0., device=x_tgt.device)


TRANSFORMER_MODELS = {
    "iTransformer": iTransformerBaseline,
    "PatchTST":     PatchTSTBaseline,
    "Informer":     InformerBaseline,
}


# ── Data helpers (same as run_52assets.py) ─────────────────────────────────────
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
    _, feat_ca      = build_features(csv_path(ca, period),    CFG["n_features"])
    price_range = float(close.max() - close.min())
    X, Xca, y, pc = make_windows(close, feat_tgt, feat_ca, CFG["window"], CFG["horizon"])
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


def train_model(model_cls, splits, seed, model_kwargs=None):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_cls(feat_dim=splits["F"], **(model_kwargs or {}))
    trainer = Trainer(
        model, lr=CFG["lr"], epochs=CFG["epochs"],
        patience=CFG["patience"], warmup=CFG["warmup"],
        batch_size=CFG["batch_size"], device=DEVICE, verbose=False,
    )
    trainer.fit(*splits["tr"], *splits["val"])
    return trainer.predict(*splits["te"][:2])


def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def main():
    os.makedirs("results", exist_ok=True)

    out_path = "results/transformer_dm_52assets.json"
    # seed from original 15-pair results
    results = {}
    orig_path = "results/transformer_dm_results.json"
    if os.path.exists(orig_path):
        with open(orig_path) as f:
            results = json.load(f)
        print(f"Seeded {len(results)} original pairs", flush=True)
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        results.update(existing)
        print(f"Loaded {len(results)} total pairs", flush=True)

    wins   = {m: 0 for m in TRANSFORMER_MODELS}
    losses = {m: 0 for m in TRANSFORMER_MODELS}
    total  = 0

    for cat, assets in CATEGORIES.items():
        for asset in assets:
            for period in DATE_TAG:
                key = f"{asset}_{period}"
                if key in results:
                    # accumulate stats from cached
                    for m in TRANSFORMER_MODELS:
                        if m in results[key]:
                            p = results[key][m]["DM_pval"]
                            s = results[key][m]["DM_stat"]
                            if p < 0.05:
                                if s < 0: wins[m]  += 1
                                else:     losses[m]+= 1
                    total += 1
                    continue

                tgt_csv = csv_path(asset, period)
                ca_csv  = csv_path(CA_MAP.get(asset, "BTC"), period)
                if not os.path.exists(tgt_csv) or not os.path.exists(ca_csv):
                    continue

                print(f"\n[{cat}]  {asset} {period}", flush=True)
                t0 = time.time()
                try:
                    splits = prepare_splits(asset, period)
                except Exception as e:
                    print(f"  ERR data: {e}", flush=True); continue

                # train MSCA seed 42 to get predictions
                torch.manual_seed(SEED); np.random.seed(SEED)
                msca = MSCABiGRU(feat_dim=splits["F"], hidden=64,
                                 n_heads=4, dropout=0.25, lambda_gate=0.05)
                tr_msca = Trainer(msca, lr=CFG["lr"], epochs=CFG["epochs"],
                                  patience=CFG["patience"], warmup=CFG["warmup"],
                                  batch_size=CFG["batch_size"], device=DEVICE,
                                  verbose=False)
                tr_msca.fit(*splits["tr"], *splits["val"])
                pred_msca = tr_msca.predict(*splits["te"][:2])
                y_te  = splits["te"][2]
                e_msca = y_te - pred_msca
                msca_nmse = tr_msca.evaluate(
                    *splits["te"], splits["pte"],
                    price_range=splits["price_range"])["NMSE"]

                row = {"MSCA_NMSE_s42": float(msca_nmse)}

                for mname, mcls in TRANSFORMER_MODELS.items():
                    try:
                        pred_t = train_model(mcls, splits, SEED)
                        e_t = y_te - pred_t
                        dm_stat, dm_pval = dm_test(e_msca, e_t, horizon=5)
                        row[mname] = {
                            "DM_stat": float(dm_stat),
                            "DM_pval": float(dm_pval),
                        }
                        flag = ""
                        if dm_pval < 0.05:
                            if dm_stat < 0: wins[mname]  += 1; flag = "WIN"
                            else:           losses[mname]+= 1; flag = "LOSS"
                        print(f"  vs {mname:15s}: DM={dm_stat:+.3f}{sig(dm_pval):3s} {flag}",
                              flush=True)
                    except Exception as e:
                        print(f"  ERR {mname}: {e}", flush=True)

                results[key] = row
                total += 1
                print(f"  ({time.time()-t0:.0f}s)", flush=True)

                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Total pairs: {total}")
    print(f"MSCA wins  : {wins}  (vs {[m for m in TRANSFORMER_MODELS]})")
    print(f"MSCA losses: {losses}")
    tw = sum(wins.values()); tl = sum(losses.values())
    print(f"Grand total: {tw}/{total*len(TRANSFORMER_MODELS)} wins, {tl} losses")


if __name__ == "__main__":
    main()
