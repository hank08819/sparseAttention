"""
run_extended_baselines.py — Extended Transformer baselines: TFT, N-BEATS, TimesNet
====================================================================================
Adds TFT, N-BEATS, TimesNet to the existing iTransformer/PatchTST/Informer
comparison.  Merges results into results/extended_baseline_results.json.
Seeds from transformer_dm_results.json (15-pair original results).

Usage:
    cd code-data-pr+/code
    python examples/run_extended_baselines.py
"""

import json, os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msca_sparsity import MSCABiGRU, Trainer, build_features
from msca_sparsity.train import dm_test

DATA    = "../data"
DEVICE  = "cpu"
SEED    = 42

CFG = dict(
    window=30, horizon=5, hidden=64, n_heads=4, dropout=0.25,
    lr=5e-4, epochs=150, patience=20, warmup=20,
    batch_size=64, train_ratio=0.70, val_frac=0.10, n_features=7,
)

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


# ── Existing baselines (from run_transformer_dm_52assets.py) ─────────────────

class iTransformerBaseline(nn.Module):
    def __init__(self, feat_dim=7, seq_len=30, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.proj_in  = nn.Linear(seq_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*2, dropout=dropout, batch_first=True)
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.proj_out = nn.Linear(d_model * feat_dim, 1)

    def forward(self, x_tgt, x_ca=None):
        x = x_tgt.transpose(1, 2)
        x = self.proj_in(x)
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)
        return self.proj_out(x).squeeze(-1), None, torch.tensor(0., device=x_tgt.device)


class PatchTSTBaseline(nn.Module):
    def __init__(self, feat_dim=7, seq_len=30, patch_len=6, d_model=64,
                 n_heads=4, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        n_patches = seq_len // patch_len
        self.proj_in  = nn.Linear(patch_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*2, dropout=dropout, batch_first=True)
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.proj_out = nn.Linear(d_model * n_patches * feat_dim, 1)
        self.n_patches = n_patches

    def forward(self, x_tgt, x_ca=None):
        B, T, F = x_tgt.shape
        x = x_tgt.reshape(B, self.n_patches, self.patch_len, F)
        x = x.permute(0, 3, 1, 2).reshape(B * F, self.n_patches, self.patch_len)
        x = self.proj_in(x)
        x = self.encoder(x)
        x = x.reshape(B, F * self.n_patches * x.size(-1))
        return self.proj_out(x).squeeze(-1), None, torch.tensor(0., device=x_tgt.device)


class InformerBaseline(nn.Module):
    def __init__(self, feat_dim=7, seq_len=30, d_model=64, n_heads=4,
                 dropout=0.1, factor=3):
        super().__init__()
        self.proj_in  = nn.Linear(feat_dim, d_model)
        self.pos_emb  = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*2, dropout=dropout, batch_first=True)
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.proj_out = nn.Linear(d_model, 1)
        self.factor   = factor

    def forward(self, x_tgt, x_ca=None):
        x = self.proj_in(x_tgt) + self.pos_emb
        B, T, D = x.shape
        k = max(1, int(self.factor * math.log(T)))
        scores = x.norm(dim=-1)
        topk_idx = scores.topk(k, dim=1).indices
        mask = torch.zeros(B, T, device=x.device)
        mask.scatter_(1, topk_idx, 1.0)
        x = x * mask.unsqueeze(-1)
        x = self.encoder(x)
        return self.proj_out(x[:, -1, :]).squeeze(-1), None, torch.tensor(0., device=x_tgt.device)


# ── TFT: Temporal Fusion Transformer (simplified, no future covariates) ───────

class _GRN(nn.Module):
    """Gated Residual Network block."""
    def __init__(self, d):
        super().__init__()
        self.fc1  = nn.Linear(d, d)
        self.fc2  = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.ln   = nn.LayerNorm(d)

    def forward(self, x):
        h  = F.elu(self.fc1(x))
        h  = self.fc2(h)
        g  = torch.sigmoid(self.gate(x))
        return self.ln(x + g * h)


class TFTBaseline(nn.Module):
    """Simplified TFT without future inputs or static covariates."""
    def __init__(self, feat_dim=7, seq_len=30, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos_emb    = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.grn_in     = _GRN(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.attn   = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.grn_out = _GRN(d_model)
        self.proj_out = nn.Linear(d_model, 1)
        self.drop     = nn.Dropout(dropout)

    def forward(self, x_tgt, x_ca=None):
        x = self.input_proj(x_tgt) + self.pos_emb
        x = self.grn_in(x)
        x = self.drop(x)
        x = self.attn(x)
        x = self.grn_out(x)
        out = self.proj_out(x[:, -1, :]).squeeze(-1)
        return out, None, torch.tensor(0., device=x_tgt.device)


# ── N-BEATS (generic stack, no trend/seasonality decomposition) ───────────────

class _NBEATSBlock(nn.Module):
    def __init__(self, seq_len, d=256, n_layers=4):
        super().__init__()
        layers = [nn.Linear(seq_len, d), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, d), nn.ReLU()]
        self.fc       = nn.Sequential(*layers)
        self.backcast = nn.Linear(d, seq_len)
        self.forecast = nn.Linear(d, 1)

    def forward(self, x):
        h = self.fc(x)
        return self.backcast(h), self.forecast(h)


class NBEATSBaseline(nn.Module):
    """N-BEATS generic configuration: 4 stacks of 4 blocks each."""
    def __init__(self, feat_dim=7, seq_len=30, d=128, n_stacks=4, n_blocks=4,
                 dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.seq_len  = seq_len
        self.input_len = seq_len * feat_dim
        self.stacks = nn.ModuleList([
            nn.ModuleList([_NBEATSBlock(self.input_len, d) for _ in range(n_blocks)])
            for _ in range(n_stacks)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, x_tgt, x_ca=None):
        B = x_tgt.size(0)
        residual = x_tgt.reshape(B, -1).float()
        forecast  = torch.zeros(B, device=x_tgt.device)
        for stack in self.stacks:
            for block in stack:
                backcast, fc = block(residual)
                residual = residual - backcast
                forecast = forecast + fc.squeeze(-1)
        return forecast, None, torch.tensor(0., device=x_tgt.device)


# ── TimesNet (FFT-based 2D convolution) ───────────────────────────────────────

class _Inception2D(nn.Module):
    """Simplified Inception block operating on (B, C, H, W)."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c_in, c_out // 2, kernel_size=5, padding=2)
        self.bn    = nn.BatchNorm2d(c_out)
        self.act   = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([self.conv1(x), self.conv2(x)], dim=1)))


class TimesNetBaseline(nn.Module):
    """TimesNet: 1D→2D via FFT top-k period detection, 2D Inception conv."""
    def __init__(self, feat_dim=7, seq_len=30, d_model=64, top_k=3, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.top_k   = top_k
        self.proj_in  = nn.Linear(feat_dim, d_model)
        self.inception = _Inception2D(d_model, d_model)
        self.proj_out = nn.Linear(d_model, 1)
        self.drop     = nn.Dropout(dropout)
        self.ln       = nn.LayerNorm(d_model)

    def forward(self, x_tgt, x_ca=None):
        B, T, _ = x_tgt.shape
        x = self.proj_in(x_tgt)        # (B, T, d_model)
        xf = torch.fft.rfft(x, dim=1)  # (B, T//2+1, d_model)
        amp = xf.abs().mean(dim=-1)    # (B, freq)
        _, topk_idx = amp.topk(self.top_k, dim=1)

        out = torch.zeros_like(x)
        for i in range(self.top_k):
            period = torch.clamp(T // (topk_idx[:, i] + 1), min=2, max=T)
            p_val  = int(period.float().mean().item())
            n_rep  = math.ceil(T / p_val)
            padded = F.pad(x.permute(0, 2, 1), (0, n_rep * p_val - T))  # (B, d, padded)
            x2d    = padded.reshape(B, x.size(-1), n_rep, p_val)          # (B, d, nrep, p)
            x2d    = self.inception(x2d)                                   # (B, d, nrep, p)
            flat   = x2d.reshape(B, x2d.size(1), -1)[:, :, :T]           # (B, d, T)
            out    = out + flat.permute(0, 2, 1)                           # (B, T, d)

        out = self.ln(out / self.top_k)
        pred = self.proj_out(self.drop(out[:, -1, :])).squeeze(-1)
        return pred, None, torch.tensor(0., device=x_tgt.device)


ALL_BASELINES = {
    "iTransformer": iTransformerBaseline,
    "PatchTST":     PatchTSTBaseline,
    "Informer":     InformerBaseline,
    "TFT":          TFTBaseline,
    "N-BEATS":      NBEATSBaseline,
    "TimesNet":     TimesNetBaseline,
}


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
    _, feat_ca      = build_features(csv_path(ca, period),    CFG["n_features"])
    price_range = float(close.max() - close.min())
    X, Xca, y, pc = make_windows(close, feat_tgt, feat_ca, CFG["window"], CFG["horizon"])
    n  = len(y)
    nt = int(n * CFG["train_ratio"])
    nv = int(n * CFG["val_frac"])
    F  = X.shape[-1]
    sc = MinMaxScaler(); sca = MinMaxScaler()
    sc.fit(X[:nt].reshape(-1, F)); sca.fit(Xca[:nt].reshape(-1, F))
    X   = sc.transform(X.reshape(-1, F)).reshape(X.shape)
    Xca = sca.transform(Xca.reshape(-1, F)).reshape(Xca.shape)
    return dict(
        tr=(X[:nt], Xca[:nt], y[:nt]),
        val=(X[nt:nt+nv], Xca[nt:nt+nv], y[nt:nt+nv]),
        te=(X[nt+nv:], Xca[nt+nv:], y[nt+nv:]),
        pte=pc[nt+nv:], price_range=price_range, F=F,
    )


def train_baseline(model_cls, splits, seed=42, model_kwargs=None):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_cls(feat_dim=splits["F"], **(model_kwargs or {}))
    trainer = Trainer(model, lr=CFG["lr"], epochs=CFG["epochs"],
                      patience=CFG["patience"], warmup=CFG["warmup"],
                      batch_size=CFG["batch_size"], device=DEVICE, verbose=False)
    trainer.fit(*splits["tr"], *splits["val"])
    return trainer.predict(*splits["te"][:2])


def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def main():
    os.makedirs("results", exist_ok=True)
    out_path = "results/extended_baseline_results.json"

    results = {}
    # seed from original 15-pair Transformer DM results
    orig_path = "results/transformer_dm_results.json"
    if os.path.exists(orig_path):
        with open(orig_path) as f:
            seed_data = json.load(f)
        for k, v in seed_data.items():
            # only carry over the 3 original baselines
            results[k] = {b: v[b] for b in ["iTransformer", "PatchTST", "Informer"]
                          if b in v}
            if "MSCA_NMSE_s42" in v:
                results[k]["MSCA_NMSE_s42"] = v["MSCA_NMSE_s42"]
        print(f"Seeded {len(results)} original pairs", flush=True)

    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        for k, v in existing.items():
            if k not in results:
                results[k] = v
            else:
                results[k].update(v)
        print(f"Merged → {len(results)} total pairs", flush=True)

    wins   = {m: 0 for m in ALL_BASELINES}
    losses = {m: 0 for m in ALL_BASELINES}
    total  = 0

    for cat, assets in CATEGORIES.items():
        for asset in assets:
            for period in DATE_TAG:
                key = f"{asset}_{period}"
                tgt_csv = csv_path(asset, period)
                ca_csv  = csv_path(CA_MAP.get(asset, "BTC"), period)
                if not os.path.exists(tgt_csv) or not os.path.exists(ca_csv):
                    continue

                # check which baselines still need running
                needed = [m for m in ALL_BASELINES
                          if key not in results or m not in results.get(key, {})]
                if not needed:
                    total += 1
                    continue

                print(f"\n[{cat}]  {asset} {period}", flush=True)
                t0 = time.time()
                try:
                    splits = prepare_splits(asset, period)
                except Exception as e:
                    print(f"  ERR data: {e}", flush=True); continue

                # train MSCA seed 42 for DM reference
                if key not in results or "MSCA_NMSE_s42" not in results.get(key, {}):
                    torch.manual_seed(SEED); np.random.seed(SEED)
                    msca = MSCABiGRU(feat_dim=splits["F"], hidden=64,
                                     n_heads=4, dropout=0.25, lambda_gate=0.05)
                    tr_msca = Trainer(msca, lr=CFG["lr"], epochs=CFG["epochs"],
                                      patience=CFG["patience"], warmup=CFG["warmup"],
                                      batch_size=CFG["batch_size"], device=DEVICE,
                                      verbose=False)
                    tr_msca.fit(*splits["tr"], *splits["val"])
                    pred_msca = tr_msca.predict(*splits["te"][:2])
                    msca_nmse = tr_msca.evaluate(*splits["te"], splits["pte"],
                                                 price_range=splits["price_range"])["NMSE"]
                    if key not in results:
                        results[key] = {}
                    results[key]["MSCA_NMSE_s42"] = float(msca_nmse)
                else:
                    # reconstruct MSCA predictions from scratch (needed for DM)
                    torch.manual_seed(SEED); np.random.seed(SEED)
                    msca = MSCABiGRU(feat_dim=splits["F"], hidden=64,
                                     n_heads=4, dropout=0.25, lambda_gate=0.05)
                    tr_msca = Trainer(msca, lr=CFG["lr"], epochs=CFG["epochs"],
                                      patience=CFG["patience"], warmup=CFG["warmup"],
                                      batch_size=CFG["batch_size"], device=DEVICE,
                                      verbose=False)
                    tr_msca.fit(*splits["tr"], *splits["val"])
                    pred_msca = tr_msca.predict(*splits["te"][:2])

                y_te   = splits["te"][2]
                e_msca = y_te - pred_msca

                for mname in needed:
                    mcls = ALL_BASELINES[mname]
                    try:
                        pred_t = train_baseline(mcls, splits, SEED)
                        e_t    = y_te - pred_t
                        dm_stat, dm_pval = dm_test(e_msca, e_t, horizon=5)
                        results[key][mname] = {
                            "DM_stat": float(dm_stat),
                            "DM_pval": float(dm_pval),
                        }
                        flag = ""
                        if dm_pval < 0.05:
                            if dm_stat < 0: wins[mname]  += 1; flag = "WIN"
                            else:           losses[mname] += 1; flag = "LOSS"
                        print(f"  vs {mname:15s}: DM={dm_stat:+.3f}{sig(dm_pval):3s} {flag}",
                              flush=True)
                    except Exception as e:
                        print(f"  ERR {mname}: {e}", flush=True)

                total += 1
                print(f"  ({time.time()-t0:.0f}s)", flush=True)
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Total pairs: {total}")
    n_baselines = len(ALL_BASELINES)
    tw = sum(wins.values()); tl = sum(losses.values())
    for m in ALL_BASELINES:
        print(f"  {m:15s}: {wins[m]:3d} wins  {losses[m]:3d} losses")
    print(f"Grand total: {tw}/{total * n_baselines} wins, {tl} losses")


if __name__ == "__main__":
    main()
