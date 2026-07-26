"""
run_soladada_baselines.py
=========================
Train MSCA-SBiGRU, LSTM, and GRU on SOL and ADA (P1/P2/P3, seeds 42/43/44),
compute DM tests (MSCA vs LSTM, MSCA vs GRU), and merge results into
results/multiseed_results.json so the DiD can be expanded to 6 altcoins.

This enables:
  - Treated: ETH, XRP  (BTC-correlation Δρ ≈ +0.30 post-ETF)
  - Control: LTC, DOGE, SOL, ADA  (Δρ ≈ +0.06–+0.09 or negative)
  - C(6,2)=15 permutations → one-sided DiD p = 1/15 ≈ 0.067

Runtime: ~50-70 min on CPU.

Usage:
    cd code-data-pr+/code
    python examples/run_soladada_baselines.py
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
# SOL and ADA both use BTC as cross-asset (same as XRP/LTC/DOGE)
CA_MAP  = {"SOL": "BTC", "ADA": "BTC"}
DATE_TAG= {"P1":"20221101","P2":"20231001","P3":"20240301"}
ASSETS  = ["SOL", "ADA"]
PERIODS = ["P1","P2","P3"]
RESULTS_PATH = "results/multiseed_results.json"


# ── Baseline models (identical to run_multiseed.py) ───────────────────────────
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


# ── Data helpers ───────────────────────────────────────────────────────────────
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("results", exist_ok=True)
    t0 = time.time()

    # load existing multiseed_results.json
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        print(f"Loaded {RESULTS_PATH} ({len(results)} existing entries)", flush=True)
    else:
        results = {}
        print("No existing multiseed_results.json — creating fresh.", flush=True)

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

            nmse_by_model = {"MSCA": [], "LSTM": [], "GRU": []}
            pred42 = {}

            for seed in SEEDS:
                label = f"s{seed}"

                nmse_m, pred_m = train_and_eval(
                    MSCABiGRU, splits, seed,
                    dict(hidden=64, n_heads=4, dropout=0.25, lambda_gate=0.05))
                nmse_by_model["MSCA"].append(nmse_m)
                if seed == 42:
                    pred42["MSCA"] = pred_m

                nmse_l, pred_l = train_and_eval(
                    LSTMBaseline, splits, seed, dict(hidden=64, dropout=0.20))
                nmse_by_model["LSTM"].append(nmse_l)
                if seed == 42:
                    pred42["LSTM"] = pred_l

                nmse_g, pred_g = train_and_eval(
                    GRUBaseline, splits, seed, dict(hidden=64, dropout=0.20))
                nmse_by_model["GRU"].append(nmse_g)
                if seed == 42:
                    pred42["GRU"] = pred_g

                print(f"  s{seed}: MSCA={nmse_m:.4f}  LSTM={nmse_l:.4f}  "
                      f"GRU={nmse_g:.4f}", flush=True)

            y_te   = splits["te"][2]
            e_msca = y_te - pred42["MSCA"]
            e_lstm = y_te - pred42["LSTM"]
            e_gru  = y_te - pred42["GRU"]

            dm_vs_lstm, p_vs_lstm = dm_test(e_msca, e_lstm, horizon=5)
            dm_vs_gru,  p_vs_gru  = dm_test(e_msca, e_gru,  horizon=5)

            print(f"  DM(MSCA vs LSTM)={dm_vs_lstm:+.3f}{sig(p_vs_lstm)}  "
                  f"DM(MSCA vs GRU)={dm_vs_gru:+.3f}{sig(p_vs_gru)}", flush=True)

            cell = {}
            for m in ["MSCA", "LSTM", "GRU"]:
                vals = nmse_by_model[m]
                cell[m] = {
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals, ddof=1)),
                    "s42":  float(vals[0]),
                    "s43":  float(vals[1]),
                    "s44":  float(vals[2]),
                }
            cell["DM_vs_LSTM"] = {"stat": float(dm_vs_lstm), "pval": float(p_vs_lstm)}
            cell["DM_vs_GRU"]  = {"stat": float(dm_vs_gru),  "pval": float(p_vs_gru)}

            results[key] = cell

            # save incrementally
            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nAll done in {elapsed/60:.1f} min", flush=True)
    print(f"Results merged → {RESULTS_PATH}", flush=True)

    # ── DiD computation (6-altcoin expanded design) ───────────────────────────
    print("\n=== Expanded DiD (6 altcoins, C(6,2)=15 permutations) ===")
    print("Treated: ETH, XRP  |  Control: LTC, DOGE, SOL, ADA")
    print("Outcome: DM(MSCA vs LSTM) statistic  |  Pre=P2, Post=P3\n")

    all_assets = ["ETH", "XRP", "LTC", "DOGE", "SOL", "ADA"]
    treated    = {"ETH", "XRP"}

    def dm_stat(asset, period, vs="LSTM"):
        k = f"{asset}_{period}"
        key_dm = f"DM_vs_{vs}"
        return results[k][key_dm]["stat"]

    # observed DiD
    did_obs = 0.0
    for a in all_assets:
        delta = dm_stat(a, "P3") - dm_stat(a, "P2")
        sign  = -1 if a in treated else +1
        did_obs += sign * delta
    did_obs /= (len(treated) + (len(all_assets) - len(treated)))

    # permutation test: all C(6,2)=15 ways to choose 2 "treated" assets
    from itertools import combinations
    perm_dids = []
    for perm_treated in combinations(all_assets, 2):
        perm_treated = set(perm_treated)
        did_perm = 0.0
        for a in all_assets:
            delta = dm_stat(a, "P3") - dm_stat(a, "P2")
            sign  = -1 if a in perm_treated else +1
            did_perm += sign * delta
        did_perm /= len(all_assets)
        perm_dids.append(did_perm)

    p_perm = sum(1 for d in perm_dids if d <= did_obs) / len(perm_dids)

    print(f"Observed DiD = {did_obs:.3f}")
    print(f"Permutation distribution ({len(perm_dids)} arrangements):")
    print(f"  min={min(perm_dids):.3f}  max={max(perm_dids):.3f}")
    print(f"One-sided p = {p_perm:.4f}  ({sum(1 for d in perm_dids if d <= did_obs)}/{len(perm_dids)})")

    print("\nPer-asset DiD contributions (ΔDM = P3 − P2, MSCA vs LSTM):")
    for a in all_assets:
        d2 = dm_stat(a, "P2"); d3 = dm_stat(a, "P3")
        role = "TREATED" if a in treated else "control"
        print(f"  {a:5s} [{role}]: P2={d2:+.3f}  P3={d3:+.3f}  Δ={d3-d2:+.3f}")

    # save DiD summary alongside results
    did_summary = {
        "design": "6-altcoin DiD: treated=ETH,XRP; control=LTC,DOGE,SOL,ADA",
        "outcome": "DM(MSCA vs LSTM)",
        "pre": "P2", "post": "P3",
        "observed_DiD": float(did_obs),
        "n_permutations": len(perm_dids),
        "one_sided_p": float(p_perm),
    }
    did_path = "results/did_6asset_results.json"
    with open(did_path, "w") as f:
        json.dump(did_summary, f, indent=2)
    print(f"\nDiD summary saved → {did_path}", flush=True)


if __name__ == "__main__":
    main()
