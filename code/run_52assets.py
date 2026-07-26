"""
run_52assets.py  —  Full 52-asset × 3-period × 3-seed experiment
=================================================================
Trains MSCA-SBiGRU, LSTM, GRU on every available asset-period pair.
Cross-asset: BTC for all non-BTC assets; ETH for BTC itself.

Loads existing results from multiseed_results.json and skips already-computed
keys, so the script is safe to resume after interruption.

Outputs
-------
results/expanded_results.json      — per-pair NMSE + DM stats
results/category_summary.json      — win-rate / DM by category & period
results/did_52asset_results.json   — multi-category DiD permutation test

Usage
-----
    cd code-data-pr+/code
    python examples/run_52assets.py
"""

import json, os, sys, time
from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msca_sparsity import MSCABiGRU, Trainer, build_features
from msca_sparsity.train import dm_test

# ── Config ─────────────────────────────────────────────────────────────────────
DATA  = "../data"
SEEDS = [42, 43, 44]
DEVICE = "cpu"

CFG = dict(
    window=30, horizon=5, hidden=64, n_heads=4, dropout=0.25,
    lr=5e-4, epochs=150, patience=20, warmup=20,
    batch_size=64, train_ratio=0.70, val_frac=0.10, n_features=7,
)

DATE_TAG = {"P1": "20221101", "P2": "20231001", "P3": "20240301"}

# ── 5-category asset map ───────────────────────────────────────────────────────
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

ASSET_TO_CAT = {a: cat for cat, assets in CATEGORIES.items() for a in assets}
ALL_ASSETS   = [a for assets in CATEGORIES.values() for a in assets]

# Cross-asset leader: BTC for all non-BTC; ETH for BTC
CA_MAP = {a: "BTC" for a in ALL_ASSETS if a != "BTC"}
CA_MAP["BTC"] = "ETH"


# ── Baselines ──────────────────────────────────────────────────────────────────
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
    _, feat_ca      = build_features(csv_path(ca, period),    CFG["n_features"])
    price_range = float(close.max() - close.min())
    if price_range < 1e-12:
        raise ValueError(f"Zero price range for {asset} {period}")
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


def train_and_eval(model_cls, splits, seed, model_kwargs=None):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_cls(feat_dim=splits["F"], **(model_kwargs or {}))
    trainer = Trainer(
        model, lr=CFG["lr"], epochs=CFG["epochs"],
        patience=CFG["patience"], warmup=CFG["warmup"],
        batch_size=CFG["batch_size"], device=DEVICE, verbose=False,
    )
    trainer.fit(*splits["tr"], *splits["val"])
    metrics = trainer.evaluate(*splits["te"], splits["pte"],
                               price_range=splits["price_range"])
    pred = trainer.predict(*splits["te"][:2])
    return metrics["NMSE"], pred


def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("results", exist_ok=True)

    # ── load existing results ─────────────────────────────────────────────────
    EXPANDED_PATH = "results/expanded_results.json"
    LEGACY_PATH   = "results/multiseed_results.json"

    results = {}
    if os.path.exists(EXPANDED_PATH):
        with open(EXPANDED_PATH) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing pairs from {EXPANDED_PATH}", flush=True)

    # seed existing multiseed_results (BTC/ETH/XRP/LTC/DOGE/SOL/ADA)
    if os.path.exists(LEGACY_PATH):
        with open(LEGACY_PATH) as f:
            legacy = json.load(f)
        for k, v in legacy.items():
            if "_" in k and not k.startswith("LNN_") and k not in results:
                results[k] = v
        print(f"Seeded {len(results)} pairs total (including legacy)", flush=True)

    periods = list(DATE_TAG.keys())
    n_total = sum(1 for a in ALL_ASSETS for p in periods
                  if os.path.exists(csv_path(a, p))
                  and os.path.exists(csv_path(CA_MAP[a], p)))
    done_count = 0
    skip_count = 0

    t_global = time.time()

    # ── training loop ─────────────────────────────────────────────────────────
    for cat, assets in CATEGORIES.items():
        for asset in assets:
            for period in periods:
                key = f"{asset}_{period}"
                tgt_csv = csv_path(asset, period)
                ca_csv  = csv_path(CA_MAP[asset], period)

                if not os.path.exists(tgt_csv):
                    print(f"  SKIP  {key}: no data file", flush=True)
                    continue
                if not os.path.exists(ca_csv):
                    print(f"  SKIP  {key}: no cross-asset file for {CA_MAP[asset]}", flush=True)
                    continue

                if key in results:
                    skip_count += 1
                    print(f"  SKIP  {key} (cached)  [{skip_count}+{done_count}/{n_total}]",
                          flush=True)
                    continue

                print(f"\n[{cat}]  {asset} {period}", flush=True)
                t0 = time.time()

                try:
                    splits = prepare_splits(asset, period)
                except Exception as e:
                    print(f"  ERR  data prep: {e}", flush=True)
                    continue

                nmse_by_model = {"MSCA": [], "LSTM": [], "GRU": []}
                pred42 = {}

                for seed in SEEDS:
                    nm, pm = train_and_eval(
                        MSCABiGRU, splits, seed,
                        dict(hidden=64, n_heads=4, dropout=0.25, lambda_gate=0.05))
                    nmse_by_model["MSCA"].append(nm)
                    if seed == 42: pred42["MSCA"] = pm

                    nl, pl = train_and_eval(
                        LSTMBaseline, splits, seed,
                        dict(hidden=64, dropout=0.20))
                    nmse_by_model["LSTM"].append(nl)
                    if seed == 42: pred42["LSTM"] = pl

                    ng, pg = train_and_eval(
                        GRUBaseline, splits, seed,
                        dict(hidden=64, dropout=0.20))
                    nmse_by_model["GRU"].append(ng)
                    if seed == 42: pred42["GRU"] = pg

                    print(f"  s{seed}: MSCA={nm:.4f}  LSTM={nl:.4f}  GRU={ng:.4f}",
                          flush=True)

                y_te   = splits["te"][2]
                e_msca = y_te - pred42["MSCA"]
                e_lstm = y_te - pred42["LSTM"]
                e_gru  = y_te - pred42["GRU"]

                dm_vs_lstm, p_vs_lstm = dm_test(e_msca, e_lstm, horizon=5)
                dm_vs_gru,  p_vs_gru  = dm_test(e_msca, e_gru,  horizon=5)

                print(f"  DM(MSCA/LSTM)={dm_vs_lstm:+.3f}{sig(p_vs_lstm)}  "
                      f"DM(MSCA/GRU)={dm_vs_gru:+.3f}{sig(p_vs_gru)}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

                cell = {"category": cat}
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
                done_count += 1

                with open(EXPANDED_PATH, "w") as f:
                    json.dump(results, f, indent=2)

    # ── category-level summary ────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("CATEGORY SUMMARY")
    print("=" * 68)

    cat_summary = {}

    for cat in CATEGORIES:
        wins_lstm = 0; losses_lstm = 0; total_lstm = 0
        wins_gru  = 0; losses_gru  = 0; total_gru  = 0
        dm_lstm_vals = []; dm_gru_vals = []

        for asset in CATEGORIES[cat]:
            for period in periods:
                key = f"{asset}_{period}"
                if key not in results: continue
                row = results[key]

                dl = row["DM_vs_LSTM"]; dg = row["DM_vs_GRU"]
                dm_lstm_vals.append(dl["stat"]); dm_gru_vals.append(dg["stat"])
                total_lstm += 1; total_gru += 1

                if dl["pval"] < 0.05 and dl["stat"] < 0: wins_lstm  += 1
                if dl["pval"] < 0.05 and dl["stat"] > 0: losses_lstm+= 1
                if dg["pval"] < 0.05 and dg["stat"] < 0: wins_gru   += 1
                if dg["pval"] < 0.05 and dg["stat"] > 0: losses_gru += 1

        cat_summary[cat] = {
            "vs_LSTM": {"wins": wins_lstm, "losses": losses_lstm, "total": total_lstm,
                        "mean_DM": float(np.mean(dm_lstm_vals)) if dm_lstm_vals else None},
            "vs_GRU":  {"wins": wins_gru,  "losses": losses_gru,  "total": total_gru,
                        "mean_DM": float(np.mean(dm_gru_vals))  if dm_gru_vals  else None},
        }
        print(f"\n{cat}  ({total_lstm} pairs)")
        print(f"  vs LSTM: {wins_lstm}/{total_lstm} wins, {losses_lstm} losses  "
              f"  mean_DM={np.mean(dm_lstm_vals) if dm_lstm_vals else 'N/A':+.3f}")
        print(f"  vs GRU:  {wins_gru}/{total_gru} wins, {losses_gru} losses    "
              f"  mean_DM={np.mean(dm_gru_vals) if dm_gru_vals else 'N/A':+.3f}")

    with open("results/category_summary.json", "w") as f:
        json.dump(cat_summary, f, indent=2)

    # ── DCC-informed DiD ─────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("MULTI-CATEGORY DiD  (C1+C2 treated vs C5 control, P2→P3)")
    print("=" * 68)

    treated_cats = ["C1_BTC_family", "C2_ETH_ecosystem"]
    control_cats = ["C5_Speculative"]

    treated_assets = [a for c in treated_cats for a in CATEGORIES[c]]
    control_assets = CATEGORIES["C5_Speculative"]

    def get_dm(asset, period, vs="LSTM"):
        key = f"{asset}_{period}"
        if key not in results: return None
        return results[key][f"DM_vs_{vs}"]["stat"]

    # ── DiD per comparison (vs LSTM and vs GRU) ───────────────────────────────
    for vs in ["LSTM", "GRU"]:
        t_assets = [a for a in treated_assets
                    if get_dm(a, "P2", vs) is not None and get_dm(a, "P3", vs) is not None]
        c_assets = [a for a in control_assets
                    if get_dm(a, "P2", vs) is not None and get_dm(a, "P3", vs) is not None]

        if not t_assets or not c_assets:
            print(f"\nvs {vs}: insufficient data"); continue

        all_avail = t_assets + c_assets
        n_t = len(t_assets)

        def compute_did(tr_set):
            tr_set = set(tr_set)
            vals = []
            for a in all_avail:
                delta = get_dm(a, "P3", vs) - get_dm(a, "P2", vs)
                sign  = -1 if a in tr_set else 1
                vals.append(sign * delta)
            return np.mean(vals)

        did_obs = compute_did(t_assets)
        perm_dids = [compute_did(perm)
                     for perm in combinations(all_avail, n_t)]

        p_left  = sum(d <= did_obs for d in perm_dids) / len(perm_dids)
        p_right = sum(d >= did_obs for d in perm_dids) / len(perm_dids)
        p_two   = min(p_left, p_right) * 2

        print(f"\nvs {vs}: treated={t_assets} (n={n_t})")
        print(f"         control={c_assets} (n={len(c_assets)})")
        print(f"         C({len(all_avail)},{n_t})={len(perm_dids)} permutations")
        print(f"         observed DiD = {did_obs:+.4f}")
        print(f"         one-sided p (left)  = {p_left:.6f}")
        print(f"         one-sided p (right) = {p_right:.6f}")
        print(f"         two-sided p         = {p_two:.6f}")

        did_out = {
            "design": f"C1+C2 treated vs C5 control, vs {vs}, P2→P3",
            "treated": t_assets, "control": c_assets,
            "n_permutations": len(perm_dids),
            "observed_DiD": float(did_obs),
            "p_left": float(p_left), "p_right": float(p_right),
            "p_two": float(p_two),
        }
        with open(f"results/did_52asset_{vs}.json", "w") as f:
            json.dump(did_out, f, indent=2)

    # ── period × category win-rate heat map ──────────────────────────────────
    print("\n" + "=" * 68)
    print("WIN RATE vs LSTM  (by category × period)")
    print(f"{'Category':<25} {'P1':>8} {'P2':>8} {'P3':>8} {'Total':>8}")
    print("-" * 68)

    for cat, assets in CATEGORIES.items():
        row_str = f"{cat:<25}"
        total_w = 0; total_n = 0
        for period in periods:
            w = 0; n = 0
            for a in assets:
                key = f"{a}_{period}"
                if key not in results: continue
                d = results[key]["DM_vs_LSTM"]
                n += 1
                if d["pval"] < 0.05 and d["stat"] < 0: w += 1
            total_w += w; total_n += n
            row_str += f" {w:>2}/{n:<2}    "
        row_str += f"  {total_w:>2}/{total_n:<2}"
        print(row_str)

    elapsed = time.time() - t_global
    print(f"\nTotal elapsed: {elapsed/60:.1f} min")
    print(f"Pairs trained: {done_count}  |  Cached: {skip_count}")


if __name__ == "__main__":
    main()
