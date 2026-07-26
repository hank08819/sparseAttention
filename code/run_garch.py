"""
run_garch.py — GARCH(1,1) + AR(1) mean baseline for all 15 asset-period pairs.

Trains AR(1)-GARCH(1,1) on the training log-return series, makes rolling
5-step-ahead cumulative-return forecasts on the test set, then computes NMSE
and DM statistics vs MSCA-SBiGRU (re-run at seed 42 for the prediction vector).

Results are merged into multiseed_results.json as GARCH_<asset>_<period> keys.

Usage:
    cd code-data-pr+/code
    python examples/run_garch.py
"""
import json, os, sys, time, warnings
import numpy as np
import torch
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msca_sparsity import MSCABiGRU, Trainer, build_features
from msca_sparsity.train import dm_test

DATA    = "../data"
JSON    = "results/multiseed_results.json"
SEED    = 42
DEVICE  = "cpu"

CFG = dict(window=30, horizon=5, hidden=64, n_heads=4, dropout=0.25,
           lr=5e-4, epochs=200, patience=25, warmup=30,
           batch_size=64, train_ratio=0.70, val_frac=0.10, n_features=7)

DATE_TAG = {"P1": "20221101", "P2": "20231001", "P3": "20240301"}
ASSETS   = ["BTC", "ETH", "XRP", "LTC", "DOGE"]
PERIODS  = ["P1", "P2", "P3"]
CA_MAP   = {"BTC": "ETH", "ETH": "BTC", "XRP": "BTC",
            "LTC": "BTC", "DOGE": "BTC"}


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

    # raw log-return series for GARCH (need the original close for this)
    close_raw, _ = build_features(csv_path(asset, period), CFG["n_features"])
    log_ret_raw  = np.diff(np.log(np.maximum(close_raw, 1e-10)))

    return dict(
        tr=(X[:nt], Xca[:nt], y[:nt]),
        val=(X[nt:nt+nv], Xca[nt:nt+nv], y[nt:nt+nv]),
        te=(X[nt+nv:], Xca[nt+nv:], y[nt+nv:]),
        pte=pc[nt+nv:], price_range=price_range, F=F,
        log_ret_raw=log_ret_raw, nt=nt, nv=nv,
        n_windows=n, close_full=close_raw,
    )


def nmse_from_price_errors(pred_prices, true_prices, price_range):
    mse = float(np.mean((pred_prices - true_prices) ** 2))
    return mse / (price_range ** 2 + 1e-12) * 1e3


def run_garch(splits):
    """Fit AR(1)-GARCH(1,1) and forecast 5-step cumulative returns on test."""
    log_ret = splits["log_ret_raw"]
    nt_raw  = splits["nt"] + CFG["window"]   # align with window start
    nv_raw  = splits["nv"]

    # training returns (align to match window offset)
    train_ret = log_ret[: nt_raw]

    gm = arch_model(train_ret * 1e4,       # scale to avoid numerical issues
                    mean="AR", lags=1,
                    vol="GARCH", p=1, q=1,
                    dist="normal")
    res = gm.fit(disp="off", show_warning=False)

    # forecast the MEAN for each test window
    # Each test sample starts at nt_raw + nv_raw + i and predicts h=5 ahead
    te_start = nt_raw + nv_raw
    n_te = len(splits["pte"])

    pred_cum_ret = np.zeros(n_te, dtype=np.float32)
    h = CFG["horizon"]

    mu  = res.params.get("Const", 0.0)        # AR constant
    phi = res.params.get("log_ret_raw[1]",     # AR(1) coefficient
                         res.params.get("AR.L1", 0.0))

    for i in range(n_te):
        # last observed return before this test window
        idx = te_start + i
        last_ret = log_ret[idx - 1] * 1e4 if idx > 0 else 0.0

        # h-step-ahead mean forecast for each step
        cum = 0.0
        for k in range(1, h + 1):
            cum += (mu + phi**k * (last_ret - mu)) / 1e4
        pred_cum_ret[i] = float(cum)

    # convert cumulative log-return to price
    close_te = splits["pte"]
    pred_prices = close_te * np.exp(pred_cum_ret)
    true_prices = close_te * np.exp(splits["te"][2])

    nmse  = nmse_from_price_errors(pred_prices, true_prices, splits["price_range"])
    errors_garch = splits["te"][2] - pred_cum_ret     # forecast errors (log-ret space)
    return nmse, pred_cum_ret, errors_garch


def run_msca(splits):
    """Re-run MSCA seed 42 to obtain prediction vector."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = MSCABiGRU(feat_dim=splits["F"],
                      hidden=64, n_heads=4, dropout=0.25, lambda_gate=0.05)
    trainer = Trainer(model, lr=CFG["lr"], epochs=CFG["epochs"],
                      patience=CFG["patience"], warmup=CFG["warmup"],
                      batch_size=CFG["batch_size"], device=DEVICE, verbose=False)
    trainer.fit(*splits["tr"], *splits["val"])
    metrics = trainer.evaluate(*splits["te"], splits["pte"],
                               price_range=splits["price_range"])
    pred = trainer.predict(*splits["te"][:2])
    errors_msca = splits["te"][2] - pred
    return metrics["NMSE"], pred, errors_msca


def main():
    torch.set_num_threads(4)
    os.makedirs("results", exist_ok=True)
    t0 = time.time()

    garch_results = {}

    for asset in ASSETS:
        for period in PERIODS:
            key = f"{asset}_{period}"
            print(f"  [{asset}] {period} ...", end=" ", flush=True)

            splits = prepare_splits(asset, period)

            # GARCH baseline
            g_nmse, g_pred, e_garch = run_garch(splits)

            # MSCA seed-42 (for DM test)
            m_nmse, m_pred, e_msca = run_msca(splits)

            # DM test: MSCA vs GARCH
            dm_stat, dm_pval = dm_test(e_msca, e_garch, horizon=5)
            sig = ("***" if dm_pval < .001 else "**" if dm_pval < .01
                   else "*" if dm_pval < .05 else "ns")

            garch_results[key] = {
                "GARCH_NMSE": round(g_nmse, 4),
                "MSCA_NMSE":  round(m_nmse, 4),
                "DM_stat":    round(dm_stat, 3),
                "DM_pval":    round(dm_pval, 4),
                "sig":        sig,
            }
            print(f"GARCH={g_nmse:.4f}  MSCA={m_nmse:.4f}  "
                  f"DM={dm_stat:+.3f}({sig})", flush=True)

    elapsed = time.time() - t0
    print(f"\n  All done in {elapsed/60:.1f} min")

    # Save separate GARCH results file
    with open("results/garch_results.json", "w") as f:
        json.dump(garch_results, f, indent=2)
    print("  Saved → results/garch_results.json")

    print("\n=== GARCH summary ===")
    for asset in ASSETS:
        for period in PERIODS:
            r = garch_results[f"{asset}_{period}"]
            print(f"  {asset} {period}: GARCH={r['GARCH_NMSE']:.4f}  "
                  f"MSCA={r['MSCA_NMSE']:.4f}  DM={r['DM_stat']:+.3f}({r['sig']})")


if __name__ == "__main__":
    main()
