"""
run_dcc_garch.py — DCC-GARCH analysis + GARCH-X forecasting baseline.

Two components:

1. GARCH-X baseline (ARX(1)-GARCH(1,1)):
   For each altcoin (ETH, XRP, LTC, DOGE), BTC lagged log-return is added
   as an exogenous variable in the mean equation. Yields a forecasting baseline
   that explicitly models cross-asset momentum in both mean and volatility.
   Rolling 5-step-ahead forecasts → NMSE + DM test vs MSCA.

2. DCC-GARCH conditional correlation (Engle 2002):
   Bivariate DCC(1,1) on (BTC, altcoin) standardised residuals.
   Mean conditional correlation extracted per period → used for causal ID
   (did BTC–altcoin co-movement increase more for treated assets ETH/XRP
   than for controls LTC/DOGE following the BTC spot ETF approval in Jan 2024?).

Output: results/dcc_garch_results.json

Usage:
    cd code-data-pr+/code
    python examples/run_dcc_garch.py
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msca_sparsity import MSCABiGRU, Trainer, build_features
from msca_sparsity.train import dm_test

DATA     = "../data"
MSCA_JSON = "results/multiseed_results.json"
SEED     = 42
DEVICE   = "cpu"

CFG = dict(window=30, horizon=5, hidden=64, n_heads=4, dropout=0.25,
           lr=5e-4, epochs=200, patience=25, warmup=30,
           batch_size=64, train_ratio=0.70, val_frac=0.10, n_features=7)

DATE_TAG = {"P1": "20221101", "P2": "20231001", "P3": "20240301"}
ASSETS   = ["ETH", "XRP", "LTC", "DOGE"]   # BTC is the cross-asset driver
PERIODS  = ["P1", "P2", "P3"]
CA_MAP   = {"ETH": "BTC", "XRP": "BTC", "LTC": "BTC", "DOGE": "BTC"}


def csv_path(asset, period):
    return os.path.join(DATA, f"{asset}_5m_{period}_{DATE_TAG[period]}.csv")


def load_close(asset, period):
    df = pd.read_csv(csv_path(asset, period))
    return df["close"].astype(float).values


def log_ret(close):
    return np.diff(np.log(np.maximum(close, 1e-10)))


def make_windows(close, feat_tgt, feat_ca, window=30, horizon=5):
    lr = log_ret(close)
    n  = min(len(feat_tgt), len(feat_ca))
    X, Xca, y, pc = [], [], [], []
    for i in range(n - window - horizon + 1):
        X.append(feat_tgt[i:i + window])
        Xca.append(feat_ca[i:i + window])
        y.append(lr[i + window - 1: i + window + horizon - 1].sum())
        pc.append(close[i + window - 1])
    return (np.array(X, np.float32), np.array(Xca, np.float32),
            np.array(y, np.float32), np.array(pc, np.float32))


def prepare_splits(asset, period):
    ca = CA_MAP[asset]
    close, feat_tgt = build_features(csv_path(asset, period), CFG["n_features"])
    _,     feat_ca  = build_features(csv_path(ca, period), CFG["n_features"])
    price_range = float(close.max() - close.min())
    X, Xca, y, pc = make_windows(close, feat_tgt, feat_ca,
                                  CFG["window"], CFG["horizon"])
    n  = len(y)
    nt = int(n * CFG["train_ratio"])
    nv = int(n * CFG["val_frac"])
    F  = X.shape[-1]
    sc = MinMaxScaler(); sca = MinMaxScaler()
    sc.fit(X[:nt].reshape(-1, F)); sca.fit(Xca[:nt].reshape(-1, F))
    X   = sc.transform(X.reshape(-1, F)).reshape(X.shape)
    Xca = sca.transform(Xca.reshape(-1, F)).reshape(Xca.shape)
    return dict(tr=(X[:nt], Xca[:nt], y[:nt]),
                val=(X[nt:nt+nv], Xca[nt:nt+nv], y[nt:nt+nv]),
                te=(X[nt+nv:], Xca[nt+nv:], y[nt+nv:]),
                pte=pc[nt+nv:], price_range=price_range, F=F,
                n_train=nt, n_val=nv)


def msca_predictions(asset, period, splits):
    """Re-run MSCA at seed 42 to get test prediction vector."""
    import torch
    torch.manual_seed(SEED); np.random.seed(SEED)
    ca = CA_MAP[asset]
    close_ca, _ = build_features(csv_path(ca, period), CFG["n_features"])
    model = MSCABiGRU(feat_dim=splits["F"], hidden=CFG["hidden"],
                      n_heads=CFG["n_heads"], dropout=CFG["dropout"])
    trainer = Trainer(model, lr=CFG["lr"], epochs=CFG["epochs"],
                      patience=CFG["patience"], warmup=CFG["warmup"],
                      batch_size=CFG["batch_size"], device=DEVICE, verbose=False)
    trainer.fit(*splits["tr"], *splits["val"])
    pred = trainer.predict(*splits["te"][:2])
    return pred


# ── Component 1: GARCH-X (ARX(1)-GARCH(1,1)) ─────────────────────────────────

def fit_garchx(asset, period, splits):
    """
    Fit ARX(1)-GARCH(1,1) where BTC lagged log-return is the exogenous
    variable in the altcoin mean equation.  Rolling 5-step forecast on test.
    """
    ca   = CA_MAP[asset]
    alt_close = load_close(asset, period)
    btc_close = load_close(ca,    period)
    alt_ret   = log_ret(alt_close) * 1e4   # scale for numerical stability
    btc_ret   = log_ret(btc_close) * 1e4

    # align lengths
    n  = min(len(alt_ret), len(btc_ret))
    alt_ret = alt_ret[:n]; btc_ret = btc_ret[:n]

    # align with window logic: first usable return index = window-1
    w   = CFG["window"]; h = CFG["horizon"]
    nt  = splits["n_train"]; nv = splits["n_val"]
    # test set starts at index nt+nv in windows; each window starts at raw index i
    # window i uses close[i:i+window], so last close = close[i+window-1]
    # returns[i+window-1 .. i+window+horizon-2] → we need alt_ret starting from w-1
    # Training indices in return space: 0 .. nt+w-2
    te_start_ret = nt + nv + w - 1   # first return index of first test window

    # Fit on training portion of returns
    train_alt = alt_ret[:nt + w - 1]
    train_btc = btc_ret[:nt + w - 1]
    btc_lag   = np.roll(train_btc, 1); btc_lag[0] = 0.0  # lag 1

    am = arch_model(train_alt, x=btc_lag.reshape(-1, 1),
                    mean='ARX', lags=1, vol='GARCH', p=1, q=1, dist='normal')
    res = am.fit(disp='off', show_warning=False)
    params = res.params   # intercept, btc_lag coef, AR(1) coef, omega, alpha, beta

    # Rolling 1-step-ahead forecast on test, accumulate into 5-step cumulative
    y_true = splits["te"][2]          # shape (n_test,)
    n_test = len(y_true)
    garchx_pred = np.zeros(n_test)

    for i in range(n_test):
        idx = te_start_ret + i
        # use actual alt return up to idx-1 and btc return at idx (contemporaneous BTC lag)
        # AR(1) forecast: E[r_t] = mu + phi*r_{t-1} + gamma*btc_{t-1}
        mu    = params.get('Const', params.iloc[0])
        # parameter ordering: Const, x[0], [r[1] if ARX], omega, alpha[1], beta[1]
        names = list(params.index)
        gamma = params.iloc[1] if len(names) > 1 else 0.0   # BTC exog coef
        phi   = params.iloc[2] if len(names) > 2 else 0.0   # AR(1) coef
        last_r = alt_ret[idx - 1] if idx > 0 else 0.0
        last_btc = btc_ret[idx - 1] if idx > 0 else 0.0
        cum = 0.0
        r_prev = last_r; btc_prev = last_btc
        for k in range(h):
            r_next = mu + phi * r_prev + gamma * btc_prev
            cum   += r_next / 1e4
            r_prev = r_next
            btc_prev = btc_ret[idx + k] if (idx + k) < len(btc_ret) else btc_prev
        garchx_pred[i] = cum

    pr = splits["price_range"]
    nmse_garchx = float(np.mean((garchx_pred - y_true)**2) / (pr**2 + 1e-12) * 1e3)
    return nmse_garchx, garchx_pred


# ── Component 2: DCC-GARCH conditional correlation ───────────────────────────

def fit_dcc(btc_ret_tr, alt_ret_tr, btc_ret_full, alt_ret_full):
    """
    Fit bivariate DCC(1,1) on training data, then filter conditional
    correlations ρ_t over the full series.  Returns the full ρ_t series.
    """
    # Step 1: Fit univariate GARCH(1,1) to each asset
    def fit_garch11(r):
        am  = arch_model(r * 1e4, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
        res = am.fit(disp='off', show_warning=False)
        return res

    res_btc = fit_garch11(btc_ret_tr)
    res_alt = fit_garch11(alt_ret_tr)

    # Step 2: Get standardised residuals on training data
    def std_resids(res, r_full):
        params = res.params
        mu     = params.get('mu', 0.0) if 'mu' in params.index else params.iloc[0]
        omega  = params.get('omega', params.iloc[-3])
        alpha  = params.get('alpha[1]', params.iloc[-2])
        beta   = params.get('beta[1]',  params.iloc[-1])
        r = r_full * 1e4
        T = len(r)
        h = np.full(T, omega / max(1 - alpha - beta, 1e-6))
        e = np.zeros(T)
        for t in range(1, T):
            e[t-1]  = r[t-1] - mu
            h[t]    = omega + alpha * e[t-1]**2 + beta * h[t-1]
        e[-1] = r[-1] - mu
        return e / np.sqrt(np.maximum(h, 1e-10))

    e_btc = std_resids(res_btc, btc_ret_full)
    e_alt = std_resids(res_alt, alt_ret_full)

    # Step 3: Fit DCC(1,1) on training standardised residuals
    n_tr = len(btc_ret_tr)
    e_btc_tr = e_btc[:n_tr]; e_alt_tr = e_alt[:n_tr]
    Q_bar = np.cov(np.vstack([e_btc_tr, e_alt_tr]))   # 2×2 unconditional cov

    # Grid-search a, b on training data (maximize log-likelihood of DCC)
    best_ll = -np.inf; best_ab = (0.05, 0.90)
    for a in np.arange(0.02, 0.20, 0.02):
        for b in np.arange(0.70, 0.98, 0.02):
            if a + b >= 1.0:
                continue
            ll = _dcc_ll(e_btc_tr, e_alt_tr, Q_bar, a, b)
            if ll > best_ll:
                best_ll = ll; best_ab = (a, b)

    a, b = best_ab
    # Step 4: Filter over full series
    rho = _dcc_filter(e_btc, e_alt, Q_bar, a, b)
    return rho, a, b


def _dcc_ll(e1, e2, Q_bar, a, b):
    T  = len(e1)
    Q  = Q_bar.copy(); ll = 0.0
    for t in range(T):
        eps = np.array([e1[t], e2[t]])
        q11 = Q[0, 0]; q22 = Q[1, 1]; q12 = Q[0, 1]
        rho = q12 / (np.sqrt(max(q11, 1e-10)) * np.sqrt(max(q22, 1e-10)))
        rho = np.clip(rho, -0.999, 0.999)
        det = 1 - rho**2
        ll -= 0.5 * (np.log(max(det, 1e-10)) + (eps[0]**2 + eps[1]**2 - 2*rho*eps[0]*eps[1]) / det)
        Q  = (1 - a - b) * Q_bar + a * np.outer(eps, eps) + b * Q
    return ll


def _dcc_filter(e1, e2, Q_bar, a, b):
    T   = len(e1)
    Q   = Q_bar.copy()
    rho = np.zeros(T)
    for t in range(T):
        eps = np.array([e1[t], e2[t]])
        q11 = Q[0, 0]; q22 = Q[1, 1]; q12 = Q[0, 1]
        rho[t] = q12 / (np.sqrt(max(q11, 1e-10)) * np.sqrt(max(q22, 1e-10)))
        rho[t] = np.clip(rho[t], -0.999, 0.999)
        Q = (1 - a - b) * Q_bar + a * np.outer(eps, eps) + b * Q
    return rho


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)
    t0 = time.time()

    with open(MSCA_JSON) as f:
        msca_data = json.load(f)

    dcc_results = {}
    garchx_results = {}

    for period in PERIODS:
        btc_close = load_close("BTC", period)
        btc_ret   = log_ret(btc_close)

        for asset in ASSETS:
            key = f"{asset}_{period}"
            print(f"  [{asset}] {period}", end=" ", flush=True)

            splits = prepare_splits(asset, period)
            nt, nv = splits["n_train"], splits["n_val"]

            # ── GARCH-X baseline ──────────────────────────────────────────
            try:
                nmse_gx, gx_pred = fit_garchx(asset, period, splits)
            except Exception as ex:
                print(f"GARCHX err: {ex}", end=" ")
                nmse_gx, gx_pred = np.nan, np.zeros(len(splits["te"][2]))

            # Re-run MSCA for DM test
            try:
                msca_pred = msca_predictions(asset, period, splits)
                y_te      = splits["te"][2]
                e_gx   = gx_pred  - y_te
                e_msca = msca_pred - y_te
                dm_stat, dm_pval = dm_test(e_gx, e_msca)
                dm_sig = "*" if dm_pval < 0.05 else "ns"
            except Exception as ex:
                print(f"DM err: {ex}", end=" ")
                dm_stat, dm_pval, dm_sig = np.nan, np.nan, "?"

            msca_nmse = msca_data.get(key, {}).get("MSCA", {}).get("mean",
                        msca_data.get(key, {}).get("mean", np.nan))

            garchx_results[key] = dict(
                GARCHX_NMSE=round(nmse_gx, 4) if not np.isnan(nmse_gx) else None,
                MSCA_NMSE=round(float(msca_nmse), 4),
                DM_stat=round(float(dm_stat), 3) if not np.isnan(dm_stat) else None,
                DM_pval=round(float(dm_pval), 4) if not np.isnan(dm_pval) else None,
                sig=dm_sig,
            )
            print(f"GARCHX={nmse_gx:.4f} MSCA={msca_nmse:.4f} DM={dm_stat:+.3f}({dm_sig})",
                  end="  ", flush=True)

            # ── DCC-GARCH correlation ─────────────────────────────────────
            try:
                alt_close = load_close(asset, period)
                alt_ret   = log_ret(alt_close)
                n         = min(len(btc_ret), len(alt_ret))
                br = btc_ret[:n]; ar = alt_ret[:n]
                # training portion in return space: index 0..nt+w-2
                w_ret = nt + CFG["window"] - 1
                rho_series, dcc_a, dcc_b = fit_dcc(br[:w_ret], ar[:w_ret], br, ar)
                mean_rho = float(np.nanmean(rho_series))
                # also compute per-period mean using full series
                dcc_results[key] = dict(
                    mean_rho=round(mean_rho, 4),
                    dcc_a=round(dcc_a, 3),
                    dcc_b=round(dcc_b, 3),
                    rho_series_len=len(rho_series),
                )
                print(f"ρ̄={mean_rho:.4f}(a={dcc_a:.2f},b={dcc_b:.2f})", flush=True)
            except Exception as ex:
                print(f"DCC err: {ex}", flush=True)
                dcc_results[key] = dict(mean_rho=None, error=str(ex))

    elapsed = time.time() - t0
    print(f"\n  All done in {elapsed/60:.1f} min")

    out = dict(garchx=garchx_results, dcc=dcc_results)
    with open("results/dcc_garch_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("  Saved → results/dcc_garch_results.json")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n=== GARCH-X summary (vs MSCA) ===")
    for asset in ASSETS:
        for period in PERIODS:
            r = garchx_results[f"{asset}_{period}"]
            print(f"  {asset} {period}: GARCHX={r['GARCHX_NMSE']}  "
                  f"MSCA={r['MSCA_NMSE']}  DM={r['DM_stat']}({r['sig']})")

    print("\n=== DCC conditional correlation ρ̄(BTC, altcoin) ===")
    print(f"  {'Asset':<6}  {'P1':>8}  {'P2':>8}  {'P3':>8}  {'ΔP2→P3':>8}")
    for asset in ASSETS:
        rhos = [dcc_results.get(f"{asset}_{p}", {}).get("mean_rho") for p in PERIODS]
        delta = (rhos[2] - rhos[1]) if all(r is not None for r in rhos[1:]) else None
        row = f"  {asset:<6}  " + "  ".join(f"{r:>8.4f}" if r is not None else f"{'?':>8}" for r in rhos)
        if delta is not None:
            row += f"  {delta:>+8.4f}"
        print(row)


if __name__ == "__main__":
    main()
