"""
download_expanded.py
====================
Download 5-min OHLCV bars for all available Coinbase assets across
three market-regime windows (P1, P2, P3).

Skips any file that already exists (BTC, ETH, XRP, LTC, DOGE, SOL, ADA, BCH).

Usage:
    cd ../data
    python download_expanded.py
"""

import json, time, os, sys, urllib.request
from datetime import datetime, timezone
import pandas as pd

GRAN  = 300   # 5-min in seconds
CHUNK = 300   # candles per request (Coinbase max)
SLEEP = 0.45  # seconds between requests

PERIODS = {
    "P1": ("2022-11-01T00:00:00Z", "2022-11-30T23:55:00Z", "20221101"),
    "P2": ("2023-10-01T00:00:00Z", "2023-10-31T23:55:00Z", "20231001"),
    "P3": ("2024-03-01T00:00:00Z", "2024-03-31T23:55:00Z", "20240301"),
}

# ── 5 scientific categories ────────────────────────────────────────────────────
CATEGORIES = {
    "C1_BTC_family":     ["BTC", "LTC", "BCH", "DOGE", "SHIB"],
    "C2_ETH_ecosystem":  ["ETH", "MATIC", "ARB", "OP", "LRC"],
    "C3_Alt_L1":         ["SOL", "ADA", "AVAX", "DOT", "ATOM",
                          "NEAR", "ALGO", "ICP", "HBAR", "VET", "APT"],
    "C4_DeFi":           ["LINK", "AAVE", "UNI", "MKR", "CRV",
                          "COMP", "SNX", "GRT", "SUSHI", "YFI", "BAL", "1INCH"],
    "C5_Speculative":    ["XRP", "EOS", "XTZ", "MANA", "SAND", "AXS",
                          "APE", "ENJ", "CHZ", "BAT", "ZRX",
                          "ANKR", "OXT", "NMR"],
}

OUTPUT_COLS = [
    "timestamp", "low", "high", "open", "close", "volume",
    "vwap", "turnover", "homeNotional", "foreignNotional", "trades",
]


def fetch_chunk(url):
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"\n    [retry] {e}", flush=True)
        time.sleep(3)
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())


def download(asset, start_iso, end_iso, label):
    base = f"https://api.exchange.coinbase.com/products/{asset}-USD/candles"
    su = int(datetime.fromisoformat(start_iso.replace("Z", "+00:00")).timestamp())
    eu = int(datetime.fromisoformat(end_iso.replace("Z", "+00:00")).timestamp())

    rows, cursor = [], su
    n_chunks = (eu - su) // GRAN // CHUNK + 1
    print(f"    {asset} {label} ({n_chunks} req)...", end=" ", flush=True)

    while cursor <= eu:
        ce = min(cursor + CHUNK * GRAN - GRAN, eu)
        s  = datetime.fromtimestamp(cursor, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        e  = datetime.fromtimestamp(ce,     tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        data = fetch_chunk(f"{base}?start={s}&end={e}&granularity={GRAN}")
        if isinstance(data, list):
            rows.extend(data)
        cursor = ce + GRAN
        time.sleep(SLEEP)

    print(f"{len(rows)} raw rows", flush=True)
    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["_ts", "low", "high", "open", "close", "volume"])
    df["_ts"] = df["_ts"].astype(int)
    df = df.drop_duplicates("_ts").sort_values("_ts").reset_index(drop=True)
    df = df[(df["_ts"] >= su) & (df["_ts"] <= eu)].reset_index(drop=True)

    # fill any gaps
    all_ts = list(range(su, eu + GRAN, GRAN))
    df = pd.DataFrame({"_ts": all_ts}).merge(df, on="_ts", how="left").ffill().bfill()

    df["timestamp"] = df["_ts"].apply(
        lambda t: datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00"))
    df["vwap"]           = ((df["open"] + df["high"] + df["low"] + df["close"]) / 4).round(6)
    df["turnover"]       = (df["volume"] * df["vwap"]).round(8)
    df["homeNotional"]   = df["volume"].round(8)
    df["foreignNotional"]= df["turnover"]
    df["trades"]         = ""
    return df[OUTPUT_COLS]


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    downloaded, skipped, failed = [], [], []

    all_assets = []
    for cat, assets in CATEGORIES.items():
        for a in assets:
            all_assets.append((cat, a))

    total = len(all_assets) * 3
    done  = 0

    for cat, asset in all_assets:
        print(f"\n[{cat}] {asset}", flush=True)
        for period, (s, e, tag) in PERIODS.items():
            fname = f"{asset}_5m_{period}_{tag}.csv"
            fpath = os.path.join(out_dir, fname)
            done += 1

            if os.path.exists(fpath):
                rows = sum(1 for _ in open(fpath)) - 1
                print(f"    SKIP {fname} ({rows} rows already)", flush=True)
                skipped.append(fname)
                continue

            try:
                df = download(asset, s, e, period)
                if df is None or len(df) < 100:
                    print(f"    FAIL {fname}: too few rows", flush=True)
                    failed.append(fname)
                    continue
                df.to_csv(fpath, index=False)
                c = df["close"].astype(float)
                print(f"    SAVE {fname}  {len(df)} bars  "
                      f"close {c.iloc[0]:.4f}→{c.iloc[-1]:.4f}  "
                      f"[{done}/{total}]", flush=True)
                downloaded.append(fname)
            except Exception as ex:
                print(f"    ERR  {fname}: {ex}", flush=True)
                failed.append(fname)

    print("\n" + "="*60)
    print(f"Downloaded: {len(downloaded)}")
    print(f"Skipped (existed): {len(skipped)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"  {failed}")


if __name__ == "__main__":
    main()
