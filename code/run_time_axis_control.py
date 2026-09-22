#!/usr/bin/env python3
"""Matched time-axis control for the cross-sectional test.

Everything is inherited from scripts/run_cross_section.py -- the predictor, the
score bound, the targets, the rolling origins, the training sizes, the test
length, the decision rule and the number of input channels. One thing changes:
what the attention weights.

  cross-section : the contemporaneous returns of the other L assets
  time axis     : the target's own past L five-minute bars

L is the number of other assets in the same panel, so the two runs have the
identical channel count, the identical target and the identical protocol. The
only difference between them is the geometry of the inputs.

  --predict-only   training-split coordinates and the rule's prediction
  --run            20 rolling origins per cell; writes results/time_axis.json
"""
from __future__ import annotations
import sys, json, argparse, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
import importlib.util
_x = importlib.util.spec_from_file_location('xs', ROOT / 'scripts' / 'run_cross_section.py')
xs = importlib.util.module_from_spec(_x); _x.loader.exec_module(xs)
cf = xs.cf

H, NS, N_ORIGIN, TEST_LEN = xs.H, xs.NS, xs.N_ORIGIN, xs.TEST_LEN


def cell_data_time(M, target):
    """Same target and horizon as xs.cell_data; inputs are the target's own
    past L bars, with L equal to the number of other assets in the panel."""
    L_lag = M.shape[1] - 1                      # size-matched channel count
    Lg = np.log(M.values)
    t = list(M.columns).index(target)
    r = Lg[1:, t] - Lg[:-1, t]                  # five-minute log returns
    y_all = Lg[1 + H:, t] - Lg[1:-H, t]         # H bars ahead, as in xs
    n = len(y_all)
    # X[i] = [r(i), r(i-1), ..., r(i-L_lag+1)], aligned with y_all[i]
    X = np.full((n, L_lag), np.nan)
    for k in range(L_lag):
        X[k:, k] = r[:n - k]
    ok = ~np.isnan(X).any(axis=1)
    return X[ok], y_all[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predict-only', action='store_true')
    ap.add_argument('--run', action='store_true')
    a = ap.parse_args()
    out = {}; t0 = time.time()
    hdr = f"{'cell':<34}{'n':>5}{'r_eff':>7}{'k_eff':>7}{'kfrac':>8}{'PRED':>6}"
    print(hdr + (f"{'OBS':>7}{'Delta':>11}{'2SE':>10}{'margin':>9}{'sp<so':>8}" if a.run else ''))
    for period in xs.PERIODS:
        M = xs.panel(period)
        if M.shape[1] < 20: continue
        for target in list(M.columns):
            X, y = cell_data_time(M, target)
            for n in NS:
                if not xs.origins(len(y), n): continue
                c = cf.coords(X[:n], y[:n])
                pred = 'win' if (c['kfrac'] <= 0.15 and n <= 256) else 'tie'
                key = f'{period}_{target}_n{n}'
                rec = dict(period=period, target=target, n_train=n,
                           n_channels=int(X.shape[1]), r_eff=c['r_eff'],
                           k_eff=c['k_eff'], kfrac=c['kfrac'], predicted=pred)
                line = (f"{period+' -> '+target:<34}{n:>5}{c['r_eff']:>7}{c['k_eff']:>7}"
                        f"{c['kfrac']:>8.3f}{pred.upper():>6}")
                if a.run:
                    r = xs.run_cell(X, y, n); rec.update(r)
                    line += (f"{r['observed'].upper():>7}{r['mean_delta']:>+11.5f}"
                             f"{2*r['se']:>10.5f}{r['margin']:>9.5f}"
                             f"{r['sparsemax_lower']:>5}/{r['n_origins']:<2}")
                out[key] = rec; print(line, flush=True)
    if a.run:
        json.dump(out, open(ROOT / 'results' / 'time_axis.json', 'w'), indent=1)
        rows = list(out.values())
        rel = np.array([r['rel_mean'] for r in rows])
        lower = sum(1 for r in rows if r['mse_sparsemax'] < r['mse_softmax'])
        ks = [r['kfrac'] for r in rows]
        print(f"\ncells {len(rows)} | sparsemax lower {lower}/{len(rows)} | "
              f"mean relative gain {rel.mean()*100:+.2f}% | "
              f"kfrac {min(ks):.2f}-{max(ks):.2f}")
        print(f"({time.time()-t0:.0f}s)  wrote results/time_axis.json")


if __name__ == '__main__': main()
