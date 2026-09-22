#!/usr/bin/env python3
"""Cross-sectional test of the frozen rule on real cryptocurrency data.

See PREREGISTRATION_crosssection.md. Direction of attention is changed: instead
of weighting past bars of one asset, the predictor weights the contemporaneous
returns of the other assets. That geometry is what the rule asks for.

  --predict-only   training-split coordinates and the rule's prediction
  --run            20 rolling origins per cell; writes results/cross_section.json
"""
from __future__ import annotations
import sys, os, json, glob, argparse, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'package' / 'misa_sparsity'))
import run_boundary_benchmarks as bb
import importlib.util
_c = importlib.util.spec_from_file_location('cf', ROOT / 'scripts' / 'run_confirmatory.py')
cf = importlib.util.module_from_spec(_c); _c.loader.exec_module(cf)

DATA = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/paper2026E/ICLR2027_P+/code-data-pr+/data'
PERIODS = ['P1', 'P2', 'P3']
H = 5                 # forecast horizon in 5-minute bars
NS = (128, 256)       # training block sizes
N_ORIGIN = 20         # replications: evenly spaced rolling origins
TEST_LEN = 500        # bars scored after each training block
REL_MARGIN = 0.005    # practical margin, as a fraction of the cell's softmax MSE
SCORE_BOUND = 8.0     # wide bound, as everywhere on real data

def panel(period):
    cols = {}
    for f in sorted(glob.glob(str(DATA / f'*_5m_{period}_*.csv'))):
        sym = os.path.basename(f).split('_')[0]
        try:
            d = pd.read_csv(f)
            c = [x for x in d.columns if x.lower() in ('close', 'c')][0]
            cols[sym] = pd.Series(d[c].values)
        except Exception:
            pass
    return pd.DataFrame(cols).dropna()

def cell_data(M, target):
    L = np.log(M.values); R = L[1:] - L[:-1]
    t = list(M.columns).index(target)
    y = L[1 + H:, t] - L[1:-H, t]
    X = np.delete(R[:-H], t, axis=1)
    return X, y

def origins(T, n):
    last = T - n - TEST_LEN
    if last < 0: return []
    step = max(1, last // (N_ORIGIN - 1))
    return [min(i * step, last) for i in range(N_ORIGIN)]

def regime(d, scale):
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d)); floor = REL_MARGIN * scale
    if m > 2 * se and m > floor: return 'win'
    if m < -2 * se and m < -floor: return 'dense'
    return 'tie'

def run_cell(X, y, n):
    bb.SCORE_BOUND = SCORE_BOUND
    ds, soft, sparse = [], [], []
    for o in origins(len(y), n):
        Xtr, ytr = X[o:o + n], y[o:o + n]
        Xte, yte = X[o + n:o + n + TEST_LEN], y[o + n:o + n + TEST_LEN]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
        A = (Xtr - mu) / sd; B = (Xte - mu) / sd
        ym, ys = ytr.mean(), ytr.std() + 1e-12
        a_y, b_y = (ytr - ym) / ys, (yte - ym) / ys
        out = {}
        for al in (1.0, 2.0):
            p, u, b, *_ = bb.fit(al, A, a_y)
            out[al] = float(np.mean((u * (B @ p) + b - b_y) ** 2))
        soft.append(out[1.0]); sparse.append(out[2.0]); ds.append(out[1.0] - out[2.0])
    d = np.array(ds); scale = float(np.mean(soft))
    soft_a = np.array(soft); sparse_a = np.array(sparse)
    rel = (soft_a - sparse_a) / soft_a          # scale-free, per origin
    return dict(observed=regime(d, scale), mean_delta=float(d.mean()),
                se=float(d.std(ddof=1) / np.sqrt(len(d))), margin=REL_MARGIN * scale,
                n_origins=len(d), sparsemax_lower=int((d > 0).sum()),
                mse_softmax=scale, mse_sparsemax=float(np.mean(sparse)),
                mse_softmax_per_origin=[float(x) for x in soft_a],
                mse_sparsemax_per_origin=[float(x) for x in sparse_a],
                rel_mean=float(rel.mean()), rel_se=float(rel.std(ddof=1) / np.sqrt(len(rel))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predict-only', action='store_true')
    ap.add_argument('--run', action='store_true')
    a = ap.parse_args()
    out = {}; t0 = time.time()
    hdr = f"{'cell':<34}{'n':>5}{'r_eff':>7}{'k_eff':>7}{'kfrac':>8}{'PRED':>6}"
    print(hdr + (f"{'OBS':>7}{'Delta':>11}{'2SE':>10}{'margin':>9}{'sp<so':>8}" if a.run else ''))
    for period in PERIODS:
        M = panel(period)
        if M.shape[1] < 20: continue
        for target in list(M.columns):
            X, y = cell_data(M, target)
            for n in NS:
                if not origins(len(y), n): continue
                c = cf.coords(X[:n], y[:n])
                pred = 'win' if (c['kfrac'] <= 0.15 and n <= 256) else 'tie'
                key = f'{period}_{target}_n{n}'
                rec = dict(period=period, target=target, n_train=n, n_channels=int(X.shape[1]),
                           r_eff=c['r_eff'], k_eff=c['k_eff'], kfrac=c['kfrac'], predicted=pred)
                line = f"{period+' -> '+target:<34}{n:>5}{c['r_eff']:>7}{c['k_eff']:>7}{c['kfrac']:>8.3f}{pred.upper():>6}"
                if a.run:
                    r = run_cell(X, y, n); rec.update(r)
                    line += (f"{r['observed'].upper():>7}{r['mean_delta']:>+11.5f}{2*r['se']:>10.5f}"
                             f"{r['margin']:>9.5f}{r['sparsemax_lower']:>5}/{r['n_origins']:<2}")
                out[key] = rec; print(line, flush=True)
    if a.run:
        json.dump(out, open(ROOT / 'results' / 'cross_section.json', 'w'), indent=1)
        rows = list(out.values())
        pw = [r for r in rows if r['predicted'] == 'win']; pt = [r for r in rows if r['predicted'] == 'tie']
        f = lambda R, k: sum(r['observed'] == k for r in R)
        print(f"\npredicted win  ({len(pw):>3}): win {f(pw,'win')}  tie {f(pw,'tie')}  dense {f(pw,'dense')}")
        print(f"predicted tie  ({len(pt):>3}): win {f(pt,'win')}  tie {f(pt,'tie')}  dense {f(pt,'dense')}")
        if pw and pt:
            print(f"win rate: predicted-win {f(pw,'win')/len(pw):.3f}  vs  predicted-tie {f(pt,'win')/len(pt):.3f}")
        ms = np.mean([r['mse_softmax'] for r in rows]); mp = np.mean([r['mse_sparsemax'] for r in rows])
        mr = np.mean([r['mse_sparsemax'] if r['predicted'] == 'win' else r['mse_softmax'] for r in rows])
        print(f"mean test MSE: always softmax {ms:.5f}  always sparsemax {mp:.5f}  rule-selected {mr:.5f}")
        print(f"({time.time()-t0:.0f}s)  wrote results/cross_section.json")

if __name__ == '__main__': main()
