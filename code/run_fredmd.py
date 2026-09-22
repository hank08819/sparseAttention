#!/usr/bin/env python3
"""FRED-MD test of the geometry claim (see PREREGISTRATION_fredmd.md).

126 monthly US macro series, 1959-2025, transformed by the codes shipped with
the file. A target series is predicted from the contemporaneous transformed
values of all the others: a high-dimensional, weakly correlated cross-section,
which is the geometry the bound describes and the one no time-axis dataset in
the paper had.

  --predict-only   training-split coordinates and the rule's prediction
  --run            rolling origins per cell; writes results/fredmd.json
"""
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, json, argparse, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'package' / 'misa_sparsity'))
import run_boundary_benchmarks as bb
import importlib.util
_c = importlib.util.spec_from_file_location('cf', ROOT / 'scripts' / 'run_confirmatory.py')
cf = importlib.util.module_from_spec(_c); _c.loader.exec_module(cf)

CSV = ROOT / 'data_external' / 'fred_md_current.csv'
H = 1                 # one month ahead
NS = (128, 256)
N_ORIGIN = 20
TEST_LEN = 120        # months scored after each training block
REL_MARGIN = 0.005
SCORE_BOUND = 8.0
# targets: the headline series of the four FRED-MD groups that are always present
TARGETS = ['INDPRO', 'PAYEMS', 'CPIAUCSL', 'FEDFUNDS', 'UNRATE', 'RPI', 'DPCERA3M086SBEA', 'S&P 500']

def load():
    raw = pd.read_csv(CSV)
    tcode = raw.iloc[0, 1:].astype(float)
    df = raw.iloc[1:].reset_index(drop=True)
    dates = pd.to_datetime(df.iloc[:, 0], format='%m/%d/%Y', errors='coerce')
    X = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    out = {}
    for c in X.columns:
        v = X[c].values.astype(float); t = int(tcode[c])
        with np.errstate(divide='ignore', invalid='ignore'):
            if   t == 1: z = v
            elif t == 2: z = np.r_[np.nan, np.diff(v)]
            elif t == 3: z = np.r_[np.nan, np.nan, np.diff(v, 2)]
            elif t == 4: z = np.log(v)
            elif t == 5: z = np.r_[np.nan, np.diff(np.log(v))]
            elif t == 6: z = np.r_[np.nan, np.nan, np.diff(np.log(v), 2)]
            elif t == 7: z = np.r_[np.nan, np.nan, np.diff(v[1:] / v[:-1] - 1.0)]
            else: z = v
        out[c] = z
    T = pd.DataFrame(out, index=dates)
    T = T.replace([np.inf, -np.inf], np.nan)
    T = T.loc['1960-01-01':]
    # Amendment 1: the standard FRED-MD pipeline (McCracken and Ng) removes
    # outliers after transformation. Without it a series that is near constant
    # in a training block and jumps later is standardized by an almost zero
    # scale, and the fitted predictor explodes: median test MSE of 2.5 and a
    # maximum of 773 on a target standardized to unit variance.
    med = T.median(); q1 = T.quantile(0.25); q3 = T.quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    T = T.mask((T - med).abs() > 10 * iqr)
    T = T.dropna(axis=1, thresh=int(0.98 * len(T)))
    T = T.interpolate(limit_direction='both').dropna()
    # drop any series that is still degenerate over the whole panel
    keep = T.std() > 1e-10 * T.abs().mean().clip(lower=1e-12)
    return T.loc[:, keep[keep].index]

def cell(T, target):
    j = list(T.columns).index(target)
    V = T.values
    y = V[H:, j]                  # one month ahead, already transformed
    X = np.delete(V[:-H], j, axis=1)
    return X, y

def origins(n_rows, n):
    last = n_rows - n - TEST_LEN
    if last < 0: return []
    step = max(1, last // (N_ORIGIN - 1))
    return [min(i * step, last) for i in range(N_ORIGIN)]

def regime(d, scale):
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d)); f = REL_MARGIN * scale
    if m > 2 * se and m > f: return 'win'
    if m < -2 * se and m < -f: return 'dense'
    return 'tie'

def run_cell(X, y, n):
    bb.SCORE_BOUND = SCORE_BOUND
    soft, sparse = [], []
    for o in origins(len(y), n):
        Xtr, ytr = X[o:o + n], y[o:o + n]
        Xte, yte = X[o + n:o + n + TEST_LEN], y[o + n:o + n + TEST_LEN]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        # Amendment 1: a column that is (near) constant on the training block
        # carries no usable scale; standardizing by it turns the test block into
        # astronomically large inputs. Such columns are dropped for that origin.
        ok = sd > 1e-8 * max(1.0, float(np.abs(Xtr).mean()))
        if ok.sum() < 5: continue
        Xtr_o, Xte_o = Xtr[:, ok], Xte[:, ok]
        mu, sd = Xtr_o.mean(0), Xtr_o.std(0)
        A = (Xtr_o - mu) / sd; B = (Xte_o - mu) / sd
        ym, ys = ytr.mean(), ytr.std() + 1e-12
        a_y, b_y = (ytr - ym) / ys, (yte - ym) / ys
        r = {}
        for al in (1.0, 2.0):
            p, u, b, *_ = bb.fit(al, A, a_y)
            r[al] = float(np.mean((u * (B @ p) + b - b_y) ** 2))
        soft.append(r[1.0]); sparse.append(r[2.0])
    s_a, p_a = np.array(soft), np.array(sparse)
    d = s_a - p_a; scale = float(s_a.mean()); rel = (s_a - p_a) / s_a
    return dict(observed=regime(d, scale), mean_delta=float(d.mean()),
                se=float(d.std(ddof=1) / np.sqrt(len(d))), margin=REL_MARGIN * scale,
                n_origins=len(d), sparsemax_lower=int((d > 0).sum()),
                mse_softmax=scale, mse_sparsemax=float(p_a.mean()),
                rel_mean=float(rel.mean()), rel_se=float(rel.std(ddof=1) / np.sqrt(len(rel))),
                mse_softmax_per_origin=[float(x) for x in s_a],
                mse_sparsemax_per_origin=[float(x) for x in p_a])

_T = None
def _init():
    global _T
    _T = load()

def _work(args):
    target, n, do_run = args
    X, y = cell(_T, target)
    if not origins(len(y), n): return None
    Xs = (X[:n] - X[:n].mean(0)) / (X[:n].std(0) + 1e-12)
    c = cf.coords(Xs, y[:n])
    pred = 'win' if (c['kfrac'] <= 0.15 and n <= 256) else 'tie'
    rec = dict(target=target, n_train=n, n_channels=int(X.shape[1]),
               r_eff=c['r_eff'], k_eff=c['k_eff'], kfrac=c['kfrac'], predicted=pred)
    if do_run: rec.update(run_cell(X, y, n))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predict-only', action='store_true'); ap.add_argument('--run', action='store_true')
    ap.add_argument('--all-targets', action='store_true', help='every series as a target')
    a = ap.parse_args()
    T = load()
    print(f'panel: {T.shape[0]} months x {T.shape[1]} series, {T.index[0].date()} to {T.index[-1].date()}')
    targets = list(T.columns) if a.all_targets else [t for t in TARGETS if t in T.columns]
    out = {}; t0 = time.time()
    print(f"\n{'target':<22}{'n':>5}{'r_eff':>7}{'k_eff':>7}{'kfrac':>8}{'PRED':>6}" +
          (f"{'OBS':>7}{'Delta':>11}{'2SE':>10}{'rel':>9}{'sp<so':>8}" if a.run else ''))
    jobs = [(t, n, a.run) for t in targets for n in NS]
    nproc = int(os.environ.get('NPROC', max(1, min((os.cpu_count() or 4) - 2, 32))))
    with ProcessPoolExecutor(max_workers=nproc, initializer=_init) as ex:
        for rec in ex.map(_work, jobs, chunksize=1):
            if rec is None: continue
            key = f"{rec['target']}_n{rec['n_train']}"; out[key] = rec
            line = (f"{rec['target']:<22}{rec['n_train']:>5}{rec['r_eff']:>7}{rec['k_eff']:>7}"
                    f"{rec['kfrac']:>8.3f}{rec['predicted'].upper():>6}")
            if a.run:
                line += (f"{rec['observed'].upper():>7}{rec['mean_delta']:>+11.5f}{2*rec['se']:>10.5f}"
                         f"{rec['rel_mean']*100:>+8.2f}%{rec['sparsemax_lower']:>5}/{rec['n_origins']:<2}")
            print(line, flush=True)

    if a.run:
        json.dump(out, open(ROOT / 'results' / 'fredmd.json', 'w'), indent=1)
        R = list(out.values())
        pw = [r for r in R if r['predicted'] == 'win']; pt = [r for r in R if r['predicted'] == 'tie']
        f = lambda S, k: sum(r['observed'] == k for r in S)
        print(f"\npredicted win ({len(pw)}): win {f(pw,'win')} tie {f(pw,'tie')} dense {f(pw,'dense')}")
        print(f"predicted tie ({len(pt)}): win {f(pt,'win')} tie {f(pt,'tie')} dense {f(pt,'dense')}")
        rel = np.array([r['rel_mean'] for r in R])
        print(f"relative gain of sparsemax over softmax: {rel.mean()*100:+.2f}%  "
              f"lower in {(rel>0).sum()}/{len(rel)} cells   ({time.time()-t0:.0f}s)")
        print('wrote results/fredmd.json')

if __name__ == '__main__': main()
