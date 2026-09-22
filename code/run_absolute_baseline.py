#!/usr/bin/env python3
"""Per-origin training-mean baseline for the three real-data runs.

The predictors are fitted and scored on a target standardized by its own
training block, so the natural reference is the training-mean forecast: predict
the training mean for every test point. On the standardized scale that is the
constant zero, and its test MSE is mean(y_test_standardized^2). That number is
NOT one in general: the test block has its own variance and its own level, so
the baseline must be computed per origin rather than assumed.

This script recomputes it for every cell and origin of the three runs, using the
same panels, targets, windows and standardization as the runs themselves, and
writes both the raw standardized MSEs and the ratio to the baseline.
"""
from __future__ import annotations
import sys, json, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def crypto_baselines(kind):
    """kind: 'cross_section' or 'time_axis'."""
    xs = load('run_cross_section')
    if kind == 'time_axis':
        ta = load('run_time_axis_control'); cell_data = ta.cell_data_time
    else:
        cell_data = xs.cell_data
    out = {}
    for period in xs.PERIODS:
        M = xs.panel(period)
        if M.shape[1] < 20: continue
        for target in list(M.columns):
            X, y = cell_data(M, target)
            for n in xs.NS:
                origs = xs.origins(len(y), n)
                if not origs: continue
                base = []
                for o in origs:
                    ytr = y[o:o + n]; yte = y[o + n:o + n + xs.TEST_LEN]
                    ym, ys = ytr.mean(), ytr.std() + 1e-12
                    b_y = (yte - ym) / ys
                    base.append(float(np.mean(b_y ** 2)))
                out[f'{period}_{target}_n{n}'] = base
    return out


def fredmd_baselines():
    fr = load('run_fredmd')
    T = fr.load(); out = {}
    for tgt in list(T.columns):
        X, y = fr.cell(T, tgt)
        for n in fr.NS:
            origs = fr.origins(len(y), n) if hasattr(fr, 'origins') else None
            if origs is None:
                last = len(y) - n - fr.TEST_LEN
                if last < 0: continue
                step = max(1, last // (fr.N_ORIGIN - 1))
                origs = [min(i * step, last) for i in range(fr.N_ORIGIN)]
            if not origs: continue
            base = []
            for o in origs:
                ytr = y[o:o + n]; yte = y[o + n:o + n + fr.TEST_LEN]
                ym, ys = ytr.mean(), ytr.std() + 1e-12
                b_y = (yte - ym) / ys
                base.append(float(np.mean(b_y ** 2)))
            out[f'{tgt}_n{n}'] = base
    return out


def summarize(run_file, base, label):
    cells = json.load(open(ROOT / 'results' / run_file))
    S, Q, Bl = [], [], []
    for k, v in cells.items():
        if not isinstance(v, dict) or 'mse_softmax_per_origin' not in v: continue
        b = base.get(k)
        if b is None: continue
        m = min(len(b), len(v['mse_softmax_per_origin']))
        S += list(v['mse_softmax_per_origin'][:m])
        Q += list(v['mse_sparsemax_per_origin'][:m])
        Bl += list(b[:m])
    S, Q, Bl = np.array(S), np.array(Q), np.array(Bl)
    rs, rq = S / Bl, Q / Bl                    # ratio to that origin's own baseline
    both = (rs < 1) & (rq < 1)
    rel = (S - Q) / S
    d = dict(label=label, origins=int(len(S)),
             baseline_median=float(np.median(Bl)),
             baseline_p05=float(np.quantile(Bl, .05)), baseline_p95=float(np.quantile(Bl, .95)),
             soft_mse_median=float(np.median(S)), spar_mse_median=float(np.median(Q)),
             soft_ratio_median=float(np.median(rs)), spar_ratio_median=float(np.median(rq)),
             soft_beat=float((rs < 1).mean()), spar_beat=float((rq < 1).mean()),
             both=int(both.sum()), both_gain=float(rel[both].mean()),
             both_win=float((rel[both] > 0).mean()))
    print(f"=== {label}: {d['origins']} origins ===")
    print(f"  baseline MSE per origin: median {d['baseline_median']:.3f} "
          f"(5-95% {d['baseline_p05']:.3f}-{d['baseline_p95']:.3f})  <- not 1")
    print(f"  raw standardized MSE  median: softmax {d['soft_mse_median']:.3f}, sparsemax {d['spar_mse_median']:.3f}")
    print(f"  MSE / own baseline    median: softmax {d['soft_ratio_median']:.3f}, sparsemax {d['spar_ratio_median']:.3f}")
    print(f"  origins below own baseline: softmax {d['soft_beat']:.1%}, sparsemax {d['spar_beat']:.1%}")
    print(f"  both below baseline: {d['both']} ({d['both']/d['origins']:.1%}), gain {d['both_gain']*100:+.2f}%, "
          f"sparsemax lower {d['both_win']:.1%}")
    return d


if __name__ == '__main__':
    out = []
    out.append(summarize('cross_section.json', crypto_baselines('cross_section'), 'cryptocurrency cross-section'))
    out.append(summarize('time_axis.json', crypto_baselines('time_axis'), 'cryptocurrency time axis, matched'))
    out.append(summarize('fredmd.json', fredmd_baselines(), 'FRED-MD cross-section'))
    json.dump(out, open(ROOT / 'results' / 'absolute_baseline.json', 'w'), indent=1)
    print('\nwrote results/absolute_baseline.json')
