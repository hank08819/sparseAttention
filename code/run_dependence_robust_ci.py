#!/usr/bin/env python3
"""Dependence-robust intervals for the cross-sectional and FRED-MD tests.

The naive interval across cells treats cells as independent. They are not: every
cell in a panel is scored on the same calendar origins, the rolling training
blocks overlap, and the target series are cross-correlated. This script reports
three intervals for the same estimand (the mean relative gain of exact zeros):

  naive       across cells, cells treated as independent
  cluster     cells clustered by target series, resampled with replacement
  block       moving-block bootstrap over the shared rolling origins, which
              respects both the overlap between windows and the cross-sectional
              dependence between series (the same origins are resampled for
              every cell at once)

The block bootstrap is the one quoted in the paper.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B = 5000
BLOCK = 4
ALPHA = 0.10
RNG = np.random.default_rng(20260920)


def cells(name):
    d = json.load(open(ROOT / 'results' / name))
    return [v for v in d.values() if isinstance(v, dict) and 'rel_mean' in v]


def per_origin(C):
    """(cells x origins) matrix of per-origin relative gains."""
    n_o = min(len(c['mse_softmax_per_origin']) for c in C)
    M = np.empty((len(C), n_o))
    for i, c in enumerate(C):
        a = np.array(c['mse_softmax_per_origin'][:n_o])
        b = np.array(c['mse_sparsemax_per_origin'][:n_o])
        M[i] = (a - b) / a
    return M


def naive_ci(x):
    m = x.mean(); se = x.std(ddof=1) / math.sqrt(len(x))
    z = 1.645
    return m, m - z * se, m + z * se


def cluster_ci(C, x):
    keys = [c.get('target', c.get('period', '')) for c in C]
    uniq = sorted(set(keys))
    idx = {u: np.array([i for i, k in enumerate(keys) if k == u]) for u in uniq}
    stats = np.empty(B)
    for b in range(B):
        pick = RNG.choice(len(uniq), size=len(uniq), replace=True)
        sel = np.concatenate([idx[uniq[p]] for p in pick])
        stats[b] = x[sel].mean()
    return np.quantile(stats, [ALPHA / 2, 1 - ALPHA / 2])


def block_ci(M):
    """Moving-block bootstrap over the shared origin axis."""
    n_c, n_o = M.shape
    n_blocks = int(math.ceil(n_o / BLOCK))
    starts = np.arange(n_o - BLOCK + 1)
    stats = np.empty(B)
    for b in range(B):
        s = RNG.choice(starts, size=n_blocks, replace=True)
        cols = np.concatenate([np.arange(t, t + BLOCK) for t in s])[:n_o]
        stats[b] = M[:, cols].mean()
    return np.quantile(stats, [ALPHA / 2, 1 - ALPHA / 2])


def report(name, label):
    C = cells(name)
    x = np.array([c['rel_mean'] for c in C])
    M = per_origin(C)
    m, lo_n, hi_n = naive_ci(x)
    lo_c, hi_c = cluster_ci(C, x)
    lo_b, hi_b = block_ci(M)
    print(f'\n=== {label} ===')
    print(f'cells {len(C)}, origins per cell {M.shape[1]}, '
          f'mean relative gain {m*100:.2f}%')
    print(f'  naive    90% CI [{lo_n*100:.2f}, {hi_n*100:.2f}]  width {(hi_n-lo_n)*100:.2f}')
    print(f'  cluster  90% CI [{lo_c*100:.2f}, {hi_c*100:.2f}]  width {(hi_c-lo_c)*100:.2f}')
    print(f'  block    90% CI [{lo_b*100:.2f}, {hi_b*100:.2f}]  width {(hi_b-lo_b)*100:.2f}')
    return dict(label=label, cells=len(C), mean=float(m),
                naive=[float(lo_n), float(hi_n)],
                cluster=[float(lo_c), float(hi_c)],
                block=[float(lo_b), float(hi_b)])


if __name__ == '__main__':
    out = []
    for f, lab in (('cross_section.json', 'cryptocurrency cross-section'),
                   ('fredmd.json', 'FRED-MD cross-section'),
                   ('time_axis.json', 'cryptocurrency time axis, matched')):
        if (ROOT / 'results' / f).exists():
            out.append(report(f, lab))
    json.dump(out, open(ROOT / 'results' / 'dependence_robust_ci.json', 'w'), indent=1)
    print('\nwrote results/dependence_robust_ci.json')
