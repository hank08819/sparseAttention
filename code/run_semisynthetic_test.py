#!/usr/bin/env python3
"""Pre-registered discriminating test -- see PREREGISTRATION_semisynthetic.md
(registered 2026-09-19 14:49 UTC, before this script was run).

Real target and real predictors from the semiconductor monthly panel; m
independent Gaussian nuisance channels appended; Definition-1 probe with free
scores; softmax vs sparsemax paired over 20 noise draws. Regime rule as Fig. 2.
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'package'))
import importlib.util
_b = importlib.util.spec_from_file_location("bb", ROOT / 'package/misa_sparsity/run_boundary_benchmarks.py')
bb = importlib.util.module_from_spec(_b); _b.loader.exec_module(bb)

FLOOR = 1e-4; N_TRAIN = 128; H = 3; SEEDS = range(20)

def load_panel(path):
    df = pd.read_csv(path).dropna()
    cols = [c for c in df.columns if c != 'month']
    V = df[cols].values.astype(np.float64)
    tgt = cols.index('ship')
    # Amendment 1: inputs are one-month log differences (stationary), not levels.
    L = np.log(np.maximum(V, 1e-9))
    D = L[1:] - L[:-1]                                   # (T-1, C) growth rates at each origin
    y = L[1 + H:, tgt] - L[1:-H, tgt]                    # 3-month-ahead shipment growth from origin
    X = D[:-H]
    return X, y, cols

def regime(d):
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    if m > 2 * se and m > FLOOR: return 'win'
    if m < -2 * se and m < -FLOOR: return 'lose'
    return 'tie'

def run_cell(X, y, m_noise, sigma_z, B, seeds):
    bb.SCORE_BOUND = B
    deltas = []; zeros = []
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        Xr = X.copy()
        if m_noise:
            Z = rng.normal(0.0, sigma_z, size=(len(X), m_noise))
            Xr = np.concatenate([Xr, Z], axis=1)
        Xtr, ytr = Xr[:N_TRAIN], y[:N_TRAIN]; Xte, yte = Xr[N_TRAIN:], y[N_TRAIN:]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
        # real channels are standardized; appended noise keeps its own scale sigma_z
        # relative to unit-variance real inputs, which is what sigma_z means in the map
        Xtr_s = Xtr.copy(); Xte_s = Xte.copy()
        nreal = X.shape[1]
        Xtr_s[:, :nreal] = (Xtr[:, :nreal] - mu[:nreal]) / sd[:nreal]; Xte_s[:, :nreal] = (Xte[:, :nreal] - mu[:nreal]) / sd[:nreal]
        ym, ys = ytr.mean(), ytr.std() + 1e-12
        ytr_s = (ytr - ym) / ys; yte_s = (yte - ym) / ys
        out = {}
        for a in (1.0, 2.0):
            p, u, b, ok, nit, _ = bb.fit(a, Xtr_s, ytr_s)
            pred = u * (Xte_s @ p) + b
            out[a] = float(np.mean((pred - yte_s) ** 2))
            if a == 2.0: zeros.append(float((p <= 1e-9).mean()))
        deltas.append(out[1.0] - out[2.0])
    d = np.array(deltas)
    return dict(mean_delta=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))),
                sparsemax_lower=int((d > 0).sum()), n_seeds=len(d), regime=regime(d),
                zero_frac_sparsemax=float(np.mean(zeros)))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--panel', required=True); a = ap.parse_args()
    X, y, cols = load_panel(a.panel)
    print(f'panel: {X.shape[0]} origins x {X.shape[1]} real channels; n_train={N_TRAIN}, test={X.shape[0]-N_TRAIN}\n')
    cells = [('P1', 19, 2.0, 'WIN'), ('P2', 19, 1.0, 'WIN'), ('P3', 19, 0.25, 'TIE'), ('P4', 0, 0.0, 'TIE')]
    results = {}
    for B in (8.0, 2.0):
        tag = 'registered (|s|<=8)' if B == 8.0 else 'secondary  (|s|<=2)'
        print(f'=== {tag} ===')
        print(f"{'':<4}{'m':>4}{'sigma_z':>9}{'predicted':>11}{'observed':>10}{'mean Delta':>12}{'2SE':>10}{'sp<soft':>9}{'zeros':>7}")
        for name, m, s, pred in cells:
            r = run_cell(X, y, m, s, B, SEEDS)
            results[f'{name}_B{int(B)}'] = dict(m=m, sigma_z=s, predicted=pred, **r)
            flag = '' if B != 8.0 else ('  <- HELD' if r['regime'].upper() == pred else '  <- FALSIFIED')
            print(f"{name:<4}{m:>4}{s:>9.2f}{pred:>11}{r['regime'].upper():>10}{r['mean_delta']:>+12.5f}{2*r['se']:>10.5f}{r['sparsemax_lower']:>6}/{r['n_seeds']:<2}{r['zero_frac_sparsemax']:>7.2f}{flag}")
        print()
    json.dump(results, open(ROOT / 'results' / 'semisynthetic_test.json', 'w'), indent=1)
    print('wrote results/semisynthetic_test.json')

if __name__ == '__main__': main()
