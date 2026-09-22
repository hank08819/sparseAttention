#!/usr/bin/env python3
"""Per-origin evaluation of the frozen rule on FRED-MD (Amendment 2).

The rule reads its coordinate off the training block in hand. A cell here holds
20 rolling origins, each with its own training block, and the coordinate moves
by a median of 0.226 across them - more than the 0.15 threshold itself. Labelling
all 20 origins with the first block's coordinate therefore measures noise. This
script applies the rule where it is defined: one prediction per origin, compared
with that origin's own paired difference.
"""
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'package' / 'misa_sparsity'))
import run_boundary_benchmarks as bb
import importlib.util
def _load(name, rel):
    sp = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
fm = _load('fm', 'scripts/run_fredmd.py')
cf = _load('cf', 'scripts/run_confirmatory.py')

_T = None
def _init():
    global _T
    _T = fm.load()

def _work(job):
    target, n = job
    X, y = fm.cell(_T, target)
    out = []
    for o in fm.origins(len(y), n):
        Xtr, ytr = X[o:o + n], y[o:o + n]
        Xte, yte = X[o + n:o + n + fm.TEST_LEN], y[o + n:o + n + fm.TEST_LEN]
        sd = Xtr.std(0); ok = sd > 1e-8 * max(1.0, float(np.abs(Xtr).mean()))
        if ok.sum() < 5: continue
        A0, B0 = Xtr[:, ok], Xte[:, ok]
        mu, s = A0.mean(0), A0.std(0)
        A, B = (A0 - mu) / s, (B0 - mu) / s
        ym, ys = ytr.mean(), ytr.std() + 1e-12
        a_y, b_y = (ytr - ym) / ys, (yte - ym) / ys
        c = cf.coords(A, a_y)                      # coordinate of THIS training block
        bb.SCORE_BOUND = fm.SCORE_BOUND
        r = {}
        for al in (1.0, 2.0):
            p, u, b, *_ = bb.fit(al, A, a_y)
            r[al] = float(np.mean((u * (B @ p) + b - b_y) ** 2))
        out.append(dict(target=target, n_train=n, origin=int(o),
                        r_eff=c['r_eff'], k_eff=c['k_eff'], kfrac=float(c['kfrac']),
                        predicted='win' if (c['kfrac'] <= 0.15 and n <= 256) else 'tie',
                        mse_softmax=r[1.0], mse_sparsemax=r[2.0],
                        rel=float((r[1.0] - r[2.0]) / r[1.0])))
    return out

def main():
    T = fm.load()
    jobs = [(t, n) for t in list(T.columns) for n in fm.NS]
    rows = []; t0 = time.time()
    nproc = int(os.environ.get('NPROC', 14))
    with ProcessPoolExecutor(max_workers=nproc, initializer=_init) as ex:
        for res in ex.map(_work, jobs, chunksize=2):
            rows.extend(res)
    json.dump(rows, open(ROOT / 'results' / 'fredmd_perorigin.json', 'w'))
    rel = np.array([r['rel'] for r in rows])
    pw = [r for r in rows if r['predicted'] == 'win']; pt = [r for r in rows if r['predicted'] == 'tie']
    f = lambda S: np.mean([r['rel'] for r in S])
    print(f"origins: {len(rows)}   ({time.time()-t0:.0f}s)")
    print(f"  predicted win {len(pw)}: mean gain {f(pw)*100:+.2f}%   lower in "
          f"{sum(r['rel']>0 for r in pw)}/{len(pw)}")
    print(f"  predicted tie {len(pt)}: mean gain {f(pt)*100:+.2f}%   lower in "
          f"{sum(r['rel']>0 for r in pt)}/{len(pt)}")
    print(f"  overall {rel.mean()*100:+.2f}%, lower in {(rel>0).sum()}/{len(rel)}")
    print('wrote results/fredmd_perorigin.json')

if __name__ == '__main__': main()
