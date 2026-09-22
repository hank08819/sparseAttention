#!/usr/bin/env python3
"""Size-matched control for the zeroed-position test.

The registered statistic Psi_zero / Psi_signal compares a perturbation of the
positions sparsemax zeroed (a median 16% of the window) against a perturbation
of the whole window, so its scale depends on how many positions were zeroed.
This script adds the control that removes that dependence: in the same frozen
model, perturb an equal number of positions drawn uniformly at random per
window, with everything else identical.

  Psi_zero / Psi_random  <  1   means the zeroed positions matter less to the
                                prediction than an equally large random set.

Writes results/zeroed_sizematched.json.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'package'))
import importlib.util
_s = importlib.util.spec_from_file_location("fmm", ROOT / 'scripts' / 'run_full_misa_mechanism.py')
fmm = importlib.util.module_from_spec(_s); _s.loader.exec_module(fmm)
_z = importlib.util.spec_from_file_location("zp", ROOT / 'scripts' / 'run_zeroed_position_test.py')
zp = importlib.util.module_from_spec(_z); _z.loader.exec_module(zp)
K = zp.K

def random_mask_like(zero, rng):
    """Per window, the same number of positions, drawn uniformly at random."""
    N, T = zero.shape; out = np.zeros_like(zero)
    counts = zero.sum(1)
    for i in range(N):
        c = int(counts[i])
        if c: out[i, rng.choice(T, size=c, replace=False)] = True
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True); ap.add_argument('--ckpt-dirs', nargs='+', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args(); fmm.DATA = Path(a.data_dir); dev = a.device
    ckpts = sorted(p for d in a.ckpt_dirs for p in Path(d).glob('*_sparsemax_*.pt'))
    out = ROOT / 'results' / 'zeroed_sizematched.json'; rows = []
    print(f'{len(ckpts)} sparsemax checkpoints', flush=True)
    for ck in ckpts:
        st = torch.load(ck, map_location='cpu'); setting, seed, r = st['setting'], st['seed'], st['r']
        twin = ck.with_name(ck.name.replace('_sparsemax_', '_softmax_'))
        if not twin.exists(): continue
        base = fmm.prepare(setting); sp = fmm.split_with_fixed_noise(base, r, 100000 + sum(map(ord, setting)) + 100 * r)
        Xte, Xca, mu, sd = sp['te'][0], sp['te'][1], sp['mu'], sp['sd']
        ms = zp.load(ck, sp['F'], dev); mf = zp.load(twin, sp['F'], dev)
        zero, _ = zp.zero_mask(ms, Xte, Xca, dev, False)
        rnd = random_mask_like(zero, np.random.default_rng(7000 + seed))
        m_zero = np.repeat(zero[:, :, None], Xte.shape[-1], axis=2)
        m_rand = np.repeat(rnd[:, :, None], Xte.shape[-1], axis=2)
        rec = dict(setting=setting, seed=seed, zero_fraction=float(zero.mean()))
        for nm, mdl in (('sparsemax', ms), ('softmax', mf)):
            rec[f'psi_zero_{nm}'] = zp.var_under(mdl, Xte, Xca, m_zero, mu, sd, dev, seed + 1)
            rec[f'psi_rand_{nm}'] = zp.var_under(mdl, Xte, Xca, m_rand, mu, sd, dev, seed + 3)
            rec[f'ratio_{nm}'] = rec[f'psi_zero_{nm}'] / max(rec[f'psi_rand_{nm}'], 1e-30)
        rows.append(rec); out.write_text(json.dumps(rows, indent=1))
        print(f"  {setting} seed={seed} f={rec['zero_fraction']:.3f}  zero/random: sparsemax {rec['ratio_sparsemax']:.3f}  softmax {rec['ratio_softmax']:.3f}", flush=True)
    R = {k: np.array([x['ratio_' + k] for x in rows]) for k in ('sparsemax', 'softmax')}
    for k, v in R.items():
        print(f"{k:<10} zero/random median {np.median(v):.4f}  90% [{np.percentile(v,5):.4f},{np.percentile(v,95):.4f}]  below 1 in {(v<1).sum()}/{len(v)}")
    from scipy.stats import wilcoxon
    print(f"sparsemax lower than softmax in {(R['sparsemax']<R['softmax']).sum()}/{len(rows)}, Wilcoxon p={wilcoxon(R['softmax'],R['sparsemax'])[1]:.2e}")

if __name__ == '__main__': main()
