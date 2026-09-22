#!/usr/bin/env python3
"""
run_zeroed_position_test.py -- perturb only the positions sparsemax set to zero.

PRE-REGISTERED MARGIN (written before any model is loaded):
  Equivalence margin for functional sensitivity to the zeroed positions:
      Psi_zero / Psi_signal  <  0.01
  i.e. resampling only the inputs sparsemax excluded may move the prediction by
  less than one percent of what resampling the whole real context moves it.
  Reported as a TOST: equivalence declared only if the 90% interval of
  Psi_zero/Psi_signal (over the 150 paired runs) lies inside [0, 0.01].
  The same positions are perturbed in the softmax twin of the same setting and
  seed, so the two normalizations are compared on identical inputs.

For each frozen (setting, seed) pair:
  1. take the test windows and the sparsemax model's scale-1 attention a1;
  2. Z = positions with a1 == 0 exactly (per window);
  3. resample those positions K times from the training marginal, hold the rest
     fixed, measure Var_z f(x, z) for the sparsemax model and for the softmax
     twin on the SAME Z;
  4. Psi_signal as in the mechanism experiment (whole real context permuted).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'package'))
from misa_sparsity import MISABiGRU
import importlib.util
_s = importlib.util.spec_from_file_location("fmm", ROOT / 'scripts' / 'run_full_misa_mechanism.py')
fmm = importlib.util.module_from_spec(_s); _s.loader.exec_module(fmm)

MARGIN = 0.01          # pre-registered, see docstring
K = 16

def load(ckpt, F, device):
    st = torch.load(ckpt, map_location=device)
    m = MISABiGRU(feat_dim=F, hidden=fmm.CFG['hidden'], n_heads=fmm.CFG['n_heads'], dropout=fmm.CFG['dropout'],
                  lambda_gate=fmm.CFG['lambda_gate'], use_multiscale=True, use_cross_asset=True,
                  use_sparsemax=(st['normalization'] == 'sparsemax')).to(device)
    m.load_state_dict(st['state_dict']); m.eval(); return m

def predict(m, X, Xca, device, batch=1024):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            p, _, _ = m(torch.tensor(X[i:i+batch], device=device), torch.tensor(Xca[i:i+batch], device=device)); out.append(p.cpu().numpy())
    return np.concatenate(out)

def zero_mask(m, X, Xca, device, strict):
    """Positions the model assigns exactly zero weight.

    strict=False: zero at scale 1 only (the first version of this test).
    strict=True : zero on every path -- scale-1 weight zero, AND the q=3 block
                  containing t has zero weight (or scale 2 is fused out), AND the
                  q=6 block has zero weight (or scale 3 is fused out).
    """
    with torch.no_grad():
        d = m.forward_diagnostics(torch.tensor(X, device=device), torch.tensor(Xca, device=device))
    a1 = d['a1'].cpu().numpy(); N, T = a1.shape
    z = a1 <= 1e-9
    if not strict or d['a2'] is None:
        return z, dict(T=T, a2=None)
    a2 = d['a2'].cpu().numpy(); a3 = d['a3'].cpu().numpy(); sw = d['scale_weights'].cpu().numpy()   # (N,3)
    t = np.arange(T)
    b2 = np.minimum(t // 3, a2.shape[1] - 1); b3 = np.minimum(t // 6, a3.shape[1] - 1)
    z2 = (a2[:, b2] <= 1e-9) | (sw[:, 1:2] <= 1e-9)
    z3 = (a3[:, b3] <= 1e-9) | (sw[:, 2:3] <= 1e-9)
    return z & z2 & z3, dict(T=T, a2=a2.shape, a3=a3.shape, sw_zero_frac=[float((sw[:, i] <= 1e-9).mean()) for i in range(3)])

def var_under(m, X, Xca, mask, mu, sd, device, seed):
    """Var over K resamples of the masked positions only."""
    rng = np.random.default_rng(seed); preds = []
    for _ in range(K):
        Xp = X.copy(); z = rng.normal(mu, sd, size=X.shape).astype(np.float32)
        Xp[mask] = z[mask]; preds.append(predict(m, Xp, Xca, device))
    P = np.stack(preds); return float(np.var(P, axis=0, ddof=1).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True); ap.add_argument('--ckpt-dirs', nargs='+', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu'); ap.add_argument('--out-tag', default='zeroed')
    ap.add_argument('--strict', action='store_true', help='zero on all three attention paths, not scale 1 only')
    a = ap.parse_args(); fmm.DATA = Path(a.data_dir); dev = a.device
    ckpts = sorted(p for d in a.ckpt_dirs for p in Path(d).glob('*_sparsemax_*.pt'))
    out = ROOT / 'results' / f'zeroed_position_{a.out_tag}.json'; rows = []
    print(f'{len(ckpts)} sparsemax checkpoints; margin={MARGIN}', flush=True)
    for ck in ckpts:
        st = torch.load(ck, map_location='cpu'); setting, seed, r = st['setting'], st['seed'], st['r']
        twin = ck.with_name(ck.name.replace('_sparsemax_', '_softmax_'))
        if not twin.exists(): print('no twin for', ck.name); continue
        base = fmm.prepare(setting); sp = fmm.split_with_fixed_noise(base, r, 100000 + sum(map(ord, setting)) + 100 * r)
        Xte, Xca, mu, sd = sp['te'][0], sp['te'][1], sp['mu'], sp['sd']
        ms = load(ck, sp['F'], dev); mf = load(twin, sp['F'], dev)
        zero, info = zero_mask(ms, Xte, Xca, dev, a.strict); zero_frac = float(zero.mean())
        if not rows: print('shapes:', info, flush=True)
        mask = np.repeat(zero[:, :, None], Xte.shape[-1], axis=2)      # (N, T, F)
        psi_zero_sp = var_under(ms, Xte, Xca, mask, mu, sd, dev, seed + 1)
        psi_zero_sf = var_under(mf, Xte, Xca, mask, mu, sd, dev, seed + 1)
        full = np.ones_like(mask)
        psi_sig_sp = var_under(ms, Xte, Xca, full, mu, sd, dev, seed + 2)
        psi_sig_sf = var_under(mf, Xte, Xca, full, mu, sd, dev, seed + 2)
        rows.append(dict(setting=setting, seed=seed, zero_fraction=zero_frac,
                         psi_zero_sparsemax=psi_zero_sp, psi_zero_softmax=psi_zero_sf,
                         psi_signal_sparsemax=psi_sig_sp, psi_signal_softmax=psi_sig_sf,
                         ratio_sparsemax=psi_zero_sp / max(psi_sig_sp, 1e-30), ratio_softmax=psi_zero_sf / max(psi_sig_sf, 1e-30)))
        out.write_text(json.dumps(rows, indent=1))
        print(f"  {setting} seed={seed} zeros={zero_frac:.3f}  ratio sp={rows[-1]['ratio_sparsemax']:.2e}  sf={rows[-1]['ratio_softmax']:.2e}", flush=True)
    R = np.array([[x['ratio_sparsemax'], x['ratio_softmax']] for x in rows])
    for j, nm in enumerate(['sparsemax', 'softmax']):
        v = R[:, j]; lo, hi = np.percentile(v, [5, 95])
        print(f"{nm}: median ratio {np.median(v):.2e}  90% interval [{lo:.2e}, {hi:.2e}]  equivalent(<{MARGIN})={'YES' if hi < MARGIN else 'NO'}")

if __name__ == '__main__': main()
