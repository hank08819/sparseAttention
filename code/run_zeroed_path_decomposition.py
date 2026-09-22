#!/usr/bin/env python3
"""Where does the influence of a zeroed position enter the model?

Follow-up to run_zeroed_position_test.py. Same frozen models, same positions
(zero at scale 1), same K resamples; four forward conditions for the sparsemax
model, so the residual sensitivity can be attributed to a path:

  end-to-end        everything recomputes                      (the main test)
  frozen-weights    attention weights at all three scales, the
                    scale-fusion weights and the target/cross
                    gate are held at their unperturbed values;
                    only representations recompute
  scale1-only       coarse scales removed (fusion weight [1,0,0]),
                    weights recompute
  frozen+scale1     both

end-to-end vs scale1-only measures the coarse-scale path; end-to-end vs
frozen-weights measures the weight-recomputation path; what is left under
frozen+scale1 is transport by the scale-1 BiGRU itself.
Writes results/zeroed_path_decomposition.json.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'package'))
from misa_sparsity import MISABiGRU
from misa_sparsity.sparsemax import sparsemax
from misa_sparsity.model import _avg_pool
import importlib.util
_s = importlib.util.spec_from_file_location("fmm", ROOT / 'scripts' / 'run_full_misa_mechanism.py')
fmm = importlib.util.module_from_spec(_s); _s.loader.exec_module(fmm)
_z = importlib.util.spec_from_file_location("zp", ROOT / 'scripts' / 'run_zeroed_position_test.py')
zp = importlib.util.module_from_spec(_z); _z.loader.exec_module(zp)
K = 16
CONDS = ['endtoend', 'frozen_weights', 'scale1_only', 'frozen_and_scale1']

def enc_ctx(enc, x, a_override=None):
    H, _ = enc.gru(x); H = enc.drop(H)
    if a_override is None:
        e = enc.v(torch.tanh(enc.W_a(H))).squeeze(-1)
        a = sparsemax(e, dim=-1) if enc.use_sparsemax else torch.softmax(e, dim=-1)
    else:
        a = a_override
    return torch.bmm(a.unsqueeze(1), H).squeeze(1), a

def fwd(m, x, xca, frz=None, scale1=False):
    """Forward with optional frozen weights and/or coarse scales removed."""
    c1, a1 = enc_ctx(m.enc1, x, None if frz is None else frz['a1'])
    if m.use_multiscale and not scale1:
        c2, a2 = enc_ctx(m.enc2, _avg_pool(x, 3), None if frz is None else frz['a2'])
        c3, a3 = enc_ctx(m.enc3, _avg_pool(x, 6), None if frz is None else frz['a3'])
        stack = torch.stack([c1, c2, c3], dim=1)
        if frz is None:
            s = m.scale_score(stack).squeeze(-1)
            sw = sparsemax(s, dim=-1) if m._use_sparsemax else torch.softmax(s, dim=-1)
        else:
            sw = frz['sw']
        fused = (stack * sw.unsqueeze(-1)).sum(dim=1)
    else:
        fused = c1
        a2 = a3 = None
        sw = None
    if m.use_cross_asset:
        H_ca, _ = m.ca_gru(xca); H_ca = m.ca_drop(H_ca)
        cross, _ = m.cross_attn(fused.unsqueeze(1), H_ca, H_ca); cross = cross.squeeze(1)
        gate = torch.sigmoid(m.gate_fc(torch.cat([fused, cross], dim=-1))) if frz is None else frz['gate']
        out = gate * fused + (1.0 - gate) * cross
    else:
        gate = None; out = fused
    pred = m.fc(m.drop_out(m.ln(out))).squeeze(-1)
    return pred, dict(a1=a1, a2=a2, a3=a3, sw=sw, gate=gate)

def batched(m, X, Xca, dev, frz_full=None, scale1=False, batch=1024, want_frz=False):
    outs, frz_parts = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], device=dev); cb = torch.tensor(Xca[i:i + batch], device=dev)
            f = None if frz_full is None else {k: (None if v is None else v[i:i + batch]) for k, v in frz_full.items()}
            p, d = fwd(m, xb, cb, f, scale1)
            outs.append(p.cpu().numpy())
            if want_frz: frz_parts.append(d)
    pred = np.concatenate(outs)
    if not want_frz: return pred, None
    keys = frz_parts[0].keys()
    frz = {k: (None if frz_parts[0][k] is None else torch.cat([p[k] for p in frz_parts], 0)) for k in keys}
    return pred, frz

def psi(m, X, Xca, mask, mu, sd, dev, seed, frz=None, scale1=False):
    rng = np.random.default_rng(seed); P = []
    for _ in range(K):
        Xp = X.copy(); z = rng.normal(mu, sd, size=X.shape).astype(np.float32); Xp[mask] = z[mask]
        p, _ = batched(m, Xp, Xca, dev, frz, scale1); P.append(p)
    return float(np.var(np.stack(P), axis=0, ddof=1).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True); ap.add_argument('--ckpt-dirs', nargs='+', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out-tag', default='path_decomposition')
    a = ap.parse_args(); fmm.DATA = Path(a.data_dir); dev = a.device
    ckpts = sorted(p for d in a.ckpt_dirs for p in Path(d).glob('*_sparsemax_*.pt'))
    out = ROOT / 'results' / f'zeroed_{a.out_tag}.json'; rows = []
    print(f'{len(ckpts)} sparsemax checkpoints', flush=True)
    for ck in ckpts:
        st = torch.load(ck, map_location='cpu'); setting, seed, r = st['setting'], st['seed'], st['r']
        base = fmm.prepare(setting); sp = fmm.split_with_fixed_noise(base, r, 100000 + sum(map(ord, setting)) + 100 * r)
        Xte, Xca, mu, sd = sp['te'][0], sp['te'][1], sp['mu'], sp['sd']
        m = zp.load(ck, sp['F'], dev)
        zero, _ = zp.zero_mask(m, Xte, Xca, dev, False)
        mask = np.repeat(zero[:, :, None], Xte.shape[-1], axis=2)
        _, frz = batched(m, Xte, Xca, dev, want_frz=True)          # unperturbed weights
        rec = dict(setting=setting, seed=seed, zero_fraction=float(zero.mean()))
        rec['psi_signal'] = psi(m, Xte, Xca, np.ones_like(mask), mu, sd, dev, seed + 2)
        for cond in CONDS:
            rec['psi_' + cond] = psi(m, Xte, Xca, mask, mu, sd, dev, seed + 1,
                                     frz=frz if 'frozen' in cond else None, scale1=('scale1' in cond))
            rec['ratio_' + cond] = rec['psi_' + cond] / max(rec['psi_signal'], 1e-30)
        rows.append(rec); out.write_text(json.dumps(rows, indent=1))
        print(f"  {setting} seed={seed} zeros={rec['zero_fraction']:.3f} " +
              "  ".join(f"{c}={rec['ratio_'+c]:.2e}" for c in CONDS), flush=True)
    R = {c: np.array([x['ratio_' + c] for x in rows]) for c in CONDS}
    print()
    for c in CONDS:
        v = R[c]; print(f"{c:<18} median {np.median(v):.3e}   90% [{np.percentile(v,5):.3e}, {np.percentile(v,95):.3e}]")
    base = R['endtoend']
    for c in CONDS[1:]:
        red = 1 - R[c] / np.maximum(base, 1e-30)
        print(f"{c:<18} removes {np.median(red)*100:.1f}% of the end-to-end sensitivity (median over pairs)")

if __name__ == '__main__': main()
