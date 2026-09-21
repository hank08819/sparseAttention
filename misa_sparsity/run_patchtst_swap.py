#!/usr/bin/env python3
"""
run_patchtst_swap.py -- normalization swap inside a Transformer with unbounded logits.

Every earlier swap in the paper sits either in the linear probe (scores clipped
to [-B, B]) or in the BiGRU-based MISA model. This script puts the same swap
inside PatchTST (Nie et al., 2023), whose attention logits q.k/sqrt(d) are
unbounded, on the same 15 cryptocurrency settings and the same ten seeds as the
frozen-model mechanism experiment. Only the attention normalization changes:

    softmax  |  alpha-entmax (alpha = 1.5)  |  sparsemax (alpha = 2)

Data pipeline, splits, features, optimizer, schedule and seeds are shared with
run_full_misa_mechanism.py, so the result is directly comparable to Fig. 4.

Usage:
  python run_patchtst_swap.py --data-dir DIR --settings BTC_P2 ... --seeds 42 ... \
                              --norms softmax entmax1.5 sparsemax --out-tag NAME
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from misa_sparsity import Trainer, build_features, get_device
from misa_sparsity.sparsemax import sparsemax as torch_sparsemax
import importlib.util
_spec = importlib.util.spec_from_file_location("fmm", ROOT / "scripts" / "run_full_misa_mechanism.py")
fmm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fmm)


# --------------------------------------------------------------------------- #
# alpha-entmax in torch (bisection on the threshold; gradient through the
# implicit function, J = diag(u) - u u^T / sum(u), u = p^(2-alpha))
# --------------------------------------------------------------------------- #
class EntmaxAlpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, alpha, iters=50):
        a1 = alpha - 1.0
        z = a1 * s
        inv = 1.0 / a1
        hi = z.max(dim=-1, keepdim=True).values
        lo = hi - 1.0
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            tot = torch.clamp(z - mid, min=0).pow(inv).sum(dim=-1, keepdim=True)
            lo = torch.where(tot > 1.0, mid, lo)
            hi = torch.where(tot > 1.0, hi, mid)
        p = torch.clamp(z - 0.5 * (lo + hi), min=0).pow(inv)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        ctx.save_for_backward(p); ctx.alpha = alpha
        return p

    @staticmethod
    def backward(ctx, g):
        (p,) = ctx.saved_tensors
        u = torch.where(p > 0, p.pow(2.0 - ctx.alpha), torch.zeros_like(p))
        gu = (u * g).sum(dim=-1, keepdim=True) / u.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return u * (g - gu), None, None


def normalize(scores, norm):
    if norm == 'softmax':
        return F.softmax(scores, dim=-1)
    if norm == 'sparsemax':
        return torch_sparsemax(scores, dim=-1)
    if norm.startswith('entmax'):
        return EntmaxAlpha.apply(scores, float(norm[6:]))
    raise ValueError(norm)


# --------------------------------------------------------------------------- #
# Transformer encoder layer with a swappable attention normalization.
# Mirrors nn.TransformerEncoderLayer(norm_first=True, batch_first=True).
# --------------------------------------------------------------------------- #
class SwapAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, norm):
        super().__init__()
        self.h = n_heads; self.d = d_model // n_heads; self.norm = norm
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.last_weights = None

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        scores = (q @ k.transpose(-2, -1)) / (self.d ** 0.5)      # unbounded logits
        w = normalize(scores, self.norm)
        self.last_weights = w.detach()
        y = (self.drop(w) @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(y)


class SwapEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, norm):
        super().__init__()
        self.attn = SwapAttention(d_model, n_heads, dropout, norm)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))
        self.n1 = nn.LayerNorm(d_model); self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.n1(x)))
        x = x + self.drop(self.ff(self.n2(x)))
        return x


class PatchTSTSwap(nn.Module):
    """PatchTST of run_transformer_dm.py with the normalization made swappable."""
    def __init__(self, feat_dim=7, seq_len=30, patch_len=8, stride=4,
                 d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.1, norm='softmax'):
        super().__init__()
        self.patch_len = patch_len; self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len, d_model)
        self.layers = nn.ModuleList([SwapEncoderLayer(d_model, n_heads, d_ff, dropout, norm)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model); self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim * self.num_patches * d_model, 1)

    def forward(self, x_tgt, x_ca=None):
        B, T, Fd = x_tgt.shape
        x = x_tgt.permute(0, 2, 1)
        patches = torch.stack([x[:, :, i * self.stride:i * self.stride + self.patch_len]
                               for i in range(self.num_patches)], dim=2)
        x = self.embed(patches.reshape(B * Fd, self.num_patches, self.patch_len))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x).reshape(B, -1)
        return self.head(self.drop(x)).squeeze(-1), None, torch.tensor(0.0, device=x_tgt.device)

    def attention_stats(self):
        """Fraction of attention weights that are exactly zero, over all heads/layers."""
        zs, tot = 0, 0
        for l in self.layers:
            w = l.attn.last_weights
            if w is not None:
                zs += int((w <= 1e-9).sum()); tot += w.numel()
        return zs / max(tot, 1)


# --------------------------------------------------------------------------- #
def train_one(setting, norm, seed, device):
    base = fmm.prepare(setting)
    nt, nv = base['nt'], base['nv']
    X, Xca, y = base['X'], base['Xca'], base['y']
    tr = (X[:nt], Xca[:nt], y[:nt]); va = (X[nt:nt+nv], Xca[nt:nt+nv], y[nt:nt+nv])
    te = (X[nt+nv:], Xca[nt+nv:], y[nt+nv:]); pte = base['pc'][nt+nv:]
    torch.manual_seed(seed); np.random.seed(seed)
    model = PatchTSTSwap(feat_dim=base['F'], seq_len=fmm.CFG['window'], norm=norm)
    T = Trainer(model, lr=fmm.CFG['lr'], epochs=fmm.CFG['epochs'], patience=fmm.CFG['patience'],
                warmup=fmm.CFG['warmup'], batch_size=fmm.CFG['batch_size'], device=device, verbose=False)
    t0 = time.time(); T.fit(*tr, *va); secs = time.time() - t0
    m = T.evaluate(*te, pte, price_range=base['price_range'])
    model.eval()
    with torch.no_grad():
        dev = next(model.parameters()).device
        model(torch.tensor(te[0][:512], device=dev), torch.tensor(te[1][:512], device=dev))
    return dict(setting=setting, normalization=norm, seed=seed, n_test=len(te[2]),
                nmse=float(m['NMSE']), rmse=float(m['RMSE']), r2=float(m['R2']),
                zero_fraction=model.attention_stats(), train_seconds=secs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--settings', nargs='+', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(42, 52)))
    ap.add_argument('--norms', nargs='+', default=['softmax', 'entmax1.5', 'sparsemax'])
    ap.add_argument('--device', default='auto')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out-tag', default='patchtst')
    a = ap.parse_args()
    fmm.DATA = Path(a.data_dir)
    torch.set_num_threads(a.threads)
    device = get_device(a.device)
    out = ROOT / 'results' / f'patchtst_swap_{a.out_tag}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.load(open(out)) if out.exists() else []
    done = {(r['setting'], r['normalization'], r['seed']) for r in rows}
    print(f'device={device} settings={a.settings} seeds={a.seeds} norms={a.norms}', flush=True)
    for s in a.settings:
        for seed in a.seeds:
            for norm in a.norms:
                if (s, norm, seed) in done: continue
                r = train_one(s, norm, seed, device); rows.append(r)
                out.write_text(json.dumps(rows, indent=1))
                print(f"  {s} {norm:10s} seed={seed} NMSE={r['nmse']:.5f} zeros={r['zero_fraction']:.3f} {r['train_seconds']:.0f}s", flush=True)


if __name__ == '__main__':
    main()
