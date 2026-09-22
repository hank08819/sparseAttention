#!/usr/bin/env python3
"""Training-only phase coordinates with a redundancy-aware support fraction.

For a dataset's TRAINING split only:
  r_eff  = participation ratio of the input correlation spectrum (independent directions)
  k_eff  = number of the top-r_eff principal directions carrying 90% of the ridge
           coefficient mass when the target is regressed on them
  k_eff / r_eff  = support fraction among independent directions (in [0, 1])
  sigma_z = sqrt(mean eigenvalue of off-support directions / mean of on-support)
  n      = training windows
The free-score (B = 8) map then returns win / tie / lose. Nothing here uses test data.
"""
import numpy as np
def coords(X, y, share=0.90):
    Xc=(X-X.mean(0))/(X.std(0)+1e-12); yc=(y-y.mean())/(y.std()+1e-12)
    w,V=np.linalg.eigh(np.cov(Xc,rowvar=False)); w=np.clip(w,1e-12,None); idx=np.argsort(w)[::-1]; w=w[idx]; V=V[:,idx]
    r_eff=max(1,int(round((w.sum()**2)/(w**2).sum())))
    Z=Xc@V[:,:r_eff]; Z=Z/(Z.std(0)+1e-12)
    beta=np.linalg.solve(Z.T@Z+np.eye(r_eff), Z.T@yc)
    mag=np.sort(np.abs(beta))[::-1]; c=np.cumsum(mag)/max(mag.sum(),1e-12); k_eff=int(np.searchsorted(c,share)+1)
    sig_z=float(np.sqrt(w[k_eff:r_eff].mean()/w[:k_eff].mean())) if r_eff>k_eff else 0.0
    return dict(k_eff=k_eff, r_eff=r_eff, kfrac=k_eff/r_eff, sigma_z=sig_z)
def regime_free_scores(kfrac, sig, n):
    """Regimes of the B = 8 controlled map (Fig. 2b): wins and losses exist only at n <= 128."""
    if n<=200:
        if kfrac<=0.15 and sig>=1: return 'win'
        if 0.2<=kfrac<=0.6 and sig>=1: return 'lose'
    return 'tie'
