#!/usr/bin/env python3
"""Constants behind the decision-rule diagnostics (Appendix "Decision-Rule Constants").

Re-runs the controlled grid, keeps every learned weight vector, and writes for
each of the 60 cells the quantities the appendix reports:

  L_closed     closed-form leak  L = r / (k e^{2B} + r)
  L_measured   mass the fitted softmax leaves on the irrelevant inputs
  bias_term    L^2 / k          (leak-induced bias of the dense side)
  noise_term   sigma_z^2 L^2 / r (Theorem 1 in equality form)
  dense_pred   bias_term + noise_term, the closed-form dense-side excess risk
  dense_meas   measured softmax excess MSE
  eps_n        sparsemax on-support excess risk (its excess MSE when the
               support is recovered)
  fail_rate    fraction of seeds in which sparsemax misses the support
  M_worst      worst observed excess of a support failure
  diagnostic   eps_n + fail_rate * M_worst, the plug-in rule value
  regime_fit   regime of the fitted comparison

The regime panel of the decision-rule figure uses the rule of Section 4 with
these measured constants; this script supplies the constants, not that panel.

Writes results/rule_evaluation.json.
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'package' / 'misa_sparsity'))
import run_entmax_sweep as sw

G = sw.G; TOL = sw.TOL

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--score-bound', type=float, default=2.0)
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    sw.SCORE_BOUND = a.score_bound; B = a.score_bound
    out = {}
    print(f"{'cell':<18}{'L_closed':>10}{'L_meas':>9}{'dense_pred':>12}{'dense_meas':>12}{'eps_n':>10}{'fail':>7}{'M':>10}{'diag':>10}  fit")
    for k in (2, 4, 8, 16, 32):
        for sig in (0.25, 0.5, 1.0, 2.0):
            for n in (128, 512, 2048):
                r = G - k
                L = r / (k * np.exp(2 * B) + r) if r else 0.0
                bias, noise = L ** 2 / k, sig ** 2 * L ** 2 / r if r else 0.0
                soft, sparse, leak, f1 = [], [], [], []
                for seed in range(a.seeds):
                    X, y = sw.sample_data(np.random.default_rng(seed), n, k, sig)
                    ev = lambda p: sw.evaluate(p, k, sig, np.random.default_rng(10_000 + seed))
                    p1, _ = sw.fit(1.0, X, y); p2, _ = sw.fit(2.0, X, y)
                    e1, irr1, _, _ = ev(p1); e2, _, f, _ = ev(p2)
                    soft.append(e1); sparse.append(e2); leak.append(irr1); f1.append(f)
                soft, sparse, f1 = np.array(soft), np.array(sparse), np.array(f1)
                ok = f1 >= 1.0 - 1e-9
                eps_n = float(sparse[ok].mean()) if ok.any() else float(sparse.mean())
                fail = float((~ok).mean())
                M = float((sparse[~ok] - eps_n).max()) if (~ok).any() else 0.0
                diag = eps_n + fail * M
                d = soft - sparse; se = d.std(ddof=1) / np.sqrt(len(d))
                key = f'k{k}_s{sig}_n{n}'
                out[key] = dict(k=k, sigma_z=sig, n=n, B=B, L_closed=float(L), L_measured=float(np.mean(leak)),
                                bias_term=float(bias), noise_term=float(noise), dense_pred=float(bias + noise),
                                dense_meas=float(soft.mean()), eps_n=eps_n, fail_rate=fail, M_worst=M,
                                diagnostic=float(diag), regime_fit=sw.regime(float(d.mean()), float(se)),
                                # aliases matching the shipped original_results/rule_evaluation.json
                                r=int(r), soft_pred=float(bias + noise), soft_obs=float(soft.mean()),
                                soft_bias=float(bias), soft_noise=float(noise), eps_meas=eps_n)
                o = out[key]
                print(f"k{k} s{sig} n{n:<6}{L:>10.4f}{o['L_measured']:>9.4f}{o['dense_pred']:>12.2e}"
                      f"{o['dense_meas']:>12.2e}{eps_n:>10.2e}{fail:>7.2f}{M:>10.2e}{diag:>10.2e}"
                      f"  {o['regime_fit']}", flush=True)
    p = Path(a.out) if a.out else ROOT / 'results' / 'rule_evaluation.json'
    json.dump(out, open(p, 'w'), indent=1); print('wrote', p)

if __name__ == '__main__': main()
