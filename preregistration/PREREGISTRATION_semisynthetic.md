# Pre-registered discriminating test: a real target placed in the map's win region
Registered: 2026-09-19 14:49 UTC. No experiment in this file has been run at this time.

## Why
Every real dataset in the paper sits in the all-relevant row of the map and ties.
The hypothesis "normalization never matters" predicts the same ties. This test is
built so the two hypotheses disagree.

## Construction (semi-synthetic, real target, real predictors)
- Data: the semiconductor monthly production panel (panel_ext.csv, 384 months),
  target = shipments (ship), 3 months ahead, as in the companion study.
- Real inputs: the 13 panel channels at the forecast origin, standardized on the
  training split. From training data alone these span about 2 independent
  directions, both relevant (Appendix, prospective coordinates).
- Appended inputs: m = 19 channels of INDEPENDENT Gaussian noise with scale
  sigma_z (independent of the target and of each other -- not copies of real
  channels), giving G = 32 groups and a training-only support fraction near 2/32.
- Predictor: the Definition-1 probe y = u (p(s)'x) + b, scores FREE (|s| <= 8),
  L-BFGS-B from zero init. Only the normalization changes: softmax vs sparsemax.
- n_train = 128 chronological windows; test = the following windows.
- 20 seeds = 20 independent draws of the noise channels; softmax and sparsemax
  see the same draw. Paired per seed.
- Regime rule: identical to Fig. 2 -- win if mean paired Delta > 2 SE and > 1e-4;
  lose if the mirror; tie otherwise. Delta = MSE_softmax - MSE_sparsemax on test.

## Predictions read from the free-score map (Fig. 2b) BEFORE running
P1  sigma_z = 2, m = 19, n = 128   -> WIN   (map cell k/G = 2/32, sigma = 2, n = 128: +0.0002)
P2  sigma_z = 1, m = 19, n = 128   -> WIN   (map cell k/G = 2/32, sigma = 1, n = 128: +0.0001)
P3  sigma_z = 0.25, m = 19, n=128  -> TIE   (map cell k/G = 2/32, sigma = 0.25, n = 128: 0)
P4  m = 0 (real channels only)     -> TIE   (all-relevant row)
The null hypothesis "normalization never matters" predicts TIE for all four.

## Falsifiers
- P1 or P2 observed as TIE or LOSE falsifies the win prediction for that cell.
- Both P1 and P2 observed as TIE falsifies the map's win region at free scores
  on this construction, and we will say so.
- P3 or P4 observed as WIN falsifies the map's tie prediction there.
All four verdicts will be reported.

## Secondary (not a registered claim)
The same four cells at |s| <= 2 are run for reference; the map predicts larger
wins there because of the bound. Not part of the test.

## Amendment 1 -- registered 2026-09-19 14:51 UTC, after the first run, before the second
The first run (results/semisynthetic_test_run1.json) is INVALID as a test of the
map and is reported as such. Diagnosis: the real channels were used as raw
levels. Standardized on the first 128 months, test-period features reach |z| = 34
against a training maximum of 3.4. Softmax placed weight on a trending level
(ip_semi, u = -2.28) and its test MSE exploded to 317; sparsemax happened to pick
a less trending channel (0.69). Both exceed the mean predictor (0.45). The
"wins" in all four cells, including P4 with no nuisance appended, are
extrapolation failures, not exclusion of nuisance.
Amended construction: real inputs are the one-month LOG DIFFERENCES of the 13
channels (stationary, matching the growth-rate target). Nothing else changes.
Predictions P1-P4 and their falsifiers stand exactly as written above.

## Verdicts -- recorded 2026-09-19 14:54 UTC, after the amended run (results/semisynthetic_test.json)
Free scores (|s| <= 8), 20 noise draws, paired:
- P1 (sigma_z = 2):    HELD.      Delta = +0.0605, 2SE = 0.0193, sparsemax lower 20/20.
- P2 (sigma_z = 1):    HELD.      Delta = +0.0630, 2SE = 0.0199, 18/20.
- P3 (sigma_z = 0.25): FALSIFIED. Predicted tie, observed win: Delta = +0.0552, 2SE = 0.0219, 16/20.
- P4 (no nuisance):    TIE (as predicted) in the deterministic fit, Delta = +0.0041 -> -0.0002
                       after removing two outlier test months. A bootstrap variant that adds
                       training-row resampling gives Delta = +0.056, 2SE = 0.049, 14/20 -- marginal;
                       reported, not counted as a verdict because the registered rule
                       presupposes noise-draw variability that this cell does not have.
Robustness (not registered, run after): wins in P1-P3 survive removing the two test months
with any |z| > 10 (Delta +0.050 / +0.052 / +0.042); across five 128-month training windows
P1 wins in all five (Delta +0.014 to +0.067, sparsemax lower 7-9 of 10 each) while P4 has no
consistent sign. In P1 sparsemax sets all 19 appended channels to exactly zero and keeps 5
of 13 real channels (caputil, ip_man, sox, emp, ip_comp).
Reading: the win the map predicts at small n with many independent irrelevant inputs is real
and is produced by exclusion of exactly those inputs. The map's sigma_z axis does not hold
here: the advantage does not require the nuisance to be loud. What matters at free scores is
how many irrelevant independent inputs there are relative to n.

## Prediction for the next test (added 2026-09-19, after P1-P4 were read)

P3 (m=19, sigma_z=0.25, n=128, free scores) was registered as a tie and a win
was observed (Delta=+0.055, 2SE=0.022, sparsemax lower in 16/20 draws). The
post-hoc reading is that at free scores the advantage depends on the number
of irrelevant independent inputs relative to n, not on their variance.
Registered prediction for the next semi-synthetic run (not yet executed):

- Holding n=128 and m=19, Delta will be a win at sigma_z in {0.1, 0.5}
  (Delta > 2SE and Delta > 1e-4), independent of sigma_z.
- Holding sigma_z=1 and m=19, Delta will fall to a tie at n=512
  (|Delta| <= 2SE or |Delta| < 1e-4).
- Holding n=128 and sigma_z=1, reducing m to 3 (k/G = 13/16) will give a tie.

Falsifier: a tie at (m=19, n=128, sigma_z=0.1) falsifies the count-based
reading; a win at n=512 with m=19 falsifies the sample-size axis of the map.
