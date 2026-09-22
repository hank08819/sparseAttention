# Pre-registered prediction: daily resolution (written before any daily experiment)
Date: 2026-07-26. No daily-resolution experiment has been run at this time.

Setting: daily OHLCV of semiconductor equities/indices (SOX, TSM, NVDA, INTC,
MU, AMD), window T=30 days, horizon 5 days, protocol identical to the
companion cryptocurrency study. Training windows ~5,000 (vs 250 monthly).

Predictions from the phase map:
P1. The complexity penalty shrinks or disappears: MISA's rank against
    LSTM/GRU improves markedly relative to the monthly panel, where it was last.
P2. The matched sparsemax-vs-softmax contrast remains a tie (support on
    daily financial series is not learnable, as in the cryptocurrency domain).
Falsifiers: MISA still last at daily resolution falsifies P1's reading of the
map; a significant sparsemax win falsifies P2 and the boundary rule itself.

## P3 addendum: 5-minute resolution (registered 2026-07-26, before any 5m experiment)
Setting: 5-minute OHLCV bars, last 60 trading days, semiconductor equities
(TSM, NVDA, INTC, MU, AMD), leader SOX; identical protocol to the companion
cryptocurrency study, which also uses 5-minute bars.
P3 (user's hypothesis): at 5-minute resolution the domain matches the
cryptocurrency regime, and MISA beats LSTM and GRU.
Falsifier: MISA loses to the recurrent baselines at 5-minute resolution.
The matched sparse-vs-soft contrast is still predicted to tie (P2 carries over).

## P4 addendum: 1-minute resolution (registered before any 1m experiment)
Setting: 1-minute bars, last 7 calendar days (Yahoo free limit), same assets
and protocol. Caveat registered up front: one week covers a single market
regime, so results are fragile and reported as a bonus rung only.
P4: the 1-minute rung behaves like the 5-minute rung (high-frequency regime).
Falsifier: a reversal relative to the 5-minute rung beyond seed noise.

## P5 addendum: decision layer (registered before any cost computation)
Setting: monthly shipments (A34SVS) as demand; three-month-ahead newsvendor
with order-up-to quantity Q = S_t * exp(yhat + z_CR * sigma_val), where
sigma_val is each model's validation residual sd; critical ratios CR in
{0.50..0.95}; realized cost = cu*max(D-Q,0) + co*max(Q-D,0), normalized by
mean demand.
P5a: the cost ranking follows the accuracy ranking; at monthly resolution
the simplest recurrent model minimizes inventory cost.
P5b: matched sparsemax and softmax yield indistinguishable costs at every
critical ratio.
Falsifiers: a complex model achieving lower cost despite higher NMSE (5a);
a systematic sparse-soft cost gap (5b).

## P6 addendum: decision-focused training (registered before any DFL run)
Setting: same shipments newsvendor as P5, but each model is trained
directly with pinball (quantile) loss at critical ratio c, so its output
is the order quantile itself and no sigma buffer is applied. Critical
ratios {0.50, 0.70, 0.90, 0.95}; models MISA, softmax twin, LSTM, GRU,
Informer; three seeds; costs evaluated exactly as in P5.
P6a: the matched sparse-soft cost tie persists under decision-focused
training at every critical ratio.
P6b: at n=250 the simplest recurrent models still achieve the lowest
cost under decision-focused training.
P6c (secondary): decision-focused training lowers cost relative to
predict-then-order at high critical ratios, per the Ban-Rudin effect.
Falsifiers: a systematic sparse-soft cost gap (6a); an attention model
cheapest (6b); DFL failing to lower cost for most models at c>=0.90 (6c).

## P7 addendum: expanded listed-semiconductor universe incl. quantum chips
(registered before any new data were downloaded or any model trained)
Universe: the major listed semiconductor firms
AVGO QCOM TXN ADI NXPI MRVL ON STM ASML AMAT LRCX KLAC TER MCHP SWKS
QRVO MPWR UMC ASX GFS plus the listed quantum-chip firms IONQ RGTI QBTS
QUBT. Daily bars, full free history per ticker, same pipeline, models
MISA / softmax twin / LSTM / GRU, three seeds, NMSE.
P7a: the matched sparse-soft tie replicates across the expanded
universe: no systematic direction, magnitudes within seed noise.
P7b: within the daily rung, architecture value tracks history length.
On short-history tickers (n_train below ~1,500: the quantum group and
recent IPOs) MISA's mean rank against LSTM and GRU is worse than on the
long-history group (n_train above ~5,000).
Falsifiers: a systematic sparse-soft gap (7a); short-history rank equal
to or better than long-history rank (7b).

## P8 addendum: decision-layer hardening (registered before any run)
Setting: the P5 newsvendor repeated on three semiconductor demand
proxies from the same panel: shipments (A34SVS), new orders (A34SNO),
and industrial production of semiconductors (IPG3344S). Models MISA,
softmax twin, LSTM, GRU; TEN seeds; cost computed PER SEED (not from
seed-averaged forecasts); critical ratios {0.50, 0.70, 0.90, 0.95}.
Plus decision-focused (pinball) training at c=0.90 on shipments, ten
seeds, per-seed costs.
P8a: the cost ranking replicates on all three proxies: a recurrent
model is cheapest and the attention models are the most expensive.
P8b: per-seed sparse-soft cost differences show no significant
advantage in either direction on any proxy (paired Wilcoxon, ten
seeds).
P8c: with per-seed costs, the headline ratios hold beyond seed noise:
attention/LSTM cost ratio above 4 under predict-then-order at c=0.90,
and above 2 after decision-focused training.
Falsifiers: an attention model cheapest on any proxy (8a); a
significant consistent sparse-soft gap (8b); ratios collapsing into
seed noise (8c).

## P9 addendum: chip-shortage stress window (registered before any run)
Setting: the ship-target newsvendor of P5/P8, ten seeds, per-seed
per-month costs at c in {0.50, 0.90}. The test period (roughly
2020-2026) contains the documented semiconductor shortage. Shortage
window fixed IN ADVANCE from press chronology: target months 2020-09
through 2022-12 inclusive; all other test months form the calm window.
P9a: mean per-month cost is higher in the shortage window than in the
calm window for every model.
P9b: the cost ranking is preserved inside the shortage window: a
recurrent model is cheapest and the attention models are the most
expensive. (This rejects the "complex models shine in turbulence"
story if it holds.)
P9c: the matched sparse-soft cost difference stays insignificant
inside the shortage window.
Falsifiers: a model with lower cost under disruption than calm (9a);
an attention model cheapest in the shortage window (9b); a significant
sparse-soft gap there (9c).

## AISTATS adaptive-control experiments (registered 2026-07-29, BEFORE runs)
Lineage: derived predictions. The 3-seed AD1-AD3 outcomes (PREREGISTRATION_iclr.md)
inform these; the new test is 10 seeds with a PAIRED fixed-sparsemax twin
(identical architecture, gate frozen at g=1), both regions, same seeds/trainer.
Runner: run_adaptive_sparse_p2.py -> results/adaptive_sparse_p2.json.

- AC1 (win region, NVDA daily): adaptive matches fixed sparsemax NMSE within
  one seed-sd. FALSIFIER: adaptive worse than fixed by >2 paired-sd.
- AC2 (direction across regions): mean learned gate on daily > mean gate on
  monthly (the dial rises with sample size, companion dial theorem).
  FALSIFIER: monthly mean gate >= daily mean gate.
- AC3 (tie region, monthly): the dial does NOT repair the tie: adaptive monthly
  NMSE not better than fixed by >2 paired-sd (an unlearnable margin cannot be
  bought back by blending). FALSIFIER: adaptive better by >2 paired-sd.
- AC4 (flat-risk signature): win-region gate seed-spread (max-min) exceeds the
  between-region mean-gate gap (near-tied risks flatten R(g), so the optimizer
  wanders). FALSIFIER: win spread < gap AND win spread < 0.10.
All four verdicts will be reported, hits and misses.

## AC verdicts (recorded 2026-07-29, after runs; results/adaptive_sparse_p2.json)
- AC1 HELD: daily adaptive 0.0175+-0.0001 vs fixed 0.0175+-0.0003; paired diff -0.26 sd.
- AC2 FIRED: daily gate 0.449 vs monthly 0.464; means indistinguishable. Diagnosis
  written into paper: prediction imported the companion's dose response, but this
  domain ties at every rung, so R(g) is flat everywhere -> no mean shift, wide wander.
  The dial confirmed the paper's own tie finding; the registration premise was wrong.
- AC3 HELD: monthly adaptive 0.314+-0.070 vs fixed 0.325+-0.094; -0.14 paired sd. No repair.
- AC4 HELD: daily gate spread 0.572 vs between-region gap 0.015 (38x); monthly spread 0.182.
  Spread ordering follows tie tightness (daily tie -0.0% is tightest, wanders widest).

## Dial-identification diagnostic (2026-07-29, post-hoc measurement, labeled as such)
run_dial_D.py, seed 42 paired twins, per-window test errors:
- monthly: D=3.71e-04, S=5.51e-03, S/D=14.9 -> m>238 seeds to pin dial to +-0.125
- daily:   D=2.30e-06, S=4.62e-03, S/D=2011 -> m>32,000 seeds
Feeds Theorem dial-identification (Le Cam, KL=Delta^2 D/(2S) exact). Not a registered
prediction; it is the measured input of a proved lower bound.
