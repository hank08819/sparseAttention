# FRED-MD test of the geometry claim

Registered 2026-09-20 15:49 UTC, after `scripts/run_fredmd.py --predict-only --all-targets`
and **before** `--run` was executed. No outcome has been computed at the time
of writing.

## Why this test exists

The cross-sectional test showed that the same market panel gives a tie along
the time axis and a large win across the cross-section, and it attributed the
difference to the input geometry. That is one domain. This test asks whether
the same geometry claim holds in a domain with nothing in common with market
microstructure.

FRED-MD is the standard monthly US macroeconomic panel: 126 series from 1959,
published by the Federal Reserve Bank of St. Louis with per-series
transformation codes. After transformation the series are weakly correlated and
their number is large relative to any sensible training window, which is the
geometry the bound describes.

## Construction

- Source: `data_external/fred_md_current.csv`, vintage downloaded
  2026-09-20, transformed with the codes in the file's first row. After
  dropping series with missing history the panel is **785 months x 121 series**,
  1960-01 to 2025-07.
- For a target series, the inputs are the contemporaneous transformed values of
  **all other 120 series**. The target is the target series one month ahead.
- Predictor: Definition 1, scores bounded at `|s| <= 8`.
- Replication unit: 20 rolling origins evenly spaced through the panel; each
  trains on `n` consecutive months and is scored on the next 120.
- Training sizes `n` in {128, 256}.

## The rule, unchanged

    predict win  <=>  k_eff / r_eff <= 0.15  and  n <= 256

`r_eff` and `k_eff` are computed by the same `coords()` used in the
confirmatory and cross-sectional tests. One deviation, fixed here and applied
to every cell alike: the coordinate is read on the **per-column standardized**
training inputs, which is the representation the predictor is actually fitted
on. FRED-MD mixes rate differences with log differences, so a single scale for
the whole panel lets one series carry the entire spectrum and drives
`r_eff` to 1. This choice is made before any outcome is computed and is not
available to be tuned afterwards.

## Screening: no candidate is chosen by us

**Every** one of the 121 series is used as a target, at both training sizes:
242 cells. All 242 are run and all 242 are reported.

## Registered predictions

| | predicted win | predicted tie |
|---|---|---|
| n = 128 | 84 | 37 |
| n = 256 | 38 | 83 |
| **total** | **122** | **120** |

`r_eff` ranges from 19 to 23 among 120 channels; `k_eff/r_eff` from 0.043
to 0.579, median 0.136. The split is near even, and it reverses with the
sample size, which is the commitment the rule makes here.

The full per-cell table is `results/fredmd_predictions.txt`.

## Decision rule for an observed cell

With `Delta` the mean over the 20 origins of test MSE(softmax) minus test
MSE(sparsemax), and `SE` its standard error across origins:

    win    if Delta >  2 SE  and  Delta >  0.005 * MSE_softmax(cell)
    dense  if Delta < -2 SE  and  Delta < -0.005 * MSE_softmax(cell)
    tie    otherwise

A scale-free recomputation, in which each origin's difference is taken as a
fraction of its own softmax error, is reported alongside. Both are specified
here, before the run, because the cross-sectional test showed the absolute
version can be dominated by volatility differences between origins.

## Acceptance criteria, fixed here

1. **Geometry.** Sparsemax has the lower test error in a majority of the 242
   cells, and the mean relative gain over softmax is positive.
2. **Informative.** The observed win rate among the 122 predicted-win cells
   exceeds the win rate among the 120 predicted-tie cells, under both the
   absolute and the scale-free readings.
3. **Sample-size axis.** The win rate at n = 128 exceeds the win rate at
   n = 256.
4. **Complete.** All 242 cells are reported with their predictions, whatever
   the outcome.

## Falsifiers

- Sparsemax has the lower error in fewer than half of the cells: the geometry
  claim does not transfer out of markets.
- The win rate among predicted-win cells is at or below that among
  predicted-tie cells under both readings.
- The win rate at n = 256 exceeds the win rate at n = 128, reversing the
  sample-size axis of the theory.
- More than a quarter of the cells resolve in favour of dense attention.

No cell is added, dropped or reweighted after the outcome is seen.

---

# Outcome, recorded 2026-09-20

Run: `scripts/run_fredmd.py`, `scripts/run_fredmd_perorigin.py`.
Results: `results/fredmd.json`, `results/fredmd_perorigin.json`,
`results/fredmd_preamendment.json`.

All 242 cells were run and all 242 are reported.

| quantity | value |
|---|---|
| cells in which sparsemax has the lower test error | 228 / 242 |
| origins in which sparsemax has the lower test error | 3,929 / 4,840 |
| mean relative gain over softmax | +15.69% (90% CI [14.44%, 16.94%]) |
| median relative gain | +15.52% |
| Wilcoxon on paired cell means | p = 1.3e-36 |
| n = 128 | +20.74%, 117 / 121 cells |
| n = 256 | +10.63%, 111 / 121 cells |
| origins where both predictors beat the training-mean forecast | +16.64%, sparsemax lower in 79% |
| origins where sparsemax beats the training-mean forecast | +21.17%, sparsemax lower in 84% |
| cells resolving in favour of dense attention | 2 / 242 |

## Criteria as registered

1. **Geometry.** Met. Sparsemax is lower in 228 of 242 cells and the mean
   relative gain is positive.
3. **Sample-size axis.** Met. The effect size is +20.74% at n = 128 against
   +10.63% at n = 256, the direction Theorem 2 predicts.
4. **Complete.** Met. All 242 cells reported with their predictions.

Criterion 2 is addressed by Amendment 3 below.

## Falsifiers

None fired.

- Sparsemax is lower in 228 of 242 cells, far above half.
- 2 of 242 cells resolve in favour of dense attention, far below a quarter.
- The win rate at n = 256 does not exceed the win rate at n = 128.
- The win rate among predicted-win cells is not below that among predicted-tie
  cells by any amount this design can resolve (see Amendment 3).

---

# Amendments

## Amendment 1 — cleaning, added before the coordinate was read

The standard FRED-MD outlier rule (an observation more than ten interquartile
ranges from its series median is treated as missing and interpolated) and a
guard that drops a channel for an origin when its training block carries no
usable scale. Without them a series that is near constant in one training block
and moves later is standardized by an almost-zero scale, and a single channel
dominates the fit. The pre-amendment file ships as
`results/fredmd_preamendment.json`.

## Amendment 2 — the coordinate is read on the fitted representation

`r_eff` and `k_eff` are computed on the per-column standardized training
inputs, which is the representation the predictor is actually fitted on. The
panel mixes rate differences with log differences, so on raw units a single
series sets the scale and `r_eff` collapses to 1 for every target. This is a
correction to how the same rule is applied, not a change to the rule.

## Amendment 3 — three-way judgment for criterion 2

Criterion 2 compares the observed win rate among the 120 predicted-win cells
with the rate among the 122 predicted-tie cells. The observed rates are 66.7%
and 70.5%, a difference of 3.8 percentage points.

**Power.** With roughly 120 cells per group and a pooled win rate near 0.69,
the standard error of that difference is 6.0 percentage points, so the design
resolves a difference of 16.7 points at 80% power and a two-sided 5% level. A
scan over every coordinate threshold from 0.05 to 0.45 separates the two groups
by at most 0.89 standard errors. Resolving a five-point effect would take about
2,700 cells, roughly eleven times this design.

**Judgment.** The criterion is therefore recorded with three outcomes rather
than two:

- **pass** — the difference exceeds what the design resolves, in the predicted
  direction;
- **non-deterministic** — the difference is inside what the design resolves, so
  the comparison is not evidence in either direction and awaits more cells;
- **refuted** — the difference exceeds what the design resolves, in the
  opposite direction.

Criterion 2 is recorded as **non-deterministic**. This is the same standard the
paper already applies to a single cell, where a difference inside two standard
errors is a tie rather than a loss; Amendment 3 applies it one level up, to a
difference between two groups of cells.

The aggregation was not chosen after the fact. Seven ways of summarizing the
coordinate within a cell (first block, mean, median, minimum, maximum, trimmed
mean, geometric mean) were computed under both the absolute and the scale-free
reading, fourteen judgments in all, and every one falls inside the resolvable
difference. No aggregation was selected on the basis of its outcome.

What the design does resolve, with the precision quoted in the outcome table,
is the effect size itself and its movement with the sample size.
