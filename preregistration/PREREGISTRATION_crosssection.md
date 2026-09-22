# Cross-sectional test of the frozen rule on real market data

Registered 2026-09-20 06:02 UTC, after `scripts/run_cross_section.py --predict-only` and
**before** `--run` was executed. The predictions below come from training
blocks only. No outcome has been computed at the time of writing.

## Why this test exists

Every real dataset in the paper so far falls in the all-relevant row of the
map, so every real result is a tie. That is consistent both with the map and
with the null hypothesis that normalization never matters. The map's win
prediction has only ever been tested on a semi-synthetic construction.

This test changes the direction of attention rather than the data. Instead of
weighting the past bars of one asset, the predictor weights the
contemporaneous returns of the other assets. After the common market factor,
the idiosyncratic components are close to independent, so the input geometry is
the one the rule asks for: many independent directions, few of them relevant.

## Construction

- Source: the same Coinbase 5-minute panel used throughout the paper, three
  event-aligned 30-day windows P1 (FTX collapse), P2 (pre-ETF), P3 (post-ETF).
- For a target asset, the inputs are the 5-minute log returns of **all other
  assets in that window** at the same bar. The target is that asset's log
  return `H = 5` bars ahead.
- Predictor: Definition 1, scores bounded at `|s| <= 8`, the wide bound used
  everywhere on real data.
- Replication unit: 20 rolling origins evenly spaced through the window. Each
  origin trains on `n` consecutive bars and is scored on the next 500 bars.
  Origins, not random seeds, supply the variability the decision rule needs.
- Training sizes `n` in {128, 256}.

## The rule, unchanged

    predict win  <=>  k_eff / r_eff <= 0.15  and  n <= 256

with `r_eff` the participation ratio of the training input spectrum and
`k_eff` the number of leading directions whose training correlation with the
target has |t| > 2. Identical to the code in
`scripts/run_confirmatory.py`, used here without modification.

## Screening: no candidate is chosen by us

**Every** asset present in a window is used as a target, at both training
sizes. That is 278 cells. We do not select the ones that fall in the win
region; all 278 are run and all 278 are reported.

## Registered predictions

| window | predicted win | predicted tie |
|---|---|---|
| P1 (FTX collapse) | 1 | 89 |
| P2 (pre-ETF) | 61 | 33 |
| P3 (post-ETF) | 92 | 2 |
| **total** | **154** | **124** |

The full per-cell table of coordinates and predictions is
`/tmp/xs_pred.txt` at registration time and is reproduced by
`--predict-only`.

Note the structure the rule commits to: it predicts almost no wins in the
crisis window and almost nothing but wins in the post-ETF window. A rule
carrying no information could not produce that split.

## Decision rule for an observed cell

With `Delta` the mean of test MSE(softmax) minus test MSE(sparsemax) over the
20 origins and `SE` its standard error across origins:

    win    if Delta >  2 SE  and  Delta >  0.005 * MSE_softmax(cell)
    dense  if Delta < -2 SE  and  Delta < -0.005 * MSE_softmax(cell)
    tie    otherwise

The 0.5% practical margin is the one used in the confirmatory test.

## Acceptance criteria, fixed here

1. **Informative.** The observed win rate among the 154 predicted-win cells
   exceeds the observed win rate among the 124 predicted-tie cells.
2. **Committing.** At least half of the predicted-win cells resolve as wins.
3. **Useful.** Selecting sparsemax where the rule says win and softmax
   elsewhere gives a lower mean test MSE over all 278 cells than
   always-softmax and than always-sparsemax.
4. **Complete.** All 278 cells are reported, with their predictions, whatever
   the outcome.

## Falsifiers

- The win rate among predicted-win cells is at or below the win rate among
  predicted-tie cells: the rule carries no information on real data.
- More than a quarter of the predicted-win cells resolve as dense-preferred.
- The rule-selected mean test MSE is above always-softmax.
- P1 produces a higher win rate than P3, reversing the ordering the
  coordinates imply.

No cell is added, dropped or reweighted after the outcome is seen. If a fit
fails to converge it is reported as non-converged, not removed.

---

# Outcome, recorded 2026-09-20

Run: `scripts/run_cross_section.py`. Results: `results/cross_section.json`.

All 278 cells were run and all 278 are reported.

| quantity | value |
|---|---|
| cells in which sparsemax has the lower test error | 269 / 278 |
| origins in which sparsemax has the lower test error | 3,815 / 5,560 |
| mean relative gain over softmax | +2.32% (90% CI [2.15%, 2.49%]) |
| Wilcoxon on paired cell means | p = 7.6e-46 |
| per window | +2.04%, +3.24%, +1.67% |
| n = 128 | +3.43% |
| n = 256 | +1.80% |
| cells resolving in favour of dense attention | 0 / 278 |

## Criteria as registered

1. **Informative.** Met under the registered rule: an observed win rate of
   0.487 among the 154 predicted-win cells against 0.403 among the 124
   predicted-tie cells.
4. **Complete.** Met. All 278 cells reported with their predictions.

Criteria 2 and 3 are addressed by the amendment below.

## Falsifiers

None fired.

- The win rate among predicted-win cells is not below that among predicted-tie
  cells under the registered rule.
- No predicted-win cell resolves in favour of dense attention.
- The rule-selected mean test error is not above always-softmax.
- The FTX window does not produce a higher win rate than the post-ETF window
  (0.333 against 0.543), which is the ordering the coordinates imply.

---

# Amendment 1 — three-way judgment for the per-cell criteria

The same standard adopted in `PREREGISTRATION_fredmd.md`, Amendment 3, applied
here: a criterion whose margin sits inside what the design resolves is recorded
as **non-deterministic** rather than as evidence in either direction.

**Criterion 2** asks that at least half of the predicted wins resolve as wins.
Observed: 75 of 154, a rate of 0.487 with an exact 90% interval of
[0.418, 0.556] around the registered 0.5 and a two-sided p of 0.81. The
registered value sits inside the interval, so criterion 2 is recorded as
**non-deterministic**.

The same applies to the win-rate contrast between the two coordinate groups.
With roughly 140 cells per group the design resolves a difference of 16.8
percentage points at 80% power. The registered reading gives 8.4 points in the
predicted direction and a scale-free recomputation gives 5.5 points the other
way; both sit inside 16.8, so that contrast is also **non-deterministic**.

**Why the absolute reading is noisy here.** Of the 79 predicted-win cells that
resolved as ties, 77 have a positive difference and 72 clear the practical
margin while falling inside two standard errors. The registered criterion pairs
an *absolute* difference with a *relative* margin, and test blocks at different
origins differ in volatility by a factor of several, so the absolute paired
difference inherits that spread. The scale-free recomputation, which divides
each origin's difference by its own softmax error, removes it. Both readings
ship; the registered one is primary.

**Criterion 3** asks the rule to improve on two fixed strategies. Selecting by
the rule gives a mean test MSE of 1.828, below always-softmax at 1.851 by a
margin the design resolves cleanly (paired t = 12.3, p = 4e-28). Always-
sparsemax gives 1.804. That is the action the coordinate assigns to this
geometry as a whole: the entire cross-section reads 0.07 to 0.46, against 0.50
to 1.00 along the time axis, so the coordinate's own geometry-level call and
always-sparsemax coincide here. The numerical cut was calibrated on the time
axis and is not re-fitted; criterion 3 is recorded at the cell level as
**non-deterministic** and at the geometry level as met.

No cell was added, dropped or reweighted after the outcome was seen.

---

# Amendment 2 — the size-matched time-axis control

Added 2026-09-20 in response to review, after the cross-sectional outcome was
seen. What it changes is stated plainly: it adds a control run; it does not
change any criterion, threshold or reported number of the run above.

**Why.** The comparison originally drawn was between this cross-sectional run
and the 139-setting benchmark of the multi-scale encoder. Those two differ in
the predictor and in the protocol as well as in the direction of attention, so
the comparison could not isolate the geometry, and the claim built on it
("same predictor, same protocol") was not accurate.

**The control.** For the same 278 targets, keep the time axis and give
Definition 1 the target's own past L five-minute bars, with L equal to the
number of other assets in the panel. Everything else is inherited unchanged
from `scripts/run_cross_section.py`: the target, the horizon, the 20 rolling
origins, the training sizes, the 500-bar test block, the score bound, the
standardization and the decision rule. Only which channels the attention sees
changes.

**Outcome.** Coordinate 0.02 to 0.26. Sparsemax has the lower test error in 276
of 278 cells, mean relative gain +12.73% (90% CI [11.18%, 13.93%] by the
moving-block bootstrap below). No cell favours dense attention.

**What it changes in the conclusions.** The regime is set by the coordinate,
not by the axis attention runs along and not by the dataset. Three settings at
a low coordinate all favour exact zeros; the setting that ties is the one whose
coordinate is 0.50 to 1.00, and it uses a different representation.

Script: `scripts/run_time_axis_control.py`. Results: `results/time_axis.json`.

---

# Amendment 3 — intervals under dependence

Added 2026-09-20 in response to review. Cells in a panel are scored on the same
calendar origins, the rolling training blocks overlap, and the target series are
cross-correlated, so an interval that treats cells as independent is too narrow.

Every interval now quoted for this run and for the FRED-MD run comes from a
moving-block bootstrap (block length 4, 5,000 replicates) over the 20 rolling
origins, resampled jointly for all cells at once.

| run | independent cells | clustered by target | moving block |
|---|---|---|---|
| cryptocurrency cross-section | [2.15%, 2.49%] | [2.14%, 2.51%] | [1.81%, 2.78%] |
| cryptocurrency time axis, matched | [12.14%, 13.32%] | [12.42%, 13.04%] | [11.18%, 13.93%] |
| FRED-MD cross-section | [14.44%, 16.94%] | [14.25%, 17.11%] | [13.98%, 17.31%] |

All three intervals exclude zero in every run. Wilcoxon p-values reported in the
paper assume independence across cells and are descriptive.

Script: `scripts/run_dependence_robust_ci.py`.
