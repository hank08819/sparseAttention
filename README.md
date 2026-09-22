# When Does Exact-Zero Attention Help? — code and results

This repository is deliberately small. It carries every number the paper
reports, the scripts that produced them, and the registration files that fixed
the predictions in advance. It does not carry the raw exchange feed.

## Where each claim in the paper is

| claim in the paper | file |
|---|---|
| cross-section, 278 cells, +2.32% | `results/cross_section.json` |
| size-matched time axis, 278 cells, +12.73% | `results/time_axis.json` |
| FRED-MD, 242 cells, +15.69% | `results/fredmd.json` |
| FRED-MD before the two cleaning amendments | `results/fredmd_preamendment.json` |
| FRED-MD per-origin coordinates, 4,840 rows | `results/fredmd_perorigin.json` |
| the training-mean reference, all three runs | `results/absolute_baseline.json` |
| dependence-robust intervals | `results/dependence_robust_ci.json` |
| exact convex references, B = 2,3,4,8 | `results/convex_reference_full*.json` |
| convex references on the confirmatory cells | `results/confirmatory_convex_B*.json` |
| representation vs optimization path | `results/optimization_probe.json` |
| size-matched zeroed-position control | `results/zeroed_sizematched.json` |
| which path carries the residual dependence | `results/zeroed_path_decomposition.json` |
| constants behind the decision-rule figure | `results/rule_evaluation.json` |
| confirmatory test of the frozen rule, 12 cells | `results/confirmatory_amended2.json` |
| semi-synthetic discriminating test | `results/semisynthetic_test.json` |
| controlled grid, 2,400 fits (B = 2 and B = 8) | `results/entmax_sweep.json`, `results/entmax_sweep_B8.json` |

Every result file is a plain JSON keyed by cell. Each cell carries its
prediction, its outcome, its per-origin errors and its coordinate, so every
number in the paper can be recomputed from these files alone, with no data and
no GPU.

## Scripts

`package/misa_sparsity/` holds the model: the MISA encoder of Section 5, the
sparsemax and alpha-entmax implementations, the feature builder, the trainer,
and the three runners that produce the controlled grid, the boundary benchmark
and the PatchTST normalization swap. Ten files, no more than the paper uses.
Most scripts in `code/` import from it, so keep the two directories together.

`code/` holds one script per experiment, named for the experiment. Each reads
result files or raw panels by path and writes the JSON above. The two that a
reader is most likely to want are `run_time_axis_control.py`, which is the
matched control that isolates the geometry, and `run_dependence_robust_ci.py`,
which recomputes every interval quoted in the paper from the shipped JSONs in
under a minute.

## Registrations

`preregistration/` holds the five files that fixed predictions, decision rules
and falsifiers before the runs they govern. Each records its amendments and its
outcome as observed, including the criteria that were not met and the power
calculation that accompanies them.

## Data

`data/fred_md_current.csv` is the FRED-MD vintage used in the paper, downloaded
2026-09-20. It is public and is shipped so that the macro results reproduce
exactly rather than against a moving vintage.

The cryptocurrency panel is a 205 MB five-minute exchange feed and is not
redistributed here. The per-cell result files above contain the fitted errors
for all 556 cryptocurrency cells, so every reported number is checkable without
it. `run_cross_section.py` and `run_time_axis_control.py` take the panel by
path and document the expected schema.

## Reproducing the headline numbers without any data

```
python code/run_dependence_robust_ci.py
```

Scripts that refit a model need the package on the path:

```
PYTHONPATH=package:package/misa_sparsity python code/run_time_axis_control.py --predict-only
```

reads `results/cross_section.json`, `results/time_axis.json` and
`results/fredmd.json` and prints the three mean relative gains and their
intervals.
