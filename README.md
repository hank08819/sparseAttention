# Mapping the Sparse-Attention Boundary for Cryptocurrency Forecasting

Anonymous code and data for the AAAI-27 submission.

**Naming note.** For historical reasons the model is implemented under the
internal identifier `MSCABiGRU` and appears as the key `MSCA` in result files.
It is the same model called **MISA** in the paper.

## Reproducing each result

| Paper item | Script | Output |
|---|---|---|
| Controlled study + Fig. 1 (bottom) | `code/run_controlled.py` | `results/controlled_results.json`, `figures/mechanism_sweep.pdf` |
| Phase map (Fig. 2) | `code/generate_phase_map.py` | `figures/fig_phase_map.pdf` |
| Temporal robustness (AR(1), heavy tails) | `code/run_controlled_temporal.py` | `results/controlled_temporal.json` |
| Neural phase map (main-text table) | `code/run_neural_phase_map.py` | `results/neural_phase_map.json` |
| Neural phase-map LaTeX rows | `code/make_neural_table.py` | (prints table body from the JSON) |
| Main 47-asset experiment | `code/run_52assets.py` | `results/expanded_results.json` |
| Supplement CSV tables | `code/export_supplement_csv.py` | `results/expanded_crypto_139.csv`, `results/expanded_crypto_asset_summary.csv` |
| 139-setting matched sparsemax-vs-softmax | `code/run_sparse_vs_soft_multiasset.py` | `results/sparse_vs_soft_multiasset.json` |
| Normalization robustness (z-score) | `code/run_norm_ablation.py` | `results/norm_ablation_zscore.json` |
| Injected-noise probe | `code/run_noise_injection.py` | `results/noise_injection.json` |
| Score-separation penalty | `code/run_score_separation.py` | `results/score_separation.json` |
| 2x2 matched factorial | `code/run_ablation_2x2.py` | `results/ablation_2x2_results.json` |
| 10-seed matched contrast | `code/run_multiseed_matched.py` | `results/multiseed_matched.json` |
| Matched neural ablation | `code/run_ablation.py` | `results/ablation_results.json` |
| Transformer DM tests | `code/run_transformer_dm_52assets.py` | `results/transformer_dm_results.json` |
| Boundary figures | `code/generate_negative_results_figs.py` | `figures/fig_injection_probe.pdf`, `figures/fig_category_gradient.pdf` |

## Model and pipeline

- `code/model.py` — MISA architecture (BiGRU encoders, sparsemax temporal and
  scale selection, gated cross-signal attention)
- `code/sparsemax.py` — sparsemax (Martins and Astudillo 2016)
- `code/features.py` — feature pipeline (correlation filter, PCA ranking)
- `code/train.py` — trainer, early stopping, Harvey-corrected DM test, NMSE

## Data

`data/` ships a sample: BTC, ETH, and SAND five-minute OHLCV for the three
30-day periods (P1 Nov 2022, P2 Oct 2023, P3 Mar 2024) and all nine hourly FX
files. The full 47-asset benchmark (about 200 MB) regenerates with:

```bash
cd data && python download_expanded.py   # public Coinbase REST API
```

Traffic uses the public METR-LA benchmark (Li et al. 2018), not redistributed.

## Environment

```bash
pip install torch>=2.0 numpy pandas scikit-learn scipy matplotlib
```

CPU-only; every experiment sets its seeds explicitly (42-44 for the main
tables, 42-51 for the 10-seed contrast, 0-19 for the controlled study).

## Quick start

```bash
cd code
python run_controlled.py --quick          # smoke test, ~1 minute
python run_experiment.py \
    --tgt ../data/BTC_5m_P2_20231001.csv \
    --ca  ../data/ETH_5m_P2_20231001.csv  # one asset-period end-to-end
```

## License

MIT.
