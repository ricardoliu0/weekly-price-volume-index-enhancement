# Weekly Price-Volume Index Enhancement Model

A research implementation of a weekly stock-scoring pipeline using price-volume sequences and a bidirectional GRU with temporal attention.

The pipeline prepares rolling windows, normalizes features and labels, trains a neural network, generates Friday stock scores, and selects up to 100 eligible stocks per scoring date. The repository documents the modeling workflow; it does not contain a complete index-relative portfolio optimization or backtesting system.

## Data Availability

The original datasets are proprietary and are not included in this repository. Processed datasets, trained model weights, training logs, stock scores, and selection outputs are also excluded. Running the full pipeline requires authorized access to compatible input data. No sample data or pretrained model is provided.

## Files

| File | Purpose |
| --- | --- |
| `step1_split_by_date_and_save_pkl.py` | Construct rolling feature windows and shifted return labels |
| `step2_apply_zscore.py` | Apply daily channel-wise MAD normalization, clipping, and label rank transformation |
| `step3_gru_final_norankloss.py` | Train the BiGRU-attention model and save the selected checkpoint |
| `step4_save_score.py` | Generate Friday stock scores using the trained model |
| `step5_TOP100.py` | Select up to 100 eligible stocks per date |
| `requirements.txt` | Python package dependencies |
| `.gitignore` | Exclude datasets, checkpoints, and generated outputs from Git tracking |

## Input Data Format

Place authorized input CSV files beside the scripts. Do not upload them to the repository.

### market_data.csv

Required columns:

| Column | Description |
| --- | --- |
| `datetime` | Observation date in YYYY-MM-DD format |
| `instrument` | Stock identifier, consistent with the eligibility tables |
| `close`, `open`, `high`, `low`, `vwap` | Price features |
| `volume` | Trading volume |
| `0c-10c-ret` | Source forward-return label |

The script shifts `0c-10c-ret` by one observation within each instrument and names the result `1c-11c-ret`. The intended return horizon must be verified against the source label definition and observation calendar.

### ST.csv

A date-by-stock table with dates in the first column and stock identifiers as column headers. The current filter treats missing entries as eligible under the ST rule; this convention must match the source dataset.

### trading_status.csv

A date-by-stock table with the same indexing convention. The current filter accepts the source value `交易` as active trading status. Date strings and stock identifiers must match the score table.

## Installation

```bash
python -m pip install -r requirements.txt
```

PyTorch uses CUDA when available and otherwise uses the CPU. Package versions are not pinned; compatibility with a particular Python or CUDA environment has not been validated for this public version.

## Running the Pipeline

Run the scripts in order after supplying authorized input data:

```bash
python step1_split_by_date_and_save_pkl.py
python step2_apply_zscore.py
python step3_gru_final_norankloss.py
python step4_save_score.py
python step5_TOP100.py
```

Paths are resolved relative to each script's directory.

| Generated item | Contents |
| --- | --- |
| `daily_data_pkl/` | Daily feature windows, metadata, labels, and date index |
| `daily_data_pkl_norm/` | Normalized daily datasets |
| `best_model.pth` | Selected model weights |
| `training_log.csv` | Epoch-level training and validation metrics |
| `score_weekly.csv` | Friday stock scores |
| `top100_filtered.csv` | Eligible stocks ranked by score, up to 100 per date |

These generated items are excluded by `.gitignore`.

## Model and Preprocessing

- Inputs contain 30 observations and six channels: close, open, high, low, VWAP, and volume.
- Volume is transformed with `log1p`; the five price channels are divided by the final close in each window.
- Within each daily dataset, each channel is centered by its median and divided by its unscaled median absolute deviation across stocks and window observations, then clipped at the 1st and 99th percentiles.
- Daily labels are transformed to normal quantiles using their cross-sectional ranks.
- The model uses a two-layer bidirectional GRU with hidden size 64, temporal attention, and 32 tanh outputs. The mean of these outputs is used as the stock score.
- The initial learning rate is 0.00005. The maximum training duration is 200 epochs, with early stopping and learning-rate reduction based on validation metrics.
- Training uses observations from 2007–2014; validation uses 2015. Scoring selects Friday observations from 2016–2025.

## Implementation Notes and Limitations

This version preserves the original numerical workflow while updating paths, comments, and console messages. It has not been rerun or validated against the proprietary data.

- Step 1 retains backward filling within each stock. This can use future observations and introduce look-ahead bias.
- Step 3 applies a date cutoff to the initial inspection loader, but recreates the training loader without that cutoff inside the epoch loop. Label overlap at the training-validation boundary therefore requires review.
- The combined loss value includes an MSE term and a RankIC-based term. RankIC is computed after detaching predictions, so only MSE contributes gradients. The combined value still affects checkpoint selection and early stopping.
- RankIC is averaged across output channels within batches and then aggregated across batches. It is not a daily cross-sectional IC evaluation.
- Friday-only scoring skips weeks whose Friday is absent; it does not substitute the final trading day of those weeks.
- Eligibility filtering depends on the conventions in the input ST and trading-status tables.
- The pipeline does not calculate transaction costs, turnover, benchmark-relative returns, or portfolio performance. Top 100 selection alone does not establish index-enhancement performance.
- Existing output directories may contain files from earlier runs. Use a clean local output directory when reproducing preprocessing.

## Scope

This repository presents research code and its implementation limitations. It makes no claim of production readiness or verified investment performance.
