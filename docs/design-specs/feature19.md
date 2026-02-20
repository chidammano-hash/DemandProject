# Feature 19: PatchTST Backtesting Implementation

## Objective
Implement PatchTST (Patch Time Series Transformer) backtesting — the first deep learning model in the platform. Uses Apple MPS GPU acceleration and supports global, per-cluster, and transfer learning strategies.

## Scope
- **Models**: Custom lightweight PyTorch PatchTST for monthly demand forecasting
- **Strategies**: Global model, per-cluster (separate model per `ml_cluster`), transfer learning (global base → per-cluster fine-tune)
- **Timeframes**: 10 expanding windows (A-J), auto-detected from data
- **GPU**: Apple MPS (Metal Performance Shaders) by default, CUDA fallback, CPU fallback
- **Main table**: Each prediction stored at the DFU's `execution_lag` from `dim_dfu`
- **Archive table**: All lags 0-4 preserved in `backtest_lag_archive`

## Model IDs

| Strategy | model_id | Description |
|----------|----------|-------------|
| Global | `patchtst_global` | One PatchTST for all DFUs |
| Per-cluster | `patchtst_cluster` | Separate PatchTST per `ml_cluster` |
| Transfer | `patchtst_transfer` | Global base → per-cluster fine-tune via weight transfer |

## PatchTST-Specific Details

### Architecture
PatchTST segments a 12-month qty sequence into overlapping 3-month patches (stride 2), producing 5 patches. Each patch is projected to `d_model=64` dimensions, enriched with learnable positional encoding and static covariate conditioning, then processed by a 2-layer Transformer encoder (4 heads, GELU activation). A flatten+MLP head produces a single next-month prediction.

~60K trainable parameters. Deliberately compact for monthly-grain data with ~50K DFUs.

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seq_len` | 12 | Lookback window (months) |
| `patch_len` | 3 | Months per patch (quarterly) |
| `stride` | 2 | Patch stride (overlapping) |
| `d_model` | 64 | Embedding dimension |
| `nhead` | 4 | Attention heads |
| `num_layers` | 2 | Transformer encoder layers |
| `dropout` | 0.1 | Regularization |
| `epochs` | 30 | Max training epochs |
| `batch_size` | 512 | Training batch size |
| `learning_rate` | 1e-3 | AdamW learning rate |
| `patience` | 5 | Early stopping patience |
| `loss` | HuberLoss | Robust to demand outliers |

### GPU Support
- Apple MPS auto-detected via `torch.backends.mps.is_available()` with verification
- CUDA fallback for Linux/NVIDIA
- CPU fallback if no GPU available
- `torch.mps.empty_cache()` called between timeframes to manage memory

### Categorical Handling
Label-encoded as integers (not embedded). Static covariates projected via `nn.Linear` and injected into the first patch token.

### Transfer Learning
- Phase 1: Train global base model on all DFUs (30 epochs)
- Phase 2: Per-cluster fine-tune via `copy.deepcopy(base_model)` + freeze patch embedding + first Transformer layer. Fine-tune for 10 epochs at 0.1× learning rate.
- Fallback: Clusters with < 20 sequences use base model predictions.

## Feature Engineering

Unlike tree-based models, PatchTST learns temporal features directly from raw qty sequences. No hand-engineered lag or rolling features.

### Input Sequence
- 12 most recent monthly `qty` values (log1p transformed)
- Zero-padded on the left if < 12 months of history available

### Covariates (11 features)
- **Numeric DFU**: `execution_lag`, `total_lt`
- **Numeric Item**: `case_weight`, `item_proof`, `bpc`
- **Calendar**: `month_sin`, `month_cos` (of target month)
- **Categorical (label-encoded)**: `ml_cluster`, `region`, `brand`, `abc_vol`

### Target
- `qty` (log1p transformed). Predictions inverse-transformed via `expm1` and floored at 0.

## Lag Strategy

Same as LGBM/CatBoost/XGBoost — see Feature 9.

### Main Table (`fact_external_forecast_monthly`)
Predictions stored at execution lag only.

### Archive Table (`backtest_lag_archive`)
All lags 0-4 preserved with `timeframe` column.

## Implementation

### Scripts

| Script | Purpose |
|--------|---------|
| `mvp/demand/scripts/patchtst_model.py` | PatchTST nn.Module, DemandDataset, train/predict functions, MPS device detection |
| `mvp/demand/scripts/run_backtest_patchtst.py` | Train PatchTST + generate predictions for all timeframes |
| `mvp/demand/scripts/load_backtest_forecasts.py` | Bulk load predictions into Postgres (shared, unchanged) |

### run_backtest_patchtst.py
Parameters: `--cluster-strategy`, `--model-id`, `--n-timeframes`, `--output-dir`, `--seq-len`, `--patch-len`, `--stride`, `--d-model`, `--nhead`, `--num-layers`, `--dropout`, `--epochs`, `--batch-size`, `--learning-rate`, `--patience`, `--weight-decay`, `--transfer-epochs`, `--transfer-min-rows`, `--transfer-freeze-layers`, `--device`

Output:
- `backtest_predictions.csv`: Execution-lag only (for main table)
- `backtest_predictions_all_lags.csv`: All lags 0-4 (for archive)
- `backtest_metadata.json`

## Makefile Targets

```makefile
backtest-patchtst:          # Global PatchTST backtest (Apple MPS GPU)
backtest-patchtst-cluster:  # Per-cluster PatchTST backtest
backtest-patchtst-transfer: # Transfer learning PatchTST backtest
backtest-load:              # Load predictions into Postgres (shared)
```

## Schema

Uses existing tables — no DDL changes needed. Same output format as LGBM/CatBoost/XGBoost.

## Verification

```bash
cd mvp/demand && uv sync                       # Install torch (~2GB)
python -c "import torch; print(torch.backends.mps.is_available())"
make backtest-patchtst                          # Run global backtest
make backtest-load                              # Load main + archive
curl "http://localhost:8000/domains/forecast/models"
make backtest-patchtst-cluster                  # Per-cluster backtest
make backtest-patchtst-transfer                 # Transfer learning backtest
make backtest-load                              # Reload
make champion-select                            # Re-run with PatchTST models
```

## Dependencies
- Feature 8 (backtesting framework)
- Feature 7 (clustering)
- Feature 4 (fact tables)
- torch >= 2.0.0 (includes MPS support), python-dateutil >= 2.8.0
