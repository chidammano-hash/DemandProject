# Tree Model Implementations


| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Accuracy, Item Analysis |

---

## Problem

A single algorithm cannot optimally forecast all demand patterns. Seasonal products behave differently from intermittent ones; high-volume items have different error profiles than slow movers. Relying on one algorithm means accepting suboptimal accuracy for large portions of the portfolio.

## Solution


## How It Works

1. Each algorithm reads its configuration from `config/forecasting/forecast_pipeline_config.yaml`
2. The shared framework loads sales data, builds the feature matrix, and generates timeframes
3. The algorithm-specific script provides `train_and_predict_per_cluster()` and `train_and_predict_global()` functions
4. The framework calls the appropriate function based on the `cluster_strategy` config key
5. Predictions are clipped to zero (demand cannot be negative) and written to CSV
6. The shared loader script inserts predictions into Postgres

### Model IDs

| Algorithm | Per-Cluster ID | Global ID |
|-----------|---------------|-----------|
| LightGBM | `lgbm_cluster` | `lgbm_global` |

## Feature Engineering

All features are strictly causal -- only data available before the target month is used.

### Feature Set

| Category | Features | Count |
|----------|----------|-------|
| Lag features | `qty_lag_1` through `qty_lag_12` (demand shifted by N months) | 12 |
| Rolling std | `rolling_std_3m`, `rolling_std_6m`, `rolling_std_12m` (shifted by 1) | 3 |
| Calendar | `month` (1-12), `quarter` (1-4), `is_quarter_end`, `is_year_end`, `days_in_month` | 5 |
| Fourier | `fourier_sin_12/6/4/3`, `fourier_cos_12/6/4/3` (sub-annual seasonality) | 8 |
| DFU attributes | `execution_lag`, `total_lt`, `region`, `brand`, `abc_vol` | 5 |
| Item attributes | `case_weight`, `item_proof`, `bpc` | 3 |
| **Total** | | **~30** |

**Note:** `ml_cluster` was removed from the feature set to eliminate data leakage — cluster assignments are computed from full history and would leak future information into early backtest timeframes. `ml_cluster` is still used for per-cluster model *partitioning* (separate models per cluster). See spec 23 for the full feature selection pipeline.

### Categorical Feature Handling

| Algorithm | Approach | `cat_dtype` |
|-----------|----------|-------------|
| LightGBM | Native pandas `category` dtype | `"category"` |

### Grid Construction

A complete DFU x month grid is built before training to ensure lag features work correctly for zero-demand months. Sales values after the training cutoff are masked to zero to prevent future leakage.

## Algorithm-Specific Details

### LightGBM

| Parameter | Default |
|-----------|---------|
| `n_estimators` | 500 |
| `learning_rate` | 0.05 |
| `num_leaves` | 31 |
| `min_child_samples` | 20 |

Auto-detects Apple GPU (OpenCL) on macOS for accelerated training.


| Parameter | Default |
|-----------|---------|
| `iterations` | 500 |
| `learning_rate` | 0.05 |
| `depth` | 6 |
| `l2_leaf_reg` | 3.0 |

Uses ordered target encoding internally for categoricals. No manual encoding needed.


| Parameter | Default |
|-----------|---------|
| `n_estimators` | 500 |
| `learning_rate` | 0.05 |
| `max_depth` | 6 |
| `min_child_weight` | 5 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |


## Pipeline

| Target | Description |
|--------|-------------|
| `make backtest-lgbm` | Run LightGBM backtest |
| `make backtest-all` | Run all three sequentially |
| `make backtest-all-parallel` | Run all three in parallel (logs to `data/backtest/logs/`) |
| `make backtest-load MODEL=<id>` | Load predictions into Postgres |
| `make backtest-load-all` | Load all models |

## Configuration

All algorithm options are in `config/forecasting/forecast_pipeline_config.yaml`. See [Forecast Pipeline Config](./19-forecast-pipeline-config.md) for the complete reference. Each script accepts only `--config`, `--model-id`, and `--n-timeframes` as CLI arguments.

## Dependencies

- [Backtest Framework](./03-backtest-framework.md) -- shared orchestrator
- [Forecast Pipeline Config](./19-forecast-pipeline-config.md) -- controls all algorithm behavior
- Clustering (in `03-demand-intelligence/`) -- provides `ml_cluster` partition metadata

## See Also

- [Advanced Backtest](./05-advanced-backtest.md) -- hyperparameter tuning, SHAP, recursive mode
- [Champion Selection](./07-champion-selection.md) -- picks the best of these three models per DFU
