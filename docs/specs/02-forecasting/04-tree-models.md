# Tree Model Implementations

> Three gradient-boosted tree algorithms (LightGBM, CatBoost, XGBoost) compete to forecast demand for every product, each with strengths for different demand patterns.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Accuracy, Item Analysis |
| **Key Files** | `scripts/run_backtest.py`, `scripts/run_backtest_catboost.py`, `scripts/run_backtest_xgboost.py`, `common/backtest_framework.py`, `common/feature_engineering.py` |

---

## Problem

A single algorithm cannot optimally forecast all demand patterns. Seasonal products behave differently from intermittent ones; high-volume items have different error profiles than slow movers. Relying on one algorithm means accepting suboptimal accuracy for large portions of the portfolio.

## Solution

Three gradient-boosted tree algorithms -- LightGBM, CatBoost, and XGBoost -- are trained using a shared backtest framework. All three use identical feature engineering and the same expanding-window evaluation protocol, making their accuracy scores directly comparable. Each algorithm can train one model per cluster (per_cluster strategy) or one model across all data (global strategy). Champion selection then picks the winner for each item-location-month.

## How It Works

1. Each algorithm reads its configuration from `config/forecast_pipeline_config.yaml`
2. The shared framework loads sales data, builds the feature matrix, and generates timeframes
3. The algorithm-specific script provides `train_and_predict_per_cluster()` and `train_and_predict_global()` functions
4. The framework calls the appropriate function based on the `cluster_strategy` config key
5. Predictions are clipped to zero (demand cannot be negative) and written to CSV
6. The shared loader script inserts predictions into Postgres

### Model IDs

| Algorithm | Per-Cluster ID | Global ID |
|-----------|---------------|-----------|
| LightGBM | `lgbm_cluster` | `lgbm_global` |
| CatBoost | `catboost_cluster` | `catboost_global` |
| XGBoost | `xgboost_cluster` | `xgboost_global` |

## Feature Engineering

All features are strictly causal -- only data available before the target month is used.

### Feature Set

| Category | Features | Count |
|----------|----------|-------|
| Lag features | `qty_lag_1` through `qty_lag_12` (demand shifted by N months) | 12 |
| Rolling means | `rolling_mean_3m`, `rolling_mean_6m`, `rolling_mean_12m` (shifted by 1) | 3 |
| Rolling std | `rolling_std_3m`, `rolling_std_6m`, `rolling_std_12m` (shifted by 1) | 3 |
| Calendar | `month` (1-12), `quarter` (1-4), `is_quarter_end`, `is_year_end`, `days_in_month` | 5 |
| Fourier | `fourier_sin_12/6/4/3`, `fourier_cos_12/6/4/3` (sub-annual seasonality) | 8 |
| DFU attributes | `ml_cluster`, `execution_lag`, `total_lt`, `region`, `brand`, `abc_vol` | 6 |
| Item attributes | `case_weight`, `item_proof`, `bpc` | 3 |
| **Total** | | **~31** |

**Key convention:** `ml_cluster` is always a hard feature -- it is never removed from the feature set in either per_cluster or global mode. In per_cluster mode it provides a constant identity signal; in global mode it provides inter-cluster discrimination.

### Categorical Feature Handling

| Algorithm | Approach | `cat_dtype` |
|-----------|----------|-------------|
| LightGBM | Native pandas `category` dtype | `"category"` |
| CatBoost | String dtype + column index list via `cat_features` parameter | `"str"` |
| XGBoost | Native `category` dtype + `enable_categorical=True` + `tree_method="hist"` | `"category"` |

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

### CatBoost

| Parameter | Default |
|-----------|---------|
| `iterations` | 500 |
| `learning_rate` | 0.05 |
| `depth` | 6 |
| `l2_leaf_reg` | 3.0 |

Uses ordered target encoding internally for categoricals. No manual encoding needed.

### XGBoost

| Parameter | Default |
|-----------|---------|
| `n_estimators` | 500 |
| `learning_rate` | 0.05 |
| `max_depth` | 6 |
| `min_child_weight` | 5 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |

Requires XGBoost >= 2.0 for native categorical support.

## Pipeline

| Target | Description |
|--------|-------------|
| `make backtest-lgbm` | Run LightGBM backtest |
| `make backtest-catboost` | Run CatBoost backtest |
| `make backtest-xgboost` | Run XGBoost backtest |
| `make backtest-all` | Run all three sequentially |
| `make backtest-all-parallel` | Run all three in parallel (logs to `data/backtest/logs/`) |
| `make backtest-load MODEL=<id>` | Load predictions into Postgres |
| `make backtest-load-all` | Load all models |

## Configuration

All algorithm options are in `config/forecast_pipeline_config.yaml`. See [Algorithm Config](./06-algorithm-config.md) for the complete reference. Each script accepts only `--config`, `--model-id`, and `--n-timeframes` as CLI arguments.

## Dependencies

- [Backtest Framework](./03-backtest-framework.md) -- shared orchestrator
- [Algorithm Config](./06-algorithm-config.md) -- controls all algorithm behavior
- Clustering (in `03-demand-intelligence/`) -- provides `ml_cluster` feature
- Python packages: `lightgbm>=4.0`, `catboost>=1.2`, `xgboost>=2.0`

## See Also

- [Advanced Backtest](./05-advanced-backtest.md) -- hyperparameter tuning, SHAP, recursive mode
- [Champion Selection](./07-champion-selection.md) -- picks the best of these three models per DFU
