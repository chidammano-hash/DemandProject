# Feature 13: XGBoost Backtesting Implementation

## Objective
Implement the backtesting framework (Feature 8) with XGBoost models, supporting both global and per-cluster training strategies. Mirrors the LGBM implementation (Feature 9) with XGBoost-specific optimizations.

## Scope
- **Models**: XGBoost regressors for monthly demand forecasting
- **Strategies**: Global model (one XGBoost, `ml_cluster` as feature) and per-cluster (separate XGBoost per cluster)
- **Timeframes**: 10 expanding windows (A-J), auto-detected from data
- **Main table**: Each prediction stored at the DFU's `execution_lag` from `dim_dfu`
- **Archive table**: All lags 0-4 preserved in `backtest_lag_archive` for accuracy reporting at any horizon

## Model IDs

| Strategy | model_id | Description |
|----------|----------|-------------|
| Global | `xgboost_global` | One model for all DFUs, `ml_cluster` as feature |
| Per-cluster | `xgboost_cluster` | Separate model per `ml_cluster` |

## XGBoost-Specific Details

### Categorical Feature Handling
- XGBoost supports native categorical features via `enable_categorical=True` with `tree_method="hist"`
- Categorical columns stored as pandas `category` dtype
- No manual one-hot encoding required (XGBoost >= 2.0)

### Default Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 500 | Number of boosting rounds |
| `learning_rate` | 0.05 | Step size shrinkage |
| `max_depth` | 6 | Maximum tree depth |
| `min_child_weight` | 5 | Minimum sum of instance weight in a child |
| `subsample` | 0.8 | Row subsampling ratio |
| `colsample_bytree` | 0.8 | Column subsampling ratio per tree |
| `tree_method` | hist | Histogram-based tree method (enables categorical support) |
| `enable_categorical` | True | Native categorical feature support |
| `random_state` | 42 | Reproducibility |

### GPU Support
- XGBoost supports GPU training via `device="cuda"`
- Auto-detected at runtime; falls back to CPU if unavailable

## Feature Engineering

Identical to Feature 9 (LGBM) — all features are **strictly causal**.

### Lag Features (12)
- `qty_lag_1` through `qty_lag_12`: Historical demand shifted by N months

### Rolling Statistics (6)
- `rolling_mean_3m`, `rolling_mean_6m`, `rolling_mean_12m` (shifted by 1)
- `rolling_std_3m`, `rolling_std_6m`, `rolling_std_12m` (shifted by 1)

### Calendar Features (4)
- `month` (1-12), `quarter` (1-4), `month_sin`, `month_cos`

### DFU Attributes
- `ml_cluster` (categorical), `execution_lag`, `total_lt`, `region`, `brand`, `abc_vol`

### Item Attributes
- `case_weight`, `item_proof`, `bpc`

### Grid Construction
- Complete (DFU x month) grid ensures lag features work for zero-demand months
- Sales data masked at `train_end` cutoff to prevent future leakage

## Lag Strategy

### Main Table (`fact_external_forecast_monthly`)
Predictions stored **only at execution lag**:
- `fcstdate = startdate - execution_lag months`
- `lag = execution_lag` for every row

### Archive Table (`backtest_lag_archive`)
All lags 0-4 preserved:
- Same prediction expanded to 5 rows (lag 0, 1, 2, 3, 4)
- Includes `timeframe` column (A-J) for traceability
- Unique on `(forecast_ck, model_id, lag)`

## Implementation

### Scripts

| Script | Purpose |
|--------|---------|
| `mvp/demand/scripts/run_backtest_xgboost.py` | XGBoost-specific training functions + argparse (imports shared framework from `common/`) |
| `mvp/demand/common/backtest_framework.py` | Shared orchestrator: data loading, timeframes, feature engineering, output saving, MLflow |
| `mvp/demand/common/feature_engineering.py` | Shared feature matrix construction (uses `cat_dtype="category"` for XGBoost's native categoricals) |
| `mvp/demand/scripts/load_backtest_forecasts.py` | Bulk load predictions into Postgres (main + archive) — shared with LGBM/CatBoost |

The script contains only three XGBoost-specific functions (`train_and_predict_global`, `train_and_predict_per_cluster`, `train_and_predict_transfer`) passed to `run_tree_backtest()` from the shared framework.

### run_backtest_xgboost.py
Parameters: `--cluster-strategy`, `--model-id`, `--n-timeframes`, `--output-dir`, `--n-estimators`, `--learning-rate`, `--max-depth`, `--min-child-weight`, `--subsample`, `--colsample-bytree`, `--verbosity`

Output:
- `backtest_predictions.csv`: Execution-lag only (for main table)
- `backtest_predictions_all_lags.csv`: All lags 0-4 (for archive)
- `backtest_metadata.json`, `feature_importance.csv`

### load_backtest_forecasts.py
Shared loader — same script used for LGBM, CatBoost, and XGBoost. Model-agnostic; operates on `model_id` column in CSV.

## Makefile Targets

```makefile
backtest-xgboost:          # Global XGBoost backtest
backtest-xgboost-cluster:  # Per-cluster XGBoost backtest
backtest-load:             # Load predictions into Postgres (main + archive) — shared
backtest-all:              # backtest-lgbm + backtest-load (unchanged)
```

## Schema

No schema changes required. Uses existing:
- `fact_external_forecast_monthly` with `model_id` support (Feature 6)
- `backtest_lag_archive` (Feature 8/9)

## Verification

```bash
cd mvp/demand && uv sync          # Install dependencies (includes xgboost)
make db-apply-sql                  # Ensure tables exist
make backtest-xgboost              # Run global backtest
make backtest-load                 # Load main + archive
curl "http://localhost:8000/domains/forecast/models"
curl "http://localhost:8000/domains/forecast/analytics?model=xgboost_global"
make backtest-xgboost-cluster      # Per-cluster backtest
make backtest-load                 # Reload
```

## Dependencies
- Feature 8 (backtesting framework)
- Feature 7 (clustering)
- Feature 4 (fact tables)
- xgboost >= 2.0.0, python-dateutil >= 2.8.0
