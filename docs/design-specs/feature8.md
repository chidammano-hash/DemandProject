# Feature 8: Backtesting Framework — Expanding Window Timeframes

## Objective
Build a generic backtesting framework that trains forecast models (LGBM, CatBoost, etc.) across multiple expanding-window timeframes, generates multi-lag predictions, stores results in `fact_external_forecast_monthly`, and measures accuracy at each DFU's execution lag.

## Motivation
- **Model comparison**: Evaluate multiple forecast algorithms side-by-side using the same backtesting structure.
- **Lag-aware accuracy**: Measuring at execution lag reflects operational reality.
- **Temporal robustness**: Expanding windows test model performance across different forecast origins.

## Core Concept: Expanding Window Timeframes

### Generic Logic
Given:
- `latest_month` = max(startdate) from `fact_sales_monthly` (detected from data)
- `earliest_month` = min(startdate) from `fact_sales_monthly` (detected from data)
- `N` = number of timeframes (default: 10, labeled A-J)
- `max_lag` = maximum forecast lag (default: 4, so lags 0-4)

For each timeframe `i` (0-indexed, A=0, B=1, ..., J=9):
```
train_end   = latest_month - (N - i) months
train_start = earliest_month
predict_start = train_end + 1 month
predict_end   = latest_month
```

### Concrete Example (current data)
Sales history: Feb 2023 - Jan 2026 (36 months). N=10.

| Timeframe | Train Period | Predict Period | Lags |
|-----------|-------------|----------------|------|
| A | Feb 2023 - Mar 2025 | Apr 2025 - Jan 2026 | 0-4 |
| B | Feb 2023 - Apr 2025 | May 2025 - Jan 2026 | 0-4 |
| ... | ... | ... | ... |
| I | Feb 2023 - Nov 2025 | Dec 2025 - Jan 2026 | 0-1 |
| J | Feb 2023 - Dec 2025 | Jan 2026 | 0 |

### Lag Matrix
Each predicted month accumulates forecasts from multiple timeframes at different lags. Months with all 5 lags: Aug 2025 - Jan 2026 (last `N - max_lag` = 6 months).

## Forecast Storage

### Mapping to `fact_external_forecast_monthly`
| Column | Value |
|--------|-------|
| `forecast_ck` | `{dmdunit}_{dmdgroup}_{loc}_{fcstdate}_{startdate}` |
| `fcstdate` | `predict_start` of the timeframe |
| `startdate` | Predicted month |
| `lag` | `month_diff(startdate, fcstdate)` — 0 to 4 |
| `execution_lag` | From `dim_dfu.execution_lag` |
| `basefcst_pref` | Model's predicted demand |
| `tothist_dmd` | Actual demand from `fact_sales_monthly` |
| `model_id` | e.g., `lgbm_v1`, `catboost_v1` |

### Storage Strategy
- **Main table** (`fact_external_forecast_monthly`): Only execution-lag forecasts
- **Archive table** (`backtest_lag_archive`): All lags 0-4 for any-horizon accuracy analysis

## Accuracy Measurement

### At Execution Lag
Each DFU's `execution_lag` in `dim_dfu` determines the forecast horizon that matters operationally. Accuracy at execution lag = filter where `lag = execution_lag` per DFU.

### Formulas (same as Feature 5)
- **WAPE**: `100 * SUM(ABS(F - A)) / ABS(SUM(A))`
- **Bias**: `(SUM(F) / SUM(A)) - 1`
- **Accuracy %**: `100 - WAPE`

## Cluster Integration (Feature 7)

| Strategy | Description |
|----------|-------------|
| Global + Cluster Feature | One model, `ml_cluster` as categorical input |
| Per-Cluster Models | Separate model per cluster |
| Hybrid | Global for large clusters, specialized for distinct ones |

The `model_id` distinguishes strategies (e.g., `lgbm_global`, `lgbm_cluster_0`).

## Feature Engineering for Tree Models

### Lag Features
- `qty_lag_1` through `qty_lag_12`: Demand from 1-12 months prior
- `qty_rolling_3m`, `qty_rolling_6m`, `qty_rolling_12m`: Rolling means
- `qty_rolling_std_3m`, `qty_rolling_std_6m`: Rolling standard deviations

### Calendar Features
- `month`, `quarter`, `month_sin`, `month_cos`

### DFU/Item Attributes
- `ml_cluster`, `execution_lag`, `total_lt`, `region`, `brand`, `case_weight`, `item_proof`

**Important**: All features must be **strictly causal** — only data available at forecast origin.

## Implementation

### Shared Framework (`common/`)
All tree-based backtest scripts (LGBM, CatBoost, XGBoost) share common logic via modules in `mvp/demand/common/`:

| Module | Purpose |
|--------|---------|
| `common/backtest_framework.py` | Orchestrator: `run_tree_backtest()`, timeframe generation, data loading, execution-lag assignment, all-lag expansion, post-processing, output saving |
| `common/feature_engineering.py` | `build_feature_matrix()`, `get_feature_columns()`, `mask_future_sales()` |
| `common/metrics.py` | `compute_accuracy_metrics()`: WAPE, bias, accuracy % |
| `common/mlflow_utils.py` | `log_backtest_run()`: generic MLflow experiment logging |
| `common/db.py` | `get_db_params()`: shared DB connection parameters |
| `common/constants.py` | `CAT_FEATURES`, `LAG_RANGE`, `ROLLING_WINDOWS`, output column ordering, thresholds |

Each model-specific script implements only three functions: `train_and_predict_global()`, `train_and_predict_per_cluster()`, `train_and_predict_transfer()`. These are passed as callables to `run_tree_backtest()`.

Prophet uses shared utilities (`generate_timeframes`, `load_backtest_data`, `postprocess_predictions`, `save_backtest_output`, `log_backtest_run`) but orchestrates its own per-DFU fitting loop.

### Model-Specific Scripts
| Script | Model |
|--------|-------|
| `mvp/demand/scripts/run_backtest.py` | LightGBM |
| `mvp/demand/scripts/run_backtest_catboost.py` | CatBoost |
| `mvp/demand/scripts/run_backtest_xgboost.py` | XGBoost |
| `mvp/demand/scripts/run_backtest_prophet.py` | Prophet |

### Loader Script: `mvp/demand/scripts/load_backtest_forecasts.py`
Parameters: `--input`, `--model-id`, `--replace`

Pattern: COPY -> staging -> INSERT with upsert -> refresh agg view. Auto-loads archive CSV if present.

### Makefile Targets
```makefile
backtest-lgbm:      # Global LGBM backtest
backtest-catboost:  # CatBoost backtest
backtest-xgboost:   # XGBoost backtest
backtest-prophet:   # Prophet backtest
backtest-load:      # Load predictions into Postgres (main + archive)
backtest-all:       # backtest-lgbm + backtest-load
```

## API Integration
No new endpoints required. Existing multi-model support (Feature 6) handles everything:
- `GET /domains/forecast/models` returns all model IDs
- `GET /domains/forecast/analytics?model=lgbm_v1` for per-model KPIs
- Optional: `GET /domains/forecast/backtest/summary?model_id=lgbm_v1`

## MLflow Integration
Experiment: `dfu_backtest`. Logs parameters, metrics (WAPE, bias, accuracy), artifacts (feature importance, accuracy plots).

## Dependencies
- Feature 3 (dimensions), Feature 4 (facts)
- Feature 5 (KPI engine), Feature 6 (multi-model)
- Feature 7 (clustering)
- lightgbm, catboost (optional), xgboost (optional)
