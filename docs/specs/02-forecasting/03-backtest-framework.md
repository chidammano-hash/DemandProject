# Backtest Framework

> Tests forecast models against historical data across 10 expanding time windows, so you can measure accuracy before deploying to production.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Accuracy |
| **Key Files** | `common/backtest_framework.py`, `common/ml/model_registry.py`, `common/feature_engineering.py`, `common/metrics.py`, `common/constants.py`, `scripts/load_backtest_forecasts.py`, `sql/010_create_backtest_lag_archive.sql` |

---

## Problem

You cannot improve what you cannot measure. Without backtesting, the only way to know if a new algorithm is better is to deploy it and wait months for actuals to come in. This is slow, risky, and provides no statistical rigor. Planners need a way to evaluate models against known history before trusting them with real purchasing decisions.

## Solution

The backtesting framework trains models on progressively larger slices of history (expanding windows) and predicts forward into periods where actuals already exist. By comparing predictions to actuals across 10 time windows and 5 lag horizons, the platform produces a statistically robust accuracy profile for every model. A dual-path storage design preserves all lag horizons in an archive while storing only the operationally relevant execution-lag prediction in the main table.

## How It Works

1. The framework generates 10 expanding-window timeframes labeled A through J
2. Timeframe A trains on the shortest history; timeframe J trains on the longest
3. For each timeframe, the model predicts all remaining months up to the latest data
4. Each prediction's **natural lag** is computed from its timeframe: `lag = months(startdate - train_end) - 1`. For a given demand month, different timeframes produce predictions at different horizons (lags 0-4), with genuinely different `basefcst_pref` values because the model was trained on different data cutoffs
5. Only the execution-lag prediction (the lag that matters operationally for each DFU) goes into the main forecast table
6. After loading, 5 materialized views are refreshed for instant accuracy queries
7. The `model_id` column distinguishes predictions from different algorithms

### Expanding Window Example

With sales data from Feb 2023 to Jan 2026 (36 months), 10 timeframes:

| Timeframe | Training Period | Prediction Period | Natural Lags (for Jan 2026) |
|-----------|----------------|-------------------|---------------------------|
| A | Feb 2023 -- Mar 2025 | Apr 2025 -- Jan 2026 (10 months) | Jan: lag 9 (filtered out, >4) |
| ... | ... | ... | ... |
| F | Feb 2023 -- Aug 2025 | Sep 2025 -- Jan 2026 (5 months) | Jan: **lag 4** (5-month-ahead) |
| G | Feb 2023 -- Sep 2025 | Oct 2025 -- Jan 2026 (4 months) | Jan: **lag 3** (4-month-ahead) |
| H | Feb 2023 -- Oct 2025 | Nov 2025 -- Jan 2026 (3 months) | Jan: **lag 2** (3-month-ahead) |
| I | Feb 2023 -- Nov 2025 | Dec 2025 -- Jan 2026 (2 months) | Jan: **lag 1** (2-month-ahead) |
| J | Feb 2023 -- Dec 2025 | Jan 2026 (1 month) | Jan: **lag 0** (1-month-ahead) |

Each lag for Jan 2026 comes from a different timeframe with a different training data cutoff, producing genuinely different predictions. Lag 0 uses the most recent data (highest accuracy), lag 4 uses 5-month-old data (lowest accuracy).

### Execution Lag

Each DFU (Demand Forecast Unit -- a unique item-location combination) has an `execution_lag` that represents how far in advance its forecast is issued. A DFU with `execution_lag = 2` means the forecast for April is issued in February. The main table stores only the prediction at this operationally relevant lag; the archive stores all 5 lags for accuracy analysis at any horizon.

**External forecast loading:** All rows in `dfu_stat_fcst.txt` are assumed to be at execution lag. The `lag` and `execution_lag` fields in the source file are ignored — the loader overwrites both from `dim_sku.execution_lag` (defaulting to 0 for unmatched DFUs). No `WHERE lag = execution_lag` filter is applied; all rows are inserted. Additionally, only the last 12 months of data (by `startdate`) are loaded, based on the current planning date.

**Backtest loading:** Backtests still produce predictions at all 5 lags (0-4). The backtest loader (`scripts/load_backtest_forecasts.py`) retains the original dual-path logic: archive gets all lags, main table gets execution-lag rows only.

## Data Model

### Main Table: `fact_external_forecast_monthly`

Stores execution-lag predictions only. One row per DFU per month per model.

### Archive Table: `backtest_lag_archive`

| Column | Type | Description |
|--------|------|-------------|
| `forecast_ck` | TEXT | Composite business key |
| `item_id`, `loc` | TEXT | Item and location |
| `fcstdate` | DATE | When the forecast was issued |
| `startdate` | DATE | Month being forecast |
| `lag` | INTEGER | 0-4 (months between issue and target) |
| `basefcst_pref` | NUMERIC | Forecast quantity |
| `tothist_dmd` | NUMERIC | Actual demand |
| `model_id` | TEXT | Algorithm identifier |
| `timeframe` | TEXT | Backtest timeframe (A-J) |

**Constraint:** `UNIQUE(forecast_ck, model_id, lag)`

### Dual-Path Loading (Critical Ordering)

The loader uses phase ordering to preserve archive integrity:
1. **12-month filter** removes staging rows with `startdate` older than 12 months from planning date
2. **Archive load** inserts remaining rows into `backtest_lag_archive` FIRST from untouched staging data (original lag values preserved)
3. **Execution lag resolution** overwrites both `lag` and `execution_lag` on staging from `dim_sku`
4. **Main table INSERT** loads all rows (no lag filter — all external forecasts are assumed at execution lag)

This ordering is critical: the archive must be loaded before the staging mutation, otherwise all rows for a DFU would have the same lag value. Backtest model rows in the archive are not affected (only `model_id='external'` rows are replaced).

## API

No new endpoints. Existing multi-model endpoints handle backtest data automatically:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/domains/forecast/models` | Lists all model_ids including backtest models |
| GET | `/forecast/accuracy/slice` | Accuracy by dimension for any model |
| GET | `/forecast/accuracy/lag-curve` | Accuracy degradation across lags 0-4 |

## Pipeline

| Target | Description |
|--------|-------------|
| `make backtest-lgbm` | Run LightGBM backtest (10 timeframes) |
| `make backtest-catboost` | Run CatBoost backtest |
| `make backtest-xgboost` | Run XGBoost backtest |
| `make backtest-all` | Run all three sequentially |
| `make backtest-all-parallel` | Run all three in parallel |
| `make backtest-load MODEL=lgbm_cluster` | Load one model's predictions into Postgres |
| `make backtest-load-all` | Load all models found under `data/backtest/*/` |
| `make backtest-list` | Show row counts per model in forecast + archive tables |
| `make backtest-clean MODELS="lgbm_cluster"` | Remove specific model predictions |

### Output Directory Structure

Each backtest writes to `data/backtest/<model_id>/`:
- `backtest_predictions.csv` -- execution-lag only (for main table)
- `backtest_predictions_all_lags.csv` -- all lags 0-4 (for archive)
- `backtest_metadata.json` -- run configuration and metrics
- `feature_importance.csv` -- model feature rankings

## Model Registry (`common/ml/model_registry.py`)

The model registry provides a centralized abstraction layer for all tree-based models, eliminating duplicate code across backtest scripts:

### Canonical Parameter Mapping

| Canonical | LGBM | CatBoost | XGBoost |
|-----------|------|----------|---------|
| `estimators` | `n_estimators` | `iterations` | `n_estimators` |
| `max_depth` | `max_depth` | `depth` | `max_depth` |
| `l2_reg` | `reg_lambda` | `l2_leaf_reg` | `reg_lambda` |
| `l1_reg` | `reg_alpha` | _(not supported)_ | `reg_alpha` |
| `min_leaf_samples` | `min_child_samples` | `min_data_in_leaf` | `min_child_weight` |
| `col_sample` | `colsample_bytree` | `colsample_bylevel` | `colsample_bytree` |

### Unified Functions

- **`fit_model()`** — single fit function replacing 3× duplicate if/elif/else blocks in `_train_single_cluster` and `train_and_predict_global`
- **`get_best_iteration()`** — abstracts `best_iteration_` (LGBM/CatBoost) vs `best_iteration` (XGBoost)
- **`compute_early_stop_patience()`** — standardized 3% of max iterations (floor 10) for all models
- **`to_native_params()` / `from_native_params()`** — bidirectional canonical ↔ native translation

### Early Stopping Standardization

All models use `compute_early_stop_patience(max_iterations, pct=0.03)`:
- LGBM 1500 iterations → 45 rounds patience
- CatBoost 3000 iterations → 90 rounds patience
- XGBoost 500 iterations → 15 rounds patience

## Configuration

Backtest behavior is controlled by `config/algorithm_config.yaml`. See [Algorithm Config](./06-algorithm-config.md) for details.

Key backtest-level settings:
- `early_stop_pct: 0.03` — early stopping patience as percentage of max iterations
- `shap_retrain_threshold: 0.10` — retrain if >= 10% of features are dropped by SHAP

## Dependencies

- [Multi-Model Support](./02-multi-model.md) -- `model_id` column in the forecast table
- Clustering (in `03-demand-intelligence/`) -- provides `ml_cluster` feature and per-cluster training

## See Also

- [Tree Models](./04-tree-models.md) -- the three algorithms that use this framework
- [Advanced Backtest](./05-advanced-backtest.md) -- tuning, SHAP, and recursive extensions
- [Algorithm Config](./06-algorithm-config.md) -- config file that controls backtest behavior
