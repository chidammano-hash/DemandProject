# Backtest Framework

> Tests forecast models against historical data across 10 expanding time windows, so you can measure accuracy before deploying to production.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Accuracy |
| **Key Files** | `common/backtest_framework.py`, `common/feature_engineering.py`, `common/metrics.py`, `common/constants.py`, `scripts/load_backtest_forecasts.py`, `sql/010_create_backtest_lag_archive.sql` |

---

## Problem

You cannot improve what you cannot measure. Without backtesting, the only way to know if a new algorithm is better is to deploy it and wait months for actuals to come in. This is slow, risky, and provides no statistical rigor. Planners need a way to evaluate models against known history before trusting them with real purchasing decisions.

## Solution

The backtesting framework trains models on progressively larger slices of history (expanding windows) and predicts forward into periods where actuals already exist. By comparing predictions to actuals across 10 time windows and 5 lag horizons, the platform produces a statistically robust accuracy profile for every model. A dual-path storage design preserves all lag horizons in an archive while storing only the operationally relevant execution-lag prediction in the main table.

## How It Works

1. The framework generates 10 expanding-window timeframes labeled A through J
2. Timeframe A trains on the shortest history; timeframe J trains on the longest
3. For each timeframe, the model predicts all remaining months up to the latest data
4. Each prediction is stored at 5 lag horizons (lag 0 through lag 4) in the archive
5. Only the execution-lag prediction (the lag that matters operationally for each DFU) goes into the main forecast table
6. After loading, 5 materialized views are refreshed for instant accuracy queries
7. The `model_id` column distinguishes predictions from different algorithms

### Expanding Window Example

With sales data from Feb 2023 to Jan 2026 (36 months), 10 timeframes:

| Timeframe | Training Period | Prediction Period | Available Lags |
|-----------|----------------|-------------------|---------------|
| A | Feb 2023 -- Mar 2025 | Apr 2025 -- Jan 2026 (10 months) | 0-4 |
| B | Feb 2023 -- Apr 2025 | May 2025 -- Jan 2026 (9 months) | 0-4 |
| ... | ... | ... | ... |
| I | Feb 2023 -- Nov 2025 | Dec 2025 -- Jan 2026 (2 months) | 0-1 |
| J | Feb 2023 -- Dec 2025 | Jan 2026 (1 month) | 0 only |

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

## Configuration

Backtest behavior is controlled by `config/algorithm_config.yaml`. See [Algorithm Config](./06-algorithm-config.md) for details.

## Dependencies

- [Multi-Model Support](./02-multi-model.md) -- `model_id` column in the forecast table
- Clustering (in `03-demand-intelligence/`) -- provides `ml_cluster` feature and per-cluster training

## See Also

- [Tree Models](./04-tree-models.md) -- the three algorithms that use this framework
- [Advanced Backtest](./05-advanced-backtest.md) -- tuning, SHAP, and recursive extensions
- [Algorithm Config](./06-algorithm-config.md) -- config file that controls backtest behavior
