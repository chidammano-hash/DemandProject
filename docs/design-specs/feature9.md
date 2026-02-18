# Feature 9: LGBM Backtesting Implementation

## Objective
Implement the backtesting framework (Feature 8) with LightGBM models, supporting both global and per-cluster training strategies.

## Scope
- **Models**: LightGBM regressors for monthly demand forecasting
- **Strategies**: Global model (one LGBM, `ml_cluster` as feature) and per-cluster (separate LGBM per cluster)
- **Timeframes**: 10 expanding windows (A-J), auto-detected from data
- **Main table**: Each prediction stored at the DFU's `execution_lag` from `dim_dfu`
- **Archive table**: All lags 0-4 preserved in `backtest_lag_archive` for accuracy reporting at any horizon

## Model IDs

| Strategy | model_id | Description |
|----------|----------|-------------|
| Global | `lgbm_global` | One model for all DFUs, `ml_cluster` as feature |
| Per-cluster | `lgbm_cluster` | Separate model per `ml_cluster` |

## Feature Engineering

All features are **strictly causal** — only data available before the target month is used.

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
- ~5x fewer rows than full lag 0-4 expansion

### Archive Table (`backtest_lag_archive`)
All lags 0-4 preserved:
- Same prediction expanded to 5 rows (lag 0, 1, 2, 3, 4)
- Includes `timeframe` column (A-J) for traceability
- Unique on `(forecast_ck, model_id, lag)`

## Implementation

### Scripts

| Script | Purpose |
|--------|---------|
| `mvp/demand/scripts/run_backtest.py` | Train LGBM + generate predictions for all timeframes |
| `mvp/demand/scripts/load_backtest_forecasts.py` | Bulk load predictions into Postgres (main + archive) |

### run_backtest.py
Parameters: `--cluster-strategy`, `--model-id`, `--n-timeframes`, `--output-dir`, `--n-estimators`, `--learning-rate`, `--num-leaves`, `--min-child-samples`

Output:
- `backtest_predictions.csv`: Execution-lag only (for main table)
- `backtest_predictions_all_lags.csv`: All lags 0-4 (for archive)
- `backtest_metadata.json`, `feature_importance.csv`

### load_backtest_forecasts.py
Parameters: `--input`, `--model-id`, `--replace`

Pattern: COPY -> temp staging -> INSERT with upsert -> refresh agg view. Auto-loads archive CSV if present.

## Makefile Targets

```makefile
backtest-lgbm:          # Global LGBM backtest
backtest-lgbm-cluster:  # Per-cluster LGBM backtest
backtest-load:          # Load predictions into Postgres (main + archive)
backtest-all:           # backtest-lgbm + backtest-load
```

## Schema

### Main table
Uses existing `fact_external_forecast_monthly` with `model_id` support (Feature 6).

### Archive table (`backtest_lag_archive`)
New table in `mvp/demand/sql/010_create_backtest_lag_archive.sql`:

| Column | Type | Description |
|--------|------|-------------|
| `archive_sk` | BIGSERIAL PK | Surrogate key |
| `forecast_ck` | TEXT NOT NULL | Composite business key |
| `dmdunit` | TEXT NOT NULL | Item |
| `dmdgroup` | TEXT NOT NULL | Product group |
| `loc` | TEXT NOT NULL | Location |
| `fcstdate` | DATE NOT NULL | Forecast creation date |
| `startdate` | DATE NOT NULL | Actual month being forecast |
| `lag` | INTEGER NOT NULL | 0-4 |
| `execution_lag` | INTEGER | DFU's execution lag |
| `basefcst_pref` | NUMERIC(18,4) | Forecast value |
| `tothist_dmd` | NUMERIC(18,4) | Actual demand |
| `model_id` | TEXT NOT NULL | Model identifier |
| `timeframe` | TEXT | Backtest timeframe (A-J) |
| `load_ts` | TIMESTAMPTZ | Record load timestamp |

Constraints: `UNIQUE(forecast_ck, model_id, lag)`, lag 0-4, month-start checks.

## Verification

```bash
cd mvp/demand && uv sync          # Install dependencies
make db-apply-sql                  # Create backtest_lag_archive
make backtest-lgbm                 # Run global backtest
make backtest-load                 # Load main + archive
curl "http://localhost:8000/domains/forecast/models"
curl "http://localhost:8000/domains/forecast/analytics?model=lgbm_global"
make backtest-lgbm-cluster         # Per-cluster backtest
make backtest-load                 # Reload
```

## Dependencies
- Feature 8 (backtesting framework)
- Feature 7 (clustering)
- Feature 4 (fact tables)
- lightgbm >= 4.0.0, python-dateutil >= 2.8.0
