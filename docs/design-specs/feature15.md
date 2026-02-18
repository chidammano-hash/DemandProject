# Feature 15: LGBM Backtesting Implementation

## Objective

Implement the backtesting framework (feature14.md) with LightGBM models, supporting both global and per-cluster training strategies.

## Scope

- **Models**: LightGBM regressors for monthly demand forecasting
- **Strategies**: Global model (one LGBM, `ml_cluster` as feature) and per-cluster (separate LGBM per cluster)
- **Timeframes**: 10 expanding windows (A–J), auto-detected from data
- **Main table**: Each prediction stored at the DFU's `execution_lag` from `dim_dfu`
- **Archive table**: All lags 0–4 preserved in `backtest_lag_archive` for accuracy reporting at any horizon

## Model IDs

| Strategy     | model_id        | Description                                      |
|-------------|-----------------|--------------------------------------------------|
| Global       | `lgbm_global`   | One model for all DFUs, `ml_cluster` as feature   |
| Per-cluster  | `lgbm_cluster`  | Separate model per `ml_cluster`, stored under one ID |

## Feature Engineering

All features are **strictly causal** — only data available before the target month is used.

### Lag Features (12)
- `qty_lag_1` through `qty_lag_12`: Historical demand shifted by N months

### Rolling Statistics (6)
- `rolling_mean_3m`, `rolling_mean_6m`, `rolling_mean_12m`: Rolling mean (shifted by 1)
- `rolling_std_3m`, `rolling_std_6m`, `rolling_std_12m`: Rolling std (shifted by 1)

### Calendar Features (4)
- `month` (1–12), `quarter` (1–4)
- `month_sin`, `month_cos`: Cyclical encoding

### DFU Attributes
- `ml_cluster` (categorical), `execution_lag`, `total_lt`
- `region`, `brand`, `abc_vol` (categorical)

### Item Attributes
- `case_weight`, `item_proof`, `bpc` (numeric)

### Grid Construction
- Complete (DFU × month) grid built to ensure lag features work for months with zero sales
- Sales data masked at `train_end` cutoff to prevent future leakage

## Timeframe Logic

See feature14.md for full specification. Summary:

```
For timeframe i (A=0, ..., J=9):
  train_end = latest_month - (N - i) months
  predict   = [train_end + 1 month, latest_month]
```

Detected automatically from `fact_sales_monthly` date range.

## Lag Strategy

### Main Table (`fact_external_forecast_monthly`)

Predictions stored **only at execution lag**:

- Each DFU has an `execution_lag` in `dim_dfu` (the lag at which the forecast is actually consumed)
- `fcstdate = startdate - execution_lag months`
- `lag = execution_lag` for every row
- Accuracy computation is direct — no filtering by `lag == execution_lag` needed
- ~5× fewer rows than full lag 0–4 expansion

### Archive Table (`backtest_lag_archive`)

All lags 0–4 preserved for accuracy reporting at any horizon:

- Same prediction expanded to 5 rows (lag 0, 1, 2, 3, 4) with `fcstdate = startdate - lag months`
- Includes `timeframe` column (A–J) for traceability
- Unique on `(forecast_ck, model_id, lag)` — one row per lag per forecast
- Enables accuracy analysis at any lag, not just execution lag

## Implementation

### Scripts

| Script | Purpose |
|--------|---------|
| `mvp/demand/scripts/run_backtest.py` | Train LGBM + generate predictions for all timeframes |
| `mvp/demand/scripts/load_backtest_forecasts.py` | Bulk load predictions into Postgres (main + archive) |

### run_backtest.py

**Parameters:**
- `--cluster-strategy`: `global` (default) or `per_cluster`
- `--model-id`: Override model_id
- `--n-timeframes`: Number of expanding windows (default: 10)
- `--output-dir`: Output directory (default: `data/backtest`)
- `--n-estimators`, `--learning-rate`, `--num-leaves`, `--min-child-samples`: LGBM hyperparams

**Output Files:**
- `data/backtest/backtest_predictions.csv`: Execution-lag only (for `fact_external_forecast_monthly`)
- `data/backtest/backtest_predictions_all_lags.csv`: All lags 0–4 (for `backtest_lag_archive`)
- `data/backtest/backtest_metadata.json`: Timeframes, params, accuracy summary
- `data/backtest/feature_importance.csv`: Feature importance (global strategy)

### load_backtest_forecasts.py

**Parameters:**
- `--input`: CSV path (default: `data/backtest/backtest_predictions.csv`)
- `--model-id`: Filter to specific model_id
- `--replace`: Delete existing rows for model_id before inserting

**Pattern:** COPY → temp staging table → INSERT with type casting + ON CONFLICT upsert → refresh agg view. Automatically loads archive CSV from same directory if present.

## Makefile Targets

```makefile
backtest-lgbm:          # Global LGBM backtest
backtest-lgbm-cluster:  # Per-cluster LGBM backtest
backtest-load:          # Load predictions into Postgres (main + archive)
backtest-all:           # backtest-lgbm + backtest-load
```

## Schema

### Main table

Uses existing `fact_external_forecast_monthly` with `model_id` support (feature11).

### Archive table (`backtest_lag_archive`)

New table in `mvp/demand/sql/010_create_backtest_lag_archive.sql`:

| Column | Type | Description |
|--------|------|-------------|
| `archive_sk` | BIGSERIAL PK | Surrogate key |
| `forecast_ck` | TEXT NOT NULL | Composite business key |
| `dmdunit` | TEXT NOT NULL | Item |
| `dmdgroup` | TEXT NOT NULL | Product group |
| `loc` | TEXT NOT NULL | Location |
| `fcstdate` | DATE NOT NULL | Forecast creation date (month-start) |
| `startdate` | DATE NOT NULL | Actual month being forecast (month-start) |
| `lag` | INTEGER NOT NULL | 0–4, months between fcstdate and startdate |
| `execution_lag` | INTEGER | DFU's execution lag |
| `basefcst_pref` | NUMERIC(18,4) | Forecast value |
| `tothist_dmd` | NUMERIC(18,4) | Actual demand |
| `model_id` | TEXT NOT NULL | Model identifier |
| `timeframe` | TEXT | Backtest timeframe label (A–J) |
| `load_ts` | TIMESTAMPTZ | Record load timestamp |

**Constraints:** `UNIQUE(forecast_ck, model_id, lag)`, lag 0–4, month-start checks, lag-matches-dates check.

## Dependencies

Added to `pyproject.toml`:
- `lightgbm>=4.0.0`
- `python-dateutil>=2.8.0`

## Verification

```bash
# 1. Install dependencies
cd mvp/demand && uv sync

# 2. Apply schema (creates backtest_lag_archive)
make db-apply-sql

# 3. Run global backtest (produces both CSVs)
make backtest-lgbm

# 4. Load into Postgres (main + archive)
make backtest-load

# 5. Check main table
curl "http://localhost:8000/domains/forecast/models"
curl "http://localhost:8000/domains/forecast/analytics?model=lgbm_global"

# 6. Check archive table
psql -h localhost -p 5440 -U demand -d demand_mvp \
  -c "SELECT model_id, lag, COUNT(*) FROM backtest_lag_archive GROUP BY 1,2 ORDER BY 1,2"

# 7. Run per-cluster backtest
make backtest-lgbm-cluster
make backtest-load
```

## Integration

- **UI**: Select `lgbm_global` or `lgbm_cluster` in the model dropdown (feature11)
- **KPIs**: Existing feature10 engine computes accuracy automatically from main table
- **Comparison**: Compare `external` vs `lgbm_global` vs `lgbm_cluster` in the model selector
- **Lag Analysis**: Query `backtest_lag_archive` directly for accuracy at any lag horizon
