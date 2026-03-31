# Spec 17 — External ML Forecast Loading

> Load externally-generated ML backtest results into the database as new model IDs so they participate in all existing accuracy and lag-curve analytics without any API changes.

| | |
|---|---|
| **Status** | Implemented |
| **Model IDs** | `ext_lgbm`, `ext_cat`, `ext_xg`, `ext_best` |
| **Script** | `scripts/etl/load_ext_ml_forecasts.py` |
| **Config** | `config/ext_ml_forecasts.yaml` |
| **Key Tables** | `backtest_lag_archive`, `fact_external_forecast_monthly` |

---

## 1. Overview

The platform's internal ML pipeline trains LGBM, CatBoost, and XGBoost models using the backtest framework and stores their predictions under model IDs `lgbm`, `catboost`, and `xgboost`. Occasionally, forecast predictions generated outside the platform (for example, by an external modelling team or a partner system) need to be evaluated alongside these internal models using the same accuracy infrastructure.

This feature defines a standalone ETL script that ingests four externally-supplied CSV files and loads them into the database as first-class model IDs: `ext_lgbm`, `ext_cat`, `ext_xg`, and `ext_best`. Once loaded, every existing accuracy API endpoint — slicing, lag-curve analysis, and available-model enumeration — returns results for the external models automatically, with no API or frontend changes required.

### Relationship to Internal Backtests

External ML forecasts are stored using the same dual-path design as internal backtests:

- All lags 0–4 are stored in `backtest_lag_archive` — enabling lag-curve analysis
- All lags 0–4 are stored in `fact_external_forecast_monthly` — enabling accuracy-slice analysis

This contrasts with the `external` model (sourced from `dfu_stat_fcst.txt`), which stores only the execution-lag row per DFU per forecast date.

| Aspect | `external` (source-system) | `ext_lgbm` / `ext_cat` / `ext_xg` / `ext_best` |
|---|---|---|
| Lags stored | Execution lag only (1 row per DFU-month) | All lags 0–4 (5 rows per DFU-month) |
| Data source | Operational planning system | External ML results CSVs |
| Lag derivation | Overwritten from `dim_sku.execution_lag` | Provided directly in CSV (`LAG` column) |
| `timeframe` column | NULL | NULL |

---

## 2. Source Data

### 2.1 Input Files

Files are placed in `data/input/` before the load script is run.

| File | Assigned `model_id` |
|---|---|
| `df_ml_lgbm_l2_extract.csv` | `ext_lgbm` |
| `df_ml_cat_l2_extract.csv` | `ext_cat` |
| `df_ml_xg_l2_extract.csv` | `ext_xg` |
| `df_ml_best.csv` | `ext_best` |

### 2.2 Input CSV Schema

All four files share the same column layout:

| Column | Type | Description |
|---|---|---|
| `DFU` | string | Composite SKU key (`item_id_customer_group_loc`). Matched against `dim_sku.sku_ck`. |
| `FORECASTDATE` | date string | When the forecast was generated. Normalized to month-start → `fcstdate`. |
| `STARTDATE` | date string | Demand period start (the month being forecast). Normalized to month-start → `startdate`. |
| `LAG` | integer | Forecast horizon: 0 (1-month-ahead) through 4 (5-month-ahead). |
| `PREDICTED_ORDERS` | numeric | Forecast quantity → `basefcst_pref`. |
| `ACTUAL_ORDERS` | numeric | Observed demand for the period → `tothist_dmd`. |
| `FILE` | string | Source file path. Discarded during load. |

All five forecast lags (0–4) are present in each file. There are no separate execution-lag-only and all-lags files.

---

## 3. Column Mapping

### 3.1 To `fact_external_forecast_monthly`

| Source | Target Column | Transformation |
|---|---|---|
| `DFU` | JOIN key | Matched to `dim_sku.sku_ck`; unmatched rows dropped |
| (from `dim_sku`) | `item_id` | Resolved via JOIN |
| (from `dim_sku`) | `customer_group` | Resolved via JOIN |
| (from `dim_sku`) | `loc` | Resolved via JOIN |
| (from `dim_sku`) | `execution_lag` | Resolved via JOIN |
| `FORECASTDATE` | `fcstdate` | Truncated to month-start (`date_trunc('month', ...)`) |
| `STARTDATE` | `startdate` | Truncated to month-start |
| `LAG` | `lag` | Integer, filtered to 0–4 |
| `PREDICTED_ORDERS` | `basefcst_pref` | Cast to NUMERIC |
| `ACTUAL_ORDERS` | `tothist_dmd` | Cast to NUMERIC |
| (CLI argument) | `model_id` | e.g. `ext_lgbm` |
| (derived) | `forecast_ck` | `item_id \|\| '_' \|\| customer_group \|\| '_' \|\| loc \|\| '_' \|\| fcstdate \|\| '_' \|\| startdate` |
| (NULL) | `timeframe` | NULL — external models have no timeframe window label |

### 3.2 To `backtest_lag_archive`

Identical to `fact_external_forecast_monthly` with the addition of:

| Source | Target Column | Transformation |
|---|---|---|
| `LAG` | `lag` | Stored as integer 0–4 |
| (NULL) | `timeframe` | NULL |

`UNIQUE (forecast_ck, model_id, lag)` in `backtest_lag_archive` prevents duplicate rows on re-run.

### 3.3 Key Differences from Internal Backtest Mapping

| Aspect | Internal LGBM/CatBoost/XGBoost | External ML (`ext_*`) |
|---|---|---|
| DFU identifier | Split columns (`item_id`, `customer_group`, `loc`) | Single `DFU` column (`sku_ck`) |
| Forecast qty column | `basefcst_pref` | `PREDICTED_ORDERS` (renamed on load) |
| Actual qty column | `tothist_dmd` | `ACTUAL_ORDERS` (renamed on load) |
| `model_id` source | Derived from script/config | Passed as CLI argument |
| Lag structure | Computed via `assign_natural_lags()` across 10 timeframes | Pre-computed in CSV |
| `timeframe` | Timeframe label (A–J) | NULL |

---

## 4. Load Pipeline

### 4.1 Diagram

```
data/input/df_ml_<model>.csv
        |
        v
  COPY into _stg_ext_ml (temp table)
        |
        v
  JOIN dim_sku ON sku_ck = DFU
  (unmatched DFUs logged and dropped)
        |
        v
  Normalize dates to month-start
  Filter lags to 0-4
  Construct forecast_ck
        |
        +---------------------------+
        |                           |
        v                           v
  INSERT all lags (0-4)       INSERT all lags (0-4)
  fact_external_forecast      backtest_lag_archive
  _monthly                    (timeframe = NULL)
  ON CONFLICT DO UPDATE       ON CONFLICT DO UPDATE
        |                           |
        +---------------------------+
                    |
                    v
        REFRESH MATERIALIZED VIEWS
        (5 views, in dependency order)
```

### 4.2 Staging Table

A temporary table `_stg_ext_ml` is created at the start of each run and dropped on completion. It mirrors the raw CSV columns exactly, allowing bulk `COPY` followed by a typed JOIN-and-insert.

```sql
CREATE TEMP TABLE _stg_ext_ml (
    dfu            TEXT,
    forecastdate   TEXT,
    startdate      TEXT,
    lag            INTEGER,
    predicted_orders NUMERIC,
    actual_orders    NUMERIC,
    file           TEXT
);
```

### 4.3 Upsert Behaviour

Both target tables use `ON CONFLICT DO UPDATE` (upsert). Re-running the loader for the same model replaces existing rows rather than raising errors or producing duplicates. This makes the script safely re-entrant after data corrections.

- `fact_external_forecast_monthly`: conflict target is `(forecast_ck, model_id)`
- `backtest_lag_archive`: conflict target is `(forecast_ck, model_id, lag)`

### 4.4 MV Refresh Order

After all rows are inserted, the following materialized views are refreshed in dependency order:

| Order | View | Purpose |
|---|---|---|
| 1 | `agg_forecast_monthly` | Base forecast aggregate |
| 2 | `agg_accuracy_by_dim` | Accuracy by dimension (powers accuracy-slice API) |
| 3 | `agg_dfu_coverage` | DFU coverage counts |
| 4 | `agg_accuracy_lag_archive` | Per-lag accuracy (powers lag-curve API) |
| 5 | `agg_dfu_coverage_lag_archive` | DFU coverage at each lag |

### 4.5 Unmatched DFU Handling

Rows whose `DFU` value does not exist in `dim_sku.sku_ck` are dropped. The count of dropped rows is logged as a warning. A high drop rate (configurable threshold in `ext_ml_forecasts.yaml`) raises a non-fatal error with a summary.

---

## 5. Target Tables

### 5.1 `fact_external_forecast_monthly`

Stores the production-facing forecast rows used by accuracy-slice endpoints. Each row represents one (DFU, model, forecast date, demand period) combination. Because all five lags share different `startdate` values, each lag produces a distinct `forecast_ck` and therefore a distinct row.

Relevant uniqueness constraint: `UNIQUE (forecast_ck, model_id)`.

### 5.2 `backtest_lag_archive`

Stores all lag predictions explicitly alongside their `lag` integer for lag-curve analysis. The `timeframe` column is NULL for all `ext_*` models — the archive does not assign these rows to an expanding-window timeframe since the model was trained externally.

Relevant uniqueness constraint: `UNIQUE (forecast_ck, model_id, lag)`.

---

## 6. No DDL Required

Both target tables already support arbitrary `model_id` string values. The `timeframe` column in `backtest_lag_archive` is already nullable. No schema changes, new tables, or new columns are needed.

The five materialized views that power the accuracy APIs already aggregate by `model_id` dynamically — adding `ext_lgbm` rows causes these views to include the new model automatically after refresh.

---

## 7. API Availability

All existing accuracy endpoints discover models from the data, not from a static list. After loading, the following endpoints return results for `ext_*` models with no code changes:

| Endpoint | Behaviour |
|---|---|
| `GET /forecast/accuracy/available-models` | Returns `ext_lgbm`, `ext_cat`, `ext_xg`, `ext_best` in the model list |
| `GET /forecast/accuracy/slice?models=ext_lgbm,ext_cat` | Returns accuracy broken down by dimension for the specified external models |
| `GET /forecast/accuracy/lag-curve?models=ext_lgbm,ext_cat` | Returns accuracy at each lag horizon (0–4) for the specified external models |

To compare an external model against an internal one, pass a comma-separated list:

```
GET /forecast/accuracy/lag-curve?models=lgbm,ext_lgbm
```

---

## 8. Configuration

`config/ext_ml_forecasts.yaml` externalises all file paths, model ID assignments, and validation thresholds.

```yaml
# External ML forecast loading configuration

input_base: "data/input"

models:
  ext_lgbm:
    file: "df_ml_lgbm_l2_extract.csv"
  ext_cat:
    file: "df_ml_cat_l2_extract.csv"
  ext_xg:
    file: "df_ml_xg_l2_extract.csv"
  ext_best:
    file: "df_ml_best.csv"

validation:
  max_unmatched_dfu_pct: 10.0   # Warn if >10% of DFUs fail dim_sku join
  valid_lags: [0, 1, 2, 3, 4]

materialized_views:
  - agg_forecast_monthly
  - agg_accuracy_by_dim
  - agg_dfu_coverage
  - agg_accuracy_lag_archive
  - agg_dfu_coverage_lag_archive
```

---

## 9. Script

`scripts/etl/load_ext_ml_forecasts.py`

```
Usage:
  python scripts/etl/load_ext_ml_forecasts.py --model ext_lgbm
  python scripts/etl/load_ext_ml_forecasts.py --model ext_cat
  python scripts/etl/load_ext_ml_forecasts.py --model ext_xg
  python scripts/etl/load_ext_ml_forecasts.py --model ext_best
  python scripts/etl/load_ext_ml_forecasts.py --all
```

`--model` selects a single model from the YAML config map. `--all` iterates through all four models in order. Both paths are idempotent.

The script follows the standard ETL conventions for this project:

- Reads DB params via `from common.db import get_db_params`
- Loads config via `load_config("ext_ml_forecasts")` from `common/utils.py`
- Uses `%s` placeholders in all SQL
- Logs via `logging.getLogger(__name__)` — no raw `print()` calls
- Catches `psycopg.Error` and `ValueError` specifically; never bare `except Exception`

---

## 10. Make Targets

```makefile
load-ext-lgbm:   ## Load ext_lgbm from df_ml_lgbm_l2_extract.csv
	$(UV) run python scripts/etl/load_ext_ml_forecasts.py --model ext_lgbm

load-ext-cat:    ## Load ext_cat from df_ml_cat_l2_extract.csv
	$(UV) run python scripts/etl/load_ext_ml_forecasts.py --model ext_cat

load-ext-xg:     ## Load ext_xg from df_ml_xg_l2_extract.csv
	$(UV) run python scripts/etl/load_ext_ml_forecasts.py --model ext_xg

load-ext-best:   ## Load ext_best from df_ml_best.csv
	$(UV) run python scripts/etl/load_ext_ml_forecasts.py --model ext_best

load-ext-all: load-ext-lgbm load-ext-cat load-ext-xg load-ext-best  ## Load all 4 external ML forecasts
```

---

## 11. Edge Cases and Validation

| Scenario | Handling |
|---|---|
| DFU not found in `dim_sku` | Row dropped; count logged as warning. If count exceeds `max_unmatched_dfu_pct`, a summary error is logged after load completes. |
| `LAG` value outside 0–4 | Row dropped; count logged. |
| Date value not parseable | Row dropped; offending value logged. |
| File not found at configured path | Script raises `FileNotFoundError` with the expected path and exits before touching the DB. |
| Re-run after data correction | Upsert replaces existing rows cleanly. No manual delete needed. |
| Partial run (script interrupted) | Transaction per target table. Either the full batch for that table commits or nothing does. MVs are refreshed only after both tables complete. |

---

## 12. Testing

### Backend unit tests (`tests/unit/test_load_ext_ml_forecasts.py`)

1. Column mapping — verify `DFU`, `PREDICTED_ORDERS`, `ACTUAL_ORDERS` are renamed correctly
2. Date normalisation — `FORECASTDATE` and `STARTDATE` are truncated to month-start
3. `forecast_ck` construction — matches expected composite key format
4. Lag filtering — rows with `LAG` outside 0–4 are dropped
5. Unmatched DFU handling — unmatched rows are excluded; matched rows retain correct `item_id`, `customer_group`, `loc`
6. Upsert idempotency — running twice produces the same row count, not doubled rows
7. `timeframe` column — always NULL for `ext_*` model IDs
8. `--all` flag iterates all four model IDs from config

### API integration verification (post-load)

After running `make load-ext-all`:

1. `GET /forecast/accuracy/available-models` lists all four `ext_*` IDs
2. `GET /forecast/accuracy/lag-curve?models=ext_lgbm` returns 5 data points (lags 0–4)
3. `GET /forecast/accuracy/slice?models=ext_lgbm,lgbm` returns results for both models in the same response
4. Re-running `make load-ext-lgbm` does not change row counts in either target table
