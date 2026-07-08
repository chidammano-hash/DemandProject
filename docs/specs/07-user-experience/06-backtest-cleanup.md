# Backtest Cleanup

> Two cleanup utilities for removing forecast data from PostgreSQL: one deletes by model identity (e.g., remove all LightGBM predictions), the other deletes by date range (e.g., remove forecasts before January 2025). Both refresh dependent materialized views after deletion.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A (backend only) |
| **Key Files** | `scripts/ml/clean_backtest_models.py`, `scripts/ml/clean_forecasts_by_date.py` |

---

## Problem

As models are retrained, backtested, and replaced, stale predictions accumulate in `fact_external_forecast_monthly` and `backtest_lag_archive`. A single backtest run can write millions of rows. Without targeted cleanup, the forecast table grows indefinitely, slowing queries and making accuracy analysis unreliable (old predictions mix with current ones). Planners need to remove specific models or date ranges without a full table truncate that would wipe production forecasts.

---

## Solution

Two CLI scripts provide targeted deletion:

1. **Model cleanup** (`clean_backtest_models.py`): Deletes rows by `model_id` from both forecast and archive tables. Protects `model_id='external'` (source-system forecasts) by default.

2. **Date cleanup** (`clean_forecasts_by_date.py`): Deletes rows by time bucket (`startdate` or `fcstdate`) with `--before`, `--after`, `--between`, or `--months` range filters.

Both scripts refresh 5 materialized views after deletion and support `--dry-run` for preview.

---

## How It Works

### Model Cleanup

| Flag | Purpose | Example |
|---|---|---|
| `--list` | Show row counts by model_id in both tables | `make backtest-list` |
| `MODELS="lgbm_cluster catboost_cluster"` | Remove specific models | `make backtest-clean MODELS="lgbm_cluster"` |
| `--all-backtest` | Remove all non-external models | Clears all ML predictions |
| `--dry-run` | Preview deletions without executing | Safe inspection |

Safety: The `external` model_id is excluded from `--all-backtest` to prevent accidental deletion of source-system forecasts.

### Date Cleanup

| Flag | Purpose | Example |
|---|---|---|
| `--before DATE` | Delete rows before a date | `--before 2025-01-01` |
| `--after DATE` | Delete rows after a date | `--after 2025-06-01` |
| `--between D1 D2` | Delete rows in a date range | `--between 2024-01-01 2024-07-01` |
| `--months M1 M2 ...` | Delete specific months | `--months 2024-03 2024-06` |
| `--model MODEL` | Filter by model_id | `--model external` |
| `--date-column COL` | Target `startdate` (default) or `fcstdate` | `--date-column fcstdate` |
| `--forecast-only` | Only clean main table | Skip archive |
| `--archive-only` | Only clean archive table | Skip main |
| `--dry-run` | Preview without deleting | Safe inspection |
| `--list` | Show row counts by model + month | Audit before cleanup |

All dates are normalized to month-start (first of the month).

### Materialized View Refresh

After deletion, both scripts refresh these 5 views to keep aggregates consistent:

| View | Purpose |
|---|---|
| `agg_forecast_monthly` | Forecast aggregation |
| `agg_sales_monthly` | Sales aggregation |
| `mv_inventory_forecast_monthly` | Inventory-forecast bridge |
| `mv_inventory_health_score` | Health scores |
| `mv_control_tower_kpis` | Control tower KPIs |

---

## Pipeline

| Command | What It Does |
|---|---|
| `make backtest-list` | Show model row counts in forecast + archive |
| `make backtest-clean MODELS="lgbm_cluster catboost_cluster"` | Remove specific models |
| `make forecast-clean-list` | Show row counts by model + month |
| `make forecast-clean ARGS="--before 2025-04-01 --model external"` | Delete external forecasts before April 2025 |
| `make forecast-clean ARGS="--between 2024-01-01 2024-07-01"` | Delete all models in date range |
| `make forecast-clean ARGS="--months 2025-01 --model external"` | Delete one month for one model |
| `make forecast-clean ARGS="--before 2025-01-01 --dry-run"` | Preview without deleting |

---

## Dependencies

| Dependency | Reason |
|---|---|
| `fact_external_forecast_monthly` | Main forecast table targeted for deletion |
| `backtest_lag_archive` | Archive table targeted for deletion |
| 5 materialized views | Refreshed after cleanup to maintain consistency |

---

## See Also

- `02-forecasting/03-backtest-framework.md` -- backtest pipeline that produces the data being cleaned
- `02-forecasting/07-champion-selection.md` -- champion rows stored in the same forecast table
- `07-user-experience/04-job-scheduler.md` -- cleanup can be scheduled as a recurring job
