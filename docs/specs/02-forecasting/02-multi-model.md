# Multi-Model Forecast Support

> Lets multiple forecasting algorithms coexist in the same table, so planners can compare them side by side.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Accuracy, Item Analysis |
| **Key Files** | `sql/008_perf_indexes_and_agg.sql`, `api/routers/forecasting/accuracy.py`, `frontend/src/tabs/AccuracyTab.tsx` |

---

## Problem

A single forecasting algorithm rarely wins for every product. Without a way to store and compare multiple models, there is no path to improvement -- the team is locked into whatever the ERP system produces. Different algorithms excel for different demand patterns (seasonal vs. intermittent vs. high-volume steady).

## Solution

A `model_id` column on the forecast table lets any number of algorithms write predictions for the same item-location-month. The uniqueness constraint is `(forecast_ck, model_id)`, meaning the same business key can appear once per model. All downstream views, KPIs, and UI components are model-aware -- they filter, compare, and aggregate by model automatically.

## How It Works

1. Every forecast row carries a `model_id` that identifies the algorithm (e.g., `external`, `lgbm_cluster`, `champion`)
2. The source ERP forecast is loaded as `model_id = 'external'` (the baseline)
4. Champion selection writes the best-of-models composite as `model_id = 'champion'`
5. The ceiling (oracle) writes as `model_id = 'ceiling'`
6. Users select models via dropdown in the Accuracy tab to compare performance

## Data Model

### Schema Change on `fact_external_forecast_monthly`

| Column | Type | Default | Notes |
|--------|------|---------|-------|
| `model_id` | TEXT NOT NULL | `'external'` | Identifies the forecasting algorithm |

**Constraint:** `UNIQUE(forecast_ck, model_id)` -- same business key allowed once per model.

### Model ID Conventions

| model_id | Source |
|----------|--------|
| `external` | ERP/source system statistical forecast |
| `lgbm_cluster` | LightGBM per-cluster backtest |
| `champion` | Best model per DFU per month (before-the-fact selection) |
| `ceiling` | Best model per DFU per month (after-the-fact oracle) |

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/domains/forecast/models` | List all distinct model_id values in the forecast table |

All accuracy and analytics endpoints accept an optional `model` parameter to scope results to a single model.

## Pipeline

No dedicated pipeline. Model IDs are assigned during data loading:
- `make load-forecast-replace` loads external forecasts as `model_id = 'external'`
- `make backtest-load MODEL=lgbm_cluster` loads backtest predictions with the specified model_id

## Configuration

No config file. Model IDs are strings -- any new algorithm can write to the table with a new model_id without schema changes.

## Dependencies

None beyond the base forecast table (`fact_external_forecast_monthly`).

## Roster visibility

Item Analysis and backtesting model selectors use the enabled stage-specific algorithm roster from
`forecast_pipeline_config.yaml`. Retired algorithms are excluded from accuracy curves and
leaderboards. Item Analysis additionally keeps the intentional `external`, `champion`, and `ceiling`
reference series. Run `uv run python scripts/ml/clean_algorithm_roster.py --execute` to purge retired
rows and generated model/backtest artifacts; without `--execute`, the command is a safe preview.

## See Also

- [Accuracy KPIs](./01-accuracy-kpis.md) -- compares models using WAPE, bias, accuracy %
- [Backtest Framework](./03-backtest-framework.md) -- generates model-specific predictions
- [Champion Selection](./07-champion-selection.md) -- picks the best model per DFU
