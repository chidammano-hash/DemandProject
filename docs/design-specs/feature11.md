# Feature 11: Multi-Model Forecast Support

## Objective
Enable multiple forecasting algorithms to coexist in the forecast pipeline. Users can filter KPIs, trends, and data by model, or view all models aggregated.

## Scope
- Dataset: `fact_external_forecast_monthly`
- New column: `model_id TEXT NOT NULL DEFAULT 'external'`
- Backward compatible: existing data defaults to `model_id = 'external'`

## Schema Change

### fact_external_forecast_monthly

Add column:
```sql
model_id TEXT NOT NULL DEFAULT 'external'
```

Uniqueness constraint change:
- Remove: `forecast_ck TEXT UNIQUE NOT NULL`
- Replace: `forecast_ck TEXT NOT NULL` + `CONSTRAINT uq_forecast_ck_model UNIQUE (forecast_ck, model_id)`

This allows the same business key (dmdunit/dmdgroup/loc/fcstdate/startdate) to appear once per model.

Add index:
```sql
CREATE INDEX idx_fact_external_forecast_monthly_model_id
  ON fact_external_forecast_monthly (model_id);
```

### agg_forecast_monthly (materialized view)

Add `model_id` to GROUP BY and indexes:
- Grain becomes: `(month_start, dmdunit, loc, model_id)`
- Unique index: `(dmdunit, loc, month_start, model_id)`

## Normalize Pipeline

In `normalize_dataset_csv.py`, forecast-specific block:
- If source CSV has a `model_id` column, use it
- If missing, default to `'external'`

## API Changes

### New endpoint: `GET /domains/forecast/models`
Returns distinct model_id values from the forecast table.

Response:
```json
{
  "domain": "forecast",
  "models": ["external", "arima", "prophet"]
}
```

### Analytics endpoint: `GET /domains/{domain}/analytics`
Add optional parameter:
- `model: str = Query(default="")` — filter by model_id

When `model` is provided and domain is `forecast`:
- Add `model_id = %s` to WHERE clause
- KPIs, trends, and summary stats are scoped to that model
- Aggregation table (`agg_forecast_monthly`) is used when model filter is compatible

When `model` is empty (default):
- All models are aggregated (same behavior as before feature 11)

### Page endpoint
No changes needed — `model_id` is a regular column visible in the data explorer. Users can filter via the existing column filter UI.

## UI Changes

### Model selector dropdown (forecast domain only)
- Location: analytics panel, alongside item/location filters
- Populated from `GET /domains/forecast/models`
- Options: "All Models" (default, no filter) + list of model_id values
- Resets to "All Models" on domain change

### Filter integration
- When a model is selected, add `model_id` to `effectiveFilters` with exact-match prefix (`=<model>`)
- This filters both the data explorer table and the analytics panel

### KPI context
- Display selected model name near the accuracy window selector
- Example: "Model: arima | Averaged across last 6 month(s)"

## Model ID Conventions
- `external` — base statistical forecast from the source system (dfu_stat_fcst.txt)
- Future models use descriptive names: `arima`, `prophet`, `xgboost`, `ensemble`, etc.
- Model IDs should be lowercase, alphanumeric with underscores

## Migration
- Existing data: `make normalize-forecast && make load-forecast` fills `model_id = 'external'`
- Schema: `make down && make up` recreates tables with new column
- Views: `make refresh-agg` rebuilds materialized views with model_id dimension
