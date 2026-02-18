# Feature 10: Multi-Dimensional Accuracy Slicing

## Objective
Enable accuracy analysis sliced by cluster, supplier, ABC volume classification, region, brand, lag, and model — powered by materialized views and a collapsible UI panel.

## Scope
- Materialized views: `agg_accuracy_by_dim`, `agg_accuracy_lag_archive`
- API endpoints for accuracy slicing and lag-horizon curves
- UI: collapsible Accuracy Comparison panel in the Forecast domain

## Materialized Views

### `agg_accuracy_by_dim`
Joins `fact_external_forecast_monthly` + `dim_dfu`. Grain: `(model_id, lag, month, cluster, supplier, abc_vol, region, brand)`.

Stores: `SUM(F)`, `SUM(A)`, `SUM(ABS(F-A))` for KPI derivation. Refreshed by `backtest-load`.

### `agg_accuracy_lag_archive`
Same structure from `backtest_lag_archive` + `dim_dfu`. Adds `timeframe` grain. Powers lag-horizon accuracy curves. Refreshed by `backtest-load`.

### DDL
`mvp/demand/sql/011_create_accuracy_slice_views.sql`

## API Endpoints

### Accuracy Slice
`GET /domains/forecast/accuracy-slice`

Parameters: `model_id`, `lag`, `cluster`, `supplier`, `abc_vol`, `region`, `brand`

Returns accuracy metrics grouped by the requested dimension.

### Lag-Horizon Curve
`GET /domains/forecast/lag-curve`

Parameters: `model_id`, optional dimension filters

Returns accuracy at each lag (0-4) for the selected model/filters — used to plot how accuracy degrades with forecast horizon.

## UI Components

### Accuracy Comparison Panel (Forecast domain)
- Collapsible panel below KPI cards
- Slice by: cluster, supplier, ABC vol, region, brand, lag
- Model comparison pivot table with best-model highlighting
- Lag-horizon accuracy curve chart (Recharts)

## Makefile Targets
```makefile
accuracy-slice-refresh:  # Refresh agg_accuracy_by_dim + agg_accuracy_lag_archive
accuracy-slice-check:    # Curl accuracy slice + lag-curve endpoints
```

## Dependencies
- Feature 7 (clustering — provides `cluster_assignment` for slicing)
- Feature 8 + 9 (backtesting — provides `backtest_lag_archive` data)
- Feature 5 (KPI formulas)
- Feature 6 (multi-model — `model_id` dimension)
