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

---

## Implementation Corrections

### Actual API Endpoint Paths
- `GET /forecast/accuracy/slice` (not `/domains/forecast/accuracy-slice`)
- `GET /forecast/accuracy/lag-curve` (not `/domains/forecast/lag-curve`)

### Full Parameter List for `/forecast/accuracy/slice`
- `group_by` (default: `cluster_assignment`) — 11 valid values: `cluster_assignment`, `ml_cluster`, `supplier_desc`, `abc_vol`, `region`, `brand_desc`, `dfu_execution_lag`, `month_start`, `lag`, `model_id`, `seasonality_profile`
- `models` (comma-separated string, not single `model_id`)
- `lag` (-1 = execution lag per DFU, 0-4)
- `month_from`, `month_to` — date range filters
- `cluster_assignment`, `supplier_desc`, `abc_vol`, `region` — dimension filters
- `seasonality_profile` — seasonality filter (Feature 32)
- `common_dfus` (bool) — intersect DFUs across models for fair comparison
- `include_dfu_count` (bool) — include distinct DFU count per bucket
- `item`, `location` — global filter passthrough

### Additional Materialized Views
- `agg_dfu_coverage` (`sql/012`) — distinct DFU count per model/lag
- `agg_dfu_coverage_lag_archive` (`sql/012`) — same for backtest archive
- All 4 views recreated with `seasonality_profile` in `sql/016`

### Additional DDL
- `sql/012_create_dfu_coverage_view.sql`
- `sql/016_add_seasonality_to_accuracy_views.sql`

### `common_dfus` Feature
- When true with 2+ models, falls back from pre-aggregated views to raw fact table with CTE intersecting DFUs present in all specified models

### Router Module
- Also implemented in `api/routers/accuracy.py`

### Frontend (`AccuracyTab.tsx`)
- Dedicated tab component (not a panel within Forecast domain)
- Slice-by dropdown (8 dimensions), lag filter (-1 to 4), models input, KPI window (1-12 months)
- Common DFUs Only checkbox with common/total DFU count badges
- Model comparison pivot table with best-accuracy star and high-bias warning
- Lag-horizon line chart with metric selector
- Champion Selection panel (Feature 15) integrated into same tab
