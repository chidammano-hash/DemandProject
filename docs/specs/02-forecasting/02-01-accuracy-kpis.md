<!-- SOURCE: feature5.md (Accuracy KPIs) -->
# Feature 5: Forecast Accuracy KPI Cards

## Objective
Define and surface forecast-accuracy KPIs for external monthly forecasts in the forecast UI.

## Scope
- Dataset: `fact_external_forecast_monthly`
- Forecast measure: `basefcst_pref`
- Actual measure: `tothist_dmd`
- Grain: monthly (`startdate`) for selected filter context (item, location, lag/date scope)

## KPI Definitions
Let:
- `F` = forecast (`basefcst_pref`)
- `A` = actual (`tothist_dmd`)
- `E` = error = `F - A`

For the selected filtered set of rows:

1. Total Forecast
- Formula: `SUM(F)`

2. Total Actual
- Formula: `SUM(A)`

3. Absolute Error
- Formula: `SUM(ABS(E))`

4. Bias
- Formula: `(SUM(F) / SUM(A)) - 1`
- If `SUM(A) = 0`, return `NULL`.

5. WAPE (%)
- Formula: `100 * SUM(ABS(E)) / ABS(SUM(A))`
- If `SUM(A) = 0`, return `NULL`.

6. MAPE (%)
- Formula: `100 * AVG(ABS(E) / ABS(A))` for rows where `A != 0`
- Rows with `A = 0` are excluded.

7. Forecast Accuracy (%)
- Formula: `100 - WAPE%`
- If `WAPE%` is `NULL`, return `NULL`.

## UI Mapping (Forecast Domain)
Show the following KPI cards in the forecast analytics panel:
- Forecast Accuracy
- WAPE
- MAPE
- Total Forecast
- Total Actual
- Absolute Error
- Bias

KPI window selector:
- Add a dropdown `Accuracy Window (Months)` with values `1..12`.
- Selected value `N` means KPI cards are computed on the latest `N` monthly buckets (by `startdate`) within active filters.
- Accuracy metrics shown on cards (`Forecast Accuracy`, `WAPE`, `MAPE`) are **average monthly** values across this `N`-month window.
- Volume/error cards (`Total Forecast`, `Total Actual`, `Absolute Error`) are summed over the same `N`-month window.
- `Bias` is computed as `(SUM(Forecast) / SUM(History)) - 1` across the `N`-month window.

Trend chart requirement:
- Support monthly `Forecast Accuracy %` series in the trend chart.
- Formula per month bucket:
  - `Forecast Accuracy % = 100 - (100 * SUM(ABS(F - A)) / ABS(SUM(A)))`
  - If monthly `SUM(A) = 0`, value is null for that month.
- This measure should be selectable alongside volume measures (`basefcst_pref`, `tothist_dmd`).

These cards are computed on the backend using the same active filters used for trend/chart queries.

## Dependencies
- Feature 4 (fact tables: `fact_sales_monthly`, `fact_external_forecast_monthly`)
- Feature 6 (`model_id` for multi-model support)
- Feature 7 (cluster dimensions for accuracy slicing)
- Feature 30 (`seasonality_profile` dimension)

---

## Implementation Details

### Backend
- `common/metrics.py`: `compute_accuracy_metrics(forecast_col, actual_col)` — shared function returning `n_rows`, `wape`, `bias`, `accuracy_pct`
- `api/core.py`: `compute_kpis()` and `forecast_accuracy_expr()` — API-level KPI computation and SQL expression generation

### Accuracy Slicing Endpoints (Feature 10 integration)
- `GET /forecast/accuracy/slice` — groups accuracy by 11 dimensions (`cluster_assignment`, `ml_cluster`, `supplier_desc`, `abc_vol`, `region`, `brand_desc`, `dfu_execution_lag`, `month_start`, `lag`, `model_id`, `seasonality_profile`)
- `GET /forecast/accuracy/lag-curve` — accuracy by lag (0-4) per model
- Both exist inline in `api/main.py` and in `api/routers/accuracy.py`
- `common_dfus` mode: when true with 2+ models, restricts to DFUs present in ALL models for fair comparison

### Pre-aggregated Materialized Views
- `agg_accuracy_by_dim` (`sql/011`) — forecast+dim_dfu join with 11-column grain
- `agg_accuracy_lag_archive` (`sql/011`) — same for backtest archive
- `agg_dfu_coverage` (`sql/012`) — distinct DFU count per model/lag
- `agg_dfu_coverage_lag_archive` (`sql/012`) — same for archive
- All 4 recreated with `seasonality_profile` in `sql/016`

### Makefile Targets
- `accuracy-slice-refresh` — refresh all 4 views
- `accuracy-slice-check` — verify view data


---

## Examples

### Example: Compute all KPIs in SQL

```sql
SELECT
    ROUND(100.0 - 100.0 * SUM(ABS(basefcst_pref - tothist_dmd))
               / NULLIF(ABS(SUM(tothist_dmd)), 0), 2)   AS accuracy_pct,
    ROUND(100.0 * SUM(ABS(basefcst_pref - tothist_dmd))
               / NULLIF(ABS(SUM(tothist_dmd)), 0), 2)   AS wape,
    ROUND((SUM(basefcst_pref) / NULLIF(SUM(tothist_dmd),0)) - 1, 4) AS bias
FROM fact_external_forecast_monthly
WHERE dmdunit='100320' AND loc='1401-BULK'
  AND model_id='external' AND lag=2
  AND startdate >= '2025-08-01';
-- accuracy_pct: 92.53 | wape: 7.47 | bias: 0.0143
```

### Example: KPI API endpoint

```bash
curl -s "http://localhost:8000/forecast/accuracy?item=100320&loc=1401-BULK&model=external&lag=2&window=6" \
  | jq '{accuracy_pct, wape, bias}'
# {"accuracy_pct": 92.53, "wape": 7.47, "bias": 0.014}
```

### Example: Lag curve — accuracy degrades at longer horizons

```bash
curl -s "http://localhost:8000/forecast/accuracy/lag-curve?model=external" \
  | jq '.rows[] | {lag, accuracy_pct}'
# {"lag": 0, "accuracy_pct": 97.1}
# {"lag": 1, "accuracy_pct": 94.3}
# {"lag": 2, "accuracy_pct": 91.8}
# {"lag": 3, "accuracy_pct": 88.6}
# {"lag": 4, "accuracy_pct": 84.2}
```


---

<!-- SOURCE: feature10.md (Multi-Dimensional Accuracy Slicing) -->
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

#### Example

```bash
# Slice accuracy by cluster assignment for lgbm_global at execution lag
curl -s "http://localhost:8000/forecast/accuracy/slice?group_by=cluster_assignment&models=lgbm_global&lag=-1" | python3 -m json.tool
# {"rows": [
#   {"cluster_assignment": "high_volume_steady",    "accuracy_pct": 94.2, "wape": 5.8,  "dfu_count": 312},
#   {"cluster_assignment": "seasonal_high_volume",  "accuracy_pct": 89.1, "wape": 10.9, "dfu_count": 187},
#   {"cluster_assignment": "low_volume_erratic",    "accuracy_pct": 71.3, "wape": 28.7, "dfu_count": 94}
# ]}

# Compare models on common DFUs only (fair comparison)
curl -s "http://localhost:8000/forecast/accuracy/slice?group_by=model_id&models=lgbm_global,catboost_global,external&common_dfus=true&lag=2" | python3 -m json.tool

# Filter by ABC volume classification and date range
curl -s "http://localhost:8000/forecast/accuracy/slice?group_by=abc_vol&models=lgbm_cluster&month_from=2025-01-01&month_to=2025-06-01" | python3 -m json.tool
```

### Lag-Horizon Curve
`GET /domains/forecast/lag-curve`

Parameters: `model_id`, optional dimension filters

Returns accuracy at each lag (0-4) for the selected model/filters — used to plot how accuracy degrades with forecast horizon.

#### Example

```bash
# Get accuracy degradation across lags 0-4 for two models
curl -s "http://localhost:8000/forecast/accuracy/lag-curve?models=lgbm_global,external" | python3 -m json.tool
# {"rows": [
#   {"model_id": "lgbm_global",  "lag": 0, "accuracy_pct": 96.1, "wape": 3.9},
#   {"model_id": "lgbm_global",  "lag": 1, "accuracy_pct": 93.8, "wape": 6.2},
#   {"model_id": "lgbm_global",  "lag": 2, "accuracy_pct": 91.2, "wape": 8.8},
#   {"model_id": "external",     "lag": 0, "accuracy_pct": 88.5, "wape": 11.5},
#   {"model_id": "external",     "lag": 2, "accuracy_pct": 82.3, "wape": 17.7}
# ]}
# Accuracy degrades with longer horizons — use this chart to pick the optimal execution lag
```

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


---

## Additional Examples

#### Example — Materialized View Refresh

```bash
# Refresh both accuracy slice views after loading new backtest predictions
docker exec demand-mvp-postgres psql -U demand -d demand_mvp -c "
  REFRESH MATERIALIZED VIEW CONCURRENTLY agg_accuracy_by_dim;
  REFRESH MATERIALIZED VIEW CONCURRENTLY agg_accuracy_lag_archive;
"
# Automatically triggered by: make backtest-load
```

#### Example — Query agg_accuracy_by_dim directly

```sql
-- Accuracy by cluster for lgbm_global at lag 2
SELECT cluster_assignment,
       ROUND(100.0 - 100.0 * SUM(sum_abs_err) / NULLIF(ABS(SUM(sum_actual)), 0), 2) AS accuracy_pct,
       COUNT(DISTINCT dmdunit || dmdgroup || loc)                                      AS dfu_count
FROM agg_accuracy_by_dim
WHERE model_id = 'lgbm_global' AND lag = 2
GROUP BY cluster_assignment
ORDER BY accuracy_pct DESC;
-- cluster_assignment          | accuracy_pct | dfu_count
-- high_volume_steady          |        94.20 |       312
-- seasonal_high_volume        |        89.10 |       187
-- low_volume_erratic          |        71.30 |        94
```

#### Example — Accuracy Comparison Panel (UI interaction)

```
User opens Accuracy tab → selects "Cluster" from the slice-by dropdown
→ selects models: lgbm_global, catboost_global, external
→ sets lag = 2, date range = 2024-07 to 2025-06
→ checks "Common DFUs Only" to ensure a fair comparison
→ pivot table renders: rows = clusters, cols = models, best accuracy highlighted with a star
→ lag-horizon chart shows accuracy at lags 0-4 for all three models
```

#### Example — Makefile targets for accuracy views

```bash
# After backtest-load, manually refresh and validate accuracy slice views
make accuracy-slice-refresh
# REFRESH MATERIALIZED VIEW agg_accuracy_by_dim;
# REFRESH MATERIALIZED VIEW agg_accuracy_lag_archive;

make accuracy-slice-check
# curl http://localhost:8000/forecast/accuracy/slice?group_by=cluster_assignment
# curl http://localhost:8000/forecast/accuracy/lag-curve?models=lgbm_global,external
```

#### Example — seasonality_profile filter (Feature 32)

```bash
# Filter accuracy slice to only seasonal DFUs
curl -s "http://localhost:8000/forecast/accuracy/slice?group_by=model_id&seasonality_profile=strong_yearly&lag=2" \
  | jq '.rows[] | {model_id, accuracy_pct}'
# {"model_id": "lgbm_global",  "accuracy_pct": 87.4}
# {"model_id": "catboost_global", "accuracy_pct": 88.1}
# Seasonal DFUs are typically harder to forecast — accuracy is lower than overall
```
