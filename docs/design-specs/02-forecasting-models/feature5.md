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
