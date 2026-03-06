# Feature 37: Inventory Planning Backtesting — Connecting Forecast Accuracy to Inventory Outcomes

## Executive Summary

Feature 37 bridges the gap between forecast accuracy and inventory outcomes. The platform already has forecast accuracy data (per model, per DFU, per month) and inventory snapshot data (on-hand, on-order, sales, lead time per item-location per month), but these two datasets were completely disconnected. There was no way to ask: "Did this model's under-forecast correlate with the stockout at location X?" or "Which algorithm correlates with the fewest excess inventory events?"

This feature joins `agg_inventory_monthly` with `fact_external_forecast_monthly` into a single materialized view (`mv_inventory_forecast_monthly`), exposes 4 API endpoints, and adds a new **Inv. Backtest** UI tab that answers:
1. **What happened** — stockout and excess events across the portfolio
2. **Why it happened** — forecast bias correlation (under-forecast → stockout correlation, over-forecast → excess correlation; see Known Limitations for causality caveats)
3. **Which algorithm performed best** — model comparison by inventory outcome metrics (not just forecast accuracy)

## Key Features

- **Materialized View:** `mv_inventory_forecast_monthly` — INNER JOIN of inventory and forecast at `item_no + loc + month_start + model_id` grain, with computed stockout/excess flags, DOS, bias direction, and DFU attributes
- **4 API Endpoints:** Summary, Trend, Root Cause, and Detail — all with shared filter parameters (models, date range, item, location, cluster, ABC, region)
- **Model Comparison:** Side-by-side comparison of forecasting algorithms by inventory outcome metrics (service level, stockout rate, excess rate, WAPE). **Note:** Model comparison uses actual historical inventory snapshots (which reflect decisions made with the operational forecast) scored against each model's retrospective forecasts. This measures forecast accuracy correlation with observed inventory outcomes, not a controlled A/B test of replenishment policies.
- **Forecast Bias Correlation:** For each stockout/excess event, shows which forecast bias direction (under-forecast, over-forecast, exact) co-occurred with the event. This is a correlation analysis, not a causal attribution — see Known Limitations.
- **DFU-Level Detail:** Paginated, sortable event table with color-coded rows and event type badges

## Business Impact

- **Inventory planners** can identify which forecasting model correlates with fewer stockouts and excess inventory for their portfolio
- **Supply chain managers** can quantify the relationship between forecast inaccuracy and inventory outcomes (not just WAPE/bias)
- **Data scientists** get a feedback loop: forecast accuracy → inventory outcomes → model selection

---

## Database Schema

### Materialized View: `mv_inventory_forecast_monthly`

**Important:** View is created `WITH NO DATA`. Query `SELECT COUNT(*) FROM mv_inventory_forecast_monthly` will return 0 until `make refresh-inv-backtest` (or `REFRESH MATERIALIZED VIEW mv_inventory_forecast_monthly`) is executed.

**Source tables:**
- `agg_inventory_monthly` (inventory) — aliased `i`
- `fact_external_forecast_monthly` (forecast) — aliased `f`
- `dim_dfu` (attributes) — aliased `d` (LEFT JOIN)

**Join conditions:**
- `i.item_no = f.dmdunit`
- `i.loc = f.loc`
- `i.month_start = f.startdate`
- `f.lag = COALESCE(d.execution_lag, 0)` — operational forecast only

**Row filters (WHERE):**
- `f.lag = COALESCE(d.execution_lag, 0)` — execution-lag aligned forecast only
- `f.tothist_dmd IS NOT NULL` — excludes future months (no actuals yet)
- `f.basefcst_pref IS NOT NULL` — excludes rows with missing base forecast

**Coverage note:** The INNER JOIN means items with inventory snapshots but no matching forecast (for the given `model_id` and `execution_lag`) are excluded. Stockouts at un-forecasted items are not captured by this view.

**Grain:** `item_no + loc + month_start + model_id`

**Columns:**

| Column | Type | Source/Derivation |
|--------|------|-------------------|
| `item_no` | text | `i.item_no` |
| `loc` | text | `i.loc` |
| `month_start` | date | `i.month_start` |
| `model_id` | text | `f.model_id` |
| `forecast` | numeric | `f.basefcst_pref AS forecast` (not `f.base_forecast`) |
| `actual_demand` | numeric | `f.tothist_dmd AS actual_demand` (not `f.actual_demand`) |
| `forecast_error` | numeric | `forecast - actual_demand` |
| `abs_error` | numeric | `ABS(forecast_error)` |
| `eom_qty_on_hand` | numeric | `i.eom_qty_on_hand` |
| `eom_qty_on_hand_on_order` | numeric | `i.eom_qty_on_hand_on_order` — end-of-month on-hand plus on-order quantity |
| `monthly_sales` | numeric | `i.monthly_sales` — MAX(mtd_sales) for the month (cumulative, not summed) |
| `snapshot_days` | numeric | `i.snapshot_days` — count of daily snapshots available in the month (partial-month detection) |
| `avg_daily_sls` | numeric | `i.avg_daily_sls` — derived from cumulative mtd_sales via LAG() window; averages only non-zero daily values. Zero-demand days excluded, so DOS reflects active selling days. |
| `dos` | numeric | `eom_qty_on_hand / avg_daily_sls` (NULL when avg_daily_sls = 0) |
| `latest_lead_time_days` | numeric | `i.latest_lead_time_days` — most recent single LT value (see Known Limitations) |
| `is_stockout` | boolean | `eom_qty_on_hand <= 0` — full stockout only; safety stock breach events (on-hand below SS target but above zero) are not detected |
| `is_excess` | boolean | Independently computed: TRUE when `avg_daily_sls > 0 AND eom_qty_on_hand / avg_daily_sls > 90`; FALSE otherwise (including when avg_daily_sls = 0 — items with zero sales are never classified as excess regardless of on-hand quantity). Not derived from the `dos` column. |
| `bias_direction` | text | `'over'` / `'under'` / `'exact'` based on forecast_error sign. `'exact'` occurs only when `basefcst_pref = tothist_dmd` precisely (numeric equality); in practice this category contains very few rows. |
| `cluster_assignment` | text | From `dim_dfu` (COALESCE default `'(unassigned)'` — with parentheses) |
| `abc_vol` | text | From `dim_dfu` (COALESCE default `'(unknown)'` — with parentheses) |
| `region` | text | From `dim_dfu` (COALESCE default `'(unknown)'` — with parentheses) |
| `brand` | text | From `dim_dfu` (COALESCE default `'(unknown)'` — with parentheses) |

**Indexes:**
1. Unique PK: `(item_no, loc, month_start, model_id)`
2. `model_id`
3. `month_start`
4. `cluster_assignment`
5. `abc_vol` — for ABC segmentation filtering
6. Partial composite index on `(model_id, month_start)` WHERE `is_stockout = TRUE`
7. Partial composite index on `(model_id, month_start)` WHERE `is_excess = TRUE`

**File:** `sql/019_inventory_forecast_view.sql`

---

## API Endpoints

All endpoints live in `api/routers/inv_backtest.py` and use a shared `_inv_backtest_filters()` helper for WHERE clause construction. The router is mounted in `api/main.py` via `app.include_router()`.

### Shared Filter Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `models` | string | `""` | Comma-separated model IDs |
| `month_from` | string | `""` | Start date (inclusive) |
| `month_to` | string | `""` | End date (inclusive) |
| `item` | string | `""` | ILIKE filter on item_no |
| `location` | string | `""` | ILIKE filter on loc |
| `cluster_assignment` | string | `""` | Exact match |
| `abc_vol` | string | `""` | Exact match |
| `region` | string | `""` | Exact match |
| `excess_dos_threshold` | int | `90` | Days threshold for excess classification (range: 1–365) |

### `GET /inventory-backtest/summary`

Per-model aggregate inventory outcome metrics. `Cache-Control: max-age=120` (2 minutes).

**Response:**
```json
{
  "models": ["external", "lgbm_cluster"],
  "excess_dos_threshold": 90,
  "by_model": {
    "external": {
      "dfu_months": 5000,
      "stockout_count": 150,
      "stockout_rate": 3.0,
      "excess_count": 400,
      "excess_rate": 8.0,
      "service_level": 97.0,
      "avg_dos": 42.0,
      "wape": 28.5,
      "bias": 3.2
    }
  }
}
```

**Note on `service_level`:** This is Cycle Service Level (CSL) — the percentage of DFU-months without a stockout event. It is NOT Fill Rate (which measures the fraction of demand units fulfilled). See Known Limitations for why Fill Rate cannot be computed from this view.

### `GET /inventory-backtest/trend`

Monthly inventory outcome trend by model. `Cache-Control: max-age=120` (2 minutes).

**Response:**
```json
{
  "trend": [{
    "month": "2025-03-01",
    "by_model": {
      "external": {
        "stockout_rate": 3.5,
        "excess_rate": 8.0,
        "avg_dos": 41.0,
        "wape": 29.0
      }
    }
  }]
}
```

### `GET /inventory-backtest/root-cause`

Stockout/excess event breakdown by forecast bias direction — showing which bias direction co-occurred most frequently with each event type. `Cache-Control: max-age=120` (2 minutes).

**Important:** This endpoint uses `model_id` (singular, required string) — NOT the shared `models` parameter (plural, optional). Omitting `model_id` returns HTTP 422.

**Note on causality:** This analysis shows correlation between forecast error direction and inventory events in the same month. It does NOT establish that the forecast error caused the event. Replenishment decisions depend on additional factors (lead time, order quantity, procurement timing) not captured in this view.

**Required param:** `model_id` (single model)

**Response:**
```json
{
  "model_id": "lgbm_cluster",
  "stockout_total": 450,
  "stockout_under_forecast": 320,
  "stockout_over_forecast": 80,
  "stockout_exact": 50,
  "excess_total": 1200,
  "excess_over_forecast": 950,
  "excess_under_forecast": 150,
  "excess_exact": 100
}
```

### `GET /inventory-backtest/detail`

Paginated DFU-level inventory event rows. `Cache-Control: max-age=60` (1 minute).

**Additional params:** `event_type` (all/stockout/excess), `limit`, `offset`, `sort_by`, `sort_dir`

**Response:**
```json
{
  "total": 15000,
  "limit": 50,
  "offset": 0,
  "rows": [{
    "item_no": "100320",
    "loc": "1401-BULK",
    "month": "2025-06-01",
    "model_id": "lgbm_cluster",
    "forecast": 120.5,
    "actual_demand": 150.0,
    "eom_qty_on_hand": 0,
    "dos": null,
    "event_type": "stockout",
    "forecast_error": -29.5,
    "pct_error": -19.7,
    "bias_direction": "under"
  }]
}
```

---

## UI Components

### InvBacktestTab (`tabs/InvBacktestTab.tsx`)

**Layout (top-to-bottom):**

1. **KPI Cards** — Best Cycle Service Level (CSL), Lowest Stockout Rate, Lowest Excess Rate, Models Compared, DFU-Months (severity-coded). Service level ≥ 95% = best; < 90% = warning.

2. **Filter Controls** — Item/Location/Cluster text inputs with debounced search (400ms), model multi-select pill buttons. On initial load, first 5 available models are auto-selected.

3. **Model Comparison Chart** — Recharts `ComposedChart` with grouped bars (stockout_rate + excess_rate per model) and WAPE line overlay on right Y-axis

4. **Forecast Bias Correlation Breakdown** — Horizontal stacked `BarChart` showing stockout/excess event counts split by bias direction (under/over/exact) for a selected model. Correlation only — not causal attribution.

5. **Monthly Trend Chart** — `LineChart` with one line per model, switchable metric (stockout_rate / excess_rate / avg_dos / wape)

6. **DFU-Level Detail Table** — Event type filter (All/Stockout/Excess), sortable columns, color-coded rows (red=stockout, amber=excess), paginated with Prev/Next (page size: 50 rows)

### Navigation

- Sidebar: `Activity` icon from lucide-react, shortcut `6`, section: `supply`
- URL: `?tab=invBacktest`
- Keyboard shortcut: `6`

---

## Known Limitations / Future Enhancements

The following data elements are absent from this view. They represent known gaps for a future inventory planning enhancement:

1. **Safety stock quantities absent.** `eom_qty_on_hand <= 0` detects full stockouts only. Safety stock breach events (on-hand drops below SS target but is still positive) are invisible. Safety stock target quantities and reorder points are not in the current schema. Stockout analysis cannot distinguish "no safety stock was set" from "safety stock was set but exhausted."

2. **Lead time variability absent.** Only `latest_lead_time_days` (a scalar point-in-time value) is captured. Lead time standard deviation (sigma_LT) is not tracked. Accurate safety stock recommendations require both mean and variability of lead time.

3. **Fill rate (beta service level) cannot be computed.** `actual_demand` (`tothist_dmd`) reflects shipments, not orders placed. Shortage quantity (unfulfilled demand units) is not available. Fill Rate would require joining with `fact_sales_monthly` using `qty_ordered` vs `qty_shipped`.

4. **Intra-month stockouts are invisible.** `eom_qty_on_hand` is the end-of-month snapshot. A DFU that was out of stock for 25 days but received a late replenishment will show positive EOM stock and be classified as NOT a stockout. Daily-granularity analysis requires querying `fact_inventory_snapshot` directly.

5. **No target inventory levels.** The view shows actual position but has no planned/target min, max, or target DOS to compare against. Inventory deviation from plan cannot be quantified.

6. **ABC-by-volume only.** XYZ (demand variability / coefficient of variation) segmentation is absent. `seasonality_profile` from `dim_dfu` is not joined into the view. High-variability items need different DOS excess thresholds than stable items.

7. **Replenishment policy data absent.** Order quantity, cycle time, MOQ, and last receipt date are not available. Root cause analysis cannot determine whether a stockout was caused by forecast error vs. replenishment execution failure (wrong order size, missed order timing, supplier delay).

8. **Cycle stock not captured.** There is no concept of order quantity or cycle stock (avg inventory that cycles between orders = EOQ/2). The view provides inventory position but not the replenishment cycle context.

9. **Model comparison is retrospective correlation, not controlled experiment.** All models are scored against the same historical inventory snapshots, which were driven by whichever model was operationally active at the time. A model showing lower stockout correlation did not actually drive different replenishment decisions in history.

---

## Makefile Targets

```bash
make db-apply-inv-backtest   # Create materialized view DDL (run once)
make refresh-inv-backtest    # Refresh with current data (required before querying)
```

---

## Testing

### Backend Tests (12 tests)
**File:** `tests/api/test_inventory_backtest.py`

- Summary: returns 200, filters, empty data, custom threshold
- Trend: returns 200, empty data
- Root cause: returns 200, missing model returns 422
- Detail: returns 200, event filter, pagination, sort fallback

### Frontend Tests (6 tests)
**File:** `tabs/__tests__/InvBacktestTab.test.tsx`

- Smoke test (renders without crashing)
- KPI cards render
- Model comparison chart renders
- Filter controls render
- Detail table renders
- Root cause section renders

---

## Files

| File | Action |
|------|--------|
| `sql/019_inventory_forecast_view.sql` | **Created** — Materialized view DDL |
| `Makefile` | Edited — 2 make targets |
| `api/routers/inv_backtest.py` | **Created** — 4 endpoints + filter helper |
| `frontend/src/types/index.ts` | Edited — 7 payload types |
| `frontend/src/api/queries.ts` | Edited — 4 fetch functions + query keys |
| `frontend/src/tabs/InvBacktestTab.tsx` | **Created** — New tab component (~700 lines) |
| `frontend/src/App.tsx` | Edited — lazy import + render block |
| `frontend/src/components/AppSidebar.tsx` | Edited — Activity icon + nav item |
| `frontend/src/hooks/useUrlState.ts` | Edited — added to VALID_TABS |
| `frontend/src/hooks/useKeyboardShortcuts.ts` | Edited — updated TAB_MAP (1-8) |
| `tests/api/test_inventory_backtest.py` | **Created** — 12 backend tests |
| `frontend/src/tabs/__tests__/InvBacktestTab.test.tsx` | **Created** — 6 frontend tests |

---

## Implementation Notes

### Actual Source Column Names
- Spec originally said `f.base_forecast` → actual: `f.basefcst_pref AS forecast`
- Spec originally said `f.actual_demand` → actual: `f.tothist_dmd AS actual_demand`

### SQL WHERE Filters
```sql
AND f.tothist_dmd IS NOT NULL
AND f.basefcst_pref IS NOT NULL
```
These exclude future months (no actuals) and rows with missing base forecast from the view.

### Summary Endpoint Parameter Ordering
The `excess_dos_threshold` appears as the first `%s` placeholder in the SQL SELECT (before WHERE clause parameters), because the `SUM(CASE WHEN dos IS NOT NULL AND dos > %s ...)` expression appears before `{where_sql}` in the query string. The endpoint corrects for this by prepending the threshold: `ordered_params = [excess_dos_threshold] + params[:threshold_idx]`. The Trend and Root Cause endpoints use `[excess_dos_threshold] + params` (threshold first) for the same reason.

### `is_excess` Behavior for Zero-Sales Items
When `avg_daily_sls = 0`, `is_excess = FALSE` (the ELSE branch fires — not NULL). Items with on-hand stock but no sales history are classified as NOT excess even though their DOS is technically infinite. This may undercount excess for slow-moving or newly introduced items.

### COALESCE Values
Uses `'(unassigned)'` and `'(unknown)'` — with parentheses — to distinguish missing DFU attributes from valid attribute values.

### View Creation
Created with `WITH NO DATA` — requires explicit `REFRESH MATERIALIZED VIEW mv_inventory_forecast_monthly` before the view is queryable.

### Caching
- Summary/Trend/Root Cause: `max_age=120` (2 minutes)
- Detail: `max_age=60` (1 minute)

### Frontend Details
- Uses `KpiCard`, `LoadingElement`, `useDebounce` (400ms), `useGlobalFilterContext`
- Auto-selects first 5 models on initial load
- KPI severity: cycle service level `>=95` best, `<90` warning
- Page size: 50 rows

---

## Examples

### Example: Inventory backtest summary — model comparison

```bash
curl -s "http://localhost:8000/inventory-backtest/summary" | \
  jq '.by_model | to_entries[] | {model_id: .key, stockout_rate: .value.stockout_rate, excess_rate: .value.excess_rate, wape: .value.wape}'
# {"model_id": "lgbm_cluster",     "stockout_rate": 2.1, "excess_rate": 8.3, "wape": 6.9}
# {"model_id": "catboost_cluster", "stockout_rate": 2.8, "excess_rate": 9.1, "wape": 7.4}
# {"model_id": "external",         "stockout_rate": 4.7, "excess_rate": 12.1, "wape": 12.8}
```

### Example: Forecast bias correlation — bias direction vs stockouts

```sql
-- Which model has the most stockouts co-occurring with systematic under-forecasting?
SELECT model_id, bias_direction, COUNT(*) AS n_events, AVG(abs_error) AS avg_abs_error
FROM mv_inventory_forecast_monthly
WHERE is_stockout = TRUE AND month_start >= '2025-08-01'
GROUP BY model_id, bias_direction
ORDER BY n_events DESC LIMIT 5;
-- external     | under | 847 | 142.3
-- external     | over  |  12 |  18.7
-- lgbm_cluster | under | 321 |  87.1
```

### Example: Monthly trend endpoint

```bash
curl -s "http://localhost:8000/inventory-backtest/trend?models=lgbm_cluster&month_from=2025-01-01" | jq '.trend[0]'
# {"month": "2025-01-01", "by_model": {"lgbm_cluster": {"stockout_rate": 1.8, "excess_rate": 7.2, "avg_dos": 38.5, "wape": 22.1}}}
```

### Example: Refresh inventory-forecast bridge

```bash
make refresh-inv-backtest
# REFRESH MATERIALIZED VIEW mv_inventory_forecast_monthly
# Joins agg_inventory_monthly + fact_external_forecast_monthly + dim_dfu
# Computes: forecast_error, abs_error, dos, is_stockout, is_excess, bias_direction
# Result: 42,847 rows (DFU × month × model grain)
```
