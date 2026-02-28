# Feature 37: Inventory Planning Backtesting — Connecting Forecast Accuracy to Inventory Outcomes

## Executive Summary

Feature 37 bridges the gap between forecast accuracy and inventory outcomes. The platform already has forecast accuracy data (per model, per DFU, per month) and inventory snapshot data (on-hand, on-order, sales, lead time per item-location per month), but these two datasets were completely disconnected. There was no way to answer: "Did this model's under-forecast cause the stockout at location X?" or "Which algorithm leads to the fewest excess inventory events?"

This feature joins `agg_inventory_monthly` with `fact_external_forecast_monthly` into a single materialized view (`mv_inventory_forecast_monthly`), exposes 4 API endpoints, and adds a new **Inv. Backtest** UI tab that answers:
1. **What happened** — stockout and excess events across the portfolio
2. **Why it happened** — root cause attribution (under-forecast → stockout, over-forecast → excess)
3. **Which algorithm performed best** — model comparison by inventory outcomes (not just forecast accuracy)

## Key Features

- **Materialized View:** `mv_inventory_forecast_monthly` — INNER JOIN of inventory and forecast at `item_no + loc + month_start + model_id` grain, with computed stockout/excess flags, DOS, bias direction, and DFU attributes
- **4 API Endpoints:** Summary, Trend, Root Cause, and Detail — all with shared filter parameters (models, date range, item, location, cluster, ABC, region)
- **Model Comparison:** Side-by-side comparison of forecasting algorithms by inventory outcome metrics (service level, stockout rate, excess rate, WAPE)
- **Root Cause Attribution:** For each stockout/excess event, correlates with forecast bias direction (under-forecast, over-forecast, exact) to explain *why* the event occurred
- **DFU-Level Detail:** Paginated, sortable event table with color-coded rows and event type badges

## Business Impact

- **Inventory planners** can identify which forecasting model minimizes stockouts and excess inventory for their portfolio
- **Supply chain managers** can quantify the cost of forecast inaccuracy in inventory terms (not just WAPE/bias)
- **Data scientists** get a feedback loop: forecast accuracy → inventory outcomes → model selection

---

## Database Schema

### Materialized View: `mv_inventory_forecast_monthly`

**Source tables:**
- `agg_inventory_monthly` (inventory) — aliased `i`
- `fact_external_forecast_monthly` (forecast) — aliased `f`
- `dim_dfu` (attributes) — aliased `d` (LEFT JOIN)

**Join conditions:**
- `i.item_no = f.dmdunit`
- `i.loc = f.loc`
- `i.month_start = f.startdate`
- `f.lag = COALESCE(d.execution_lag, 0)` — operational forecast only

**Grain:** `item_no + loc + month_start + model_id`

**Columns:**

| Column | Type | Source/Derivation |
|--------|------|-------------------|
| `item_no` | text | `i.item_no` |
| `loc` | text | `i.loc` |
| `month_start` | date | `i.month_start` |
| `model_id` | text | `f.model_id` |
| `forecast` | numeric | `f.base_forecast` |
| `actual_demand` | numeric | `f.actual_demand` |
| `forecast_error` | numeric | `forecast - actual_demand` |
| `abs_error` | numeric | `ABS(forecast_error)` |
| `eom_qty_on_hand` | numeric | `i.eom_qty_on_hand` |
| `avg_daily_sls` | numeric | `i.avg_daily_sls` |
| `dos` | numeric | `eom_qty_on_hand / avg_daily_sls` (NULL if zero sales) |
| `latest_lead_time_days` | numeric | `i.latest_lead_time_days` |
| `is_stockout` | boolean | `eom_qty_on_hand <= 0` |
| `is_excess` | boolean | `dos > 90` |
| `bias_direction` | text | `'over'` / `'under'` / `'exact'` based on forecast_error |
| `cluster_assignment` | text | From `dim_dfu` (COALESCE default `'unassigned'`) |
| `abc_vol` | text | From `dim_dfu` (COALESCE default `'unknown'`) |
| `region` | text | From `dim_dfu` (COALESCE default `'unknown'`) |
| `brand` | text | From `dim_dfu` (COALESCE default `'unknown'`) |

**Indexes:**
1. Unique PK: `(item_no, loc, month_start, model_id)`
2. `model_id`
3. `month_start`
4. `cluster_assignment`
5. Partial index on `is_stockout = TRUE`
6. Partial index on `is_excess = TRUE`

**File:** `sql/019_inventory_forecast_view.sql`

---

## API Endpoints

All endpoints live in `api/main.py` and use a shared `_inv_backtest_filters()` helper for WHERE clause construction.

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
| `excess_dos_threshold` | int | `90` | Days threshold for excess classification |

### `GET /inventory-backtest/summary`

Per-model aggregate inventory outcome metrics.

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

### `GET /inventory-backtest/trend`

Monthly inventory outcome trend by model.

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

Stockout/excess event root cause breakdown by forecast bias direction.

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

Paginated DFU-level inventory event rows.

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

1. **KPI Cards** — Best Service Level, Lowest Stockout Rate, Lowest Excess Rate, Models Compared, DFU-Months (severity-coded)

2. **Filter Controls** — Item/Location/Cluster text inputs with debounced search, model multi-select pill buttons

3. **Model Comparison Chart** — Recharts `ComposedChart` with grouped bars (stockout_rate + excess_rate per model) and WAPE line overlay on right Y-axis

4. **Root Cause Breakdown** — Horizontal stacked `BarChart` showing stockout/excess event counts split by bias direction (under/over/exact) for a selected model

5. **Monthly Trend Chart** — `LineChart` with one line per model, switchable metric (stockout_rate / excess_rate / avg_dos / wape)

6. **DFU-Level Detail Table** — Event type filter (All/Stockout/Excess), sortable columns, color-coded rows (red=stockout, amber=excess), paginated with Prev/Next

### Navigation

- Sidebar: `Activity` icon from lucide-react, shortcut `6`, section: `supply`
- URL: `?tab=invBacktest`
- Keyboard shortcut: `6`

---

## Makefile Targets

```bash
make db-apply-inv-backtest   # Create materialized view DDL
make refresh-inv-backtest    # Refresh with current data
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
| `api/main.py` | Edited — 4 endpoints + filter helper (~250 lines) |
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

## Implementation Corrections

### Materialized View Schema
3 additional columns implemented but not in spec:
- `eom_qty_on_hand_on_order` (from `i.eom_qty_on_hand_on_order`)
- `monthly_sales` (from `i.monthly_sales`)
- `snapshot_days` (from `i.snapshot_days`)

### Source Column Names
- Spec says `f.base_forecast` → actual: `f.basefcst_pref AS forecast`
- Spec says `f.actual_demand` → actual: `f.tothist_dmd AS actual_demand`

### SQL WHERE Filters (not in spec)
```sql
AND f.tothist_dmd IS NOT NULL
AND f.basefcst_pref IS NOT NULL
```

### COALESCE Values
- Actual uses `'(unassigned)'` and `'(unknown)'` (with parentheses, not plain `unassigned`/`unknown`)

### View Creation
- Created with `WITH NO DATA` — requires explicit `REFRESH MATERIALIZED VIEW`

### Additional Index
- `idx_mv_inv_fcst_abc` on `abc_vol` (7th index, not documented in spec's 6)

### Caching
- Summary/Trend/Root Cause: `max_age=120` (2 minutes)
- Detail: `max_age=60` (1 minute)

### Frontend Details
- Uses `KpiCard`, `LoadingElement`, `useDebounce` (400ms), `useGlobalFilterContext`
- Auto-selects first 5 models on initial load
- KPI severity: service level `>=95` best, `<90` warning
- Page size: 50 rows


---

## Examples

### Example: Inventory backtest summary — model comparison

```bash
curl -s "http://localhost:8000/inv-backtest/summary" | jq '.rows[] | {model_id, stockout_pct, excess_pct, wape}'
# {"model_id": "lgbm_cluster", "stockout_pct": 2.1, "excess_pct":  8.3, "wape":  6.9}
# {"model_id": "lgbm_global",  "stockout_pct": 3.4, "excess_pct":  9.8, "wape":  8.5}
# {"model_id": "external",     "stockout_pct": 4.7, "excess_pct": 12.1, "wape": 12.8}
```

### Example: Root cause attribution — bias direction vs stockouts

```sql
-- Which model causes most stockouts due to systematic under-forecasting?
SELECT model_id, bias_direction, COUNT(*) AS n_events, AVG(abs_error) AS avg_abs_error
FROM mv_inventory_forecast_monthly
WHERE is_stockout = TRUE AND month_start >= '2025-08-01'
GROUP BY model_id, bias_direction
ORDER BY n_events DESC LIMIT 5;
-- external  | under | 847 | 142.3
-- external  | over  |  12 |  18.7
-- lgbm_global | under | 321 |  87.1
```

### Example: Refresh inventory-forecast bridge

```bash
make refresh-inv-backtest
# REFRESH MATERIALIZED VIEW mv_inventory_forecast_monthly
# Joins agg_inventory_monthly + fact_external_forecast_monthly + dim_dfu
# Computes: forecast_error, abs_error, dos, is_stockout, is_excess, bias_direction
# Result: 42,847 rows (DFU × month × model grain)
```

### Example: Monthly trend endpoint

```bash
curl -s "http://localhost:8000/inv-backtest/trend?model=lgbm_cluster&months=6" | jq '.rows[0]'
# {"month": "2025-08-01", "stockout_pct": 1.8, "excess_pct": 7.2, "service_level": 98.2}
```
