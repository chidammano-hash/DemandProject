# Feature 17 — DFU Analysis Tab (Sales vs Multi-Model Forecast Overlay)

## Overview

A unified analysis tab that overlays sales history and multi-model forecast predictions on a single chart for a given DFU scope. Users can analyze demand patterns at three levels of granularity: single DFU (item @ location), all items at a location, or an item across all locations.

## Problem

The existing Sales and Forecast tabs show data independently. Users need to visually compare actual sales against multiple forecasting models on the same chart to:
1. Identify which models track demand patterns most closely
2. Spot forecast divergence from actuals at specific time periods
3. Compare model KPIs side-by-side for a specific DFU or aggregation scope

## Architecture

### Data Flow

```
React UI (DFU Analysis tab)
        ↓
  GET /dfu/analysis?mode=&item=&location=&points=&kpi_months=&sales_metric=
        ↓
  ┌─────────────────────────────────────┐
  │ 1. Sales trend (agg_sales_monthly)  │
  │ 2. Forecast trend per model         │
  │    (agg_forecast_monthly)           │
  │ 3. KPIs per model (windowed)        │
  └─────────────────────────────────────┘
        ↓
  Pre-pivoted series + per-model KPIs
        ↓
  Recharts LineChart with toggleable measures
```

### Analysis Modes

| Mode | Filter | Aggregation |
|------|--------|-------------|
| `item_location` | dmdunit = X AND loc = Y | Single DFU |
| `all_items_at_location` | loc = Y | SUM across all items at location |
| `item_at_all_locations` | dmdunit = X | SUM across all locations for item |

### Backend Endpoint

**`GET /dfu/analysis`** — New endpoint in `api/main.py`

Parameters:
- `mode`: `item_location` | `all_items_at_location` | `item_at_all_locations`
- `item`: dmdunit value
- `location`: loc value
- `points`: Number of monthly data points (3–120, default 36)
- `kpi_months`: Rolling KPI window (1–24, default 12)
- `sales_metric`: `qty` | `qty_shipped` | `qty_ordered` (default `qty`)

Queries pre-aggregated materialized views (`agg_sales_monthly`, `agg_forecast_monthly`) for performance. Returns pre-pivoted series data with `sales` and `forecast_{model_id}` keys for direct chart consumption.

Response shape:
```json
{
  "mode": "item_location",
  "item": "100320",
  "location": "1401-BULK",
  "models": ["external", "lgbm_global", "champion"],
  "series": [
    {"month": "2023-01-01", "sales": 1234.0, "forecast_external": 1100.0, ...}
  ],
  "kpis": {
    "external": {"accuracy_pct": 85.3, "wape": 14.7, "bias": -0.05, "sum_forecast": 12000, "sum_actual": 12600, "months_covered": 12}
  }
}
```

### Frontend Tab

**Tab element:** `Da` (DFU Analysis), number 6, cyan color scheme

**Controls:**
- Analysis Scope dropdown (3 modes)
- Item input with typeahead (via `/domains/sales/suggest`)
- Location input with typeahead (via `/domains/sales/suggest`)
- Sales measure selector (qty/qty_shipped/qty_ordered)
- Points and KPI window selectors

**Chart:** Recharts LineChart with single Y-axis (left). Sales line is thicker/solid (2.5px), forecast lines are thinner/dashed (1.5px). Champion model gets solid line. Each series has a color-coded checkbox toggle.

**KPI Cards:** One card per visible model showing Accuracy %, WAPE %, Bias, Total Forecast, Total Actual. Color-coded to match chart line colors.

## Key Files

| File | Purpose |
|------|---------|
| `mvp/demand/api/main.py` | `GET /dfu/analysis` endpoint |
| `mvp/demand/frontend/src/App.tsx` | DFU Analysis tab UI |
| `mvp/demand/sql/008_perf_indexes_and_agg.sql` | Source materialized views (`agg_sales_monthly`, `agg_forecast_monthly`) |

## Dependencies

- Reuses existing `_compute_kpis()` helper for per-model KPI computation
- Reuses existing `/domains/sales/suggest` and `/domains/sales/sample-pair` endpoints for typeahead and auto-sampling
- Reuses existing `/domains/forecast/models` for model discovery
- Requires `agg_sales_monthly` and `agg_forecast_monthly` materialized views to be populated
