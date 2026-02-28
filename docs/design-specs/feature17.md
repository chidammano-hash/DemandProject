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

---

## Implementation Corrections

### Actual Backend Parameters
- `mode`, `item`, `location`, `points`, `seasonality_profile` (NOT `kpi_months` or `sales_metric`)

### Actual Response Shape
- Series keys: `tothist_dmd`, `qty_shipped`, `qty_ordered` (not single `sales` key)
- `model_monthly`: raw monthly forecast/actual pairs for client-side KPI computation (not server-side `kpis`)
- `dfu_attributes`: up to 20 DFU attribute records from `dim_dfu` (34+ columns including seasonality)
- `scope_count`: count of distinct locations or items for aggregated modes
- `points` included in response

### KPI Computation
- KPIs computed client-side in `DfuAnalysisTab.tsx` via `useMemo` with configurable `dfuKpiMonths` window (NOT server-side)

### File Locations
- Tab: `frontend/src/tabs/DfuAnalysisTab.tsx` (not `App.tsx`)
- Router: `api/routers/analysis.py` (also in `api/main.py` inline)

### Frontend Features (not in original spec)
- Global Filter Integration: syncs item/location from `GlobalFilterContext`
- Auto-Sampling: fetches sample pair from `/domains/sales/sample-pair` on first visit
- DFU Attributes Panel: collapsible `<details>` section showing all DFU attributes
- Time Range Filter: From/To month selectors with "Show All" and "Default" buttons
- Visible Measures Toggles: checkboxes for sales measures AND forecast models with Select All/Deselect All
- Cross-filtered typeahead: item suggestions filtered by selected location and vice versa
- Scope count badge on chart title for aggregated modes

### Key Files
| File | Purpose |
|------|---------|
| `mvp/demand/api/routers/analysis.py` | Router version of the endpoint |
| `mvp/demand/frontend/src/tabs/DfuAnalysisTab.tsx` | Extracted tab component |
| `mvp/demand/frontend/src/types/index.ts` | `DfuAnalysisMode`, `DfuAnalysisKpis`, `DfuModelMonthly`, `DfuAnalysisPayload` |
| `mvp/demand/frontend/src/api/queries.ts` | `fetchDfuAnalysis()` |
| `mvp/demand/frontend/src/constants/colors.ts` | `DFU_SALES_COLORS`, `dfuModelColor()` |
| `mvp/demand/tests/api/test_dfu_analysis.py` | Backend API tests |
| `mvp/demand/frontend/src/tabs/__tests__/DfuAnalysisTab.test.tsx` | Frontend smoke test |


---

## Examples

### Example: DFU Analysis endpoint — item+location overlay

```bash
curl -s "http://localhost:8000/dfu/analysis?item=100320&loc=1401-BULK&mode=item_location&window=12" \
  | jq '{series_count: (.series | length), models: [.model_monthly[].model_id] | unique}'
# {"series_count": 6, "models": ["ceiling","champion","external","lgbm_cluster","lgbm_global"]}
```

### Example: Three analysis scope modes

```typescript
// Mode 1: single DFU
fetchDfuAnalysis({ item: '100320', loc: '1401-BULK', mode: 'item_location' })

// Mode 2: item across all locations
fetchDfuAnalysis({ item: '100320', loc: '', mode: 'all_locations_for_item' })

// Mode 3: all items at one location
fetchDfuAnalysis({ item: '', loc: '1401-BULK', mode: 'all_items_at_location' })
```

### Example: Per-model KPI cards computed client-side

```typescript
// useMemo computes WAPE, bias, accuracy% from model_monthly data
const kpis = useMemo(() => {
  return modelMonthly
    .filter(r => r.model_id === selectedModel)
    .reduce((acc, r) => ({
      totalForecast: acc.totalForecast + r.basefcst_pref,
      totalActual: acc.totalActual + r.tothist_dmd,
      absError: acc.absError + Math.abs(r.basefcst_pref - r.tothist_dmd),
    }), { totalForecast: 0, totalActual: 0, absError: 0 })
}, [modelMonthly, selectedModel])
// accuracy_pct = 100 - (100 * absError / Math.abs(totalActual))
```

### Example: Toggleable measure checkboxes — Recharts series visibility

```typescript
// DfuAnalysisTab.tsx — visible measures state drives chart rendering
const [visibleMeasures, setVisibleMeasures] = useState<Set<string>>(
  new Set(['tothist_dmd', 'forecast_external', 'forecast_champion'])
)

// Recharts LineChart: only render Lines for visible measures
{series.length > 0 && visibleMeasures.has('tothist_dmd') && (
  <Line dataKey="tothist_dmd" stroke="#4ade80" strokeWidth={2.5}
        dot={false} name="Sales (History)" />
)}
{models.map(model =>
  visibleMeasures.has(`forecast_${model}`) && (
    <Line key={model} dataKey={`forecast_${model}`}
          stroke={dfuModelColor(model)} strokeWidth={1.5}
          strokeDasharray={model === 'champion' ? undefined : '4 2'}
          dot={false} name={model} />
  )
)}

// Checkbox panel — "Select All" / "Deselect All"
<button onClick={() => setVisibleMeasures(new Set(allKeys))}>Select All</button>
<button onClick={() => setVisibleMeasures(new Set())}>Deselect All</button>
```

### Example: Error boundary wrapping for DFU Analysis tab

```typescript
// App.tsx — each tab is wrapped in a Suspense + ErrorBoundary
<ErrorBoundary fallback={<TabErrorFallback tab="DFU Analysis" />}>
  <Suspense fallback={<ChemistryLoader element="Da" label="DFU Analysis" />}>
    <DfuAnalysisTab />
  </Suspense>
</ErrorBoundary>

// If /dfu/analysis throws 500, error boundary catches and renders:
// "DFU Analysis tab failed to load. Check API connection."
// User can click "Retry" to reset the boundary and re-fetch.
```
