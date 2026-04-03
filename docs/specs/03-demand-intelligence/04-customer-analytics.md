# Spec: Customer Analytics Dashboard

**Spec ID:** 03-04
**Domain:** Demand Intelligence
**Status:** Implementation
**Date:** 2026-04-03

---

## Overview

Enhance the customer geographic visualization from a simple count-of-customers
choropleth to a demand-aware analytics dashboard. The new **Customer Analytics**
tab joins `fact_customer_demand_monthly` with `dim_customer`, `dim_location`,
and `dim_item` to show demand volume, fill rate, OOS hotspots, channel mix,
and customer concentration across 7 panel types.

## Data Sources

| Table | Role |
|-------|------|
| `fact_customer_demand_monthly` | Demand, sales, OOS volumes (monthly grain) |
| `dim_customer` | Geography (state/city/zip), channel, store type, chain |
| `dim_location` | Warehouse state |
| `dim_item` | Item descriptions for picker / heatmap axis |

## API Router

**File:** `api/routers/core/customer_analytics.py`
**Prefix:** `/customer-analytics`

### Endpoints

| # | Method | Path | Description |
|---|--------|------|-------------|
| 1 | GET | `/customer-analytics/map` | Demand-aware map data by state/city/zip |
| 2 | GET | `/customer-analytics/treemap` | Customer concentration hierarchy |
| 3 | GET | `/customer-analytics/heatmap` | Item x State demand matrix |
| 4 | GET | `/customer-analytics/channel-mix` | Channel/store-type sunburst hierarchy |
| 5 | GET | `/customer-analytics/segment-trends` | Monthly demand by segment (sparklines) |
| 6 | GET | `/customer-analytics/ranking` | Top/bottom customers by metric |
| 7 | GET | `/customer-analytics/oos-impact` | Bubble chart: demand vs fill rate |
| 8 | GET | `/customer-analytics/items` | Typeahead item search for filters |

### Common Filter Parameters

All endpoints (except `/items`) accept:
- `item_id` (optional) — filter to single item
- `date_from`, `date_to` (optional) — date range, default last 12 months
- `channel` (optional) — `rpt_channel_desc` filter
- `store_type` (optional) — `store_type_desc` filter

### SQL Pattern

```sql
FROM fact_customer_demand_monthly f
JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
WHERE f.startdate BETWEEN %s AND %s
  [AND f.item_id = %s]
  [AND c.rpt_channel_desc = %s]
GROUP BY ...
```

## Frontend Components

### Tab: `CustomerAnalyticsTab.tsx`

Shared filter bar at top (item picker, date range, channel, store type)
with 7 panels in a 2-column grid below.

### Panels

| Panel | Chart Library | Component File |
|-------|---------------|----------------|
| Enhanced Demand Map | Leaflet + circle markers | `CustomerDemandMap.tsx` |
| Customer Treemap | ECharts treemap | `CustomerTreemap.tsx` |
| Item x State Heatmap | ECharts heatmap | `CustomerHeatmap.tsx` |
| Channel Mix Sunburst | ECharts sunburst | `ChannelSunburst.tsx` |
| Segment Trend Sparklines | Recharts area | `SegmentSparklines.tsx` |
| Customer Ranking | Recharts bar | `CustomerRanking.tsx` |
| OOS Impact Bubble | ECharts scatter | `OosImpactBubble.tsx` |

### Query Module

`frontend/src/api/queries/customer-analytics.ts` — fetch functions + query keys.

## Wiring

1. Router mounted in `api/main.py` before `domains.py`
2. Vite proxy: `/customer-analytics` -> `:8000`
3. Sidebar: `customerAnalytics` entry in `demand` section (after Customer Map)
4. App.tsx: lazy import + TabPanel block
5. `useUrlState.ts`: add `"customerAnalytics"` to VALID_TABS

## Testing

- Backend: `tests/api/test_customer_analytics.py` — mock DB, test all 8 endpoints
- Frontend: `frontend/src/tabs/__tests__/CustomerAnalyticsTab.test.tsx` — render, filter, panels
