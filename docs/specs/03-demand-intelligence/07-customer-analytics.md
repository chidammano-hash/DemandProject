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

**Package:** `api/routers/intelligence/customer_analytics/` (split from the
former 1726-LoC `customer_analytics.py` module — endpoints and `/customer-analytics`
prefix unchanged)
**Prefix:** `/customer-analytics`

### Sub-Routers

| Sub-Router File | Concern | Endpoints |
|---|---|---|
| `geo.py` | Geographic visualizations | `/map`, `/treemap`, `/heatmap`, `/demand-flow` |
| `segments.py` | Channel / segment breakdowns | `/channel-mix`, `/segment-trends`, `/filter-options` |
| `ranking.py` | Customer ranking + behavior | `/ranking`, `/oos-impact`, `/affinity`, `/order-patterns` |
| `lifecycle.py` | Lifecycle + risk views | `/lifecycle`, `/demand-at-risk` |
| `kpis.py` | Item picker + KPI tiles + alerts | `/items`, `/kpis`, `/alerts` |
| `_helpers.py` | Geocoding cache + WHERE-clause builders (shared) | (helpers, no routes) |
| `__init__.py` | Aggregates sub-router includes; re-exports `_get_nomi` and `get_planning_date` for test patching | (package init) |

### Endpoints

| # | Method | Path | Sub-Router | Description |
|---|--------|------|------------|-------------|
| 1 | GET | `/customer-analytics/map` | `geo.py` | Demand-aware map data by state/city/zip |
| 2 | GET | `/customer-analytics/treemap` | `geo.py` | Customer concentration hierarchy |
| 3 | GET | `/customer-analytics/heatmap` | `geo.py` | Item x State demand matrix |
| 4 | GET | `/customer-analytics/demand-flow` | `geo.py` | Origin/destination demand flow |
| 5 | GET | `/customer-analytics/channel-mix` | `segments.py` | Channel/store-type sunburst hierarchy |
| 6 | GET | `/customer-analytics/segment-trends` | `segments.py` | Monthly demand by segment (sparklines) |
| 7 | GET | `/customer-analytics/filter-options` | `segments.py` | Distinct values for filter dropdowns |
| 8 | GET | `/customer-analytics/ranking` | `ranking.py` | Top/bottom customers by metric |
| 9 | GET | `/customer-analytics/oos-impact` | `ranking.py` | Bubble chart: demand vs fill rate |
| 10 | GET | `/customer-analytics/affinity` | `ranking.py` | Customer-item affinity scores |
| 11 | GET | `/customer-analytics/order-patterns` | `ranking.py` | Inter-order intervals / cadence |
| 12 | GET | `/customer-analytics/lifecycle` | `lifecycle.py` | New / growing / declining / churned cohorts |
| 13 | GET | `/customer-analytics/demand-at-risk` | `lifecycle.py` | Demand exposure from at-risk customers |
| 14 | GET | `/customer-analytics/items` | `kpis.py` | Typeahead item search for filters |
| 15 | GET | `/customer-analytics/kpis` | `kpis.py` | Aggregate KPI tiles |
| 16 | GET | `/customer-analytics/alerts` | `kpis.py` | Threshold-based alert feed |

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

- Backend: `tests/api/test_customer_analytics.py` — mock DB, test all 16 endpoints. Patch
  targets live on the package (`api.routers.intelligence.customer_analytics._get_nomi`,
  `…customer_analytics.get_planning_date`), which the package `__init__.py` re-exports
  for test convenience.
- Frontend: `frontend/src/tabs/__tests__/CustomerAnalyticsTab.test.tsx` — render, filter, panels
