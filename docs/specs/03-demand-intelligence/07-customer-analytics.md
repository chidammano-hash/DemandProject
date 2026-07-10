# Spec: Customer Analytics Dashboard

**Spec ID:** 03-07
**Domain:** Demand Intelligence
**Status:** Implemented
**Date:** 2026-04-03

---

## Overview

Enhance the customer geographic visualization from a simple count-of-customers
choropleth to a demand-aware, task-oriented analytics workspace. The **Customer
Analytics** tab joins `fact_customer_demand_monthly` with `dim_customer`,
`dim_location`, and `dim_item` to show demand volume, fill rate, OOS hotspots,
channel mix, customer concentration, lifecycle, and buying behavior. Instead of
mounting every chart in one long page, it groups analysis into five focused
views: **Overview, Customers, Segments, Service risk, and Buying behavior**.

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
| `assistant.py` | Grounded customer-intelligence Q&A | `/ask` |
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
| 17 | POST | `/customer-analytics/ask` | `assistant.py` | Answer from filtered KPIs and customer rankings |

### Common Filter Parameters

All endpoints (except `/items`) accept:
- `item_id` (optional) — filter to single item
- `date_from`, `date_to` (optional) — date range, default last 12 months
- `channel` (optional) — `rpt_channel_desc` filter
- `store_type` (optional) — `store_type_desc` filter
- `state` (optional) — normalized customer-state filter

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

Shared collapsible filter bar at top (item picker, date range, state, channel,
store type), KPI summary, embedded **Customer Intelligence** question bar, and
a five-item task navigation. Only the active view mounts, avoiding the previous
wall of charts while preserving one filter context across planning questions.

### Task Views

| View | Primary planning question | Panels |
|---|---|---|
| Overview | Where is demand and how concentrated is it? | Demand Map, Customer Treemap |
| Customers | Which customers drive value or need retention attention? | Ranking, Lifecycle, Demand at Risk |
| Segments | Which channel/store segments are growing or weakening? | Channel Mix, Segment Trends, Item × State Heatmap |
| Service risk | Where are fill-rate and OOS losses concentrated? | OOS Impact, Demand at Risk, Fill-rate Heatmap |
| Buying behavior | What cadence, affinity, and flow patterns stand out? | Customer-Item Affinity, Order Patterns, Demand Flow |

### Embedded Customer Intelligence

`CustomerAnalyticsAssistant.tsx` posts the question, active view, current
filters, and a bounded six-message history to `POST /customer-analytics/ask`.
The backend refreshes database evidence for that exact scope: aggregate KPIs,
the top five demand customers, and the five lowest-fill-rate customers. The
model must use only this evidence, disclose when it is insufficient, and avoid
claiming causation from correlation.

- **Laptop development:** the default `codex` runtime invokes the same
  read-only `codex exec` path and GPT model tiers as SKU Chat, reusing saved
  Codex/ChatGPT login without a separate API key.
- **Production:** set `CUSTOMER_ANALYTICS_AI_RUNTIME=openai` and provide
  `OPENAI_API_KEY`; `config/ai/customer_analytics_assistant_config.yaml`
  selects the production GPT model and cost controls.

Every response displays provider, model, tier, and evidence-set count. The
global SKU chat drawer is suppressed on this tab so users see one relevant,
database-grounded assistant.

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
- AI runtime: `tests/unit/test_customer_analytics_assistant.py` — Codex-local and OpenAI-production routing
- AI API: `tests/api/test_customer_analytics_assistant.py` — filtered evidence grounding and contract
- Scale: `tests/scale/test_customer_analytics_scale.py` — synthetic 100K-row run by default;
  nightly `make scale-test SCALE=10000000` (10M rows ≈ 40× production) verifies all
  MV-routed endpoints clear the 30s `statement_timeout` ceiling.

---

## Performance Architecture (Items 8 / 9 / 14 / 19 / 24)

The dashboard's first cut hit `fact_customer_demand_monthly` (~50M+ rows) for every
panel. At 1× scale `/kpis` ran 10,805 ms; at 40× scale several panels exceeded the
30s `statement_timeout`. The perf rework shifts almost all "no-item-filter" reads
onto narrow MVs, parallelizes the heaviest panel, and routes selected reads to a
read-replica pool. Net result: `/kpis` 10,805 ms → 63 ms at 1× and the entire tab
survives the 40× scale test.

### MV Topology

| MV | DDL | Role |
|---|---|---|
| `mv_customer_activity_monthly` | `sql/174_extend_mv_customer_activity_geo.sql` | Hot path. Extended with `state`, `city`, `zip`, `customer_name`, `rpt_sub_channel_desc`, `chain_type_desc`, `location_id` so the geo + segment + ranking panels read from a single MV without re-joining `dim_customer` / `dim_location`. 9 of 16 endpoints route through this MV when `item_id` is null. |
| `mv_customer_filter_options` | `sql/173_create_mv_customer_filter_options.sql` | 3-row materialization that replaces three `ARRAY_AGG(DISTINCT ...)` scans of `dim_customer` previously executed on every filter-bar mount. |
| `mv_ca_segment_trends` | `sql/180_create_mv_ca_segment_trends.sql` | Pre-aggregates 12-month segment trend rollups for `/segment-trends`. |
| `mv_ca_demand_at_risk` | `sql/181_create_mv_ca_demand_at_risk.sql` | Pre-computes lifecycle-derived demand-at-risk windows for `/demand-at-risk`. |
| `mv_ca_order_patterns` | `sql/182_create_mv_ca_order_patterns.sql` | Inter-order interval / cadence rollup for `/order-patterns`. |

All five MVs are nightly-refreshed; the underlying tables (`dim_customer`,
`fact_customer_demand_monthly`) change on the same cadence so end-of-day
freshness is sufficient.

**On-demand recalculation.** The tab header carries a **Recalculate** button
(`RecalculateButton`, mirroring SKU Features' Compute button). It `POST`s to
`/customer-analytics/recalculate` (write endpoint, `require_api_key`), which
submits the `refresh_customer_analytics` background job
(`_run_refresh_customer_analytics` in `common/services/job_state.py`). The job
`REFRESH MATERIALIZED VIEW CONCURRENTLY` over all six MVs above —
`mv_customer_activity_monthly` first as the core rollup. The button polls
`/jobs/{id}` and, on completion, invalidates every `customer-analytics-*` React
Query so the panels repaint. Same refresh as the `refresh-customer-mv` /
`refresh-customer-filter-options` / `refresh-ca-mvs` Make targets, run off the
request thread.

### Endpoint Routing Strategy

| Endpoint | Default Source | Falls Back To Fact Table When |
|---|---|---|
| `/kpis` | `mv_customer_activity_monthly` | `item_id` is set |
| `/map` | `mv_customer_activity_monthly` | `item_id` is set |
| `/treemap` | `mv_customer_activity_monthly` | `item_id` is set |
| `/heatmap` | `mv_customer_activity_monthly` | `item_id` is set |
| `/channel-mix` | `mv_customer_activity_monthly` | `item_id` is set |
| `/segment-trends` | `mv_ca_segment_trends` | `item_id` is set |
| `/ranking` | `mv_customer_activity_monthly` | `item_id` is set |
| `/demand-at-risk` | `mv_ca_demand_at_risk` | `item_id` is set |
| `/order-patterns` | `mv_ca_order_patterns` | `item_id` is set |
| `/filter-options` | `mv_customer_filter_options` | (always — 3 rows) |
| `/lifecycle`, `/oos-impact`, `/affinity`, `/demand-flow`, `/alerts`, `/items` | fact tables / dim_customer | n/a |

When a single `item_id` is supplied the panels fall through to the original
fact-table query path so item-grain detail still works — only the dashboard-wide
"all items" view (the dominant access pattern) is MV-accelerated.

### Single-Pass `/kpis` Aggregate

`/kpis` originally ran four CTEs comma-joined together (one base scan per CTE,
then a 4-way Cartesian on single-row results). It is now rewritten as a single
pass over the MV with `FILTER (WHERE …)` aggregates — one base scan, no joins —
which is the bulk of the 10,805 ms → 63 ms win at 1× scale.

### Async + Read-Replica Routing (Item 19 pilot)

All 16 endpoints have been converted to `async def` against the async pool,
freeing the anyio threadpool tokens that previously throttled concurrent dashboard
loads. Cache decorators use `cached_async` from `common/services/cache.py`.

Seven panels route through the read-replica pool via `get_async_read_only_conn()`
(set by `READ_REPLICA_URL`):

- `/kpis`, `/map`, `/treemap`, `/heatmap`, `/channel-mix`, `/segment-trends`,
  `/ranking`

These are pure GETs against MVs that tolerate replica lag, so reads peel off the
primary entirely. The remaining endpoints stay on the primary (`get_async_conn()`)
because they touch `dim_*` lookups whose currency matters more than throughput.

### Parallel Lifecycle Aggregation

`/lifecycle` previously ran four cohort queries serially via a `ThreadPoolExecutor`
that occupied threadpool tokens for the duration of the slowest query. It now uses
`asyncio.gather()` against `get_async_conn()`, which runs the four queries
concurrently on a single worker without tying up the threadpool.

### In-Process Caches

Two long-TTL caches sit in front of dim-customer-cadence data:

| Cache | Endpoints | TTL | Rationale |
|---|---|---|---|
| `_CA_CACHE` | `/items` | 24 h | Item picker payload changes only when `dim_item` reloads. |
| `_CA_FILTER_OPTIONS_CACHE` | `/filter-options` | 24 h | Filter dropdown values change with `dim_customer` reloads. |

Both invalidate on the existing `dim_customer` / `dim_item` reload hooks, so
freshness still tracks ETL.

### Frontend Bundle + Render Wins

| Panel | Before | After | Why |
|---|---|---|---|
| 8 ECharts panels | echarts (canvas, ~1 MB shared chunk) | recharts (SVG) | Reduced raw bundle by ~728 KB across the tab; SVG renders faster on the small datasets these panels return. |
| Specialized panels | All mounted in one long page | Mounted only for the active task view | Reduces initial work and removes the repeated loading-placeholder wall. |

### Performance Snapshot

| Metric | Before | After |
|---|---|---|
| `/kpis` p50 (1× scale) | 10,805 ms | 63 ms |
| `/kpis` at 40× scale | timeout (>30 s) | <30 s |
| Initial-paint panel queries fired | 13 | 2 overview panels + shared KPI/filter lookups |
| Tab raw bundle | baseline | −728 KB |
