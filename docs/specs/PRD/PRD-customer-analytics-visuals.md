# PRD: Customer Analytics & Enhanced Map Visuals

**Status:** Draft
**Author:** Auto-generated
**Date:** 2026-04-03
**Domain:** Demand Intelligence / Customer Analytics

---

## 1. Problem Statement

The current Customer Map tab (`CustomerMapTab.tsx`) shows a simple choropleth of **customer count by state/city/zip** using `dim_customer` alone. It does not incorporate any transactional data from `fact_customer_demand_monthly` — meaning the map cannot answer the most critical supply chain questions:

- Which states/regions generate the most **demand volume** for a given item?
- How many **unique customers** order a specific item, and where are they?
- Where is **unfulfilled demand** (OOS) concentrated geographically?
- How does customer **demand mix** vary by channel, store type, or chain?

With `fact_customer_demand_monthly` now loaded, we have item-level, customer-level, location-level monthly demand/sales/OOS data ready to power a significantly richer analytics experience.

---

## 2. Goals

| # | Goal | Success Metric |
|---|------|----------------|
| G1 | Show demand volume on the map, not just customer count | Map bubbles/choropleth reflect `demand_qty` and `sales_qty` |
| G2 | Enable item-level filtering on the map | User can select an item and see its customer footprint |
| G3 | Surface OOS/fill-rate geography | Users identify stockout hotspots by region |
| G4 | Add industry-standard supply chain visuals beyond the map | 4+ new chart types shipped |
| G5 | Leverage existing segmentation fields | Channel, store type, chain type visible in analytics |

---

## 3. Data Foundation

### Available Tables

| Table | Grain | Key Columns for This Feature |
|-------|-------|------------------------------|
| `fact_customer_demand_monthly` | item_id + customer_no + location_id + month | `demand_qty`, `sales_qty`, `oos_qty`, `startdate` |
| `dim_customer` | customer_no + site | `city`, `state`, `zip`, `rpt_channel_desc`, `store_type_desc`, `chain_type_desc`, `corp_chain_name`, `customer_name`, `status` |
| `dim_location` | location_id | `state_id`, `site_id` |
| `dim_item` | item_id | `item_desc`, `class`, `dept` (for item picker) |

### Data Volume Estimates

- `dim_customer`: ~10K-50K rows (US customers with zip/state)
- `fact_customer_demand_monthly`: millions of rows (monthly grain, partitioned)
- Aggregation will happen server-side; frontend receives pre-aggregated JSON

---

## 4. Feature Specifications

### 4.1 Enhanced Customer Demand Map (Replaces Current Map)

**What changes:** The existing customer map evolves from a "count of customers" view to a **demand-aware geographic visualization**.

#### 4.1.1 Map Metric Selector

Add a metric toggle to control what the map displays:

| Metric | Source | Bubble/Choropleth Meaning |
|--------|--------|--------------------------|
| Customer Count | `COUNT(DISTINCT customer_no)` | Number of unique customers (current behavior) |
| Demand Volume | `SUM(demand_qty)` | Total cases ordered |
| Sales Volume | `SUM(sales_qty)` | Total cases shipped |
| OOS Volume | `SUM(oos_qty)` | Total unfulfilled cases |
| Fill Rate % | `SUM(sales_qty) / NULLIF(SUM(demand_qty), 0) * 100` | Service level percentage |

**Default metric:** Demand Volume

#### 4.1.2 Item Filter

- Searchable dropdown (typeahead) backed by `dim_item` distinct values
- When an item is selected:
  - Map shows only customers who ordered that item
  - Metrics aggregate to the selected item
  - Title updates: "Customer Demand Map — {item_desc}"
- When no item is selected (default): shows all-item aggregate

#### 4.1.3 Time Range Filter

- Month range picker (min/max from `fact_customer_demand_monthly.startdate`)
- Default: last 12 months
- Aggregates metrics across the selected time window

#### 4.1.4 Dual-Layer Map

- **Choropleth base layer** (state level): shaded by the selected metric
- **Bubble overlay**: sized proportionally to metric value, colored by a secondary dimension:
  - Default: bubble color = fill rate gradient (green 95%+ to red <80%)
  - Option: bubble color = channel/store type categorical palette

#### 4.1.5 Map Interactions

- **Hover tooltip**: state name, metric value, customer count, top 3 items (by demand)
- **Click to drill**: state click zooms in and re-aggregates to city/zip level within that state
- **Legend**: dynamic legend reflecting current metric and color scale

---

### 4.2 Customer Concentration Treemap

**Industry Standard:** Treemap visualizations are widely used in supply chain analytics for showing hierarchical demand concentration (Pareto analysis).

**Visual:** ECharts treemap (`echarts-for-react`)

**Hierarchy:** State > Channel > Top Customers

**Purpose:** Answer "Which customers drive the most demand for this item?"

| Level | Label | Size | Color |
|-------|-------|------|-------|
| State | State name | SUM(demand_qty) | Fill rate gradient |
| Channel | `rpt_channel_desc` | SUM(demand_qty) | Channel categorical |
| Customer | `customer_name` | SUM(demand_qty) | Fill rate gradient |

**Interactions:**
- Click to drill from state to channel to customer
- Hover shows: customer name, demand qty, sales qty, OOS qty, fill rate %
- Item filter and time range filter shared with the map

---

### 4.3 Customer Demand Heatmap (Item x State Matrix)

**Industry Standard:** Heatmaps are a core visualization in demand planning for identifying which products sell where and spotting white-space opportunities.

**Visual:** ECharts heatmap

**Axes:**
- **Y-axis:** Top N items (by total demand, default N=25, configurable)
- **X-axis:** States (sorted by total demand descending)
- **Cell color:** Demand volume (sequential color scale: white to dark blue)
- **Cell tooltip:** Item, state, demand qty, customer count, fill rate %

**Variant toggle:**
- "Demand Volume" (default) — `SUM(demand_qty)`
- "Customer Reach" — `COUNT(DISTINCT customer_no)`
- "Fill Rate" — diverging color scale (red < 85%, yellow 85-95%, green > 95%)

**Purpose:** Quickly identify:
- Items with narrow vs. wide geographic distribution
- States with demand gaps (white space) for specific items
- Fill rate problem spots by item x geography

---

### 4.4 Channel Mix Sunburst

**Industry Standard:** Sunburst charts show hierarchical composition — standard in retail/CPG analytics for understanding channel performance.

**Visual:** ECharts sunburst chart

**Hierarchy (inside-out):**
1. `rpt_channel_desc` (innermost ring)
2. `store_type_desc`
3. `rpt_sub_channel_desc`

**Metric:** Ring thickness proportional to `SUM(demand_qty)`

**Color:** Each channel gets a distinct base hue; shades darken at deeper levels

**Interactions:**
- Click a segment to zoom into that channel
- Hover shows: segment name, demand qty, % of total, customer count, fill rate
- Center text shows current selection context

**Filters:** Item filter, time range, state filter (optional)

---

### 4.5 Demand Trend Sparklines by Customer Segment

**Industry Standard:** Small multiples / sparkline grids are a best practice for comparing trends across segments without chart clutter.

**Visual:** Recharts sparkline grid (small area charts in a table layout)

**Layout:** Table with one row per segment, sparkline column for demand trend

| Segment | Customers | Demand (12M) | Trend (12 months) | Fill Rate | MoM Change |
|---------|-----------|--------------|---------------------|-----------|------------|
| On-Premise | 1,245 | 450K | [sparkline] | 94.2% | +2.1% |
| Off-Premise | 3,891 | 1.2M | [sparkline] | 91.8% | -0.4% |
| ... | ... | ... | ... | ... | ... |

**Segment dimension selector:** `rpt_channel_desc` (default), `store_type_desc`, `chain_type_desc`, `state`

**Purpose:** Compare demand trajectories across customer segments at a glance. Spot segments growing, declining, or with service issues.

---

### 4.6 Top/Bottom Customer Ranking Bar Chart

**Industry Standard:** Ranked bar charts are a staple in account management dashboards for prioritization.

**Visual:** Recharts horizontal bar chart (bidirectional)

**Two views (tab toggle):**

**A) Top N Customers by Demand**
- Horizontal bars sorted descending
- Bar length = `SUM(demand_qty)`
- Bar color = fill rate gradient
- Labels: customer name, demand qty, fill rate %, state

**B) Bottom N Customers by Fill Rate (with minimum demand threshold)**
- Horizontal bars sorted ascending by fill rate
- Bar length = fill rate %
- Bar color = red to yellow gradient
- Filter: only customers with `SUM(demand_qty) > threshold` (avoid noise from tiny accounts)
- Labels: customer name, fill rate %, OOS qty, state

**Default N:** 20 (configurable slider 10-50)

**Filters:** Item, time range, state, channel

---

### 4.7 OOS (Out-of-Stock) Impact Bubble Chart

**Industry Standard:** Bubble charts are used in supply chain analytics to show the relationship between multiple dimensions simultaneously — a common lens for prioritizing stockout remediation.

**Visual:** ECharts scatter/bubble chart

**Axes:**
- **X-axis:** Total demand volume (`SUM(demand_qty)`)
- **Y-axis:** Fill rate % (`SUM(sales_qty) / SUM(demand_qty) * 100`)
- **Bubble size:** OOS volume (`SUM(oos_qty)`)
- **Bubble color:** Channel (categorical)

**Grain:** One bubble per customer (or per state, togglable)

**Quadrant logic:**
- Top-right (high demand, high fill rate): healthy accounts
- Top-left (low demand, high fill rate): small but well-served
- Bottom-right (high demand, low fill rate): **critical action items**
- Bottom-left (low demand, low fill rate): deprioritize or investigate

**Interactions:**
- Hover: customer name, demand, sales, OOS, fill rate, channel, state
- Click: drill into customer's monthly trend
- Lasso select: group selection for export / action

---

## 5. API Endpoints (New / Modified)

### 5.1 New Endpoints

All endpoints go in a new router: `api/routers/core/customer_analytics.py`
Prefix: `/customer-analytics`

| Method | Path | Purpose | Query Params |
|--------|------|---------|--------------|
| GET | `/customer-analytics/map` | Enhanced demand-aware map data | `metric`, `group_by`, `item_id`, `date_from`, `date_to`, `channel`, `store_type` |
| GET | `/customer-analytics/treemap` | Concentration treemap data | `item_id`, `date_from`, `date_to` |
| GET | `/customer-analytics/heatmap` | Item x State heatmap matrix | `metric`, `top_n`, `date_from`, `date_to` |
| GET | `/customer-analytics/channel-mix` | Sunburst hierarchy data | `item_id`, `date_from`, `date_to`, `state` |
| GET | `/customer-analytics/segment-trends` | Sparkline trend data by segment | `segment_by`, `item_id`, `date_from`, `date_to` |
| GET | `/customer-analytics/ranking` | Top/bottom customer ranking | `metric`, `sort`, `top_n`, `item_id`, `date_from`, `date_to`, `min_demand` |
| GET | `/customer-analytics/oos-impact` | Bubble chart data | `grain`, `item_id`, `date_from`, `date_to`, `channel` |
| GET | `/customer-analytics/items` | Item picker dropdown data | `search` (typeahead, limit 50) |

### 5.2 Modified Endpoints

| Endpoint | Change |
|----------|--------|
| `GET /dashboard/customer-map` | Deprecate in favor of `/customer-analytics/map`. Keep for backward compat with redirect note. |

### 5.3 Common Query Pattern

All endpoints join:
```sql
FROM fact_customer_demand_monthly f
JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
JOIN dim_location l ON l.location_id = f.location_id
LEFT JOIN dim_item i ON i.item_id = f.item_id
WHERE f.startdate BETWEEN %(date_from)s AND %(date_to)s
  [AND f.item_id = %(item_id)s]
  [AND c.rpt_channel_desc = %(channel)s]
  [AND c.store_type_desc = %(store_type)s]
```

Aggregation happens in SQL (GROUP BY state/city/zip/customer depending on endpoint). Frontend receives pre-aggregated JSON arrays.

---

## 6. Frontend Architecture

### 6.1 New Tab: CustomerAnalyticsTab

**Location:** `frontend/src/tabs/CustomerAnalyticsTab.tsx`

**Layout:** Full-width dashboard with a shared filter bar at the top and a panel grid below.

```
+------------------------------------------------------------------+
|  [Item Filter v]  [Date Range]  [Channel v]  [Store Type v]      |
+------------------------------------------------------------------+
|                                                                    |
|  +-----------------------------+  +-----------------------------+  |
|  |                             |  |                             |  |
|  |    Enhanced Demand Map      |  |   Customer Concentration    |  |
|  |    (Leaflet + ECharts)      |  |   Treemap (ECharts)         |  |
|  |                             |  |                             |  |
|  +-----------------------------+  +-----------------------------+  |
|                                                                    |
|  +-----------------------------+  +-----------------------------+  |
|  |                             |  |                             |  |
|  |  Item x State Heatmap      |  |   Channel Mix Sunburst      |  |
|  |  (ECharts)                  |  |   (ECharts)                 |  |
|  |                             |  |                             |  |
|  +-----------------------------+  +-----------------------------+  |
|                                                                    |
|  +-----------------------------+  +-----------------------------+  |
|  |                             |  |                             |  |
|  |  Segment Trend Sparklines   |  |   OOS Impact Bubble Chart   |  |
|  |  (Recharts)                 |  |   (ECharts)                 |  |
|  |                             |  |                             |  |
|  +-----------------------------+  +-----------------------------+  |
|                                                                    |
|  +--------------------------------------------------------------+  |
|  |  Top/Bottom Customer Ranking (Recharts bar chart)            |  |
|  +--------------------------------------------------------------+  |
+------------------------------------------------------------------+
```

### 6.2 Shared Filter State

Use React context or lifted state in the tab component:

```typescript
interface CustomerAnalyticsFilters {
  itemId: string | null;
  dateFrom: string;      // YYYY-MM-DD
  dateTo: string;        // YYYY-MM-DD
  channel: string | null;
  storeType: string | null;
  state: string | null;
}
```

All child panels consume these filters and pass them to their respective API queries.

### 6.3 Query Module

**Location:** `frontend/src/api/queries/customerAnalytics.ts`

Export individual fetch functions + TanStack React Query hooks for each endpoint.

### 6.4 Existing Tab Disposition

- `CustomerMapTab.tsx` remains as-is for backward compatibility
- Sidebar navigation adds "Customer Analytics" entry pointing to the new tab
- Old "Customer Map" entry can be removed or kept as a lightweight shortcut

---

## 7. Technical Considerations

### 7.1 Performance

| Concern | Mitigation |
|---------|------------|
| `fact_customer_demand_monthly` is large (millions of rows) | All aggregation in SQL with partition pruning on `startdate` |
| Heatmap with 25 items x 50 states = 1,250 cells | Lightweight payload, no concern |
| Treemap with deep drill-down | Lazy-load children on click (API returns one level at a time) |
| Geocoding for zip-level bubbles | Cache in-memory (pgeocode); limit to 500 markers |
| Multiple panels loading simultaneously | `staleTime: 5 * 60 * 1000` on React Query; panels load independently |

### 7.2 Index Coverage

Existing indexes on `fact_customer_demand_monthly` cover the main query patterns:
- `idx_cust_demand_item` — item filter
- `idx_cust_demand_customer` — customer joins
- `idx_cust_demand_startdate` — date range pruning (partition key)
- `idx_cust_demand_item_loc` — item + location compound

**May need:** Composite index on `(item_id, startdate)` if item-filtered date-range queries are slow. Evaluate after initial deployment.

### 7.3 No New Dependencies

All visuals use **existing libraries** (Leaflet, ECharts, Recharts). No new npm packages required.

---

## 8. Phased Delivery

### Phase 1: Core Map Enhancement (High Impact, Medium Effort)

- [ ] New API: `/customer-analytics/map` with metric/item/date filters
- [ ] New API: `/customer-analytics/items` (item picker)
- [ ] Enhanced map component with metric selector + item filter + time range
- [ ] Dual-layer map (choropleth + demand-sized bubbles)
- [ ] Tests (API + frontend)

### Phase 2: Concentration & Distribution Visuals (High Impact, Medium Effort)

- [ ] New API: `/customer-analytics/heatmap`
- [ ] New API: `/customer-analytics/treemap`
- [ ] Item x State heatmap panel
- [ ] Customer concentration treemap panel
- [ ] Tests

### Phase 3: Segmentation & Channel Analytics (Medium Impact, Low Effort)

- [ ] New API: `/customer-analytics/channel-mix`
- [ ] New API: `/customer-analytics/segment-trends`
- [ ] Channel mix sunburst panel
- [ ] Segment trend sparklines panel
- [ ] Tests

### Phase 4: Actionable Insights (High Impact, Low Effort)

- [ ] New API: `/customer-analytics/ranking`
- [ ] New API: `/customer-analytics/oos-impact`
- [ ] Top/bottom customer ranking bar chart
- [ ] OOS impact bubble chart with quadrant annotations
- [ ] Tests

### Phase 5: Tab Assembly & Polish

- [ ] Assemble `CustomerAnalyticsTab.tsx` with shared filter bar
- [ ] Add sidebar navigation entry
- [ ] Add Vite proxy entry for `/customer-analytics`
- [ ] Mount router in `api/main.py`
- [ ] E2E test for tab navigation
- [ ] Documentation updates

---

## 9. Out of Scope

- **Real-time streaming data** — all data is monthly batch
- **Customer-level forecasting** — separate future initiative
- **International geographies** — US-only (pgeocode limitation for now)
- **Export to Excel/PDF** — can be added later as a cross-cutting feature
- **Customer master data editing** — read-only analytics

---

## 10. Open Questions

| # | Question | Impact |
|---|----------|--------|
| Q1 | Should the new Customer Analytics tab replace or coexist with the existing Customer Map tab? | Navigation structure |
| Q2 | Do we need customer-level drill-through to a detail page (order history, trend, etc.)? | Scope of Phase 4+ |
| Q3 | Is there a need for customer grouping beyond what `dim_customer` provides (e.g., custom segments)? | Data model extension |
| Q4 | Should we add lat/lon columns to `dim_customer` or `dim_location` for faster geocoding? | Performance vs. simplicity |
| Q5 | What is the minimum demand threshold for the "Bottom N by Fill Rate" chart? | Business rule from stakeholders |

---

## 11. Visual Reference (Industry Benchmarks)

These visualization types are standard in leading supply chain analytics platforms:

| Visual | Used By | Our Equivalent |
|--------|---------|----------------|
| Geographic demand heatmap | SAP IBP, Kinaxis, o9 | Section 4.1 — Enhanced Demand Map |
| Customer concentration treemap | Tableau Supply Chain, Power BI | Section 4.2 — Treemap |
| Product x Geography matrix | RELEX, Blue Yonder | Section 4.3 — Item x State Heatmap |
| Channel mix sunburst | Nielsen, IRI | Section 4.4 — Channel Mix Sunburst |
| Segment trend sparklines | Gartner dashboards, Looker | Section 4.5 — Sparklines |
| Account ranking waterfall | Salesforce, SAP CRM | Section 4.6 — Ranking Bar Chart |
| Service level bubble matrix | Manhattan Associates, Llamasoft | Section 4.7 — OOS Impact Bubble |

---

## 12. Success Criteria

| Criteria | Target |
|----------|--------|
| All 7 visualization panels render with real data | 100% |
| API response time for any endpoint (p95) | < 2 seconds |
| Frontend bundle size increase | < 50KB gzipped (no new deps) |
| Test coverage for new code | > 80% |
| User can answer "where do my customers buy item X" in < 3 clicks | Yes |
| User can identify top OOS hotspot states in < 10 seconds | Yes |
