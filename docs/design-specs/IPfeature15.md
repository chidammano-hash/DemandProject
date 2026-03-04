# IPfeature15 — Unified Inventory Control Tower

## EPIC
InventoryPlanning

## Status
Planned

## Priority
P2 — Should Have

## Effort
L (Large)

## Expert Perspectives
- **Supply Chain Control Tower Expert** (lead) — command center design, real-time visibility, executive KPIs
- **UI/UX Expert** — 5-zone grid layout, unified alert merging, drill-down flows
- **Inventory Planning Expert** — KPI selection, top-critical item ranking, portfolio health aggregation
- **Demand Planning Expert** — fill rate and demand signal KPI integration
- **Simulation Expert** — SS coverage and portfolio CSL visualization

---

## Problem Statement

After implementing IPfeature1–IPfeature14, the platform has 9+ data sources across separate panels in `InvPlanningTab.tsx`:

- Health scores (IPfeature6)
- Exception queue (IPfeature7)
- Fill rate trend (IPfeature8)
- Demand signals (IPfeature9)
- Intra-month stockouts (IPfeature14)
- Safety stock coverage (IPfeature3)
- Investment opportunity (IPfeature13)

A supply chain executive running a **daily supply review** must jump between 9 panels and 4 tabs to get a complete picture. This is cognitively expensive and error-prone — critical items requiring action are buried in detail tables.

**The Control Tower collapses all upstream features into a single command center** — one screen answering: "What needs my attention today and why?"

---

## User Story

> As a supply chain executive, I want a single command center showing health score distribution, open exception counts, fill rate trend, demand signals, intra-month stockouts, SS coverage, and top-10 critical items — all on one screen — so I can run my daily supply review in under 5 minutes.

---

## Business Value

- Reduces executive daily review from 30–45 minutes (across tabs) to < 5 minutes
- Creates a single authoritative view of supply chain health for weekly business reviews
- Surfaces emerging crises before they hit customers (demand signals + intra-month stockouts)
- Closes the loop between prescriptive recommendations (exceptions) and strategic position (health)
- Provides the "headline number" every day: "Portfolio health: 72/100, 14 open critical exceptions"

---

## Data Requirements

### New DDL: `mvp/demand/sql/035_create_control_tower_kpis.sql`

New materialized view `mv_control_tower_kpis` — thin aggregation layer for fast KPI query:

```sql
CREATE MATERIALIZED VIEW mv_control_tower_kpis AS
SELECT
    -- Computed timestamp
    NOW() AS computed_at,

    -- === Health Score KPIs ===
    COUNT(*)                                          AS total_dfus,
    COUNT(*) FILTER (WHERE health_tier = 'healthy')  AS healthy_count,
    COUNT(*) FILTER (WHERE health_tier = 'monitor')  AS monitor_count,
    COUNT(*) FILTER (WHERE health_tier = 'at_risk')  AS at_risk_count,
    COUNT(*) FILTER (WHERE health_tier = 'critical') AS critical_count,
    AVG(health_score)                                 AS avg_health_score,
    AVG(ss_coverage)                                  AS avg_ss_coverage,
    COUNT(*) FILTER (WHERE is_below_ss = TRUE)        AS below_ss_count,
    CASE WHEN COUNT(*) > 0
         THEN COUNT(*) FILTER (WHERE is_below_ss = TRUE)::NUMERIC / COUNT(*)
         ELSE 0 END                                   AS below_ss_pct,
    AVG(current_dos)                                  AS avg_portfolio_dos,

    -- === Exception KPIs (open only) ===
    (SELECT COUNT(*) FROM fact_replenishment_exceptions
     WHERE status = 'open') AS open_exceptions_total,
    (SELECT COUNT(*) FROM fact_replenishment_exceptions
     WHERE status = 'open' AND severity = 'critical') AS critical_exceptions,
    (SELECT COUNT(*) FROM fact_replenishment_exceptions
     WHERE status = 'open' AND severity = 'high') AS high_exceptions,
    (SELECT SUM(recommended_order_qty * unit_cost)
     FROM fact_replenishment_exceptions e
     JOIN fact_safety_stock_targets s
         ON e.item_no = s.item_no AND e.loc = s.loc AND s.policy_version = 'v1'
     WHERE e.status = 'open') AS recommended_order_value,

    -- === Fill Rate KPIs (latest 3 months) ===
    (SELECT SUM(total_shipped)::NUMERIC / NULLIF(SUM(total_ordered), 0)
     FROM mv_fill_rate_monthly
     WHERE month_start >= (SELECT MAX(month_start) FROM mv_fill_rate_monthly)
                          - INTERVAL '2 months') AS portfolio_fill_rate_3m,
    (SELECT SUM(shortage_qty)
     FROM mv_fill_rate_monthly
     WHERE month_start >= (SELECT MAX(month_start) FROM mv_fill_rate_monthly)
                          - INTERVAL '2 months') AS total_shortage_qty_3m,

    -- === Demand Signal KPIs (today) ===
    (SELECT COUNT(*) FROM fact_demand_signals
     WHERE signal_date = CURRENT_DATE AND alert_priority = 'urgent') AS urgent_demand_signals,
    (SELECT COUNT(*) FROM fact_demand_signals
     WHERE signal_date = CURRENT_DATE AND projected_stockout = TRUE) AS projected_stockouts_today,

    -- === Intra-Month Stockout KPIs (current month) ===
    (SELECT COUNT(*) FROM mv_intramonth_stockout
     WHERE month_start = DATE_TRUNC('month', CURRENT_DATE)
       AND had_full_stockout = TRUE) AS items_with_stockout_this_month,
    (SELECT COUNT(*) FROM mv_intramonth_stockout
     WHERE month_start = DATE_TRUNC('month', CURRENT_DATE)
       AND had_extended_stockout = TRUE) AS extended_stockouts_this_month

FROM mv_inventory_health_score
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_ct_kpis_singleton
    ON mv_control_tower_kpis ((1));   -- single-row view; singleton index for fast refresh
```

---

## API Endpoints

**New Router:** `mvp/demand/api/routers/control_tower.py` (mounted at `/control-tower`)

```
GET /control-tower/kpis
  Response: {
    computed_at: datetime,
    health: {
      total_dfus, healthy_count, monitor_count, at_risk_count, critical_count,
      avg_health_score, avg_ss_coverage, below_ss_count, below_ss_pct, avg_portfolio_dos
    },
    exceptions: {
      open_exceptions_total, critical_exceptions, high_exceptions, recommended_order_value
    },
    fill_rate: {
      portfolio_fill_rate_3m, total_shortage_qty_3m
    },
    demand_signals: {
      urgent_demand_signals, projected_stockouts_today
    },
    intramonth: {
      items_with_stockout_this_month, extended_stockouts_this_month
    }
  }
  Cache: max-age=120s

GET /control-tower/alerts
  Query params: limit (default 20), severity (critical | high | medium | low)
  Response: {
    total: int,
    alerts: [
      {
        alert_id: str,           -- unique composite key
        source: str,             -- 'exception' | 'demand_signal' | 'health_drop'
        severity: str,           -- critical | high | medium | low
        item_no: str,
        loc: str,
        alert_type: str,         -- exception_type | 'below_plan' | 'health_critical'
        description: str,        -- human-readable: "Item X at Loc Y is below ROP by 120 units"
        action: str,             -- "Order 240 units by 2024-03-15" | "Monitor demand pace"
        alert_ts: datetime,      -- when generated
        abc_vol: str
      }
    ]
  }
  -- Merges from 3 sources:
  --   1. fact_replenishment_exceptions (status='open', sorted by severity then exception_date)
  --   2. fact_demand_signals (alert_priority='urgent', signal_date=today)
  --   3. mv_inventory_health_score (health_tier='critical', no open exception already listed)
  -- Sorted: critical first, then high, then by alert_ts desc (most recent)
  Cache: max-age=60s

GET /control-tower/top-critical
  Query params: limit (default 10)
  Response: {
    items: [
      {
        item_no, loc, abc_vol, abc_xyz_segment,
        health_score, health_tier,
        ss_coverage, is_below_ss,
        current_dos, target_dos_min, target_dos_max,
        open_exception_count,
        recommended_order_qty,   -- from fact_replenishment_exceptions if exists
        order_by_date,           -- from exception if exists
        fill_rate_last_3m,       -- from mv_fill_rate_monthly
        stockout_days_this_month -- from mv_intramonth_stockout
      }
    ]
  }
  -- Source: mv_inventory_health_score ORDER BY health_score ASC LIMIT N
  -- Enriched with exception + fill rate + intramonth joins
  Cache: max-age=120s

GET /control-tower/trend
  Query params: months (default 6)
  Response: {
    trend: [
      {
        month_start: date,
        avg_health_score: float,
        fill_rate: float,
        stockout_day_rate: float,    -- avg across all item-locs
        below_ss_pct: float,
        avg_dos: float,
        open_exception_count: int    -- snapshot at month end
      }
    ]
  }
  -- NOTE: health_score, fill_rate, stockout trends from respective views
  -- open_exception_count from job_history or exceptions with monthly bucketing
  Cache: max-age=3600s
```

**Vite proxy:** Add `/control-tower` entry to `mvp/demand/frontend/vite.config.ts`

---

## Frontend UI

### New Tab: `frontend/src/tabs/ControlTowerTab.tsx`

**Sidebar entry:**
- Section: Supply Chain
- Icon: `Monitor` (from lucide-react)
- Label: "Control Tower"
- Keyboard shortcut: `8`
- Shortcut registered in `useKeyboardShortcuts.ts`

**5-Zone Grid Layout:**

```
┌─────────────────────────────────────────────────────────────┐
│  ZONE 1: KPI Strip — 6 cards across top                     │
│  [Portfolio Health] [Open Exceptions] [Fill Rate 3m]        │
│  [Stockout Days]    [Below SS %]      [Urgent Signals]      │
├────────────────────┬────────────────────────────────────────┤
│  ZONE 2: Health    │  ZONE 3: Exception Queue               │
│  Tier Donut Chart  │  Top 10 open exceptions sorted by      │
│  + ABC×Variability │  severity; inline Acknowledge buttons   │
│  Class Heatmap     │                                        │
├────────────────────┴────────────────────────────────────────┤
│  ZONE 4: Top-10 Critical Items (horizontal scrollable)      │
│  Each card: item, loc, health_score, health_tier badge,     │
│  is_below_SS indicator, current_dos vs target range,        │
│  recommended_order_qty, order_by_date, fill_rate badge,     │
│  stockout_days_this_month                                    │
├─────────────────────────────────────────────────────────────┤
│  ZONE 5: 6-Month Trend — dual Y-axis line chart             │
│  Lines: avg_health_score (left, 0-100), fill_rate (left),   │
│  below_ss_pct (right, %), stockout_day_rate (right, %)      │
│  Reference line: 95% fill rate target (horizontal)         │
└─────────────────────────────────────────────────────────────┘
```

**Zone 1: KPI Strip**

| Card | Value | Threshold |
|---|---|---|
| Portfolio Health | avg_health_score / 100 | Green ≥80, amber 60–79, red <60 |
| Open Exceptions | open_exceptions_total (badge: critical count in red) | Red if critical > 0 |
| Fill Rate (3m) | portfolio_fill_rate_3m % | Green ≥95%, amber 90–95%, red <90% |
| Stockout Days (MTD) | items_with_stockout_this_month | Red if extended_stockouts > 0 |
| Below SS % | below_ss_pct % | Green <5%, amber 5-20%, red >20% |
| Urgent Signals | urgent_demand_signals count | Red if > 0 |

**Zone 2: Health Overview**
- Left: Donut chart — 4 segments (healthy/monitor/at_risk/critical), center shows avg score
- Right: ABC × variability_class heatmap (3×4 cells), avg_health_score color-coded green→red
- Click cell: opens IPfeature6 health detail pre-filtered to that segment
- Auto-refresh: every 5 minutes

**Zone 3: Exception Queue**
- Shows top 10 alerts from `/control-tower/alerts`
- Source badge: "EXCEPTION" (red), "SIGNAL" (amber), "HEALTH" (orange)
- Each row: source badge | item + loc | description | action text | "Acknowledge" button
- Acknowledge button calls `PUT /inv-planning/exceptions/{id}/acknowledge` (auth required)
- "View All" link → navigates to InvPlanningTab Exception Queue panel
- Auto-refresh: every 2 minutes

**Zone 4: Top-10 Critical Items**
- Horizontally scrollable card rail (10 cards)
- Each card (compact):
  ```
  Item: 100320  Loc: 1401-BULK
  Health: 23/100 [CRITICAL]
  SS Coverage: 0.12 ↓ (below SS)
  DOS: 2.1d | Target: 14-28d
  Recommend: Order 480 units by Mar 10
  Fill Rate: 67% | Stockout Days MTD: 12
  ```
- Card border color: red if critical, orange if at_risk
- Click card → navigates to InvPlanningTab with item pre-selected in health detail table

**Zone 5: 6-Month Trend Chart**
- Dual Y-axis line chart (Recharts LineChart with ResponsiveContainer)
- Left Y: avg_health_score (0–100) + fill_rate (0–100%)
- Right Y: below_ss_pct (%) + stockout_day_rate (%)
- 4 lines, each color-coded with legend
- Reference: horizontal dashed line at 95 (fill rate target)
- X-axis: month_start labels (MMM YY format)

**Refresh behavior:**
- Initial load: fetch all 4 endpoints on mount
- Auto-refresh `/control-tower/kpis` every 2 minutes (most volatile)
- Auto-refresh `/control-tower/alerts` every 2 minutes
- Manual refresh: "Refresh Now" button top-right → invalidates all 4 queries

---

## Backend Script

No dedicated script needed — `mv_control_tower_kpis` is a thin aggregation of other views.

**Refresh strategy:** Call `REFRESH MATERIALIZED VIEW CONCURRENTLY mv_control_tower_kpis` at the end of each upstream view refresh:
- After `make health-refresh` → also refresh control tower
- After `make exceptions-generate` → also refresh control tower
- After `make fill-rate-refresh` → also refresh control tower

**Makefile Targets:**
```makefile
control-tower-schema:
	# apply sql/035_create_control_tower_kpis.sql (CREATE MAT VIEW WITH NO DATA)

control-tower-refresh:
	uv run python -c "
import asyncio
from common.db import get_conn
async def refresh():
    conn = await get_conn()
    await conn.execute('REFRESH MATERIALIZED VIEW CONCURRENTLY mv_control_tower_kpis')
asyncio.run(refresh())
"

control-tower-all: control-tower-schema control-tower-refresh
```

---

## Dependencies

| Dependency | Type | Notes |
|---|---|---|
| `mv_inventory_health_score` | IPfeature6 | Health KPIs, top-critical items, ABC×variability heatmap |
| `fact_replenishment_exceptions` | IPfeature7 | Open exceptions, recommended order qty, order_by_date |
| `mv_fill_rate_monthly` | IPfeature8 | Portfolio fill rate 3-month average |
| `fact_demand_signals` | IPfeature9 | Urgent demand alerts, projected stockouts |
| `mv_intramonth_stockout` | IPfeature14 | MTD stockout items count |
| `fact_safety_stock_targets` | IPfeature3 | SS coverage, below_ss, unit_cost for order value |
| `useKeyboardShortcuts.ts` | Existing | Register shortcut `8` for Control Tower tab |
| `AppSidebar.tsx` | Existing | Add Control Tower nav item |
| `App.tsx` | Existing | Lazy-load ControlTowerTab with error boundary |

---

## Testing Requirements

### Backend API Tests: `mvp/demand/tests/api/test_control_tower.py`

Minimum 10 tests:
- `GET /control-tower/kpis` → 200 OK, response has health + exceptions + fill_rate + demand_signals + intramonth keys
- `GET /control-tower/kpis` → avg_health_score between 0 and 100
- `GET /control-tower/kpis` → below_ss_pct between 0 and 1
- `GET /control-tower/alerts` → 200 OK, alerts list, each alert has source + severity + item_no
- Alerts sorted: severity critical before high before medium
- Alert sources: all 3 source types present when data available (exception, demand_signal, health_drop)
- `GET /control-tower/alerts?severity=critical` → all alerts have severity='critical'
- `GET /control-tower/top-critical` → items list, health_score ascending (worst first)
- `GET /control-tower/trend?months=6` → trend list with ≤6 entries
- Empty upstream data → returns zeros, not 500

### Frontend Tests: `frontend/src/tabs/__tests__/ControlTowerTab.test.tsx`

Smoke tests with mocked API:
- Tab renders without crashing
- Zone 1 KPI strip: 6 cards rendered
- Zone 3 Exception Queue: renders with "No open exceptions" when empty
- Zone 4 Critical Items: renders scrollable card rail
- Zone 5 Trend chart: renders with LineChart
- "Refresh Now" button calls queryClient.invalidateQueries

---

## Acceptance Criteria

- [ ] All 4 API endpoints return non-empty data after upstream features populated
- [ ] Alert list correctly merges and sorts across 3 source tables (exceptions, demand_signals, health)
- [ ] Top-10 critical items match bottom of `mv_inventory_health_score` sorted by health_score ASC
- [ ] Control Tower tab accessible via keyboard shortcut `8`
- [ ] `/control-tower` proxy entry in `frontend/vite.config.ts`
- [ ] `mv_control_tower_kpis` refreshes in < 5s (single-row thin aggregation)
- [ ] Auto-refresh every 2 minutes without full page reload (TanStack Query refetchInterval)
- [ ] `make test-all` passes

---

## Files Created / Modified

| File | Action |
|---|---|
| `mvp/demand/sql/035_create_control_tower_kpis.sql` | Create |
| `mvp/demand/api/routers/control_tower.py` | Create |
| `mvp/demand/api/main.py` | Modify — mount control_tower router |
| `mvp/demand/frontend/vite.config.ts` | Modify — add `/control-tower` proxy |
| `mvp/demand/frontend/src/tabs/ControlTowerTab.tsx` | Create |
| `mvp/demand/frontend/src/tabs/__tests__/ControlTowerTab.test.tsx` | Create |
| `mvp/demand/frontend/src/components/AppSidebar.tsx` | Modify — add Control Tower nav item |
| `mvp/demand/frontend/src/hooks/useKeyboardShortcuts.ts` | Modify — add shortcut `8` |
| `mvp/demand/frontend/src/App.tsx` | Modify — lazy-load ControlTowerTab |
| `mvp/demand/tests/api/test_control_tower.py` | Create |
| `mvp/demand/Makefile` | Modify — add control-tower-* targets |
| `docs/design-specs/IPfeature15.md` | Create (this file) |
