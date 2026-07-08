# Control Tower

> A unified operational dashboard that aggregates KPIs (Key Performance Indicators) from across all supply chain domains -- inventory health, forecast accuracy, fill rate, demand signals, and stockout events -- into a single command-center view with active alerts and top-critical items.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | CommandCenterTab (see UI Integration note below) |
| **Key Files** | `CommandCenterTab.tsx`, `api/routers/operations/control_tower.py`, `sql/035_create_control_tower_kpis.sql` |

---

## Problem

Supply chain managers juggle multiple dashboards: one for inventory, one for forecast accuracy, one for fill rate, one for exceptions. Switching between tabs to assess overall health is slow and error-prone. When a problem spans multiple domains (e.g., forecast bias causes excess inventory which triggers a policy exception), no single view connects the dots. Managers need a one-screen summary that surfaces the most urgent issues across all domains.

---

## UI Integration

The dedicated `ControlTowerTab` screen was retired and consolidated into `CommandCenterTab` (U3.10). `CommandCenterTab`
merges Control Tower KPIs with AI Planner insights and Storyboard exceptions into one unified triage feed
(`frontend/src/tabs/CommandCenterTab.tsx`, fed by `fetchControlTowerKpis`/`fetchControlTowerTrend` from
`src/api/queries`). The old `?tab=controlTower` URL key still resolves to `commandCenter` via `TAB_REDIRECTS` in
`useUrlState.ts`, so existing bookmarks keep working. A regression test
(`frontend/src/tabs/__tests__/no-retired-tabs.test.ts`) fails the build if `ControlTowerTab.tsx` is ever
reintroduced under `src/tabs`. The data shape documented below (KPI zones, alerts, top-critical items) is
unchanged by the consolidation -- only the hosting tab moved.

---

## Solution

A single materialized view (`mv_control_tower_kpis`) joins data from inventory, forecast, fill rate, demand signal, and stockout materialized views to produce a unified KPI row. The frontend renders this as a five-zone grid: KPI cards across the top, an active alert list on the left, top-critical items on the right, and a trend chart below. Alerts are generated from threshold breaches across any domain.

---

## How It Works

### KPI Aggregation

The materialized view computes cross-dimensional KPIs in a single SQL query:

| Zone | KPIs | Source View |
|---|---|---|
| Health | Avg DOS, % items below safety stock, avg health score | `mv_inventory_health_score` |
| Exceptions | Open exception count, critical count, unacknowledged count | `fact_replenishment_exceptions` |
| Fill Rate | Avg fill rate %, items below 95% fill rate | `mv_fill_rate_monthly` |
| Demand Signals | Signal count, avg signal strength, anomaly count | `fact_demand_signals` |
| Intramonth | Stockout event count, affected items, affected locations | `mv_intramonth_stockout` |

### Alert Generation

Alerts are dynamically generated from threshold breaches, not stored in a separate table:

| Alert Condition | Severity | Source |
|---|---|---|
| Fill rate below 90% for any item-location | Critical | `mv_fill_rate_monthly` |
| DOS below lead time for 50+ DFUs | High | `mv_inventory_health_score` |
| WAPE above 40% for any model | Medium | `agg_forecast_monthly` |
| Intramonth stockout count rising week-over-week | High | `mv_intramonth_stockout` |
| Open critical exceptions above 10 | Critical | `fact_replenishment_exceptions` |

### Top-Critical Items

The top-critical list ranks items by a composite urgency score combining: financial impact (inventory value at risk), service level impact (fill rate deviation), and forecast risk (WAPE trend direction). The top 20 items are returned with their primary risk factor.

---

## Data Model

| Object | Type | Purpose |
|---|---|---|
| `mv_control_tower_kpis` | Materialized view | Single-row aggregate KPIs across all domains |

The view joins 5 existing source tables/views. No new fact or dimension tables are created.

### Refresh

`REFRESH MATERIALIZED VIEW CONCURRENTLY mv_control_tower_kpis` -- can be triggered manually or scheduled via the job scheduler.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/control-tower/kpis` | All KPI zones in a single response |
| GET | `/control-tower/alerts` | Active alerts with severity and source |
| GET | `/control-tower/top-critical` | Top-20 items ranked by composite urgency |
| GET | `/control-tower/trend` | Monthly KPI trend for charting |

All endpoints are read-only. No authentication required for reads.

---

## Pipeline

| Step | Command | What It Does |
|---|---|---|
| Schema | `make control-tower-schema` | Creates `mv_control_tower_kpis` materialized view |
| Refresh | `make control-tower-refresh` | Refreshes the materialized view with current data |
| Full | `make control-tower-all` | Schema + refresh |

---

## Dependencies

| Dependency | Reason |
|---|---|
| Inventory health (03-04) | `mv_inventory_health_score` for DOS and health KPIs |
| Fill rate (03-06) | `mv_fill_rate_monthly` for fill rate zone |
| Demand signals (03-06) | `fact_demand_signals` for signal zone |
| Intramonth stockouts (03-06) | `mv_intramonth_stockout` for stockout zone |
| Exception queue (03-05) | `fact_replenishment_exceptions` for exception zone |

All source views must be refreshed before the control tower view is refreshed.

---

## Financial & lifecycle KPIs

- **$-denominated KPIs** — `GET /control-tower/kpis-financial` (inventory value, below-SS gap, excess, exception exposure).
- **Working-capital analytics** — `GET /analytics/working-capital` (cash-to-cash, turns, DIO/DPO/DSO; `api/routers/inventory/working_capital.py`).
- **Exception lifecycle + SLA** — `fact_exception_lifecycle` (append-only transitions + MTTR views) and SLA / root-cause grouping in `common/engines/exception_engine.py` (`config/operations/exception_sla.yaml`).

## See Also

- `06-ai-platform/01-ai-planning-agent.md` -- AI agent queries control tower KPIs via `get_portfolio_health_summary` tool
- `06-ai-platform/04-storyboard.md` -- exception details behind the exception count KPI
- `03-inventory-planning/06-analytics.md` -- fill rate, demand signals, and intramonth stockout source data
