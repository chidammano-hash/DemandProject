# Fill Rate, Demand Signals & Intramonth Stockouts

> Three materialized views and a computation script that provide historical fill rate analytics, short-horizon demand signals from intra-month sales patterns, and detection of mid-month stockout events invisible in end-of-month snapshots.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inventory Planning |
| **Key Files** | `scripts/inventory/compute_demand_signals.py`, `api/routers/operations/fill_rate.py`, `sql/028_create_fill_rate_monthly.sql`, `sql/034_create_intramonth_stockout.sql` |

---

## Problem

Three monitoring gaps: (1) planners lack a historical fill rate (percentage of demand fulfilled from stock) trend to measure service performance over time; (2) short-horizon demand signals from recent sales velocity are not captured for near-term planning; (3) stockouts that occur mid-month and recover by month-end are invisible in end-of-month snapshots.

---

## Solution

Three materialized views and a computation script that address each gap: fill rate analytics from inventory-to-sales ratios, demand signal extraction from intra-month sales patterns, and within-month stockout detection from daily snapshot sequences.

---

## How It Works

### Fill Rate Analytics (IPfeature8)

`mv_fill_rate_monthly` aggregates item-location-month fill rates from inventory snapshot data:

| Metric | Formula |
|---|---|
| Fill rate % | MIN(1.0, qty_shipped / qty_ordered) per order line, averaged |
| Service level | % of order lines with fill rate = 100% |
| Backorder rate | 1 - fill rate |

Supports trending over time and slicing by location, category, ABC class.

### Demand Signals (IPfeature9)

`scripts/inventory/compute_demand_signals.py` extracts short-horizon signals:

| Signal | Derivation | Purpose |
|---|---|---|
| Sales velocity | Recent daily sales rate vs trailing average | Acceleration/deceleration |
| Inventory movement | Rate of stock depletion | Days-to-stockout projection |
| Intra-month projection | Current MTD extrapolated to full month | Early warning vs forecast |

Written to `fact_demand_signals` at grain: item_id + loc + signal_date. Consumed by blended demand and exception detection.

### Intramonth Stockout Detection (IPfeature14)

`mv_intramonth_stockout` identifies stockout events that occur *during* a month but may not appear in end-of-month snapshots:

| Detection Logic | Detail |
|---|---|
| Daily on-hand check | Any day with qty_on_hand <= 0 within a month |
| Duration | Count of consecutive zero-stock days |
| Recovery | Did stock recover before month-end? |
| Daily sales | Derived via LAG() on cumulative MTD (same method as `agg_inventory_monthly`) |
| Lost sales estimate | Stockout days * avg daily sales |

This catches "hidden" stockouts where a replenishment arrives before the EOM snapshot, masking the event in monthly aggregates.

---

## Data Model

| Table / View | Grain | Row Count |
|---|---|---|
| `mv_fill_rate_monthly` | item_id + loc + month | ~7.2M rows |
| `fact_demand_signals` | item_id + loc + signal_date | ~483K rows |
| `mv_intramonth_stockout` | item_id + loc + month + event | ~8.2M rows |

DDL: `sql/028_create_fill_rate_monthly.sql`, `sql/029_create_demand_signals.sql`, `sql/034_create_intramonth_stockout.sql`

---

## API

Fill rate:

| Method | Path | Purpose |
|---|---|---|
| GET | `/fill-rate/summary` | Portfolio fill rate KPIs, by-ABC breakdown, worst items, trend |
| GET | `/fill-rate/trend` | Monthly fill rate trend |
| GET | `/fill-rate/detail` | Per-DFU fill rate detail, paginated |
| GET | `/fill-rate/gap-analysis` | Fill rate gap decomposition by cause (SS shortfall, demand spike, lead time delay) |
| GET | `/inv-planning/service-level/waterfall` | Service-level target-to-actual bridge (waterfall) chart data |

Demand signals:

| Method | Path | Purpose |
|---|---|---|
| GET | `/inv-planning/demand-signals/summary` | Signal distribution overview for a given date |
| GET | `/inv-planning/demand-signals` | Paginated demand signals list, filterable and sortable |
| GET | `/inv-planning/demand-signals/item` | Single item-location signal with daily MTD series |

Intramonth stockouts:

| Method | Path | Purpose |
|---|---|---|
| GET | `/inv-planning/intramonth-stockouts/summary` | Hidden stockout frequency, portfolio-level |
| GET | `/inv-planning/intramonth-stockouts/detail` | Per-DFU stockout events, paginated |
| GET | `/inv-planning/intramonth-stockouts/daily` | Daily on-hand/sales drill-down for one item-location |

Routers: `api/routers/operations/fill_rate.py`, `api/routers/inventory/inv_planning_demand_signals.py`, `api/routers/inventory/inv_planning_intramonth.py`

Note: `/inv-planning/service-level/waterfall` is defined in `fill_rate.py` despite its
`/inv-planning` path prefix -- it shares the fill-rate materialized view and service-level
target logic, so it was kept in the same router rather than split out.

---

## Pipeline

```
make fill-rate-all          # fill-rate-schema + fill-rate-refresh
make demand-signals-all     # demand-signals-schema + demand-signals-compute
make intramonth-all         # intramonth-schema + intramonth-refresh
```

---

## Configuration

Demand signals use thresholds from the exception config for velocity change alerts. Fill rate and intramonth stockouts are materialized views with no external configuration.

---

## Dependencies

- **Upstream:** `fact_inventory_snapshot`, `agg_inventory_monthly`, `fact_sales_monthly`
- **Downstream:** Blended demand (sensing signals), exception queue (velocity alerts), control tower (fill rate KPIs)

---

## See Also

- [01-inventory-snapshot](01-inventory-snapshot.md) -- Source snapshot data
- [../03-demand-intelligence/03-blended-demand](../03-demand-intelligence/03-blended-demand.md) -- Consumes demand signals
- [05-exception-queue](05-exception-queue.md) -- Velocity changes trigger exceptions
