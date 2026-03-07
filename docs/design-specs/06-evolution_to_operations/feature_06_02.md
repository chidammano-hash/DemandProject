# F1.2 — Forward Inventory Projection

**Phase:** Evolution to Operations — Phase 1
**Feature Number:** F1.2
**Status:** Not Started
**Priority:** Critical (blocker for F2.1)
**Depends On:** F1.1 (Production Forecast Generation), F1.3 (Open PO Integration)

---

## 1. Problem Statement

Today, Demand Studio can tell a planner: "Item 100320 at LOC 1401-BULK is 15 units below safety stock." This is a snapshot — a single point in time. It cannot answer any of the questions that determine what action to take:

- "How many days until I stock out if I do nothing?"
- "If I receive the 200-unit PO arriving April 5, when does my safety stock buffer expire?"
- "Is this a genuine emergency reorder or will the projected demand soften next month?"
- "If I approve the planned 300-unit order today with 14-day lead time, what does my inventory position look like in May?"

Without answers to these questions, planners default to gut feel or static reorder rules. The result is over-ordering (excess inventory) and under-ordering (stockouts) occurring simultaneously across the portfolio.

### Concrete Failure Scenario

Item 100320. March 6, 2026:
- `qty_on_hand = 120`
- `safety_stock = 60`
- `qty_on_order = 200` (from inventory snapshot field — but **no delivery date**)
- External ERP forecast: April = 400 units → average daily demand = 13.3 units/day

The system currently shows: "Health score: Monitor. Gap to SS: +60 units (above SS)."

The planner assumes she has 4–5 days of cushion above SS. She does not order.

**What the projection would have revealed:**
- The 200-unit `qty_on_order` has a confirmed delivery on April 12 (from the ERP PO — currently not in Demand Studio)
- The ML forecast (F1.1) predicts April demand = 490 units, not 400
- Daily demand = 16.3 units/day
- By April 4 (29 days from today), qty_on_hand drops from 120 to 0: **stockout**
- The inbound PO arrives April 12 — **8 days after stockout begins**

The projection would have flagged this as a CRITICAL risk on March 6, giving 4 weeks to expedite the PO or place an emergency order.

---

## 2. Projection Formula

```
projected_qty[0] = current_qty_on_hand

For each day t = 1, 2, ..., horizon_days:
    receipts[t]  = sum of confirmed receipt qtys due on day t  (from fact_open_purchase_orders)
    demand[t]    = daily_demand_rate  (disaggregated from monthly forecast)
    projected_qty[t] = max(0, projected_qty[t-1] + receipts[t] - demand[t])

    reorder_triggered[t] = (projected_qty[t] <= reorder_point)
    stockout_risk[t]     = (projected_qty[t] <= 0)
    excess_risk[t]       = (projected_qty[t] > max_coverage_qty)

Key dates derived:
    reorder_trigger_date = first t where reorder_triggered[t] = TRUE
    stockout_date        = first t where stockout_risk[t] = TRUE
    excess_liquidation_date = first t where excess_risk[t] = TRUE
    days_of_supply_today = current_qty_on_hand / daily_demand_rate
```

### Daily Demand Disaggregation

Monthly forecast (from `fact_production_forecast`) is disaggregated to daily:

```python
def disaggregate_monthly_to_daily(monthly_forecasts: dict, start_date: date,
                                   horizon_days: int) -> dict[date, float]:
    """
    Spread monthly forecast qty evenly across business days in the month.
    Returns {date: daily_demand_qty} for each day in [start_date, start_date+horizon_days].
    """
    daily = {}
    for month_start, monthly_qty in monthly_forecasts.items():
        days_in_month = calendar.monthrange(month_start.year, month_start.month)[1]
        daily_rate = monthly_qty / days_in_month  # calendar days (no weekend adjustment)
        for d in range(days_in_month):
            day = month_start + timedelta(days=d)
            if start_date <= day < start_date + timedelta(days=horizon_days):
                daily[day] = daily_rate
    return daily
```

---

## 3. Three Projection Scenarios

| Scenario | Description | Receipt Input |
|---|---|---|
| `no_order` | Current on-hand only, no inbound assumed | Zero receipts for all future days |
| `with_open_po` | Current on-hand + confirmed open POs only | Receipts from `fact_open_purchase_orders` where `status IN ('open','partially_received')` |
| `with_planned_orders` | Open POs + system-recommended planned orders | Open POs + approved rows from `fact_planned_orders` |

All three scenarios run in parallel for every DFU projection. The UI shows all three curves on the same chart.

---

## 4. Missing Input Data — Dependencies

### From F1.1 (Production Forecast — Required)

| Data Needed | Source | Notes |
|---|---|---|
| `forecast_qty` for months T+1 through T+6 | `fact_production_forecast` | Must exist before projection runs |
| `plan_version` (latest) | `fact_production_forecast` | Projection always uses latest plan version |

**Without F1.1:** Projection falls back to last 3-month average actuals as a flat daily demand rate. This is explicitly flagged in the API response as `forecast_source: "fallback_avg"`.

### From F1.3 (Open PO Integration — Required for `with_open_po` scenario)

| Data Needed | Source | Notes |
|---|---|---|
| Confirmed delivery dates | `fact_open_purchase_orders.confirmed_delivery_date` | If absent, `with_open_po` scenario = `no_order` scenario |
| Ordered / confirmed qty | `fact_open_purchase_orders.confirmed_qty` | Used as receipt qty on delivery date |

**Without F1.3:** The `with_open_po` scenario silently degrades to `no_order`. The API response includes `open_po_data_available: false` in the metadata.

---

## 5. Data Model

### 5.1 New Table: `fact_inventory_projection`

```sql
CREATE TABLE IF NOT EXISTS fact_inventory_projection (
    id                      BIGSERIAL PRIMARY KEY,
    projection_run_id       UUID            NOT NULL,   -- one run = one DFU + scenario batch
    item_no                 VARCHAR(50)     NOT NULL,
    loc                     VARCHAR(50)     NOT NULL,
    projection_date         DATE            NOT NULL,   -- the specific future date
    scenario                VARCHAR(30)     NOT NULL,   -- 'no_order', 'with_open_po', 'with_planned_orders'
    projected_qty           NUMERIC(12, 2)  NOT NULL,   -- projected on-hand qty
    projected_dos           NUMERIC(8, 2),              -- projected days-of-supply
    forecast_qty_consumed   NUMERIC(12, 2)  NOT NULL,   -- cumulative demand consumed through this date
    receipts_expected       NUMERIC(12, 2)  NOT NULL DEFAULT 0,  -- receipts landing on this date
    reorder_triggered       BOOLEAN         NOT NULL DEFAULT FALSE,
    stockout_risk           BOOLEAN         NOT NULL DEFAULT FALSE,
    excess_risk             BOOLEAN         NOT NULL DEFAULT FALSE,
    daily_demand_rate       NUMERIC(10, 4)  NOT NULL,   -- demand/day used for this date
    forecast_source         VARCHAR(30)     NOT NULL DEFAULT 'production_forecast',  -- or 'fallback_avg'
    plan_version            VARCHAR(30),                -- from fact_production_forecast
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_inv_proj_item_loc_scenario
    ON fact_inventory_projection (item_no, loc, scenario, projection_date);

CREATE INDEX idx_inv_proj_run_id
    ON fact_inventory_projection (projection_run_id);

CREATE INDEX idx_inv_proj_stockout
    ON fact_inventory_projection (item_no, loc, scenario, projection_date)
    WHERE stockout_risk = TRUE;
```

**Grain:** one row per `(projection_run_id, item_no, loc, scenario, projection_date)`
**Retention:** only the most recent run per DFU is kept (older runs deleted on refresh)
**Horizon:** 90 days by default (configurable)

---

### 5.2 Derived Summary View: `mv_inventory_projection_summary`

```sql
CREATE MATERIALIZED VIEW mv_inventory_projection_summary AS
SELECT
    p.item_no,
    p.loc,
    p.scenario,
    p.projection_run_id,
    MIN(CASE WHEN p.reorder_triggered THEN p.projection_date END) AS reorder_trigger_date,
    MIN(CASE WHEN p.stockout_risk     THEN p.projection_date END) AS stockout_date,
    MIN(CASE WHEN p.excess_risk       THEN p.projection_date END) AS excess_date,
    (MIN(CASE WHEN p.stockout_risk THEN p.projection_date END) - CURRENT_DATE)
        AS days_until_stockout,
    MAX(p.created_at) AS last_computed_at
FROM fact_inventory_projection p
GROUP BY p.item_no, p.loc, p.scenario, p.projection_run_id;

CREATE UNIQUE INDEX uq_mv_proj_summary
    ON mv_inventory_projection_summary (item_no, loc, scenario, projection_run_id);
```

---

### 5.3 New Config: `config/projection_config.yaml`

```yaml
projection:
  horizon_days: 90                  # how many days forward to project
  scenarios:
    - no_order
    - with_open_po
    - with_planned_orders
  daily_demand_method: calendar_days # 'calendar_days' or 'business_days'
  fallback_history_months: 3        # months of history to average if no production forecast

thresholds:
  reorder_point_source: safety_stock  # 'safety_stock' or 'reorder_point_config'
  excess_coverage_months: 6          # projected_qty > 6 months coverage = excess_risk

scheduler:
  job_type: compute_inventory_projection
  cron: "0 7 2 * *"                 # Run after production forecast (which runs 06:00)
  horizon_days: 90
```

---

## 6. Python Scripts / Pipeline

### 6.1 `scripts/compute_inventory_projection.py`

```python
"""
compute_inventory_projection.py

Computes day-by-day forward inventory projections for all active DFUs (or a single DFU).
Writes results to fact_inventory_projection and refreshes mv_inventory_projection_summary.

Usage:
    uv run python scripts/compute_inventory_projection.py [--horizon 90] [--dfu ITEM LOC] [--dry-run]

Key functions:
    main()
    get_active_dfus(conn) -> list[tuple[str, str]]
    get_current_inventory(item_no, loc, conn) -> dict
    get_daily_demand_rates(item_no, loc, start_date, horizon_days, conn) -> dict[date, float]
    get_open_po_receipts(item_no, loc, start_date, horizon_days, conn) -> dict[date, float]
    get_safety_stock(item_no, loc, conn) -> float
    run_projection_scenario(current_qty, demand_by_day, receipts_by_day,
                            safety_stock, max_coverage_qty, horizon_days,
                            scenario, forecast_source) -> list[dict]
    write_projection_rows(rows, dry_run, conn) -> int
    refresh_summary_view(conn) -> None
"""

import argparse, yaml, uuid
import pandas as pd
from datetime import date, timedelta, datetime
from common.db import get_db_params
import psycopg

CONFIG_PATH = "config/projection_config.yaml"


def get_current_inventory(item_no: str, loc: str, conn) -> dict:
    """
    Pull the latest snapshot values for this DFU.
    Returns {"qty_on_hand": float, "lead_time_days": int}
    """
    sql = """
        SELECT qty_on_hand, lead_time_days
        FROM fact_inventory_snapshot
        WHERE item_no = %s AND loc = %s
        ORDER BY snapshot_date DESC
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (item_no, loc))
        row = cur.fetchone()
    if not row:
        return {"qty_on_hand": 0.0, "lead_time_days": 14}
    return {"qty_on_hand": float(row[0]), "lead_time_days": int(row[1] or 14)}


def get_daily_demand_rates(item_no: str, loc: str,
                            start_date: date, horizon_days: int,
                            conn) -> dict:
    """
    Pulls production forecast and disaggregates to daily rates.
    Falls back to 3-month average actuals if no production forecast exists.
    Returns {date: daily_qty, ...} and {"source": "production_forecast" | "fallback_avg"}
    """
    import calendar
    from dateutil.relativedelta import relativedelta

    # Try production forecast first
    sql = """
        SELECT forecast_month, forecast_qty
        FROM fact_production_forecast
        WHERE item_no = %s AND loc = %s
          AND plan_version = (
              SELECT plan_version FROM fact_production_forecast
              WHERE item_no = %s AND loc = %s
              ORDER BY generated_at DESC LIMIT 1
          )
        ORDER BY forecast_month
    """
    with conn.cursor() as cur:
        cur.execute(sql, (item_no, loc, item_no, loc))
        rows = cur.fetchall()

    if rows:
        source = "production_forecast"
        monthly = {r[0]: float(r[1]) for r in rows}
    else:
        # Fallback: 3-month average actuals
        source = "fallback_avg"
        sql_fallback = """
            SELECT AVG(qty) / 30.0 AS daily_rate
            FROM fact_sales_monthly
            WHERE dmdunit = %s AND loc = %s AND type = 1
              AND startdate >= CURRENT_DATE - INTERVAL '90 days'
        """
        with conn.cursor() as cur:
            cur.execute(sql_fallback, (item_no, loc))
            row = cur.fetchone()
        daily_rate = float(row[0] or 0.0) if row else 0.0
        return {start_date + timedelta(days=i): daily_rate
                for i in range(horizon_days)}, source

    # Disaggregate monthly to daily
    daily = {}
    for i in range(horizon_days):
        d = start_date + timedelta(days=i)
        month_start = d.replace(day=1)
        days_in_month = calendar.monthrange(d.year, d.month)[1]
        monthly_qty = monthly.get(month_start, 0.0)
        daily[d] = monthly_qty / days_in_month
    return daily, source


def run_projection_scenario(current_qty: float,
                             demand_by_day: dict,
                             receipts_by_day: dict,
                             safety_stock: float,
                             max_coverage_qty: float,
                             horizon_days: int,
                             start_date: date,
                             scenario: str,
                             forecast_source: str) -> list:
    """
    Simulate inventory day by day for a single scenario.
    Returns list of row dicts for insertion into fact_inventory_projection.
    """
    rows = []
    qty = current_qty
    cumulative_demand = 0.0

    for i in range(horizon_days):
        d = start_date + timedelta(days=i)
        daily_demand = demand_by_day.get(d, 0.0)
        daily_receipts = receipts_by_day.get(d, 0.0) if scenario != "no_order" else 0.0

        qty = max(0.0, qty + daily_receipts - daily_demand)
        cumulative_demand += daily_demand
        daily_rate = daily_demand
        dos = qty / daily_rate if daily_rate > 0 else 9999.0

        rows.append({
            "projection_date": d,
            "scenario": scenario,
            "projected_qty": round(qty, 2),
            "projected_dos": round(dos, 2),
            "forecast_qty_consumed": round(cumulative_demand, 2),
            "receipts_expected": round(daily_receipts, 2),
            "reorder_triggered": qty <= safety_stock,
            "stockout_risk": qty <= 0,
            "excess_risk": qty > max_coverage_qty,
            "daily_demand_rate": round(daily_demand, 4),
            "forecast_source": forecast_source,
        })

    return rows
```

### 6.2 Makefile Targets

```makefile
## Inventory Projection
projection-schema:
	uv run python -c "import psycopg; ..."

projection-compute:
	uv run python scripts/compute_inventory_projection.py --horizon 90

projection-compute-dfu:
	uv run python scripts/compute_inventory_projection.py --dfu $(ITEM) $(LOC) --horizon 90

projection-dry:
	uv run python scripts/compute_inventory_projection.py --dry-run --horizon 90

projection-all: projection-schema projection-compute
```

---

## 7. API Endpoints

### 7.1 `GET /inv-planning/projection`

**Query params:**
| Param | Type | Default | Description |
|---|---|---|---|
| `item_no` | string | required | Item number |
| `loc` | string | required | Location code |
| `horizon_days` | int | 90 | Days to project forward |
| `scenario` | string | all | `no_order`, `with_open_po`, `with_planned_orders`, or `all` |

**Response:**
```json
{
  "item_no": "100320",
  "loc": "1401-BULK",
  "current_qty_on_hand": 120.0,
  "safety_stock": 60.0,
  "reorder_point": 60.0,
  "forecast_source": "production_forecast",
  "plan_version": "2026-03",
  "open_po_data_available": true,
  "computed_at": "2026-03-02T07:04:11Z",
  "key_dates": {
    "no_order": {
      "reorder_trigger_date": "2026-03-10",
      "stockout_date": "2026-04-04",
      "days_until_stockout": 29
    },
    "with_open_po": {
      "reorder_trigger_date": "2026-04-22",
      "stockout_date": null,
      "days_until_stockout": null
    }
  },
  "projection": [
    {
      "projection_date": "2026-03-07",
      "no_order_qty": 103.7,
      "with_open_po_qty": 103.7,
      "with_planned_orders_qty": 103.7,
      "daily_demand_rate": 16.3,
      "receipts_expected": 0.0
    },
    {
      "projection_date": "2026-03-08",
      "no_order_qty": 87.4,
      "with_open_po_qty": 87.4,
      "with_planned_orders_qty": 87.4,
      "daily_demand_rate": 16.3,
      "receipts_expected": 0.0
    }
  ]
}
```

### 7.2 `GET /inv-planning/projection/at-risk`

Returns all DFUs with `stockout_date` within N days in the `with_open_po` scenario.

**Query params:** `horizon_days` (default 30), `page`, `page_size`

**Response:**
```json
{
  "total": 47,
  "items": [
    {
      "item_no": "100320",
      "loc": "1401-BULK",
      "current_qty": 120.0,
      "safety_stock": 60.0,
      "stockout_date": "2026-04-04",
      "days_until_stockout": 29,
      "open_po_qty": 200.0,
      "open_po_date": "2026-04-12",
      "severity": "critical"
    }
  ]
}
```

### 7.3 `POST /inv-planning/projection/refresh`

Triggers a synchronous projection recompute for one DFU (for on-demand UI use).

**Body:**
```json
{"item_no": "100320", "loc": "1401-BULK", "horizon_days": 90}
```

**Response:** `{"status": "ok", "rows_written": 270, "run_id": "<uuid>"}`

---

## 8. Frontend Components

### 8.1 New Panel: `ProjectionPanel` in Inv. Planning Tab

**File:** `frontend/src/tabs/inv-planning/ProjectionPanel.tsx`

```
┌───────────────────────────────────────────────────────────────────────────┐
│  INVENTORY PROJECTION                              [Refresh]  [Export CSV] │
│  Item: [100320────────] Loc: [1401-BULK────────]  Horizon: [90 days ▼]   │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ⚠ STOCKOUT RISK: No-order scenario stocks out in 29 days (Apr 4)        │
│  ✓ With confirmed PO (200u on Apr 12): safe until Jun 3                  │
│                                                                           │
│  Qty                                                                      │
│  320 │   ╮                          ← PO receipt bump (with_open_po)     │
│  200 │   │╲   ╭────────────────                                          │
│  120 │───┤ ╲ ╱              ── ── ── safety stock (60)                   │
│   60 │   │  ╲/  ← no_order curve                                         │
│    0 │───┼──────!────────────────────                                     │
│        Mar 6  Apr 4  Apr 12  May 1  Jun 1                                │
│              ↑ stockout   ↑ PO arrives                                   │
│              (no_order)   (with PO: safe)                                │
├───────────────────────────────────────────────────────────────────────────┤
│  KEY DATES               │ NO ORDER      │ WITH OPEN PO  │ WITH PLANNED  │
│  Reorder trigger         │ Mar 10        │ Apr 22        │ May 15        │
│  Stockout date           │ Apr 4 ⚠       │ —             │ —             │
│  Days until stockout     │ 29 days ⚠     │ —             │ —             │
│  Excess threshold date   │ —             │ —             │ Jun 3         │
├───────────────────────────────────────────────────────────────────────────┤
│  DATE        │ QTY (No Order) │ QTY (With PO) │ DEMAND/DAY │ RECEIPTS    │
│  Mar 07      │    103.7       │    103.7      │    16.3    │    0        │
│  Mar 14      │     17.7       │     17.7      │    16.3    │    0        │
│  Mar 21      │      0.0 ⚠     │      0.0 ⚠   │    16.3    │    0        │
│  Apr 05      │      0.0 ⚠     │      0.0 ⚠   │    16.3    │    0        │
│  Apr 12      │      0.0 ⚠     │    183.7      │    16.3    │  200        │
│  Apr 19      │      0.0 ⚠     │     70.0      │    16.3    │    0        │
└───────────────────────────────────────────────────────────────────────────┘
```

**Chart implementation:**
- Recharts `ComposedChart`
- `Line` for each scenario (blue=with_planned, green=with_open_po, red=no_order)
- `ReferenceLine` at y=safety_stock (dashed, labeled "Safety Stock")
- `ReferenceLine` at x=stockout_date (vertical, red, labeled "STOCKOUT")
- `Bar` for receipts (secondary y-axis, green)
- Shaded red zone below y=0

---

## 9. Worked Example: Item 100320, LOC 1401-BULK

**As of March 6, 2026:**

| Parameter | Value | Source |
|---|---|---|
| `qty_on_hand` | 120 units | Latest `fact_inventory_snapshot` |
| `safety_stock` | 60 units | `fact_safety_stock_targets` |
| `reorder_point` | 60 units | = safety_stock (simplified) |
| Monthly forecast (Apr 2026) | 490 units | `fact_production_forecast` (F1.1) |
| Daily demand rate | 16.3 units/day | 490 / 30 = 16.333 |
| Open PO | 200 units arriving Apr 12 | `fact_open_purchase_orders` (F1.3) |
| Max coverage | 300 units | 6 months × 490 / 6 = 490 (use 300 for simplicity) |

### Day-by-Day Table (first 45 days)

| Date | Day# | Demand/Day | Receipts (PO) | No-Order Qty | With-PO Qty | Reorder? | Stockout? |
|---|---|---|---|---|---|---|---|
| Mar 06 | 0 | — | — | 120.0 | 120.0 | No | No |
| Mar 07 | 1 | 16.3 | 0 | 103.7 | 103.7 | No | No |
| Mar 10 | 4 | 16.3 | 0 | **59.8** | 59.8 | **YES** | No |
| Mar 14 | 8 | 16.3 | 0 | 9.6 | 9.6 | Yes | No |
| Mar 18 | 12 | 16.3 | 0 | 0 *(clamped)* | 0 | Yes | **YES** |
| Apr 04 | 29 | 16.3 | 0 | 0.0 | 0.0 | Yes | Yes |
| Apr 12 | 37 | 16.3 | **200** | 0.0 | **196.4** | Yes → No | Yes → **No** |
| Apr 22 | 47 | 16.3 | 0 | 0.0 | **59.5** | No | No |
| May 01 | 56 | 16.0* | 0 | 0.0 | 59.5 | No | No |
| Jun 03 | 89 | 16.0* | 0 | 0.0 | **0.1** | Yes | No |

*May/Jun daily rate based on forecast 480/30=16.0*

### Key Date Summary

| Scenario | Reorder Trigger | Stockout Date | Days Until Stockout |
|---|---|---|---|
| No Order | Mar 10 (day 4) | Mar 18 (day 12) | **12 days** |
| With Open PO | Mar 10 (day 4) | Not reached | — |
| With Planned | Mar 10 (day 4) | Not reached | — |

**Planner action implied:** Place emergency order OR expedite the Apr 12 PO to arrive by Mar 10. The system would flag this as CRITICAL in the Exception Queue (IPfeature7) with severity=critical.

---

## 10. Edge Cases

### 10.1 Zero Demand Items
- `daily_demand_rate = 0` for all days
- Projection stays flat at `current_qty` (no stockout, no reorder trigger)
- `projected_dos = 9999` (capped sentinel value)

### 10.2 Intermittent Demand Items
- Monthly forecast may be 0 for some months and 40+ for others
- Daily disaggregation still works: months with forecast=0 contribute 0 demand
- Reorder trigger fires when cumulative demand approaches on-hand

### 10.3 Multiple Open POs
- `receipts_by_day` is a dict: `{date: sum_of_all_po_qtys_arriving_that_day}`
- Multiple POs arriving same day are summed before simulation
- Split deliveries (partial receipt on 2 dates) are two separate entries in `fact_open_purchase_orders` after receipt posting

### 10.4 No Production Forecast Available
- Falls back to 3-month trailing average
- `forecast_source = "fallback_avg"` in response
- UI shows warning banner: "Production forecast not available. Using 3-month average."

---

## 11. Dependencies

| Dependency | Status | Notes |
|---|---|---|
| F1.1 Production Forecast | Required | `fact_production_forecast` must exist; graceful fallback if missing |
| F1.3 Open PO Integration | Required for `with_open_po` scenario | Degrades gracefully if table empty |
| `fact_safety_stock_targets` | Exists (IPfeature3) | SS value used as reorder_point |
| `fact_inventory_snapshot` | Exists | `qty_on_hand` as starting position |
| F2.1 Order Recommendation | Downstream | Consumes projection to determine when to order |

---

## 12. Out of Scope

- Lot sizing optimization (handled in F2.1)
- Multi-echelon projection (warehouse → store cascade)
- Probabilistic projection (Monte Carlo demand simulation — separate feature)
- Real-time intraday inventory updates (snapshot-based only)
- Customer order backlog as demand signal (requires SO data integration)

---

## 13. Test Requirements

### Backend Unit Tests (`tests/unit/test_inventory_projection.py`)
- `test_run_projection_no_order_depletes` — qty monotonically decreases to 0 with no receipts
- `test_run_projection_po_bumps_qty` — receipt on day N increases projected_qty
- `test_run_projection_qty_never_negative` — clamping to 0 works correctly
- `test_stockout_date_correct` — `stockout_date` = first day projected_qty <= 0
- `test_reorder_trigger_correct` — `reorder_trigger_date` = first day qty <= safety_stock
- `test_excess_flag_correct` — `excess_risk = TRUE` when projected_qty > max_coverage_qty
- `test_disaggregate_monthly_to_daily_even_split` — 490 units / 30 days = 16.333
- `test_disaggregate_fallback_avg` — fallback returns flat daily rate when no forecast
- `test_zero_demand_item_stable_qty` — qty unchanged when daily_demand_rate = 0
- `test_multiple_pos_same_day_summed` — two POs on same day are additive

### Backend API Tests (`tests/api/test_inv_planning_projection.py`)
- `test_get_projection_success_all_scenarios` — 200, returns 3 scenario curves
- `test_get_projection_stockout_in_key_dates` — stockout_date present in response
- `test_get_projection_no_po_data` — `open_po_data_available: false` in response
- `test_get_projection_at_risk_list` — filters correctly by horizon_days
- `test_post_projection_refresh` — 200, rows_written > 0

### Frontend Tests (`src/tabs/__tests__/InvPlanningTab.test.tsx`)
- `test_projection_panel_renders`
- `test_projection_panel_shows_stockout_warning`
- `test_projection_panel_shows_key_dates_table`
