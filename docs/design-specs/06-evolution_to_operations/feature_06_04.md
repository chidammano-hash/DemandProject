# F2.1 — Order Recommendation Engine

**Phase:** Evolution to Operations — Phase 2
**Feature Number:** F2.1
**Status:** Not Started
**Priority:** High
**Depends On:** F1.1 (Production Forecast), F1.2 (Inventory Projection), F1.3 (Open PO Integration)

---

## 1. Problem Statement

The current system generates replenishment exceptions with a static heuristic:

```python
# From scripts/generate_replenishment_exceptions.py (current)
recommended_order_qty = gap + EOQ / 2
```

Where `gap = safety_stock - current_qty_on_hand`.

This formula has four fundamental problems:

### Problem A: Ignores Future Demand Trajectory
The formula treats demand as flat (the current snapshot). If April demand is projected at 490 units and May at 510 units, the system still recommends based on the current SS gap. It does not account for the fact that demand is accelerating and a single order may only last 3 weeks instead of the expected 6.

**Example:** Item 100320. `qty_on_hand=120`, `SS=60`, `EOQ=300`.
- Current heuristic: `gap = 60 - 120 = -60` → gap is negative → no exception generated (item appears safe)
- True situation: ML forecast says April demand = 490, daily rate = 16.3. Item stocks out in 3.7 days.
- **The current system does not generate an order recommendation. The planner orders nothing. Stockout occurs.**

### Problem B: Ignores Already-Confirmed Inbound Supply
The heuristic does not subtract `qty_on_order` from the recommendation. If 200 units are already ordered and arriving next week, the system recommends ordering another 200 — creating 4 months of excess.

### Problem C: Ignores Review Cycle Timing
Replenishment policies have a review cycle (daily, weekly, monthly). The recommendation should only be triggered when the next review date has been reached and the projected position at that point falls below the reorder point.

### Problem D: Ignores Minimum Order Quantities / Price Breaks
The formula produces continuous quantities (e.g., 127.3 units). MOQ rounding, pallet quantities, and price break thresholds are ignored, causing infeasible purchase orders.

---

## 2. Net Requirements Calculation

The correct approach replaces the static heuristic with a time-phased net requirements calculation derived from the material requirements planning (MRP) logic.

### 2.1 Core Formula

```
projected_position[t] = current_qty_on_hand
                      + SUM(confirmed_receipts[now : t])     ← from fact_open_purchase_orders
                      - SUM(forecast_demand[now : t])         ← from fact_production_forecast

trigger_date = first t where projected_position[t] ≤ reorder_point

# At trigger_date, compute the deficit covering the lead time window:
lt_demand = SUM(forecast_demand[trigger_date : trigger_date + lead_time_days])
net_requirement = safety_stock + lt_demand - projected_position[trigger_date]
net_requirement = max(0, net_requirement)

# Round up to MOQ
order_qty = max(MOQ, ceil(net_requirement / MOQ) * MOQ)

# Schedule the order
order_by_date = trigger_date - 0 days   # = today if trigger_date is in past
expected_receipt_date = order_by_date + lead_time_days

# Order value
order_value = order_qty × unit_cost
```

### 2.2 Timeline Diagram

```
TIME AXIS ─────────────────────────────────────────────────────────────────────►

         TODAY        ORDER BY   RECEIPT     COVERAGE WINDOW
           │          DATE       DATE        │
           ▼          ▼          ▼           ▼
Qty:  120 ──┐
            │ demand consumed (16.3/day)
         60 ┤── reorder_point (= SS)
            │
          0 ┤────────────────────┐           ┌────────── +300 units (order receipt)
            │    net deficit     │           │
            │  during LT window │           │ projected with order
            │                   └───────────┘
            │
           TRIGGER              RECEIPT
           DATE                 DATE
           (Mar 10)             (Mar 24, LT=14d)

ORDER BY DATE = TRIGGER DATE - (days to place order, typically 0 for immediate)
                              ^ if trigger_date is already past → flag as PAST DUE order
```

### 2.3 MOQ Rounding Example

```
net_requirement = 220 units
MOQ = 100 units

naive:  220
        220 / 100 = 2.2 → ceil(2.2) = 3
        order_qty = 3 × 100 = 300 units
```

If there is a price break at 250 units (`cost = $10.00` vs `$12.50` below 250), the system should also evaluate ordering 250 units instead of 300 (lower cost, still meets requirement). This price-break optimization is a stretch goal for MVP; base implementation uses MOQ rounding only.

---

## 3. Planned Order Approval Workflow

```
STATES:   proposed → approved → released → [received] → closed
                   ↘ rejected
                   ↘ cancelled (after approved/released)

proposed:   System generates; awaits planner review
approved:   Planner confirms; ready to send to ERP
released:   Transmitted to ERP / supplier (PO created externally)
closed:     Corresponding open PO receipt confirmed
rejected:   Planner dismissed the recommendation
cancelled:  Planner cancels after approval but before release
```

State transitions:
- `proposed → approved`: `PUT /supply/planned-orders/{id}/approve`
- `proposed → rejected`: `PUT /supply/planned-orders/{id}/reject`
- `approved → released`: `PUT /supply/planned-orders/{id}/release`
- `approved/released → cancelled`: `PUT /supply/planned-orders/{id}/cancel`
- `released → closed`: Automatic when a matching receipt arrives in `fact_po_receipts` (future automation)

---

## 4. Data Model

### 4.1 New Table: `fact_planned_orders`

```sql
CREATE TABLE IF NOT EXISTS fact_planned_orders (
    id                      BIGSERIAL PRIMARY KEY,
    item_no                 VARCHAR(50)     NOT NULL,
    loc                     VARCHAR(50)     NOT NULL,
    supplier_id             VARCHAR(50)     REFERENCES dim_supplier(supplier_id),
    policy_id               INTEGER         REFERENCES dim_replenishment_policy(id),

    -- Quantities
    net_requirement_qty     NUMERIC(12, 2)  NOT NULL,       -- raw net requirement before rounding
    recommended_qty         NUMERIC(12, 2)  NOT NULL,       -- after MOQ rounding
    moq                     NUMERIC(12, 2)  NOT NULL DEFAULT 1,
    unit_cost               NUMERIC(12, 4),
    order_value             NUMERIC(14, 2)  GENERATED ALWAYS AS
                                (recommended_qty * COALESCE(unit_cost, 0)) STORED,
    currency                CHAR(3)         NOT NULL DEFAULT 'USD',

    -- Timing
    trigger_date            DATE            NOT NULL,        -- when proj_position <= reorder_point
    trigger_reason          VARCHAR(50)     NOT NULL,        -- 'projected_below_ss' / 'ss_gap_today' / etc.
    order_by_date           DATE            NOT NULL,        -- when order must be placed
    expected_receipt_date   DATE            NOT NULL,        -- order_by_date + lead_time_days
    lead_time_days          INTEGER         NOT NULL,
    review_cycle_days       INTEGER,                         -- from replenishment policy
    is_past_due             BOOLEAN GENERATED ALWAYS AS (order_by_date < CURRENT_DATE) STORED,

    -- Demand inputs used in calculation
    current_qty_on_hand     NUMERIC(12, 2)  NOT NULL,
    safety_stock            NUMERIC(12, 2)  NOT NULL,
    reorder_point           NUMERIC(12, 2)  NOT NULL,
    confirmed_inbound_qty   NUMERIC(12, 2)  NOT NULL DEFAULT 0,  -- sum of open PO open_qty before trigger_date
    lt_forecast_demand      NUMERIC(12, 2)  NOT NULL,            -- demand during lead time window
    plan_version            VARCHAR(30),                          -- which production forecast was used

    -- Confidence
    confidence_score        NUMERIC(4, 3),                       -- 0.000 to 1.000
    confidence_reason       TEXT,

    -- Workflow
    status                  VARCHAR(20)     NOT NULL DEFAULT 'proposed',
                            -- proposed / approved / released / rejected / cancelled / closed
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    approved_by             VARCHAR(100),
    approved_at             TIMESTAMPTZ,
    released_at             TIMESTAMPTZ,
    cancelled_at            TIMESTAMPTZ,
    rejection_reason        TEXT,
    run_id                  UUID            NOT NULL            -- ties to a generation run
);

CREATE INDEX idx_planned_orders_item_loc
    ON fact_planned_orders (item_no, loc, status);

CREATE INDEX idx_planned_orders_order_by_date
    ON fact_planned_orders (order_by_date, status)
    WHERE status IN ('proposed', 'approved');

CREATE INDEX idx_planned_orders_past_due
    ON fact_planned_orders (item_no, loc, order_by_date)
    WHERE is_past_due AND status IN ('proposed', 'approved');

CREATE INDEX idx_planned_orders_status_created
    ON fact_planned_orders (status, created_at DESC);
```

**Grain:** one row per recommended order event (one DFU may have multiple if horizon covers multiple reorder cycles)

---

### 4.2 New Config: `config/order_recommendation_config.yaml`

```yaml
recommendation:
  horizon_days: 90                  # how far forward to look for trigger dates
  max_orders_per_dfu: 3            # generate at most N planned orders per DFU per run
  include_past_due: true            # generate past-due orders (order_by_date < today)

  # Confidence scoring rules
  confidence:
    high_threshold: 0.80           # ≥ this = High confidence
    low_threshold: 0.50            # < this = Low confidence (show warning)
    # Factors that reduce confidence:
    penalty_no_open_po_data: 0.15  # open PO data unavailable
    penalty_fallback_forecast: 0.20 # using fallback avg instead of production forecast
    penalty_past_due_order: 0.10   # order_by_date is in the past

moq_handling:
  rounding_strategy: ceil_to_moq   # 'ceil_to_moq' | 'nearest_moq' | 'floor_to_moq'
  # Price break evaluation (future feature):
  evaluate_price_breaks: false

budget_cap:
  enabled: false                   # true = enforce portfolio budget cap
  monthly_budget_usd: 500000.0     # max order value per month across all DFUs
  priority_by: abc_class           # A-class items get budget first

scheduler:
  job_type: generate_planned_orders
  cron: "0 8 2 * *"               # Run after projection (which runs 07:00)
```

---

## 5. Python Script: `scripts/generate_planned_orders.py`

```python
"""
generate_planned_orders.py

Generates time-phased planned order recommendations for all active DFUs (or a single DFU).
Reads: fact_inventory_projection, fact_open_purchase_orders, fact_production_forecast,
       fact_safety_stock_targets, dim_replenishment_policy, dim_item_supplier.
Writes: fact_planned_orders.

Usage:
    uv run python scripts/generate_planned_orders.py [--dfu ITEM LOC] [--dry-run]

Key functions:
    main()
    get_active_dfus_for_recommendation(conn) -> list[tuple[str, str]]
    get_dfu_inputs(item_no, loc, conn) -> dict
    compute_net_requirements(inputs: dict, config: dict) -> list[dict]
    round_to_moq(qty: float, moq: float, strategy: str) -> float
    compute_confidence_score(inputs: dict, config: dict) -> tuple[float, str]
    write_planned_orders(orders: list[dict], dry_run: bool, conn) -> int
"""

import argparse, yaml, uuid
import pandas as pd
import math
from datetime import date, timedelta
from common.db import get_db_params
import psycopg

CONFIG_PATH = "config/order_recommendation_config.yaml"


def round_to_moq(qty: float, moq: float, strategy: str = "ceil_to_moq") -> float:
    """
    Round a net requirement quantity up to the nearest MOQ multiple.

    Examples:
        round_to_moq(220, 100, 'ceil_to_moq') → 300
        round_to_moq(200, 100, 'ceil_to_moq') → 200  (exact multiple, no change)
        round_to_moq(  1, 100, 'ceil_to_moq') → 100  (min = 1 MOQ)
    """
    if moq <= 0:
        moq = 1.0
    if strategy == "ceil_to_moq":
        return max(moq, math.ceil(qty / moq) * moq)
    elif strategy == "nearest_moq":
        return max(moq, round(qty / moq) * moq)
    else:
        return max(moq, math.floor(qty / moq) * moq)


def get_dfu_inputs(item_no: str, loc: str, conn) -> dict:
    """
    Assembles all inputs needed for net requirement calculation for one DFU.
    Returns a dict with keys:
        current_qty_on_hand, safety_stock, reorder_point, lead_time_days,
        moq, unit_cost, policy_id, review_cycle_days, supplier_id,
        daily_demand_by_date, confirmed_receipts_by_date,
        plan_version, forecast_source
    """
    # 1. Current inventory position
    inv_sql = """
        SELECT qty_on_hand, lead_time_days
        FROM fact_inventory_snapshot
        WHERE item_no = %s AND loc = %s
        ORDER BY snapshot_date DESC LIMIT 1
    """
    # 2. Safety stock
    ss_sql = """
        SELECT ss_combined, reorder_point
        FROM fact_safety_stock_targets
        WHERE item_no = %s AND loc = %s
        ORDER BY computed_at DESC LIMIT 1
    """
    # 3. Policy (for review_cycle_days and moq)
    policy_sql = """
        SELECT p.id, p.review_cycle_days, p.moq, s.lead_time_days AS supplier_lt,
               s.price_per_unit, s.supplier_id
        FROM fact_dfu_policy_assignment pa
        JOIN dim_replenishment_policy p ON p.id = pa.policy_id
        LEFT JOIN dim_item_supplier s ON s.item_no = pa.item_no AND s.loc = pa.loc
                                        AND s.is_preferred = TRUE
        WHERE pa.item_no = %s AND pa.loc = %s
        LIMIT 1
    """
    # 4. Daily demand rates (from fact_inventory_projection, no_order scenario)
    demand_sql = """
        SELECT projection_date, daily_demand_rate, forecast_source
        FROM fact_inventory_projection
        WHERE item_no = %s AND loc = %s AND scenario = 'no_order'
        ORDER BY projection_date
    """
    # 5. Confirmed receipts from open POs
    receipts_sql = """
        SELECT effective_delivery_date, SUM(open_qty) AS expected_qty
        FROM fact_open_purchase_orders
        WHERE item_no = %s AND loc = %s
          AND line_status NOT IN ('closed', 'cancelled')
          AND effective_delivery_date >= CURRENT_DATE
        GROUP BY effective_delivery_date
    """
    # ... execute all queries and assemble dict
    pass


def compute_net_requirements(inputs: dict, config: dict) -> list:
    """
    Runs the time-phased net requirements calculation.
    Returns list of planned order dicts, one per reorder cycle within the horizon.

    Algorithm:
    1. Simulate projected position day-by-day (receipts - demand), using open POs as receipts
    2. When projected_position[t] <= reorder_point, that is trigger_date
    3. Compute lt_demand = sum of daily demand over [trigger_date, trigger_date + LT]
    4. Compute net_requirement = SS + lt_demand - projected_position[trigger_date]
    5. Round to MOQ
    6. Add the planned order receipt to the simulation (projected receipt on trigger_date + LT)
    7. Continue simulation; find next trigger within horizon (for multi-cycle ordering)
    """
    horizon_days = config["recommendation"]["horizon_days"]
    max_orders = config["recommendation"]["max_orders_per_dfu"]

    qty = inputs["current_qty_on_hand"]
    ss = inputs["safety_stock"]
    rp = inputs["reorder_point"]
    lt = inputs["lead_time_days"]
    moq = inputs["moq"]
    moq_strategy = config["moq_handling"]["rounding_strategy"]
    unit_cost = inputs.get("unit_cost", 0.0)

    planned_receipts = dict(inputs["confirmed_receipts_by_date"])  # copy, will add planned orders
    demand_by_day = inputs["daily_demand_by_date"]

    orders = []
    today = date.today()

    for i in range(horizon_days):
        d = today + timedelta(days=i)
        daily_receipts = planned_receipts.get(d, 0.0)
        daily_demand = demand_by_day.get(d, 0.0)
        qty = max(0.0, qty + daily_receipts - daily_demand)

        if qty <= rp and len(orders) < max_orders:
            # Triggered: compute net requirement
            trigger_date = d
            order_by_date = trigger_date  # immediate (no placement lead time in MVP)

            # Sum forecast demand during lead time window
            lt_demand = sum(
                demand_by_day.get(trigger_date + timedelta(days=j), 0.0)
                for j in range(lt)
            )

            net_req = max(0.0, ss + lt_demand - qty)
            order_qty = round_to_moq(net_req, moq, moq_strategy)

            receipt_date = order_by_date + timedelta(days=lt)

            orders.append({
                "item_no": inputs["item_no"],
                "loc": inputs["loc"],
                "supplier_id": inputs.get("supplier_id"),
                "policy_id": inputs.get("policy_id"),
                "net_requirement_qty": round(net_req, 2),
                "recommended_qty": order_qty,
                "moq": moq,
                "unit_cost": unit_cost,
                "trigger_date": trigger_date,
                "trigger_reason": "projected_below_ss",
                "order_by_date": order_by_date,
                "expected_receipt_date": receipt_date,
                "lead_time_days": lt,
                "current_qty_on_hand": inputs["current_qty_on_hand"],
                "safety_stock": ss,
                "reorder_point": rp,
                "confirmed_inbound_qty": sum(inputs["confirmed_receipts_by_date"].values()),
                "lt_forecast_demand": round(lt_demand, 2),
                "plan_version": inputs.get("plan_version"),
                "status": "proposed",
                "run_id": inputs["run_id"],
            })

            # Add planned receipt to simulation so next cycle is correctly projected
            planned_receipts[receipt_date] = planned_receipts.get(receipt_date, 0.0) + order_qty
            qty += order_qty  # immediately account for planned receipt in projection

    return orders


def compute_confidence_score(inputs: dict, config: dict) -> tuple:
    """
    Returns (score: float, reason: str).
    Score degrades based on data quality flags.
    """
    score = 1.0
    reasons = []
    penalties = config["recommendation"]["confidence"]

    if inputs.get("forecast_source") == "fallback_avg":
        score -= penalties["penalty_fallback_forecast"]
        reasons.append("using fallback demand average (no production forecast)")

    if not inputs.get("open_po_data_available", True):
        score -= penalties["penalty_no_open_po_data"]
        reasons.append("open PO delivery dates unavailable")

    # Check if order_by_date is already past
    for order in inputs.get("orders", []):
        if order["order_by_date"] < date.today():
            score -= penalties["penalty_past_due_order"]
            reasons.append("order already past due")
            break

    score = round(max(0.0, min(1.0, score)), 3)
    return score, "; ".join(reasons) if reasons else "all data sources available"
```

### 5.1 Makefile Targets

```makefile
## Order Recommendation Engine
planned-orders-schema:
	uv run python -c "import psycopg; conn=psycopg.connect(**__import__('common.db',fromlist=['get_db_params']).get_db_params()); conn.autocommit=True; conn.execute(open('sql/043_create_planned_orders.sql').read()); print('done')"

planned-orders-generate:
	uv run python scripts/generate_planned_orders.py

planned-orders-generate-dfu:
	uv run python scripts/generate_planned_orders.py --dfu $(ITEM) $(LOC)

planned-orders-dry:
	uv run python scripts/generate_planned_orders.py --dry-run

planned-orders-all: planned-orders-schema planned-orders-generate
```

---

## 6. API Endpoints

### 6.1 `GET /supply/planned-orders`

Returns planned orders, filterable by status, item, supplier, or urgency.

**Query params:**
| Param | Type | Default | Description |
|---|---|---|---|
| `item_no` | string | optional | Filter by item |
| `loc` | string | optional | Filter by location |
| `status` | string | `proposed,approved` | Comma-sep status filter |
| `past_due_only` | bool | false | Filter to orders where order_by_date < today |
| `supplier_id` | string | optional | Filter by supplier |
| `page` | int | 1 | Pagination |
| `page_size` | int | 50 | Rows per page |

**Response:**
```json
{
  "total": 1482,
  "total_order_value_usd": 1842340.00,
  "past_due_count": 34,
  "items": [
    {
      "id": 1001,
      "item_no": "100320",
      "loc": "1401-BULK",
      "supplier_id": "VENDOR-0042",
      "supplier_name": "Acme Supply Co.",
      "net_requirement_qty": 220.0,
      "recommended_qty": 300.0,
      "moq": 100.0,
      "unit_cost": 12.50,
      "order_value": 3750.00,
      "trigger_date": "2026-03-10",
      "trigger_reason": "projected_below_ss",
      "order_by_date": "2026-03-10",
      "expected_receipt_date": "2026-03-24",
      "lead_time_days": 14,
      "current_qty_on_hand": 120.0,
      "safety_stock": 60.0,
      "confirmed_inbound_qty": 200.0,
      "lt_forecast_demand": 228.2,
      "plan_version": "2026-03",
      "confidence_score": 0.950,
      "confidence_reason": "all data sources available",
      "is_past_due": false,
      "status": "proposed",
      "created_at": "2026-03-02T08:04:22Z"
    }
  ]
}
```

### 6.2 `PUT /supply/planned-orders/{id}/approve`

Approves a proposed planned order. Requires API key.

**Body:** `{"approved_by": "jane.smith@company.com"}`

**Response:**
```json
{
  "id": 1001,
  "status": "approved",
  "approved_by": "jane.smith@company.com",
  "approved_at": "2026-03-06T09:15:42Z"
}
```

### 6.3 `PUT /supply/planned-orders/{id}/reject`

Rejects a proposed order.

**Body:** `{"rejection_reason": "Demand is expected to fall — holding off"}`

**Response:** `{"id": 1001, "status": "rejected"}`

### 6.4 `PUT /supply/planned-orders/{id}/release`

Marks an approved order as released (transmitted to ERP). Requires API key.

**Response:** `{"id": 1001, "status": "released", "released_at": "..."}`

### 6.5 `POST /supply/planned-orders/generate`

Triggers planned order generation as an on-demand job. Requires API key.

**Body:** `{"item_no": "100320", "loc": "1401-BULK"}` (optional; omit for full portfolio scan)

**Response:** `{"status": "accepted", "job_id": "<uuid>"}` (HTTP 202)

### 6.6 `GET /supply/planned-orders/summary`

Portfolio-level KPI summary for the planner dashboard.

**Response:**
```json
{
  "status_counts": {
    "proposed": 1482,
    "approved": 203,
    "released": 87,
    "rejected": 44
  },
  "total_proposed_value_usd": 1842340.00,
  "total_approved_value_usd": 284150.00,
  "past_due_proposed_count": 34,
  "past_due_proposed_value_usd": 41200.00,
  "avg_confidence_score": 0.873,
  "low_confidence_count": 142,
  "generated_at": "2026-03-02T08:04:22Z"
}
```

---

## 7. Frontend Components

### 7.1 New Panel: `PlannedOrdersPanel` in Inv. Planning Tab

**File:** `frontend/src/tabs/inv-planning/PlannedOrdersPanel.tsx`

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  PLANNED ORDERS          Last generated: Mar 2 08:04     [Generate] [Export] │
├─────────────────┬──────────────────┬────────────────┬────────────────────────┤
│  1,482 Proposed │  $1.84M          │  34 PAST DUE   │  203 Approved          │
│  awaiting review│  total order val │  need action   │  ready to release      │
├─────────────────┴──────────────────┴────────────────┴────────────────────────┤
│  Filters: Item [──────────] Loc [──────────] Supplier [──────── ▼]          │
│           Status: [Proposed ▼]   [✓ Past-Due Only]   Confidence: [All ▼]    │
├──────────────────────────────────────────────────────────────────────────────┤
│  Item   │ Loc        │ Supplier     │ Rec. Qty │ Value  │ Order By │ Status   │
│  100320 │ 1401-BULK  │ Acme Supply  │   300    │ $3,750 │ Mar 10   │ Proposed │
│         │            │ (LT: 14d)   │          │        │ ← 4 days │ [Approve]│
│                                                                      [Reject] │
│  200147 │ 1401-BULK  │ Beta Comp.   │   400    │ $5,000 │ Feb 20 ⚠ │ Proposed │
│         │            │ (LT: 21d)   │          │        │ PAST DUE │ [Approve]│
├──────────────────────────────────────────────────────────────────────────────┤
│  DETAIL DRAWER (opens on row click):                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ Item 100320 @ 1401-BULK — Order Recommendation Detail                  │  │
│  │                                                                        │  │
│  │ Current on-hand: 120 units    Safety stock: 60 units                  │  │
│  │ Confirmed inbound: 200 units  (PO-4521: 150u Mar 14, PO-4522: 50u 21) │  │
│  │ Trigger date: Mar 10          (projected position drops to 59.8 ≤ 60) │  │
│  │ Lead time: 14 days            Expected receipt: Mar 24                 │  │
│  │ LT demand: 228.2 units        (from ML forecast: 16.3 units/day × 14) │  │
│  │ Net requirement: 220 units    (60 SS + 228.2 LT demand - 59.8 pos.)   │  │
│  │ MOQ: 100 units                Rounded up: 300 units                    │  │
│  │ Unit cost: $12.50             Total value: $3,750.00                   │  │
│  │ Confidence: 95.0% (high)                                               │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Approve/Reject flow:**
- `[Approve]` button triggers `PUT /supply/planned-orders/{id}/approve` with optimistic UI update
- `[Reject]` opens a modal requesting a rejection reason (free text, 200 chars max)
- Bulk approval: checkbox select multiple rows → "Approve Selected (N)" button

---

## 8. Worked Example: Item 100320, LOC 1401-BULK

**Date:** March 6, 2026

### Inputs

| Parameter | Value | Source |
|---|---|---|
| `qty_on_hand` | 120 units | `fact_inventory_snapshot` |
| `safety_stock` | 60 units | `fact_safety_stock_targets` |
| `reorder_point` | 60 units | = SS (simplified) |
| `lead_time_days` | 14 days | `dim_item_supplier` (preferred supplier) |
| `moq` | 100 units | `dim_item_supplier` |
| `unit_cost` | $12.50 | `dim_item_supplier` |
| `policy_id` | 3 (Review policy: continuous) | `fact_dfu_policy_assignment` |

**Confirmed Receipts (from `fact_open_purchase_orders`):**
```
Mar 14: +150 units (PO-4521)
Mar 21: +  50 units (PO-4522)
```

**Daily Demand (from `fact_production_forecast` → F1.1 → disaggregated):**
```
Mar 06–31: 16.3 units/day  (490 units / 30 days)
Apr 01–30: 16.0 units/day  (480 units / 30 days)
May 01–31: 16.5 units/day  (510 units / 31 days)
```

### Projection Simulation (with Open POs included)

| Day | Date | Demand | Receipts | Qty After |
|---|---|---|---|---|
| 0 | Mar 06 | — | — | 120.0 |
| 1 | Mar 07 | 16.3 | 0 | 103.7 |
| 2 | Mar 08 | 16.3 | 0 | 87.4 |
| 3 | Mar 09 | 16.3 | 0 | 71.1 |
| 4 | **Mar 10** | 16.3 | 0 | **54.8** ← **TRIGGER** (54.8 ≤ 60) |
| 5 | Mar 11 | 16.3 | 0 | 38.5 |
| 8 | Mar 14 | 16.3 | +150 | 186.9 |
| 15 | Mar 21 | 16.3 | +50 | 131.8 |
| ... | ... | ... | ... | ... |

### Net Requirement Calculation

```
trigger_date = Mar 10
projected_position[Mar 10] = 54.8

Lead time window: Mar 10 to Mar 24 (14 days)
lt_demand = 14 × 16.3 = 228.2 units

But wait: PO-4521 arrives Mar 14 (within the LT window!)
Should we include confirmed receipts during LT in the requirement calculation?
Answer: YES — confirmed receipts during LT reduce the net requirement.

lt_demand_net_of_receipts:
  demand Mar 10–13: 4 × 16.3 = 65.2
  PO-4521 arrives Mar 14: +150 → net position goes from ~22 to ~172
  demand Mar 14–23: 10 × 16.3 = 163
  Net position at Mar 24: 172 - 163 = 9.0

Projected position at receipt date (Mar 24): 9.0 units
net_requirement = max(0, SS + lt_demand - pos_at_trigger)
                = max(0, 60 + 228.2 - 54.8)
                = 233.4 units

[Note: The simplified formula uses trigger_date position, not LT-adjusted.
A more accurate version subtracts confirmed receipts during LT.
MVP uses simplified formula; LT-adjusted version is a future enhancement.]

net_requirement = 233.4 units
round_to_moq(233.4, 100, 'ceil_to_moq'):
  233.4 / 100 = 2.334 → ceil = 3 → 3 × 100 = 300 units

order_qty = 300 units
order_by_date = Mar 10 (today is Mar 6 → 4 days to act)
expected_receipt_date = Mar 10 + 14 = Mar 24
order_value = 300 × $12.50 = $3,750.00
```

### Output Row Written to `fact_planned_orders`

```
id                    = (next sequence)
item_no               = '100320'
loc                   = '1401-BULK'
supplier_id           = 'VENDOR-0042'
policy_id             = 3
net_requirement_qty   = 233.40
recommended_qty       = 300.00
moq                   = 100.00
unit_cost             = 12.50
order_value           = 3750.00   (computed column)
trigger_date          = 2026-03-10
trigger_reason        = 'projected_below_ss'
order_by_date         = 2026-03-10
expected_receipt_date = 2026-03-24
lead_time_days        = 14
current_qty_on_hand   = 120.00
safety_stock          = 60.00
reorder_point         = 60.00
confirmed_inbound_qty = 200.00   (150 + 50)
lt_forecast_demand    = 228.20
plan_version          = '2026-03'
confidence_score      = 0.950
status                = 'proposed'
is_past_due           = FALSE   (order_by_date is Mar 10, today is Mar 6)
```

---

## 9. Budget Cap Handling

When `budget_cap.enabled = true` in config:

1. After generating all proposed orders for the portfolio, sort by priority:
   - ABC class A first, then B, then C
   - Within class, sort by `confidence_score DESC`
   - Within confidence, sort by `order_by_date ASC` (most urgent first)

2. Accumulate `order_value` until `monthly_budget_usd` is reached.

3. Orders beyond the budget cap are written with `status = 'proposed'` and a flag `confidence_reason += '; budget_cap_exceeded'`. They are not automatically rejected — the planner can still approve them (with an override warning).

This prevents automatic approval workflows from placing $10M of orders when the budget is $500K.

---

## 10. Dependencies

| Dependency | Status | Notes |
|---|---|---|
| F1.1 Production Forecast | Required | Provides `daily_demand_by_date` via `fact_production_forecast` |
| F1.2 Inventory Projection | Required | Provides pre-computed `projected_position[t]` in `fact_inventory_projection` |
| F1.3 Open PO Integration | Required | Provides `confirmed_receipts_by_date` from `fact_open_purchase_orders` |
| `fact_safety_stock_targets` | Exists (IPfeature3) | SS and reorder_point values |
| `dim_replenishment_policy` | Exists (IPfeature5) | MOQ, review_cycle_days |
| `dim_item_supplier` | New (F1.3) | Lead time, unit_cost, preferred supplier |
| `dim_supplier` | New (F1.3) | Supplier name for UI display |

---

## 11. Out of Scope

- ERP write-back (auto-creation of POs in SAP/Oracle from approved planned orders)
- Dynamic safety stock recalculation as part of the order recommendation (SS from `fact_safety_stock_targets` is read-only input)
- Multi-supplier optimization (choosing cheapest supplier per order)
- Price-break quantity optimization (ordering 250 instead of 300 to hit a lower price tier)
- Consignment inventory handling (supplier-owned inventory on site)
- Cross-location lateral transfers as an alternative to ordering
- Demand sensing integration (sub-weekly signal adjustments to planned orders)

---

## 12. Test Requirements

### Backend Unit Tests (`tests/unit/test_planned_orders.py`)
- `test_round_to_moq_ceil_exact_multiple` — 200 units, MOQ=100 → 200 (no rounding needed)
- `test_round_to_moq_ceil_partial` — 220 units, MOQ=100 → 300
- `test_round_to_moq_ceil_minimum_moq` — 1 unit, MOQ=100 → 100
- `test_round_to_moq_zero_net_req` — 0 units, MOQ=100 → 100 (minimum 1 MOQ)
- `test_compute_net_requirements_trigger_on_day_4` — with example inputs, trigger on Mar 10
- `test_compute_net_requirements_correct_order_qty` — net_req=233.4 → recommended_qty=300
- `test_compute_net_requirements_no_trigger_sufficient_stock` — no orders when stock adequate
- `test_compute_net_requirements_past_due_trigger` — order_by_date < today still generated
- `test_compute_net_requirements_multi_cycle` — second trigger found after first planned receipt
- `test_compute_confidence_score_all_sources_available` — returns 0.95+ score
- `test_compute_confidence_score_fallback_forecast_penalty` — score reduced by 0.20
- `test_compute_confidence_score_no_po_data_penalty` — score reduced by 0.15
- `test_compute_confidence_score_past_due_penalty` — score reduced by 0.10
- `test_order_value_computed_correctly` — 300 × $12.50 = $3,750.00

### Backend API Tests (`tests/api/test_planned_orders.py`)
- `test_get_planned_orders_success` — 200, returns list
- `test_get_planned_orders_status_filter` — filters by 'proposed'
- `test_get_planned_orders_past_due_only` — filters by is_past_due
- `test_approve_planned_order_success` — 200, status changed to 'approved'
- `test_approve_planned_order_requires_auth` — 401 when API_KEY set
- `test_reject_planned_order_records_reason` — rejection_reason stored
- `test_release_planned_order_success` — 200, status changed to 'released'
- `test_generate_planned_orders_async_202` — returns 202 accepted
- `test_get_planned_orders_summary_success` — 200 with KPI counts and values

### Frontend Tests (`src/tabs/__tests__/InvPlanningTab.test.tsx`)
- `test_planned_orders_panel_renders`
- `test_planned_orders_panel_shows_kpi_cards`
- `test_planned_orders_panel_approve_button_calls_api`
- `test_planned_orders_panel_reject_opens_modal`
- `test_planned_orders_panel_past_due_warning_badge`
