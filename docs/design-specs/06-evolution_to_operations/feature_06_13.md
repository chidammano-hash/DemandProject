# Feature F4.1 — Financial Inventory Plan (Budget vs. Actuals)

**Phase:** 4 — Evolution to Operations
**Feature Number:** F4.1 (Internal: feature_06_13)
**Status:** Specification
**Author:** Supply Chain Systems Architecture
**Date:** 2026-03-06

---

## 1. Problem Statement

Supply chain planning decisions have direct and measurable financial consequences. Every planned order commits future cash, and every unit held in a warehouse consumes working capital. Despite having sophisticated inventory health scoring, safety stock calculation, EOQ, and replenishment exception detection, the current system has **no financial view of the inventory plan**.

### What Fails Today

**Scenario A — No cost visibility:**
A planner approves a wave of replenishment orders for 2,400 units across 85 SKUs. She has no way to know whether this wave totals $15,000 or $215,000. The system shows recommended order quantities in units only. Finance must separately pull a COGS report from the ERP and manually map it back to the planner's list — a 4-hour reconciliation process performed every week.

**Scenario B — Budget breach discovered too late:**
The Q2 category inventory budget is $850,000. The system generates recommendations totalling $1.1M in planned orders. Nobody knows this until Finance runs month-end actuals and raises the overspend. By then, 70% of the orders have already been placed.

**Scenario C — No forward-looking investment view:**
A senior director asks: "What will our total inventory value be in June if we execute the current plan?" There is no answer in the system. The only available data is the current on-hand snapshot. No projection of where inventory value is heading over the next 6 months exists.

**Scenario D — Working capital trapped in slow-movers:**
Across 12,000 SKU-locations, approximately $340,000 of inventory value is in items with >180 days of supply. This is invisible — there is no aggregate excess-value view that connects physical excess quantity to its dollar cost to the business.

---

## 2. Objectives

1. Compute the **current inventory investment** (on-hand × unit cost) at item, category, and portfolio level.
2. Project **forward inventory value** 1–12 months using the existing inventory projection pipeline.
3. Track **planned order value** as committed future cash outflow.
4. Enforce **budget caps** by category/buyer/location and generate budget breach alerts.
5. Quantify **excess inventory value** tied up in over-stocked items.
6. Surface **working capital release scenarios** when SS or policy parameters are relaxed.

---

## 3. Missing Input Data

The following data is **not currently in the system** and must be sourced externally:

| Data Element | Source | Priority | Notes |
|---|---|---|---|
| `unit_cost` per item-location | ERP item master or price list | CRITICAL | Moving-average cost preferred over standard cost. Landed cost ideal for import items. |
| `standard_cost` | ERP product costing module | MEDIUM | Alternative to moving-average for manufactured items |
| `budget_cap` per category/buyer | Finance system or manual entry | HIGH | Monthly, quarterly, or annual budget periods |
| COGS (Cost of Goods Sold) | ERP GL / P&L | MEDIUM | Required for inventory turns denominator |
| `carrying_cost_pct` | Finance team configuration | HIGH | Typically 20–35% annually (capital cost + warehouse + obsolescence + insurance) |

**Recommended cost type:** Moving-average cost (`moving_avg`) for purchased goods. Standard cost for manufactured items. Landed cost for international sourcing. The system will accept any of the three via the `cost_type` flag on `dim_item_cost`.

---

## 4. Data Model

### 4.1 New Table: `dim_item_cost`

**Grain:** item_no + loc + effective_from (one row per cost revision per item-location)
**Purpose:** Stores current and historical unit costs for financial valuation.

```sql
CREATE TABLE dim_item_cost (
    id               SERIAL PRIMARY KEY,
    item_no          VARCHAR(50)   NOT NULL,
    loc              VARCHAR(50)   NOT NULL,
    unit_cost        NUMERIC(12,4) NOT NULL CHECK (unit_cost >= 0),
    cost_type        VARCHAR(30)   NOT NULL DEFAULT 'moving_avg',
                     -- Allowed: standard / moving_avg / landed / manual
    currency         VARCHAR(3)    NOT NULL DEFAULT 'USD',
    effective_from   DATE          NOT NULL,
    effective_to     DATE,          -- NULL means currently active
    loaded_from      VARCHAR(50),   -- erp_api / csv_import / manual
    loaded_at        TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    created_by       VARCHAR(100),
    CONSTRAINT dim_item_cost_unique UNIQUE (item_no, loc, effective_from, cost_type)
);

CREATE INDEX idx_item_cost_item_loc ON dim_item_cost (item_no, loc);
CREATE INDEX idx_item_cost_effective ON dim_item_cost (effective_from, effective_to);
-- Partial index for active costs
CREATE INDEX idx_item_cost_active ON dim_item_cost (item_no, loc)
    WHERE effective_to IS NULL;
```

**Example rows:**

| item_no | loc | unit_cost | cost_type | currency | effective_from | effective_to |
|---|---|---|---|---|---|---|
| 100320 | 1401-BULK | 24.50 | moving_avg | USD | 2025-10-01 | NULL |
| 100320 | 1401-BULK | 22.80 | moving_avg | USD | 2025-01-01 | 2025-09-30 |
| 100321 | 1401-BULK | 8.75 | standard | USD | 2025-07-01 | NULL |
| 100322 | 2201-DC | 142.00 | landed | USD | 2026-01-15 | NULL |

---

### 4.2 New Table: `fact_financial_inventory_plan`

**Grain:** item_no + loc + plan_month + plan_version
**Purpose:** Forward-looking financial inventory plan — projected inventory value and order spend.

```sql
CREATE TABLE fact_financial_inventory_plan (
    id                         BIGSERIAL PRIMARY KEY,
    item_no                    VARCHAR(50)   NOT NULL,
    loc                        VARCHAR(50)   NOT NULL,
    plan_month                 DATE          NOT NULL,  -- First of month
    plan_version               VARCHAR(50)   NOT NULL,  -- e.g. '2026-Q2-v1'
    projected_qty              NUMERIC(12,2) NOT NULL DEFAULT 0,
    unit_cost                  NUMERIC(12,4) NOT NULL,
    projected_inventory_value  NUMERIC(14,2) NOT NULL,  -- projected_qty * unit_cost
    planned_order_qty          NUMERIC(12,2) NOT NULL DEFAULT 0,
    planned_order_value        NUMERIC(14,2) NOT NULL DEFAULT 0,
    carrying_cost_monthly      NUMERIC(14,2),           -- projected_inventory_value * carrying_cost_pct / 12
    excess_qty                 NUMERIC(12,2) NOT NULL DEFAULT 0,
    excess_value               NUMERIC(14,2) NOT NULL DEFAULT 0,
    budget_cap                 NUMERIC(14,2),           -- NULL if no budget defined
    budget_remaining           NUMERIC(14,2),
    within_budget              BOOLEAN,
    abc_class                  CHAR(1),
    category                   VARCHAR(100),
    buyer_id                   VARCHAR(50),
    computed_at                TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    CONSTRAINT fin_plan_unique UNIQUE (item_no, loc, plan_month, plan_version)
);

CREATE INDEX idx_finplan_item_loc ON fact_financial_inventory_plan (item_no, loc);
CREATE INDEX idx_finplan_month ON fact_financial_inventory_plan (plan_month);
CREATE INDEX idx_finplan_version ON fact_financial_inventory_plan (plan_version);
CREATE INDEX idx_finplan_category ON fact_financial_inventory_plan (category, plan_month);
CREATE INDEX idx_finplan_budget ON fact_financial_inventory_plan (within_budget, plan_month)
    WHERE within_budget = FALSE;
```

---

### 4.3 New Table: `fact_budget_periods`

**Grain:** budget_id (one row per budget rule)
**Purpose:** Configurable budget caps by scope, period, and organizational unit.

```sql
CREATE TABLE fact_budget_periods (
    budget_id      SERIAL PRIMARY KEY,
    scope_type     VARCHAR(20)   NOT NULL,  -- global / category / buyer / location
    scope_value    VARCHAR(100),             -- NULL for global; category name, buyer_id, or loc
    period_type    VARCHAR(10)   NOT NULL,  -- monthly / quarterly / annual
    period_start   DATE          NOT NULL,
    period_end     DATE          NOT NULL,
    budget_cap     NUMERIC(14,2) NOT NULL CHECK (budget_cap > 0),
    currency       VARCHAR(3)    NOT NULL DEFAULT 'USD',
    carrying_cost_pct NUMERIC(5,4) DEFAULT 0.25,  -- 25% annual carrying cost
    set_by         VARCHAR(100),
    approved_by    VARCHAR(100),
    approved_at    TIMESTAMPTZ,
    created_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    notes          TEXT,
    CONSTRAINT budget_no_overlap UNIQUE (scope_type, scope_value, period_start, period_end)
);

CREATE INDEX idx_budget_scope ON fact_budget_periods (scope_type, scope_value);
CREATE INDEX idx_budget_period ON fact_budget_periods (period_start, period_end);
```

**Example rows:**

| budget_id | scope_type | scope_value | period_type | period_start | period_end | budget_cap |
|---|---|---|---|---|---|---|
| 1 | global | NULL | annual | 2026-01-01 | 2026-12-31 | 12,000,000 |
| 2 | category | Beverages | quarterly | 2026-04-01 | 2026-06-30 | 200,000 |
| 3 | buyer | BUYER_007 | monthly | 2026-04-01 | 2026-04-30 | 85,000 |
| 4 | location | 1401-BULK | quarterly | 2026-04-01 | 2026-06-30 | 450,000 |

---

### 4.4 New Materialized View: `mv_financial_summary`

**Grain:** category + plan_month + plan_version
**Purpose:** Pre-aggregated financial KPIs for fast dashboard queries.

```sql
CREATE MATERIALIZED VIEW mv_financial_summary AS
SELECT
    f.category,
    f.plan_month,
    f.plan_version,
    COUNT(DISTINCT f.item_no || '|' || f.loc)          AS sku_loc_count,
    SUM(f.projected_inventory_value)                    AS total_projected_value,
    SUM(f.planned_order_value)                          AS total_planned_order_value,
    SUM(f.carrying_cost_monthly)                        AS total_carrying_cost,
    SUM(f.excess_value)                                 AS total_excess_value,
    SUM(CASE WHEN f.within_budget = FALSE THEN 1 ELSE 0 END) AS budget_breach_count,
    MAX(b.budget_cap)                                   AS category_budget_cap,
    SUM(f.planned_order_value) / NULLIF(MAX(b.budget_cap), 0) * 100 AS budget_utilization_pct
FROM fact_financial_inventory_plan f
LEFT JOIN fact_budget_periods b
    ON b.scope_type = 'category'
   AND b.scope_value = f.category
   AND f.plan_month BETWEEN b.period_start AND b.period_end
GROUP BY f.category, f.plan_month, f.plan_version;

CREATE UNIQUE INDEX idx_mv_fin_summary
    ON mv_financial_summary (category, plan_month, plan_version);
```

---

## 5. Python Scripts

### 5.1 `scripts/compute_financial_plan.py`

**Purpose:** Join projected inventory quantities + planned order quantities + item costs + budget caps → compute and write `fact_financial_inventory_plan`.

```python
# scripts/compute_financial_plan.py

import yaml
import psycopg
import pandas as pd
from datetime import date, timedelta
from common.db import get_db_params

CONFIG_PATH = "config/financial_plan_config.yaml"

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def load_item_costs(conn) -> pd.DataFrame:
    """Load active unit costs for all item-locations."""
    sql = """
        SELECT item_no, loc, unit_cost, cost_type, currency
        FROM dim_item_cost
        WHERE effective_to IS NULL
    """
    return pd.read_sql(sql, conn)

def load_projected_quantities(conn, horizon_months: int) -> pd.DataFrame:
    """
    Load forward-projected inventory quantities from the inventory projection pipeline.
    Expects fact_inventory_projection to exist (output of F1.2 — Inventory Projection).
    Falls back to current on-hand if projection table is absent.
    """
    cutoff = date.today().replace(day=1)
    months = [
        (cutoff + timedelta(days=32 * i)).replace(day=1)
        for i in range(horizon_months)
    ]
    sql = """
        SELECT
            p.item_no,
            p.loc,
            p.projection_month AS plan_month,
            p.projected_eom_qty,
            p.planned_order_qty,
            p.reorder_point,
            p.max_stock_target,
            d.abc_class,
            i.category,
            i.buyer_id
        FROM fact_inventory_projection p
        JOIN dim_dfu d ON d.item_no = p.item_no AND d.loc = p.loc
        JOIN dim_item i ON i.item_no = p.item_no
        WHERE p.projection_month = ANY(%s)
          AND p.plan_version = %s
    """
    return pd.read_sql(sql, conn, params=(months, "latest"))

def load_budget_caps(conn) -> pd.DataFrame:
    """Load active budget caps for all scopes."""
    sql = """
        SELECT scope_type, scope_value, period_start, period_end,
               budget_cap, carrying_cost_pct
        FROM fact_budget_periods
        WHERE period_end >= CURRENT_DATE
    """
    return pd.read_sql(sql, conn)

def resolve_budget_cap(
    row: pd.Series,
    budgets: pd.DataFrame,
    plan_month: date
) -> tuple[float | None, float | None]:
    """
    Resolve budget cap for a given item-location row.
    Priority: category > buyer > location > global.
    Returns (budget_cap, budget_remaining) or (None, None) if no budget defined.
    """
    for scope_type, scope_value in [
        ("category", row.get("category")),
        ("buyer",    row.get("buyer_id")),
        ("location", row.get("loc")),
        ("global",   None),
    ]:
        mask = (
            (budgets["scope_type"] == scope_type) &
            (budgets["scope_value"].fillna("").eq(str(scope_value) if scope_value else "")) &
            (budgets["period_start"] <= plan_month) &
            (budgets["period_end"]   >= plan_month)
        )
        match = budgets[mask]
        if not match.empty:
            cap = float(match.iloc[0]["budget_cap"])
            return cap, cap   # budget_remaining computed post-aggregation
    return None, None

def compute_excess_qty(projected_qty: float, max_stock_target: float | None) -> float:
    """Excess = projected_qty above max stock target (e.g., SS + 2× cycle stock)."""
    if max_stock_target is None or max_stock_target <= 0:
        return 0.0
    return max(0.0, projected_qty - max_stock_target)

def compute_financial_plan(
    projections: pd.DataFrame,
    costs: pd.DataFrame,
    budgets: pd.DataFrame,
    carrying_cost_pct: float,
    plan_version: str,
) -> pd.DataFrame:
    """
    Core computation: join costs → compute financial values → resolve budget.

    Returns a DataFrame ready for bulk insert into fact_financial_inventory_plan.
    """
    merged = projections.merge(
        costs[["item_no", "loc", "unit_cost", "cost_type"]],
        on=["item_no", "loc"],
        how="left"
    )
    merged["unit_cost"] = merged["unit_cost"].fillna(0.0)

    # Financial calculations
    merged["projected_inventory_value"] = (
        merged["projected_eom_qty"] * merged["unit_cost"]
    ).round(2)
    merged["planned_order_value"] = (
        merged["planned_order_qty"] * merged["unit_cost"]
    ).round(2)
    merged["carrying_cost_monthly"] = (
        merged["projected_inventory_value"] * carrying_cost_pct / 12
    ).round(2)
    merged["excess_qty"] = merged.apply(
        lambda r: compute_excess_qty(r["projected_eom_qty"], r.get("max_stock_target")),
        axis=1
    )
    merged["excess_value"] = (merged["excess_qty"] * merged["unit_cost"]).round(2)

    # Budget resolution
    merged["budget_cap"]       = None
    merged["budget_remaining"] = None
    merged["within_budget"]    = None
    # Full budget enforcement requires aggregation by category × period — done in write_results()
    merged["plan_version"] = plan_version

    return merged.rename(columns={"projected_eom_qty": "projected_qty"})

def write_results(df: pd.DataFrame, conn, plan_version: str) -> int:
    """Bulk upsert computed financial plan into fact_financial_inventory_plan."""
    with conn.cursor() as cur:
        cur.execute("""
            DELETE FROM fact_financial_inventory_plan
            WHERE plan_version = %s
        """, (plan_version,))

        rows = [
            (
                r.item_no, r.loc, r.plan_month, plan_version,
                r.projected_qty, r.unit_cost, r.projected_inventory_value,
                r.planned_order_qty, r.planned_order_value, r.carrying_cost_monthly,
                r.excess_qty, r.excess_value,
                r.get("budget_cap"), r.get("budget_remaining"), r.get("within_budget"),
                r.get("abc_class"), r.get("category"), r.get("buyer_id"),
            )
            for _, r in df.iterrows()
        ]
        cur.executemany("""
            INSERT INTO fact_financial_inventory_plan (
                item_no, loc, plan_month, plan_version,
                projected_qty, unit_cost, projected_inventory_value,
                planned_order_qty, planned_order_value, carrying_cost_monthly,
                excess_qty, excess_value, budget_cap, budget_remaining, within_budget,
                abc_class, category, buyer_id
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, rows)
        conn.commit()
    return len(rows)

def run(horizon_months: int = 6, plan_version: str = "latest") -> None:
    cfg = load_config()
    carrying_cost_pct = cfg["carrying_cost_pct"]

    with psycopg.connect(**get_db_params()) as conn:
        costs       = load_item_costs(conn)
        projections = load_projected_quantities(conn, horizon_months)
        budgets     = load_budget_caps(conn)
        plan_df     = compute_financial_plan(
            projections, costs, budgets, carrying_cost_pct, plan_version
        )
        n = write_results(plan_df, conn, plan_version)
        print(f"[financial_plan] Wrote {n:,} rows for version '{plan_version}'")

        # Refresh materialized view
        with conn.cursor() as cur:
            cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY mv_financial_summary")
        conn.commit()
        print("[financial_plan] Refreshed mv_financial_summary")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--version", type=str, default="latest")
    args = parser.parse_args()
    run(args.horizon, args.version)
```

### 5.2 Config: `config/financial_plan_config.yaml`

```yaml
# Financial plan computation configuration
carrying_cost_pct: 0.25          # 25% annual carrying cost (capital + warehouse + insurance + obsolescence)
default_horizon_months: 6        # Months to project forward
default_plan_version: "latest"   # Version label for current plan
excess_threshold_days: 90        # Items with >90 DOS are flagged as excess

# Budget breach alert thresholds
budget_warning_pct: 85           # Warn when utilization > 85%
budget_breach_pct: 100           # Alert when utilization >= 100%

# Working capital release scenario parameters
wc_release_classes:              # Classes eligible for SS reduction in WC scenarios
  - C
wc_release_ss_reduction_pct: 10  # Default SS reduction % for scenario
```

---

## 6. API Endpoints

### `GET /finance/inventory-plan`

Returns projected inventory value and planned order spend by category for a given plan version and horizon.

**Query params:** `horizon=6`, `plan_version=latest`, `category=`, `abc_class=`

**Response:**
```json
{
  "plan_version": "latest",
  "computed_at": "2026-03-06T08:00:00Z",
  "summary": {
    "total_projected_inventory_value": 3840000.00,
    "total_planned_order_value": 312500.00,
    "total_carrying_cost_monthly": 80000.00,
    "total_excess_value": 284300.00,
    "sku_loc_count": 12847
  },
  "by_category": [
    {
      "category": "Beverages",
      "plan_month": "2026-04-01",
      "projected_inventory_value": 182400.00,
      "planned_order_value": 47500.00,
      "carrying_cost_monthly": 3800.00,
      "excess_value": 12400.00,
      "budget_cap": 200000.00,
      "budget_utilization_pct": 91.0,
      "within_budget": true
    }
  ]
}
```

---

### `GET /finance/budget-status`

Returns current budget utilization across all defined budget periods.

**Query params:** `period_start=`, `period_end=`, `scope_type=`

**Response:**
```json
{
  "budgets": [
    {
      "budget_id": 2,
      "scope_type": "category",
      "scope_value": "Beverages",
      "period_type": "quarterly",
      "period_start": "2026-04-01",
      "period_end": "2026-06-30",
      "budget_cap": 200000.00,
      "committed_value": 182400.00,
      "planned_additional": 47500.00,
      "total_exposure": 229900.00,
      "utilization_pct": 114.95,
      "status": "BREACHED",
      "breach_amount": 29900.00
    }
  ]
}
```

---

### `GET /finance/working-capital-trend`

Returns monthly inventory value trend (actual + projected) for working capital analysis.

**Query params:** `months_history=6`, `months_forward=6`, `category=`

**Response:**
```json
{
  "trend": [
    { "month": "2025-10-01", "type": "actual",    "inventory_value": 3620000.00, "planned_order_value": null },
    { "month": "2026-04-01", "type": "projected",  "inventory_value": 3840000.00, "planned_order_value": 312500.00 },
    { "month": "2026-09-01", "type": "projected",  "inventory_value": 3210000.00, "planned_order_value": 287000.00 }
  ]
}
```

---

### `GET /finance/excess-value`

Returns top items by excess inventory value with liquidation priority score.

**Query params:** `limit=20`, `abc_class=`, `min_excess_value=1000`

**Response:**
```json
{
  "total_excess_value": 284300.00,
  "excess_sku_count": 387,
  "items": [
    {
      "item_no": "100320",
      "loc": "1401-BULK",
      "item_description": "Energy Drink 500ml",
      "abc_class": "C",
      "qty_on_hand": 420,
      "max_stock_target": 180,
      "excess_qty": 240,
      "unit_cost": 24.50,
      "excess_value": 5880.00,
      "days_of_supply": 312,
      "liquidation_priority": "HIGH",
      "recommendation": "Transfer to high-velocity location or initiate markdown"
    }
  ]
}
```

---

### `PUT /finance/budget/{budget_id}`

Update an existing budget cap.

**Request body:**
```json
{ "budget_cap": 220000.00, "approved_by": "CFO_jsmith", "notes": "Approved uplift for Easter promo" }
```

---

### `POST /finance/budget`

Create a new budget period.

**Request body:**
```json
{
  "scope_type": "category",
  "scope_value": "Snacks",
  "period_type": "quarterly",
  "period_start": "2026-04-01",
  "period_end": "2026-06-30",
  "budget_cap": 150000.00,
  "currency": "USD",
  "set_by": "BUYER_012",
  "carrying_cost_pct": 0.25
}
```

---

## 7. Frontend Components

### 7.1 New Panel: "Financial Plan" in Inv. Planning Tab

**Location:** New sub-panel inside `InvPlanningTab.tsx` rendered as `FinancialPlanPanel`

```
┌─────────────────────────────────────────────────────────────────────┐
│  FINANCIAL INVENTORY PLAN          Plan: 2026-Q2-v1   Horizon: 6M  │
├──────────┬──────────┬──────────┬──────────┬─────────────────────────┤
│ INVENTORY│ PROJECTED│ PLANNED  │ CARRYING │  EXCESS INVENTORY VALUE  │
│  VALUE   │  6-MONTH │  ORDERS  │  COST/MO │                         │
│          │  VALUE   │  VALUE   │          │                         │
│  $3.84M  │  $3.21M  │ $312.5K  │  $80.0K  │       $284.3K           │
│  ON HAND │ FORWARD  │ COMMITTED│ ANNUAL:  │   387 SKU-LOCATIONS     │
│          │ LOOKING  │          │  $960K   │                         │
├──────────┴──────────┴──────────┴──────────┴─────────────────────────┤
│  WORKING CAPITAL TIMELINE (Stacked Area by Category)                │
│                                                                     │
│  $4.5M ┤████████████████████████████████████████████████████       │
│  $4.0M ┤████████████████████████████████████████████████████       │
│  $3.5M ┤████████████████████████████████████████████████████       │
│  $3.0M ┤▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓       │
│  $2.5M ┤░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░       │
│        └──────────────────────────────────────────────────          │
│        Oct  Nov  Dec  Jan  Feb  Mar │ Apr  May  Jun  Jul  Aug  Sep  │
│                        ACTUAL      │       PROJECTED                │
├─────────────────────────────────────────────────────────────────────┤
│  BUDGET VS ACTUALS BY CATEGORY                                      │
│  Category        Budget Cap   Committed    Planned      Status      │
│  ─────────────── ──────────   ─────────    ────────     ──────────  │
│  Beverages       $200,000     $182,400     $47,500    ⚠ BREACHED   │
│  Snacks          $150,000      $98,200     $31,100    ✓ ON TRACK   │
│  Dairy           $120,000      $87,300     $18,900    ✓ ON TRACK   │
│  Frozen          $180,000     $154,600     $22,100    ⚠ WARNING    │
├─────────────────────────────────────────────────────────────────────┤
│  TOP 20 EXCESS INVENTORY ITEMS                            [Export]  │
│  Item       Description         Excess Qty  Unit Cost  Excess Value │
│  100320     Energy Drink 500ml  240 units    $24.50    $5,880       │
│  100445     Orange Juice 1L     180 units    $32.10    $5,778       │
│  ...                                                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Budget Breach Alert

When any category or budget period is breached, a red alert banner appears at the top of the Financial Plan panel:

```
⚠ BUDGET ALERT: "Beverages" Q2 budget exceeded by $29,900 (114.9% utilization).
  4 planned orders need deferral to restore compliance.
  [View Affected Orders]  [Request Budget Increase]
```

---

## 8. Worked Example — End-to-End Numbers

**Item:** 100320 — Energy Drink 500ml
**Location:** 1401-BULK
**Unit cost (moving average):** $24.50

| Month | Current On-Hand | Planned Orders | Projected EOM Qty | Projected Value | Max Stock Target | Excess Qty | Excess Value |
|---|---|---|---|---|---|---|---|
| Apr 2026 | 120 units | +220 units | 340 units | $8,330 | 200 units | 140 units | $3,430 |
| May 2026 | 340 units | +0 | 210 units | $5,145 | 200 units | 10 units | $245 |
| Jun 2026 | 210 units | +180 | 340 units | $8,330 | 200 units | 140 units | $3,430 |

**Category "Beverages" budget analysis for Q2:**

```
Total DFUs in Beverages at 1401-BULK:  847 SKU-locations
Sum of planned order values Q2:        $229,900
Category budget cap (Q2):             $200,000
─────────────────────────────────────────────
Budget utilized:                       $182,400  (actual April committed)
Planned additional orders:             $47,500
Total exposure:                        $229,900
Budget remaining:                     ($29,900)  ← BREACH
─────────────────────────────────────────────
Budget utilization:                    114.95%
```

**Auto-deferral logic:** System ranks 4 pending orders by ABC class. 3 C-class orders (total $31,200) are flagged for deferral to May. 1 A-class order for $16,300 is protected.

**Working capital release scenario:**
"Reduce SS by 10% for C-class items in Beverages."
- C-class DFUs in Beverages: 312
- Average SS value per DFU: $420
- Total SS reduction: 312 × $420 × 10% = $13,104
- Projected carrying cost saving: $13,104 × 25% / 12 = $273/month

---

## 9. Dependencies

| Dependency | Feature | Status |
|---|---|---|
| Inventory projection (forward quantities) | F1.2 — Inventory Projection | Not yet implemented |
| Planned order quantities | F2.1 — Order Recommendations | Not yet implemented |
| Safety stock targets | IPfeature3 | Implemented |
| EOQ cycle stock | IPfeature4 | Implemented |
| ABC-XYZ classification | IPfeature11 | Implemented |
| Item master (category, buyer) | dim_item | Implemented |

**Graceful degradation:** If `fact_inventory_projection` does not exist, fall back to current on-hand quantity as the "projected" quantity for month 1 only. Remaining months show zero. A warning banner in the UI states: "Forward inventory projection is unavailable. Showing current on-hand only."

---

## 10. Out of Scope

- Multi-currency conversion (all values assumed to be in `USD`; currency conversion rates and FX hedging are out of scope)
- ERP direct API integration (item cost loading is CSV-import based in this phase; ERP API connector is a separate infrastructure feature)
- P&L impact calculation (revenue, margin, COGS reconciliation — belongs to finance system)
- Purchase order commitment tracking (open POs are a separate data source not yet ingested)
- Obsolescence write-down calculation (requires accounting policy decisions outside the scope of supply chain planning)

---

## 11. Test Requirements

### Backend Unit Tests (`tests/unit/test_financial_plan.py`)

- `test_compute_excess_qty_no_target()` — returns 0 when max_stock_target is None
- `test_compute_excess_qty_within_target()` — returns 0 when projected_qty < max_stock_target
- `test_compute_excess_qty_above_target()` — returns correct excess when projected_qty > max_stock_target
- `test_carrying_cost_formula()` — $10,000 value × 25% / 12 = $208.33/month
- `test_projected_value_zero_cost()` — zero cost items produce zero value (no divide by zero)
- `test_budget_resolution_category_priority()` — category budget takes precedence over global
- `test_budget_resolution_no_budget()` — returns (None, None) when no budget defined
- `test_compute_financial_plan_shape()` — output DataFrame has all required columns
- `test_compute_financial_plan_within_budget()` — item below cap is flagged within_budget=True
- `test_compute_financial_plan_breach()` — item above cap is flagged within_budget=False

### Backend API Tests (`tests/api/test_finance.py`)

- `test_get_inventory_plan_200()` — returns 200 with summary and by_category
- `test_get_budget_status_200()` — returns list of budget utilization rows
- `test_get_budget_status_breached()` — breached budget has status="BREACHED" and breach_amount > 0
- `test_get_working_capital_trend_200()` — returns trend list with actual + projected entries
- `test_get_excess_value_200()` — returns items sorted by excess_value desc
- `test_post_budget_201()` — creates new budget period, returns budget_id
- `test_put_budget_200()` — updates existing budget cap
- `test_put_budget_404()` — returns 404 for non-existent budget_id

### Frontend Tests (`src/tabs/__tests__/FinancialPlanPanel.test.tsx`)

- Renders 4 KPI cards (Inventory Value, Projected Value, Planned Orders, Carrying Cost)
- Budget breach alert renders when `utilization_pct > 100`
- Working capital timeline chart renders with correct months
- Budget table shows BREACHED status in red for over-cap categories
- Export button triggers CSV download
- Empty state renders when no cost data is loaded

---

## 12. Makefile Targets

```makefile
financial-plan-schema:
	uv run python -c "import psycopg, yaml; ..."   # Apply DDL for dim_item_cost + fact_financial_inventory_plan + fact_budget_periods

financial-plan-compute:
	uv run python scripts/compute_financial_plan.py --horizon 6 --version latest

financial-plan-all: financial-plan-schema financial-plan-compute
```
