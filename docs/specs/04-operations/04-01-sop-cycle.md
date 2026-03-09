# Feature F4.2 — Sales & Operations Planning (S&OP) Module

**Phase:** 4 — Evolution to Operations
**Feature Number:** F4.2 (Internal: feature_06_14)
**Status:** Specification
**Author:** Supply Chain Systems Architecture
**Date:** 2026-03-06

---

## 1. Problem Statement

Demand Studio currently produces excellent statistical forecasts, backtest results, exception queues, and inventory recommendations in isolation. However, there is **no structured consensus process** that brings demand, supply, and finance into a single agreed plan that the organization can commit to and act on.

### What Fails Today

**Scenario A — Disconnected functional views:**
The commercial team works from a spreadsheet with their promotional uplift estimates. The supply team works from a separate inventory planning system. Finance has a budget model in Excel. None of these inputs are reconciled. When a supply constraint emerges (e.g., a supplier cuts capacity by 30%), the commercial team is not informed and continues selling as if inventory is available.

**Scenario B — No version control on the plan:**
When the supply plan changes, there is no record of what was approved vs. what changed and why. Accountability is unclear. Finance cannot trace a budget overspend back to a specific planning decision made in week 2 of the month.

**Scenario C — Supply gaps discovered at execution:**
A DC manager discovers a 2,100-unit shortfall in "Beverages" 3 days before the promotional window opens. This gap should have been identified 3 weeks earlier in a structured Supply Review, not at execution. The absence of a Pre-S&OP gap analysis leaves critical risks undiscovered until it is too late to mitigate.

**Scenario D — No approved plan published to planning engine:**
Statistical forecasts and commercial overrides currently coexist without a clear hierarchy. There is no concept of an "approved plan" that supersedes all other demand signals for a given month. The planning engine has no authoritative demand signal for the future.

---

## 2. Objectives

1. Implement the **standard 4-stage monthly S&OP cycle** as a structured workflow in the system.
2. Capture **demand review inputs** — statistical forecasts, commercial overrides, and consensus demand — at category level.
3. Capture **supply review inputs** — supplier capacity constraints, DC capacity events, lead time changes.
4. Automate **Pre-S&OP gap analysis** — compute demand vs supply gaps and identify unresolved risks.
5. Support **executive S&OP approval** — lock an approved plan that becomes the authoritative demand signal.
6. Publish the **approved plan** to `fact_sop_approved_plan` which planning engines consume.

---

## 3. S&OP Process Architecture

```
  WEEK 1           WEEK 2           WEEK 3           WEEK 4
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐
│  DEMAND     │  │  SUPPLY     │  │  PRE-S&OP   │  │  EXECUTIVE   │
│  REVIEW     │→ │  REVIEW     │→ │  GAP        │→ │  S&OP        │
│             │  │             │  │  ANALYSIS   │  │  APPROVAL    │
│ Statistical │  │ Supplier    │  │ Demand vs   │  │ CEO/CFO/VP   │
│ Forecast    │  │ Capacity    │  │ Supply Gap  │  │ Supply Chain │
│ Commercial  │  │ DC Capacity │  │ Budget Gap  │  │ Lock Plan    │
│ Overrides   │  │ Lead Time   │  │ Escalations │  │ Publish      │
│ Consensus   │  │ Constraints │  │ Resolutions │  │              │
└─────────────┘  └─────────────┘  └─────────────┘  └──────────────┘
       │                │                │                 │
       ▼                ▼                ▼                 ▼
 fact_sop_demand  fact_sop_supply  fact_sop_gaps    fact_sop_approved
 _review          _constraints                      _plan
```

### Stage Transitions

| Stage | Trigger | Responsible | Output |
|---|---|---|---|
| `demand_review` | Cycle created (auto or manual) | Demand Planner | `fact_sop_demand_review` rows |
| `supply_review` | Demand review submitted | Supply Planner | `fact_sop_supply_constraints` rows |
| `pre_sop` | Supply review submitted | S&OP Facilitator | `fact_sop_gaps` rows, escalation list |
| `executive_sop` | Pre-S&OP complete | VP Supply Chain | Gap resolutions approved |
| `approved` | Executive sign-off | CEO / CFO | `fact_sop_approved_plan` populated |
| `closed` | Month-end actuals loaded | Automated | Performance vs plan computed |

---

## 4. Data Model

### 4.1 New Table: `fact_sop_cycles`

**Grain:** cycle_id (one row per monthly S&OP cycle)
**Purpose:** Master record for each monthly S&OP cycle, tracking status and key timestamps.

```sql
CREATE TABLE fact_sop_cycles (
    cycle_id                SERIAL PRIMARY KEY,
    cycle_month             DATE          NOT NULL UNIQUE,  -- First of month (e.g., 2026-04-01)
    status                  VARCHAR(30)   NOT NULL DEFAULT 'demand_review',
    -- Allowed: demand_review / supply_review / pre_sop / executive_sop / approved / closed
    demand_plan_version     VARCHAR(50),   -- Label for the demand plan produced in this cycle
    supply_plan_version     VARCHAR(50),
    approved_plan_version   VARCHAR(50),
    facilitated_by          VARCHAR(100),  -- S&OP facilitator (user ID)
    approved_by             VARCHAR(100),  -- Executive who approved
    demand_review_at        TIMESTAMPTZ,
    supply_review_at        TIMESTAMPTZ,
    pre_sop_at              TIMESTAMPTZ,
    exec_sop_at             TIMESTAMPTZ,
    approved_at             TIMESTAMPTZ,
    closed_at               TIMESTAMPTZ,
    notes                   TEXT,
    created_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sop_cycles_month  ON fact_sop_cycles (cycle_month);
CREATE INDEX idx_sop_cycles_status ON fact_sop_cycles (status);
```

---

### 4.2 New Table: `fact_sop_demand_review`

**Grain:** cycle_id + item_category (one row per category per cycle)
**Purpose:** Demand review submissions — statistical vs commercial vs consensus demand.

```sql
CREATE TABLE fact_sop_demand_review (
    id                        BIGSERIAL PRIMARY KEY,
    cycle_id                  INTEGER       NOT NULL REFERENCES fact_sop_cycles(cycle_id),
    item_category             VARCHAR(100)  NOT NULL,
    plan_month                DATE          NOT NULL,
    statistical_forecast_qty  NUMERIC(12,2) NOT NULL,
    statistical_forecast_val  NUMERIC(14,2),
    commercial_override_qty   NUMERIC(12,2) NOT NULL DEFAULT 0,
    commercial_override_val   NUMERIC(14,2) DEFAULT 0,
    commercial_override_reason TEXT,
    consensus_demand_qty      NUMERIC(12,2),          -- Agreed after demand review meeting
    consensus_demand_val      NUMERIC(14,2),
    submitted_by              VARCHAR(100),
    submitted_at              TIMESTAMPTZ,
    review_status             VARCHAR(20)   NOT NULL DEFAULT 'draft',
    -- Allowed: draft / submitted / reviewed / approved
    reviewer_notes            TEXT,
    reviewed_by               VARCHAR(100),
    reviewed_at               TIMESTAMPTZ,
    CONSTRAINT sop_demand_review_unique UNIQUE (cycle_id, item_category)
);

CREATE INDEX idx_sop_demand_cycle    ON fact_sop_demand_review (cycle_id);
CREATE INDEX idx_sop_demand_category ON fact_sop_demand_review (item_category);
```

**Example rows (April 2026 cycle):**

| cycle_id | item_category | stat_forecast_qty | commercial_override_qty | consensus_demand_qty | review_status |
|---|---|---|---|---|---|
| 12 | Beverages | 12,500 | +800 (promo) -200 (phase-out) = +600 | 13,100 | approved |
| 12 | Snacks | 8,200 | +150 | 8,350 | approved |
| 12 | Dairy | 4,100 | 0 | 4,100 | approved |

---

### 4.3 New Table: `fact_sop_supply_constraints`

**Grain:** cycle_id + constraint_id
**Purpose:** Supply review — all identified supply constraints for a cycle.

```sql
CREATE TABLE fact_sop_supply_constraints (
    constraint_id         BIGSERIAL PRIMARY KEY,
    cycle_id              INTEGER       NOT NULL REFERENCES fact_sop_cycles(cycle_id),
    constraint_type       VARCHAR(50)   NOT NULL,
    -- Allowed: supplier_capacity / dc_capacity / lead_time_change / seasonal_allocation / port_delay / quality_hold
    supplier_id           VARCHAR(50),
    supplier_name         VARCHAR(200),
    item_category         VARCHAR(100),
    item_no               VARCHAR(50),   -- NULL means category-wide constraint
    constraint_description TEXT         NOT NULL,
    impact_qty            NUMERIC(12,2), -- Units unavailable due to constraint
    impact_value          NUMERIC(14,2),
    impact_period_start   DATE          NOT NULL,
    impact_period_end     DATE          NOT NULL,
    affected_locations    JSONB,         -- List of loc codes, or null for all
    mitigation_action     TEXT,
    mitigation_status     VARCHAR(20)   DEFAULT 'open',
    -- Allowed: open / in_progress / resolved / accepted
    submitted_by          VARCHAR(100),
    submitted_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    resolved_by           VARCHAR(100),
    resolved_at           TIMESTAMPTZ
);

CREATE INDEX idx_sop_supply_cycle    ON fact_sop_supply_constraints (cycle_id);
CREATE INDEX idx_sop_supply_type     ON fact_sop_supply_constraints (constraint_type);
CREATE INDEX idx_sop_supply_supplier ON fact_sop_supply_constraints (supplier_id);
```

---

### 4.4 New Table: `fact_sop_gaps`

**Grain:** cycle_id + gap_id
**Purpose:** Pre-S&OP gap analysis — computed and manually entered gaps between demand plan and supply capability.

```sql
CREATE TABLE fact_sop_gaps (
    gap_id            BIGSERIAL PRIMARY KEY,
    cycle_id          INTEGER       NOT NULL REFERENCES fact_sop_cycles(cycle_id),
    gap_type          VARCHAR(50)   NOT NULL,
    -- Allowed: demand_supply_gap / budget_gap / service_level_gap / lead_time_gap / capacity_gap
    item_category     VARCHAR(100),
    plan_month        DATE,
    gap_qty           NUMERIC(12,2),  -- Units of unmet demand or excess supply
    gap_value         NUMERIC(14,2),  -- Financial value of gap
    gap_description   TEXT,
    severity          VARCHAR(10)   NOT NULL DEFAULT 'medium',
    -- Allowed: critical / high / medium / low
    resolution_options JSONB,         -- Array of text options discussed in pre-S&OP
    resolution_status VARCHAR(20)   NOT NULL DEFAULT 'open',
    -- Allowed: open / mitigated / accepted / escalated / resolved
    resolution_notes  TEXT,
    resolved_by       VARCHAR(100),
    resolved_at       TIMESTAMPTZ,
    created_at        TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sop_gaps_cycle    ON fact_sop_gaps (cycle_id);
CREATE INDEX idx_sop_gaps_status   ON fact_sop_gaps (resolution_status);
CREATE INDEX idx_sop_gaps_severity ON fact_sop_gaps (severity)
    WHERE resolution_status IN ('open', 'escalated');
```

---

### 4.5 New Table: `fact_sop_approved_plan`

**Grain:** cycle_id + item_no + loc + plan_month
**Purpose:** The locked, executive-approved demand plan. This is the authoritative demand signal for the planning engine.

```sql
CREATE TABLE fact_sop_approved_plan (
    id              BIGSERIAL PRIMARY KEY,
    cycle_id        INTEGER       NOT NULL REFERENCES fact_sop_cycles(cycle_id),
    item_no         VARCHAR(50)   NOT NULL,
    loc             VARCHAR(50)   NOT NULL,
    item_category   VARCHAR(100),
    plan_month      DATE          NOT NULL,
    approved_qty    NUMERIC(12,2) NOT NULL,
    approved_value  NUMERIC(14,2),
    source          VARCHAR(30)   NOT NULL,
    -- Allowed: consensus / statistical / commercial_override / sop_adjusted
    statistical_qty NUMERIC(12,2),  -- Original statistical forecast for comparison
    override_qty    NUMERIC(12,2),  -- Commercial override component
    sop_adjustment  NUMERIC(12,2),  -- Additional adjustment made at executive S&OP
    approved_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    approved_by     VARCHAR(100),
    locked          BOOLEAN         NOT NULL DEFAULT FALSE,
    -- Once locked = TRUE, cannot be modified without cycle re-opening
    plan_version    VARCHAR(50)     NOT NULL,
    CONSTRAINT sop_approved_unique UNIQUE (cycle_id, item_no, loc, plan_month)
);

CREATE INDEX idx_sop_approved_cycle ON fact_sop_approved_plan (cycle_id);
CREATE INDEX idx_sop_approved_item  ON fact_sop_approved_plan (item_no, loc);
CREATE INDEX idx_sop_approved_month ON fact_sop_approved_plan (plan_month);
CREATE INDEX idx_sop_approved_locked ON fact_sop_approved_plan (locked, plan_month)
    WHERE locked = TRUE;
```

---

## 5. Python Scripts

### 5.1 `scripts/run_sop_cycle.py`

**Purpose:** Orchestrates the 4-step S&OP cycle: generate demand review summary, compile supply constraints, compute gaps, and publish approved plan.

```python
# scripts/run_sop_cycle.py

import yaml
import psycopg
import pandas as pd
from datetime import date
from typing import Literal
from common.db import get_db_params

CONFIG_PATH = "config/sop_config.yaml"

SopStage = Literal[
    "demand_review", "supply_review", "pre_sop",
    "executive_sop", "approved", "closed"
]
STAGE_ORDER = [
    "demand_review", "supply_review", "pre_sop",
    "executive_sop", "approved", "closed"
]

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def get_or_create_cycle(conn, cycle_month: date) -> int:
    """Fetch existing cycle or create a new one for the given month."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT cycle_id FROM fact_sop_cycles WHERE cycle_month = %s",
            (cycle_month,)
        )
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute("""
            INSERT INTO fact_sop_cycles (cycle_month, status)
            VALUES (%s, 'demand_review')
            RETURNING cycle_id
        """, (cycle_month,))
        conn.commit()
        return cur.fetchone()[0]

def generate_demand_review_summary(conn, cycle_id: int, cycle_month: date) -> pd.DataFrame:
    """
    Pull statistical forecasts from fact_external_forecast_monthly (model_id='external'),
    grouped by item_category. One row per category.

    Returns a DataFrame with columns:
      [item_category, statistical_forecast_qty, statistical_forecast_val]
    """
    sql = """
        SELECT
            i.category                         AS item_category,
            SUM(f.qty)                         AS statistical_forecast_qty,
            SUM(f.qty * COALESCE(c.unit_cost, 0)) AS statistical_forecast_val
        FROM fact_external_forecast_monthly f
        JOIN dim_item i ON i.item_no = f.dmdunit
        LEFT JOIN dim_item_cost c
            ON c.item_no = f.dmdunit
           AND c.loc = f.loc
           AND c.effective_to IS NULL
        WHERE f.startdate = %s
          AND f.model_id = 'external'
          AND f.lag = 0
        GROUP BY i.category
    """
    return pd.read_sql(sql, conn, params=(cycle_month,))

def upsert_demand_review(conn, cycle_id: int, df: pd.DataFrame) -> int:
    """Insert category-level demand review rows (draft status)."""
    with conn.cursor() as cur:
        rows_written = 0
        for _, row in df.iterrows():
            cur.execute("""
                INSERT INTO fact_sop_demand_review
                    (cycle_id, item_category, plan_month, statistical_forecast_qty,
                     statistical_forecast_val, review_status)
                VALUES (%s, %s, %s, %s, %s, 'draft')
                ON CONFLICT (cycle_id, item_category) DO UPDATE SET
                    statistical_forecast_qty = EXCLUDED.statistical_forecast_qty,
                    statistical_forecast_val = EXCLUDED.statistical_forecast_val
            """, (
                cycle_id,
                row["item_category"],
                df.attrs.get("cycle_month"),
                row["statistical_forecast_qty"],
                row.get("statistical_forecast_val"),
            ))
            rows_written += 1
        conn.commit()
    return rows_written

def compute_supply_demand_gaps(conn, cycle_id: int) -> list[dict]:
    """
    Compare consensus demand plan vs supply constraints.
    Returns a list of gap dicts ready for insertion into fact_sop_gaps.
    """
    sql_demand = """
        SELECT item_category, consensus_demand_qty
        FROM fact_sop_demand_review
        WHERE cycle_id = %s AND review_status = 'approved'
    """
    sql_supply = """
        SELECT item_category, SUM(impact_qty) AS constrained_qty
        FROM fact_sop_supply_constraints
        WHERE cycle_id = %s AND mitigation_status = 'open'
        GROUP BY item_category
    """
    with conn.cursor() as cur:
        cur.execute(sql_demand, (cycle_id,))
        demand_rows = {r[0]: r[1] for r in cur.fetchall()}
        cur.execute(sql_supply, (cycle_id,))
        supply_rows = {r[0]: r[1] for r in cur.fetchall()}

    gaps = []
    for category, constrained_qty in supply_rows.items():
        consensus = demand_rows.get(category, 0)
        gap_qty = consensus - (consensus - constrained_qty)
        if gap_qty > 0:
            severity = "critical" if gap_qty / max(consensus, 1) > 0.20 else "high"
            gaps.append({
                "cycle_id": cycle_id,
                "gap_type": "demand_supply_gap",
                "item_category": category,
                "gap_qty": gap_qty,
                "severity": severity,
                "resolution_status": "open",
                "gap_description": (
                    f"Supply constraint reduces available supply by {gap_qty:.0f} units "
                    f"({gap_qty / max(consensus, 1) * 100:.1f}% of consensus demand)."
                ),
                "resolution_options": [
                    "Source from alternate supplier (+15% cost premium)",
                    "Pre-build demand to prior month",
                    "Accept stockout risk on C-class items, protect A-class",
                ],
            })
    return gaps

def write_gaps(conn, gaps: list[dict]) -> int:
    """Insert gaps into fact_sop_gaps."""
    with conn.cursor() as cur:
        import json
        for g in gaps:
            cur.execute("""
                INSERT INTO fact_sop_gaps
                    (cycle_id, gap_type, item_category, gap_qty, severity,
                     resolution_status, gap_description, resolution_options)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                g["cycle_id"], g["gap_type"], g["item_category"],
                g["gap_qty"], g["severity"], g["resolution_status"],
                g["gap_description"], json.dumps(g.get("resolution_options", [])),
            ))
        conn.commit()
    return len(gaps)

def publish_approved_plan(
    conn,
    cycle_id: int,
    cycle_month: date,
    approved_by: str,
    plan_version: str,
) -> int:
    """
    After executive approval, write the agreed demand quantities to fact_sop_approved_plan.
    Uses consensus demand qty from demand review. Falls back to statistical forecast
    for categories without a consensus entry.
    """
    sql = """
        SELECT
            f.dmdunit AS item_no,
            f.loc,
            i.category AS item_category,
            f.startdate AS plan_month,
            COALESCE(dr.consensus_demand_qty / NULLIF(cat_total.cat_qty, 0), 1.0) * f.qty AS approved_qty,
            f.qty  AS statistical_qty,
            dr.commercial_override_qty AS override_qty
        FROM fact_external_forecast_monthly f
        JOIN dim_item i ON i.item_no = f.dmdunit
        LEFT JOIN fact_sop_demand_review dr
            ON dr.cycle_id = %s AND dr.item_category = i.category
        LEFT JOIN (
            SELECT item_category, SUM(statistical_forecast_qty) AS cat_qty
            FROM fact_sop_demand_review
            WHERE cycle_id = %s
            GROUP BY item_category
        ) cat_total ON cat_total.item_category = i.category
        WHERE f.startdate = %s AND f.model_id = 'external' AND f.lag = 0
    """
    plan_df = pd.read_sql(sql, conn, params=(cycle_id, cycle_id, cycle_month))

    with conn.cursor() as cur:
        rows = [
            (
                cycle_id, r.item_no, r.loc, r.item_category, r.plan_month,
                r.approved_qty, r.statistical_qty, r.override_qty,
                "consensus" if pd.notna(r.override_qty) else "statistical",
                approved_by, True, plan_version,
            )
            for _, r in plan_df.iterrows()
        ]
        cur.executemany("""
            INSERT INTO fact_sop_approved_plan
                (cycle_id, item_no, loc, item_category, plan_month, approved_qty,
                 statistical_qty, override_qty, source, approved_by, locked, plan_version)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (cycle_id, item_no, loc, plan_month) DO NOTHING
        """, rows)
        conn.commit()
    return len(rows)

def advance_cycle(conn, cycle_id: int, facilitated_by: str) -> str:
    """Advance cycle to next stage. Returns new stage name."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status FROM fact_sop_cycles WHERE cycle_id = %s",
            (cycle_id,)
        )
        current = cur.fetchone()[0]
        idx = STAGE_ORDER.index(current)
        if idx >= len(STAGE_ORDER) - 1:
            raise ValueError(f"Cycle {cycle_id} is already in final stage '{current}'")
        next_stage = STAGE_ORDER[idx + 1]
        timestamp_col = {
            "supply_review":  "demand_review_at",
            "pre_sop":        "supply_review_at",
            "executive_sop":  "pre_sop_at",
            "approved":       "exec_sop_at",
            "closed":         "approved_at",
        }.get(next_stage)
        if timestamp_col:
            cur.execute(
                f"UPDATE fact_sop_cycles SET status = %s, {timestamp_col} = NOW(), "
                f"facilitated_by = %s WHERE cycle_id = %s",
                (next_stage, facilitated_by, cycle_id)
            )
        else:
            cur.execute(
                "UPDATE fact_sop_cycles SET status = %s WHERE cycle_id = %s",
                (next_stage, cycle_id)
            )
        conn.commit()
    return next_stage

def run(action: str, cycle_month: date, **kwargs) -> None:
    with psycopg.connect(**get_db_params()) as conn:
        if action == "init":
            cycle_id = get_or_create_cycle(conn, cycle_month)
            df = generate_demand_review_summary(conn, cycle_id, cycle_month)
            df.attrs["cycle_month"] = cycle_month
            n = upsert_demand_review(conn, cycle_id, df)
            print(f"[sop] Initialized cycle {cycle_id} with {n} demand review rows")

        elif action == "gaps":
            cycle_id = kwargs["cycle_id"]
            gaps = compute_supply_demand_gaps(conn, cycle_id)
            n = write_gaps(conn, gaps)
            print(f"[sop] Computed {n} supply/demand gaps for cycle {cycle_id}")

        elif action == "approve":
            cycle_id   = kwargs["cycle_id"]
            approved_by = kwargs.get("approved_by", "admin")
            version    = kwargs.get("plan_version", f"sop-{cycle_month}")
            n = publish_approved_plan(conn, cycle_id, cycle_month, approved_by, version)
            advance_cycle(conn, cycle_id, approved_by)
            print(f"[sop] Published {n:,} approved plan rows, cycle advanced to 'approved'")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["init", "gaps", "approve"])
    p.add_argument("--month", required=True, help="YYYY-MM-DD (first of month)")
    p.add_argument("--cycle-id", type=int)
    p.add_argument("--approved-by", default="admin")
    p.add_argument("--version", default="latest")
    args = p.parse_args()
    run(
        args.action,
        date.fromisoformat(args.month),
        cycle_id=args.cycle_id,
        approved_by=args.approved_by,
        plan_version=args.version,
    )
```

### 5.2 Config: `config/sop_config.yaml`

```yaml
# S&OP cycle configuration
default_facilitator: "sop_facilitator"
demand_review_deadline_day: 7       # Day of month by which demand review must be submitted
supply_review_deadline_day: 14
pre_sop_deadline_day: 21
exec_sop_deadline_day: 28

# Gap analysis thresholds
critical_gap_threshold_pct: 20      # Gap > 20% of consensus demand → critical severity
high_gap_threshold_pct: 10          # Gap > 10% → high severity

# Approved plan publishing
approved_plan_model_id: "sop_approved"   # Stored as model_id in downstream systems
lock_approved_plan: true                 # Prevent modifications after approval

# Categories to include in S&OP
included_categories:                     # NULL means all categories
  - Beverages
  - Snacks
  - Dairy
  - Frozen
```

---

## 6. API Endpoints

### `GET /sop/cycles`

Returns list of all S&OP cycles with current status.

**Response:**
```json
{
  "cycles": [
    {
      "cycle_id": 12,
      "cycle_month": "2026-04-01",
      "status": "pre_sop",
      "facilitated_by": "jane.doe",
      "demand_review_at": "2026-04-07T16:30:00Z",
      "supply_review_at": "2026-04-14T17:00:00Z",
      "pre_sop_at": null,
      "approved_at": null,
      "open_gaps": 3
    }
  ]
}
```

---

### `GET /sop/cycles/{cycle_id}`

Returns full detail for a single cycle including demand review rows, supply constraints, and gaps.

**Response:**
```json
{
  "cycle_id": 12,
  "cycle_month": "2026-04-01",
  "status": "pre_sop",
  "demand_review": [
    {
      "item_category": "Beverages",
      "statistical_forecast_qty": 12500,
      "commercial_override_qty": 600,
      "consensus_demand_qty": 13100,
      "review_status": "approved"
    }
  ],
  "supply_constraints": [
    {
      "constraint_id": 45,
      "constraint_type": "supplier_capacity",
      "supplier_name": "ABC Trading",
      "item_category": "Beverages",
      "impact_qty": 2100,
      "impact_period_start": "2026-04-01",
      "impact_period_end": "2026-04-30",
      "mitigation_status": "open"
    }
  ],
  "gaps": [
    {
      "gap_id": 8,
      "gap_type": "demand_supply_gap",
      "item_category": "Beverages",
      "gap_qty": 2100,
      "gap_value": 252000.00,
      "severity": "critical",
      "resolution_status": "open"
    }
  ]
}
```

---

### `POST /sop/cycles/{cycle_id}/advance`

Advance cycle to next S&OP stage.

**Request body:**
```json
{ "advanced_by": "jane.doe", "notes": "Demand review complete. All categories approved." }
```

**Response:**
```json
{ "cycle_id": 12, "previous_stage": "demand_review", "new_stage": "supply_review" }
```

---

### `POST /sop/cycles/{cycle_id}/approve`

Executive approval — locks the plan and publishes to `fact_sop_approved_plan`.

**Request body:**
```json
{ "approved_by": "ceo_jsmith", "plan_version": "2026-04-sop-v1", "notes": "Approved with alternate supplier sourcing for Beverages gap." }
```

**Response:**
```json
{
  "cycle_id": 12,
  "status": "approved",
  "approved_at": "2026-04-28T14:00:00Z",
  "rows_published": 12847,
  "plan_version": "2026-04-sop-v1"
}
```

---

### `GET /sop/cycles/{cycle_id}/gaps`

Returns all gaps for the cycle with resolution status.

**Query params:** `severity=`, `resolution_status=open`

**Response:**
```json
{
  "cycle_id": 12,
  "open_gap_count": 3,
  "total_gap_value": 347500.00,
  "gaps": [
    {
      "gap_id": 8,
      "gap_type": "demand_supply_gap",
      "item_category": "Beverages",
      "gap_qty": 2100,
      "gap_value": 252000.00,
      "severity": "critical",
      "resolution_options": [
        "Source from alternate supplier (+15% cost premium)",
        "Pre-build demand to prior month",
        "Accept stockout risk on C-class items, protect A-class"
      ],
      "resolution_status": "open"
    }
  ]
}
```

---

### `GET /sop/approved-plan`

Returns the approved demand plan for a given cycle and optional item/location filter.

**Query params:** `cycle_id=12`, `item_no=100320`, `loc=1401-BULK`, `category=Beverages`

**Response:**
```json
{
  "cycle_id": 12,
  "plan_version": "2026-04-sop-v1",
  "approved_at": "2026-04-28T14:00:00Z",
  "rows": [
    {
      "item_no": "100320",
      "loc": "1401-BULK",
      "item_category": "Beverages",
      "plan_month": "2026-04-01",
      "approved_qty": 486,
      "statistical_qty": 450,
      "override_qty": 36,
      "source": "consensus",
      "locked": true
    }
  ]
}
```

---

## 7. Frontend Components

### 7.1 New "S&OP" Tab in Navigation Sidebar

```
┌─────────────────────────────────────────────────────────────────────┐
│  S&OP CYCLE: April 2026              Status: PRE-S&OP (Week 3)     │
├──────────────┬───────────────┬───────────────┬──────────────────────┤
│  DEMAND      │    SUPPLY     │   PRE-S&OP    │  EXECUTIVE S&OP      │
│  REVIEW      │    REVIEW     │   GAP         │  APPROVAL            │
│  ✓ COMPLETE  │  ✓ COMPLETE   │  ● ACTIVE     │  ○ PENDING           │
│  Apr 7       │  Apr 14       │  Apr 21       │  Apr 28              │
│  jane.doe    │  bob.supply   │  ---          │  CEO sign-off        │
└──────────────┴───────────────┴───────────────┴──────────────────────┘
│  DEMAND REVIEW TABLE                                                │
│  Category      Stat Fcst   Override  Consensus  Status   Notes      │
│  ──────────    ─────────   ────────  ─────────  ──────   ─────      │
│  Beverages      12,500      +600     13,100    ✓ Appr.  Easter+    │
│  Snacks          8,200      +150      8,350    ✓ Appr.  None       │
│  Dairy           4,100         0      4,100    ✓ Appr.  None       │
├─────────────────────────────────────────────────────────────────────┤
│  SUPPLY CONSTRAINTS                              [+ Add Constraint] │
│  Type              Supplier      Category   Impact Qty  Status      │
│  Supplier Capacity ABC Trading   Beverages  2,100 units OPEN        │
│  DC Capacity       —             All        —          RESOLVED     │
├─────────────────────────────────────────────────────────────────────┤
│  GAP ANALYSIS                    3 open gaps   $347,500 at risk     │
│  ⚠ CRITICAL: Beverages — 2,100 unit supply shortfall               │
│    Options: (A) Alt supplier +15% | (B) Pre-build | (C) Accept risk │
│    [Resolve]                                                        │
│  ⚠ HIGH: Frozen — $78,400 budget gap                               │
│    [Resolve]                                                        │
├─────────────────────────────────────────────────────────────────────┤
│  EXECUTIVE DASHBOARD (unlocked after pre-S&OP complete)            │
│  Total Plan Value: $1,574,200   Open Gaps: 3   Budget Headroom: 7% │
│                          [APPROVE PLAN]  (requires VP Supply Chain) │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Worked Example — April 2026 S&OP Cycle

### Week 1: Demand Review

| Category | Statistical Forecast | Commercial Override | Reason | Consensus Demand |
|---|---|---|---|---|
| Beverages | 12,500 units | +800 (Easter promo) | Promotional uplift 6.4% | — |
| Beverages | — | -200 (phase-out, SKU 100399) | End of life | — |
| Beverages | — | Net +600 | — | 13,100 units |
| Snacks | 8,200 units | +150 | New store opening 1401 | 8,350 units |
| Dairy | 4,100 units | 0 | — | 4,100 units |
| Frozen | 3,900 units | -300 | Competitor price increase expected | 3,600 units |

**Total consensus demand April:** 29,150 units | $1,574,200 estimated value

### Week 2: Supply Review

Supplier ABC Trading (30% of Beverages volume) flags capacity constraint: factory maintenance shutdown April 14–21. Maximum supply April: 11,000 units (of 13,100 demanded).

**Constraint entered:**
```
type: supplier_capacity
supplier: ABC Trading
category: Beverages
impact_qty: 2,100
impact_period: 2026-04-01 → 2026-04-30
description: Factory maintenance Apr 14-21. Max April supply = 11,000 units.
```

### Week 3: Pre-S&OP Gap Analysis

System computes gaps automatically:

```
Gap Analysis Results:
─────────────────────────────────────────────────────
Gap 1: DEMAND_SUPPLY_GAP — Beverages
  Consensus demand:    13,100 units
  Supply available:    11,000 units
  Gap:                  2,100 units
  Gap value:          $252,000  (at avg $120/unit)
  Severity:           CRITICAL (16% of consensus demand)

  Resolution Options Generated:
    A) Source 2,100 units from alternate supplier → +$31,500 (15% premium)
    B) Pre-build 1,000 units in March + accept 1,100 unit shortfall on C-class
    C) Accept stockout risk on C-class (340 units), protect A-class (1,760 units)

Gap 2: BUDGET_GAP — Beverages
  Committed orders:    $182,400
  Planned orders:      $47,500
  Total exposure:      $229,900
  Budget cap Q2:       $200,000
  Budget gap:          $29,900
  Severity:            HIGH
─────────────────────────────────────────────────────
Total open gaps: 2   Total at-risk value: $347,500
```

### Week 4: Executive S&OP

Executive team selects: **Option A** (alternate supplier) + pre-build 1,000 units in March.

Decision logged:
```
approved_by: ceo_jsmith
resolution: Source 2,100 units from alternate supplier at $31,500 premium.
            Pre-build 1,000 units in March.
            Final approved plan: 13,100 units at $157,200 total value.
```

`fact_sop_approved_plan` is populated with 12,847 rows. The approved plan becomes the demand signal for the planning engine, overriding the statistical forecast for April 2026.

---

## 9. Dependencies

| Dependency | Feature | Status |
|---|---|---|
| Statistical forecast | fact_external_forecast_monthly | Implemented |
| Commercial overrides | F2.3 — Forecast Override (spec) | Not yet implemented |
| Item cost (for gap value) | F4.1 — Financial Inventory Plan | Specified in feature_06_13 |
| ABC-XYZ classification | IPfeature11 | Implemented |
| Budget caps | F4.1 — Financial Inventory Plan | Specified in feature_06_13 |

---

## 10. Out of Scope

- Automated ERP sales order confirmation against approved plan
- Multi-currency S&OP (single currency USD assumed)
- Resource/capacity planning for manufacturing (production scheduling)
- Automated supply constraint sourcing from supplier APIs
- Version control / rollback of approved plans (current cycle only; historical cycles preserved read-only)
- Role-based access control (RBAC) enforcement at database level (application-level only in this phase)

---

## 11. Test Requirements

### Backend Unit Tests (`tests/unit/test_sop_cycle.py`)

- `test_stage_advance_order()` — advancing from demand_review → supply_review → pre_sop → executive_sop → approved → closed in correct order
- `test_stage_advance_final_stage_raises()` — advancing from 'closed' raises ValueError
- `test_compute_supply_demand_gaps_no_constraints()` — returns empty list when no constraints
- `test_compute_supply_demand_gaps_critical()` — gap > 20% consensus → severity=critical
- `test_compute_supply_demand_gaps_high()` — gap 10–20% → severity=high
- `test_compute_supply_demand_gaps_multi_category()` — only constrained categories generate gaps
- `test_gap_value_calculation()` — gap_qty × avg_unit_cost = gap_value
- `test_publish_approved_plan_source_consensus()` — source='consensus' when override present
- `test_publish_approved_plan_source_statistical()` — source='statistical' when no override

### Backend API Tests (`tests/api/test_sop.py`)

- `test_get_cycles_200()` — returns list with status and cycle_month
- `test_get_cycle_detail_200()` — returns demand_review + supply_constraints + gaps
- `test_get_cycle_404()` — returns 404 for non-existent cycle_id
- `test_advance_cycle_200()` — returns new stage name
- `test_advance_cycle_final_stage_400()` — returns 400 when cycle is closed
- `test_approve_cycle_200()` — sets status=approved, returns rows_published
- `test_get_gaps_open_only()` — resolution_status=open filter works
- `test_get_gaps_severity_filter()` — severity=critical filter works
- `test_get_approved_plan_200()` — returns locked rows with correct source
- `test_get_approved_plan_item_filter()` — item_no filter scopes results

### Frontend Tests (`src/tabs/__tests__/SopTab.test.tsx`)

- Cycle status timeline renders 4 stage milestones
- Demand review table renders category rows with correct column values
- Supply constraints list renders with mitigation_status badge
- Gap analysis shows critical gap in red with resolution options
- Approve button renders only when status=executive_sop
- Empty state renders when no cycles exist
