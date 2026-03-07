# Feature F2.3 — Consensus Forecasting & Planner Overrides

**Phase:** Evolution to Operations — Phase 2 (Demand Planning)
**Feature Number:** F2.3
**Status:** Design — Not Implemented
**Depends On:** F2.2 (Multi-Horizon Demand Plan / Quantile Forecasts), IPfeature3 (Safety Stock)

---

## 1. Problem Statement

### What Fails Today

Every forecast in Demand Studio is 100% model-driven. The statistical ML models (LightGBM/CatBoost/XGBoost) produce predictions based entirely on historical sales patterns. There is no mechanism for:

- A demand planner to adjust a forecast because of an upcoming promotion
- A sales manager to add volume for a new product launch not captured in history
- A commercial team to suppress demand for a product being phased out
- A supply chain director to lock a volume commitment for a capacity-constrained item

**Concrete failure — what happens today without this feature:**

Company runs a 40%-off promotion on Item 100320 (Bulk Cleaning Solution) in May 2026, starting May 15. The promotion is agreed with marketing 6 weeks in advance.

The ML model sees no signal for this — its training data shows normal May demand of ~450 units. The model predicts 450. The system orders inventory based on 450. The actual demand during the promotion runs 630 units (450 × 1.40).

Result: stockout on May 22 — 7 days before the promotion ends. Lost sales ≈ 180 units × $24 average selling price = $4,320 revenue lost. Customer satisfaction drops.

The planner "knew" about this promotion but had no way to tell the system.

### The Consensus Forecasting Process

```
Step 1: Statistical Baseline
    ML model generates P10/P50/P90 for all DFUs
    (from F2.2 fact_demand_plan)

Step 2: Commercial Override
    Demand planners, sales managers, or commercial analysts
    submit overrides via UI for events not in the model

Step 3: Approval Workflow
    Manager reviews overrides > threshold ($5K impact or > 20% lift)
    Approved overrides are locked

Step 4: Consensus Plan
    generate_consensus_plan.py merges statistical baseline
    with approved overrides → fact_consensus_plan

Step 5: Downstream consumption
    Replenishment engine, safety stock, MRP all read from
    fact_consensus_plan (not raw model output)
```

---

## 2. Override Types

| Override Type | Description | Math | Example |
|---|---|---|---|
| `PROMO` | Promotional lift over a date range | `base * multiplier` | 40% off sale: multiplier=1.40 |
| `LAUNCH` | New product ramp — manual qty by month | `additive_qty` | Month 1: 200, Month 2: 350, Month 3: 500 |
| `PHASE_OUT` | End-of-life demand decay | `base * decay_factor` | Last buy: multiplier=0.30 in final month |
| `MARKET_EVENT` | External market factor (competitor exit, regulatory change) | `base * multiplier` | Competitor discontinued: multiplier=1.25 |
| `CAPACITY_LOCK` | Agreed volume with supplier or customer | `locked_qty` (hard override) | Contractual volume: 800 units |
| `MANUAL` | Arbitrary planner judgment, no formula | `override_qty` (replaces base) | Planner sets 350, no reason required |

### Override Math

**Multiplier + additive (most common):**

```
consensus_qty = (base_statistical_qty * override_multiplier) + override_additive_qty
```

**Hard override (CAPACITY_LOCK, MANUAL with is_hard_override=TRUE):**

```
consensus_qty = override_qty  -- completely replaces statistical forecast
```

**Conflict resolution (two overrides on same DFU-month):**

Priority rules applied in order:
1. `CAPACITY_LOCK` wins over all other types (contractual commitment)
2. Higher `priority_rank` (1=highest) wins
3. Most recent `created_at` wins as tiebreaker
4. System logs a warning when conflict is detected and resolved

**Auto-expiry:**

Any override where `valid_to < current_plan_run_date` is automatically excluded. The override record is not deleted — it is left with `status='expired'` for audit purposes.

---

## 3. Input Data Required

### Available Today

| Source | Table | Data |
|---|---|---|
| Demand Plan (F2.2) | `fact_demand_plan` | Statistical P50 baseline per DFU per month |
| DFU dimension | `dim_dfu` | Item/location metadata for override UI |
| Safety stock | `fact_safety_stock_targets` | Downstream consumer of consensus plan |

### Missing — New or External

| Data | Source | Gap |
|---|---|---|
| Planner override submissions | New UI in this feature | Tables defined below |
| Approval workflow users | Identity provider / LDAP | User identity not currently in system |
| Promotion calendar | Marketing system / manual entry | Not currently ingested |
| Product lifecycle stage | PIM / manual flag | Not currently tracked in `dim_dfu` |
| Customer volume commitments | CRM / ERP contracts | Not currently ingested |

---

## 4. Data Model

### 4.1 New Table: `fact_forecast_overrides`

**Grain:** `item_no + loc + override_month + override_type + created_by + created_at`

One override per submission. Multiple overrides on the same DFU-month are allowed (resolved by conflict rules at consensus-plan generation time).

```sql
CREATE TABLE fact_forecast_overrides (
    override_id             BIGSERIAL       PRIMARY KEY,
    item_no                 VARCHAR(50)     NOT NULL,
    loc                     VARCHAR(50)     NOT NULL,
    override_month          DATE            NOT NULL,       -- First of month (YYYY-MM-01)
    override_type           VARCHAR(20)     NOT NULL,       -- PROMO, LAUNCH, PHASE_OUT, MARKET_EVENT, CAPACITY_LOCK, MANUAL
    override_qty            NUMERIC(12,2),                  -- Absolute quantity (for MANUAL, LAUNCH, CAPACITY_LOCK)
    override_multiplier     NUMERIC(6,4),                   -- Multiplicative factor (for PROMO, MARKET_EVENT, PHASE_OUT)
    override_additive_qty   NUMERIC(12,2)   DEFAULT 0,      -- Additive lift on top of multiplied base
    is_hard_override        BOOLEAN         NOT NULL DEFAULT FALSE,  -- If TRUE, completely replaces statistical qty
    override_reason         TEXT            NOT NULL,       -- Structured reason (selectable)
    override_note           TEXT,                           -- Free-text planner note
    created_by              VARCHAR(100)    NOT NULL,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    valid_from              DATE            NOT NULL,
    valid_to                DATE            NOT NULL,
    approved_by             VARCHAR(100),
    approved_at             TIMESTAMPTZ,
    rejected_by             VARCHAR(100),
    rejected_at             TIMESTAMPTZ,
    rejection_reason        TEXT,
    status                  VARCHAR(20)     NOT NULL DEFAULT 'draft',  -- draft, pending_approval, approved, rejected, expired, superseded
    requires_approval       BOOLEAN         NOT NULL DEFAULT TRUE,
    priority_rank           INTEGER         NOT NULL DEFAULT 5,     -- 1=highest priority in conflict resolution
    statistical_qty_at_creation NUMERIC(12,2),                      -- Snapshot of P50 when override was submitted
    estimated_impact_units  NUMERIC(12,2),                          -- computed at submission: override effect in units
    estimated_impact_value  NUMERIC(14,2),                          -- impact * avg unit cost
    currency                VARCHAR(3)      DEFAULT 'USD',
    expires_auto            BOOLEAN         NOT NULL DEFAULT TRUE,  -- Auto-expire when valid_to < plan_run_date
    plan_version_applied    VARCHAR(50),                            -- Which consensus plan version consumed this override
    parent_override_id      BIGINT          REFERENCES fact_forecast_overrides(override_id)  -- For amendment tracking
);

CREATE INDEX idx_override_item_loc_month
    ON fact_forecast_overrides (item_no, loc, override_month);

CREATE INDEX idx_override_status
    ON fact_forecast_overrides (status, override_month);

CREATE INDEX idx_override_created_by
    ON fact_forecast_overrides (created_by, created_at DESC);

CREATE INDEX idx_override_pending_approval
    ON fact_forecast_overrides (status, requires_approval)
    WHERE status = 'pending_approval';

CREATE INDEX idx_override_valid_dates
    ON fact_forecast_overrides (valid_from, valid_to);
```

### 4.2 New Table: `fact_consensus_plan`

**Grain:** `item_no + loc + plan_month + plan_version`

One row per DFU per month per plan version. This is the authoritative demand plan used by all downstream processes.

```sql
CREATE TABLE fact_consensus_plan (
    id                      BIGSERIAL       PRIMARY KEY,
    item_no                 VARCHAR(50)     NOT NULL,
    loc                     VARCHAR(50)     NOT NULL,
    plan_month              DATE            NOT NULL,
    plan_version            VARCHAR(50)     NOT NULL,
    statistical_qty         NUMERIC(12,2)   NOT NULL,   -- P50 from fact_demand_plan
    statistical_p10         NUMERIC(12,2),
    statistical_p90         NUMERIC(12,2),
    override_qty            NUMERIC(12,2)   DEFAULT 0,  -- Net override effect in units
    consensus_qty           NUMERIC(12,2)   NOT NULL,   -- Final: stat + override
    consensus_p10           NUMERIC(12,2),              -- Adjusted lower bound
    consensus_p90           NUMERIC(12,2),              -- Adjusted upper bound
    override_applied        BOOLEAN         NOT NULL DEFAULT FALSE,
    override_id             BIGINT          REFERENCES fact_forecast_overrides(override_id),
    override_type           VARCHAR(20),
    override_multiplier     NUMERIC(6,4),
    is_hard_override        BOOLEAN         DEFAULT FALSE,
    overrider               VARCHAR(100),
    approver                VARCHAR(100),
    uplift_pct              NUMERIC(8,4),               -- (consensus - statistical) / statistical * 100
    generated_at            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_consensus_plan UNIQUE (item_no, loc, plan_month, plan_version)
);

CREATE INDEX idx_consensus_plan_item_loc_month
    ON fact_consensus_plan (item_no, loc, plan_month);

CREATE INDEX idx_consensus_plan_version
    ON fact_consensus_plan (plan_version, plan_month);

CREATE INDEX idx_consensus_plan_overridden
    ON fact_consensus_plan (plan_version)
    WHERE override_applied = TRUE;
```

### 4.3 New Table: `fact_override_audit_log`

Complete audit trail. Immutable — rows are never updated, only inserted.

```sql
CREATE TABLE fact_override_audit_log (
    log_id          BIGSERIAL       PRIMARY KEY,
    override_id     BIGINT          NOT NULL REFERENCES fact_forecast_overrides(override_id),
    action          VARCHAR(30)     NOT NULL,   -- submitted, updated, approved, rejected, expired, applied, superseded
    performed_by    VARCHAR(100)    NOT NULL,
    performed_at    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    old_status      VARCHAR(20),
    new_status      VARCHAR(20),
    old_qty         NUMERIC(12,2),
    new_qty         NUMERIC(12,2),
    old_multiplier  NUMERIC(6,4),
    new_multiplier  NUMERIC(6,4),
    reason          TEXT,
    system_note     TEXT            -- Automated notes (e.g., 'auto-expired', 'conflict resolved by priority_rank')
);

CREATE INDEX idx_override_audit_override_id
    ON fact_override_audit_log (override_id, performed_at DESC);
```

---

## 5. Python Scripts

### 5.1 `scripts/generate_consensus_plan.py`

```python
"""
Generate the consensus demand plan by merging the statistical baseline
(fact_demand_plan P50) with approved planner overrides (fact_forecast_overrides).

Usage:
    uv run scripts/generate_consensus_plan.py \
        --plan-version 2026-04-01_production \
        --months-ahead 12 \
        --dry-run

Config: config/consensus_config.yaml
Output: fact_consensus_plan
"""

import yaml
import pandas as pd
import psycopg
from datetime import date
from common.db import get_db_params

OVERRIDE_PRIORITY = {
    "CAPACITY_LOCK": 1,
    "PROMO": 2,
    "LAUNCH": 2,
    "PHASE_OUT": 3,
    "MARKET_EVENT": 3,
    "MANUAL": 4,
}


def load_statistical_baseline(
    plan_version: str,
    conn: psycopg.Connection,
) -> pd.DataFrame:
    """
    Load P10/P50/P90 from fact_demand_plan for a given plan version.

    Returns DataFrame with columns:
        item_no, loc, plan_month, p10, p50, p90
    """
    sql = """
        SELECT
            item_no, loc, plan_month,
            MAX(CASE WHEN quantile = 0.10 THEN forecast_qty END) AS p10,
            MAX(CASE WHEN quantile = 0.50 THEN forecast_qty END) AS p50,
            MAX(CASE WHEN quantile = 0.90 THEN forecast_qty END) AS p90
        FROM fact_demand_plan
        WHERE plan_version = %s
        GROUP BY item_no, loc, plan_month
    """
    with conn.cursor() as cur:
        cur.execute(sql, (plan_version,))
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["item_no", "loc", "plan_month", "p10", "p50", "p90"])


def load_approved_overrides(
    plan_run_date: date,
    months: list[date],
    conn: psycopg.Connection,
) -> pd.DataFrame:
    """
    Load overrides that are approved and currently valid.
    Excludes expired overrides (valid_to < plan_run_date).
    Applies priority ranking for conflict detection.
    """
    sql = """
        SELECT
            override_id, item_no, loc, override_month,
            override_type, override_qty, override_multiplier,
            override_additive_qty, is_hard_override,
            priority_rank, created_at, approved_by, approved_at
        FROM fact_forecast_overrides
        WHERE status = 'approved'
          AND valid_from <= %s
          AND valid_to   >= %s
          AND override_month = ANY(%s)
        ORDER BY item_no, loc, override_month,
                 priority_rank ASC, created_at DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (plan_run_date, plan_run_date, months))
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=[
        "override_id", "item_no", "loc", "override_month",
        "override_type", "override_qty", "override_multiplier",
        "override_additive_qty", "is_hard_override",
        "priority_rank", "created_at", "approved_by", "approved_at"
    ])


def apply_override(
    statistical_qty: float,
    override_type: str,
    override_qty: float | None,
    override_multiplier: float | None,
    override_additive_qty: float,
    is_hard_override: bool,
) -> tuple[float, float]:
    """
    Apply a single override to the statistical baseline quantity.

    Args:
        statistical_qty: P50 from fact_demand_plan
        override_type: PROMO, LAUNCH, PHASE_OUT, MARKET_EVENT, CAPACITY_LOCK, MANUAL
        override_qty: Absolute quantity (for MANUAL, LAUNCH, CAPACITY_LOCK)
        override_multiplier: Multiplicative factor (for PROMO, MARKET_EVENT, PHASE_OUT)
        override_additive_qty: Additive lift (added after multiplier)
        is_hard_override: If True, replaces statistical qty entirely

    Returns:
        Tuple of (consensus_qty, override_delta_units)
    """
    if is_hard_override and override_qty is not None:
        consensus_qty = max(0.0, override_qty)
        delta = consensus_qty - statistical_qty
        return round(consensus_qty, 2), round(delta, 2)

    multiplier = override_multiplier if override_multiplier is not None else 1.0
    additive = override_additive_qty or 0.0
    consensus_qty = max(0.0, statistical_qty * multiplier + additive)
    delta = consensus_qty - statistical_qty
    return round(consensus_qty, 2), round(delta, 2)


def resolve_conflicts(
    overrides_for_dfu_month: pd.DataFrame,
) -> pd.Series:
    """
    Given multiple overrides for the same DFU-month, select the winner.
    Priority: CAPACITY_LOCK type > highest priority_rank > most recent created_at.
    Logs a warning row for each conflict resolved.
    """
    if len(overrides_for_dfu_month) == 1:
        return overrides_for_dfu_month.iloc[0]

    # Apply type priority
    overrides_for_dfu_month = overrides_for_dfu_month.copy()
    overrides_for_dfu_month["type_priority"] = overrides_for_dfu_month[
        "override_type"
    ].map(OVERRIDE_PRIORITY)
    winner = overrides_for_dfu_month.sort_values(
        ["type_priority", "priority_rank", "created_at"],
        ascending=[True, True, False],
    ).iloc[0]

    # Log conflict warning
    n_conflicts = len(overrides_for_dfu_month) - 1
    print(
        f"[CONFLICT] {winner['item_no']}@{winner['loc']} "
        f"{winner['override_month']}: {n_conflicts} override(s) superseded "
        f"by {winner['override_type']} (override_id={winner['override_id']})"
    )
    return winner


def generate_consensus_plan(
    plan_version: str,
    plan_run_date: date,
    months_ahead: int,
    dry_run: bool = False,
) -> dict:
    """
    Main entry point: merge statistical baseline with approved overrides.

    Returns dict with summary: total_rows, overridden_rows, total_override_impact_units
    """
    pass  # Full implementation follows the load → merge → write pattern
```

### 5.2 `config/consensus_config.yaml`

```yaml
consensus_plan:
  approval_required_threshold_units: 100     # Overrides > 100 units require approval
  approval_required_threshold_pct: 0.20      # Or > 20% uplift require approval
  approval_required_threshold_value: 5000    # Or > $5,000 estimated impact

  conflict_resolution:
    priority_order:
      - CAPACITY_LOCK
      - PROMO
      - LAUNCH
      - PHASE_OUT
      - MARKET_EVENT
      - MANUAL
    log_all_conflicts: true

  auto_expiry:
    enabled: true
    status_on_expiry: expired          # Set status to 'expired' (do not delete)

  consensus_plan_output:
    adjust_p10_p90_proportionally: true   # Scale P10/P90 by same ratio as P50 override
    floor_qty: 0                          # Never allow negative consensus qty
    ceiling_multiplier: 5.0              # Override cannot multiply base by more than 5x

  notifications:
    notify_on_submission: true
    notify_on_approval: true
    notify_on_rejection: true
    approver_roles: ["demand_manager", "supply_chain_director"]
```

---

## 6. API Endpoints

### 6.1 `GET /forecast/overrides`

List planner overrides with filters.

**Parameters:**
- `item_no` (optional)
- `loc` (optional)
- `status` (optional): `draft | pending_approval | approved | rejected | expired`
- `override_type` (optional)
- `month_from` (optional): DATE
- `month_to` (optional): DATE
- `page` (default: 1), `page_size` (default: 50)

**Response:**

```json
{
  "total": 3,
  "page": 1,
  "overrides": [
    {
      "override_id": 42,
      "item_no": "100320",
      "loc": "1401-BULK",
      "override_month": "2026-05-01",
      "override_type": "PROMO",
      "override_multiplier": 1.40,
      "override_additive_qty": 0,
      "is_hard_override": false,
      "override_reason": "May promotion — 40% off campaign, May 15-31",
      "statistical_qty_at_creation": 450.00,
      "estimated_impact_units": 180.00,
      "estimated_impact_value": 4320.00,
      "status": "approved",
      "created_by": "jane.smith@company.com",
      "created_at": "2026-04-02T09:14:22Z",
      "approved_by": "mike.jones@company.com",
      "approved_at": "2026-04-02T14:03:01Z",
      "valid_from": "2026-05-01",
      "valid_to": "2026-05-31"
    }
  ]
}
```

### 6.2 `POST /forecast/overrides`

Submit a new planner override.

**Request body:**

```json
{
  "item_no": "100320",
  "loc": "1401-BULK",
  "override_month": "2026-05-01",
  "override_type": "PROMO",
  "override_multiplier": 1.40,
  "override_additive_qty": 0,
  "is_hard_override": false,
  "override_reason": "May promotion — 40% off campaign, May 15-31",
  "override_note": "Marketing confirmed with customer service. Briefing doc: mkt-2026-0412.pdf",
  "valid_from": "2026-05-01",
  "valid_to": "2026-05-31",
  "priority_rank": 2
}
```

**Response (201 Created):**

```json
{
  "override_id": 42,
  "status": "pending_approval",
  "requires_approval": true,
  "estimated_impact_units": 180.00,
  "estimated_impact_value": 4320.00,
  "message": "Override submitted. Pending approval from demand_manager role."
}
```

**Validation rules enforced server-side:**
- `override_month` must be >= today (cannot retroactively override past months)
- `override_multiplier` must be within [0.10, 5.00]
- `override_type` must be in the allowed set
- If `estimated_impact_value > 5000`, automatically set `requires_approval=true`

### 6.3 `PUT /forecast/overrides/{id}/approve`

Manager approves a pending override.

**Request body:**

```json
{
  "approved_by": "mike.jones@company.com",
  "approval_note": "Confirmed with marketing director. Launch confirmed."
}
```

**Response:**

```json
{
  "override_id": 42,
  "status": "approved",
  "approved_by": "mike.jones@company.com",
  "approved_at": "2026-04-02T14:03:01Z"
}
```

### 6.4 `PUT /forecast/overrides/{id}/reject`

**Request body:**

```json
{
  "rejected_by": "mike.jones@company.com",
  "rejection_reason": "Promotion cancelled — see email from SVP Sales 2026-04-02"
}
```

**Response:** `{ "override_id": 42, "status": "rejected" }`

### 6.5 `DELETE /forecast/overrides/{id}`

Soft-delete: sets status to `superseded`. Does not physically delete. Only callable by `created_by` or manager role. Returns `{ "override_id": 42, "status": "superseded" }`.

### 6.6 `GET /forecast/consensus-plan`

Retrieve the current consensus plan (merged statistical + overrides).

**Parameters:** `item_no`, `loc`, `plan_version`, `month_from`, `month_to`

**Response:**

```json
{
  "plan_version": "2026-04-01_production",
  "item_no": "100320",
  "loc": "1401-BULK",
  "months": [
    {
      "plan_month": "2026-05-01",
      "statistical_qty": 450.00,
      "consensus_qty": 630.00,
      "override_applied": true,
      "override_type": "PROMO",
      "override_multiplier": 1.40,
      "uplift_pct": 40.00,
      "overrider": "jane.smith@company.com",
      "approver": "mike.jones@company.com",
      "consensus_p10": 448.00,
      "consensus_p90": 812.00
    },
    {
      "plan_month": "2026-06-01",
      "statistical_qty": 460.00,
      "consensus_qty": 460.00,
      "override_applied": false,
      "uplift_pct": 0.0
    }
  ]
}
```

### 6.7 `GET /forecast/overrides/summary`

Portfolio-level summary of overrides for the current plan cycle.

**Response:**

```json
{
  "plan_version": "2026-04-01_production",
  "total_overrides": 47,
  "by_status": {
    "pending_approval": 12,
    "approved": 31,
    "rejected": 4
  },
  "by_type": {
    "PROMO": 18,
    "MANUAL": 14,
    "LAUNCH": 8,
    "PHASE_OUT": 5,
    "MARKET_EVENT": 2
  },
  "total_uplift_units": 3240.0,
  "total_uplift_value": 77760.0,
  "dfu_count_overridden": 38
}
```

---

## 7. Frontend Components

### 7.1 Override Button on Demand Plan Panel

Located in: `frontend/src/tabs/inv-planning/DemandPlanPanel.tsx`

An "Override" button appears on each month row in the demand plan table. Clicking opens the Override Modal.

```
┌────────────────────────────────────────────────────────────────┐
│  DEMAND PLAN — Item 100320  Loc 1401-BULK                      │
│  [Version: 2026-04-01_production ▼]                            │
├─────────┬──────────┬──────────┬────────────┬───────────────────┤
│ Month   │ Stat P50 │ Override │ Consensus  │ Action            │
├─────────┼──────────┼──────────┼────────────┼───────────────────┤
│ Apr '26 │ 452      │ —        │ 452        │ [Override]        │
│ May '26 │ 450      │ +180 🟠  │ 630        │ [View Override]   │
│ Jun '26 │ 460      │ —        │ 460        │ [Override]        │
└─────────┴──────────┴──────────┴────────────┴───────────────────┘
```

Orange highlight on rows with an active override. Override delta shown in orange.

### 7.2 Override Modal

```
┌──────────────────────────────────────────────────────────────────┐
│  ADD OVERRIDE — Item 100320  Loc 1401-BULK  May 2026             │
│                                                             [✕]  │
├──────────────────────────────────────────────────────────────────┤
│  Override Type:  [PROMO ▼]                                       │
│                                                                  │
│  Statistical Forecast (P50):    450 units                        │
│                                                                  │
│  Multiplier:   [  1.40  ]    Additive Qty:  [   0   ]           │
│  ──────────────────────────────────────────────────────          │
│  Projected Consensus Qty:    630 units  (+180 / +40%)           │
│  Est. Impact Value:          $4,320                              │
│                                                                  │
│  Reason:  [May promotion — 40% off campaign ____________________]│
│                                                                  │
│  Note:    [Marketing confirmed. Briefing doc: mkt-2026-0412 ___] │
│                                                                  │
│  Valid From:  [2026-05-01]    Valid To:  [2026-05-31]           │
│                                                                  │
│  ⚠️  This override exceeds $5,000 and requires manager approval. │
│                                                                  │
│                    [Cancel]     [Submit Override]                │
└──────────────────────────────────────────────────────────────────┘
```

Consensus qty preview updates live as the planner types multiplier/additive values.

### 7.3 Override Queue Panel

Separate panel in Inv. Planning: "Override Queue" showing all pending approvals.

```
┌───────────────────────────────────────────────────────────────────┐
│  OVERRIDE QUEUE                [Filter: Pending ▼] [My Items ▼]   │
│  12 pending approval  •  31 approved  •  4 rejected               │
├────────┬──────────┬──────────┬────────┬──────────┬───────────────┤
│ Item   │ Loc      │ Month    │ Type   │ Impact   │ Actions       │
├────────┼──────────┼──────────┼────────┼──────────┼───────────────┤
│ 100320 │ 1401-BLK │ May 2026 │ PROMO  │ +$4,320  │ [✓ Approve]  │
│        │          │          │        │          │ [✗ Reject]   │
├────────┼──────────┼──────────┼────────┼──────────┼───────────────┤
│ 204771 │ 2203-STD │ Jun 2026 │ LAUNCH │ +$12,800 │ [✓ Approve]  │
│        │          │          │        │          │ [✗ Reject]   │
└────────┴──────────┴──────────┴────────┴──────────┴───────────────┘
```

### 7.4 Fan Chart Override Visualization

On the demand plan fan chart (ECharts), approved overrides appear as:
- Vertical dashed orange line at override month
- Orange annotation label: "PROMO +40%"
- The fan chart bands (P10/P90) scale proportionally: if P50 is multiplied by 1.40, P10 and P90 are also multiplied by 1.40 (configurable via `adjust_p10_p90_proportionally` in config)

---

## 8. Worked Example: End-to-End

### Scenario

Item 100320 (Bulk Cleaning Solution), Loc 1401-BULK.
May 2026 promotion: 40% off, May 15-31. Agreed by marketing on April 2.

**Step 1: Statistical Baseline (from F2.2)**

```
fact_demand_plan for May 2026, plan_version = '2026-04-01_production':
  P10 = 320.00
  P50 = 450.00
  P90 = 580.00
```

**Step 2: Planner Submits Override**

Jane Smith submits at 9:14 AM on April 2:

```json
{
  "override_type": "PROMO",
  "override_multiplier": 1.40,
  "valid_from": "2026-05-01",
  "valid_to": "2026-05-31"
}
```

System computes estimated impact:
```
estimated_impact_units = 450 * (1.40 - 1.0) = 180 units
estimated_impact_value = 180 * $24.00 (avg unit cost) = $4,320
```

Since $4,320 < $5,000 threshold... BUT override_multiplier drives > 20% uplift (40%>20%), so `requires_approval = true`. Status → `pending_approval`.

**Step 3: Manager Approves**

Mike Jones approves at 2:03 PM. Status → `approved`. Audit log row inserted.

**Step 4: Consensus Plan Generation**

`generate_consensus_plan.py` runs nightly (or on-demand):

```
Input:
  statistical_qty (P50) = 450.00
  override_multiplier   = 1.40
  is_hard_override      = FALSE

Computation:
  consensus_qty = 450.00 * 1.40 + 0 = 630.00
  override_qty  = 630.00 - 450.00 = 180.00
  uplift_pct    = 180 / 450 * 100 = 40.00%

  consensus_p10 = 320.00 * 1.40 = 448.00
  consensus_p90 = 580.00 * 1.40 = 812.00
```

`fact_consensus_plan` row inserted:

```
item_no          = '100320'
loc              = '1401-BULK'
plan_month       = 2026-05-01
plan_version     = '2026-04-01_production'
statistical_qty  = 450.00
override_qty     = 180.00
consensus_qty    = 630.00
override_applied = TRUE
override_type    = 'PROMO'
override_id      = 42
uplift_pct       = 40.00
```

**Step 5: Downstream Impact**

Replenishment engine reads `consensus_qty = 630` instead of `statistical_qty = 450`.

Planned order for May:
```
Target stock = consensus_qty + safety_stock = 630 + 186 = 816 units
Current on-hand (Apr 30) = 300 units
Open PO arriving = 200 units (from March order)
Net planned order = 816 - 300 - 200 = 316 units

Without override:
  Net planned order = (450 + 186) - 300 - 200 = 136 units
  Stockout on May 22 (as described in problem statement)

With override:
  Net planned order = 316 units ordered → arrives by May 10
  No stockout during promotion
```

**Step 6: Conflict Resolution Example**

Suppose both Jane (Sales) and Tom (Supply Chain) submit overrides for the same DFU-month:

```
Override A (Jane): PROMO, multiplier=1.40, priority_rank=2
Override B (Tom):  MANUAL, override_qty=500, priority_rank=5
```

Conflict resolution:
1. PROMO has type_priority=2, MANUAL has type_priority=4
2. PROMO wins. Override B status → `superseded`.
3. System logs: "[CONFLICT] 100320@1401-BULK 2026-05-01: 1 override superseded by PROMO (override_id=42)"

---

## 9. Dependencies

| Dependency | Type | Status |
|---|---|---|
| F2.2 — Quantile Forecasts | Hard | Design |
| IPfeature3 — Safety Stock | Soft (consumer) | Implemented |
| User identity / roles | Hard | Not in system (need auth layer) |
| Notification system (email/Slack) | Soft | Not in system |

---

## 10. Out of Scope

- Integration with promotional planning systems (SAP Trade Promotion Management, etc.)
- Collaborative planning with customer VMI (Vendor Managed Inventory)
- AI-assisted override suggestion ("Based on past promos, +38% is more accurate than +40%")
- Real-time push notifications (WebSocket alerts for approval events)
- Multi-level approval chains (this spec supports single-level only)
- Override version history diff viewer (which fields changed across override amendments)

---

## 11. Makefile Targets

```makefile
consensus-schema:
    @psql $$(uv run python -c "from common.db import get_db_params; ...") \
        -f sql/040_create_consensus_plan.sql

consensus-generate:
    uv run scripts/generate_consensus_plan.py \
        --plan-version $(VERSION) \
        --months-ahead 12

consensus-generate-dry:
    uv run scripts/generate_consensus_plan.py \
        --plan-version $(VERSION) \
        --months-ahead 12 \
        --dry-run

consensus-all:
    make consensus-schema && make consensus-generate
```

---

## 12. Test Requirements

### Backend Unit Tests (`tests/unit/test_consensus_plan.py`)

- `test_apply_override_promo_multiplier`: `apply_override(450, 'PROMO', None, 1.40, 0, False)` → `(630.0, 180.0)`
- `test_apply_override_hard_override`: `apply_override(450, 'MANUAL', 500, None, 0, True)` → `(500.0, 50.0)`
- `test_apply_override_phase_out`: multiplier=0.30 → `(135.0, -315.0)`
- `test_apply_override_floor_at_zero`: multiplier=0.0 → `(0.0, -450.0)`
- `test_resolve_conflicts_capacity_lock_wins`: CAPACITY_LOCK beats PROMO regardless of priority_rank
- `test_resolve_conflicts_priority_rank_tiebreak`: PROMO priority_rank=1 beats PROMO priority_rank=3
- `test_resolve_conflicts_recency_tiebreak`: Same type + same rank → most recent created_at wins
- `test_auto_expiry_excludes_past_valid_to`: Override with valid_to=yesterday not included
- `test_p10_p90_scaled_proportionally`: If P50 multiplied by 1.40, P10 and P90 also scaled by 1.40

### Backend API Tests (`tests/api/test_consensus_plan.py`)

- `test_post_override_requires_approval_above_threshold`: $6,000 impact → `requires_approval=true`
- `test_post_override_invalid_type_422`: Unknown override_type returns 422
- `test_post_override_past_month_422`: override_month < today returns 422
- `test_approve_override_sets_status_approved`: PUT approve changes status
- `test_reject_override_sets_status_rejected`: PUT reject with reason
- `test_get_consensus_plan_shows_override_applied`: Response includes `override_applied=true`, `uplift_pct=40.0`
- `test_get_overrides_filter_by_status`: Filter by `status=pending_approval` returns only pending rows
- `test_get_override_summary_counts`: Summary totals match mock data

### Frontend Tests (`frontend/src/tabs/__tests__/ConsensusOverride.test.tsx`)

- Override modal opens on click
- Consensus qty preview updates when multiplier input changes
- Warning badge shown when impact > $5,000
- Override type selector populates correct form fields (multiplier vs additive)
- Override queue renders pending items with Approve/Reject buttons
