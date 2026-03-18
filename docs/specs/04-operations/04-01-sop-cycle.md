# Feature F4.2 — Sales & Operations Planning (S&OP) Cycle Management

**Phase:** 4 — Evolution to Operations
**Feature Number:** F4.2
**Status:** Implemented
**Author:** Supply Chain Systems Architecture
**Date:** 2026-03-06 (spec), updated 2026-03-17 (implementation alignment)

---

## 1. Overview

The S&OP module provides a structured monthly **Sales & Operations Planning** process within Demand Studio. Each cycle progresses through a 6-stage state machine, capturing demand review inputs, supply constraints, supply-demand gap analysis, executive approval, and a locked approved plan that becomes the authoritative demand signal for downstream planning engines.

### What it Solves

- **Disconnected functional views** — demand, supply, and finance previously operated from separate data sources with no reconciliation point.
- **No version control on the plan** — there was no record of what was approved vs what changed and why.
- **Supply gaps discovered at execution** — shortfalls surfaced days before promotional windows instead of weeks earlier in a structured review.
- **No approved plan published to planning engine** — statistical forecasts and commercial overrides coexisted without a clear hierarchy.

---

## 2. Stage Machine

Each S&OP cycle progresses through 6 stages in fixed order:

```
demand_review → supply_review → pre_sop → executive_sop → approved → closed
```

| Stage | Week | Responsible | Output |
|---|---|---|---|
| `demand_review` | 1 | Demand Planner | `fact_sop_demand_review` rows (statistical + consensus demand) |
| `supply_review` | 2 | Supply Planner | `fact_sop_supply_constraints` rows |
| `pre_sop` | 3 | S&OP Facilitator | `fact_sop_gaps` rows (demand vs supply gap analysis) |
| `executive_sop` | 4 | VP Supply Chain | Gap resolutions approved, plan finalized |
| `approved` | — | CEO / CFO | `fact_sop_approved_plan` populated and locked |
| `closed` | — | Automated | Month-end actuals loaded, performance vs plan |

Stage transitions are performed via `POST /sop/cycles/{cycle_id}/advance` (auth required). Executive approval uses `POST /sop/cycles/{cycle_id}/approve` which directly sets status to `approved`.

```
  WEEK 1           WEEK 2           WEEK 3           WEEK 4
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐
│  DEMAND     │  │  SUPPLY     │  │  PRE-S&OP   │  │  EXECUTIVE   │
│  REVIEW     │→ │  REVIEW     │→ │  GAP        │→ │  S&OP        │
│             │  │             │  │  ANALYSIS   │  │  APPROVAL    │
│ Statistical │  │ Supplier    │  │ Demand vs   │  │ CEO/CFO/VP   │
│ Forecast    │  │ Capacity    │  │ Supply Gap  │  │ Supply Chain │
│ Consensus   │  │ Constraints │  │ Escalations │  │ Lock Plan    │
└─────────────┘  └─────────────┘  └─────────────┘  └──────────────┘
       │                │                │                 │
       ▼                ▼                ▼                 ▼
 fact_sop_demand  fact_sop_supply  fact_sop_gaps    fact_sop_approved
 _review          _constraints                      _plan
```

---

## 3. Database Schema

**DDL:** `sql/056_create_sop_module.sql`

### 3.1 `fact_sop_cycles`

**Grain:** `cycle_id` (one row per monthly S&OP cycle)

| Column | Type | Notes |
|---|---|---|
| `cycle_id` | `BIGSERIAL PK` | Auto-increment |
| `cycle_month` | `DATE NOT NULL UNIQUE` | First of planning month (e.g. `2026-05-01`) |
| `status` | `VARCHAR(30)` | Current stage. Default `'demand_review'` |
| `demand_plan_version` | `VARCHAR(50)` | Label for the demand plan produced |
| `supply_plan_version` | `VARCHAR(50)` | Label for the supply plan produced |
| `approved_plan_version` | `VARCHAR(50)` | Set on executive approval |
| `facilitated_by` | `VARCHAR(100)` | S&OP facilitator user |
| `approved_by` | `VARCHAR(100)` | Executive who approved |
| `demand_review_at` | `TIMESTAMPTZ` | Stage completion timestamp |
| `supply_review_at` | `TIMESTAMPTZ` | Stage completion timestamp |
| `pre_sop_at` | `TIMESTAMPTZ` | Stage completion timestamp |
| `executive_sop_at` | `TIMESTAMPTZ` | Stage completion timestamp |
| `approved_at` | `TIMESTAMPTZ` | Approval timestamp |
| `run_by` | `VARCHAR(100)` | Script runner |
| `created_at` | `TIMESTAMPTZ` | Default `NOW()` |
| `updated_at` | `TIMESTAMPTZ` | Default `NOW()` |

**Index:** `idx_sop_cycles_month` on `(cycle_month DESC)`

### 3.2 `fact_sop_demand_review`

**Grain:** `cycle_id + item_category`

| Column | Type | Notes |
|---|---|---|
| `id` | `BIGSERIAL PK` | |
| `cycle_id` | `BIGINT FK` | References `fact_sop_cycles` |
| `item_category` | `VARCHAR(100)` | |
| `statistical_demand_qty` | `NUMERIC(14,2)` | |
| `commercial_demand_qty` | `NUMERIC(14,2)` | |
| `consensus_demand_qty` | `NUMERIC(14,2)` | |
| `statistical_demand_val` | `NUMERIC(16,2)` | |
| `commercial_demand_val` | `NUMERIC(16,2)` | |
| `consensus_demand_val` | `NUMERIC(16,2)` | |
| `review_status` | `VARCHAR(20)` | Default `'pending'` |

**Unique constraint:** `(cycle_id, item_category)`

### 3.3 `fact_sop_supply_constraints`

**Grain:** `constraint_id`

| Column | Type | Notes |
|---|---|---|
| `constraint_id` | `BIGSERIAL PK` | |
| `cycle_id` | `BIGINT FK` | References `fact_sop_cycles` |
| `constraint_type` | `VARCHAR(50)` | e.g. `supplier_capacity`, `dc_capacity`, `lead_time_change` |
| `supplier_id` | `VARCHAR(50)` | |
| `impact_qty` | `NUMERIC(14,2)` | Units unavailable due to constraint |
| `impact_period` | `DATE` | |
| `mitigation_status` | `VARCHAR(30)` | Default `'open'`. Values: `open`, `in_progress`, `resolved`, `accepted` |

**Index:** `idx_sop_constraints_cycle` on `(cycle_id, mitigation_status)`

### 3.4 `fact_sop_gaps`

**Grain:** `gap_id`

| Column | Type | Notes |
|---|---|---|
| `gap_id` | `BIGSERIAL PK` | |
| `cycle_id` | `BIGINT FK` | References `fact_sop_cycles` |
| `gap_type` | `VARCHAR(50)` | e.g. `demand_supply_gap`, `budget_gap` |
| `gap_qty` | `NUMERIC(14,2)` | |
| `gap_value` | `NUMERIC(16,2)` | Financial value of the gap |
| `severity` | `VARCHAR(20)` | Default `'medium'`. Values: `critical`, `high`, `medium`, `low` |
| `resolution_options` | `JSONB` | Array of resolution text options |
| `resolution_status` | `VARCHAR(20)` | Default `'open'`. Values: `open`, `mitigated`, `accepted`, `escalated`, `resolved` |

**Index:** `idx_sop_gaps_cycle` on `(cycle_id, severity, resolution_status)`

### 3.5 `fact_sop_approved_plan`

**Grain:** `cycle_id + item_no + loc + plan_month`

| Column | Type | Notes |
|---|---|---|
| `id` | `BIGSERIAL PK` | |
| `cycle_id` | `BIGINT FK` | References `fact_sop_cycles` |
| `item_no` | `VARCHAR(50)` | |
| `loc` | `VARCHAR(50)` | |
| `plan_month` | `DATE` | |
| `approved_qty` | `NUMERIC(12,2)` | The locked authoritative demand signal |
| `statistical_qty` | `NUMERIC(12,2)` | Original statistical forecast for comparison |
| `override_qty` | `NUMERIC(12,2)` | Commercial override component |
| `source` | `VARCHAR(30)` | Default `'consensus'`. Values: `consensus`, `statistical`, `commercial_override`, `sop_adjusted` |
| `locked` | `BOOLEAN` | Default `TRUE`. Once locked, cannot be modified without cycle re-opening |

**Unique constraint:** `(cycle_id, item_no, loc, plan_month)`
**Index:** `idx_sop_approved_item_loc` on `(item_no, loc, plan_month)`

---

## 4. API Endpoints

**Router:** `api/routers/sop.py` (tag: `sop`)
**Auth:** Mutation endpoints (`advance`, `approve`) require `X-API-Key` header when `API_KEY` env var is set.
**Connection pattern:** Uses `get_conn()` directly (same as all inv_planning routers).

### `GET /sop/cycles`

List all S&OP cycles with status. Paginated.

**Query params:** `page` (default 1), `page_size` (default 20, max 100)

**Response:**
```json
{
  "total": 5,
  "page": 1,
  "cycles": [
    {
      "cycle_id": 12,
      "cycle_month": "2026-04-01",
      "current_stage": "pre_sop",
      "facilitated_by": "jane.doe",
      "approved_by": null,
      "approved_plan_version": null,
      "created_at": "2026-03-01T10:00:00",
      "updated_at": "2026-03-15T14:30:00"
    }
  ]
}
```

### `GET /sop/cycles/{cycle_id}`

Full cycle detail including demand review rows and supply constraints.

**Response:** Same as cycle fields above, plus:
- `demand_review` — array of `{item_category, statistical_demand_qty, commercial_demand_qty, consensus_demand_qty, review_status}`
- `supply_constraints` — array of `{constraint_type, supplier_id, impact_qty, impact_period, mitigation_status}`

Returns 404 if cycle not found.

### `POST /sop/cycles/{cycle_id}/advance`

Advance cycle to the next stage in the state machine. Auth required.

**Request body:**
```json
{ "facilitated_by": "jane.doe", "notes": "Demand review complete." }
```

**Response:**
```json
{ "cycle_id": 12, "previous_status": "demand_review", "new_status": "supply_review" }
```

Returns 400 if cycle is already at the final stage (`closed`). Returns 404 if cycle not found.

### `POST /sop/cycles/{cycle_id}/approve`

Executive approval — sets status to `approved`, records approver and plan version. Auth required.

**Request body:**
```json
{ "approved_by": "ceo_jsmith", "plan_version": "v2026-04-28" }
```

**Response:**
```json
{
  "cycle_id": 12,
  "status": "approved",
  "approved_by": "ceo_jsmith",
  "plan_version": "v2026-04-28"
}
```

Returns 404 if cycle not found.

### `GET /sop/cycles/{cycle_id}/gaps`

Supply-demand gap analysis for a cycle. Ordered by severity (critical first).

**Query params:** `severity` (optional filter), `resolution_status` (default `"open"`)

**Response:**
```json
{
  "cycle_id": 12,
  "gaps": [
    {
      "gap_id": 8,
      "gap_type": "demand_supply_gap",
      "gap_qty": 2100.00,
      "gap_value": 252000.00,
      "severity": "critical",
      "resolution_options": ["Source from alternate supplier", "Pre-build"],
      "resolution_status": "open"
    }
  ]
}
```

### `GET /sop/approved-plan`

Locked approved demand from the approved S&OP plan. Paginated with optional filters.

**Query params:** `cycle_id`, `item_no`, `loc`, `category`, `page` (default 1), `page_size` (default 50, max 200)

**Response:**
```json
{
  "total": 12847,
  "page": 1,
  "approved_plan": [
    {
      "cycle_id": 12,
      "item_no": "100320",
      "loc": "1401-BULK",
      "plan_month": "2026-04-01",
      "approved_qty": 486.00,
      "statistical_qty": 450.00,
      "override_qty": 36.00,
      "source": "consensus",
      "locked": true
    }
  ]
}
```

---

## 5. Configuration

**File:** `config/sop_config.yaml`

```yaml
sop:
  planning_horizon_months: 12
  demand_review_day: 5       # 5th of the month before
  supply_review_day: 10
  pre_sop_day: 15
  executive_sop_day: 20      # Final sign-off day

  stages:
    - demand_review
    - supply_review
    - pre_sop
    - executive_sop
    - approved
    - closed

  demand_baseline_model: champion
  supply_gap_alert_pct: 0.10          # Alert when supply < 90% of consensus demand

  approvers:                           # Informational only (not enforced server-side)
    demand_review: demand_planner
    supply_review: supply_planner
    pre_sop: supply_chain_manager
    executive_sop: vp_supply_chain

scheduler:
  job_type: run_sop_cycle
  cron: "0 7 1 * *"   # Monthly, 7am on the 1st
```

---

## 6. CLI Script

**File:** `scripts/run_sop_cycle.py`

The script provides 3 actions for managing S&OP cycles from the command line:

| Action | Description | Required args |
|---|---|---|
| `create` | Create a new S&OP cycle for a given month | `--cycle-month YYYY-MM-DD` |
| `advance` | Advance an existing cycle to the next stage | `--cycle-id N` |
| `populate-demand` | Populate demand review + supply constraints | `--cycle-id N` |

**Usage:**
```bash
uv run python scripts/run_sop_cycle.py --action create --cycle-month 2026-05-01
uv run python scripts/run_sop_cycle.py --action advance --cycle-id 1
uv run python scripts/run_sop_cycle.py --action populate-demand --cycle-id 1
uv run python scripts/run_sop_cycle.py --dry-run --action create --cycle-month 2026-05-01
```

### Key functions

- `create_sop_cycle(conn, cycle_month, cycle_name, cfg, created_by)` — inserts into `fact_sop_cycles` with computed scheduled stage dates (based on config day-of-month in prior month)
- `advance_sop_cycle(conn, cycle_id, performed_by, notes)` — advances to next stage, sets stage-specific completion timestamp columns
- `populate_demand_review(conn, cycle_id, cycle_month, horizon_months)` — reads `fact_external_forecast_monthly` (model_id=`'champion'`) and inserts into `fact_sop_demand_review`
- `populate_supply_constraints(conn, cycle_id)` — reads open `fact_replenishment_exceptions` and inserts into `fact_sop_supply_constraints`
- `generate_approved_plan_snapshot(conn, cycle_id, cycle_month, approved_by)` — copies consensus demand from `fact_sop_demand_review` into `fact_sop_approved_plan` (triggered automatically when advancing to `approved` stage)

All functions support `--dry-run` for preview without writes.

---

## 7. Frontend UI

**File:** `frontend/src/tabs/SopTab.tsx`
**Navigation:** "S&OP" item in sidebar (CalendarDays icon)
**Keyboard shortcut:** None (beyond sidebar's numeric shortcuts)

### Layout

The tab has three sections:

1. **Cycle List** (left 1/3) — clickable cards showing each cycle's month and current stage with a `StageTimeline` component (numbered circles with checkmarks for completed stages, chevrons between stages)

2. **Cycle Detail** (right 2/3) — appears when a cycle is selected:
   - **Stage actions card** — full `StageTimeline`, "Facilitated by" input, "Advance Stage" button (disabled when `closed`), "Approve Plan" button with "Approved by" and "Plan version" inputs (shown only at `executive_sop` stage)
   - **Supply/Demand Gaps card** — severity-colored gap cards (critical=red, high=orange, medium=amber, low=blue) with gap_type, qty, value, resolution_options, and mitigation_status badge. Shows critical count badge in header.

3. **Approved Plan** (full width, bottom) — filterable table with month picker and item number input. Columns: Item, Location, Plan Month, Approved Qty, Approved By, Approved At.

### Frontend query keys and fetch functions

Defined in `frontend/src/api/queries/evolution.ts`:

- `sopKeys.cycles(params)` — `["sop", "cycles", params]`
- `sopKeys.cycle(cycle_id)` — `["sop", "cycle", cycle_id]`
- `sopKeys.gaps(cycle_id)` — `["sop", "gaps", cycle_id]`
- `sopKeys.approvedPlan(params)` — `["sop", "approved-plan", params]`

TypeScript interfaces: `SopCycle`, `SopGap`, `ApprovedPlanRow`

### Mutations

- **Advance stage:** `POST /sop/cycles/{cycle_id}/advance` via `useMutation`, invalidates `sopKeys.cycles` on success
- **Approve plan:** Direct `fetch()` call to `POST /sop/cycles/{cycle_id}/approve`, invalidates `sopKeys.cycles`

---

## 8. Makefile Targets

```bash
make sop-schema                      # Apply DDL (sql/056_create_sop_module.sql)
make sop-create CYCLE_MONTH=2026-05-01  # Create new cycle
make sop-advance CYCLE_ID=1          # Advance cycle to next stage
make sop-populate CYCLE_ID=1         # Populate demand review + supply constraints
make sop-all                         # sop-schema (full pipeline)
```

---

## 9. Vite Proxy

The `/sop` prefix is proxied in `frontend/vite.config.ts` to the FastAPI backend at `http://127.0.0.1:8000`.

---

## 10. Dependencies

| Dependency | Source | Status |
|---|---|---|
| Statistical forecast | `fact_external_forecast_monthly` (model_id=`'champion'`) | Implemented |
| Replenishment exceptions | `fact_replenishment_exceptions` (for supply constraints) | Implemented |
| S&OP config | `config/sop_config.yaml` | Implemented |

---

## 11. Out of Scope

- Automated ERP sales order confirmation against approved plan
- Multi-currency S&OP (single currency USD assumed)
- Resource/capacity planning for manufacturing (production scheduling)
- Automated supply constraint sourcing from supplier APIs
- Version control / rollback of approved plans (historical cycles preserved read-only)
- Role-based access control enforcement at database level (application-level auth only)

---

## 12. Test Requirements

### Backend API Tests (`tests/api/test_sop.py`)

- `test_get_cycles_200()` — returns list with status and cycle_month
- `test_get_cycle_detail_200()` — returns demand_review + supply_constraints
- `test_get_cycle_404()` — returns 404 for non-existent cycle_id
- `test_advance_cycle_200()` — returns previous_status and new_status
- `test_advance_cycle_final_stage_400()` — returns 400 when cycle is at final stage
- `test_approve_cycle_200()` — sets status=approved, returns plan_version
- `test_get_gaps_200()` — returns gaps with severity ordering
- `test_get_gaps_severity_filter()` — severity query param scopes results
- `test_get_approved_plan_200()` — returns locked rows with correct columns
- `test_get_approved_plan_item_filter()` — item_no query param scopes results

### Frontend Tests (`src/tabs/__tests__/SopTab.test.tsx`)

- Stage timeline renders 6 stage milestones
- Cycle list renders with current_stage badge
- Gap cards render with severity color coding
- Approve button renders only when current_stage=`executive_sop`
- Advance button disabled when current_stage=`closed`
- Empty state renders when no cycles exist
- Approved plan table renders rows with correct columns

### Backend Unit Tests (`tests/unit/test_sop_cycle.py`)

- `test_next_stage_order()` — advancing through all stages in correct order
- `test_next_stage_final_raises()` — advancing from `closed` raises ValueError
- `test_is_terminal_stage()` — only `closed` returns True
- `test_compute_cycle_dates()` — scheduled dates computed from config day-of-month in prior month
