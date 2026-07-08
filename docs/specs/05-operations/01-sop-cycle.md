# S&OP Cycle

> A six-stage state machine that guides cross-functional teams through monthly Sales & Operations Planning from demand review to executive approval.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | SopTab |
| **Key Files** | `SopTab.tsx`, `api/routers/operations/sop.py`, `scripts/ops/run_sop_cycle.py`, `config/operations/sop_config.yaml`, `sql/056_create_sop_module.sql` |

---

## Problem

Planners, supply managers, and finance leads make decisions in isolation. Without a structured cadence, gaps between demand plans and supply constraints surface too late -- after purchase orders are already placed. Teams need a shared workflow that moves from data collection through executive sign-off with clear ownership at each stage.

---

## Solution

A database-backed state machine enforces six sequential stages. Each stage gates advancement on required inputs (review data, constraint flags, gap analysis). The frontend renders stage progress, pending actions, and the final approved plan. Planners advance stages through the UI; the system prevents skipping or reverting.

---

## How It Works

### Stage Progression

| Stage | Name | Owner | Gate to Advance |
|---|---|---|---|
| 1 | Demand Review | Planner | Statistical + override forecasts loaded |
| 2 | Supply Review | Supply Planner | Capacity and material constraints flagged |
| 3 | Gap Analysis | S&OP Lead | All gaps identified with root cause |
| 4 | Pre-S&OP Meeting | Cross-Functional | Resolution proposals for each gap |
| 5 | Executive Review | VP/Director | Financial impact reviewed |
| 6 | Approved Plan | System | Executive approval recorded |

### Cycle Lifecycle

1. A new cycle is created (monthly cadence, configurable via YAML).
2. The system seeds initial demand review rows from the latest production forecast.
3. Planners advance through stages by submitting required data at each gate.
4. At Stage 3, the system auto-generates gap cards comparing demand vs. supply vs. financial targets.
5. Stage 6 locks the cycle -- the approved plan becomes the consensus baseline for the next month.

### Gap Detection

Gaps are computed as the difference between demand forecast totals, supply-constrained quantities, and budget limits. Each gap record includes: item scope, gap type (demand-supply, budget, capacity), magnitude, and recommended resolution.

---

## Data Model

| Table | Purpose | Grain | Key Columns |
|---|---|---|---|
| `fact_sop_cycles` | Cycle header | One row per `cycle_month` | `cycle_id`, `cycle_month`, `status`, `demand_plan_version`, `supply_plan_version`, `approved_plan_version`, `facilitated_by`, `approved_by`, `demand_review_at`, `supply_review_at`, `pre_sop_at`, `executive_sop_at`, `approved_at`, `created_at`, `updated_at` |
| `fact_sop_demand_review` | Demand inputs | **Category grain**, not item/location -- one row per `(cycle_id, item_category)` | `cycle_id`, `item_category`, `statistical_demand_qty`, `commercial_demand_qty`, `consensus_demand_qty`, `statistical_demand_val`, `commercial_demand_val`, `consensus_demand_val`, `review_status` |
| `fact_sop_supply_constraints` | Supply flags | **Category grain**, not item/location -- one row per `(cycle_id, constraint_type)` | `constraint_id`, `cycle_id`, `constraint_type`, `supplier_id`, `impact_qty`, `impact_period`, `mitigation_status` |
| `fact_sop_gaps` | Identified gaps | One row per `(cycle_id, gap_type)` | `gap_id`, `cycle_id`, `gap_type`, `gap_qty`, `gap_value`, `severity`, `resolution_options` (JSONB), `resolution_status` |
| `fact_sop_approved_plan` | Locked consensus plan | Item + location grain -- one row per `(cycle_id, item_id, loc, plan_month)` | `id`, `cycle_id`, `item_id`, `loc`, `plan_month`, `approved_qty`, `statistical_qty`, `override_qty`, `source`, `locked` |

All five tables share `cycle_id` as the join key. Only `fact_sop_approved_plan` carries item/location detail --
the demand-review, supply-constraint, and gap tables roll up to `item_category` (or `constraint_type`/`gap_type`)
for the cycle as a whole.

The `status` column on `fact_sop_cycles` holds the actual stage value written by the API (`demand_review` ->
`supply_review` -> `pre_sop` -> `executive_sop` -> `approved` -> `closed`), which is a terser machine name for
the same six stages described above.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/sop/cycles` | List cycles, paginated, newest `cycle_month` first |
| POST | `/sop/cycles` | Create a new cycle for a month (auth); defaults to the current planning-date month, 409 on duplicate |
| GET | `/sop/cycles/{cycle_id}` | Cycle detail including demand review and supply constraints |
| POST | `/sop/cycles/{cycle_id}/advance` | Move cycle to the next stage in `status` (auth) |
| POST | `/sop/cycles/{cycle_id}/approve` | Lock the cycle at `approved`, record `approved_by`/`approved_plan_version` (auth) |
| GET | `/sop/cycles/{cycle_id}/gaps` | Gap cards for a cycle, filterable by `severity` and `resolution_status` |
| GET | `/sop/approved-plan` | Locked approved demand, filterable by `cycle_id`, `item_id`, `loc` |

Advance and approve are `POST`, not `PUT` -- both require API key authentication.

---

## Pipeline

| Step | Command | What It Does |
|---|---|---|
| Schema | `make sop-schema` | Creates the 5 S&OP tables |
| Seed | `make sop-seed` | Creates an initial cycle for the current planning period |
| Full | `make sop-all` | Schema + seed |
| Create | `make sop-create CYCLE_MONTH=YYYY-MM-DD` | Create a cycle for a specific month |
| Advance | `make sop-advance CYCLE_ID=n` | Advance a cycle to its next stage |
| Populate | `make sop-populate CYCLE_ID=n` | Populate demand-review rows for a cycle |
| Run | `scripts/ops/run_sop_cycle.py --action {create,advance,populate-demand}` | Programmatic cycle operations (used by job scheduler) |

---

## Configuration

File: `config/operations/sop_config.yaml`

| Key | Purpose | Default |
|---|---|---|
| `sop.planning_horizon_months` | Forward planning horizon | `12` |
| `sop.demand_review_day` | Day of the prior month the demand-review meeting is scheduled | `5` |
| `sop.supply_review_day` | Day of the prior month the supply-review meeting is scheduled | `10` |
| `sop.pre_sop_day` | Day of the prior month the pre-S&OP meeting is scheduled | `15` |
| `sop.executive_sop_day` | Day of the prior month the executive sign-off is scheduled | `20` |
| `sop.stages` | Ordered stage machine (matches `status` above) | `demand_review, supply_review, pre_sop, executive_sop, approved, closed` |
| `sop.demand_baseline_model` | `model_id` from `fact_external_forecast_monthly` used as the consensus baseline | `champion` |
| `sop.supply_gap_alert_pct` | Alert when supply falls below this fraction of consensus demand | `0.10` |
| `sop.approvers` | Role required at each stage (informational only, not server-enforced) | `demand_planner`, `supply_planner`, `supply_chain_manager`, `vp_supply_chain` |
| `scheduler.cron` | Scheduled cycle-advancement cadence | `0 7 1 * *` |

---

## Dependencies

| Dependency | Reason |
|---|---|
| Production forecast (02-08) | Seeds Stage 1 demand review quantities |
| Supply constraints (manual entry or import) | Required for Stage 2 gate |
| Financial plan (05-02) | Budget targets used in gap analysis |
| Job scheduler (07-04) | Optional: auto-advance cycles on schedule |

---

## See Also

- `05-operations/02-financial-planning.md` -- budget targets used in gap analysis
- `05-operations/03-event-calendar.md` -- event uplifts feed into demand review
- `02-forecasting/08-production-forecast.md` -- forecast that seeds demand review
- `07-user-experience/04-job-scheduler.md` -- scheduling automated cycle advancement
