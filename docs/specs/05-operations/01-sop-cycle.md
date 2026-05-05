# S&OP Cycle

> A six-stage state machine that guides cross-functional teams through monthly Sales & Operations Planning from demand review to executive approval.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | SopTab |
| **Key Files** | `SopTab.tsx`, `api/routers/operations/sop.py`, `scripts/run_sop_cycle.py`, `config/operations/sop_config.yaml`, `sql/056_create_sop_module.sql` |

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

| Table | Purpose | Key Columns |
|---|---|---|
| `fact_sop_cycles` | Cycle header | `cycle_id`, `period`, `current_stage`, `created_at`, `closed_at` |
| `fact_sop_demand_review` | Demand inputs per DFU (Demand Forecast Unit -- item + location) | `cycle_id`, `item_id`, `loc`, `statistical_qty`, `override_qty`, `final_qty` |
| `fact_sop_supply_constraints` | Supply flags per item-location | `cycle_id`, `item_id`, `loc`, `constraint_type`, `constrained_qty`, `notes` |
| `fact_sop_gaps` | Identified gaps | `cycle_id`, `gap_type`, `item_id`, `loc`, `gap_qty`, `resolution`, `status` |
| `fact_sop_approved_plan` | Locked consensus plan | `cycle_id`, `item_id`, `loc`, `approved_qty`, `approved_by`, `approved_at` |

All five tables share `cycle_id` as the join key.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/sop/cycles` | List all cycles with stage and date range |
| POST | `/sop/cycles` | Create a new cycle for a given period |
| GET | `/sop/cycles/{id}` | Cycle detail including demand review, constraints, gaps |
| PUT | `/sop/cycles/{id}/advance` | Move cycle to next stage (validates gate conditions) |
| PUT | `/sop/cycles/{id}/approve` | Lock cycle at Stage 6, write approved plan |
| GET | `/sop/gaps/{cycle_id}` | Gap cards for a specific cycle |

All mutation endpoints require API key authentication when `API_KEY` is set.

---

## Pipeline

| Step | Command | What It Does |
|---|---|---|
| Schema | `make sop-schema` | Creates the 5 S&OP tables |
| Seed | `make sop-seed` | Creates an initial cycle for the current planning period |
| Full | `make sop-all` | Schema + seed |
| Run | `scripts/run_sop_cycle.py` | Programmatic cycle advancement (used by job scheduler) |

---

## Configuration

File: `config/operations/sop_config.yaml`

| Key | Purpose | Default |
|---|---|---|
| `cycle_cadence` | How often a new cycle is created | `monthly` |
| `stages` | Ordered list of stage names and gate rules | 6 stages as listed above |
| `gap_thresholds.demand_supply_pct` | Minimum percent gap to flag | `5.0` |
| `gap_thresholds.budget_variance_pct` | Budget deviation threshold | `10.0` |
| `auto_seed_demand` | Whether to pre-populate demand review from production forecast | `true` |

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
