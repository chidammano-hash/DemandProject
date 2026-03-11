# 03-12 Inventory Rebalancing — Cross-Location Transfer Optimization

## EPIC
InventoryPlanning

## Status
Implemented

## Priority
P2 — Should Have

## Effort
XL (Extra Large)

## Expert Perspectives
- **Network Optimization Expert** (lead) — graph-based transfer routing, LP formulation, solver selection
- **Warehouse Operations Expert** — transfer lane constraints, receiving capacity, batch sizing, frozen periods
- **Finance & Cost Accounting Expert** — transfer cost modeling, carrying cost savings, stockout avoidance valuation, ROI
- **Demand Planning Expert** — DOS-based imbalance detection, safety stock integration, ABC priority

---

## 1. Overview / Problem Statement

Distribution networks routinely develop **spatial inventory imbalances**: one warehouse holds 90 days of supply for an item while another warehouse, selling the same item, sits below its safety stock target and risks stockout. This mismatch occurs because replenishment is typically planned per-location independently, without considering the network as a whole.

The cost of inaction is twofold:
1. **Excess carrying cost** at over-stocked locations (capital tied up, warehouse space consumed, obsolescence risk).
2. **Lost sales and service failures** at under-stocked locations (stockouts, expedited shipments, customer penalties).

Inventory Rebalancing solves this by:
- Detecting items with simultaneous excess at one location and shortage at another.
- Computing cost-optimal inter-location transfer recommendations using a greedy heuristic or LP solver.
- Providing a full financial model (transfer cost vs. carrying savings + stockout avoidance) so planners can evaluate ROI before approving.
- Supporting an approval workflow (individual or bulk) with audit trail.

The feature bridges the gap between location-level inventory planning (safety stock, EOQ, replenishment policies) and network-level optimization.

---

## 2. Stakeholder Perspectives

### Warehouse Operations
- Need visibility into inbound transfers: quantities, planned ship dates, expected arrivals.
- Must respect receiving capacity constraints (`max_receiving_units_per_period`), batch sizes, and frozen periods.
- Require per-transfer approve/reject workflow with reason tracking.

### Finance
- Demand ROI justification for every recommended transfer: transfer cost vs. avoided stockout value + carrying cost savings.
- Need plan-level financial summary: total cost, total avoided stockout value, net ROI.
- Budget cap support: constrain total transfer spend within an approved budget.

### Network Optimization
- Two solver options: fast greedy heuristic for daily operations, LP solver for periodic strategic rebalancing.
- Objective functions: minimize cost (`min_cost`), maximize service (`max_service`), or equalize DOS across locations (`equalize_dos`).
- Network balance score (1 - avg DOS CV) as a single metric for spatial health.

### Demand Planning
- Integration with safety stock targets: excess/shortage defined relative to `fact_safety_stock_targets.ss_combined`.
- ABC-class priority: A-class items are rebalanced first (configurable priority order).
- Urgency assignment based on destination DOS and ABC class: critical items with <3 DOS are flagged immediately.

---

## 3. Data Model

### 3.1 Transfer Network Topology — `sql/071_create_transfer_network.sql`

**Table: `dim_transfer_lane`**

Defines which locations can ship to which, at what cost, with what constraints. One row per (source, destination, mode) combination.

```sql
CREATE TABLE IF NOT EXISTS dim_transfer_lane (
    lane_sk                    BIGSERIAL PRIMARY KEY,
    lane_id                    TEXT UNIQUE NOT NULL DEFAULT gen_random_uuid()::TEXT,
    source_loc                 TEXT NOT NULL,
    dest_loc                   TEXT NOT NULL,
    transfer_mode              TEXT NOT NULL DEFAULT 'truck'
                               CHECK (transfer_mode IN ('truck','rail','air','parcel')),
    -- Cost model ($/unit)
    cost_per_unit              NUMERIC(10,4) NOT NULL,
    handling_cost              NUMERIC(10,4) DEFAULT 0,
    freight_cost               NUMERIC(10,4) DEFAULT 0,
    receiving_cost             NUMERIC(10,4) DEFAULT 0,
    fixed_cost_per_shipment    NUMERIC(10,2) DEFAULT 0,
    -- Lead time
    transfer_lt_days           INTEGER NOT NULL DEFAULT 3,
    -- Constraints
    min_transfer_qty           INTEGER DEFAULT 1,
    max_transfer_qty           INTEGER,
    batch_size                 INTEGER DEFAULT 1,
    max_shipments_per_week     INTEGER DEFAULT 5,
    max_receiving_units_per_period INTEGER,
    -- Status
    is_active                  BOOLEAN DEFAULT TRUE,
    effective_from             DATE,
    effective_to               DATE,
    -- Metadata
    load_ts                    TIMESTAMPTZ DEFAULT NOW(),
    modified_ts                TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source_loc, dest_loc, transfer_mode)
);
```

**Indexes:**
- `idx_lane_source` — B-tree on `source_loc` WHERE `is_active` (filtered)
- `idx_lane_dest` — B-tree on `dest_loc` WHERE `is_active` (filtered)
- `idx_lane_pair` — B-tree on `(source_loc, dest_loc)` for lane lookups

**Key design decisions:**
- **Soft delete** via `is_active` flag — lanes are never physically deleted.
- **Effective dating** (`effective_from`, `effective_to`) — supports seasonal or contract-based lane availability.
- **Four transfer modes** — truck, rail, air, parcel — each with independent cost/constraint profiles.
- **Granular cost model** — `cost_per_unit` (total) plus decomposed `handling_cost` + `freight_cost` + `receiving_cost` for reporting, plus `fixed_cost_per_shipment` for shipment-level overhead.

### 3.2 Rebalancing Plan & Transfers — `sql/072_create_rebalancing_plan.sql`

**Table: `fact_rebalancing_plan`** — Plan header (one row per computation run)

```sql
CREATE TABLE IF NOT EXISTS fact_rebalancing_plan (
    plan_sk                    BIGSERIAL PRIMARY KEY,
    plan_id                    TEXT UNIQUE NOT NULL DEFAULT gen_random_uuid()::TEXT,
    computation_date           DATE NOT NULL,
    horizon_weeks              INTEGER NOT NULL DEFAULT 4,
    solver_method              TEXT NOT NULL DEFAULT 'greedy',
    objective                  TEXT NOT NULL DEFAULT 'min_cost',
    -- Plan-level KPIs
    total_transfer_qty         NUMERIC(15,2),
    total_transfer_cost        NUMERIC(12,2),
    total_avoided_stockout_value NUMERIC(12,2),
    net_roi                    NUMERIC(10,4),
    network_balance_before     NUMERIC(6,4),
    network_balance_after      NUMERIC(6,4),
    items_rebalanced           INTEGER,
    lanes_used                 INTEGER,
    -- Workflow status
    status                     TEXT NOT NULL DEFAULT 'draft'
                               CHECK (status IN ('draft','pending_approval','approved',
                                                  'partially_approved','executing',
                                                  'completed','cancelled')),
    approved_by                TEXT,
    approved_ts                TIMESTAMPTZ,
    -- Metadata
    solver_runtime_ms          INTEGER,
    created_ts                 TIMESTAMPTZ DEFAULT NOW(),
    modified_ts                TIMESTAMPTZ DEFAULT NOW()
);
```

**Table: `fact_rebalancing_transfer`** — Individual transfer recommendations within a plan

```sql
CREATE TABLE IF NOT EXISTS fact_rebalancing_transfer (
    transfer_sk                BIGSERIAL PRIMARY KEY,
    transfer_id                TEXT UNIQUE NOT NULL DEFAULT gen_random_uuid()::TEXT,
    plan_id                    TEXT NOT NULL REFERENCES fact_rebalancing_plan(plan_id),
    item_no                    TEXT NOT NULL,
    source_loc                 TEXT NOT NULL,
    dest_loc                   TEXT NOT NULL,
    lane_id                    TEXT,
    transfer_mode              TEXT DEFAULT 'truck',
    -- Quantities
    recommended_qty            NUMERIC(15,4) NOT NULL,
    approved_qty               NUMERIC(15,4),
    -- Source context (snapshot at computation time)
    source_on_hand             NUMERIC(15,4),
    source_dos                 NUMERIC(10,2),
    source_ss_target           NUMERIC(15,4),
    source_excess_qty          NUMERIC(15,4),
    -- Destination context
    dest_on_hand               NUMERIC(15,4),
    dest_dos                   NUMERIC(10,2),
    dest_ss_target             NUMERIC(15,4),
    dest_shortage_qty          NUMERIC(15,4),
    -- Financial
    transfer_cost              NUMERIC(12,2),
    carrying_cost_saved        NUMERIC(12,2),
    stockout_cost_avoided      NUMERIC(12,2),
    net_benefit                NUMERIC(12,2),
    roi                        NUMERIC(10,4),
    -- Scheduling
    planned_ship_date          DATE,
    expected_arrival_date      DATE,
    transfer_lt_days           INTEGER,
    -- Priority
    priority_score             NUMERIC(10,4),
    abc_class                  TEXT,
    urgency                    TEXT CHECK (urgency IN ('critical','high','medium','low')),
    -- Workflow
    status                     TEXT NOT NULL DEFAULT 'recommended'
                               CHECK (status IN ('recommended','approved','rejected',
                                                  'hold','in_transit','received','cancelled')),
    approved_by                TEXT,
    approved_ts                TIMESTAMPTZ,
    rejection_reason           TEXT,
    notes                      TEXT,
    -- Metadata
    created_ts                 TIMESTAMPTZ DEFAULT NOW(),
    modified_ts                TIMESTAMPTZ DEFAULT NOW()
);
```

**Indexes on `fact_rebalancing_transfer`:**
- `idx_rebal_transfer_plan` — B-tree on `plan_id` (FK join)
- `idx_rebal_transfer_item` — B-tree on `item_no` (item search)
- `idx_rebal_transfer_source` — B-tree on `source_loc`
- `idx_rebal_transfer_dest` — B-tree on `dest_loc`
- `idx_rebal_transfer_status` — B-tree on `status`
- `idx_rebal_transfer_urgency` — B-tree on `(urgency, priority_score DESC)` WHERE `status = 'recommended'` (filtered, for urgency-sorted work queue)

**Plan status lifecycle:**
```
draft --> pending_approval --> approved --> executing --> completed
                          \-> partially_approved --> executing --> completed
                          \-> cancelled
```

**Transfer status lifecycle:**
```
recommended --> approved --> in_transit --> received
            \-> rejected
            \-> hold --> approved | rejected | cancelled
            \-> cancelled
```

### 3.3 Network Balance Materialized View — `sql/073_create_rebalancing_views.sql`

**View: `mv_network_balance`**

Per-item DOS variance across locations for imbalance detection. Only items present at 2+ locations are included.

```sql
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_network_balance AS
SELECT
    a.item_no,
    COUNT(DISTINCT a.loc)             AS location_count,
    AVG(a.eom_qty_on_hand)           AS avg_on_hand,
    STDDEV(a.eom_qty_on_hand)        AS stddev_on_hand,
    AVG(dos)                          AS avg_dos,
    STDDEV(dos)                       AS stddev_dos,
    dos_cv                            AS dos_cv,          -- CV = stddev/mean
    excess_loc_count,                                      -- locs where on_hand > 1.5x SS
    shortage_loc_count                                     -- locs where on_hand < SS
FROM agg_inventory_monthly a
LEFT JOIN fact_safety_stock_targets s ON a.item_no = s.item_no AND a.loc = s.loc
WHERE a.month_start = (SELECT MAX(month_start) FROM agg_inventory_monthly)
GROUP BY a.item_no
HAVING COUNT(DISTINCT a.loc) >= 2;
```

**Unique index:** `idx_mv_netbal_item` on `item_no` (enables `REFRESH MATERIALIZED VIEW CONCURRENTLY`).

**Key computed columns:**
| Column | Formula | Purpose |
|---|---|---|
| `dos_cv` | `STDDEV(DOS) / AVG(DOS)` | Coefficient of variation of DOS across locations — primary imbalance indicator |
| `excess_loc_count` | Count where `on_hand > ss_combined * 1.5` | Number of over-stocked locations |
| `shortage_loc_count` | Count where `on_hand < ss_combined` | Number of under-stocked locations |

---

## 4. Configuration Reference

**File:** `mvp/demand/config/rebalancing_config.yaml`

### `network` — Default lane parameters (used when no lane exists in `dim_transfer_lane`)

| Key | Type | Default | Description |
|---|---|---|---|
| `default_transfer_lt_days` | int | 3 | Default transit time for lanes not in the DB |
| `default_cost_per_unit` | float | 0.50 | Default per-unit transfer cost ($/unit) |
| `default_min_transfer_qty` | int | 10 | Minimum units to justify a transfer |
| `default_batch_size` | int | 1 | Round transfer quantities to multiples of this |

### `optimization` — Solver settings

| Key | Type | Default | Description |
|---|---|---|---|
| `solver` | string | `"greedy"` | Solver method: `greedy` or `lp` |
| `horizon_weeks` | int | 4 | Rolling planning horizon in weeks |
| `time_limit_seconds` | int | 60 | Wall-clock time limit for LP solver |
| `objective` | string | `"min_cost"` | Objective function: `min_cost`, `max_service`, `equalize_dos` |

### `triggers` — Imbalance detection thresholds

| Key | Type | Default | Description |
|---|---|---|---|
| `excess_threshold_pct` | float | 1.50 | `on_hand / ss_target > 150%` classifies location as excess |
| `shortage_threshold_pct` | float | 0.80 | `on_hand / ss_target < 80%` classifies location as shortage |
| `dos_cv_threshold` | float | 0.40 | Network DOS CV > 0.40 flags item as imbalanced |
| `min_benefit_per_transfer` | float | 5.0 | Minimum net benefit ($) to include a transfer in the plan |

### `costs` — Financial model parameters

| Key | Type | Default | Description |
|---|---|---|---|
| `default_handling_cost` | float | 0.10 | $/unit handling at source warehouse |
| `default_freight_cost` | float | 0.30 | $/unit inter-location freight |
| `default_receiving_cost` | float | 0.10 | $/unit receiving at destination warehouse |
| `stockout_cost_multiplier` | float | 5.0 | Revenue multiplier for missed sales valuation |
| `carrying_cost_annual_pct` | float | 0.25 | Annual holding cost as percentage of unit value |

### `constraints` — Operational guardrails

| Key | Type | Default | Description |
|---|---|---|---|
| `frozen_period_days` | int | 7 | No transfers scheduled within first N days of an approved plan |
| `max_source_drawdown_pct` | float | 0.30 | Never take more than 30% of source `on_hand` in a single plan |
| `abc_priority_order` | list | `["A", "B", "C"]` | ABC classes processed in this priority order |

### `scheduling` — Automated execution

| Key | Type | Default | Description |
|---|---|---|---|
| `auto_compute_cron` | string | `"0 5 * * 1"` | Cron schedule for automatic plan generation (Monday 5 AM) |
| `auto_compute_enabled` | bool | `false` | Whether automatic scheduling is active |

---

## 5. Optimization Engine

**Script:** `mvp/demand/scripts/compute_rebalancing.py`

### 5.1 Pipeline Overview

```
1. Load config (rebalancing_config.yaml)
2. Load active transfer lanes (dim_transfer_lane)
3. Load inventory state (agg_inventory_monthly + fact_safety_stock_targets + dim_dfu)
4. Detect imbalances (classify each item-loc as excess / shortage / balanced)
5. Build transfer candidates (all feasible excess -> shortage pairs)
6. Compute financials (cost, savings, ROI per candidate)
7. Assign urgency (critical / high / medium / low)
8. Run solver (greedy or LP)
9. Apply budget cap (if specified)
10. Write plan + transfers to DB
```

### 5.2 Imbalance Detection

For each item-location pair with a safety stock target:

```
ratio = on_hand / ss_target

If ratio > excess_threshold_pct (default 1.5):  -> EXCESS
If ratio < shortage_threshold_pct (default 0.8): -> SHORTAGE
Otherwise:                                       -> BALANCED
```

Only items that have **both** at least one excess location **and** at least one shortage location are considered for rebalancing. Items that are uniformly over-stocked or uniformly under-stocked across all locations are excluded (those are replenishment problems, not rebalancing problems).

### 5.3 Transfer Candidate Generation

For each (excess_location, shortage_location) pair of the same item:

```
excess_qty     = source.on_hand - source.ss_target
shortage_qty   = dest.ss_target - dest.on_hand
drawdown_limit = source.on_hand * max_source_drawdown_pct

raw_qty = min(excess_qty, shortage_qty, drawdown_limit)

If lane exists: apply lane.max_transfer_qty cap
Round down to batch_size: raw_qty = floor(raw_qty / batch) * batch
If raw_qty < min_transfer_qty: discard candidate
```

Lane lookup uses `(source_loc, dest_loc)` from `dim_transfer_lane`. If no lane exists, default network parameters from config are used.

### 5.4 Greedy Solver

The greedy solver is the default and handles the majority of operational use cases.

**Formulation:**

1. **Filter**: Remove candidates where `net_benefit < min_benefit_per_transfer`.

2. **Score** each candidate:
   ```
   shortage_severity = dest_shortage_qty / max(dest_ss_target, 1)
   priority_score = urgency_weight * shortage_severity + max(ROI, 0)
   ```
   Where urgency weights are: critical=4, high=3, medium=2, low=1.

3. **Sort** candidates by `priority_score` descending.

4. **Greedy assignment**: Iterate through sorted candidates. For each candidate:
   - Check remaining excess at source: `avail_excess[source]`
   - Check remaining shortage at destination: `avail_shortage[dest]`
   - Assign `qty = min(recommended_qty, avail_excess, avail_shortage)`
   - If `qty > 0`: accept transfer, decrement remaining pools
   - If `qty <= 0`: skip (already satisfied by earlier transfers)

**Complexity:** O(n log n) for the sort + O(n) for the greedy pass, where n = number of candidates.

**Properties:**
- Prioritizes high-urgency, high-ROI transfers.
- Respects source drawdown limits (never depletes a source below its SS target).
- Deterministic: same input always produces the same output.
- No inter-item trade-offs (each item is independent).

### 5.5 LP Solver

The LP solver uses `scipy.optimize.linprog` (HiGHS backend) for mathematically optimal solutions.

**Formal notation:**

Let:
- $x_i \in \mathbb{R}_{\geq 0}$ be the transfer quantity for candidate $i$, $i \in \{1, \ldots, n\}$
- $b_i$ be the net benefit per unit for candidate $i$
- $q_i$ be the maximum recommended quantity for candidate $i$
- $S_j$ be the set of candidates sourcing from source $j$
- $D_k$ be the set of candidates delivering to destination $k$
- $e_j$ be the available excess at source $j$
- $s_k$ be the shortage quantity at destination $k$

**Objective (maximize total net benefit):**

$$\max \sum_{i=1}^{n} b_i \cdot x_i$$

Equivalently, minimize $-b_i \cdot x_i$ (as `linprog` minimizes).

**Subject to:**

1. **Per-candidate upper bound:**
   $$0 \leq x_i \leq q_i \quad \forall i$$

2. **Source capacity constraint** (total transfers from each source cannot exceed its excess):
   $$\sum_{i \in S_j} x_i \leq e_j \quad \forall j$$

3. **Destination demand constraint** (total transfers to each destination cannot exceed its shortage):
   $$\sum_{i \in D_k} x_i \leq s_k \quad \forall k$$

**Implementation details:**
- Method: `highs` (state-of-the-art LP solver, ships with scipy).
- Time limit: configurable via `time_limit_seconds` (default 60s).
- Post-solve rounding: solutions with $x_i < 0.5$ are discarded; remaining are rounded to integers.
- Fallback: if LP solver fails (infeasible or timeout), falls back to greedy solver with a warning.
- Priority scores are computed post-solve for display ordering (not part of the LP objective).

**When to use LP vs. Greedy:**
| Scenario | Recommended Solver |
|---|---|
| Daily operational rebalancing | Greedy (fast, interpretable) |
| Weekly strategic rebalancing | LP (globally optimal) |
| Network with many shared sources | LP (handles contention better) |
| Budget-constrained plans | Either (budget cap applied post-solve) |

---

## 6. Financial Model

### 6.1 Transfer Cost

Per-transfer cost combines variable and fixed components:

```
transfer_cost = recommended_qty * cost_per_unit + fixed_cost_per_shipment
```

Where `cost_per_unit` is the lane-specific rate (or default from config), and `fixed_cost_per_shipment` covers shipment-level overhead (documentation, loading, etc.).

### 6.2 Carrying Cost Saved

Removing excess inventory from the source saves annual carrying cost, prorated to the planning horizon:

```
carrying_cost_saved = source_excess_qty * unit_cost * carrying_cost_annual_pct * (horizon_days / 365)
```

Where `unit_cost` defaults to $1.00 (no `unit_cost` column in current schema; future enhancement to integrate with `dim_item` or `fact_safety_stock_targets`).

### 6.3 Stockout Cost Avoided

Filling a shortage at the destination avoids stockout-driven revenue loss:

```
shortage_severity = min(1.0, dest_shortage_qty / max(dest_ss_target, 1))
stockout_cost_avoided = min(qty, dest_shortage_qty) * unit_cost * stockout_cost_multiplier * shortage_severity
```

The `shortage_severity` factor (0 to 1) scales the avoidance value proportionally to how far below the SS target the destination sits. A location at 0% of SS receives full stockout valuation; a location at 80% receives partial.

### 6.4 Net Benefit and ROI

```
net_benefit = stockout_cost_avoided + carrying_cost_saved - transfer_cost

roi = net_benefit / max(transfer_cost, 0.01)
```

Only transfers with `net_benefit >= min_benefit_per_transfer` (default $5.00) are included in the plan. ROI is per-transfer; the plan-level `net_roi` aggregates across all selected transfers.

### 6.5 Network Balance Score

The network balance score is a portfolio-level metric derived from the coefficient of variation (CV) of DOS across locations:

```
For each item with 2+ locations:
    cv_i = stddev(DOS across locations) / mean(DOS across locations)

network_dos_cv = mean(cv_i)  across all items
network_balance_score = (1 - network_dos_cv) * 100
```

A score of 100% means perfectly balanced (all locations have identical DOS for every item). A score of 60% means the average item has 40% variation in DOS across locations. The score is computed before and after the plan to measure improvement.

---

## 7. Urgency & Priority Assignment

Each transfer candidate is assigned an urgency level based on the destination's inventory health:

| Urgency | Condition | Weight |
|---|---|---|
| **Critical** | Destination DOS < 3 days **AND** ABC class = A | 4 |
| **High** | Destination DOS < 7 days | 3 |
| **Medium** | Destination DOS < 14 days | 2 |
| **Low** | All others | 1 |

**Priority score** (used for sorting in the greedy solver and for display ordering):

```
shortage_severity = dest_shortage_qty / max(dest_ss_target, 1)
priority_score = urgency_weight * shortage_severity + max(ROI, 0)
```

This composite score balances service urgency (urgency weight * severity) with economic efficiency (ROI). Critical items with deep shortages surface first; among equal-urgency items, higher-ROI transfers take precedence.

---

## 8. API Contract

**Router:** `mvp/demand/api/routers/inv_planning_rebalancing.py`
**Tags:** `Inventory Rebalancing`
**Auth:** Mutation endpoints require `require_api_key`; read endpoints are public.
**DB pattern:** All endpoints use `get_conn()` directly (not `Depends(_get_pool)`).

### 8.1 KPIs

```
GET /inv-planning/rebalancing/kpis
```

**Response:**
```json
{
  "total_multi_loc_items": 1234,
  "avg_dos_cv": 0.42,
  "network_balance_score": 58.0,
  "imbalanced_items": 312,
  "total_excess_locs": 580,
  "total_shortage_locs": 445,
  "latest_plan": {
    "plan_id": "uuid",
    "total_transfer_qty": 15000.0,
    "total_transfer_cost": 7500.0,
    "total_avoided_stockout_value": 45000.0,
    "net_roi": 5.0,
    "items_rebalanced": 120,
    "status": "draft",
    "computation_date": "2026-02-24"
  }
}
```
Cache: 120s.

### 8.2 Network Topology — List Lanes

```
GET /inv-planning/rebalancing/network
  Query: source_loc, dest_loc, limit (100), offset (0)
```

**Response:** `{ total, rows: [{ lane_id, source_loc, dest_loc, transfer_mode, cost_per_unit, handling_cost, freight_cost, receiving_cost, fixed_cost_per_shipment, transfer_lt_days, min_transfer_qty, max_transfer_qty, batch_size }] }`

Cache: 300s.

### 8.3 Network Topology — Create/Update Lane

```
POST /inv-planning/rebalancing/network
  Auth: require_api_key
  Body: LaneCreateBody {
    source_loc, dest_loc, transfer_mode ("truck"),
    cost_per_unit, handling_cost (0), freight_cost (0), receiving_cost (0),
    fixed_cost_per_shipment (0), transfer_lt_days (3),
    min_transfer_qty (1), max_transfer_qty (null), batch_size (1),
    max_shipments_per_week (5), max_receiving_units_per_period (null)
  }
```

**Response:** `{ lane_id, status: "created" }`

Upserts on `(source_loc, dest_loc, transfer_mode)` unique constraint.

### 8.4 Network Topology — Deactivate Lane

```
DELETE /inv-planning/rebalancing/network/{lane_id}
  Auth: require_api_key
```

**Response:** `{ lane_id, status: "deactivated" }` | 404

Soft-delete: sets `is_active = FALSE`.

### 8.5 Imbalance Detection

```
GET /inv-planning/rebalancing/imbalances
  Query: item, limit (50), offset (0)
```

**Response:** `{ total, rows: [{ item_no, location_count, avg_on_hand, avg_dos, dos_cv, excess_loc_count, shortage_loc_count }] }`

Sorted by `dos_cv DESC` (most imbalanced first). Cache: 60s.

### 8.6 Compute Plan (Async)

```
POST /inv-planning/rebalancing/compute  [202 Accepted]
  Auth: require_api_key
  Body: ComputeBody { solver ("greedy"), horizon_weeks (4), budget_cap (null) }
```

**Response:** `{ status: "accepted", message: "Rebalancing computation started" }`

Runs in a background `ThreadPoolExecutor`. Callers should poll `GET /plans` for completion.

### 8.7 List Plans

```
GET /inv-planning/rebalancing/plans
  Query: status, limit (10), offset (0)
```

**Response:** `{ total, rows: [{ plan_id, computation_date, solver_method, objective, total_transfer_qty, total_transfer_cost, total_avoided_stockout_value, net_roi, items_rebalanced, lanes_used, status, solver_runtime_ms, created_ts }] }`

Sorted by `computation_date DESC, created_ts DESC`. Cache: 30s.

### 8.8 Plan Detail

```
GET /inv-planning/rebalancing/plans/{plan_id}
```

**Response:** Full plan record including `horizon_weeks`, `network_balance_before`, `network_balance_after`, `approved_by`, `approved_ts`. | 404

Cache: 30s.

### 8.9 List Plan Transfers

```
GET /inv-planning/rebalancing/plans/{plan_id}/transfers
  Query: urgency, status, item, sort_by (priority_score), sort_dir (desc), limit (50), offset (0)
```

**Valid sort columns:** `priority_score`, `recommended_qty`, `transfer_cost`, `net_benefit`, `roi`, `urgency`, `item_no`.

**Response:** `{ total, rows: [RebalancingTransfer] }`

Each transfer includes full source/destination context snapshot, financial breakdown, scheduling dates, priority/urgency, and workflow status. Cache: 30s.

### 8.10 Approve Transfer

```
POST /inv-planning/rebalancing/transfers/{transfer_id}/approve
  Auth: require_api_key
  Body: TransferApproveBody { approved_by, approved_qty (null = use recommended), notes (null) }
```

**Response:** `{ transfer_id, item_no, source_loc, dest_loc, approved_qty, status: "approved" }` | 404

Only transitions from `status = 'recommended'`.

### 8.11 Reject Transfer

```
POST /inv-planning/rebalancing/transfers/{transfer_id}/reject
  Auth: require_api_key
  Body: TransferRejectBody { rejection_reason (required), notes (null) }
```

**Response:** `{ transfer_id, item_no, source_loc, dest_loc, status: "rejected" }` | 404 | 422 (empty reason)

Only transitions from `status IN ('recommended', 'hold')`.

### 8.12 Bulk Approve All

```
POST /inv-planning/rebalancing/plans/{plan_id}/approve-all
  Auth: require_api_key
  Body: TransferApproveBody { approved_by, notes (null) }
```

**Response:** `{ plan_id, approved_count, status: "approved" }`

Bulk-approves all `status = 'recommended'` transfers in the plan. Also transitions the plan itself to `status = 'approved'`.

---

## 9. UI Design

### Panel: "Rebalancing" in `InvPlanningTab.tsx`

**Component:** `frontend/src/tabs/inv-planning/RebalancingPanel.tsx`

The panel is organized into 6 sections:

#### Section 1: Info Banner
A concise description of the rebalancing concept displayed in a muted background card at the top.

#### Section 2: KPI Cards (4-column grid)

| Card | Source | Color |
|---|---|---|
| Transfer Opportunities | `kpis.imbalanced_items` | Blue |
| Est. Cost Savings | `kpis.latest_plan.total_avoided_stockout_value` | Green |
| Urgent Transfers | Count of critical + high urgency in current page | Amber |
| Network Balance | `kpis.network_balance_score` (%) | Default |

#### Section 3: Action Bar
- **Solver selector**: dropdown (`Greedy Solver` / `LP Solver`)
- **Horizon input**: numeric input (1-12 weeks)
- **"Compute Plan" button**: triggers `POST /compute` (202), shows status message
- **"Approve All" button**: visible only when latest plan is in `draft` status; bulk-approves all recommended transfers

#### Section 4: Top Transfer Routes Bar Chart
Horizontal bar chart (Recharts `BarChart` with `layout="vertical"`) showing the top 10 transfers by quantity. Each bar is labeled with `source_loc -> dest_loc`. Provides an at-a-glance view of the largest material movements.

#### Section 5: Transfer Work Queue Table
Paginated table (50 rows per page) with columns:

| Column | Alignment | Notes |
|---|---|---|
| Item | Left | Monospace font |
| Source | Left | |
| Dest | Left | |
| Qty | Right | Formatted integer |
| Cost | Right | Red text, dollar format |
| Benefit | Right | Green text, dollar format |
| ROI | Right | 2 decimal places |
| Urgency | Center | Color-coded badge (red/amber/yellow/neutral) |
| Status | Center | Color-coded badge (blue/green/red/yellow) |
| Actions | Right | Approve / Reject buttons (visible for `recommended` status) |

Row background is urgency-tinted: critical rows get a subtle red background, high gets amber, medium gets yellow, low gets no tint.

**Reject modal**: Clicking "Reject" opens a centered modal overlay with a required `rejection_reason` textarea. Cancel and Reject buttons.

**Pagination**: Prev / Next buttons with page counter and total transfer count.

#### Section 6: Cost vs. Benefit Scatter Chart
Recharts `ScatterChart` plotting each transfer as a dot:
- X-axis: transfer cost ($)
- Y-axis: net benefit ($)
- Z-axis (bubble size): transfer quantity
- Reference line at y=0 (break-even)
- Dots above the line have positive ROI

Includes an explanatory muted-text description below the chart title.

### Query Module

**File:** `frontend/src/api/queries/inv-planning-rebalancing.ts`

**TypeScript interfaces:**
- `RebalancingKpis` — KPI response with nested `latest_plan`
- `TransferLane` — lane topology record
- `Imbalance` — per-item network imbalance summary
- `RebalancingPlan` — plan header
- `RebalancingTransfer` — individual transfer with full context

**Query key factory:** `rebalancingKeys` with keys for `kpis`, `network`, `imbalances`, `plans`, `planDetail`, `transfers`.

**Fetch functions:** `fetchRebalancingKpis`, `fetchTransferLanes`, `fetchImbalances`, `fetchRebalancingPlans`, `fetchPlanDetail`, `fetchPlanTransfers`, `computeRebalancingPlan`, `approveTransfer`, `rejectTransfer`, `approveAllTransfers`.

---

## 10. Integration Points

### Exception Queue (IPfeature7)
Rebalancing transfers with urgency `critical` or `high` should surface as exceptions in `fact_replenishment_exceptions` with `exception_type = 'rebalancing_required'`. The exception engine can detect items with high DOS CV from `mv_network_balance`.

### S&OP Cycle (F4.2)
Rebalancing plans in `pending_approval` or `approved` status feed into the S&OP review cycle. The plan's financial summary (total cost, avoided stockouts, net ROI) is presented as a decision point during the Supply Review phase.

### Financial Plan (F4.1)
Transfer costs from approved plans contribute to the `inventory_carrying_cost` and `logistics_cost` line items in the financial planning module. Budget cap integration ensures rebalancing plans respect the approved logistics budget.

### Job Scheduler (Feature 39)
The rebalancing computation can be registered as a job type in `common/job_registry.py` for scheduled execution (e.g., weekly Monday 5 AM per `auto_compute_cron`). The `POST /compute` endpoint uses a `ThreadPoolExecutor` for immediate background execution; the scheduler provides recurring automation.

### Safety Stock (IPfeature3)
Rebalancing is downstream of safety stock computation. The `fact_safety_stock_targets.ss_combined` column is the threshold for classifying locations as excess or shortage. Changes to safety stock targets directly affect which transfers are recommended.

### ABC-XYZ Classification (IPfeature11)
ABC class from `dim_dfu.abc_vol` drives urgency assignment (A-class items get `critical` urgency at low DOS) and `abc_priority_order` in the config controls processing order.

---

## 11. Operational Considerations

### Scheduling
- **Manual**: Planners trigger computation via the UI "Compute Plan" button or `make rebalancing-compute`.
- **Automated**: Set `auto_compute_enabled: true` in config and register as a scheduled job. Default cron: `0 5 * * 1` (Monday 5 AM).
- **Horizon**: Default 4 weeks. Shorter horizons (1-2 weeks) for tactical; longer (8-12 weeks) for strategic network optimization.

### Dry-Run Mode
The script supports `--dry-run` which previews the top 10 recommended transfers without writing to the database. Useful for validation before committing a plan.

```bash
uv run python scripts/compute_rebalancing.py --dry-run
```

### Budget Cap
Optional `--budget-cap` parameter (also exposed in the API `ComputeBody`) sorts selected transfers by ROI descending and includes only those whose cumulative transfer cost fits within the cap.

### Audit Trail
- Every plan gets a unique `plan_id` (UUID).
- Transfer approval/rejection records `approved_by`, `approved_ts`, `rejection_reason`, and `notes`.
- Plan status transitions are tracked via `status` column with CHECK constraints.
- `created_ts` and `modified_ts` on both tables provide temporal audit.

### Materialized View Refresh
`mv_network_balance` should be refreshed after:
1. New inventory data is loaded (`make load-inventory`).
2. Safety stock targets are recomputed (`make ss-compute`).

Refresh command:
```sql
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_network_balance;
```

### Single-Item Mode
The script supports `--item ITEM` to restrict computation to a single item, useful for debugging or targeted rebalancing of a specific SKU.

---

## 12. Make Targets

```makefile
# Inventory Rebalancing
rebalancing-schema:          # Apply DDL: dim_transfer_lane + fact_rebalancing_plan/transfer + mv_network_balance
rebalancing-refresh:         # REFRESH MATERIALIZED VIEW CONCURRENTLY mv_network_balance
rebalancing-compute:         # Run rebalancing optimizer (greedy, 4-week horizon)
rebalancing-compute-lp:      # Run rebalancing optimizer with LP solver
rebalancing-compute-dry:     # Preview rebalancing plan without writing (--dry-run)
rebalancing-all:             # rebalancing-schema + rebalancing-refresh + rebalancing-compute
```

**CLI usage:**

```bash
# Full pipeline (first time)
make rebalancing-all

# Subsequent runs
make rebalancing-refresh && make rebalancing-compute

# LP solver
make rebalancing-compute-lp

# Single item
uv run python scripts/compute_rebalancing.py --item 100320

# Budget-constrained
uv run python scripts/compute_rebalancing.py --budget-cap 10000

# Dry run preview
make rebalancing-compute-dry
```

---

## Dependencies

| Dependency | Type | Notes |
|---|---|---|
| `agg_inventory_monthly` | Existing MV | Source of latest on-hand, daily sales, DOS |
| `fact_safety_stock_targets` | IPfeature3 | SS targets for excess/shortage classification |
| `dim_dfu` | Existing table | ABC class lookup |
| `scipy` | Python package | LP solver (`linprog` with HiGHS backend); optional, falls back to greedy |
| `common/db.py` | Existing module | `get_db_params()` for DB connections |
| `common/planning_date.py` | Existing module | `get_planning_date()` for computation date |

---

## Testing Requirements

### Backend Unit Tests: `mvp/demand/tests/unit/test_rebalancing.py`

- Imbalance detection: items with both excess and shortage are included; items with only one are excluded.
- Candidate generation: respects `max_source_drawdown_pct`, `min_transfer_qty`, `batch_size` rounding.
- Greedy solver: highest priority_score transfer is selected first.
- Greedy solver: source excess is decremented after each assignment.
- Financial model: `net_benefit = stockout_cost_avoided + carrying_cost_saved - transfer_cost`.
- Urgency assignment: A-class item with DOS < 3 gets `critical`; DOS < 7 gets `high`.
- Budget cap: transfers are capped at cumulative cost <= budget.
- Network balance computation: CV calculation correctness.
- LP solver fallback: returns greedy result when scipy is unavailable.
- LP solver: respects source and destination constraints.

### Backend API Tests: `mvp/demand/tests/api/test_rebalancing.py`

- `GET /inv-planning/rebalancing/kpis` returns expected shape with `latest_plan`.
- `GET /inv-planning/rebalancing/network` pagination works.
- `POST /inv-planning/rebalancing/network` creates a lane (returns `lane_id`).
- `DELETE /inv-planning/rebalancing/network/{id}` deactivates (soft delete).
- `GET /inv-planning/rebalancing/imbalances` sorted by `dos_cv DESC`.
- `POST /inv-planning/rebalancing/compute` returns 202.
- `GET /inv-planning/rebalancing/plans` returns plan list.
- `GET /inv-planning/rebalancing/plans/{id}` returns 404 for missing plan.
- `GET /inv-planning/rebalancing/plans/{id}/transfers` pagination and filtering.
- `POST /transfers/{id}/approve` transitions status correctly.
- `POST /transfers/{id}/reject` requires `rejection_reason`.
- `POST /plans/{id}/approve-all` bulk-approves and updates plan status.

### Frontend Tests: `mvp/demand/frontend/src/tabs/__tests__/RebalancingPanel.test.tsx`

- Renders KPI cards with mock data.
- Shows empty state when no plan exists.
- Renders transfer table with urgency badges.
- Compute button triggers mutation.
- Approve/Reject buttons appear only for `recommended` status.
- Reject modal requires non-empty reason.

---

## Acceptance Criteria

- [ ] `dim_transfer_lane` supports 4 transfer modes with independent cost/constraint profiles
- [ ] Imbalance detection identifies items with simultaneous excess and shortage across locations
- [ ] Greedy solver respects `max_source_drawdown_pct` and `min_transfer_qty`
- [ ] LP solver produces mathematically optimal solution (fallback to greedy on failure)
- [ ] Financial model computes per-transfer ROI with transfer cost, carrying savings, and stockout avoidance
- [ ] Urgency assignment: critical for A-class items with DOS < 3 days
- [ ] Plan status lifecycle: draft -> approved -> executing -> completed
- [ ] Transfer approval/rejection with audit trail (who, when, reason)
- [ ] Budget cap constrains total plan cost
- [ ] UI shows KPI cards, transfer routes chart, work queue table, cost-benefit scatter
- [ ] `make rebalancing-all` runs full pipeline end-to-end
- [ ] `make test-all` passes

---

## Files Created / Modified

| File | Action |
|---|---|
| `mvp/demand/sql/071_create_transfer_network.sql` | Create |
| `mvp/demand/sql/072_create_rebalancing_plan.sql` | Create |
| `mvp/demand/sql/073_create_rebalancing_views.sql` | Create |
| `mvp/demand/config/rebalancing_config.yaml` | Create |
| `mvp/demand/scripts/compute_rebalancing.py` | Create |
| `mvp/demand/api/routers/inv_planning_rebalancing.py` | Create |
| `mvp/demand/api/main.py` | Modify — mount rebalancing router |
| `mvp/demand/frontend/src/tabs/inv-planning/RebalancingPanel.tsx` | Create |
| `mvp/demand/frontend/src/api/queries/inv-planning-rebalancing.ts` | Create |
| `mvp/demand/frontend/src/api/queries/index.ts` | Modify — re-export rebalancing module |
| `mvp/demand/frontend/src/tabs/InvPlanningTab.tsx` | Modify — add Rebalancing sub-tab |
| `mvp/demand/frontend/vite.config.ts` | Verify — `/inv-planning` proxy already covers this path |
| `mvp/demand/Makefile` | Modify — add rebalancing-* targets |
| `mvp/demand/tests/unit/test_rebalancing.py` | Create |
| `mvp/demand/tests/api/test_rebalancing.py` | Create |
| `mvp/demand/frontend/src/tabs/__tests__/RebalancingPanel.test.tsx` | Create |
| `docs/specs/03-inventory-planning/03-12-inventory-rebalancing.md` | Create (this file) |
