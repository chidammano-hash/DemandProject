<!-- SOURCE: IPfeature4.md (EOQ & Cycle Stock) -->
# IPfeature4 — EOQ & Cycle Stock Calculator

## EPIC
InventoryPlanning

## Status
Planned

## Priority
P1 — Must Have (Foundation)

## Effort
M (Medium)

## Expert Perspectives
- **Inventory Planning Expert** (lead) — EOQ theory, cycle stock, MOQ constraints
- **Warehouse Space Utilization Expert** — capacity constraints on order quantity
- **Statistical Analyst** — cost function minimization, sensitivity analysis

---

## Problem Statement

IPfeature3 answers "how much safety stock do I need?" IPfeature4 answers the complementary question: "how much should I order each time?"

Without EOQ:
- Over-ordering wastes capital (high holding cost)
- Under-ordering means frequent small orders (high transaction cost)
- No systematic way to set `target_max` or `target_dos_max` (upper inventory bound)

EOQ minimizes total annual cost = ordering cost + holding cost. Combined with SS from IPfeature3, this gives planners the complete `(when to order = ROP, how much to order = EOQ)` prescription.

---

## User Story

> As an inventory planner, I want Economic Order Quantities and cycle stock targets per SKU-location, accounting for ordering costs and holding costs and subject to minimum order quantities, so that I know the optimal replenishment quantity that minimizes total annual inventory cost.

---

## Business Value

- Completes the replenishment prescription: ROP (from IP3) tells *when* to order; EOQ tells *how much*
- Quantifies total annual inventory cost (holding + ordering) as a portfolio metric
- Enables warehouse space planning: `total_target_stock = SS + cycle_stock`
- Enables IPfeature13 (investment optimization) with unit-cost-aware cost modeling

---

## Key Formulas

```
# Annual demand
D_annual = demand_mean_monthly × 12

# Economic Order Quantity (Wilson formula)
EOQ = sqrt(2 × D_annual × ordering_cost / (holding_cost_pct × unit_cost))

# Constrained by MOQ and capital limits
effective_EOQ = max(EOQ, MOQ)
effective_EOQ = min(effective_EOQ, max_eoq_months_supply × demand_mean_monthly)

# Cycle stock (average inventory from replenishment cycle)
cycle_stock = effective_EOQ / 2

# Total target inventory
total_target_stock = SS_combined + cycle_stock

# Order frequency (orders per year)
order_frequency = D_annual / effective_EOQ

# Annual cost components
annual_holding_cost = holding_cost_pct × unit_cost × (effective_EOQ/2 + SS_combined)
annual_order_cost   = ordering_cost × D_annual / effective_EOQ
total_annual_cost   = annual_holding_cost + annual_order_cost

# Target DOS bounds
target_dos_min = SS_combined / D_avg_daily          (SS floor, from IPfeature3)
target_dos_max = total_target_stock / D_avg_daily   (SS + full cycle stock ceiling)
```

**Verification example:**
```
D=1200/yr, ordering_cost=50, holding_cost_pct=0.25, unit_cost=10
EOQ = sqrt(2 × 1200 × 50 / (0.25 × 10)) = sqrt(120000/2.5) = sqrt(48000) ≈ 219.1 units
```

---

## Data Requirements

### DDL Extension: `mvp/demand/sql/024_create_safety_stock_targets.sql` (ALTER TABLE)

The script `compute_eoq.py` runs `ALTER TABLE fact_safety_stock_targets ADD COLUMN IF NOT EXISTS ...` for each new column:

| Column | Type | Description |
|---|---|---|
| `ordering_cost` | NUMERIC(10,2) | $ per purchase order |
| `holding_cost_pct` | NUMERIC(6,4) | Annual holding cost as % of unit value |
| `unit_cost` | NUMERIC(12,4) | $ per unit (from dim_item or config default) |
| `annual_demand` | NUMERIC(15,4) | demand_mean_monthly × 12 |
| `eoq` | NUMERIC(15,4) | Wilson EOQ formula result |
| `eoq_cycle_stock` | NUMERIC(15,4) | EOQ / 2 |
| `total_target_stock` | NUMERIC(15,4) | SS_combined + cycle_stock |
| `order_frequency` | NUMERIC(10,2) | D_annual / effective_EOQ (orders/yr) |
| `moq` | NUMERIC(15,4) | Minimum order quantity |
| `effective_eoq` | NUMERIC(15,4) | max(EOQ, MOQ), capped by months supply |
| `annual_holding_cost` | NUMERIC(12,2) | Holding cost per year |
| `annual_order_cost` | NUMERIC(12,2) | Ordering cost per year |
| `total_annual_cost` | NUMERIC(12,2) | Sum of holding + ordering |

Also updates (overwrites from IPfeature3):
- `target_dos_max` — now set to `total_target_stock / D_avg_daily`
- `target_max_qty` — now set to `total_target_stock`

### New Config: `mvp/demand/config/eoq_config.yaml`

```yaml
eoq:
  default_ordering_cost: 50.00        # $ per PO (override per item via dim_item if available)
  default_holding_cost_pct: 0.25      # 25%/yr (typical CPG/beverage industry)
  default_unit_cost: 1.00             # fallback if dim_item has no cost field
  moq_source: config                  # 'config' (use default_moq) | 'dim_item' (use pack_case)
  default_moq: 1
  max_eoq_months_supply: 6            # cap EOQ at 6 months of demand
```

---

## API Endpoints

**Router:** `mvp/demand/api/routers/inv_planning.py` (extends existing router)

```
GET /inv-planning/eoq/summary
  Query params: item, location, abc_vol
  Response: {
    total_dfus: int,
    avg_eoq: float,
    total_cycle_stock: float,
    avg_order_frequency: float,        -- orders per year
    total_annual_cost: float,
    by_abc: {
      A: { avg_eoq, avg_cycle_stock, avg_order_frequency, total_annual_cost },
      B: { ... },
      C: { ... }
    }
  }
  Cache: max-age=300s

GET /inv-planning/eoq/detail
  Query params: item, location, abc_vol, limit, offset,
                sort_by (eoq | cycle_stock | total_annual_cost | order_frequency), sort_dir
  Response: {
    total: int,
    rows: [ {item_no, loc, abc_vol, annual_demand, eoq, moq, effective_eoq,
             eoq_cycle_stock, total_target_stock, order_frequency,
             annual_holding_cost, annual_order_cost, total_annual_cost,
             target_dos_min, target_dos_max} ]
  }
  Cache: max-age=120s

GET /inv-planning/eoq/sensitivity
  Query params: item (required), location (required),
                ordering_cost_min (default: 10), ordering_cost_max (default: 200), steps (default: 10)
  Response: {
    item_no: str,
    loc: str,
    unit_cost: float,
    holding_cost_pct: float,
    annual_demand: float,
    moq: float,
    curve: [ {ordering_cost, eoq, cycle_stock, annual_holding_cost, annual_order_cost, total_annual_cost} ]
  }
  Cache: max-age=600s
```

---

## Frontend UI

### Panel: "EOQ & Cycle Stock" in `InvPlanningTab.tsx`

**KPI Cards (row of 4):**
| Card | Value |
|---|---|
| Total Cycle Stock | sum(eoq_cycle_stock) across portfolio |
| Avg EOQ Size | avg(effective_eoq) |
| Avg Order Frequency | avg(order_frequency) per year |
| Total Annual Inventory Cost | sum(total_annual_cost) |

**EOQ Sensitivity Chart:**
- Shows how EOQ changes as ordering_cost varies from min to max
- Two lines: EOQ line (left axis) + total_annual_cost line (right axis)
- Vertical marker at current `default_ordering_cost`
- Reveals the trade-off: lower ordering cost → smaller EOQ → more frequent orders

**EOQ Detail Table:**
- Columns: item, loc, annual_demand, eoq, moq, effective_eoq, cycle_stock, total_target_stock (SS+cycle), total_annual_cost
- Sorted by total_annual_cost descending (highest cost items first)
- Color-coded: effective_eoq = moq (MOQ-bound) highlighted amber (EOQ smaller than MOQ)

---

## Backend Script

### `mvp/demand/scripts/compute_eoq.py`

```python
# Algorithm:
# 1. Load eoq_config.yaml
# 2. Load fact_safety_stock_targets: item_no, loc, ss_combined, demand_mean_monthly,
#    avg_daily_demand, target_dos_min (from IPfeature3)
# 3. Load dim_item: item_no, unit_cost (or fallback to config default)
#    If moq_source='dim_item': use pack_case as MOQ
# 4. For each DFU:
#    a. D_annual = demand_mean_monthly * 12
#    b. EOQ = sqrt(2 * D_annual * ordering_cost / (holding_cost_pct * unit_cost))
#       Handle zero D_annual or zero unit_cost edge cases
#    c. effective_EOQ = max(EOQ, MOQ)
#    d. effective_EOQ = min(effective_EOQ, max_eoq_months_supply * demand_mean_monthly)
#    e. cycle_stock = effective_EOQ / 2
#    f. total_target_stock = ss_combined + cycle_stock
#    g. order_frequency = D_annual / effective_EOQ
#    h. annual_holding = holding_cost_pct * unit_cost * (effective_EOQ/2 + ss_combined)
#    i. annual_order = ordering_cost * D_annual / effective_EOQ
#    j. total_annual = annual_holding + annual_order
#    k. target_dos_max = total_target_stock / avg_daily_demand (NULL if zero demand)
#    l. target_max_qty = total_target_stock
# 5. UPDATE fact_safety_stock_targets SET eoq=..., cycle_stock=..., etc.
#    WHERE item_no=... AND loc=... AND policy_version='v1'
```

**CLI Usage:**
```bash
uv run python scripts/compute_eoq.py
uv run python scripts/compute_eoq.py --config config/eoq_config.yaml
```

---

## Makefile Targets

```makefile
eoq-compute:
	uv run python scripts/compute_eoq.py
	# Note: requires ss-compute to have run first

eoq-all: ss-all eoq-compute
```

---

## Dependencies

| Dependency | Type | Notes |
|---|---|---|
| `fact_safety_stock_targets` | IPfeature3 | Must be populated first |
| `dim_item` | Existing | For unit_cost and optional MOQ |
| `config/eoq_config.yaml` | New | Ordering cost, holding cost, MOQ source |

---

## Testing Requirements

### Backend Unit Tests: `mvp/demand/tests/unit/test_eoq.py`

Minimum 15 tests:

**EOQ formula:**
- Verified: D=1200, S=50, H=0.25, C=10 → EOQ ≈ 219.1 (assert abs(result - 219.1) < 0.1)
- D=0: effective_EOQ = MOQ (no demand → order minimum)
- unit_cost=0: fallback to default_unit_cost from config

**MOQ enforcement:**
- EOQ=50, MOQ=100 → effective_EOQ = 100 (MOQ wins)
- EOQ=200, MOQ=100 → effective_EOQ = 200 (EOQ wins)

**Months supply cap:**
- max_eoq_months_supply=6, demand_mean_monthly=100, EOQ=800 → capped at 600
- EOQ=400, cap=600 → not capped (400 < 600)

**Cost computation:**
- annual_holding = H × C × (EOQ/2 + SS): verified numerically
- annual_order = S × D / EOQ: verified numerically
- total = holding + order: always positive

**target_dos_max:**
- D_avg_daily=0: target_dos_max = NULL
- D_avg_daily=10, total_target=200 → target_dos_max = 20 days

**Sensitivity curve:**
- 10 ordering_cost steps from 10 to 200 → 10 EOQ values, all positive
- EOQ increases as ordering_cost increases (monotonic)

### Backend API Tests: `mvp/demand/tests/api/test_inv_planning_eoq.py`

Minimum 8 tests:
- `GET /inv-planning/eoq/summary` → 200 OK, by_abc has A/B/C keys
- `GET /inv-planning/eoq/detail` → rows have effective_eoq >= moq always
- `GET /inv-planning/eoq/sensitivity?item=X&location=Y` → curve with 10 points
- `GET /inv-planning/eoq/sensitivity` (no item) → 422
- Pagination + sort by total_annual_cost desc

---

## Acceptance Criteria

- [ ] EOQ formula verified: D=1200, S=50, H=0.25, C=10 → EOQ ≈ 219.1 in unit tests
- [ ] `effective_eoq >= moq` for every DFU always
- [ ] EOQ capped at `max_eoq_months_supply × demand_mean_monthly`
- [ ] `target_dos_max` = `total_target_stock / avg_daily_demand` (NULL if zero demand)
- [ ] Sensitivity curve is monotonically non-decreasing in EOQ as ordering_cost increases
- [ ] `make test-all` passes

---

## Files Created / Modified

| File | Action |
|---|---|
| `mvp/demand/config/eoq_config.yaml` | Create |
| `mvp/demand/scripts/compute_eoq.py` | Create |
| `mvp/demand/api/routers/inv_planning.py` | Modify — add EOQ endpoints |
| `mvp/demand/frontend/src/tabs/InvPlanningTab.tsx` | Modify — add EOQ panel |
| `mvp/demand/tests/unit/test_eoq.py` | Create |
| `mvp/demand/tests/api/test_inv_planning_eoq.py` | Create |
| `mvp/demand/Makefile` | Modify — add eoq-compute, eoq-all targets |
| `docs/design-specs/IPfeature4.md` | Create (this file) |


---

<!-- SOURCE: IPfeature5.md (Replenishment Policies) -->
# IPfeature5 — Replenishment Policy Management

## EPIC
InventoryPlanning

## Status
Planned

## Priority
P2 — Should Have

## Effort
M (Medium)

## Expert Perspectives
- **Inventory Planning Expert** (lead) — policy types, review cycles, segment mapping
- **UI/UX Expert** — policy configuration UI, compliance gauges, auto-assign flow
- **Warehouse Space Utilization Expert** — review cycle and space utilization trade-offs

---

## Problem Statement

IPfeature3 and IPfeature4 compute SS and EOQ numbers, but these numbers have no enforcement path. There is no concept of *which policy type applies to which DFU* — no framework for "A-class items should be on continuous review, C-class on min-max."

Without a policy layer:
- Planners cannot communicate standard operating procedures to the system
- The system cannot distinguish items that *should* be reviewed daily from those checked monthly
- Compliance cannot be measured (how many items actually follow their policy?)

---

## User Story

> As an inventory manager, I want to define replenishment policy types (continuous review ROP/EOQ, periodic review, min-max, manual) per item segment, assign them automatically to DFUs, and track policy compliance so I have a systematic replenishment framework rather than ad-hoc decisions.

---

## Business Value

- Creates the governance layer connecting DFUs to replenishment rules
- Enables IPfeature7 (exceptions) to know the correct `recommended_order_by` date per policy type
- Enables reporting on "% of portfolio with a defined policy"
- Provides managers a lever to change policy segments en masse (e.g., "reclassify all CZ items to manual review")

---

## Policy Types

| Type | Description | Review Trigger |
|---|---|---|
| `continuous_rop` | Monitor daily; order when position ≤ ROP | On-hand ≤ ROP |
| `periodic_review` | Check inventory every N days; order up to max | Every `review_cycle_days` |
| `min_max` | Order up to max when position falls below min | On-hand ≤ target_min_qty |
| `manual` | No automatic trigger; planner decides | Manual review |

---

## Data Requirements

### New DDL: `mvp/demand/sql/025_create_replenishment_policy.sql`

**Table 1: `dim_replenishment_policy`**
```sql
CREATE TABLE IF NOT EXISTS dim_replenishment_policy (
    policy_sk          BIGSERIAL PRIMARY KEY,
    policy_id          TEXT UNIQUE NOT NULL,
    policy_name        TEXT NOT NULL,
    policy_type        TEXT NOT NULL CHECK (policy_type IN
                       ('continuous_rop','periodic_review','min_max','manual')),
    segment            TEXT,             -- descriptive (e.g. 'A', 'CZ', 'lumpy')
    review_cycle_days  INTEGER,          -- for periodic_review: days between checks
    service_level      NUMERIC(6,4),
    use_eoq            BOOLEAN DEFAULT TRUE,
    use_safety_stock   BOOLEAN DEFAULT TRUE,
    active             BOOLEAN DEFAULT TRUE,
    notes              TEXT,
    created_ts         TIMESTAMPTZ DEFAULT NOW(),
    modified_ts        TIMESTAMPTZ DEFAULT NOW()
);

**Table 2: `fact_dfu_policy_assignment`**
CREATE TABLE IF NOT EXISTS fact_dfu_policy_assignment (
    assignment_sk      BIGSERIAL PRIMARY KEY,
    item_no            TEXT NOT NULL,
    loc                TEXT NOT NULL,
    policy_id          TEXT NOT NULL REFERENCES dim_replenishment_policy(policy_id),
    override_reason    TEXT,
    assigned_by        TEXT DEFAULT 'system',   -- 'system' | 'manual'
    effective_date     DATE NOT NULL,
    created_ts         TIMESTAMPTZ DEFAULT NOW(),
    modified_ts        TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (item_no, loc)
);
CREATE INDEX IF NOT EXISTS idx_dfu_policy_policy_id ON fact_dfu_policy_assignment (policy_id);
CREATE INDEX IF NOT EXISTS idx_dfu_policy_item_loc  ON fact_dfu_policy_assignment (item_no, loc);
```

### New Config: `mvp/demand/config/replenishment_policy_config.yaml`

```yaml
policies:
  - id: A_continuous_v1
    name: "A-Class Continuous Review (ROP/EOQ)"
    type: continuous_rop
    segment: "A"
    service_level: 0.98
    use_eoq: true
    use_safety_stock: true

  - id: B_periodic_v1
    name: "B-Class Periodic Review (4-week)"
    type: periodic_review
    segment: "B"
    review_cycle_days: 28
    service_level: 0.95
    use_eoq: true
    use_safety_stock: true

  - id: C_min_max_v1
    name: "C-Class Min-Max"
    type: min_max
    segment: "C"
    service_level: 0.90
    use_eoq: false
    use_safety_stock: true

  - id: lumpy_manual_v1
    name: "Lumpy/Intermittent — Manual Review"
    type: manual
    segment: "lumpy"
    service_level: 0.85
    use_eoq: false
    use_safety_stock: false

auto_assign:
  enabled: true
  # Priority: variability_class overrides abc_vol
  # lumpy variability_class → lumpy_manual_v1 regardless of ABC class
  # else: A → A_continuous_v1, B → B_periodic_v1, C → C_min_max_v1
  variability_override:
    lumpy: lumpy_manual_v1
```

---

## API Endpoints

**Router:** `mvp/demand/api/routers/inv_planning.py`

```
GET /inv-planning/policies
  Response: {
    policies: [ {policy_id, policy_name, policy_type, segment, review_cycle_days,
                 service_level, use_eoq, use_safety_stock, active, dfu_count} ]
  }
  Cache: max-age=300s

POST /inv-planning/policies
  Auth: require_api_key
  Body: { policy_id, policy_name, policy_type, segment, review_cycle_days,
          service_level, use_eoq, use_safety_stock, notes }
  Response: created policy object

PUT /inv-planning/policies/{policy_id}
  Auth: require_api_key
  Body: (any subset of policy fields)
  Response: updated policy object

GET /inv-planning/policy-assignments
  Query params: item, location, policy_id, assigned_by, limit, offset
  Response: {
    total: int,
    rows: [ {item_no, loc, policy_id, policy_name, policy_type,
             override_reason, assigned_by, effective_date} ]
  }
  Cache: max-age=120s

POST /inv-planning/policy-assignments/assign
  Auth: require_api_key
  Body (individual): { item_no, loc, policy_id, override_reason }
     OR
  Body (bulk by segment): { segment, policy_id }   -- assigns all DFUs in that segment
  Response: { assigned_count, failed_count, already_assigned_count }

GET /inv-planning/policy-assignments/compliance
  Response: {
    total_dfus: int,
    assigned_count: int,
    unassigned_count: int,
    assignment_pct: float,
    by_policy: {
      policy_id: {
        policy_name, policy_type, dfu_count,
        below_ss_pct, avg_ss_coverage, avg_dos
      }
    }
  }
  Cache: max-age=300s
```

---

## Frontend UI

### Panel: "Policy Management" in `InvPlanningTab.tsx`

**Policy Cards (grid):**
- One card per active policy
- Card content: policy_name, policy_type badge (colored), segment, service_level %, DFU count, review_cycle_days (if periodic)
- "Edit" button → opens modal to update policy parameters (calls PUT endpoint, auth required)

**Compliance Section:**
- Ring gauge chart: % of DFUs with a policy assigned (green ring fills)
- KPI: `assigned_count / total_dfus × 100%`
- "Auto-assign All" button → calls POST assign with each policy's segment
- After auto-assign: ring gauge updates

**Policy Compliance Table:**
- One row per policy
- Columns: policy_name, type, DFU count, below_ss_pct, avg_ss_coverage, avg_dos
- Sorted by below_ss_pct descending (policies with worst compliance at top)

---

## Backend Script

### `mvp/demand/scripts/assign_replenishment_policies.py`

```python
# Algorithm:
# 1. Load replenishment_policy_config.yaml
# 2. Upsert all policies into dim_replenishment_policy
#    ON CONFLICT (policy_id) DO UPDATE
# 3. If auto_assign.enabled:
#    Load dim_dfu: item_no (=dmdunit), loc, abc_vol, variability_class
#    For each DFU:
#      a. If variability_class in variability_override → use override policy
#      b. Else: match abc_vol (A/B/C) → corresponding policy
#      c. If abc_vol is NULL or unrecognized → skip (don't assign)
#    Batch upsert into fact_dfu_policy_assignment (assigned_by='system')
#    ON CONFLICT (item_no, loc) DO UPDATE only if assigned_by='system'
#    (manual overrides preserved: assigned_by='manual' rows are NOT overwritten)
# 4. Print summary: {assigned, skipped, preserved_manual}
```

**CLI Usage:**
```bash
uv run python scripts/assign_replenishment_policies.py
uv run python scripts/assign_replenishment_policies.py --dry-run
uv run python scripts/assign_replenishment_policies.py --force-overwrite  # overwrite manual assignments
```

---

## Makefile Targets

```makefile
policy-schema:
	# apply sql/025_create_replenishment_policy.sql

policy-assign:
	uv run python scripts/assign_replenishment_policies.py

policy-all: policy-schema policy-assign
```

---

## Dependencies

| Dependency | Type | Notes |
|---|---|---|
| `dim_dfu.abc_vol` | Existing | ABC class for auto-assignment |
| `dim_dfu.variability_class` | IPfeature1 | variability_class for lumpy override |
| `fact_safety_stock_targets` | IPfeature3 | For compliance reporting (below_ss_pct) |

---

## Testing Requirements

### Backend Unit Tests: `mvp/demand/tests/unit/test_replenishment_policy.py`

Minimum 12 tests:
- Config loading: 4 policies parsed correctly from YAML
- Auto-assign: abc_vol='A', variability_class='low' → A_continuous_v1
- Auto-assign: abc_vol='B', variability_class='medium' → B_periodic_v1
- Auto-assign: abc_vol='A', variability_class='lumpy' → lumpy_manual_v1 (variability_class overrides ABC)
- Auto-assign: abc_vol='C', variability_class='lumpy' → lumpy_manual_v1
- Manual override preserved: assigned_by='manual' not overwritten by auto-assign
- Unknown abc_vol: skip DFU (no assignment)
- Dry-run: no DB writes
- Policy upsert: ON CONFLICT updates policy_name and service_level

### Backend API Tests: `mvp/demand/tests/api/test_inv_planning_policy.py`

Minimum 10 tests:
- `GET /inv-planning/policies` → 200 OK, list of policies with dfu_count
- `POST /inv-planning/policies` without auth → 403
- `POST /inv-planning/policies` with auth → 201 Created
- `PUT /inv-planning/policies/A_continuous_v1` → 200 OK, updated service_level
- `GET /inv-planning/policy-assignments` → paginated rows
- `POST /inv-planning/policy-assignments/assign` (individual) → assigned_count=1
- `POST /inv-planning/policy-assignments/assign` (bulk by segment) → assigned_count>0
- `GET /inv-planning/policy-assignments/compliance` → assignment_pct between 0-100
- Compliance table shows by_policy dict with policy_ids as keys

---

## Acceptance Criteria

- [ ] All 4 default policies upserted into `dim_replenishment_policy` after `make policy-assign`
- [ ] All DFUs with known abc_vol auto-assigned to a policy
- [ ] lumpy variability_class DFUs assigned to `lumpy_manual_v1` regardless of ABC class
- [ ] Manual overrides (`assigned_by='manual'`) not overwritten by auto-assign
- [ ] Compliance endpoint returns `assignment_pct > 0`
- [ ] `make test-all` passes

---

## Files Created / Modified

| File | Action |
|---|---|
| `mvp/demand/sql/025_create_replenishment_policy.sql` | Create |
| `mvp/demand/config/replenishment_policy_config.yaml` | Create |
| `mvp/demand/scripts/assign_replenishment_policies.py` | Create |
| `mvp/demand/api/routers/inv_planning.py` | Modify — add policy endpoints |
| `mvp/demand/frontend/src/tabs/InvPlanningTab.tsx` | Modify — add Policy Management panel |
| `mvp/demand/tests/unit/test_replenishment_policy.py` | Create |
| `mvp/demand/tests/api/test_inv_planning_policy.py` | Create |
| `mvp/demand/Makefile` | Modify — add policy-* targets |
| `docs/design-specs/IPfeature5.md` | Create (this file) |


---

<!-- SOURCE: IPfeature6.md (Health Score) -->
# IPfeature6 — Inventory Health Score Dashboard

## EPIC
InventoryPlanning

## Status
Planned

## Priority
P2 — Should Have

## Effort
M (Medium)

## Expert Perspectives
- **Supply Chain Control Tower Expert** (lead) — composite scoring, triage hierarchy
- **UI/UX Expert** — health tier visualization, drill-down, segment heatmap
- **Inventory Planning Expert** — score component weighting and thresholds

---

## Problem Statement

With IPfeature3 computing safety stock targets and IPfeature6 extending them with EOQ, planners now have the numbers — but reviewing 10,000 DFUs to find which need attention is still impossible.

A planner needs a single number per SKU-location that says: "this item is healthy, monitor, at-risk, or critical." Without it, all items look equal until a stockout occurs.

---

## User Story

> As a supply chain director, I want a single 0–100 health score for each SKU-location combining SS coverage, DOS target adherence, stockout risk history, and forecast accuracy, so that I can instantly see which items need attention without reviewing every row manually.

---

## Business Value

- Converts raw SS/EOQ/stockout data into a single triage signal
- Feeds IPfeature7 (exception queue priority) and IPfeature15 (control tower)
- Enables "critical items first" drill-down and segment health heatmaps
- Provides managers a daily headline: "Portfolio health: 72/100 (↓3 vs last month)"

---

## Health Score Formula

**4 components, 25 points each, summed to 0–100:**

### Component 1: SS Coverage (25 pts)
```
ss_coverage = current_qty_on_hand / ss_combined
score_ss_coverage:
  ss_coverage ≥ 1.5  → 25 (well-stocked above SS)
  ss_coverage ≥ 1.0  → 18 (at SS target)
  ss_coverage ≥ 0.5  → 10 (below SS but some buffer)
  ss_coverage < 0.5  → 0  (critically low)
  SS not computed    → 12 (neutral — no target set yet)
```

### Component 2: DOS Target Adherence (25 pts)
```
current_dos = eom_qty_on_hand / avg_daily_sls
score_dos_target:
  current_dos BETWEEN target_dos_min AND target_dos_max → 25 (within target)
  current_dos > target_dos_max                          → 10 (excess inventory)
  current_dos < target_dos_min AND current_dos > 0      → 5  (below minimum)
  current_dos = 0 (stockout)                            → 0
  No target set                                         → 15 (neutral)
```

### Component 3: Stockout Risk History (25 pts)
```
From mv_inventory_forecast_monthly, count stockout months in last 3 months:
  0 stockout months in last 3  → 25 (no recent stockouts)
  1 stockout month             → 15 (one recent event)
  2 stockout months            → 8  (repeated stockouts)
  3 stockout months            → 0  (chronic stockout)
  No forecast data available   → 20 (assume OK, not critical)
```

### Component 4: Forecast Accuracy (25 pts)
```
recent_wape = 3-month avg WAPE from mv_inventory_forecast_monthly (champion or external)
score_forecast_accuracy:
  recent_wape < 15%  → 25 (excellent)
  recent_wape < 25%  → 20 (good)
  recent_wape < 40%  → 15 (fair)
  recent_wape < 60%  → 8  (poor)
  recent_wape ≥ 60%  → 0  (very poor)
  No data            → 15 (neutral)
```

### Composite Score & Tier
```
health_score = score_ss_coverage + score_dos_target + score_stockout_risk + score_forecast_accuracy
health_tier:
  health_score ≥ 80 → 'healthy'
  health_score ≥ 60 → 'monitor'
  health_score ≥ 40 → 'at_risk'
  health_score < 40 → 'critical'
```

---

## Data Requirements

### New DDL: `mvp/demand/sql/026_create_inventory_health_score.sql`

New materialized view `mv_inventory_health_score`:

```sql
CREATE MATERIALIZED VIEW mv_inventory_health_score AS
WITH latest_inv AS (
    SELECT DISTINCT ON (item_no, loc)
        item_no, loc, month_start,
        eom_qty_on_hand, monthly_sales, avg_daily_sls, latest_lead_time_days
    FROM agg_inventory_monthly
    ORDER BY item_no, loc, month_start DESC
),
recent_stockout AS (
    SELECT item_no, loc,
        SUM(CASE WHEN is_stockout THEN 1 ELSE 0 END) AS stockout_count_3m
    FROM mv_inventory_forecast_monthly
    WHERE month_start >= (SELECT MAX(month_start) FROM mv_inventory_forecast_monthly)
                          - INTERVAL '2 months'
      AND model_id IN ('champion', 'external')
    GROUP BY item_no, loc
),
recent_accuracy AS (
    SELECT item_no, loc,
        SUM(abs_error) / NULLIF(ABS(SUM(actual_demand)), 0) AS recent_wape
    FROM mv_inventory_forecast_monthly
    WHERE month_start >= (SELECT MAX(month_start) FROM mv_inventory_forecast_monthly)
                          - INTERVAL '2 months'
      AND model_id IN ('champion', 'external')
    GROUP BY item_no, loc
),
ss AS (
    SELECT item_no, loc, ss_combined, reorder_point, is_below_ss,
           ss_coverage, target_dos_min, target_dos_max
    FROM fact_safety_stock_targets
    WHERE policy_version = 'v1'
)
SELECT
    l.item_no, l.loc, l.month_start,
    d.cluster_assignment, d.abc_vol, d.variability_class,
    d.seasonality_profile, d.region, d.xyz_class, d.abc_xyz_segment,
    -- Current position
    l.eom_qty_on_hand, l.avg_daily_sls,
    CASE WHEN l.avg_daily_sls > 0
         THEN l.eom_qty_on_hand / l.avg_daily_sls ELSE NULL END AS current_dos,
    -- SS targets
    s.ss_combined, s.reorder_point, s.is_below_ss, s.ss_coverage,
    s.target_dos_min, s.target_dos_max,
    -- Forecast accuracy
    ra.recent_wape,
    -- Score components
    CASE WHEN s.ss_combined IS NULL THEN 12
         WHEN s.ss_coverage >= 1.5 THEN 25
         WHEN s.ss_coverage >= 1.0 THEN 18
         WHEN s.ss_coverage >= 0.5 THEN 10
         ELSE 0 END AS score_ss_coverage,
    CASE WHEN s.target_dos_min IS NULL THEN 15
         WHEN l.avg_daily_sls = 0 THEN 5
         WHEN (l.eom_qty_on_hand / l.avg_daily_sls)
              BETWEEN s.target_dos_min AND s.target_dos_max THEN 25
         WHEN (l.eom_qty_on_hand / l.avg_daily_sls) > s.target_dos_max THEN 10
         ELSE 5 END AS score_dos_target,
    CASE WHEN rs.stockout_count_3m IS NULL THEN 20
         WHEN rs.stockout_count_3m = 0 THEN 25
         WHEN rs.stockout_count_3m = 1 THEN 15
         WHEN rs.stockout_count_3m = 2 THEN 8
         ELSE 0 END AS score_stockout_risk,
    CASE WHEN ra.recent_wape IS NULL THEN 15
         WHEN ra.recent_wape < 0.15 THEN 25
         WHEN ra.recent_wape < 0.25 THEN 20
         WHEN ra.recent_wape < 0.40 THEN 15
         WHEN ra.recent_wape < 0.60 THEN 8
         ELSE 0 END AS score_forecast_accuracy,
    -- Composite
    (score_ss_coverage + score_dos_target + score_stockout_risk
     + score_forecast_accuracy)::INTEGER AS health_score,
    CASE WHEN (composite) >= 80 THEN 'healthy'
         WHEN (composite) >= 60 THEN 'monitor'
         WHEN (composite) >= 40 THEN 'at_risk'
         ELSE 'critical' END AS health_tier
FROM latest_inv l
LEFT JOIN dim_dfu d ON l.item_no = d.dmdunit AND l.loc = d.loc
LEFT JOIN ss s ON l.item_no = s.item_no AND l.loc = s.loc
LEFT JOIN recent_stockout rs ON l.item_no = rs.item_no AND l.loc = rs.loc
LEFT JOIN recent_accuracy ra ON l.item_no = ra.item_no AND l.loc = ra.loc
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_health_score_pk
    ON mv_inventory_health_score (item_no, loc);
CREATE INDEX IF NOT EXISTS idx_health_score_tier
    ON mv_inventory_health_score (health_tier);
CREATE INDEX IF NOT EXISTS idx_health_score_critical
    ON mv_inventory_health_score (health_score)
    WHERE health_tier = 'critical';
CREATE INDEX IF NOT EXISTS idx_health_score_abc
    ON mv_inventory_health_score (abc_vol, health_tier);
```

---

## API Endpoints

**Router:** `mvp/demand/api/routers/inv_planning.py`

```
GET /inv-planning/health/summary
  Query params: abc_vol, cluster_assignment, region, variability_class
  Response: {
    total_dfus: int,
    by_tier: { healthy: int, monitor: int, at_risk: int, critical: int },
    avg_health_score: float,
    score_histogram: [ {bin_start, bin_end, count} × 10 bins ],
    component_avgs: {
      score_ss_coverage, score_dos_target, score_stockout_risk, score_forecast_accuracy
    }
  }
  Cache: max-age=120s

GET /inv-planning/health/detail
  Query params: item, location, health_tier, abc_vol, cluster_assignment, variability_class,
                limit, offset, sort_by (health_score | ss_coverage | current_dos), sort_dir
  Response: {
    total: int,
    rows: [ {item_no, loc, abc_vol, variability_class, health_score, health_tier,
             score_ss_coverage, score_dos_target, score_stockout_risk, score_forecast_accuracy,
             ss_coverage, current_dos, target_dos_min, target_dos_max, is_below_ss,
             recent_wape} ]
  }
  Cache: max-age=120s

GET /inv-planning/health/heatmap
  Query params: group_x (abc_vol | xyz_class), group_y (variability_class | cluster_assignment | region)
  Response: {
    x_labels: [ str ],
    y_labels: [ str ],
    cells: [ {x, y, avg_health_score, count, critical_count} ]
  }
  Cache: max-age=300s
```

---

## Frontend UI

### Panel: "Portfolio Health" in `InvPlanningTab.tsx` (landing section — shown first on tab load)

**4 Status KPI Cards:**
| Tier | Color | Card content |
|---|---|---|
| Healthy | Green | count + %, health_score ≥ 80 |
| Monitor | Yellow | count + %, health_score 60–79 |
| At Risk | Orange | count + %, health_score 40–59 |
| Critical | Red | count + %, health_score < 40 |

**Health Tier Donut Chart:**
- 4 segments, colored by tier
- Center label: "Avg Score: 72"

**ABC × Variability Class Heatmap:**
- Rows: ABC class (A / B / C)
- Columns: variability_class (low / medium / high / lumpy)
- Cell: avg_health_score + count
- Color scale: 80–100=green, 60–79=yellow, 40–59=orange, <40=red
- Click cell → filters detail table below

**Portfolio Health Detail Table:**
- Default sort: health_score ascending (worst first)
- Columns: item, loc, abc_vol, variability_class, health_score, health_tier badge, is_below_ss, current_dos, target_dos_min/max, recent_wape
- Row color: red if critical, orange if at_risk, yellow if monitor, white if healthy
- Filter bar: health_tier dropdown, item input, location input

---

## Backend Script

### `mvp/demand/scripts/refresh_health_scores.py`

Thin wrapper:
```python
# Executes: REFRESH MATERIALIZED VIEW CONCURRENTLY mv_inventory_health_score
# Logs: rows updated, timing
```

**Makefile Targets:**
```makefile
health-schema:
	# apply sql/026_create_inventory_health_score.sql (CREATE MAT VIEW WITH NO DATA)

health-refresh:
	uv run python scripts/refresh_health_scores.py
```

---

## Dependencies

| Dependency | Type | Notes |
|---|---|---|
| `dim_dfu.variability_class` | IPfeature1 | Required for heatmap segmentation |
| `fact_safety_stock_targets` | IPfeature3 | SS coverage component |
| `mv_inventory_forecast_monthly` | Feature 37 | Stockout risk + forecast accuracy components |
| `agg_inventory_monthly` | Existing | Current DOS computation |

---

## Testing Requirements

### Backend API Tests: `mvp/demand/tests/api/test_inv_planning_health.py`

Minimum 8 tests:
- `GET /inv-planning/health/summary` → 200 OK, by_tier has 4 keys
- `GET /inv-planning/health/summary?abc_vol=A` → filtered (only A-class items)
- `GET /inv-planning/health/detail` → rows have health_score in 0–100 range
- `GET /inv-planning/health/detail?health_tier=critical` → all rows health_tier='critical'
- `GET /inv-planning/health/detail` sort by health_score asc → first row has lowest score
- `GET /inv-planning/health/heatmap` → cells list, each with x, y, avg_health_score
- Empty view → returns zeros, not 500
- health_score bounds: all rows 0 ≤ health_score ≤ 100

### Unit Test Coverage (within existing ss tests or new file)
- score formula: ss_coverage=1.6 → score_ss_coverage=25
- score formula: ss_coverage=0.9 → score_ss_coverage=18
- score formula: ss_coverage=0.4 → score_ss_coverage=0
- tier classification: 82 → 'healthy', 61 → 'monitor', 45 → 'at_risk', 38 → 'critical'
- wape score: 0.12 → 25, 0.38 → 15, 0.65 → 0

### Frontend Tests: extend `InvPlanningTab.test.tsx`
- Health panel renders 4 KPI cards
- Donut chart has 4 segments
- Heatmap renders grid cells
- Detail table renders with health_score column

---

## Acceptance Criteria

- [ ] Every DFU has a `health_score` 0–100 and `health_tier` after `make health-refresh`
- [ ] `health_score = sum of 4 components`, each 0–25
- [ ] Items with no SS target get neutral component scores (not 0)
- [ ] `health_tier = 'critical'` for health_score < 40
- [ ] Heatmap shows non-trivial distribution across ABC × variability segments
- [ ] `make test-all` passes

---

## Files Created / Modified

| File | Action |
|---|---|
| `mvp/demand/sql/026_create_inventory_health_score.sql` | Create |
| `mvp/demand/scripts/refresh_health_scores.py` | Create |
| `mvp/demand/api/routers/inv_planning.py` | Modify — add health endpoints |
| `mvp/demand/frontend/src/tabs/InvPlanningTab.tsx` | Modify — add Health panel |
| `mvp/demand/tests/api/test_inv_planning_health.py` | Create |
| `mvp/demand/Makefile` | Modify — add health-* targets |
| `docs/design-specs/IPfeature6.md` | Create (this file) |
