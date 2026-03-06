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
