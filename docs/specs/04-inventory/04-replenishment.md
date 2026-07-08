# EOQ, Replenishment Policies & Health Scores

> Three interconnected modules: an EOQ calculator using the Wilson formula, a policy engine that auto-assigns Fixed Quantity/Min-Max/Periodic Review/DDMRP policies by ABC-XYZ segment, and a 4-component 100-point health score surfacing the weakest inventory positions.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inventory Planning |
| **Key Files** | `scripts/compute_eoq.py`, `scripts/assign_replenishment_policy.py`, `config/inventory/eoq_config.yaml`, `sql/026_create_inventory_health_score.sql` |

---

## Problem

Three interconnected replenishment questions: (1) *How much* to order each cycle -- the Economic Order Quantity (EOQ) balances ordering cost vs holding cost; (2) *Which policy* governs each DFU -- fixed quantity, min/max, periodic review, or demand-driven; (3) *How healthy* is each DFU's inventory position -- a composite score that flags items needing attention.

---

## Solution

Three modules that work together: an EOQ calculator using the Wilson formula, a policy management engine that auto-assigns policies by ABC-XYZ segment, and a 4-component health score (100-point scale) that surfaces the weakest inventory positions.

---

## How It Works

### EOQ (IPfeature4)

**Wilson formula:** `EOQ = sqrt(2 * D * S / H)`

Where D = annual demand, S = ordering cost per order, H = holding cost per unit per year.

| Parameter | Source | Default |
|---|---|---|
| D (annual demand) | `agg_inventory_monthly` trailing 12m sales * 12 | Computed |
| S (ordering cost) | Config | $50 |
| H (holding cost) | Config % * unit cost | 25% of unit cost |

Effective EOQ applies guard rails: MOQ (minimum order quantity) floor, max months-of-supply cap. Sensitivity analysis shows cost curve at +/- 20% around optimal EOQ.

Output: `fact_eoq_targets` table with `eoq_qty`, `effective_eoq`, `annual_ordering_cost`, `annual_holding_cost`, `total_cost`.

### Replenishment Policies (IPfeature5)

4 policy types:

| Policy | Trigger | Order Qty | Best For |
|---|---|---|---|
| Fixed Quantity (s,Q) | Stock hits reorder point | Fixed Q (= EOQ) | Stable demand, A items |
| Min/Max (s,S) | Stock hits min | Up to max level | Variable demand |
| Periodic Review (R,S) | Fixed review interval | Up to target S | B/C items, batch ordering |
| Demand-Driven (DDMRP) | Buffer penetration | To green zone | High-variability items |

Auto-assignment rules map ABC-XYZ segments to default policies:

| Segment | Default Policy |
|---|---|
| AX, AY | Fixed Quantity |
| AZ, BX | Min/Max |
| BY, BZ, CX | Periodic Review |
| CY, CZ | Periodic Review (longer interval) |

Planners can override assignments. Compliance tracking compares actual ordering against policy parameters.

### Health Scores (IPfeature6)

4 components, 25 points each (100-point scale):

| Component | 25 pts (good) | 0 pts (poor) |
|---|---|---|
| SS coverage | Current stock >= SS target | Stock << SS |
| DOS target | DOS within policy range | DOS far outside range |
| Stockout risk | No stockout in 3 months | Active or recent stockout |
| Forecast accuracy | WAPE < 15% | WAPE > 40% |

Materialized view `mv_inventory_health_score` LEFT JOINs `fact_safety_stock_targets`. When SS targets are not yet computed, neutral scores (12.5/25) flow automatically via the stub table pattern.

---

## Data Model

| Table / View | Purpose |
|---|---|
| `fact_eoq_targets` | EOQ computation output per DFU |
| `dim_replenishment_policy` | 4 policy type definitions |
| `fact_dfu_policy_assignment` | DFU-to-policy mapping |
| `mv_inventory_health_score` | 4-component health score per DFU |

DDL: `sql/024_create_eoq_targets.sql`, `sql/025_create_replenishment_policy.sql`, `sql/026_create_inventory_health_score.sql`

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/inv-planning/eoq/summary` | Portfolio EOQ summary with by-ABC breakdown |
| GET | `/inv-planning/eoq/detail` | Paginated per-item-location EOQ detail (filter, sort) |
| GET | `/inv-planning/eoq/sensitivity` | EOQ sensitivity curve as ordering cost varies +/- 20% |
| GET | `/inv-planning/policies` | List all policy definitions with DFU assignment counts |
| POST | `/inv-planning/policies` | Create a new policy definition (201, auth required) |
| PUT | `/inv-planning/policies/{policy_id}` | Update an existing policy by ID (auth required) |
| GET | `/inv-planning/policy-assignments` | Paginated DFU-to-policy assignments |
| POST | `/inv-planning/policy-assignments/assign` | Assign a policy to one DFU, or bulk by segment (auth required) |
| GET | `/inv-planning/policy-assignments/compliance` | Portfolio-level policy compliance metrics |
| GET | `/inv-planning/health/summary` | Score distribution + worst performers |
| GET | `/inv-planning/health/detail` | Per-DFU component breakdown |
| GET | `/inv-planning/health/heatmap` | Location x category health heatmap |

Routers: `inv_planning_eoq.py`, `inv_planning_policy.py`, `inv_planning_health.py`

Policy endpoints span two resource families: `/inv-planning/policies` is CRUD on policy
definitions (`dim_replenishment_policy`); `/inv-planning/policy-assignments` is DFU-to-policy
assignment and compliance tracking (`fact_dfu_policy_assignment`). The assign endpoint accepts
either an individual body (`item_id`, `loc`, `policy_id`, `override_reason`) or a bulk body
(`segment`, `policy_id`) that assigns every DFU matching the ABC or variability segment.

---

## Pipeline

```
make eoq-all       # eoq-schema + eoq-compute
make policy-all    # policy-schema + policy-assign
make health-all    # health-schema + health-refresh
```

---

## Configuration

File: `config/inventory/eoq_config.yaml`

```yaml
ordering_cost: 50.0
holding_cost_pct: 0.25
moq: 1
max_eoq_months_supply: 6
```

File: `config/inventory/replenishment_policy_config.yaml`

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
    type: periodic_review
    segment: "B"
    review_cycle_days: 28
    service_level: 0.95
  - id: C_min_max_v1
    type: min_max
    segment: "C"
    service_level: 0.90
    use_eoq: false
  - id: lumpy_manual_v1
    type: manual
    segment: "lumpy"
    service_level: 0.85
    use_eoq: false
    use_safety_stock: false

auto_assign:
  enabled: true
  variability_override:      # overrides the ABC-based mapping above
    lumpy: lumpy_manual_v1

# Separate 9-cell ABC-XYZ policy matrix (DOS targets + service levels, consumed by
# classify_abc_xyz.py rather than the policy assignment above):
abc_xyz_policies:
  AX: {dos_min: 14, dos_max: 21, service_level: 0.98}
  AZ: {dos_min: 28, dos_max: 42, service_level: 0.95}
  BY: {dos_min: 28, dos_max: 35, service_level: 0.93}
```

---

## Dependencies

- **Upstream:** `agg_inventory_monthly`, `fact_safety_stock_targets`, `dim_sku` (ABC-XYZ class), forecast accuracy views
- **Downstream:** Exception queue (policy violations), replenishment plan (policy parameters), investment optimization (EOQ costs)

---

## See Also

- [03-safety-stock](03-safety-stock.md) -- SS targets feed health score and ROP calculation
- [05-exception-queue](05-exception-queue.md) -- Policy violations generate exceptions
- [07-abc-xyz-supplier](07-abc-xyz-supplier.md) -- Segments drive auto-assignment
