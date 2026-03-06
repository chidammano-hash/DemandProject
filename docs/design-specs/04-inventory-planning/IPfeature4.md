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
