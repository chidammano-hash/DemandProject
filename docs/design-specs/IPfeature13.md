# IPfeature13 — Capital & Space Investment Optimization

## EPIC
InventoryPlanning

## Status
Planned

## Priority
P3 — Nice to Have

## Effort
XL (Extra Large)

## Expert Perspectives
- **Warehouse Space Utilization Expert** (lead) — capital constraints, space vs. service trade-offs
- **Inventory Planning Expert** — efficient frontier theory, marginal SS investment
- **Simulation Expert** — CSL estimation per SS level, curve construction

---

## Problem Statement

Businesses face capital constraints: "We have $5M budgeted for inventory. Where should we invest it to maximize service level?"

Current system computes recommended SS for each DFU independently (IPfeature3), but doesn't answer the portfolio question: **"Given limited capital, which items' SS should I fund first to get the biggest service level improvement per dollar?"**

This is a **resource allocation problem** — classic operations research. The efficient frontier shows every achievable (capital, service_level) combination, and the marginal ROI ranking tells planners exactly which items to fund first.

---

## User Story

> As a supply chain director, I want to see the portfolio-level trade-off between total inventory investment and service level — the efficient frontier — and a ranked list of items by marginal CSL gain per dollar of additional safety stock, so I can make defensible capital allocation decisions under budget constraints.

---

## Business Value

- Answers "where should we invest our inventory budget?" with data
- Provides a defensible business case for increasing inventory investment (shows expected CSL gain)
- Enables scenario planning: "If we cut inventory by $2M, what service level will we lose?"
- Positions the platform as a strategic decision-support tool, not just an analytics tool

---

## Key Math

```
# Per DFU
current_ss_value       = current_qty_on_hand × unit_cost
recommended_ss_value   = recommended_ss_qty × unit_cost     (from IPfeature3)
investment_increment   = (recommended_ss - current_ss) × unit_cost  [if positive; i.e., underfunded]

# CSL estimation
current_csl    = from simulation (IPfeature10) if available, else analytical estimate
                 analytical estimate: P(demand_during_lt <= current_on_hand)
                 = Φ((current_on_hand - D_avg_daily × LT_mean) / (Z_normalizer))
                 ≈ 1 - (1 - ss_coverage × service_level_target)  [simplified]
recommended_csl = service_level_target (from safety_stock_config for abc_vol)
csl_increment   = recommended_csl - current_csl

# Marginal ROI
marginal_roi = csl_increment / max(investment_increment, 1)   [avoid division by zero]
              (CSL gain per $ invested)

# Efficient frontier construction
Sort items by marginal_roi descending (fund highest-ROI items first)
Cumulative sums:
  cumulative_investment[i] = sum(investment_increment[0..i])
  cumulative_csl_gain[i]   = approximated portfolio CSL improvement
                             (complex: requires portfolio-level CSL model)
                             Simplified: use avg(csl_gain per item) × funded_items / total_items

Budget constraint: Find all items where cumulative_investment <= budget
→ Show achievable portfolio CSL at that budget
```

**Efficient Frontier interpretation:**
- The curve shows every achievable (budget, CSL) combination
- Steep part of curve = high ROI region (cheap CSL gains)
- Flat part of curve = diminishing returns (expensive CSL gains)
- Budget-optimal point: where the curve "bends" (elbow of the frontier)

---

## Data Requirements

### New DDL: `mvp/demand/sql/033_create_investment_plan.sql`

New table `fact_inventory_investment_plan`:

```sql
CREATE TABLE IF NOT EXISTS fact_inventory_investment_plan (
    plan_sk                BIGSERIAL PRIMARY KEY,
    plan_id                TEXT NOT NULL,
    item_no                TEXT NOT NULL,
    loc                    TEXT NOT NULL,
    computation_date       DATE NOT NULL,
    -- Current state
    current_ss_qty         NUMERIC(15,4),
    current_ss_value       NUMERIC(12,2),
    current_csl            NUMERIC(6,4),
    -- Recommended state
    recommended_ss_qty     NUMERIC(15,4),
    recommended_ss_value   NUMERIC(12,2),
    recommended_csl        NUMERIC(6,4),
    -- Incremental analysis
    ss_increment_qty       NUMERIC(15,4),      -- recommended - current (positive = need more SS)
    investment_increment   NUMERIC(12,2),      -- $ to fund the gap
    csl_increment          NUMERIC(6,4),       -- expected CSL gain
    marginal_roi           NUMERIC(10,4),      -- csl_increment / investment_increment
    -- Ranking
    investment_rank        INTEGER,            -- sorted by marginal_roi DESC
    cumulative_investment  NUMERIC(15,2),      -- running sum of investment_increment
    -- Metadata
    abc_vol                TEXT,
    abc_xyz_segment        TEXT,
    unit_cost              NUMERIC(12,4),
    created_ts             TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (plan_id, item_no, loc)
);
CREATE INDEX IF NOT EXISTS idx_inv_plan_plan_id
    ON fact_inventory_investment_plan (plan_id, investment_rank);
CREATE INDEX IF NOT EXISTS idx_inv_plan_item_loc
    ON fact_inventory_investment_plan (item_no, loc);
CREATE INDEX IF NOT EXISTS idx_inv_plan_marginal_roi
    ON fact_inventory_investment_plan (marginal_roi DESC);

-- Efficient frontier: pre-computed budget-to-CSL mapping for the plan
CREATE TABLE IF NOT EXISTS fact_efficient_frontier (
    frontier_sk       BIGSERIAL PRIMARY KEY,
    plan_id           TEXT NOT NULL,
    budget_point      NUMERIC(15,2),          -- cumulative_investment value at this point
    items_funded      INTEGER,                -- how many items funded at this budget
    achievable_csl    NUMERIC(6,4),           -- estimated portfolio CSL at this budget
    marginal_item     TEXT,                   -- item_no of next item to fund
    created_ts        TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_frontier_plan ON fact_efficient_frontier (plan_id, budget_point);
```

---

## API Endpoints

**Router:** `mvp/demand/api/routers/inv_planning.py`

```
GET /inv-planning/investment/efficient-frontier
  Query params: plan_id (default: latest)
  Response: {
    plan_id: str,
    computation_date: date,
    total_items: int,
    current_portfolio_investment: float,
    recommended_portfolio_investment: float,
    current_portfolio_csl: float,
    recommended_portfolio_csl: float,
    curve: [
      { budget, items_funded, achievable_csl, marginal_item }
      -- one point per item, ordered by investment_rank
    ]
  }
  Cache: max-age=300s

GET /inv-planning/investment/summary
  Query params: plan_id (default: latest)
  Response: {
    total_items: int,
    total_current_investment: float,
    total_recommended_investment: float,
    total_investment_gap: float,
    portfolio_csl_current: float,
    portfolio_csl_recommended: float,
    top_roi_items: [ {item_no, loc, marginal_roi, investment_increment, csl_increment} × 10 ]
  }
  Cache: max-age=300s

GET /inv-planning/investment/detail
  Query params: plan_id (default: latest), item, location, abc_vol, abc_xyz_segment,
                limit, offset, sort_by (marginal_roi | investment_increment | csl_increment), sort_dir
  Response: {
    total: int,
    rows: [ {item_no, loc, abc_vol, abc_xyz_segment,
             current_ss_qty, current_ss_value, current_csl,
             recommended_ss_qty, recommended_ss_value, recommended_csl,
             ss_increment_qty, investment_increment, csl_increment, marginal_roi,
             investment_rank, cumulative_investment} ]
  }
  Cache: max-age=120s

POST /inv-planning/investment/plan
  Auth: require_api_key
  Body: {
    budget_constraint: float (optional),    -- if set, only fund items within budget
    target_csl: float (optional)            -- portfolio CSL target
  }
  Response: { plan_id: str, status: 'queued' }
  -- Runs as background job
```

---

## Frontend UI

### Panel: "Investment Optimization" in `InvPlanningTab.tsx`

**KPI Cards (row of 4):**
| Card | Value |
|---|---|
| Current Portfolio Investment | total_current_investment $ |
| Recommended Investment | total_recommended_investment $ |
| CSL Gap | portfolio_csl_recommended - portfolio_csl_current (%) |
| Top ROI Item | item_no of investment_rank=1 |

**Efficient Frontier Chart (main visualization):**
- X-axis: cumulative investment budget ($)
- Y-axis: achievable portfolio CSL %
- Curve: monotonically non-decreasing (each funded item adds to CSL)
- Vertical dashed line: current portfolio investment
- Vertical slider: user-draggable budget constraint → shows achievable CSL
- Horizontal dashed line: target CSL (from config)
- Tooltip at each point: items_funded, achievable_csl, marginal_item (next item to fund)
- Curve "elbow" highlighted: optimal budget point (biggest CSL gain per dollar)

**Budget Slider:**
- Draggable slider from $0 to max(recommended_portfolio_investment)
- Shows: "At $X budget: fund N items, achieve Y% CSL"
- Updates annotation on chart in real time (client-side, no API call)

**"Run Plan" Button:**
- Opens modal with optional budget_constraint and target_csl inputs
- Calls POST /investment/plan → shows queued status (polls job status)
- On completion: refreshes chart and table

**Item Ranking Table:**
- Sorted by investment_rank ascending (fund first = top row)
- Columns: rank, item, loc, abc_xyz_segment, current_ss, recommended_ss, investment_increment, csl_increment, marginal_roi
- Rows colored: funded (within budget, green background) vs. unfunded (gray)
- As budget slider moves, funded rows update dynamically

---

## Backend Script

### `mvp/demand/scripts/compute_investment_plan.py`

```python
# Registered as job type 'investment_plan' in common/job_registry.py

def compute_investment_plan_job(budget_constraint=None, target_csl=None):
    # 1. Generate plan_id = uuid4()
    # 2. Load fact_safety_stock_targets: item_no, loc, ss_combined, unit_cost, abc_vol
    # 3. Load agg_inventory_monthly: latest eom_qty_on_hand per item-loc
    # 4. Load dim_dfu: abc_xyz_segment
    # 5. For each DFU:
    #    current_ss_qty = current_on_hand (using current position as proxy)
    #    current_ss_value = current_ss_qty × unit_cost
    #    recommended_ss_qty = ss_combined (from IPfeature3)
    #    recommended_ss_value = recommended_ss_qty × unit_cost
    #    recommended_csl = service_level_target (from abc_vol → ss_config)
    #    # Estimate current_csl analytically:
    #    if current_ss_qty >= recommended_ss_qty:
    #        current_csl = recommended_csl   (already at or above SS target)
    #    else:
    #        # Linear interpolation: scale CSL by coverage ratio
    #        coverage = current_ss_qty / max(recommended_ss_qty, 1)
    #        current_csl = base_csl + (recommended_csl - base_csl) × coverage
    #        # base_csl = CSL at SS=0 ≈ 0.5 (median service level by definition)
    #    ss_increment_qty = max(0, recommended_ss_qty - current_ss_qty)
    #    investment_increment = ss_increment_qty × unit_cost
    #    csl_increment = recommended_csl - current_csl
    #    marginal_roi = csl_increment / max(investment_increment, 1)
    #
    # 6. Sort by marginal_roi DESC → assign investment_rank
    # 7. Compute cumulative_investment (running sum)
    # 8. Build efficient frontier:
    #    For each rank i:
    #      budget_point = cumulative_investment[i]
    #      achievable_csl = weighted avg of (current_csl[j<i] updated to recommended + others staying current)
    #      INSERT INTO fact_efficient_frontier
    # 9. INSERT INTO fact_inventory_investment_plan
    # 10. Update job_history with plan_id
```

**Job Registration:** Add to `common/job_registry.py`:
```python
JOB_TYPE_REGISTRY['investment_plan'] = {
    'group': 'champion',    # reuse existing group
    'callable': compute_investment_plan_job,
    'description': 'Capital Investment Optimization Plan'
}
```

**Makefile Targets:**
```makefile
investment-schema:
	# apply sql/033_create_investment_plan.sql

investment-plan:
	uv run python scripts/compute_investment_plan.py
	# Runs as background job
```

---

## Dependencies

| Dependency | Type | Notes |
|---|---|---|
| `fact_safety_stock_targets` | IPfeature3 | SS quantities and service level targets |
| `fact_safety_stock_targets.unit_cost` | IPfeature4 | Dollar value computation |
| `fact_ss_simulation_results` | IPfeature10 | Better CSL estimates (optional, falls back to analytical) |
| `dim_dfu.abc_xyz_segment` | IPfeature11 | Segment for display in ranking table |
| `common/job_registry.py` | Existing | Background job execution |
| `agg_inventory_monthly` | Existing | Current on-hand position |

---

## Testing Requirements

### Backend Unit Tests: `mvp/demand/tests/unit/test_investment_plan.py`

Minimum 15 tests:
- Efficient frontier is monotonically non-decreasing: `achievable_csl[i+1] >= achievable_csl[i]`
- Items sorted by marginal_roi descending: first item has highest ROI
- `marginal_roi = csl_increment / investment_increment` (numerically verified)
- investment_increment = 0 if current_ss >= recommended_ss (already funded)
- current_csl = recommended_csl if current_ss >= recommended_ss
- Tie-breaking: equal marginal_roi → secondary sort by investment_increment ascending (cheapest first)
- budget_constraint = 0 → no items funded, achievable_csl = portfolio_csl_current
- budget_constraint = Infinity → all items funded, achievable_csl = portfolio_csl_recommended
- cumulative_investment[0] = investment_increment of rank-1 item
- cumulative_investment[n] = sum of all investment_increments
- plan_id uniqueness: two runs produce different plan_ids

### Backend API Tests: `mvp/demand/tests/api/test_inv_planning_investment.py`

Minimum 10 tests:
- `GET /inv-planning/investment/efficient-frontier` → 200 OK, curve is list
- Curve points: `budget` values are monotonically increasing
- Curve points: `achievable_csl` values are monotonically non-decreasing
- `GET /inv-planning/investment/summary` → top_roi_items is list of ≤10 items
- `GET /inv-planning/investment/detail` → rows with investment_rank starting at 1
- Sort by marginal_roi desc → first row has highest marginal_roi
- `POST /inv-planning/investment/plan` without auth → 403
- `POST /inv-planning/investment/plan` with auth → {plan_id, status='queued'}
- Pagination on detail endpoint

---

## Acceptance Criteria

- [ ] Efficient frontier is monotonically non-decreasing (more budget → equal or better CSL)
- [ ] Items ranked by `marginal_roi` descending: investment_rank=1 always has highest ROI
- [ ] `cumulative_investment[N] = sum(investment_increment[1..N])`
- [ ] Budget slider on frontend updates funded items count dynamically (client-side)
- [ ] Plan runs as background job via JobManager
- [ ] `make test-all` passes

---

## Files Created / Modified

| File | Action |
|---|---|
| `mvp/demand/sql/033_create_investment_plan.sql` | Create |
| `mvp/demand/scripts/compute_investment_plan.py` | Create |
| `mvp/demand/common/job_registry.py` | Modify — add investment_plan job type |
| `mvp/demand/api/routers/inv_planning.py` | Modify — add investment endpoints |
| `mvp/demand/frontend/src/tabs/InvPlanningTab.tsx` | Modify — add Investment Optimization panel |
| `mvp/demand/tests/unit/test_investment_plan.py` | Create |
| `mvp/demand/tests/api/test_inv_planning_investment.py` | Create |
| `mvp/demand/Makefile` | Modify — add investment-* targets |
| `docs/design-specs/IPfeature13.md` | Create (this file) |
