# Investment Optimization

> Computes an efficient frontier of maximum achievable service level at each budget level by ranking DFUs by marginal return on safety stock investment and allocating capital greedily to highest-ROI units first.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inventory Planning |
| **Key Files** | `scripts/inventory/compute_investment_plan.py`, `api/routers/inventory/inv_planning_investment.py`, `sql/033_create_investment_plan.sql` |

---

## Problem

Inventory investment is constrained by capital budgets. Planners need to allocate limited dollars across thousands of DFUs to maximize service level -- but the relationship between incremental inventory spend and service improvement is nonlinear. Adding $10K to an A-item near stockout yields more service gain than $10K to an overstocked C-item.

---

## Solution

An efficient frontier (the curve showing maximum achievable service level for each budget level) computation engine that ranks DFUs by marginal return on inventory investment, allocates budget optimally, and outputs a capital plan per DFU.

---

## How It Works

### Efficient Frontier Construction

1. For each DFU, compute the marginal ROI (return on investment) of adding one unit of safety stock:
   - ROI = (service level gain) / (unit cost * holding cost rate)
   - Service level gain estimated from the safety stock vs service level curve (Monte Carlo output or analytical approximation)

2. Rank all DFU-unit pairs by marginal ROI (highest first)

3. Allocate budget greedily: fund the highest-ROI unit first, then the next, until budget is exhausted

4. Record the cumulative (budget, service level) curve -- this is the efficient frontier

### Budget Allocation Output

Each DFU row records its current and recommended state side by side, plus the incremental delta:

| Output Column | Meaning |
|---|---|
| `current_ss_qty` / `current_ss_value` / `current_csl` | SS quantity, dollar value, and CSL before allocation |
| `recommended_ss_qty` / `recommended_ss_value` / `recommended_csl` | SS quantity, dollar value, and CSL after allocation |
| `ss_increment_qty` / `investment_increment` / `csl_increment` | Delta in units, dollars, and CSL |
| `marginal_roi` | CSL gain per incremental dollar for this DFU |
| `investment_rank` | Position in the greedy allocation sequence |
| `cumulative_investment` | Running total spend up to and including this DFU's rank |

### Scenario Support

Planners can run multiple budget scenarios (e.g., $1M, $2M, $5M) to see how the frontier shifts and which DFUs enter/exit the funded set.

---

## Data Model

| Table | Grain | Purpose |
|---|---|---|
| `fact_inventory_investment_plan` | plan_id + item_id + loc (unique) | Per-DFU current vs. recommended allocation and rank |
| `fact_efficient_frontier` | plan_id + budget_point | Frontier curve data points |

DDL: `sql/033_create_investment_plan.sql`

---

## API

| Method | Path | Params | Purpose |
|---|---|---|---|
| GET | `/inv-planning/investment/efficient-frontier` | `plan_id` (defaults to latest) | Frontier curve plus portfolio-level current vs. recommended investment/CSL |
| GET | `/inv-planning/investment/summary` | none | Latest-plan allocation summary with top-10 ROI items |
| GET | `/inv-planning/investment/detail` | `plan_id`, `item`, `location`, `abc_vol`, `abc_xyz_segment`, `limit` (1-1000, default 50), `offset`, `sort_by`, `sort_dir` | Paginated per-DFU allocation detail |
| POST | `/inv-planning/investment/plan` | `budget_constraint`, `target_csl` (requires API key) | Trigger capital investment optimization; returns `plan_id` |

Router: `api/routers/inventory/inv_planning_investment.py`

---

## Pipeline

```
make investment-all    # investment-schema + investment-plan
```

| Step | Script | Output |
|---|---|---|
| Compute | `scripts/inventory/compute_investment_plan.py` | `fact_inventory_investment_plan` + `fact_efficient_frontier` |

---

## Configuration

Investment parameters are passed at runtime (budget amount, holding cost rate). The script reads safety stock targets and unit costs from existing tables.

---

## Dependencies

- **Upstream:** `fact_safety_stock_targets` (SS quantities), `dim_sku` (ABC class, unit cost), `agg_inventory_monthly` (latest on-hand position)
- **Downstream:** Financial planning (budget utilization), S&OP (investment approval)

---

## See Also

- [03-safety-stock](03-safety-stock.md) -- SS targets are the input
- [07-abc-xyz-supplier](07-abc-xyz-supplier.md) -- ABC class influences prioritization
- [../05-operations/02-financial-planning](../05-operations/02-financial-planning.md) -- Carries investment totals into financial plan
