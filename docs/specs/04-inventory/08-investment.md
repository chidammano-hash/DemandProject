# Investment Optimization

> Computes an efficient frontier of maximum achievable service level at each budget level by ranking DFUs by marginal return on safety stock investment and allocating capital greedily to highest-ROI units first.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inventory Planning |
| **Key Files** | `scripts/compute_investment_plan.py`, `api/routers/inventory/inv_planning_investment.py`, `sql/033_create_investment_plan.sql` |

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

Each DFU receives an allocated quantity and dollar amount:

| Output Column | Meaning |
|---|---|
| `allocated_qty` | Additional SS units funded by budget |
| `allocated_value` | Dollar amount assigned |
| `projected_service_level` | Expected SL after allocation |
| `marginal_roi` | Service gain per dollar for this DFU |
| `priority_rank` | Position in allocation sequence |

### Scenario Support

Planners can run multiple budget scenarios (e.g., $1M, $2M, $5M) to see how the frontier shifts and which DFUs enter/exit the funded set.

---

## Data Model

| Table | Grain | Purpose |
|---|---|---|
| `fact_inventory_investment_plan` | item_id + loc + plan_date | Per-DFU budget allocation |
| `fact_efficient_frontier` | budget_level + plan_date | Frontier curve data points |

DDL: `sql/033_create_investment_plan.sql`

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/inv-planning/investment/summary` | Current allocation summary |
| GET | `/inv-planning/investment/frontier` | Efficient frontier curve data |
| GET | `/inv-planning/investment/detail` | Per-DFU allocation detail |
| POST | `/inv-planning/investment/compute` | Trigger computation with budget param |

Router: `inv_planning_investment.py`

---

## Pipeline

```
make investment-all    # investment-schema + investment-plan
```

| Step | Script | Output |
|---|---|---|
| Compute | `scripts/compute_investment_plan.py` | `fact_inventory_investment_plan` + `fact_efficient_frontier` |

---

## Configuration

Investment parameters are passed at runtime (budget amount, holding cost rate). The script reads safety stock targets and unit costs from existing tables.

---

## Dependencies

- **Upstream:** `fact_safety_stock_targets` (SS quantities and unit costs), `fact_ss_simulation_results` (service level curves), `dim_sku` (ABC class for prioritization)
- **Downstream:** Financial planning (budget utilization), S&OP (investment approval)

---

## See Also

- [03-safety-stock](03-safety-stock.md) -- SS targets and simulation curves are the input
- [07-abc-xyz-supplier](07-abc-xyz-supplier.md) -- ABC class influences prioritization
- [../05-operations/02-financial-planning](../05-operations/02-financial-planning.md) -- Carries investment totals into financial plan
