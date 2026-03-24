# Financial Planning

Translates inventory positions and demand forecasts into financial terms -- inventory value, carrying cost, and budget utilization -- so finance teams can plan capital allocation alongside supply chain operations.

| Field | Value |
|---|---|
| Status | Implemented |
| Spec | 05-operations/02-financial-planning |
| Frontend | `inv-planning/FinancialPlanPanel.tsx` |
| Backend | `api/routers/financial_plan.py`, `scripts/compute_financial_plan.py` |
| Config | `config/financial_plan_config.yaml` |
| SQL | `sql/057_create_financial_plan.sql` |

---

## Problem

Supply chain teams track units; finance tracks dollars. Without a bridge, inventory decisions happen without visibility into capital impact. A planner might approve a safety stock increase that looks reasonable in units but represents a $2M carrying cost increase that blows the quarterly budget. Finance needs forward-looking inventory value projections in the same system where planners manage replenishment.

---

## Solution

A computation pipeline joins on-hand inventory, safety stock targets, production forecasts, and item-level unit costs to produce a monthly financial plan. The plan projects inventory value, carrying cost (holding cost as a percentage of inventory value), and budget utilization for each item-location over a configurable horizon. The API aggregates data from `fact_financial_inventory_plan` directly for dashboard display.

---

## How It Works

### Computation Flow

1. Load current inventory positions from `agg_inventory_monthly`.
2. Join `dim_item_cost` for unit cost per item (standard cost or weighted average cost).
3. Join `fact_safety_stock_targets` for target stock levels.
4. Join `fact_production_forecast` for expected future demand.
5. For each month in the horizon, compute: projected ending inventory, inventory value (units x unit cost), carrying cost (inventory value x annual holding cost rate / 12), and cumulative budget consumption.
6. Write results to `fact_financial_inventory_plan`.
7. Data available for API queries.

### Key Formulas

| Metric | Formula |
|---|---|
| Inventory Value | `ending_inventory_qty x unit_cost` |
| Monthly Carrying Cost | `inventory_value x (annual_holding_cost_pct / 12)` |
| Budget Utilization | `cumulative_carrying_cost / budget_period_amount x 100` |
| Excess Value | `MAX(0, ending_qty - safety_stock_target) x unit_cost` |

---

## Data Model

| Table | Purpose | Key Columns |
|---|---|---|
| `dim_item_cost` | Unit cost per item | `item_id`, `unit_cost`, `cost_type`, `effective_date` |
| `fact_financial_inventory_plan` | Monthly projections per DFU | `item_id`, `loc`, `month`, `ending_qty`, `inventory_value`, `carrying_cost`, `budget_utilization_pct` |
| `fact_budget_periods` | Budget allocations by location and period | `loc`, `period_start`, `period_end`, `budget_amount` |

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/financial-plan/summary` | Aggregated financial KPIs by location or category |
| GET | `/financial-plan/detail` | Item-location level monthly projections |
| GET | `/financial-plan/budget` | Budget periods with utilization tracking |
| POST | `/financial-plan/compute` | Trigger financial plan recomputation |
| GET | `/financial-plan/trend` | Monthly time series for charting |
| GET | `/financial-plan/excess` | Items with inventory value exceeding targets |

---

## Pipeline

| Step | Command | What It Does |
|---|---|---|
| Schema | `make financial-schema` | Creates `dim_item_cost`, `fact_financial_inventory_plan`, `fact_budget_periods` |
| Compute | `make financial-compute` | Runs the full financial plan computation |
| Full | `make financial-all` | Schema + compute |

---

## Configuration

File: `config/financial_plan_config.yaml`

| Key | Purpose | Default |
|---|---|---|
| `horizon_months` | Number of months to project forward | `12` |
| `annual_holding_cost_pct` | Annual carrying cost as percent of inventory value | `0.25` |
| `cost_type` | Which cost to use from `dim_item_cost` | `standard` |
| `budget_warning_threshold_pct` | Utilization level that triggers a warning | `80` |
| `budget_critical_threshold_pct` | Utilization level that triggers a critical alert | `95` |

---

## Dependencies

| Dependency | Reason |
|---|---|
| Inventory snapshot (03-01) | Current on-hand quantities |
| Safety stock targets (03-03) | Target stock levels for excess computation |
| Production forecast (02-08) | Future demand for forward projection |
| S&OP cycle (05-01) | Budget utilization feeds into gap analysis at Stage 3 |

---

## See Also

- `05-operations/01-sop-cycle.md` -- financial gaps surface during S&OP Stage 3
- `03-inventory-planning/08-investment-optimization.md` -- efficient frontier uses financial plan data
- `03-inventory-planning/03-safety-stock.md` -- safety stock targets drive carrying cost projections
