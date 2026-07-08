# Financial Planning

> Translates inventory positions and demand forecasts into financial terms -- inventory value, carrying cost, and budget utilization -- so finance teams can plan capital allocation alongside supply chain operations.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inventory Planning (FinancialPlanPanel) |
| **Key Files** | `inv-planning/FinancialPlanPanel.tsx`, `api/routers/operations/financial_plan.py`, `scripts/ops/compute_financial_plan.py`, `config/operations/financial_plan_config.yaml`, `sql/055_create_financial_plan.sql` |

---

## Problem

Supply chain teams track units; finance tracks dollars. Without a bridge, inventory decisions happen without visibility into capital impact. A planner might approve a safety stock increase that looks reasonable in units but represents a $2M carrying cost increase that blows the quarterly budget. Finance needs forward-looking inventory value projections in the same system where planners manage replenishment.

---

## Solution

A computation pipeline joins on-hand inventory, safety stock targets, production forecasts, and item-level unit costs to produce a monthly financial plan. The plan projects inventory value, carrying cost (holding cost as a percentage of inventory value), and budget utilization for each item-location over a configurable horizon. The API aggregates data from `fact_financial_inventory_plan` directly for dashboard display.

---

## How It Works

### Computation Flow

1. Load the latest on-hand quantity and average daily sales per item-location from `fact_inventory_snapshot`.
2. Join `dim_item_cost` for unit cost per item-location (current row where `effective_from`/`effective_to` cover the plan date).
3. Load committed planned-order spend from open `fact_replenishment_exceptions` rows due within the horizon.
4. Load active budget caps from `fact_budget_periods` for the plan date.
5. For each item-location, compute: inventory value, carrying cost, excess value (stock beyond `excess_dos_threshold` days of average daily demand), and budget utilization against the applicable budget cap.
6. Write results to `fact_financial_inventory_plan` (upsert on `item_id, loc, plan_month, plan_version`).
7. Data available for API queries.

### Key Formulas

| Metric | Formula | Stored As |
|---|---|---|
| Inventory Value | `qty_on_hand x unit_cost` | `fact_financial_inventory_plan.projected_inventory_value` |
| Monthly Carrying Cost | `inventory_value x (carrying_cost_pct / 12)` | `fact_financial_inventory_plan.carrying_cost_monthly` |
| Excess Value | `MAX(0, qty_on_hand - avg_daily_demand x excess_dos_threshold) x unit_cost` | `fact_financial_inventory_plan.excess_value` |
| Budget Utilization | `planned_order_value (committed spend) / budget_cap x 100` | Computed live by `GET /finance/budget-status`, not stored |

---

## Data Model

| Table | Purpose | Key Columns |
|---|---|---|
| `dim_item_cost` | Unit cost per item-location | `item_id`, `loc`, `unit_cost`, `cost_type` (`standard`\|`moving_avg`\|`last_purchase`), `currency`, `effective_from`, `effective_to` |
| `fact_budget_periods` | Budget allocations by scope and period | `budget_id`, `scope_type` (`global`\|`category`\|`buyer`\|`location`), `scope_value`, `period_type`, `budget_start`, `budget_end`, `budget_cap`, `carrying_cost_pct` |
| `fact_financial_inventory_plan` | Monthly projections per item-location | `item_id`, `loc`, `plan_month`, `plan_version`, `projected_inventory_value`, `planned_order_value`, `carrying_cost_monthly`, `excess_qty`, `excess_value`, `max_stock_target`, `budget_cap`, `within_budget` |

Unique key on `fact_financial_inventory_plan` is `(item_id, loc, plan_month, plan_version)` -- `plan_version` defaults
to `'latest'` and lets a recompute run alongside a prior snapshot.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/finance/inventory-plan` | Summary financial KPIs + by-category breakdown for a `plan_version` |
| GET | `/finance/budget-status` | Budget utilization across all budget periods |
| GET | `/finance/working-capital-trend` | Monthly time series (history + forward) for charting |
| GET | `/finance/excess-value` | Top SKU-locations ranked by excess inventory value |
| POST | `/finance/budget` | Create a new budget period (auth) |
| PUT | `/finance/budget/{budget_id}` | Update a budget period's cap (auth) |

The API prefix is `/finance`, not `/financial-plan`. There is no compute-trigger endpoint -- recomputation only
happens via the `scripts/ops/compute_financial_plan.py` pipeline below.

---

## Pipeline

| Step | Command | What It Does |
|---|---|---|
| Schema | `make financial-plan-schema` | Creates `dim_item_cost`, `fact_budget_periods`, `fact_financial_inventory_plan` |
| Compute | `make financial-plan-compute` | Runs the full financial plan computation |
| Dry Run | `make financial-plan-dry` | Computes and logs without writing |
| Full | `make financial-plan-all` | Schema + compute |

---

## Configuration

File: `config/operations/financial_plan_config.yaml`

| Key | Purpose | Default |
|---|---|---|
| `financial_plan.months_ahead` | Number of months to project forward | `6` |
| `financial_plan.excess_dos_threshold` | Days-of-supply above which stock is classified as excess | `180` |
| `financial_plan.budget_breach_alert_pct` | Utilization level that triggers a breach alert | `0.90` |
| `financial_plan.target_inventory_turns` | Annual inventory turns target | `8.0` |
| `financial_plan.target_dos` | Target days of supply | `45` |
| `financial_plan.cost_type_priority` | `dim_item_cost.cost_type` fallback order | `[moving_avg, standard, landed, manual]` |
| `carrying_cost_pct` | Annual carrying cost as percent of inventory value (from `shared_constants.yaml` via `_includes`) | inherited |
| `scheduler.cron` | Monthly compute schedule | `0 5 1 * *` |

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
