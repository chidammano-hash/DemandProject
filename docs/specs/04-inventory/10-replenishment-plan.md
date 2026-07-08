# Replenishment Plan

> Forward-looking replenishment plan generator that combines production forecast confidence intervals, safety stock targets, current inventory positions, and policy parameters to produce a month-by-month order schedule per DFU, with each row compared against its stored historical safety-stock baseline.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inventory Planning |
| **Key Files** | `scripts/inventory/compute_replenishment_plan.py`, `api/routers/inventory/inv_planning_replenishment.py`, `config/inventory/replenishment_plan_config.yaml` |

---

## Problem

Safety stock targets and replenishment policies define *what* to maintain, but planners need a forward-looking *action plan*: which items to order, when, and how much, covering the next several months. Without a consolidated replenishment plan, planners manually calculate order quantities per DFU each cycle.

---

## Solution

A forward-looking replenishment plan generator that combines production forecast confidence intervals (CI bands), safety stock targets, current inventory positions, and policy parameters to produce a month-by-month order schedule per DFU. Each row also carries the DFU's `historical_ss` (from `fact_safety_stock_targets`) alongside the newly-computed `ss_combined`, so planners can see whether the forward-looking recommendation is raising or lowering safety stock versus the currently-live target.

---

## How It Works

### Plan Generation Logic

For each DFU, for each planning month in the horizon:

1. **Project inventory forward:** Starting position - forecast demand + scheduled receipts (open POs)
2. **Check policy trigger:** Does projected position hit the reorder point or review date?
3. **Compute order qty:** Based on policy type:

| Policy | Order Quantity |
|---|---|
| Fixed Quantity (s,Q) | EOQ when position <= ROP |
| Min/Max (s,S) | Up to max level when position <= min |
| Periodic Review (R,S) | Up to target at each review interval |
| Demand-Driven (DDMRP) | Refill to green zone when buffer penetrated |

4. **Derive demand variability from CI bands:** `sigma_demand_monthly = (forecast_qty_upper - forecast_qty_lower) / (2 * ci_z_score)`, using the production forecast's P10/P90 confidence interval as an 80%-confidence spread (`sigma_method = 'ci_spread'`). If CI bands are NULL, falls back to `dim_sku.demand_std` (`sigma_method = 'historical_fallback'`).

5. **Compare to the historical SS baseline:** Each row also carries `historical_ss` (the DFU's currently-live `fact_safety_stock_targets.ss_combined`) alongside the newly-computed `ss_combined`, plus the delta (`ss_delta`, `ss_delta_pct`). This is a safety-stock comparison, not an order-quantity trend line - there is no trailing-order-history baseline in this pipeline.

### Output Columns

Selected columns from `fact_replenishment_plan` (full DDL: `sql/041_create_replenishment_plan.sql`):

| Column | Type | Purpose |
|---|---|---|
| `plan_version`, `item_id`, `loc`, `plan_month` | TEXT/TEXT/TEXT/DATE | Unique key |
| `horizon_months` | SMALLINT | Forward horizon position (1 = T+1, 2 = T+2, ...) |
| `policy_id`, `policy_type`, `abc_vol` | TEXT | Policy and ABC context snapshot at compute time |
| `forecast_qty`, `forecast_qty_lower`, `forecast_qty_upper` | NUMERIC | Point forecast and P10/P90 CI bounds |
| `sigma_method` | TEXT | `ci_spread` or `historical_fallback` |
| `ss_combined` | NUMERIC | Recommended forward-looking safety stock |
| `historical_ss`, `ss_delta`, `ss_delta_pct` | NUMERIC | Currently-live SS baseline and the delta vs. `ss_combined` |
| `eoq`, `effective_eoq`, `cycle_stock` | NUMERIC | EOQ and cycle stock |
| `reorder_point`, `order_qty`, `order_up_to_level` | NUMERIC | Policy-specific replenishment parameters (nullable by policy type) |
| `is_below_ss` | BOOLEAN | `current_qty_on_hand < ss_combined` |

---

## Data Model

| Table | Grain | Purpose |
|---|---|---|
| `fact_replenishment_plan` | plan_version + item_id + loc + plan_month (unique) | Forward order schedule, one row per DFU per forward month per plan version |

DDL: `sql/041_create_replenishment_plan.sql`

---

## API

| Method | Path | Params | Purpose |
|---|---|---|---|
| GET | `/inv-planning/replenishment/summary` | `plan_version`, `policy_type`, `abc_vol` | Portfolio totals for the first plan month, plus a by-policy-type breakdown |
| GET | `/inv-planning/replenishment/detail` | `item`, `location`, `policy_type`, `abc_vol`, `is_below_ss`, `plan_version`, `plan_month`, `limit` (1-500, default 50), `offset`, `sort_by`, `sort_dir` | Paginated per-DFU monthly plan rows |
| GET | `/inv-planning/replenishment/comparison` | `plan_version`, `abc_vol`, `policy_type` | Forecast SS (`ss_combined`) vs. historical SS by ABC class, with increased/decreased/unchanged counts |
| GET | `/inv-planning/replenishment/dfu` | `item_id` (required), `loc` (required), `plan_version` | Full forward time series for a single DFU; 404 if none found |

Router: `api/routers/inventory/inv_planning_replenishment.py`

There is no plan-generation endpoint - the plan is written by the pipeline script below, not triggered via the API.

---

## Pipeline

```
make replplan-schema    # create fact_replenishment_plan table (one-time)
make replplan-compute   # compute forward replenishment plan from production forecast CI bands
make replplan-all       # replplan-schema + replplan-compute
```

| Step | Script | Output |
|---|---|---|
| Generate plan | `scripts/inventory/compute_replenishment_plan.py` | `fact_replenishment_plan` rows |

---

## Configuration

File: `config/inventory/replenishment_plan_config.yaml`, `replenishment_plan` section (includes `shared_constants.yaml` for service levels, z-table, and cost defaults):

```yaml
replenishment_plan:
  sigma_method: ci_spread        # derive sigma_demand_monthly from CI bands
  ci_confidence: 0.80            # P10/P90 = 80% interval
  ci_z_score: 1.282              # z for 80% CI
  fallback_to_historical: true   # use dim_sku.demand_std when CI bands are NULL
  lt_default_days: 14            # fallback when dim_item_lead_time_profile has no entry
  eoq_annualization_months: 12   # forward months summed for annualized demand
```

---

## Dependencies

- **Upstream:** `fact_production_forecast` (demand + CI bands), `fact_dfu_policy_assignment` (policy type + params), `fact_safety_stock_targets` (historical SS baseline), `fact_inventory_snapshot` (current position), `dim_sku` (historical demand_std fallback)
- **Downstream:** Procurement, financial planning (projected spend)

---

## See Also

- [04-replenishment](04-replenishment.md) -- Policy definitions that govern order logic
- [03-safety-stock](03-safety-stock.md) -- SS targets define reorder points
- [../02-forecasting/08-production-forecast](../02-forecasting/08-production-forecast.md) -- Forecast + CI bands
- [11-rebalancing](11-rebalancing.md) -- Transfers may substitute for new orders
