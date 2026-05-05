# Replenishment Plan

> Forward-looking replenishment plan generator that combines production forecast confidence intervals, safety stock targets, current inventory positions, and policy parameters to produce a month-by-month order schedule per DFU with baseline deviation flagging.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inventory Planning |
| **Key Files** | `scripts/compute_replenishment_plan.py`, `api/routers/inventory/inv_planning_replenishment.py`, `config/inventory/inventory_planning_config.yaml` (projection section) |

---

## Problem

Safety stock targets and replenishment policies define *what* to maintain, but planners need a forward-looking *action plan*: which items to order, when, and how much, covering the next several months. Without a consolidated replenishment plan, planners manually calculate order quantities per DFU each cycle.

---

## Solution

A forward-looking replenishment plan generator that combines production forecast confidence intervals (CI bands), safety stock targets, current inventory positions, and policy parameters to produce a month-by-month order schedule per DFU. The plan compares against a historical baseline to highlight changes.

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

4. **Apply CI bands:** The production forecast's confidence interval (from `fact_production_forecast`) provides a forward sigma estimate. The plan uses the upper CI bound for safety calculations in the first 1-3 months (conservative) and the point forecast for months 4+.

5. **Compare to baseline:** Historical average order pattern (trailing 6 months) serves as the comparison. Deviations flagged with direction and magnitude.

### Output Columns

| Column | Type | Purpose |
|---|---|---|
| `item_id`, `loc` | TEXT | DFU identity |
| `plan_month` | DATE | Planning period |
| `forecast_qty` | NUMERIC | Expected demand (point forecast) |
| `forecast_upper_ci` | NUMERIC | Upper confidence bound |
| `projected_position` | NUMERIC | Inventory before order |
| `order_qty` | NUMERIC | Recommended order |
| `order_date` | DATE | When to place (plan_month - lead_time) |
| `policy_id` | TEXT | Governing policy |
| `vs_baseline_pct` | NUMERIC | Change vs historical average |

---

## Data Model

| Table | Grain | Purpose |
|---|---|---|
| `fact_replenishment_plan` | item_id + loc + plan_month | Forward order schedule |

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/inv-planning/replenishment-plan/summary` | Horizon-level totals (units, value, order count) |
| GET | `/inv-planning/replenishment-plan/detail` | Per-DFU monthly plan |
| GET | `/inv-planning/replenishment-plan/changes` | Items with significant baseline deviations |
| POST | `/inv-planning/replenishment-plan/generate` | Trigger plan generation |

Router: `inv_planning_replenishment.py`

---

## Pipeline

| Step | Script | Output |
|---|---|---|
| Generate plan | `scripts/compute_replenishment_plan.py` | `fact_replenishment_plan` rows |

---

## Configuration

The replenishment plan reads policy parameters from `config/inventory/replenishment_policy_config.yaml` and projection settings from `config/inventory/inventory_planning_config.yaml` (projection section). Planning horizon length and CI band usage rules are configurable.

```yaml
# In inventory_planning_config.yaml, projection section
planning_horizon_months: 6
ci_conservative_months: 3    # Use upper CI for months 1-3
baseline_lookback_months: 6
change_threshold_pct: 20     # Flag deviations > 20%
```

---

## Dependencies

- **Upstream:** `fact_production_forecast` (demand + CI bands), `fact_safety_stock_targets` (ROP), `fact_dfu_policy_assignment` (policy type + params), `agg_inventory_monthly` (current position), open POs
- **Downstream:** Planned order generation, procurement, financial planning (projected spend)

---

## See Also

- [04-replenishment](04-replenishment.md) -- Policy definitions that govern order logic
- [03-safety-stock](03-safety-stock.md) -- SS targets define reorder points
- [../02-forecasting/02-08-production-forecast](../02-forecasting/02-08-production-forecast.md) -- Forecast + CI bands
- [11-rebalancing](11-rebalancing.md) -- Transfers may substitute for new orders
