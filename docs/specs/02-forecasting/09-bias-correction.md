# Bias Correction & Forward Inventory Projection

> Detects systematic forecast bias and projects day-by-day inventory positions so planners can see stockouts weeks before they happen.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inv. Planning (Projection panel), Accuracy (Bias Corrections panel) |
| **Key Files** | `scripts/compute_bias_corrections.py`, `scripts/compute_inventory_projection.py`, `api/routers/forecasting/bias_corrections.py`, `api/routers/inventory/inv_planning_projection.py`, `config/bias_correction_config.yaml`, `config/inventory/inventory_planning_config.yaml` (projection section) |

---

## Problem

Point-in-time inventory snapshots tell planners what they have today but not what they will have tomorrow. A planner sees "120 units on hand, safety stock = 60" and assumes a comfortable buffer. But if the ML forecast predicts 490 units of demand next month (not the ERP's flat 400), the item stocks out in 8 days -- and the signal was there all along. Meanwhile, models that systematically over- or under-forecast create compounding inventory errors that are invisible until stockouts or excess pile up.

## Solution

Two complementary capabilities: (1) Bias detection flags DFUs where the forecast consistently runs high or low, enabling planners to adjust or investigate. (2) Forward inventory projection simulates day-by-day inventory positions over 90 days using the production forecast, open purchase orders, and planned orders -- revealing stockout dates, reorder trigger dates, and excess risk weeks in advance.

---

## Part 1: Bias Detection & Correction

### How It Works

1. For each DFU, compute rolling bias over a configurable window (default: 6 months)
2. Bias = `(SUM(Forecast) / SUM(Actual)) - 1` -- positive means over-forecasting
3. Flag DFUs where absolute bias exceeds a threshold (configurable by ABC class)
4. Provide correction recommendations: multiplicative adjustment factor = `1 / (1 + bias)`
5. Planners review flagged items in the Bias Corrections panel and can accept or dismiss

### Configuration: `config/bias_correction_config.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `rolling_window_months` | 6 | Months of history for bias computation |
| `bias_threshold_a` | 0.10 | Flag A-class items with >10% absolute bias |
| `bias_threshold_b` | 0.15 | Flag B-class items with >15% absolute bias |
| `bias_threshold_c` | 0.20 | Flag C-class items with >20% absolute bias |
| `min_history_months` | 3 | Minimum months of data before flagging |

### API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/forecast/bias-corrections` | List flagged DFUs with bias metrics |
| GET | `/forecast/bias-corrections/summary` | Aggregate bias statistics |

---

## Part 2: Forward Inventory Projection

### How It Works

1. Start with current on-hand quantity from the latest inventory snapshot
2. Disaggregate the monthly production forecast into daily demand rates
3. Simulate three scenarios in parallel:
   - **No order:** Current on-hand only, no inbound assumed
   - **With open POs:** Current on-hand + confirmed purchase orders with delivery dates
   - **With planned orders:** Open POs + system-recommended planned orders
4. For each day, compute: `projected_qty = max(0, previous_qty + receipts - demand)`
5. Flag key dates: reorder trigger, stockout risk, excess risk
6. Write results to `fact_inventory_projection`

### Three Scenarios

| Scenario | Receipt Input | Use Case |
|----------|--------------|----------|
| `no_order` | Zero receipts | Worst case -- what happens if we do nothing? |
| `with_open_po` | Confirmed POs from `fact_open_purchase_orders` | Realistic -- what does our pipeline cover? |
| `with_planned_orders` | Open POs + approved planned orders | Best case -- what if we execute all plans? |

### Fallback Behavior

If no production forecast exists for a DFU, the projection falls back to a 3-month average of actual sales as a flat daily rate. The API response includes `forecast_source: "fallback_avg"` to flag this.

### Data Model: `fact_inventory_projection`

| Column | Type | Description |
|--------|------|-------------|
| `item_id`, `loc` | VARCHAR | DFU identifier |
| `projection_date` | DATE | Specific future date |
| `scenario` | VARCHAR(30) | Which scenario |
| `projected_qty` | NUMERIC(12,2) | Projected on-hand |
| `projected_dos` | NUMERIC(8,2) | Projected days of supply |
| `reorder_triggered` | BOOLEAN | Below reorder point? |
| `stockout_risk` | BOOLEAN | Projected qty <= 0? |
| `daily_demand_rate` | NUMERIC(10,4) | Demand/day used |
| `forecast_source` | VARCHAR(30) | "production_forecast" or "fallback_avg" |

**Grain:** `(projection_run_id, item_id, loc, scenario, projection_date)`

### API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/inv-planning/projection` | Day-by-day projection for a DFU (all 3 scenarios) |
| GET | `/inv-planning/projection/summary` | Portfolio-level risk summary |
| GET | `/inv-planning/projection/at-risk` | DFUs with near-term stockout risk |

### Configuration: `config/inventory/inventory_planning_config.yaml` (projection section)

| Key | Default | Description |
|-----|---------|-------------|
| `horizon_days` | 90 | How many days forward to project |
| `daily_demand_method` | calendar_days | How to disaggregate monthly to daily |
| `excess_coverage_months` | 6 | Threshold for excess risk flagging |

## Pipeline

| Target | Description |
|--------|-------------|
| `make projection-compute` | Run projection for all active DFUs |
| `make projection-compute-sku ITEM=100320 LOC=1401-BULK` | Single DFU projection |
| `make projection-dry` | Preview without writing |

**Scheduler:** Runs daily at 07:00 UTC (after production forecast at 06:00).

## Dependencies

- [Production Forecast](./08-production-forecast.md) -- provides the demand signal for projection
- Safety Stock (in `03-inventory-planning/`) -- provides reorder points
- Open POs (in `03-inventory-planning/`) -- provides confirmed receipt dates

## See Also

- [Forecast CI Bands](./10-forecast-ci-bands.md) -- uncertainty ranges complement projection scenarios
- [Production Forecast](./08-production-forecast.md) -- upstream data source
