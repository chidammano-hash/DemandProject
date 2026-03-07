# Next Evolution — Demand Studio: From Analytics to Operational Planning

**Date:** 2026-03-06
**Author perspective:** Supply Chain Planning System Architect

---

## The Honest Baseline

The current system is a **retrospective analytics platform**, not an operational planning system.

What exists today:
- Inventory snapshots (historical stock positions, no open-order integration)
- Backtesting framework (ML model accuracy evaluated against past actuals)
- Champion model selection (best historical model per DFU per past month)
- Rule-based exception alerts from current inventory vs. static thresholds
- EOQ and safety stock calculations from historical demand statistics

What does **not** exist:
- No deployed forecasting engine generating future-period predictions
- No forward inventory projection (current stock + on-order − future demand)
- No live open purchase order data
- No order execution or PO generation
- No supplier lead time actuals (only item-level estimates)
- No ERP/WMS integration of any kind
- No consensus or collaborative planning layer

The gap between where the system is and a true **operational planning system** is significant. The roadmap below defines three phases to close it.

---

## Phase 1 — Deploy the Forecast Engine (Foundation)

*Pre-requisite for everything else. Without real forecasts nothing below is possible.*

### F1.1 — Production Forecast Generation Pipeline

**Problem:** The ML models (LGBM, CatBoost, XGBoost) exist only in backtesting. They have never been run to generate predictions for future periods.

**What to build:**
- A `scripts/generate_future_forecasts.py` pipeline that:
  1. Loads the trained champion model weights (or retrains on full history)
  2. Builds a feature matrix for future months (lag features rolled forward, no future actuals)
  3. Generates `horizon = 1–12` month predictions per DFU
  4. Writes output to `fact_external_forecast_monthly` with `model_id='production_<date>'` and future `startdate` values
- A Makefile target `make forecast-generate` and an APScheduler job (`generate_production_forecast`) running monthly
- A `/forecast/future` API endpoint returning predictions per DFU with horizon selection

**Why it matters:** Without this, every downstream planning calculation (inventory projection, order recommendation, safety stock) is based on historical averages, not forward-looking signal.

---

### F1.2 — Forward Inventory Projection

**Problem:** The system knows current stock-on-hand but cannot answer "what will my inventory be in 6 weeks if I don't order?"

**What to build:**
- A `scripts/project_inventory_forward.py` that runs the following calculation per DFU per day `t`:
  ```
  projected_qty[t] = projected_qty[t-1] + receipts[t] - forecast_demand[t]
  ```
- Identifies the first day where `projected_qty[t] ≤ safety_stock` (the **reorder trigger date**)
- Identifies the first day where `projected_qty[t] ≤ 0` (the **projected stockout date**)
- A new table `fact_inventory_projection` (grain: item_no + loc + projection_date + horizon_days)
- A `/inv-planning/projection` API endpoint returning a time series chart per DFU
- UI panel in Inv. Planning showing projected inventory curve with SS and stockout threshold lines

**Why it matters:** Replaces the current reactive "you're below SS right now" alert with a proactive "you will stock out in 14 days" warning — planners can act before the problem occurs.

---

### F1.3 — Open Purchase Order Integration

**Problem:** The inventory snapshot contains `qty_on_hand_on_order` (a derived field) but there is no data on *when* that inventory is expected to arrive or against which supplier PO it belongs to.

**What to build:**
- A `fact_open_purchase_orders` table (grain: po_number + line_number):
  - Columns: `item_no`, `loc`, `supplier_id`, `ordered_qty`, `confirmed_qty`, `po_date`, `expected_receipt_date`, `status`
- A CSV/API ingest pipeline from the source ERP (or a manual upload flow)
- Wire open PO receipts into the forward inventory projection (`F1.2`) so the curve reflects confirmed inbound supply
- UI column on the Inventory Position table: "Next Receipt: 200 units on 2026-04-12"

**Why it matters:** Without open PO data, the inventory projection assumes no inbound supply. Every reorder recommendation is over-stated. The system cannot differentiate "we need to order" from "we already ordered and it's arriving next week."

---

## Phase 2 — Operational Planning Engine

*Transforms the system from "here is what happened" to "here is what you should do."*

### F2.1 — Order Recommendation Engine

**Problem:** The current `compute_recommendation()` in `generate_replenishment_exceptions.py` produces a single static quantity based on the current gap vs safety stock. It does not account for:
- Future demand trajectory
- Already-confirmed inbound supply (open POs)
- Lead time variability
- Review cycle alignment (don't order if next planned order is in 3 days)
- Budget constraints

**What to build:**
- Replace the current heuristic with a **net requirements calculation**:
  ```
  Net Requirement[t] = max(0, SS + forecast_demand[t:t+LT] - projected_qty[t] - open_order_qty[t])
  ```
- Round up to the nearest MOQ/EOQ
- Group by supplier and review date to produce a **planned order schedule** (not just a single order event)
- A new table `fact_planned_orders` (grain: item_no + loc + planned_order_date):
  - Columns: `recommended_qty`, `latest_order_date`, `expected_receipt_date`, `order_value`, `policy_id`, `confidence_score`
- A `/inv-planning/planned-orders` API endpoint
- A "Planned Orders" panel in Inv. Planning with approve/reject/modify workflow

**Why it matters:** Planners currently get an alert that says "order something." They need to know exactly what to order, when to order it, which supplier to use, and what it will cost.

---

### F2.2 — Multi-Horizon Demand Plan

**Problem:** The current ML forecast generates point estimates per month. A planning system needs:
- Probabilistic forecast intervals (P50, P80, P95) to size safety stock correctly
- Short-horizon (1–4 week) disaggregation for execution scheduling
- Long-horizon (12–18 month) for supplier capacity booking and budget planning

**What to build:**
- Extend `generate_future_forecasts.py` to produce **forecast quantiles** (P10, P50, P90) using:
  - Quantile regression loss on LightGBM, or
  - Monte Carlo sampling over the demand signal distribution
- A `fact_demand_plan` table (grain: item_no + loc + plan_month + quantile):
  - Columns: `forecast_qty`, `lower_bound`, `upper_bound`, `model_id`, `plan_version`
- Plan versioning: each `generate_production_forecast` run creates a new version; planners compare versions
- A "Demand Plan" sub-panel in Inv. Planning showing probabilistic fan chart (P10/P50/P90 bands)

**Why it matters:** Safety stock sized on P50 creates stockouts ~50% of the time. You need P80 or P95 coverage targets tied to ABC classification, and you need the probabilistic output to compute that correctly.

---

### F2.3 — Consensus Forecasting & Planner Overrides

**Problem:** The system has no mechanism for a demand planner, sales manager, or commercial team to inject their knowledge (promotions, new product launches, market intelligence) into the statistical forecast.

**What to build:**
- A `fact_forecast_overrides` table (grain: item_no + loc + override_month):
  - Columns: `override_qty`, `override_reason`, `override_type` (promo/launch/market_event/manual), `created_by`, `valid_from`, `valid_to`, `approved_by`
- An "Override" button on each DFU row in the Demand Plan panel
- The override propagates into `fact_demand_plan` as a new plan version: `model_id = 'consensus'`
- The inventory projection and order recommendation engine uses the consensus plan, not the raw statistical forecast
- An override audit trail and expiry mechanism (overrides auto-expire unless renewed)

**Why it matters:** Statistical forecasts are systematically wrong for promotional periods, new launches, and phase-outs. Every mature planning organization layers human judgment on top of the statistical baseline.

---

### F2.4 — Procurement Workflow & Order Release

**Problem:** The planned order recommendations exist only in the database. There is no mechanism to convert them into actual purchase orders that reach a supplier.

**What to build:**
- An order approval workflow in the Planned Orders panel:
  - States: `proposed → planner_approved → buyer_released → po_sent → confirmed → received → closed`
  - Role-based actions: planners approve quantities, buyers release to supplier
- A `POST /inv-planning/planned-orders/{id}/release` endpoint that:
  - Changes status to `buyer_released`
  - Optionally calls an ERP webhook (SAP, Oracle, NetSuite) or generates a CSV/EDI file
- An ERP integration configuration page (endpoint URL, auth, field mapping)
- Email/Slack notification to buyers when orders require release action

**Why it matters:** Without execution integration the system remains advisory. The ROI of planning optimization is only realized when recommendations translate into procurement actions.

---

## Phase 3 — Closed-Loop Planning Intelligence

*Completes the feedback loop: plan → execute → measure → learn.*

### F3.1 — Forecast Bias Correction Engine

**Problem:** ML forecasts have systematic biases by cluster, seasonality profile, and product lifecycle stage. The system can measure these biases (it does in the Accuracy tab) but cannot automatically correct for them in the next planning cycle.

**What to build:**
- A `scripts/compute_bias_correction.py` that:
  1. Computes rolling 3-month WAPE and bias by DFU segment (ABC × lifecycle × cluster)
  2. Derives a multiplicative correction factor: `correction = 1 / (1 + rolling_bias)`
  3. Applies correction to next-period statistical forecast before plan generation
- A `fact_bias_corrections` table logging correction factors per DFU per plan cycle
- Surfaced as a "Bias Corrected" indicator in the Demand Plan panel

**Why it matters:** A forecasting system that measures its own errors but does not correct for them is not learning. Automatic bias correction reduces systematic over- or under-forecasting without model retraining.

---

### F3.2 — Service Level Actuals vs. Targets Tracking

**Problem:** The system computes fill rate from inventory snapshots (IPfeature8) but there is no tracking of:
- Whether planned service level targets were actually achieved
- Which DFUs are chronically missing their service level target
- Whether safety stock changes produced the intended service level improvement

**What to build:**
- A `fact_service_level_performance` table (grain: item_no + loc + month):
  - Columns: `target_service_level`, `achieved_fill_rate`, `stockout_events`, `projected_ss_days`, `actual_ss_days`, `gap`
- A "Service Level Performance" panel in the Control Tower: RAG-coded table of DFUs missing targets with trend
- An alert: "14 A-class items have missed their 98% service level target for 3+ consecutive months"
- Feedback loop: when a DFU consistently misses its target, flag for safety stock review in the AI Planner

**Why it matters:** Safety stock is sized to achieve a target. Without tracking whether the target was actually achieved, there is no way to know if the safety stock parameters are correct.

---

### F3.3 — Supplier Performance & Lead Time Learning

**Problem:** The current `mv_supplier_performance` view aggregates receipt data but does not:
- Track promised vs. actual delivery dates at the PO line level
- Build a statistical model of each supplier's lead time distribution
- Feed lead time actuals back into safety stock calculation

**What to build:**
- Add `promised_receipt_date` and `actual_receipt_date` to `fact_open_purchase_orders` (`F1.3`)
- A `scripts/update_lead_time_actuals.py` that:
  1. Computes mean and standard deviation of lead time per supplier per item category
  2. Updates `dim_lead_time_profile` with rolling actuals
  3. Triggers a safety stock recalculation for DFUs where lead time variance has changed significantly
- A "Lead Time Reliability" column in the Supplier Performance panel: promised vs. actual delivery rate
- Alert: "Supplier X lead time has increased by 8 days over 90 days — 23 DFUs need SS review"

**Why it matters:** Safety stock is a function of lead time variability. If supplier reliability degrades, safety stock must increase or stockout risk rises. Currently this relationship is static.

---

### F3.4 — Demand Sensing Integration (Short-Horizon Override)

**Problem:** The current demand sensing module (`fact_demand_signals`) computes a short-horizon signal from inventory velocity but does not use it to override the monthly statistical forecast in the planning engine.

**What to build:**
- Define a **demand sensing horizon** (default: 4 weeks)
- Within the sensing horizon, replace statistical forecast with the velocity-based signal from `fact_demand_signals`
- Beyond the sensing horizon, blend: `blended_forecast = α * sensing_signal + (1-α) * statistical_forecast`
  - `α` decays from 1.0 at T+1 to 0.0 at T+4 weeks
- Wire the blended forecast into inventory projection (`F1.2`) and order recommendation (`F2.1`)
- Show "Sensing Override Active" badge on affected DFUs in the Demand Plan panel

**Why it matters:** Statistical models trained on monthly data cannot respond to a spike in daily sales velocity that happened two weeks ago. Short-horizon sensing improves immediate order accuracy significantly.

---

### F3.5 — Network / Multi-Echelon Planning

**Problem:** The system plans each item-location independently. In practice, inventory at a distribution center should be planned jointly with inventory at retail locations it replenishes — a stockout at the DC propagates to all downstream stores.

**What to build:**
- A `dim_supply_network` table defining parent-child relationships:
  - Columns: `from_loc`, `to_loc`, `replenishment_lead_time`, `min_transfer_qty`, `transfer_frequency`
- An `scripts/compute_echelon_targets.py` that:
  1. Starts at leaf (store) nodes — computes their demand, safety stock, and reorder points
  2. Aggregates upstream demand to parent (DC) nodes accounting for pooling benefits
  3. Sets DC safety stock based on downstream demand variability, not historical DC shipments
- Surfaces echelon coverage in the Inv. Planning Health panel: "DC covers 3.4 months of downstream demand"
- Flags when DC stock cannot meet downstream replenishment requirements

**Why it matters:** Planning each location in isolation ignores demand pooling at the DC, leads to unnecessary safety stock duplication, and misses the systemic risk of a DC-level stockout cascading to many stores.

---

## Phase 4 — S&OP / IBP Integration (Long-Horizon)

*Connects operational planning to financial and strategic planning.*

### F4.1 — Financial Inventory Plan (Budget vs. Actuals)

Build a rolling 13-month financial inventory plan showing:
- Projected inventory value (units × cost) at each planning horizon
- Budget vs. forecast inventory investment by category
- Working capital release/lock-up from planned order schedule

### F4.2 — Sales & Operations Planning (S&OP) Module

A monthly consensus process workflow:
- Demand review: statistical forecast + commercial overrides → agreed demand plan
- Supply review: capacity constraints, supplier allocations, planned shutdowns
- Pre-S&OP: gap analysis (demand plan vs. supply capability)
- Executive S&OP: approved plan published to planning engine

### F4.3 — Promotion & Event Planning

An event calendar where planners register:
- Promotions (item + location + date range + uplift %)
- New product launches (item + introduction date + ramp curve)
- Phase-outs (item + last-order date + end-of-life inventory target)
- Market events (sporting seasons, holidays, weather)

Registered events automatically apply multiplicative adjustments to the statistical forecast during consensus generation.

### F4.4 — Scenario Planning ("What-If" for Supply Chain)

Extend the existing clustering What-If framework to cover:
- Demand shock scenarios: "what if demand drops 20% for 3 months?"
- Lead time shock scenarios: "what if Supplier X doubles its lead time?"
- Inventory investment scenarios: "what service level can we achieve with $X less working capital?"
- Network disruption: "what if DC East goes offline for 30 days?"

---

## Implementation Priority Summary

| Phase | Feature | Priority | Dependency |
|---|---|---|---|
| 1 | F1.1 Production Forecast Generation | **Critical** | None — must be first |
| 1 | F1.2 Forward Inventory Projection | **Critical** | F1.1, F1.3 |
| 1 | F1.3 Open PO Integration | **Critical** | ERP data access |
| 2 | F2.1 Order Recommendation Engine | **High** | F1.1, F1.2, F1.3 |
| 2 | F2.2 Multi-Horizon Demand Plan | **High** | F1.1 |
| 2 | F2.3 Consensus Forecasting & Overrides | **High** | F2.2 |
| 2 | F2.4 Procurement Workflow & Order Release | **High** | F2.1, ERP integration |
| 3 | F3.1 Bias Correction Engine | **Medium** | F1.1 |
| 3 | F3.2 Service Level Actuals Tracking | **Medium** | F1.3, F2.1 |
| 3 | F3.3 Supplier Lead Time Learning | **Medium** | F1.3 |
| 3 | F3.4 Demand Sensing Integration | **Medium** | F1.1, F2.2 |
| 3 | F3.5 Network / Multi-Echelon Planning | **Medium** | F2.1, dim_supply_network |
| 4 | F4.1 Financial Inventory Plan | **Low** | F2.1, F2.2 |
| 4 | F4.2 S&OP Module | **Low** | F2.3 |
| 4 | F4.3 Promotion & Event Planning | **Low** | F2.3 |
| 4 | F4.4 What-If Scenario Planning | **Low** | All Phase 2 |

---

## The Minimum Viable Planning System

If only three things get built, they are:

1. **F1.1 — Production Forecast Generation:** Deploy the trained ML models to generate actual future predictions. Without this, every "planning" feature downstream is based on historical averages, not forecasts.

2. **F1.2 — Forward Inventory Projection:** Project stock levels forward using forecast demand and confirmed receipts. This transforms "you are below safety stock today" into "you will stock out on April 14."

3. **F1.3 — Open PO Integration:** Ingest actual purchase order receipt schedules. Without this, the projection model assumes zero inbound supply and every order recommendation is wrong.

These three features convert the current system from a **reporting tool** into a **planning tool**. Everything else in Phase 2–4 builds on this foundation.

---

*Current system characterization: retrospective analytics engine with real-time alerting.*
*Target system characterization: forward-looking operational planning platform with closed-loop execution.*
