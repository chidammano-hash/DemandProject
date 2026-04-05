# Scenario Planning

> A what-if simulation engine that models supply chain disruptions -- demand shocks, lead time delays, supplier failures, and budget changes -- to quantify financial and service-level impact before they happen.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inventory Planning (ScenarioPlanningPanel) |
| **Key Files** | `inv-planning/ScenarioPlanningPanel.tsx`, `api/routers/supply_scenarios.py`, `scripts/run_supply_chain_scenario.py`, `config/supply_scenario_config.yaml`, `sql/059_create_supply_scenarios.sql` |

---

## Problem

Supply chains are brittle. A single-source supplier outage, a 30% demand spike, or a two-week lead time extension can cascade into stockouts, excess inventory, and missed revenue targets. Planners need to model these disruptions before they happen so they can pre-position inventory, qualify backup suppliers, or adjust budgets. Without structured scenario analysis, contingency planning happens in ad-hoc spreadsheets that are disconnected from live inventory and forecast data.

---

## Solution

A scenario engine accepts disruption parameters (type, magnitude, scope, duration), applies them to the current inventory position and demand forecast, and simulates forward to compute projected stockouts, excess inventory, service level changes, and financial impact. Results are stored for comparison across scenarios. The S&OP team uses scenario results during Stage 4 (Pre-S&OP Meeting) to evaluate trade-offs.

---

## How It Works

### Scenario Types

| Type | What It Models | Key Parameters |
|---|---|---|
| Demand Shock | Sudden demand increase or decrease | `demand_multiplier`, affected items/locations, duration |
| Lead Time Shock | Supplier or transit delay | `lead_time_add_days`, affected items/locations |
| Supplier Disruption | Partial or full supplier failure | `supply_reduction_pct`, affected supplier, duration |
| DC Disruption | Distribution center (warehouse) outage | `dc_location`, outage duration, reroute options |
| Investment | Budget increase or decrease | `budget_change_pct`, affected locations |
| Policy Change | Replenishment policy adjustment | `new_policy_type`, `new_service_level`, affected scope |

### Simulation Flow

1. Clone current inventory positions and forecast into a simulation workspace.
2. Apply disruption parameters to the cloned data (e.g., multiply demand by 1.3 for demand shock).
3. Run forward projection month-by-month using the modified inputs.
4. At each month, compute: projected on-hand, stockout events, excess over target, fill rate, carrying cost.
5. Compare results to the baseline (no disruption) to quantify delta impact.
6. Write results to `fact_scenario_results` with a `scenario_id` for retrieval.

### Impact Metrics

| Metric | How Computed |
|---|---|
| Stockout Events | Count of item-location-months where projected on-hand falls below zero |
| Revenue at Risk | Stockout units x average selling price |
| Excess Inventory Value | Units above safety stock target x unit cost |
| Service Level Delta | Scenario fill rate minus baseline fill rate |
| Incremental Carrying Cost | Additional holding cost from changed inventory levels |

---

## Data Model

| Table | Purpose | Key Columns |
|---|---|---|
| `fact_supply_scenarios` | Scenario definitions | `scenario_id`, `scenario_type`, `name`, `parameters` (JSONB), `status`, `created_at` |
| `fact_scenario_results` | Simulation output per item-location-month | `scenario_id`, `item_id`, `loc`, `month`, `projected_oh`, `stockout_flag`, `excess_qty`, `carrying_cost` |
| `fact_scenario_comparison` | Baseline vs. scenario deltas | `scenario_id`, `metric`, `baseline_value`, `scenario_value`, `delta`, `delta_pct` |
| `fact_scenario_audit` | Execution history | `scenario_id`, `run_at`, `duration_seconds`, `row_count` |

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/supply-scenarios` | List scenarios with filtering by type and status |
| POST | `/supply-scenarios` | Create and queue a new scenario |
| GET | `/supply-scenarios/{id}` | Scenario detail with parameters and status |
| POST | `/supply-scenarios/{id}/run` | Execute simulation (returns 202, runs in background) |
| GET | `/supply-scenarios/{id}/results` | Simulation results with item-level detail |
| GET | `/supply-scenarios/{id}/comparison` | Baseline vs. scenario comparison metrics |
| DELETE | `/supply-scenarios/{id}` | Remove scenario and its results |

Simulation runs asynchronously via background thread. Poll status via the detail endpoint.

---

## Pipeline

| Step | Command | What It Does |
|---|---|---|
| Schema | `make scenario-schema` | Creates scenario tables |
| Run | `scripts/run_supply_chain_scenario.py --scenario-id X` | Execute a specific scenario |
| Dry Run | Add `--dry-run` flag | Preview impact without writing results |

---

## Configuration

File: `config/supply_scenario_config.yaml`

| Key | Purpose | Default |
|---|---|---|
| `projection_horizon_months` | How far forward to simulate | `6` |
| `max_concurrent_scenarios` | Parallel simulation limit | `2` |
| `demand_shock.max_multiplier` | Cap on demand shock magnitude | `3.0` |
| `lead_time_shock.max_add_days` | Cap on lead time extension | `90` |
| `supplier_disruption.max_reduction_pct` | Cap on supply reduction | `100` |
| `baseline_source` | Where baseline data comes from | `production_forecast` |

---

## Dependencies

| Dependency | Reason |
|---|---|
| Inventory snapshot (03-01) | Starting inventory positions for simulation |
| Production forecast (02-08) | Baseline demand for forward projection |
| Safety stock targets (03-03) | Excess computed relative to targets |
| Financial plan (05-02) | Unit costs for financial impact computation |
| S&OP cycle (05-01) | Scenario results reviewed during Stage 4 |

---

## See Also

- `05-operations/01-sop-cycle.md` -- scenario results inform S&OP trade-off decisions
- `05-operations/02-financial-planning.md` -- financial impact uses same cost inputs
- `03-inventory-planning/03-safety-stock.md` -- policy change scenarios adjust safety stock levels
- `03-inventory-planning/12-inventory-rebalancing.md` -- DC disruption scenarios may trigger rebalancing
