# Scenario Planning

> A what-if simulation engine that models supply chain disruptions -- demand shocks, lead time delays, capacity constraints, and logistics disruptions -- to quantify stockout risk and financial impact before they happen.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inventory Planning (ScenarioPlanningPanel) |
| **Key Files** | `inv-planning/ScenarioPlanningPanel.tsx`, `api/routers/operations/supply_scenarios.py`, `scripts/inventory/run_supply_chain_scenario.py`, `config/operations/supply_scenario_config.yaml`, `sql/058_create_supply_scenarios.sql` |

---

## Problem

Supply chains are brittle. A single-source supplier outage, a 30% demand spike, or a two-week lead time extension can cascade into stockouts, excess inventory, and missed revenue targets. Planners need to model these disruptions before they happen so they can pre-position inventory, qualify backup suppliers, or adjust budgets. Without structured scenario analysis, contingency planning happens in ad-hoc spreadsheets that are disconnected from live inventory and forecast data.

---

## Solution

A scenario engine accepts disruption parameters (type, magnitude, scope, duration), applies them to the current inventory position and demand forecast, and simulates forward to compute projected stockouts, excess inventory, service level changes, and financial impact. Results are stored for comparison across scenarios. The S&OP team uses scenario results during Stage 4 (Pre-S&OP Meeting) to evaluate trade-offs.

---

## How It Works

### Scenario Types

`scenario_type` is a free-text `VARCHAR(50)` on `fact_supply_scenarios` -- there is no DB-enforced enum, and two
different entry points use two different (overlapping) vocabularies for it.

`POST /scenarios/supply` (the API path) documents this convention in its `_ScenarioCreate` model:

| Type | `scenario_type` value | What It Models |
|---|---|---|
| Demand Shock | `demand_shock` | Sudden demand increase or decrease |
| Lead Time Shock | `lead_time_shock` | Supplier or transit delay |
| Capacity Constraint | `capacity_constraint` | Supplier can only ship a fraction of ordered quantity |
| Logistics Disruption | `logistics_disruption` | Freight/transport unavailable |

"Investment" and "Policy Change" scenario types do not exist in the codebase -- there is no budget-change or
replenishment-policy scenario type.

The standalone CLI script (`scripts/inventory/run_supply_chain_scenario.py --disruption-type`) writes into the
same `scenario_type` column using a different, overlapping vocabulary: `supplier_delay`, `capacity_constraint`,
`demand_shock`, `transport_disruption`, `quality_hold` -- also the basis for `disruption_defaults` in
`config/operations/supply_scenario_config.yaml`. The two paths are not reconciled; check which one produced a
given scenario before filtering on `scenario_type`.

### Simulation Flow

Simulation is executed by the CLI script (`--action run`), not the API -- `POST /scenarios/supply/{id}/run` only
flips `status` to `running` and stamps `run_by`/`last_run_at`; the actual numbers come from a separate
`scripts/inventory/run_supply_chain_scenario.py --action run --scenario-id N` invocation.

1. Fetch the scenario's `disruption_type`/`impact_pct`/`duration_weeks` (from `shock_parameters`) and the
   affected item-locations (scoped to `affected_items`/`affected_locations` if set, else the whole portfolio)
   from `fact_inventory_snapshot`.
2. For `supplier_delay`/`transport_disruption`: extend lead time by `duration_weeks x 7 x (impact_pct / 100)`
   days. For `capacity_constraint`/`quality_hold`: reduce available supply by `impact_pct`%. Other disruption
   types leave lead time and supply unchanged.
3. Estimate stockout days as `MAX(0, adjusted_lead_time_days - (on_hand + available_supply) / daily_demand)`.
4. Write one row per item-location to `fact_scenario_results`; mark the scenario `completed` and stamp
   `last_run_at`.
5. A financial-impact estimate (`stockout_units x stockout_cost_per_unit`, from `supply_scenario_config.yaml`)
   is computed and returned in the CLI's result dict but is not persisted to any table.

### Impact Metrics

| Metric | Column | How Computed |
|---|---|---|
| Baseline Supply | `baseline_qty` | 30-day average-demand estimate (not a live on-hand snapshot) |
| Scenario Supply | `scenario_qty` | Available supply after the disruption is applied |
| Impact Qty | `impact_qty` | Supply shortfall: `baseline_qty - scenario_qty` |
| Impact % | `impact_pct` | `impact_qty / baseline_qty x 100` |
| Stockout Risk | `stockout_risk_days` | Estimated stockout days from the depletion-time formula above |
| Excess Risk | `excess_risk_qty` | Not populated by the CLI script -- always `0` |
| Mitigation Option | `mitigation_option` | `"expedite"` if a stockout is projected, else `"none"` |

---

## Data Model

| Table | Purpose | Key Columns |
|---|---|---|
| `fact_supply_scenarios` | Scenario definitions | `scenario_id`, `scenario_name`, `scenario_type`, `description`, `shock_parameters` (JSONB), `affected_items`/`affected_locations`/`affected_suppliers` (JSONB), `horizon_months`, `status` (`draft`\|`running`\|`completed`\|`failed`), `created_by`, `run_by`, `created_at`, `last_run_at`, `run_duration_ms` |
| `fact_scenario_results` | Simulation output per item-location | `id`, `scenario_id`, `item_id`, `loc`, `plan_month`, `baseline_qty`, `scenario_qty`, `impact_qty`, `impact_pct`, `stockout_risk_days`, `excess_risk_qty`, `mitigation_option`, `computed_at` |

There is no `fact_scenario_comparison` or `fact_scenario_audit` table. Baseline-vs-scenario comparison and
execution history are not persisted separately -- the API reads `fact_supply_scenarios` and
`fact_scenario_results` directly.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/scenarios/supply` | List scenarios, filterable by `scenario_type` and `status` |
| POST | `/scenarios/supply` | Create a new scenario, starts at `status = draft` (auth) |
| GET | `/scenarios/supply/{scenario_id}` | Scenario definition + most recent run metadata |
| POST | `/scenarios/supply/{scenario_id}/run` | Mark the scenario `running`, record `run_by`/`last_run_at` (auth) |
| GET | `/scenarios/supply/{scenario_id}/results` | Simulation results, paginated |

The path prefix is `/scenarios/supply`, not `/supply-scenarios`. The router's module docstring also lists a
`POST /scenarios/supply/{id}/compare` endpoint, but it is not implemented; there is no `DELETE` endpoint either.
`POST .../run` only updates scenario status -- it does not itself run the simulation (see Simulation Flow above).

---

## Pipeline

| Step | Command | What It Does |
|---|---|---|
| Schema | `make scenarios-schema` | Creates scenario tables |
| List | `make scenarios-list` | List scenarios via the CLI script |
| Run | `make scenarios-run SCENARIO_ID=n` | Execute a specific scenario and write `fact_scenario_results` |
| Run (dry) | `make scenarios-run-dry SCENARIO_ID=n` | Preview impact without writing results |
| Full | `make scenarios-all` | Currently just runs `scenarios-schema` -- no default scenario is seeded |
| Create | `scripts/inventory/run_supply_chain_scenario.py --action create --scenario-name ... --disruption-type ... --impact-pct N --duration-weeks N` | Create a scenario via the CLI (no Make target) |

---

## Configuration

File: `config/operations/supply_scenario_config.yaml`

| Key | Purpose | Default |
|---|---|---|
| `supply_scenario.simulation_horizon_weeks` | Simulation lookahead | `13` |
| `supply_scenario.service_level_target` | Target fill rate for impact assessment | `0.95` |
| `supply_scenario.stockout_cost_per_unit` | $ per unit of lost sales (margin proxy) | `10.0` |
| `supply_scenario.excess_holding_cost_pct` | Annual holding cost as a fraction of unit cost | `0.25` |
| `supply_scenario.disruption_defaults.<type>.typical_impact_pct` | Default severity for each of the 5 CLI disruption types | varies (`25`-`100`) |
| `supply_scenario.disruption_defaults.<type>.typical_duration_weeks` | Default duration for each disruption type | varies (`2`-`8`) |
| `supply_scenario.alert_threshold_usd` | Financial-impact level that surfaces a control-tower alert | `100000.0` |
| `scheduler.cron` | Scheduled run cadence | `null` (on-demand only) |

---

## Dependencies

| Dependency | Reason |
|---|---|
| Inventory snapshot (03-01) | `fact_inventory_snapshot` supplies on-hand qty and average daily demand |
| Item cost (`dim_item_cost`) | Unit cost for financial-impact estimation |
| Lead time profile (`dim_lead_time_profile`) | Base lead time before disruption adjustment |
| S&OP cycle (05-01) | Scenario results reviewed during Stage 4 |

---

## See Also

- `05-operations/01-sop-cycle.md` -- scenario results inform S&OP trade-off decisions
- `05-operations/02-financial-planning.md` -- `dim_item_cost` and holding-cost assumptions are shared
- `03-inventory-planning/03-safety-stock.md` -- lead-time and stockout-risk concepts shared with safety stock
