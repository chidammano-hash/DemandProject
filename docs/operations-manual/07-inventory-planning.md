# Section 7 — Inventory Planning

End-to-end operations for the inventory planning module: prerequisites, pipeline orchestration, per-script reference, configs, API, UI, materialized views, and routine maintenance.

---

## 7.1 Data Prerequisites

Before any inventory pipeline runs, the following must already be populated:

| Source | Loaded by | Used for |
|---|---|---|
| `fact_inventory_snapshot` (~198M rows, monthly partitioned by `snapshot_date`) | `make pipeline-inventory` / `make load-all` | EOM on-hand, DOS, stock projections, fill rate |
| `fact_production_forecast` (PROMOTED — see Section 6) | `POST /backtest-management/{model_id}/promote` | Demand input for SS, projection, replenishment |
| `fact_customer_demand_monthly` (monthly partitioned by `startdate`) | `make pipeline-customer-demand` | Variability, lead-time actuals, customer-level fill |
| `dim_sourcing` + `fact_purchase_order` | `make load-all` (sourcing + purchase_order domains) | Lead time, open POs, planned orders |
| `dim_sku` + `current_sku_cluster_assignment` | `make features-compute` + `make cluster-all` (Section 3) | Per-cluster policy assignment |

**If production forecast is missing or stale, every downstream metric (SS, projection, replenishment, exceptions) will be wrong.** Verify with:

```bash
psql "$DATABASE_URL" -c "SELECT COUNT(*), MAX(forecast_month) FROM fact_production_forecast;"
```

---

## 7.2 Pipeline Orchestration

Two top-level Make targets cover the full module:

```bash
make setup-inv-planning      # Inventory side: SS, EOQ, policies, exceptions, fill rate, KPIs
make setup-demand-planning   # Demand side: forecasts, projections, orders, replenishment
```

Expanded from `Makefile`:

```
setup-inv-planning   = eoq-all policy-all ss-all exceptions-generate fill-rate-all
                       health-all supplier-perf-all investment-all intramonth-all
                       control-tower-all rebalancing-all

setup-demand-planning = forecast-prod-all projection-all po-all quantile-all
                        consensus-all planned-orders-all replplan-all bias-all
                        blended-all service-level-all lead-time-all echelon-all
```

Recommended order on a fresh dataset:

1. `make setup-data` (Section 2) → all domains loaded
2. `make features-compute` + `make cluster-all` (Section 3)
3. `make backtest-all` + `make champion-all` (Sections 4–5) → promote a model (Section 6)
4. `make setup-demand-planning` → forecast-driven artifacts
5. `make setup-inv-planning` → policies, SS, exceptions, KPIs

---

## 7.3 Per-Script Reference

Scripts live at `scripts/` root (the `scripts/inventory/` subdirectory does not yet exist on this branch — new inventory scripts should go there per CLAUDE.md placement rules).

### Demand-side scripts (run via `setup-demand-planning`)

| Script | Make target | Output table | Purpose |
|---|---|---|---|
| `compute_inventory_projection.py` | `projection-all` | projection table | Forward inventory position by month |
| `compute_lead_time_variability.py` | `lead-time-all` | lead-time stats | LT mean + variance per supplier × item |
| `update_lead_time_actuals.py` | (within `lead-time-all`) | lead-time actuals | Refresh actuals from PO history |
| `compute_bias_corrections.py` | `bias-all` | bias-correction table | Adjusts forecast for known bias |
| `compute_service_level_actuals.py` | `service-level-all` | service-level table | Actual SL achieved vs target |
| `generate_consensus_plan.py` | `consensus-all` | consensus plan | Combines forecast variants |
| `generate_planned_orders.py` | `planned-orders-all` | planned orders | Time-phased order plan |
| `release_planned_orders.py` | (manual) | open POs | Promotes planned → released |
| `compute_replenishment_plan.py` | `replplan-all` | replenishment plan | Period-by-period replenishment qty |
| `compute_echelon_targets.py` | `echelon-all` | echelon targets | Multi-tier inventory targets |

### Inventory-side scripts (run via `setup-inv-planning`)

| Script | Make target | Output | Purpose |
|---|---|---|---|
| `compute_safety_stock.py` | `ss-all` | SS table | Z × σLT formula (see 7.4) |
| `compute_eoq.py` | `eoq-all` | EOQ table | Wilson EOQ per SKU |
| `assign_replenishment_policies.py` | `policy-all` | policy table | Min/Max, ROP, R/Q assignment per ABC×XYZ class |
| `generate_replenishment_exceptions.py` | `exceptions-generate` | exception queue | Stockout risk, overstock, expiring, lead-time breach |
| `compute_investment_plan.py` | `investment-all` | investment plan | Working-capital allocation |
| `compute_rebalancing.py` | `rebalancing-all` | rebalancing plan | Inter-DC stock moves |
| `refresh_intramonth_stockout.py` | `intramonth-all` | `mv_intramonth_stockout` | Mid-month stockout signal |
| `run_sop_cycle.py` | (Section 8) | SOP outputs | S&OP cycle (cross-module) |

### Algorithm comparison + backtest

```bash
# Inventory backtest — no Make target; runs via the job registry
# (job type "inventory_backtest" → scripts/inventory/run_inventory_backtest.py).
# Trigger from the Jobs UI/API, or invoke the script directly:
uv run python scripts/inventory/run_inventory_backtest.py   # backtest SS/policy choices

make algo-comparison    # scripts/inventory/compare_inventory_algorithms.py — A/B between policy variants
```

Surfaces in the **Inventory Algorithm Comparison** API router for the UI A/B view.

### Granular per-feature targets (`IPfeatureN`)

Each inventory-planning computation is tracked as an `IPfeatureN` feature and exposes granular `-schema` / `-compute` / `-refresh` Make targets in addition to the `-all` rollups invoked by `setup-inv-planning`. Run individually when re-running a single stage. Dependencies are noted inline.

```bash
# Safety stock (IPfeature3 — requires sales + inventory loaded)
make ss-compute              # Compute Z-score safety stock targets per DFU

# EOQ & cycle stock (IPfeature4 — requires inventory loaded)
make eoq-compute             # Wilson EOQ formula → fact_eoq_targets

# Replenishment policies (IPfeature5)
make policy-assign           # Upsert 4 default policies + auto-assign DFUs by ABC segment

# Health score (IPfeature6 — requires safety stock computed)
make health-refresh          # Refresh mv_inventory_health_score

# Exception queue (IPfeature7 — requires EOQ + safety stock)
# IMPORTANT: Must run AFTER ss-compute completes (fact_safety_stock_targets must have rows)
#   Dependency chain: make ss-compute → make exceptions-generate
make exceptions-generate     # Detect stockout/excess/below-ROP exceptions → DB

# Fill rate (IPfeature8 — requires inventory loaded)
make fill-rate-refresh       # Refresh mv_fill_rate_monthly

# Lead time variability (IPfeature2/3 — requires inventory loaded)
make lt-profile-compute      # LT CV, reliability bands → fact_lead_time_profile

# Demand signals (IPfeature9 — requires inventory + sales)
make demand-signals-compute  # Short-horizon signals → fact_demand_signals

# Monte Carlo simulation (IPfeature10 — requires safety stock)
make sim-run                 # Monte Carlo SS simulation → fact_ss_simulation_results

# ABC-XYZ segmentation (IPfeature11 — requires sales loaded)
make abc-xyz-classify        # Volume × variability classification → dim_sku

# Supplier performance (IPfeature12 — requires inventory loaded)
make supplier-perf-refresh   # Refresh mv_supplier_performance

# Capital investment plan (IPfeature13 — requires safety stock + EOQ)
make investment-plan         # Efficient frontier → fact_inventory_investment_plan

# Intramonth stockout (IPfeature14 — requires inventory loaded)
make intramonth-refresh      # Refresh mv_intramonth_stockout

# Inventory Rebalancing (requires agg_inventory_monthly + fact_safety_stock_targets)
make rebalancing-refresh     # Refresh mv_network_balance (network surplus/deficit view)
make rebalancing-compute     # Compute rebalancing recommendations → fact_rebalancing_recommendations
# preview without writing:
make rebalancing-compute-dry # Preview recommendations (--dry-run)
# or all-in-one:
make rebalancing-all         # rebalancing-schema + rebalancing-refresh + rebalancing-compute

# Control Tower KPIs (IPfeature15 — requires all above)
make control-tower-refresh   # Refresh mv_control_tower_kpis
```

> NOTE: Demand variability (`IPfeature1/3`) no longer has a standalone `variability-compute` target — CV, dispersion, and volatility profiles are now computed as part of `make features-compute` (Section 3) and written to `dim_sku`.

**Per-feature target breakdown** (schema / compute / refresh variants):

**EOQ computation** (IPfeature4 — requires inventory loaded):
```bash
make eoq-all         # Apply schema + compute EOQ metrics → fact_eoq_targets
make eoq-schema      # Apply DDL only
make eoq-compute     # Compute + upsert only
```

**Replenishment policies** (IPfeature5):
```bash
make policy-all      # Apply schema + upsert policies + auto-assign DFUs
make policy-schema   # Apply DDL only
make policy-assign   # Upsert policies + auto-assign DFUs from config
```

**Inventory Health Score** (IPfeature6 — requires inventory loaded):
```bash
make health-all      # Apply schema + refresh health score view
make health-schema   # Apply DDL + create materialized view
make health-refresh  # REFRESH MATERIALIZED VIEW CONCURRENTLY mv_inventory_health_score
```

**Exception Queue** (IPfeature7 — requires inventory + EOQ computed):
```bash
make exceptions-schema        # Apply DDL for fact_replenishment_exceptions (one-time)
make exceptions-generate      # Detect exceptions + write to DB
make exceptions-generate-dry  # Preview exceptions without writing to DB
```

**Fill Rate Analytics** (IPfeature8 — requires inventory loaded):
```bash
make fill-rate-all      # Apply schema + refresh fill rate view
make fill-rate-schema   # Apply DDL only
make fill-rate-refresh  # REFRESH MATERIALIZED VIEW CONCURRENTLY mv_fill_rate_monthly
```

**Demand Signals** (IPfeature9 — requires inventory loaded):
```bash
make demand-signals-all      # Apply schema + compute demand signals
make demand-signals-schema   # Apply DDL only
make demand-signals-compute  # Compute demand signals → fact_demand_signals
```

**Safety Stock Simulation** (IPfeature10 — requires inventory loaded):
```bash
make sim-schema  # Apply DDL for fact_ss_simulation_results (one-time)
make sim-run     # Run Monte Carlo safety stock simulation (reads config/inventory/inventory_planning_config.yaml simulation section)
```

**ABC-XYZ Classification** (IPfeature11 — requires sales + inventory loaded):
```bash
make abc-xyz-all      # Apply schema + run classification
make abc-xyz-schema   # Apply DDL only
make abc-xyz-classify # Run ABC-XYZ classification + write to dim_sku
```

**Supplier Performance** (IPfeature12 — requires inventory loaded):
```bash
make supplier-perf-all      # Apply schema + refresh supplier performance view
make supplier-perf-schema   # Apply DDL only
make supplier-perf-refresh  # REFRESH MATERIALIZED VIEW CONCURRENTLY mv_supplier_performance
```

**Investment Plan** (IPfeature13 — requires EOQ + policy data):
```bash
make investment-all    # Apply schema + compute investment plan
make investment-schema # Apply DDL only
make investment-plan   # Compute investment plan + efficient frontier → fact tables
```

**Intramonth Stockout** (IPfeature14 — requires inventory loaded):
```bash
make intramonth-all      # Apply schema + refresh intramonth stockout view
make intramonth-schema   # Apply DDL only
make intramonth-refresh  # REFRESH MATERIALIZED VIEW CONCURRENTLY mv_intramonth_stockout
```

**Control Tower** (IPfeature15 — requires all inv planning data):
```bash
make control-tower-all      # Apply schema + refresh control tower KPIs view
make control-tower-schema   # Apply DDL only
make control-tower-refresh  # REFRESH MATERIALIZED VIEW CONCURRENTLY mv_control_tower_kpis
```

**Inventory Rebalancing** (requires inventory + safety stock data loaded):
```bash
make rebalancing-all           # Apply schema + compute rebalancing plan (full pipeline)
make rebalancing-schema        # Apply DDL: dim_transfer_lane, fact_rebalancing_plan, fact_rebalancing_transfer, mv_network_balance (one-time)
make rebalancing-compute       # Compute rebalancing plan from inventory positions + safety stock targets
make rebalancing-compute-dry   # Preview rebalancing computation without writing to DB (--dry-run)
make rebalancing-refresh       # REFRESH MATERIALIZED VIEW CONCURRENTLY mv_network_balance
```

Rebalancing tables:
- `dim_transfer_lane` — valid transfer lanes between locations (source -> destination, lead time, cost)
- `fact_rebalancing_plan` — computed rebalancing recommendations (item, source/dest, qty, priority)
- `fact_rebalancing_transfer` — executed/planned transfer records with status tracking
- `mv_network_balance` — materialized view aggregating network-wide inventory balance metrics

Rebalancing SQL files: `sql/071_create_transfer_network.sql` (dim_transfer_lane), `sql/072_create_rebalancing_plan.sql` (fact_rebalancing_plan + fact_rebalancing_transfer), `sql/073_create_rebalancing_views.sql` (mv_network_balance)

Rebalancing config: `config/inventory/rebalancing_config.yaml` — transfer cost thresholds, minimum transfer qty, priority scoring weights, network constraints.

---

## 7.4 Configuration

All knobs live in YAML — never hardcode in scripts.

| File | Owns |
|---|---|
| `config/inventory/inventory_planning_config.yaml` | Merged config: `lead_time`, `simulation` (n_simulations: 10000), `projection` |
| `config/inventory/safety_stock_config.yaml` | Service-level targets, σ floor, MOQ rounding |
| `config/inventory/eoq_config.yaml` | Holding cost %, ordering cost, min-order-qty rules |
| `config/inventory/replenishment_policy_config.yaml` | ABC×XYZ → policy mapping (Min/Max vs R/Q vs s/S) |
| `config/inventory/replenishment_plan_config.yaml` | Period buckets, lookahead horizon |
| `config/operations/sop_config.yaml` | S&OP cycle timing (Section 8) |
| `config/planning_config.yaml` | Global `PLANNING_DATE` override behavior |
| `config/shared_constants.yaml` | Service-level Z-table, financial defaults, guard rails (inherited via `_includes`) |

Service-level Z-table reference (from `shared_constants.yaml`): SL 95% → Z=1.645, SL 98% → Z=2.054, SL 99% → Z=2.326.

Safety-stock formula:

```
SS = Z(SL) × √(LT_mean × σ_demand² + demand_mean² × σ_LT²)
```

Bounded by `min_safety_stock_days` and `max_safety_stock_days` from `safety_stock_config.yaml`.

---

## 7.5 API Surface

Six routers in `api/routers/inventory/`:

| Router | Endpoint prefix | Notes |
|---|---|---|
| `inventory_main.py` | `/inventory/*` | Core EOM, on-hand, DOS, stock-out lookups |
| `sourcing.py` | `/sourcing/*` | Supplier/DC mapping, lead-time data |
| `purchase_orders.py` | `/purchase-orders/*` | Open + closed PO views |
| `demand_history.py` | `/demand-history/*` | Customer demand drill-down |
| `integrated_targets.py` | `/inv-planning/*` (and others) | SS / EOQ / policy / projection / exceptions / fill rate |
| `inv_planning_algorithm_comparison.py` | `/inv-planning/algorithm-comparison/*` | A/B view for SS/policy variants |
| `working_capital.py` | `/inv-planning/working-capital/*` | Investment / cash-flow drill |

**CRITICAL RULE (from CLAUDE.md):** `inv_planning_*.py` routers MUST use `get_conn()` directly — never `Depends(_get_pool)`. FastAPI inspects MagicMock signatures and returns 422 in tests when `Depends(_get_pool)` is used.

All write endpoints (overrides, policy edits, scenario promotion) are guarded by `dependencies=[Depends(require_api_key)]`.

---

## 7.6 UI — InvPlanningTab

Entry: `frontend/src/tabs/InvPlanningTab.tsx`. Sub-panels live under `frontend/src/tabs/inv-planning/`.

**Default panel: `actionfeed`** (NOT `exceptions`). Common gotcha when restoring tabs from URL state.

Panels (38 total) grouped by purpose:

| Group | Panels |
|---|---|
| Triage / action | `ActionFeedPanel`, `ExceptionQueuePanel`, `OverrideQueuePanel`, `TodaysPlanBanner` |
| Demand view | `DemandForecastPanel`, `DemandPlanPanel`, `BlendedDemandPanel`, `DemandIntelligencePanel`, `DemandSignalsPanel`, `VariabilityPanel` |
| Inventory view | `SafetyStockPanel`, `EoqPanel`, `ProjectionPanel`, `RebalancingPanel`, `IntramonthPanel`, `LeadTimePanel`, `EchelonPanel` |
| Procurement | `OpenPOPanel`, `PurchaseOrdersPanel`, `ProcurementPanel`, `SourcingPanel`, `SupplierPanel`, `PlannedOrdersPanel`, `ReplenishmentPlanPanel` |
| Policy / segmentation | `PolicyManagementPanel`, `AbcXyzPanel`, `SegmentDashboardPanel`, `PortfolioHealthPanel` |
| Service / KPI | `FillRatePanel`, `ServiceLevelWaterfallPanel`, `PlanningScorecardPanel`, `NetworkHeatmapPanel` |
| Financial | `FinancialPlanPanel`, `CashFlowPanel`, `InvestmentPanel` |
| Scenario | `ScenarioPlanningPanel`, `SimulationPanel`, `ConstrainedOptPanel`, `EventCalendarPanel` |

Use `useThemeContext()` for theme — never pass `theme` as a prop.

---

## 7.7 Exception Engine

Lives at `common/engines/exception_engine.py`. Triggered by `make exceptions-generate` (script: `generate_replenishment_exceptions.py`).

Exception types:

| Type | Trigger | Severity inputs |
|---|---|---|
| Stockout risk | Projected on-hand < safety stock within LT window | DOS, demand variability |
| Overstock | Projected on-hand > max threshold | Holding cost, expiry |
| Lead-time breach | Actual LT > planned LT + tolerance | Supplier perf history |
| Expiring inventory | Shelf-life remaining < threshold | Days to expiry |
| Forecast bias | |MAPE| > threshold for N consecutive months | bias-corrections table |

Exceptions surface in:
- **API:** `/inv-planning/exceptions/*`
- **UI:** `ExceptionQueuePanel`, `ActionFeedPanel`
- **Webhooks:** registered subscribers fire on new high-severity exceptions (Section 8)

---

## 7.8 Materialized Views

| MV | Refresh target | Purpose |
|---|---|---|
| `agg_inventory_monthly` | `refresh-agg` | EOM on-hand, sales, DOS, lead time |
| `mv_inventory_forecast_monthly` | `refresh-mvs-tiered` | Inventory↔forecast bridge for root cause attribution |
| `mv_fill_rate_monthly` | `fill-rate-all` | Monthly fill rate per item × loc × customer |
| `mv_supplier_performance` | `supplier-perf-all` | Supplier OTIF, LT variance |
| `mv_intramonth_stockout` | `intramonth-all` | Mid-month stockout signal |
| `mv_network_balance` | `refresh-mvs-tiered` | Cross-DC inventory balance |
| `mv_control_tower_kpis` | `control-tower-all` | Tile KPIs for Control Tower (Section 8) |

**Refresh order matters** — tiered refresh ensures base aggregates refresh BEFORE derived MVs:

```bash
make refresh-mvs-tiered
```

Direct refresh (skipping tier order) can produce stale joins.

---

## 7.9 Verification

```bash
make health                     # API + DB liveness
make check-all                  # row counts across all key tables

# Spot checks
psql "$DATABASE_URL" <<'SQL'
SELECT COUNT(*) AS ss_rows FROM safety_stock;
SELECT COUNT(*) AS open_exceptions FROM replenishment_exceptions WHERE status = 'open';
SELECT MAX(snapshot_date) AS latest_snapshot FROM fact_inventory_snapshot;
SELECT MAX(forecast_month) AS latest_forecast FROM fact_production_forecast;
SELECT class, COUNT(*) FROM replenishment_policy GROUP BY class ORDER BY class;
SQL
```

UI sanity: open `InvPlanningTab` → confirm `ActionFeedPanel` loads with non-empty rows, `ExceptionQueuePanel` shows mixed severities, `FillRatePanel` returns this-month and rolling-12 numbers.

---

## 7.10 Re-Run Cadence

| Trigger | Re-run |
|---|---|
| New inventory snapshot loaded (daily) | `make refresh-agg` + `make intramonth-all` + `make exceptions-generate` |
| New customer demand month closed | `make pipeline-customer-demand` + `make service-level-all` + `make bias-all` |
| Forecast promoted (Section 6) | `make setup-demand-planning` + `make ss-all` + `make exceptions-generate` |
| Cluster experiment promoted (Section 3) | `make policy-all` (re-assigns policies by new ABC×XYZ class) |
| Lead-time data refresh (weekly) | `make lead-time-all` + `make ss-all` |
| Quarterly policy review | run the `inventory_backtest` job (or `uv run python scripts/inventory/run_inventory_backtest.py`) + `make algo-comparison` → review → update `replenishment_policy_config.yaml` → `make policy-all` |

---

## 7.11 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| 422 in tests for new `/inv-planning/*` endpoint | Router used `Depends(_get_pool)` | Switch to `get_conn()` per CLAUDE.md |
| Exceptions empty after `exceptions-generate` | No promoted forecast or stale snapshot | Verify `fact_production_forecast` populated; check `MAX(snapshot_date)` |
| Fill rate always 100% | `mv_fill_rate_monthly` not refreshed after demand load | `make fill-rate-all` |
| Safety stock = 0 for many SKUs | `σ_demand` floor too high or insufficient history | Inspect `safety_stock_config.yaml`; SKU may be cold-start (Section 6) |
| Projection panel shows flat line | No planned orders / replenishment plan | Run `make planned-orders-all` then `make replplan-all` |
| Control Tower KPIs stale | `mv_control_tower_kpis` not refreshed | `make control-tower-all` |
| Frontend "HTML instead of JSON" on new endpoint | Missing Vite proxy entry | Add prefix to `frontend/vite.config.ts`; run `make audit-routers` |
| `make exceptions-generate` slow (>10 min) | Sequential per-DFU, no batching | Reduce `exception_lookback_days` in config; consider per-cluster parallelism |

---

## 7.12 Adding a New Inventory Calculation

Follow CLAUDE.md "Feature Integration Checklist". Inventory-specific reminders:

1. New script → `scripts/inventory/` (create the directory; CLAUDE.md mandates this placement for new files)
2. New table → DDL in `sql/`, add `TRUNCATE` to `db-truncate-data`, add to RUNBOOK cleanup
3. New MV → register it in `MV_SOURCES` in `common/core/mv_refresh.py` (dependency order; `tests/unit/test_mv_refresh.py` fails until you do)
4. New router → place in `api/routers/inventory/`, use `get_conn()`, add Vite proxy if new prefix
5. New UI panel → `frontend/src/tabs/inv-planning/`, register in `InvPlanningTab.tsx`, co-located test
6. Backend test in `tests/api/`, frontend test in `src/tabs/__tests__/`
7. `make audit-routers` + `make test-all`
