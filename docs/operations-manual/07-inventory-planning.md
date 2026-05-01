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
| `dim_sku` with `ml_cluster` populated | `make features-compute` + `make cluster-all` (Section 3) | Per-cluster policy assignment |

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
make ip-backtest        # run_inventory_backtest.py — backtest SS/policy choices
make ip-compare         # compare_inventory_algorithms.py — A/B between policy variants
```

Surfaces in the **Inventory Algorithm Comparison** API router for the UI A/B view.

---

## 7.4 Configuration

All knobs live in YAML — never hardcode in scripts.

| File | Owns |
|---|---|
| `config/inventory_planning_config.yaml` | Merged config: `lead_time`, `simulation` (n_simulations: 10000), `projection` |
| `config/safety_stock_config.yaml` | Service-level targets, σ floor, MOQ rounding |
| `config/eoq_config.yaml` | Holding cost %, ordering cost, min-order-qty rules |
| `config/replenishment_policy_config.yaml` | ABC×XYZ → policy mapping (Min/Max vs R/Q vs s/S) |
| `config/replenishment_plan_config.yaml` | Period buckets, lookahead horizon |
| `config/sop_config.yaml` | S&OP cycle timing (Section 8) |
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
| Quarterly policy review | `make ip-backtest` + `make ip-compare` → review → update `replenishment_policy_config.yaml` → `make policy-all` |

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
3. New MV → add `REFRESH MATERIALIZED VIEW` to the correct tier in `refresh-mvs-tiered`
4. New router → place in `api/routers/inventory/`, use `get_conn()`, add Vite proxy if new prefix
5. New UI panel → `frontend/src/tabs/inv-planning/`, register in `InvPlanningTab.tsx`, co-located test
6. Backend test in `tests/api/`, frontend test in `src/tabs/__tests__/`
7. `make audit-routers` + `make test-all`
