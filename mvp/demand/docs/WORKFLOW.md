# Demand Studio — End-to-End Workflow

All commands run from `mvp/demand/` unless noted.

This document covers the full pipeline from a clean environment through data
ingestion, ML training, and production-ready UI. Follow the phases in order on
first setup; for subsequent runs only re-run the phases that have new data.

---

## Phase 0: One-Time Environment Setup

Run once per machine / new environment.

```bash
# 1. Python environment
make init              # Create .venv, install uv, sync all dependencies

# 2. Infrastructure (Docker)
make up                # Start Postgres, MLflow

# 3. Frontend dependencies
make ui-init           # npm install for React dev server
```

---

## Phase 1: Schema Setup (One-Time per Environment)

Apply all DDL in order. Safe to re-run (`IF NOT EXISTS` guards on every statement).

```bash
# Core tables
make db-apply-sql              # All core tables, indexes, materialized views

# Inventory
make db-apply-inventory        # fact_inventory_snapshot + indexes + agg view
make db-apply-inv-backtest     # mv_inventory_forecast_monthly bridge view

# Jobs
make db-apply-jobs             # job_history + job_schedule tables

# Inventory Planning (IPfeature3–15)
make ss-schema                 # fact_safety_stock_targets + indexes
make eoq-schema                # fact_eoq_targets
make policy-schema             # dim_replenishment_policy + fact_dfu_policy_assignment
make health-schema             # mv_inventory_health_score materialized view
make exceptions-schema         # fact_replenishment_exceptions
make fill-rate-schema          # mv_fill_rate_monthly
make demand-signals-schema     # fact_demand_signals
make sim-schema                # fact_ss_simulation_results
make abc-xyz-schema            # XYZ classification columns on dim_dfu
make supplier-perf-schema      # mv_supplier_performance
make investment-schema         # fact_inventory_investment_plan + fact_efficient_frontier
make intramonth-schema         # mv_intramonth_stockout
make control-tower-schema      # mv_control_tower_kpis

# AI Planning Agent
make ai-insights-schema        # ai_insights + ai_planning_memos + ai_call_log + ai_recommendation_outcomes

# Production Forecast (F1.1)
make forecast-prod-schema      # fact_production_forecast + fact_model_registry + source_model_id migration

# Forward-Looking Replenishment Plan (CI Bands + Repl. Plan)
make replplan-schema           # fact_replenishment_plan

# Chat
make db-apply-chat             # pgvector + chat_embeddings (requires OPENAI_API_KEY)
make generate-embeddings       # Embed schema descriptions (requires OPENAI_API_KEY)

# Storyboard
make storyboard-schema         # fact_storyboard_exceptions
```

> **Tip:** `make db-apply-sql` covers the majority of tables. The remaining `make *-schema` commands add feature-specific tables on top.

---

## Phase 2: Data Ingestion

### 2.1 Core Datasets (Dimensions + Facts)

```bash
make normalize-all    # Normalize all 8 datasets: CSV → clean CSV
make load-all         # Load all 8 datasets into Postgres + refresh materialized views
```

Or individually:

```bash
make normalize-item && make load-item
make normalize-location && make load-location
make normalize-customer && make load-customer
make normalize-time && make load-time       # Auto-generates 2020–2035 time dimension
make normalize-dfu && make load-dfu
make normalize-sales && make load-sales     # TYPE=1 rows only
make normalize-forecast && make load-forecast
```

**Re-loading external forecast only** (preserves backtest/champion/ceiling data):
```bash
make load-forecast-replace                  # Replace external rows, keep ML model rows
make load-forecast-replace-no-archive       # Same, skip 45M-row archive (faster)
```

### 2.2 Inventory Snapshots

```bash
make inventory-pipeline    # normalize + load + refresh (all-in-one, ~190M rows)
```

---

## Phase 3: Inventory Planning Computations

Run after Phase 2 completes. These derive planning signals from inventory + sales data.

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
make exceptions-generate     # Detect stockout/excess/below-ROP exceptions → DB

# Fill rate (IPfeature8 — requires inventory loaded)
make fill-rate-refresh       # Refresh mv_fill_rate_monthly

# Demand variability (IPfeature1/3 — requires sales loaded)
make variability-compute     # CV, dispersion, volatility profiles → dim_dfu

# Lead time variability (IPfeature2/3 — requires inventory loaded)
make lt-profile-compute      # LT CV, reliability bands → fact_lead_time_profile

# Demand signals (IPfeature9 — requires inventory + sales)
make demand-signals-compute  # Short-horizon signals → fact_demand_signals

# Monte Carlo simulation (IPfeature10 — requires safety stock)
make sim-run                 # Monte Carlo SS simulation → fact_ss_simulation_results

# ABC-XYZ segmentation (IPfeature11 — requires sales loaded)
make abc-xyz-classify        # Volume × variability classification → dim_dfu

# Supplier performance (IPfeature12 — requires inventory loaded)
make supplier-perf-refresh   # Refresh mv_supplier_performance

# Capital investment plan (IPfeature13 — requires safety stock + EOQ)
make investment-plan         # Efficient frontier → fact_inventory_investment_plan

# Intramonth stockout (IPfeature14 — requires inventory loaded)
make intramonth-refresh      # Refresh mv_intramonth_stockout

# Control Tower KPIs (IPfeature15 — requires all above)
make control-tower-refresh   # Refresh mv_control_tower_kpis
```

---

## Phase 4: ML — Clustering & Seasonality

Run after Phase 2. These enrich `dim_dfu` with segment labels used as ML features.

```bash
# DFU clustering (groups DFUs by demand pattern)
make cluster-all             # features → train → label → update dim_dfu

# Seasonality profiles
make seasonality-all         # detect → update dim_dfu
```

---

## Phase 5: ML — Backtesting

Run after Phase 4 (clustering needed for cluster-based models).
Each backtest trains models AND persists `.pkl` artifacts to `data/models/<model_id>/` for production forecasting.

```bash
# Run all three model backtests
make backtest-all            # LGBM → CatBoost → XGBoost (sequential)
# or in parallel:
make backtest-all-parallel

# Load backtest predictions into Postgres
make backtest-load-all       # Scan data/backtest/*/ and upsert all models
```

**Optional: Hyperparameter tuning** (before backtests, improves accuracy):
```bash
make tune-all                # Bayesian Optuna tuning → data/tuning/best_params_*.json
# Then set params_file in config/algorithm_config.yaml before running backtests
```

---

## Phase 6: ML — Champion Model Selection

Run after Phase 5 (requires ≥2 backtest models in DB).

```bash
# Apply source_model_id schema (one-time after sql/041 migration)
# Already included in: make forecast-prod-schema

# Select per-DFU champion + ceiling models
make champion-select         # Writes champion + ceiling rows to fact_external_forecast_monthly
                             # Stores source_model_id (e.g. lgbm_cluster) per champion row
```

**Optional: Simulate strategies / train meta-learner:**
```bash
make champion-simulate       # Compare all strategies vs ceiling accuracy
make champion-train-meta     # Train meta-learner classifier (for meta_learner strategy)
make champion-all            # train-meta + simulate + select (full pipeline)
```

---

## Phase 7: Production Forecast Generation (F1.1)

Run after Phase 6. Generates future-period (T+1 to T+12) demand forecasts using champion ML model artifacts.

```bash
make forecast-generate       # Generate 12-month forward forecasts → fact_production_forecast
# or for a single DFU:
make forecast-generate-dfu ITEM=100320 LOC=1401-BULK
# preview without writing:
make forecast-generate-dry
```

**Dependency chain for `make forecast-generate`:**
1. `data/models/lgbm_cluster/cluster_*.pkl` must exist (from Phase 5)
2. Champion assignments with `source_model_id` must exist (from Phase 6)
3. Recent sales history must be loaded (from Phase 2)

---

## Phase 7b: Forward-Looking Replenishment Plan (CI Bands + Repl. Plan)

Run after Phase 7 (production forecast must exist). Computes forward SS, EOQ, and order quantities from CI-band forecasts.

```bash
make replplan-compute        # Compute 12-month replenishment plan → fact_replenishment_plan
# preview without writing:
make replplan-compute-dry
```

**Dependency chain for `make replplan-compute`:**
1. `fact_production_forecast` must have rows (from Phase 7)
2. `fact_safety_stock_targets` must have rows (from Phase 3 `ss-compute`)
3. `fact_eoq_targets` must have rows (from Phase 3 `eoq-compute`)

---

## Phase 8: AI Planning Agent

Run after Phases 2–7 to give the agent full context.

```bash
make ai-insights-scan        # Portfolio-wide AI exception scan → ai_insights table
make ai-insights-dfu ITEM=100320 LOC=1401-BULK  # Single DFU analysis
```

---

## Phase 9: Storyboard Exceptions

```bash
make storyboard-generate     # Detect planning exceptions → fact_storyboard_exceptions
```

---

## Phase 10: Start Services

```bash
# Terminal 1 — FastAPI backend
make api                     # FastAPI on :8000

# Terminal 2 — React frontend
make ui                      # Vite dev server on :5173
```

Open browser: `http://localhost:5173`

---

## Full First-Time Run (New Environment)

```bash
# 0. Setup
make init && make up && make ui-init

# 1. Schema (one-time)
make db-apply-sql
make db-apply-inventory db-apply-inv-backtest db-apply-jobs
make ss-schema eoq-schema policy-schema health-schema exceptions-schema
make fill-rate-schema demand-signals-schema sim-schema abc-xyz-schema
make supplier-perf-schema investment-schema intramonth-schema control-tower-schema
make ai-insights-schema storyboard-schema forecast-prod-schema

# 2. Ingest
make normalize-all && make load-all
make inventory-pipeline

# 3. Inventory Planning
make ss-compute eoq-compute policy-assign health-refresh
make exceptions-generate fill-rate-refresh variability-compute lt-profile-compute
make demand-signals-compute sim-run abc-xyz-classify
make supplier-perf-refresh investment-plan intramonth-refresh control-tower-refresh

# 4. Clustering + Seasonality
make cluster-all && make seasonality-all

# 5. Backtesting
make backtest-all && make backtest-load-all

# 6. Champion selection
make champion-select

# 7. Production forecasts
make forecast-generate

# 8. AI insights
make ai-insights-scan

# 9. Storyboard
make storyboard-generate

# 10. Start services
make api   # terminal 1
make ui    # terminal 2
```

---

## Incremental Refresh (New Data Arrives)

When new monthly data files are added:

```bash
# Re-ingest changed datasets
make load-forecast-replace       # New external forecast (preserves ML rows)
make inventory-pipeline          # New inventory snapshots

# Re-compute dependent views
make health-refresh fill-rate-refresh intramonth-refresh
make demand-signals-compute control-tower-refresh

# Re-run backtests (if model needs refreshing)
make backtest-all && make backtest-load-all
make champion-select

# Regenerate production forecasts
make forecast-generate

# Refresh AI insights
make ai-insights-scan
```

---

## Validation

```bash
make check-db    # Table row counts in Postgres
make check-api   # curl API health + sample endpoints
make test-all    # Full test suite (backend + frontend)
```

---

## Key Dependencies Between Phases

```
Phase 2 (Ingest)
    └─► Phase 3 (Inv Planning)
    └─► Phase 4 (Clustering + Seasonality)  ──► Phase 5 (Backtesting)
                                                    └─► Phase 6 (Champion Select)
                                                            └─► Phase 7 (Prod Forecast)
    └─► Phase 8 (AI Insights)  [needs 3 + 6 + 7]
    └─► Phase 9 (Storyboard)   [needs 3]
```
