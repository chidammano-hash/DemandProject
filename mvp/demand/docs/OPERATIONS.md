# Supply Chain Command Center — Operations Guide

All commands run from `mvp/demand/` unless noted.

This document is the single operations reference: environment setup, data pipeline,
ML training, platform services, testing, cleanup, and troubleshooting. Follow the
phases in order on first setup; for subsequent runs only re-run the phases that have
new data.

---

## Phase 0: One-Time Environment Setup

Run once per machine / new environment.

```bash
# 1. Python environment
make init              # Create .venv, install uv, sync all dependencies

# 2. Infrastructure (Docker)
make up                # Start Postgres, MLflow
make down              # Stop all services

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
make db-apply-inv-backtest     # mv_inventory_forecast_monthly bridge view (Feature 37)

# Jobs
make db-apply-jobs             # job_history + job_schedule tables + indexes

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

# Inventory Rebalancing
make rebalancing-schema        # mv_network_balance + fact_rebalancing_recommendations

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

# Auth / RBAC (Spec 08-02)
make auth-schema               # dim_user + fact_audit_log (sql/062)

# Data Quality (Spec 08-01)
make dq-schema                 # dim_dq_check_catalog + fact_dq_check_results + mv_dq_dashboard (sql/063)

# Cache Performance Tracking (Spec 08-03)
make cache-perf-schema         # fact_query_performance (sql/064)

# Notifications (Spec 08-04)
make notification-schema       # dim_notification_channel + fact_notification_log (sql/065)

# Collaboration (Spec 08-05)
make collaboration-schema      # fact_annotations (sql/066)

# External Signals (Spec 08-06)
make external-signals-schema   # fact_external_signals (sql/067)

# FVA Tracking (Spec 08-07)
make fva-schema                # fact_fva_tracking (sql/068)

# Reporting (Spec 08-08)
make report-schema             # dim_report_template + fact_report_schedule + fact_report_delivery (sql/069)

# Medallion Pipeline (layered ETL)
make medallion-schema          # audit_load_batch, bronze_*, silver_*, quarantine, lineage (sql/080-086)
```

> **Tip:** `make db-apply-sql` covers the majority of tables (including DDL 062-070 for auth, data quality, cache, notifications, webhooks, reports, rate limiting). The remaining `make *-schema` commands add feature-specific tables on top.

---

## Phase 1b: Auth & RBAC Setup

Run after schema setup. Seeds default admin user and configures JWT-based authentication.

```bash
# Auth config lives in config/auth_config.yaml (JWT secret, token TTL, role hierarchy)
# common/auth.py provides: CurrentUser, get_current_user, require_role dependencies
# api/routers/auth_router.py provides: POST /auth/login, POST /auth/refresh
# api/routers/users.py provides: CRUD for dim_user (admin-only)

# No Make target needed — auth is auto-initialized when API starts.
# All mutation endpoints use require_role() for RBAC enforcement.
# Audit log entries written to fact_audit_log on every state-changing request.
```

---

## Phase 2: Data Ingestion

### 2.1 All Datasets (Recommended Starting Point)

```bash
make normalize-all     # Normalize all 8 datasets (CSV → clean CSV)
make load-all          # Load all datasets into Postgres + refresh materialized views
```

### 2.2 Individual Datasets

```bash
make normalize-item && make load-item
make normalize-location && make load-location
make normalize-customer && make load-customer
make normalize-time && make load-time       # Auto-generates 2020–2035 time dimension
make normalize-dfu && make load-dfu
make normalize-sales && make load-sales     # TYPE=1 rows only
make normalize-forecast && make load-forecast
```

**Forecast loading flags:**

```bash
make load-forecast                      # TRUNCATE all + reload external (exec-lag → main, all lags → archive)
make load-forecast-replace              # Replace external only (preserves backtest/champion/ceiling)
make load-forecast-replace-no-archive   # Replace external only, skip 45M-row archive INSERT (fast)
```

### 2.3 Inventory Snapshots (14 Monthly CSVs, ~190M Rows)

```bash
make normalize-inventory    # Merge 14 monthly CSVs → single clean CSV
make load-inventory         # Load into Postgres + refresh agg view
make inventory-pipeline     # normalize + load + refresh (all-in-one)
```

### 2.4 Chatbot Embeddings (Requires `OPENAI_API_KEY` in `.env`)

```bash
make generate-embeddings
```

Parses all domain specs, generates OpenAI embeddings for schema metadata, and stores them in the `chat_embeddings` pgvector table. Re-run after adding new datasets or changing schema.

---

## Phase 2b: Data Quality Checks

Run after Phase 2 ingestion to validate loaded data. Can also be scheduled as a recurring pipeline step.

```bash
# Data quality checks are config-driven (config/data_quality_config.yaml)
# common/dq_engine.py runs SQL-based checks: freshness, completeness, uniqueness,
#   range validation, volume delta, referential integrity
# Results written to fact_dq_check_results; domain health scores in mv_dq_dashboard

make dq-run                  # Run all enabled DQ checks → fact_dq_check_results
make dq-run-dry              # Preview checks without writing (--dry-run)
```

API endpoints (`/data-quality/*`):
- `GET /data-quality/dashboard` — domain health scores and recent check trends
- `GET /data-quality/checks` — list all check definitions from catalog
- `GET /data-quality/results` — recent check results with pass/fail/warn status
- `POST /data-quality/run` — trigger ad-hoc DQ check run (requires planner role)

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
# IMPORTANT: Must run AFTER ss-compute completes (fact_safety_stock_targets must have rows)
#   Dependency chain: make ss-compute → make exceptions-generate
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

**Inventory Planning feature details:**

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
make sim-run     # Run Monte Carlo safety stock simulation (reads config/simulation_config.yaml)
```

**ABC-XYZ Classification** (IPfeature11 — requires sales + inventory loaded):
```bash
make abc-xyz-all      # Apply schema + run classification
make abc-xyz-schema   # Apply DDL only
make abc-xyz-classify # Run ABC-XYZ classification + write to dim_dfu
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

**Tables:**
- `dim_transfer_lane` — valid transfer lanes between locations (source -> destination, lead time, cost)
- `fact_rebalancing_plan` — computed rebalancing recommendations (item, source/dest, qty, priority)
- `fact_rebalancing_transfer` — executed/planned transfer records with status tracking
- `mv_network_balance` — materialized view aggregating network-wide inventory balance metrics

**SQL files:** `sql/071_create_transfer_network.sql` (dim_transfer_lane), `sql/072_create_rebalancing_plan.sql` (fact_rebalancing_plan + fact_rebalancing_transfer), `sql/073_create_rebalancing_views.sql` (mv_network_balance)

**Config:** `config/rebalancing_config.yaml` — transfer cost thresholds, minimum transfer qty, priority scoring weights, network constraints.

---

## Phase 4: ML — Clustering & Seasonality

Run after Phase 2. These enrich `dim_dfu` with segment labels used as ML features.

```bash
# DFU clustering (groups DFUs by demand pattern)
make cluster-all             # features → train → label → update dim_dfu

# Seasonality profiles
make seasonality-all         # detect → update dim_dfu
```

**Clustering individual steps:**

```bash
make cluster-features  # Extract 14 core features (volume, trend, seasonality, periodicity, intermittency, lifecycle) from 36-month sales history
make cluster-train     # KMeans with combined Silhouette + Calinski-Harabasz scoring, 5% min cluster size, k_range [5,18] (logged to MLflow: dfu_clustering)
make cluster-label     # Priority-ordered taxonomy labeling: Intermittency → Periodicity → Seasonality → Trend → Volatility → Volume (5 tiers)
make cluster-update    # Write cluster_assignment to dim_dfu in Postgres
```

Cluster assignments are filterable in the Data Explorer via the `cluster_assignment` column and viewable via `/domains/dfu/clusters`.

---

## Phase 5: ML — Backtesting

Run after Phase 4 (clustering needed for cluster-based models).
Each backtest trains models AND persists `.pkl` artifacts to `data/models/<model_id>/` for production forecasting.

### Configuration (Feature 44)

Before running any backtest, edit `config/algorithm_config.yaml` to set options for each algorithm:

```yaml
lgbm:
  cluster_strategy: "per_cluster"  # "per_cluster" or "global" — ml_cluster always a hard feature
  recursive: false       # Set true for recursive multi-step inference (Feature 43)
  shap_select: false     # Set true for per-timeframe SHAP feature selection (Feature 42)
  shap_threshold: 0.95   # Cumulative SHAP importance threshold
  shap_top_n: null       # Exact top-N features (overrides threshold)
  shap_sample_size: 500
  tune_inline: false     # Set true for per-timeframe causal tuning (PL-002)
  params_file: null      # Set to data/tuning/best_params_lgbm.json for pre-tuned params
  # Default hyperparameters used when params_file is null and tune_inline is false
  n_estimators: 500
  learning_rate: 0.05
  # ... (see full file for all keys)
```

Same structure for `catboost:` and `xgboost:` sections.

**Architecture (Feature 44):** Tree-based models (LGBM, CatBoost, XGBoost) share `common/backtest_framework.py` via `run_tree_backtest()`. Each script provides both `train_and_predict_per_cluster()` and `train_and_predict_global()`, selecting based on the `cluster_strategy` config key (`per_cluster` or `global`). **`ml_cluster` is always a hard feature** — never stripped from feature_cols in either strategy. Algorithm options (cluster_strategy, recursive, shap_select, tune_inline, params_file, hyperparameters) are read from `config/algorithm_config.yaml`. Shared modules: `feature_engineering.py`, `metrics.py`, `mlflow_utils.py`, `db.py`, `constants.py`, `tuning.py`, `shap_selector.py`.

### Optional: Hyperparameter Tuning (Before Backtests)

Two modes — choose based on your goal:

| Mode | When to use | How to activate (Feature 44) |
|---|---|---|
| **Global** — tune once on full history, apply via `params_file` | Production scoring, fastest path | `make tune-lgbm` then set `params_file: data/tuning/best_params_lgbm.json` in `config/algorithm_config.yaml` |
| **Inline** — per-timeframe causal tuning inside each backtest fold | Honest backtest evaluation, no future leakage (PL-002) | Set `tune_inline: true` in `config/algorithm_config.yaml`, then `make backtest-lgbm` |

> **Warning:** Using a globally tuned `params_file` in backtests introduces temporal leakage — the tuner sees future timeframes. Use `tune_inline: true` for unbiased backtest accuracy.

**Global tuning (Mode A):**
```bash
make tune-lgbm      # ~20–40 min  → data/tuning/best_params_lgbm.json
make tune-catboost  # ~30–60 min  → data/tuning/best_params_catboost.json
make tune-xgboost   # ~25–50 min  → data/tuning/best_params_xgboost.json
make tune-all       # All three sequentially
```

Each JSON file contains `best_params`, `best_n_estimators`, per-cluster WAPEs, and CV fold metrics. Apply in backtest by setting `params_file: data/tuning/best_params_<model>.json` in `config/algorithm_config.yaml` (Feature 44).

**Inline per-timeframe tuning (Mode B, PL-002 fix):**

Set `tune_inline: true` in `config/algorithm_config.yaml` for the relevant algorithm section, then run `make backtest-lgbm` (or catboost/xgboost). Each of the 10 timeframes (~20 trials x 3 folds, ~2-3x slower) tunes on only data available up to its training cutoff — no future leakage into backtest accuracy metrics.

### Run Backtests

#### LGBM (per-cluster)

```bash
# Edit config/algorithm_config.yaml lgbm: section, then:
make backtest-lgbm
make backtest-load MODEL=lgbm_cluster
```

#### CatBoost (per-cluster)

```bash
# Edit config/algorithm_config.yaml catboost: section, then:
make backtest-catboost
make backtest-load MODEL=catboost_cluster
```

#### XGBoost (per-cluster)

```bash
# Edit config/algorithm_config.yaml xgboost: section, then:
make backtest-xgboost
make backtest-load MODEL=xgboost_cluster
```

#### Run All Three

```bash
# Sequential (safe default)
make backtest-all
make backtest-load-all

# Parallel (faster on servers with 16+ cores / 32GB+ RAM)
# Each process logs to data/backtest/logs/<model>.log — no interleaved output
make backtest-all-parallel
make backtest-load-all
```

> **Note:** `backtest-all-parallel` fires all three processes simultaneously. LGBM and XGBoost use `n_jobs=-1` (all cores); running them in parallel fully saturates CPU and RAM. Use sequential on laptops or machines with limited resources.

### Loading Predictions

```bash
make backtest-load MODEL=<model_id>   # Load one model
make backtest-load-all                # Scan data/backtest/*/ and load all models
```

`--replace` is built into `backtest-load` — it only deletes rows for the loaded `model_id`, leaving all other models untouched. Accuracy materialized views are refreshed automatically after every load.

Each backtest run writes to a **model-scoped subdirectory** so multiple models can run back-to-back without overwriting each other:
- `data/backtest/<model_id>/backtest_predictions.csv` — execution-lag predictions (-> `fact_external_forecast_monthly`)
- `data/backtest/<model_id>/backtest_predictions_all_lags.csv` — lag 0-4 archive (-> `backtest_lag_archive`)

**Available model IDs (Feature 44):**

| Framework | Per-Cluster (default) | Global |
|---|---|---|
| LGBM | `lgbm_cluster` | `lgbm_global` |
| CatBoost | `catboost_cluster` | `catboost_global` |
| XGBoost | `xgboost_cluster` | `xgboost_global` |

Verify archive data:
```bash
docker exec demand-mvp-postgres psql -U demand -d demand_mvp \
  -c "SELECT model_id, lag, COUNT(*) FROM backtest_lag_archive GROUP BY 1,2 ORDER BY 1,2"
```

---

## Phase 6: Champion Model Selection

Run after Phase 5 (requires 2+ backtest models in DB).

Identifies the best model per DFU per month. All strategies enforce **strict exec-lag-aware causality** — selection for month T uses only data from months where `startdate < fcstdate`.

### Strategies

| Strategy | Key Idea |
|---|---|
| `expanding` *(default)* | Cumulative WAPE over all prior months, equal weight |
| `rolling` | Last N months only (`window_months`, default 6) |
| `decay` | Exponential decay — recent months weighted more (`decay_factor`, default 0.9) |
| `ensemble` | Blend top-K models by inverse-WAPE weights (`top_k`, default 3) |
| `meta_learner` | ML classifier trained on DFU features + performance stats |

### Commands

```bash
make champion-select          # Run with strategy from config/model_competition.yaml
make champion-train-meta      # Train meta-learner (required before using meta_learner strategy)
make champion-simulate        # Simulate all strategies, compare accuracy vs ceiling
make champion-all             # train-meta + simulate + select (full pipeline)

# Override strategy:
.venv/bin/python -m scripts.run_champion_selection --strategy rolling
```

### Config (`config/model_competition.yaml`)

```yaml
competition:
  metric: accuracy_pct
  lag: execution
  min_dfu_rows: 3
  models: [lgbm_cluster, catboost_cluster, xgboost_cluster]
  strategy: expanding          # expanding | rolling | decay | ensemble | meta_learner
  strategy_params:
    window_months: 6
    decay_factor: 0.90
    top_k: 3
    performance_window: 6
  meta_learner:
    model_type: random_forest  # random_forest | xgboost
    n_estimators: 200
    max_depth: 15
    test_months: 3
    performance_window: 6
```

### Output

- `model_id='champion'` rows in `fact_external_forecast_monthly`
- `model_id='ceiling'` rows (oracle best-with-hindsight) in `fact_external_forecast_monthly`
- `data/champion/champion_summary.json` — champion + ceiling metrics
- `data/champion/simulation_results.json` — strategy comparison
- All accuracy materialized views refreshed automatically; running again is idempotent

### API

```bash
curl http://localhost:8000/competition/config
curl -X PUT http://localhost:8000/competition/config \
  -H "Content-Type: application/json" \
  -d '{"metric": "wape", "lag": "execution", "min_dfu_rows": 3,
       "models": ["lgbm_cluster", "catboost_cluster"],
       "strategy": "rolling", "strategy_params": {"window_months": 6}}'
curl -X POST http://localhost:8000/competition/run
curl http://localhost:8000/competition/summary
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

## Phase 8-9: AI, Storyboard, DQ, FVA, S&OP, Notifications, Reports, Medallion

### AI Planning Agent (IPAIfeature1)

Run after Phases 2-7 to give the agent full context. Requires `ANTHROPIC_API_KEY` in `.env`.

```bash
make ai-insights-scan        # Portfolio-wide AI exception scan → ai_insights table
make ai-insights-dfu ITEM=100320 LOC=1401-BULK  # Single DFU analysis
make ai-insights-all         # ai-insights-schema + ai-insights-scan (full pipeline)
```

### Storyboard Exceptions (Feature 40)

```bash
make storyboard-generate     # Detect planning exceptions → fact_storyboard_exceptions
```

### Data Quality Pipeline (Automated, 08-01)

Beyond the one-off Phase 2b checks, Data Quality runs as a recurring automated pipeline.

```bash
make dq-schema               # dim_dq_check_catalog + fact_dq_check_results + mv_dq_dashboard
make dq-populate             # Populate check catalog from config/data_quality_config.yaml
make dq-run                  # Run all enabled DQ checks → fact_dq_check_results
make dq-all                  # Full pipeline: schema + populate + run
```

**Automated schedule:** Every 4 hours via APScheduler (`dq_check` job type in job_registry).
The scheduler triggers `common/dq_engine.py` which evaluates SQL-based rules (freshness,
completeness, uniqueness, range, volume delta, referential integrity), writes results to
`fact_dq_check_results`, and refreshes `mv_dq_dashboard` domain health scores.

Dashboard: `GET /data-quality/dashboard` shows pass/warn/fail trends per domain.

```bash
# Run checks via API
curl http://localhost:8000/data-quality/run          # Execute all configured checks
curl http://localhost:8000/data-quality/results       # View latest results
curl http://localhost:8000/data-quality/catalog        # View/manage check catalog
```

### FVA Tracking (08-07)

FVA (Forecast Value Add) tracks whether human forecast interventions improve or degrade accuracy.

```bash
make fva-schema              # fact_fva_tracking (sql/068)
```

> FVA interventions are populated through user actions in the UI (override queue, manual adjustments). No batch seed step needed.

API endpoints (`/fva/*`):
- `GET /fva/waterfall` — step-by-step accuracy waterfall (statistical -> override -> consensus -> final)
- `GET /fva/roi` — ROI dashboard: intervention count, accuracy lift, cost of touch
- `GET /fva/detail` — per-DFU intervention history with before/after metrics

Dashboard: FVA tab shows waterfall chart + ROI summary. Config in `config/fva_config.yaml`.

### S&OP Cycle (F4.2)

S&OP (Sales & Operations Planning) runs as a multi-stage approval workflow.

```bash
make sop-schema     # Apply DDL (one-time)
make sop-seed       # Seed initial S&OP cycle
make sop-all        # Full setup: schema + seed
```

**Cycle stages** (sequential, each requires approval to advance):

1. **Demand Review** — Review statistical + ML forecasts, apply overrides
2. **Supply Review** — Validate inventory positions, capacity, lead times
3. **Pre-S&OP** — Cross-functional alignment, gap identification
4. **Executive S&OP** — Final executive review and trade-off decisions
5. **Approved** — Plan locked, execution begins
6. **Closed** — Cycle archived after month-end actuals reconciliation

API endpoints (`/sop/*`):
- `POST /sop/cycles` — seed a new cycle
- `PUT /sop/cycles/{id}/advance` — advance to next stage (requires planner/admin role)
- `GET /sop/cycles` — list cycles with current stage + gap metrics
- `GET /sop/cycles/{id}/plan` — approved plan detail

Dashboard: S&OP tab (sidebar "S&OP") shows stage machine, gap cards, advance/approve buttons.
Config in `config/sop_config.yaml`.

### Notification & Webhook Configuration (08-04, 08-10)

Configure notification channels and webhook subscriptions. These fire automatically
when pipeline events occur (DQ failures, AI insights, exception alerts, report delivery).

```bash
# Notification channels configured via config/notification_config.yaml
# common/notification_engine.py dispatches to Slack, Teams, Email, PagerDuty
# Delivery history stored in fact_notification_log (sql/065)
```

API endpoints (`/notifications/*`):
- `GET /notifications/history` — past notification deliveries with status
- `POST /notifications/test` — send a test notification to verify channel config

```bash
# Webhook subscriptions registered via API (no Make target needed)
# common/webhook_dispatcher.py signs and dispatches HTTPS callbacks on events
# Event types: dq.check.failed, insight.created, exception.detected,
#              forecast.generated, report.delivered, etc.
```

API endpoints (`/webhooks/*`):
- `POST /webhooks/register` — register a webhook URL + event types (requires admin)
- `GET /webhooks/list` — list active webhook subscriptions
- `DELETE /webhooks/{id}` — deactivate a webhook subscription

### Report Generation Pipeline (08-08)

Configure and schedule automated report generation and distribution.

```bash
# Report templates + schedules configured via config/reporting_config.yaml
# dim_report_template stores reusable report definitions (SQL query config + layout)
# fact_report_schedule stores cron-based recurring report jobs
# fact_report_delivery tracks generation + distribution history
```

API endpoints (`/reports/*`):
- `GET /reports/templates` — list available report templates (system + user-created)
- `POST /reports/schedule` — create a recurring report schedule (cron + recipients)
- `GET /reports/schedules` — list active report schedules
- `POST /reports/generate` — trigger ad-hoc report generation
- `GET /reports/history` — past report deliveries with download links

```bash
# List available report templates
curl http://localhost:8000/reports/templates

# Schedule a recurring report
curl -X POST http://localhost:8000/reports/schedule \
  -H "Content-Type: application/json" \
  -d '{"template_id": "portfolio_summary", "schedule": "0 8 * * 1", "recipients": ["team@example.com"]}'
```

Templates define the data queries, layout, and output format (PDF/CSV/Excel).

### Medallion Data Pipeline

Bronze -> Silver -> Gold layered ETL with DQ gate checks, auto-fix, row lineage, and quarantine.

```bash
# One-time schema (if not done in Phase 1):
make medallion-schema        # Apply DDL: sql/080-086 (audit_load_batch, bronze_*, silver_*,
                             # silver_quarantine, dq_corrections_audit, row_lineage, fact_sales_original)

# Ingest data through medallion layers:
make medallion-load-sales     # Sales only: bronze → silver → gold (fact_sales_monthly)
make medallion-load-all       # All domains: bronze → silver → gold
make medallion-load-all-fix   # All domains with DQ auto-fix enabled

# Maintenance:
make medallion-prune          # Remove expired rows per retention config (bronze 90d, silver 30d)

# All-in-one:
make medallion-all            # schema + load-all + prune
```

Config: `config/medallion_config.yaml` — layer retention (bronze 90d, silver 30d), promotion gates (`min_pass_rate: 95%`), auto-fix strategies per domain, sales dual-track (original + corrected).

**Dependency:** Requires Phase 2 data loaded first (needs populated dimension + fact tables).

---

## Phase 10: Start Services

### API Middleware Stack

The FastAPI backend includes platform middleware applied in `api/main.py`:
- **GZip compression** — responses > 1KB are gzip-compressed
- **CORS** — allows `localhost:5173` (dev) origins
- **Cache layer** — `common/cache.py` provides `@cached` decorator with TTL-based caching.
  Two backends: Redis (when `REDIS_URL` env var set) or in-memory fallback.
  Config in `config/cache_config.yaml`. Cache invalidation on write endpoints.
- **Query performance tracking** — endpoint latency + DB query counts logged to
  `fact_query_performance` for API governance and observability.

### Start Services

```bash
# Terminal 1 — FastAPI backend
make api                     # FastAPI on :8000

# Terminal 2 — React frontend
make ui                      # Vite dev server on :5173
```

Open in browser:
- **UI:** `http://localhost:5173`
- **API:** `http://localhost:8000`
- **MLflow:** `http://localhost:5003`

> **Vite proxy:** Every new API path prefix must be added to `frontend/vite.config.ts` — otherwise the frontend receives HTML instead of JSON. Restart `make ui` after changes.

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
make rebalancing-schema
make ai-insights-schema storyboard-schema forecast-prod-schema
make auth-schema dq-schema cache-perf-schema notification-schema
make collaboration-schema external-signals-schema fva-schema report-schema
make medallion-schema

# 2. Ingest
make normalize-all && make load-all
make inventory-pipeline

# 2b. Data Quality
make dq-run

# 3. Inventory Planning
make ss-compute eoq-compute policy-assign health-refresh
make exceptions-generate fill-rate-refresh variability-compute lt-profile-compute
make demand-signals-compute sim-run abc-xyz-classify
make supplier-perf-refresh investment-plan intramonth-refresh
make rebalancing-refresh rebalancing-compute
make control-tower-refresh

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

# 9b. Data Quality (also runs automatically every 4h)
make dq-run

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

# Validate ingested data
make dq-run                      # Run data quality checks on refreshed data

# Re-compute dependent views
make health-refresh fill-rate-refresh intramonth-refresh
make demand-signals-compute
make rebalancing-refresh rebalancing-compute
make control-tower-refresh

# Re-run backtests (if model needs refreshing)
make backtest-all && make backtest-load-all
make champion-select

# Regenerate production forecasts
make forecast-generate

# Refresh AI insights
make ai-insights-scan

# Notifications & webhooks fire automatically on pipeline events
# (DQ failures, new AI insights, exception alerts, forecast generation)
```

---

## Platform Services Configuration

### Authentication & Authorization (Spec 08-02)

Set `JWT_SECRET` in `.env` to enable JWT-based auth:

```bash
# .env
JWT_SECRET=your-secret-key-here    # Required for token signing/verification
JWT_ALGORITHM=HS256                # Default algorithm (optional, defaults to HS256)
JWT_EXPIRY_MINUTES=60              # Token lifetime (optional, defaults to 60)
```

**Default dev mode:** When `JWT_SECRET` is not set, the API runs in anonymous admin mode — all requests are treated as authenticated with admin privileges. Set `JWT_SECRET` in any non-local environment.

Passwords are hashed with bcrypt (12 rounds). Plaintext passwords are never stored.

**New Python dependencies** (added to `pyproject.toml`):
- `bcrypt` — password hashing for user accounts
- `PyJWT` — JWT token creation and validation

### Cache Management (Spec 08-03)

Default backend is **InMemory** (no external dependencies). For production, configure Redis:

```bash
# .env (optional — InMemory is the default)
CACHE_BACKEND=redis               # "memory" (default) or "redis"
CACHE_REDIS_URL=redis://localhost:6379/0
CACHE_DEFAULT_TTL=300             # Default TTL in seconds (optional)
```

InMemory cache is cleared on process restart. Redis cache persists across restarts and is shared across workers.

### Notifications (Spec 08-04)

Configure notification channels in `config/notification_config.yaml`:

```yaml
channels:
  email:
    enabled: false
    smtp_host: smtp.example.com
    smtp_port: 587
    from_address: alerts@example.com
  slack:
    enabled: false
    webhook_url: https://hooks.slack.com/services/...
  webhook:
    enabled: false
    url: https://your-endpoint.com/notify
```

Channels are disabled by default. Enable and configure as needed. Notification preferences are per-user and managed via the API.

### Webhooks (Spec 08-10)

Webhooks use **HMAC-SHA256 signing** for payload verification. Each webhook subscription has its own signing secret.

```bash
# Create a webhook subscription via API
curl -X POST http://localhost:8000/webhooks/subscriptions \
  -H "Content-Type: application/json" \
  -d '{"url": "https://your-endpoint.com/hook", "events": ["insight.created", "exception.generated"]}'
```

Failed deliveries are retried with exponential backoff (3 attempts by default). Delivery history is available via `GET /webhooks/deliveries`.

### Reports (Spec 08-08)

Report templates are managed via API. Schedules use the existing job engine:

```bash
# List available report templates
curl http://localhost:8000/reports/templates

# Schedule a recurring report
curl -X POST http://localhost:8000/reports/schedule \
  -H "Content-Type: application/json" \
  -d '{"template_id": "portfolio_summary", "schedule": "0 8 * * 1", "recipients": ["team@example.com"]}'
```

Templates define the data queries, layout, and output format (PDF/CSV/Excel).

### Rate Limiting (Spec 08-09)

Configure rate limits in `config/api_governance_config.yaml`:

```yaml
rate_limiting:
  enabled: true
  default_limit: "100/minute"
  burst_limit: "20/second"
  per_endpoint:
    /ai-planner/portfolio-scan: "5/hour"
    /chat: "30/minute"
  per_user: true          # Apply limits per authenticated user (vs global)
  backend: "memory"       # "memory" or "redis" (for multi-worker deployments)
```

When rate limited, the API returns HTTP 429 with `Retry-After` header. Limits are applied per authenticated user when `per_user: true`; otherwise globally.

---

## Validation

```bash
make check-db    # Table row counts in Postgres
make check-api   # curl API health + sample endpoints
make check-all   # DB + API (full)
make test-all    # Full test suite (backend + frontend)

# E2E smoke tests (requires API + UI running)
make e2e-install # Install Playwright browsers (one-time)
make e2e         # Run all E2E smoke tests (headless)
```

---

## Testing

```bash
make test          # All backend tests (~0.7s, no infra needed — DB is mocked)
make test-unit     # Unit tests only (common/ modules)
make test-api      # API endpoint tests only
make test-cov      # Backend tests with coverage report
make ui-test       # All frontend tests (Vitest + RTL, ~1.5s)
make test-all      # Backend + frontend (1636+ backend tests, 457+ frontend tests, <3s)
```

**Test structure:**
```
tests/
├── conftest.py              # Shared fixtures (sample DataFrames)
├── unit/
│   ├── test_metrics.py      # WAPE, bias, accuracy %
│   ├── test_constants.py    # LAG_RANGE, ROLLING_WINDOWS, thresholds
│   ├── test_domain_specs.py # All 8 domains (parametrized)
│   ├── test_backtest_framework.py
│   ├── test_mlflow_utils.py
│   ├── test_db.py
│   └── test_load_dataset_postgres.py
└── api/
    ├── conftest.py          # Mock pool + async httpx client
    ├── test_health.py
    ├── test_domains.py
    ├── test_forecast_accuracy.py
    ├── test_dfu_analysis.py
    ├── test_competition.py
    ├── test_clusters.py
    ├── test_inventory.py
    ├── test_distinct.py
    ├── test_dashboard.py
    ├── test_jobs.py         # 16 tests: types, submit, list, cancel, delete, stats, schedules, pipeline
    ├── test_inv_planning_eoq.py   # 10 tests: EOQ summary, detail, sensitivity endpoints
    ├── test_inv_planning_policy.py  # 13 tests: policy CRUD, assign, compliance endpoints
    ├── test_inv_planning_health.py  # 12 tests: health summary, detail, heatmap endpoints
    ├── test_inv_planning_exceptions.py  # 13 tests: exception list, summary, ack, status, generate
    ├── test_fill_rate.py            # fill rate summary, trend, detail endpoints
    ├── test_inv_planning_demand_signals.py  # demand signals endpoints
    ├── test_inv_planning_simulation.py      # simulation run, results, compare, status
    ├── test_inv_planning_abc_xyz.py         # ABC-XYZ matrix, summary, detail
    ├── test_inv_planning_supplier.py        # supplier performance endpoints
    ├── test_inv_planning_investment.py      # investment plan endpoints
    ├── test_inv_planning_intramonth.py      # intramonth stockout endpoints
    ├── test_control_tower.py               # control tower kpis, alerts, top-critical, trend
    ├── test_ai_planner.py                  # 18 tests: tool functions, agent loop, dry-run mode
    └── test_ai_planner_api.py              # 10 tests: insights CRUD, portfolio scan 202, memo list

frontend/src/
├── hooks/__tests__/         # useTheme, useUrlState, useKeyboardShortcuts, useSidebar, useGlobalFilters
├── lib/__tests__/           # formatters, export
├── api/__tests__/           # TanStack Query keys
├── context/__tests__/       # JobNotificationContext, ScenarioNotificationContext
├── components/__tests__/    # Skeleton, KeyboardShortcutHelp, EChartContainer, AppSidebar, ...
└── tabs/__tests__/          # All tab components (smoke tests)
```

Run `make test-all` after every code change. Every new feature must include tests — see `docs/design-specs/feature31.md`.

### E2E Testing (Playwright)

E2E tests run against the full stack — both API (`make api`) and UI (`make ui`) must be running.

**One-time setup:**
```bash
make e2e-install       # Install Playwright browsers (Chromium)
```

**Run E2E smoke tests:**
```bash
make e2e               # Headless (CI-friendly)
make e2e-headed        # Headed browser (local debugging)
make e2e-ui            # Playwright UI mode (interactive step-through)
make e2e-report        # Open HTML report from last run
```

> **Prerequisite:** Start API and UI in separate terminals before running E2E tests: `make api` (terminal 1), `make ui` (terminal 2).

**Test coverage:** 8 E2E test files — navigation, dashboard, accuracy, global-filters, inv-planning, ai-planner, control-tower, theme.

**Config:** `frontend/e2e/playwright.config.ts`. Shared fixtures: `frontend/e2e/fixtures/base.ts`.

---

## Data Cleanup

### Remove Backtest Model Predictions

```bash
make backtest-list                                 # Row counts per model_id
make backtest-clean MODELS="--dry-run lgbm_cluster" # Preview before deleting
make backtest-clean MODELS="lgbm_cluster catboost_cluster"  # Delete specific models
make backtest-clean MODELS="--all-backtest"        # Delete all non-external backtest models
```

Removes rows from `fact_external_forecast_monthly` and `backtest_lag_archive`, then refreshes all 5 dependent materialized views. `--all-backtest` never deletes `model_id='external'`.

### Remove Forecasts by Date Range

```bash
make forecast-clean-list                                               # Row counts by model + month

make forecast-clean ARGS="--before 2025-04-01 --dry-run"              # Preview
make forecast-clean ARGS="--before 2025-04-01 --model external"        # Delete external before Apr 2025
make forecast-clean ARGS="--between 2024-01-01 2024-07-01"             # Delete all models Jan–Jun 2024
make forecast-clean ARGS="--months 2024-03 2024-06 2024-09"            # Delete specific months
make forecast-clean ARGS="--months 2025-01 --model external"           # One month, one model
make forecast-clean ARGS="--after 2025-06-01 --forecast-only"          # Main table only
make forecast-clean ARGS="--before 2025-01-01 --date-column fcstdate --archive-only"  # Archive only
```

Accepted date formats: `YYYY-MM-DD`, `YYYY-MM`, `MM/DD/YYYY` (all normalized to month-start).

After any cleanup that affects champion/ceiling rows, re-run `make champion-select`.

---

## Troubleshooting

### Environment / Setup

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` in backend tests | `make init` — installs pytest, httpx, pytest-asyncio |
| Frontend tests fail to start | `make ui-init` — installs vitest, @testing-library/react |
| API tests fail with DB errors | Verify `tests/api/conftest.py` has mock pool fixtures |
| `localStorage.clear is not a function` | jsdom localStorage mock required — see `frontend/src/hooks/__tests__/useTheme.test.ts` |

### Infrastructure

| Problem | Fix |
|---|---|
| API returns DB errors | Verify `.env` DB values and `make up` status |
| pgvector extension not found | Ensure `docker-compose.yml` uses `pgvector/pgvector:pg16`; rebuild: `make down && docker volume rm demand_pg_data && make up` |
| MLflow connection refused (port 5003) | Run `make up`; verify: `docker ps \| grep mlflow`; MLflow only runs when stack is up |

### Data Ingestion

| Problem | Fix |
|---|---|
| Forecast load fails with "missing data for column model_id" | `make normalize-forecast && make load-forecast` |
| Inventory normalize fails | Verify 14 CSVs in `datafiles/`: `ls datafiles/Inventory_Snapshot_*.csv \| wc -l` (expect 14); files must be UTF-8 CSV with columns `exec_date,item,loc,lead_time,tot_oh,tot_oh_oo,mtd_sls` |
| Inventory load fails | Apply DDL first: `make db-apply-inventory`; verify `ls -lh data/inventory_clean.csv` |
| Inventory tab shows no data | Verify load: `docker exec demand-mvp-postgres psql -U demand -d demand_mvp -c "SELECT COUNT(*) FROM fact_inventory_snapshot"`; refresh view: `make refresh-agg-inventory` |

### ML Pipelines

| Problem | Fix |
|---|---|
| Clustering fails | Load sales first: `make load-sales`; verify MLflow: `docker ps \| grep mlflow`; check: `ls -lh data/clustering_features.csv` |
| Cluster assignments not updating | Preview with `--dry-run`; verify DFU key format matches database; check Postgres connection |
| Backtest fails | Run clustering first: `make cluster-all`; load sales: `make load-sales`; install deps: `uv sync` |
| Champion selection finds no DFUs | Load backtest predictions: `make backtest-load`; lower `min_dfu_rows` in `config/model_competition.yaml`; verify models exist: `SELECT DISTINCT model_id FROM fact_external_forecast_monthly` |
| Chat endpoint errors | Set `OPENAI_API_KEY` in `.env`; run `make generate-embeddings`; check API logs for rate limit errors |
| AI Planner errors | Set `ANTHROPIC_API_KEY` in `.env`; verify insight schema exists: `make ai-insights-schema`; check API logs for rate limit or tool dispatch errors |

### Exception Queue

| Problem | Fix |
|---|---|
| Exception queue is empty | Safety stock targets must exist first. Run `make ss-all` to compute safety stock targets, then `make exceptions-generate` to detect exceptions. Dependency chain: safety stock targets -> exception generation |
| `make exceptions-generate` finds no exceptions | Verify `fact_safety_stock_targets` has rows: `SELECT COUNT(*) FROM fact_safety_stock_targets`; check thresholds in `config/exception_config.yaml` |

### Inventory Rebalancing

| Problem | Fix |
|---|---|
| `mv_network_balance` refresh fails | Ensure dependent tables are populated first: `fact_rebalancing_plan`, `dim_transfer_lane`; run `make rebalancing-refresh` after data is loaded |
| Rebalancing compute finds no candidates | Verify safety stock targets exist (`SELECT COUNT(*) FROM fact_safety_stock_targets`); verify inventory snapshots are loaded; check `config/rebalancing_config.yaml` thresholds are not too restrictive |
| Dry-run shows transfers but compute writes nothing | Confirm you are running `make rebalancing-compute` (not `make rebalancing-compute-dry`); check DB connectivity and table permissions |

### Platform Services (Specs 08-01 through 08-10)

| Problem | Fix |
|---|---|
| JWT auth returns 401 on all requests | Verify `JWT_SECRET` is set in `.env`; check token expiry; re-authenticate to get a fresh token |
| `bcrypt` or `PyJWT` import error | Run `make init` or `uv sync` to install new dependencies |
| Data quality checks return empty results | Run `make db-apply-sql` to ensure DDL 062-070 are applied; verify check catalog has entries via `GET /data-quality/catalog` |
| Cache not working (stale data) | Check `CACHE_BACKEND` env var; for Redis, verify connectivity: `redis-cli ping`; InMemory cache clears on restart |
| Notifications not sending | Verify channel is `enabled: true` in `config/notification_config.yaml`; check SMTP credentials / Slack webhook URL |
| Webhook deliveries failing | Check delivery history via `GET /webhooks/deliveries`; verify target URL is reachable; check HMAC signature validation on receiver |
| Rate limiting too aggressive | Adjust limits in `config/api_governance_config.yaml`; increase `default_limit` or add per-endpoint overrides |
| Anonymous admin mode in production | Set `JWT_SECRET` in `.env` — without it, all requests bypass auth |

### Frontend / API

| Problem | Fix |
|---|---|
| Jobs tab returns HTML instead of JSON | Add `/jobs` proxy to `frontend/vite.config.ts`; restart `make ui`; verify backend: `curl http://localhost:8000/jobs/stats` |
| New API route returns HTML in UI | Add the path prefix to Vite proxy config in `frontend/vite.config.ts`; restart `make ui` |

---

## Key Dependencies Between Phases

```
Phase 1  (Schema)
    └─► Phase 1b (Auth/RBAC)  [auto-init on API start]
Phase 2  (Ingest)
    └─► Phase 2b (Data Quality Checks)
    └─► Phase 3  (Inv Planning)
    └─► Phase 4  (Clustering + Seasonality)  ──► Phase 5 (Backtesting)
                                                     └─► Phase 6 (Champion Select)
                                                             └─► Phase 7 (Prod Forecast)
    └─► Phase 8  (AI Insights)  [needs 3 + 6 + 7]
    └─► Phase 9  (Storyboard)   [needs 3]
    └─► Phase 9b (DQ Pipeline — auto)  [needs 2, runs every 4h via APScheduler]
    └─► Phase 9c (FVA Tracking)       [needs user interventions + actuals]
    └─► Phase 9d (S&OP Cycle)         [needs 3 + 7, multi-stage approval]
    └─► Phase 9e (Notifications + Webhooks)  [config-only, no data deps]
    └─► Phase 9f (Report Generation)  [needs loaded data for report queries]
    └─► Phase 9g (Medallion Pipeline) [needs 2, layered ETL with DQ gates]
Phase 10 (Start Services)
    └─► Cache layer active (Redis or in-memory fallback)
    └─► Query performance tracking active (fact_query_performance)
    └─► Auth/RBAC middleware enforced on mutation endpoints
```
