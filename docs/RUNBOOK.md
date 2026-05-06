# Supply Chain Command Center — Operations Guide

All commands run from the project root (`DemandProject/`) unless noted.

This document is the single operations reference: environment setup, data pipeline,
ML training, platform services, testing, cleanup, and troubleshooting. Follow the
phases in order on first setup; for subsequent runs only re-run the phases that have
new data.

---

## Table of Contents

- [Phase 0: One-Time Environment Setup](#phase-0-one-time-environment-setup)
- [Phase 1: Schema Setup](#phase-1-schema-setup-one-time-per-environment)
- [Phase 1b: Auth & RBAC Setup](#phase-1b-auth--rbac-setup)
- [Phase 2: Data Ingestion](#phase-2-data-ingestion)
- [Phase 2b: Data Quality Checks](#phase-2b-data-quality-checks)
- [Phase 3: Inventory Planning Computations](#phase-3-inventory-planning-computations)
- [Phase 4: ML — Clustering & Seasonality](#phase-4-ml--clustering--seasonality)
- [Phase 5: ML — Backtesting](#phase-5-ml--backtesting)
- [Phase 5b: Unified Model Tuning Studio](#phase-5b-unified-model-tuning-studio-feature-46)
- [Phase 6: Champion Model Selection](#phase-6-champion-model-selection)
- [Phase 6b: Expert Panel Algorithm Selection](#phase-6b-expert-panel-algorithm-selection-feature-49)
- [Phase 7: Production Forecast Generation](#phase-7-production-forecast-generation-f11)
- [Phase 7b: Forward-Looking Replenishment Plan](#phase-7b-forward-looking-replenishment-plan-ci-bands--repl-plan)
- [Phase 8-9: AI, Storyboard, DQ, FVA, S&OP, Notifications, Reports](#phase-8-9-ai-storyboard-dq-fva-sop-notifications-reports)
- [Phase 10: Start Services](#phase-10-start-services)
- [Full First-Time Run (New Environment)](#full-first-time-run-new-environment)
- [Incremental Refresh (New Data Arrives)](#incremental-refresh-new-data-arrives)
- [Platform Services Configuration](#platform-services-configuration)
- [Validation](#validation)
- [Testing](#testing)
- [Database Cleanup & Fresh Recreate](#database-cleanup--fresh-recreate)
- [Data Cleanup](#data-cleanup)
- [Troubleshooting](#troubleshooting)
- [Key Dependencies Between Phases](#key-dependencies-between-phases)

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
make abc-xyz-schema            # XYZ classification columns on dim_sku
make supplier-perf-schema      # mv_supplier_performance
make investment-schema         # fact_inventory_investment_plan + fact_efficient_frontier
make intramonth-schema         # mv_intramonth_stockout
make control-tower-schema      # mv_control_tower_kpis

# Inventory Rebalancing
make rebalancing-schema        # mv_network_balance + fact_rebalancing_recommendations

# AI Planning Agent
make ai-insights-schema        # ai_insights + ai_planning_memos + ai_call_log + ai_recommendation_outcomes

# Production Forecast (F1.1)
make forecast-prod-schema      # fact_production_forecast (source_model_id included in base DDL)
                               # + fact_candidate_forecast (staging) + model_promotion_log (audit trail)
                               # DDL: sql/121_candidate_forecast_and_promotion.sql

# Forward-Looking Replenishment Plan (CI Bands + Repl. Plan)
make replplan-schema           # fact_replenishment_plan

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

```

> **Tip:** `make db-apply-sql` covers the majority of tables (including DDL 062-070 for auth, data quality, cache, notifications, webhooks, reports, rate limiting). The remaining `make *-schema` commands add feature-specific tables on top.

---

## Phase 1b: Auth & RBAC Setup

Run after schema setup. Seeds default admin user and configures JWT-based authentication.

```bash
# Auth config lives in config/platform/auth_config.yaml (JWT secret, token TTL, role hierarchy)
# common/auth.py provides: CurrentUser, get_current_user, require_role dependencies
# api/routers/platform/auth_router.py provides: POST /auth/login, POST /auth/refresh
# api/routers/platform/users.py provides: CRUD for dim_user (admin-only)

# No Make target needed — auth is auto-initialized when API starts.
# All mutation endpoints use require_role() for RBAC enforcement.
# Audit log entries written to fact_audit_log on every state-changing request.
```

---

## Phase 2: Data Ingestion

### 2.1 All Datasets (Recommended Starting Point)

```bash
make normalize-all     # Normalize all 10 datasets (CSV → clean CSV)
make load-all          # Load all datasets into Postgres + refresh materialized views
```

### 2.2 Individual Datasets

```bash
make normalize-item && make load-item
make normalize-location && make load-location
make normalize-customer && make load-customer
make normalize-time && make load-time       # Auto-generates 2020–2035 time dimension
make normalize-sku && make load-sku
make normalize-sales && make load-sales     # TYPE=1 rows only
make normalize-forecast && make load-forecast
make normalize-sourcing && make load-sourcing
make normalize-purchase-order && make load-purchase-order
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

## Phase 2b: Data Quality Checks

Run after Phase 2 ingestion to validate loaded data. Can also be scheduled as a recurring pipeline step.

```bash
# Data quality checks are config-driven (config/platform/data_quality_config.yaml)
# common/engines/dq_engine.py runs SQL-based checks: freshness, completeness, uniqueness,
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
make variability-compute     # CV, dispersion, volatility profiles → dim_sku

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

**Tables:**
- `dim_transfer_lane` — valid transfer lanes between locations (source -> destination, lead time, cost)
- `fact_rebalancing_plan` — computed rebalancing recommendations (item, source/dest, qty, priority)
- `fact_rebalancing_transfer` — executed/planned transfer records with status tracking
- `mv_network_balance` — materialized view aggregating network-wide inventory balance metrics

**SQL files:** `sql/071_create_transfer_network.sql` (dim_transfer_lane), `sql/072_create_rebalancing_plan.sql` (fact_rebalancing_plan + fact_rebalancing_transfer), `sql/073_create_rebalancing_views.sql` (mv_network_balance)

**Config:** `config/inventory/rebalancing_config.yaml` — transfer cost thresholds, minimum transfer qty, priority scoring weights, network constraints.

---

## Phase 4: ML — Clustering & Seasonality

Run after Phase 2. These enrich `dim_sku` with segment labels used as ML features.

```bash
# SKU clustering (groups SKUs by demand pattern)
make cluster-all             # features → train → label → update dim_sku

# Seasonality profiles
make seasonality-all         # detect → update dim_sku
```

**Clustering individual steps:**

```bash
make cluster-all  # Unified pipeline: features → train → label → auto-promote (creates cluster_experiment row)
```

Cluster assignments are filterable in the Data Explorer via the `cluster_assignment` column and viewable via `/domains/sku/clusters`.

---

## Phase 5: ML — Backtesting

Run after Phase 4 (clustering needed for cluster-based models).
Each backtest trains models AND persists `.pkl` artifacts to `data/models/<model_id>/` for production forecasting.

### Configuration (Feature 44)

Before running any backtest, edit the algorithm entry in `config/forecasting/forecast_pipeline_config.yaml` under `algorithms.<model_id>`:

```yaml
algorithms:
  lgbm_cluster:
    type: tree
    cluster_strategy: per_cluster  # "per_cluster" or "global" — ml_cluster used for partitioning only, not a model feature
    recursive: false       # Set true for recursive multi-step inference (Feature 43)
    shap_select: false     # Set true for per-timeframe SHAP feature selection (Feature 42)
    shap_threshold: 0.95   # Cumulative SHAP importance threshold
    shap_top_n: null       # Exact top-N features (overrides threshold)
    shap_sample_size: 500
    tune_inline: false     # Set true for per-timeframe causal tuning (PL-002)
    params_file: null      # Set to data/tuning/best_params_lgbm.json for pre-tuned params
    params:                # Inline hyperparameters (replaces old algorithm_config.yaml)
      n_estimators: 500
      learning_rate: 0.05
      # ... (see full file for all keys)
```

Same structure for `catboost_cluster:` and `xgboost_cluster:` entries.

**Architecture (Feature 44):** Tree-based models (LGBM, CatBoost, XGBoost) share `common/ml/backtest_framework.py` via `run_tree_backtest()`. Each script provides both `train_and_predict_per_cluster()` and `train_and_predict_global()`, selecting based on the `cluster_strategy` config key (`per_cluster` or `global`). **`ml_cluster` is excluded from model features** (listed in `METADATA_COLS`) — it is merged into the grid as a metadata column for per-cluster partitioning only, never passed to models as an input feature. Algorithm options (cluster_strategy, recursive, shap_select, tune_inline, params_file, hyperparameters) are read from `config/forecasting/forecast_pipeline_config.yaml` under `algorithms.<model_id>`. Use `get_algorithm_params(model_id)` from `common/core/utils.py` to retrieve hyperparameters. Shared modules: `feature_engineering.py`, `metrics.py`, `mlflow_utils.py`, `db.py`, `constants.py`, `tuning.py`, `shap_selector.py`.

### GPU Acceleration

GPU acceleration is available for backtesting and simulation pipelines via optional dependencies.

**Backtesting (LGBM):** Controlled by the `DEMAND_GPU` environment variable:

| Value | Behavior |
|---|---|
| `auto` (default) | Auto-detect GPU availability at runtime |
| `on` | Force GPU usage (fails if unavailable) |
| `off` | Disable GPU, use CPU only |

```bash
DEMAND_GPU=on make backtest-lgbm    # Force GPU
DEMAND_GPU=off make backtest-lgbm   # Force CPU
```

CatBoost and XGBoost also auto-detect GPU at runtime (`task_type="GPU"` and `device="cuda"` respectively).

**Monte Carlo Simulation:** Uses CuPy for GPU-accelerated array operations when available, falls back to NumPy.

**Seasonality Detection:** Uses Numba JIT compilation for numerical kernels when available, falls back to pure NumPy.

**Optional dependencies** (not required -- all scripts fall back gracefully):
- `cupy` -- GPU array library for Monte Carlo simulation
- `numba` -- JIT compilation for seasonality detection kernels

### Optional: Hyperparameter Tuning (Before Backtests)

Three modes — choose based on your goal:

| Mode | When to use | How to activate | Output |
|---|---|---|---|
| **Global** — tune once on all data, one param set | Quick baseline tuning | `make tune-lgbm` | `data/tuning/best_params_lgbm.json` |
| **Per-cluster** — tune independently per `ml_cluster` | Best accuracy (recommended) | `make tune-lgbm-clusters` | `config/forecasting/cluster_tuning_profiles.yaml` |
| **Inline** — per-timeframe tuning inside each backtest fold | Unbiased backtest evaluation | Set `tune_inline: true` in config, then `make backtest-lgbm` | Params applied per timeframe |

**Recommended workflow:**
1. `make tune-lgbm` → sets good global base params in `forecast_pipeline_config.yaml`
2. `make tune-lgbm-clusters` → tunes per cluster, writes `cluster_tuning_profiles.yaml`
3. `make backtest-lgbm` → uses base params + per-cluster overrides

> **Warning:** Using a globally tuned `params_file` in backtests introduces temporal leakage — the tuner sees future timeframes. Use `tune_inline: true` for unbiased backtest accuracy.

**Global tuning (Mode A):**
```bash
make tune-lgbm      # ~20–40 min  → data/tuning/best_params_lgbm.json
make tune-catboost  # ~30–60 min  → data/tuning/best_params_catboost.json
make tune-xgboost   # ~25–50 min  → data/tuning/best_params_xgboost.json
make tune-all       # All three sequentially
```

Each JSON file contains `best_params`, `best_n_estimators`, per-cluster WAPEs, and CV fold metrics. Apply in backtest by setting `params_file: data/tuning/best_params_<model>.json` in `config/forecasting/forecast_pipeline_config.yaml` under `algorithms.<model_id>` (Feature 44).

**Per-cluster tuning (Mode B — recommended):**
```bash
make tune-lgbm-clusters      # ~45–60 min  → config/forecasting/cluster_tuning_profiles.yaml
make tune-catboost-clusters   # ~60–90 min
make tune-xgboost-clusters    # ~45–60 min
make tune-clusters            # All three sequentially
```

Tunes Optuna Bayesian optimization (30 trials) independently for each `ml_cluster` from `dim_sku`. Writes per-cluster params to `config/forecasting/cluster_tuning_profiles.yaml` with `cluster_name`-based matching. During backtest, each cluster first checks for an exact name match in profiles; if found, those tuned overrides are applied on top of base params. If not found, base params are used and the log shows "using global params (no profile match)".

Options: `--trials 30` (default), `--clusters L2_1 L2_3` (tune specific clusters), `--min-rows 500` (skip clusters with fewer rows).

**Disabling per-cluster profiles (use global params only):**

Set `enabled: false` in `config/forecasting/cluster_tuning_profiles.yaml`. All clusters will use the base params from `forecast_pipeline_config.yaml`. The profiles are ignored but preserved — set back to `true` to re-enable.

**End-to-end LGBM workflow after new clustering:**
```bash
# 1. Run clustering (updates dim_sku.ml_cluster with new labels)
make cluster-all

# 2. (Optional) Tune global base params
make tune-lgbm                       # → forecast_pipeline_config.yaml

# 3. (Optional) Tune per-cluster params — requires step 1 first
make tune-lgbm-clusters              # → cluster_tuning_profiles.yaml

# 4. Run backtest with tuned params
make backtest-lgbm                   # Uses base + per-cluster overrides

# 5. Load predictions into Postgres
make backtest-load-bulk MODELS=lgbm_cluster

# 6. Run champion selection
make champion-all                    # Selects best model per DFU
```

Skip step 3 to use only global params. Skip steps 2-3 to use the existing params as-is.

**Inline per-timeframe tuning (Mode C, PL-002 fix):**

Set `tune_inline: true` in `config/forecasting/forecast_pipeline_config.yaml` for the relevant algorithm entry, then run `make backtest-lgbm` (or catboost/xgboost). Each of the 10 timeframes (~20 trials x 3 folds, ~2-3x slower) tunes on only data available up to its training cutoff — no future leakage into backtest accuracy metrics.

### Run Backtests

#### LGBM (per-cluster)

```bash
# Edit forecast_pipeline_config.yaml algorithms.lgbm_cluster section, then:
make backtest-lgbm
make backtest-load MODEL=lgbm_cluster
```

#### CatBoost (per-cluster)

```bash
# Edit forecast_pipeline_config.yaml algorithms.catboost_cluster section, then:
make backtest-catboost
make backtest-load MODEL=catboost_cluster
```

#### XGBoost (per-cluster)

```bash
# Edit forecast_pipeline_config.yaml algorithms.xgboost_cluster section, then:
make backtest-xgboost
make backtest-load MODEL=xgboost_cluster
```

#### Run All Tree Models

```bash
# Sequential (safe default)
make backtest-lgbm && make backtest-catboost && make backtest-xgboost
make backtest-load-all
```

#### Foundation Models (Chronos)

```bash
# Chronos T5 (46M params, ~2.5h) — original zero-shot model
make backtest-chronos
make backtest-load-chronos

# Chronos Bolt (205M params, ~12min) — fastest, comparable to T5-large accuracy
make backtest-bolt
make backtest-load-bolt

# Chronos 2 (821M params, ~5.5h) — latest generation, zero-shot
make backtest-chronos2
make backtest-load-chronos2

# Chronos 2 Enriched (821M + 31 covariates, ~6h) — best accuracy potential
make backtest-chronos2e
make backtest-load-chronos2e

# Resume any crashed run (uses checkpoints — skips completed timeframes):
uv run python -m scripts.ml.run_backtest_chronos_bolt --resume
uv run python -m scripts.ml.run_backtest_chronos2_enriched --resume
```

See `docs/specs/02-forecasting/18-chronos-foundation-models.md` for full architecture comparison.

#### Statistical Baselines

```bash
# Seasonal Naive — repeats last year's pattern
make backtest-seasonal-naive
make backtest-load-seasonal-naive

# Rolling Mean — simple rolling average
make backtest-rolling-mean
make backtest-load-rolling-mean

# MSTL (Multiple Seasonal-Trend decomposition using LOESS)
make backtest-mstl
make backtest-load-mstl

# Run both baselines together:
make backtest-baselines
```

#### Deep Learning Models

```bash
# N-HiTS (Neural Hierarchical Interpolation for Time Series)
make backtest-nhits
make backtest-load-nhits

# N-BEATS (Neural Basis Expansion Analysis)
make backtest-nbeats
make backtest-load-nbeats
```

#### Run All (Tree + Foundation + Statistical + Deep Learning)

```bash
# Sequential (safe default)
make backtest-all
make backtest-load-all

# Parallel (faster on servers with 16+ cores / 32GB+ RAM)
# Each process logs to data/backtest/logs/<model>.log — no interleaved output
make backtest-all-parallel
make backtest-load-all
```

> **Note:** `backtest-all-parallel` fires all processes simultaneously. Tree models use `n_jobs=-1` (all CPU cores); foundation models use GPU. Running tree + foundation in parallel is generally safe (CPU vs GPU). Avoid running multiple foundation models simultaneously — they compete for GPU memory.

### Loading Predictions

```bash
make backtest-load MODEL=<model_id>   # Load one model
make backtest-load-all                # Scan data/backtest/*/ and load all models
make backtest-load-all-bulk           # Load all models with single index cycle (~4x faster)
```

`--replace` is built into `backtest-load` — it only deletes rows for the loaded `model_id`, leaving all other models untouched. Accuracy materialized views are refreshed automatically after every load.

Each backtest run writes to a **model-scoped subdirectory** so multiple models can run back-to-back without overwriting each other:
- `data/backtest/<model_id>/backtest_predictions.csv` — execution-lag predictions (-> `fact_external_forecast_monthly`)
- `data/backtest/<model_id>/backtest_predictions_all_lags.csv` — lag 0-4 archive (-> `backtest_lag_archive`)

#### Advanced Loading Flags

The `load_backtest_forecasts.py` script supports several flags for efficient multi-model loading:

| Flag | Description |
|---|---|
| `--model M` | Load a single model by ID |
| `--models M1 M2 ...` | Load specific models (space-separated list) |
| `--all` | Auto-discover and load all models from `data/backtest/*/` |
| `--replace` | Delete existing rows for each loaded `model_id` before inserting |
| `--bulk` | With `--replace`: drop/recreate indexes ONCE across all models instead of per-model (~4x faster for multi-model loads) |
| `--main-only` | Load only `fact_external_forecast_monthly` (skip archive table) |
| `--archive-only` | Load only `backtest_lag_archive` (skip main table) |

#### Multi-Model Loading Examples

```bash
# Load 4 models with single index cycle (fastest):
uv run python scripts/etl/load_backtest_forecasts.py \
  --models lgbm_cluster catboost_cluster xgboost_cluster chronos \
  --replace --bulk

# Load main table only (skip archive):
uv run python scripts/etl/load_backtest_forecasts.py \
  --models lgbm_cluster chronos --replace --bulk --main-only

# Load archive only:
uv run python scripts/etl/load_backtest_forecasts.py \
  --models lgbm_cluster chronos --replace --bulk --archive-only

# Makefile convenience targets:
make backtest-load-bulk                              # All 4 core models, bulk mode
make backtest-load-main-only MODELS="lgbm_cluster chronos"   # Main table only
make backtest-load-archive-only MODELS="lgbm_cluster chronos" # Archive only
```

> **Tip:** Use `--bulk` whenever loading 2+ models with `--replace`. It drops indexes before the first model and recreates them after the last, avoiding redundant index rebuilds. For a single model, `--bulk` has no benefit.

**Available model IDs (Feature 44):**

| Framework | Per-Cluster (default) | Global |
|---|---|---|
| LGBM | `lgbm_cluster` | `lgbm_global` |
| CatBoost | `catboost_cluster` | `catboost_global` |
| XGBoost | `xgboost_cluster` | `xgboost_global` |
| Chronos T5 | `chronos` | — |
| Chronos Bolt | `chronos_bolt` | — |
| Chronos 2 | `chronos2` | — |
| Chronos 2 Enriched | `chronos2_enriched` | — |
| Seasonal Naive | `seasonal_naive` | — |
| Rolling Mean | `rolling_mean` | — |
| MSTL | `mstl` | — |
| N-HiTS | `nhits` | — |
| N-BEATS | `nbeats` | — |

Verify archive data:
```bash
docker compose exec -T postgres psql -U demand -d demand_mvp \
  -c "SELECT model_id, lag, COUNT(*) FROM backtest_lag_archive GROUP BY 1,2 ORDER BY 1,2"
```

---

## Phase 5b: Unified Model Tuning Studio (Feature 46)

Optional — use the UI-driven tuning studio to run experiments interactively instead of CLI-based tuning.

The Unified Model Tuning Studio lets users configure hyperparameters, launch backtest experiments, compare results, and promote winners — all from the browser for LightGBM, CatBoost, and XGBoost.

### Accessing the Studio

Navigate to **Model Tuning** in the sidebar (Demand section). Select a model type tab (LGBM, CatBoost, XGBoost) at the top.

### Workflow

1. **Create Experiment** — Click "New Experiment", select a template (production baseline, expert recommendation, or custom), adjust parameters, and submit
2. **Monitor** — Watch the experiment in the Jobs tab with live log streaming
3. **Compare** — Select two completed runs for side-by-side comparison with per-lag, per-cluster, per-month accuracy breakdowns, parameter diffs, and feature diffs
4. **Promote** — Promote the winning run to production via the confirmation modal (writes to `forecast_pipeline_config.yaml` under `algorithms.<model_id>.params`)

### API Endpoints

All endpoints are under `/model-tuning/{model}/` where `{model}` is `lgbm`, `catboost`, or `xgboost`.

```
GET  /model-tuning/{model}/experiments           # List experiments (paginated, filterable)
GET  /model-tuning/{model}/experiments/{id}       # Experiment detail with timeframes
GET  /model-tuning/{model}/experiments/{id}/lags  # Per-execution-lag accuracy breakdown
GET  /model-tuning/{model}/experiments/{id}/clusters  # Per-cluster accuracy (filterable by exec_lag)
GET  /model-tuning/{model}/experiments/{id}/months    # Per-month accuracy
GET  /model-tuning/{model}/experiments/{id}/logs      # Incremental log streaming (offset-based)
GET  /model-tuning/{model}/compare                    # Pairwise comparison with deltas
GET  /model-tuning/{model}/templates                  # Available experiment templates
GET  /model-tuning/{model}/promoted                   # Currently promoted (champion) run
GET  /model-tuning/{model}/promotions                 # Promotion audit trail
POST /model-tuning/{model}/experiments                # Create and launch experiment
POST /model-tuning/{model}/experiments/{id}/promote   # Promote to production
POST /model-tuning/{model}/experiments/{id}/cancel    # Cancel running/queued experiment
DELETE /model-tuning/{model}/experiments/{id}          # Delete completed/failed experiment
```

### Schema

Reuses existing `lgbm_tuning_run` table with `model_id` column (`lgbm_cluster`, `catboost_cluster`, `xgboost_cluster`) to discriminate model types. DDL: `sql/098_add_promoted_to_tuning.sql` adds `is_promoted` and `promoted_at` columns with a partial unique index per model.

```bash
# Apply schema (one-time)
make db-apply-sql    # Includes sql/098_add_promoted_to_tuning.sql
```

### Configuration

Experiment templates are loaded from `config/forecasting/forecast_pipeline_config.yaml` (each algorithm's `params` section) plus the unified strategy file `config/forecasting/tune_strategies.yaml` (model-keyed sections: `lgbm`, `catboost`, `xgboost`).

### Key Files

| File | Purpose |
|------|---------|
| `api/routers/forecasting/tuning/` | Unified router package (14 endpoints, split by concern) |
| `frontend/src/api/queries/unified-model-tuning.ts` | Frontend query module |
| `frontend/src/tabs/LgbmTuningTab.tsx` | Main UI component |
| `tests/api/test_unified_model_tuning.py` | Backend tests (40 tests) |

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
make champion-select          # Run with strategy from config/forecasting/forecast_pipeline_config.yaml champion section
make champion-train-meta      # Train meta-learner (required before using meta_learner strategy)
make champion-simulate        # Simulate all strategies, compare accuracy vs ceiling
make champion-all             # train-meta + simulate + select (full pipeline)

# Override strategy:
.venv/bin/python -m scripts.ml.run_champion_selection --strategy rolling
```

### Config (`config/forecasting/forecast_pipeline_config.yaml`, `champion` section)

> **Note:** All champion selection settings live in `config/forecasting/forecast_pipeline_config.yaml` under the `champion` section. The legacy `config/model_competition.yaml` has been deleted.
>
> The competing models list is derived from `algorithms[*].compete == true` in the master config rather than an explicit list.

```yaml
competition:
  metric: accuracy_pct
  lag: execution
  min_dfu_rows: 3
  models: [catboost_cluster, xgboost_cluster, chronos2, chronos2_enriched, chronos_bolt, seasonal_naive, rolling_mean]
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

## Phase 6b: Expert Panel Algorithm Selection (Feature 49)

Tests whether a mix of statistical + ML + deep learning algorithms outperforms the current tree-only approach. Runs 14 algorithms (Holt-Winters, Simple ES, Croston SBA, Auto-ARIMA, Theta, MSTL, LGBM, CatBoost, XGBoost, Random Forest-via-Ridge, Seasonal Naive, Rolling Mean, N-HiTS, N-BEATS) across demand archetypes classified by the Syntetos-Boylan ADI × CV² framework.

### Prerequisites

- Phase 5 completed (backtests loaded) — needed for baseline comparison
- Phase 6 completed (champion selection) — needed for champion baseline
- `statsmodels` installed (added to `pyproject.toml` dependencies)
- Optional: `pmdarima` for Auto-ARIMA (`uv pip install pmdarima`)

### Commands

```bash
# Full test: 5000 DFUs, 5 timeframes (~30 min)
make expert-panel

# Quick test: 1000 DFUs, 3 timeframes (~8 min)
make expert-panel-quick

# Minimal smoke test: 200 DFUs, 2 timeframes (~2 min)
make expert-panel-mini

# Location-scoped: all DFUs at one site (sequential, no sampling)
make expert-panel-loc LOC=1401-BULK

# Advanced panel (adds stat upgrades, DL, foundation models)
make adv-expert-panel
make adv-expert-panel-quick
make adv-expert-panel-mini
make adv-expert-panel-loc LOC=1401-BULK

# Custom parameters
uv run python -m algorithm_testing.run_expert_panel \
    --n-dfus 2000 --n-timeframes 4 --seed 123

# Location + fewer timeframes
uv run python -m algorithm_testing.run_expert_panel \
    --loc 1401-BULK --n-timeframes 3
```

### What It Does

1. **Builds a golden set** — stratified sample from `dim_sku` by cluster, or all DFUs at a specified location (`--loc`)
2. **Classifies demand** — Syntetos-Boylan: smooth, erratic, intermittent, lumpy × high/low volume (8 archetypes)
3. **Runs 12 algorithms** per timeframe — statistical models fit per-DFU (parallel or sequential depending on set size), tree models fit per-cluster
4. **Builds affinity matrix** — segment × algorithm accuracy heatmap
5. **Optimizes portfolio** — assigns best algorithm per segment (max 6 algorithms)
6. **Compares vs baselines** — Seasonal Naive, External Forecast, Current Tree Champion

### Output

All results written to `algorithm_testing/results/` (or `adv_algorithm_testing/results/`):

| File | Content |
|------|---------|
| `experiment_report.txt` | Human-readable summary with lift numbers |
| `comparison.json` | Portfolio vs baselines (the key result) |
| `affinity_matrix.csv` | Segment × algorithm accuracy heatmap |
| `assignments.csv` | Best algorithm per demand segment |
| `classification.csv` | Per-DFU archetype classification |
| `affinity_detail.csv` | Per-segment accuracy detail |
| `all_predictions.parquet` | All model predictions (large) |
| `metadata.json` | Runtime, DFU count, loc_filter, algorithm counts |

### Interpreting Results

The key metric is **lift in basis points (bps)** in `comparison.json`:

- `lift.vs_naive_bps` — improvement over seasonal naive (should be large, 500+ bps)
- `lift.vs_external_bps` — improvement over ERP forecast (target: positive)
- `lift.vs_champion_bps` — improvement over current tree champion (the money number)

A positive `vs_champion_bps` means the algorithm mix outperforms the tree-only approach. Each 100 bps ≈ 1% accuracy improvement ≈ ~1% safety stock reduction.

### Configuration

All parameters in `algorithm_testing/config.yaml` (or `adv_algorithm_testing/config.yaml`):

- `experiment.n_dfus` — golden set size (default: 5000; ignored when `loc_filter` is set)
- `experiment.n_timeframes` — backtest depth (default: 5, production uses 10)
- `experiment.loc_filter` — run on all DFUs at a specific location instead of sampling (e.g. `1401-BULK`); also settable via `--loc` CLI flag
- `experiment.n_workers` — parallel worker count (default: 8; sets auto auto-switch to sequential when DFU count ≤ 200)
- `statistical_models.*` — enable/disable each statistical method
- `portfolio_optimizer.max_algorithms` — complexity budget (default: 6)

### Execution Mode

Statistical models auto-select between two execution paths:

| DFU count | Mode | Reason |
|-----------|------|--------|
| ≤ 200 | **Sequential** | Process-pool startup overhead exceeds model runtime for small sets |
| > 200 | **Parallel** (`ProcessPoolExecutor`) | `enabled_models` and `predict_months` serialized once per worker via initializer; each task carries only compact numpy arrays |

When `--loc` is used, most sites produce < 200 DFUs → sequential mode is used automatically.

### Key Files

| File | Purpose |
|------|---------|
| `algorithm_testing/run_expert_panel.py` | Main orchestrator |
| `algorithm_testing/golden_set.py` | Golden set sampling + loc-based selection |
| `algorithm_testing/config.yaml` | Experiment configuration |
| `algorithm_testing/demand_classifier.py` | Syntetos-Boylan classification |
| `algorithm_testing/statistical_models.py` | HW, ES, Croston, ARIMA, Theta (DFU-first, all models per DFU) |
| `algorithm_testing/tree_models.py` | LGBM/CatBoost/XGBoost wrapper |
| `algorithm_testing/affinity_matrix.py` | Segment × algorithm matrix |
| `algorithm_testing/portfolio_optimizer.py` | Greedy + constrained optimizer |
| `algorithm_testing/comparison.py` | Portfolio vs baselines |
| `adv_algorithm_testing/run_adv_expert_panel.py` | Advanced panel orchestrator (+ DL, foundation models) |
| `docs/specs/02-forecasting/15-expert-panel-algorithm-selection.md` | Full design spec |

---

## Phase 7: Production Forecast Generation (F1.1)

Run after Phase 6. Generates future-period (T+1 to T+24) demand forecasts using champion ML model artifacts. Uses 36 months of lookback history. Cold-start routing sends DFUs with < 12 months history to rolling_mean; DFUs with < 3 months are skipped entirely.

**Staged promotion workflow:** Predictions are first written to `fact_candidate_forecast` (staging table), then promoted to `fact_production_forecast` after validation. The `model_promotion_log` table tracks all promotion events as an audit trail.

```
Train → Generate → Load (→ fact_candidate_forecast) → Promote (→ fact_production_forecast)
```

Config: `config/forecasting/forecast_pipeline_config.yaml` (production_forecast section).

```bash
make forecast-generate       # Generate 24-month forward forecasts → fact_candidate_forecast
# or for a single DFU:
make forecast-generate-sku ITEM=100320 LOC=1401-BULK
# preview without writing:
make forecast-generate-dry
```

**Promotion:** After candidate forecasts are generated, promote them to production:
- **Champion promotion:** Uses per-DFU champion assignments to select the best model per DFU from candidates, then copies those rows to `fact_production_forecast`.
- **Single model promotion:** Copies all candidate rows for a specified model to `fact_production_forecast`.

Each promotion event is logged in `model_promotion_log` with the promotion type, model(s), row counts, and timestamp.

**Dependency chain for `make forecast-generate`:**
1. `data/models/lgbm_cluster/cluster_*.pkl` must exist (from Phase 5)
2. Champion assignments with `source_model_id` must exist (from Phase 6)
3. Recent sales history must be loaded (from Phase 2)

**Cold-start routing:**
- DFUs with >= 12 months history: use champion model (normal path)
- DFUs with 3-11 months history: routed to `cold_start_model_id` (rolling_mean)
- DFUs with < 3 months history: skipped (absolute floor)

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

## Phase 8-9: AI, Storyboard, DQ, FVA, S&OP, Notifications, Reports

### AI Planning Agent (IPAIfeature1)

Run after Phases 2-7 to give the agent full context. Requires `ANTHROPIC_API_KEY` in `.env`.

```bash
make ai-insights-scan        # Portfolio-wide AI exception scan → ai_insights table
make ai-insights-sku ITEM=100320 LOC=1401-BULK  # Single DFU analysis
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
make dq-populate             # Populate check catalog from config/platform/data_quality_config.yaml
make dq-run                  # Run all enabled DQ checks → fact_dq_check_results
make dq-all                  # Full pipeline: schema + populate + run
```

**Automated schedule:** Every 4 hours via APScheduler (`dq_check` job type in job_registry).
The scheduler triggers `common/engines/dq_engine.py` which evaluates SQL-based rules (freshness,
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
make fva-schema              # fact_intervention_metrics (sql/068_create_fva_tracking.sql)
```

> FVA interventions are populated through user actions in the UI (override queue, manual adjustments). No batch seed step needed.

API endpoints (`/fva/*`):
- `GET /fva/waterfall` — staged ladder for `naive seasonal -> external -> champion`, plus planned `AI adjusted` and `planner adjusted` placeholders and a separate ceiling benchmark
- `GET /fva/interventions` — intervention history with before/after metrics
- `GET /fva/roi-summary` — ROI dashboard: intervention count and estimated vs. actual financial impact

Dashboard: FVA tab shows the Forecast Value Ladder, ceiling benchmark, ROI summary, and recent interventions. FVA config was removed (dead, no consumers); settings are inline in the router.

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
Config in `config/operations/sop_config.yaml`.

### Notification & Webhook Configuration (08-04, 08-10)

Configure notification channels and webhook subscriptions. These fire automatically
when pipeline events occur (DQ failures, AI insights, exception alerts, report delivery).

```bash
# Notification channels configured via config/platform/notification_config.yaml
# common/services/notification_engine.py dispatches to Slack, Teams, Email, PagerDuty
# Delivery history stored in fact_notification_log (sql/065)
```

API endpoints (`/notifications/*`):
- `GET /notifications/history` — past notification deliveries with status
- `POST /notifications/test` — send a test notification to verify channel config

```bash
# Webhook subscriptions registered via API (no Make target needed)
# common/services/webhook_dispatcher.py signs and dispatches HTTPS callbacks on events
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
# Report templates + schedules configured via reporting tables (reporting_config.yaml was removed as dead config)
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

### Unified Pipeline Orchestrator

Single-command data pipeline that handles normalize, load, and MV refresh for all 10 domains. Data loads directly from CSV into main tables via `scripts/etl/load_dataset_postgres.py` (single-pass: COPY to staging, type-cast + dedup, INSERT). Batch tracking via `audit_load_batch`. Two modes: **full reload** (wipe and reload everything) and **incremental refresh** (detect changes, reload only deltas).

```bash
# Full reload — all domains, parallel normalization
make pipeline-full

# Incremental refresh — only changed files reloaded
make pipeline-refresh

# Inventory only
make pipeline-inventory              # Full reload
make pipeline-inventory-refresh      # Incremental
```

**Full reload flow** (`--mode full`):
1. Runs `data/input/cleanup_input.py` (input cleanup)
2. Normalizes all requested domains (parallel for non-inventory, sequential for inventory)
3. Loads each domain directly into target tables (COPY to staging, type-cast + dedup, INSERT)
4. Refreshes all affected materialized views
5. Prints summary table (rows loaded, elapsed time per domain)

**Incremental refresh flow** (`--mode refresh`):
1. Compares SHA256 hashes of clean CSVs against `audit_load_batch.source_hash`
2. For inventory: per-file hash comparison of each `Inventory_Snapshot_YYYY_MM.csv`
3. Normalizes only changed domains
4. Loads changed domains (inventory uses targeted DELETE by month range instead of TRUNCATE)
5. Refreshes only MVs affected by the changed domains
6. Unchanged domains are skipped entirely

**CLI flags:**

| Flag | Description |
|---|---|
| `--mode full\|refresh` | Full wipe-and-reload vs incremental delta |
| `--domains item,sales` | Comma-separated subset (default: all 10) |
| `--parallel` | Normalize non-inventory domains in parallel |
| `--dry-run` | Preview what would be done without making changes |
| `--data-dir /path` | Override source directory (default: `data/input`) |

**Config:** `config/etl/etl_config.yaml` — domain load order, parallel workers, MV refresh mapping per domain, always-refresh list.

**Error handling:** If normalization fails for a domain, that domain is skipped during loading (logged as `(skipped)` in the summary table). Other domains continue normally.

**Examples:**

```bash
# Preview a full reload
~/.local/bin/uv run python scripts/etl/run_pipeline.py --mode full --parallel --dry-run

# Reload only sales and forecast
~/.local/bin/uv run python scripts/etl/run_pipeline.py --mode full --domains sales,forecast

# Incremental refresh after adding a new inventory snapshot
~/.local/bin/uv run python scripts/etl/run_pipeline.py --mode refresh --domains inventory

```

---

## Phase 10: Start Services

### API Middleware Stack

The FastAPI backend includes platform middleware applied in `api/main.py`:
- **GZip compression** — responses > 1KB are gzip-compressed
- **CORS** — allows `localhost:5173` (dev) origins
- **Cache layer** — `common/services/cache.py` provides `@cached` decorator with TTL-based caching.
  Two backends: Redis (when `REDIS_URL` env var set) or in-memory fallback.
  Config in `config/platform/cache_config.yaml`. Cache invalidation on write endpoints.
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

### Option A: Automated Setup Targets (Recommended)

Use the orchestrated `setup-*` targets that handle dependency ordering automatically:

```bash
# 0. Environment + schema
make init && make up && make ui-init
make db-apply-sql
make db-apply-inventory db-apply-inv-backtest db-apply-jobs

# 1. Full setup — data + ML + inventory + demand + ops (everything)
make setup-all

# 2. Start services
make api   # terminal 1
make ui    # terminal 2
```

**Available setup targets:**

| Target | What it does |
|---|---|
| `make setup-data` | Normalize + load all 10 domains into Postgres |
| `make setup-planning` | Data load + inventory planning (no ML — fastest path to a working UI) |
| `make setup-all` | Full pipeline: data + features + backtests + champion + inv planning + demand planning + ops |

**Intermediate targets** (called by `setup-all` in dependency order):

| Target | Phase |
|---|---|
| `make setup-features` | Clustering, seasonality, variability, lead time, ABC-XYZ, demand signals |
| `make setup-backtest` | All backtests + champion selection (depends on setup-features) |
| `make setup-inv-planning` | Safety stock, EOQ, policies, exceptions, health, rebalancing, control tower |
| `make setup-demand-planning` | Production forecasts, projections, orders, replenishment, consensus |
| `make setup-ops` | S&OP, events, financial plan, storyboard, DQ |

### Option B: Manual Step-by-Step

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
# 2. Ingest (Option A: unified pipeline — recommended)
make pipeline-full               # Normalize + load + refresh MVs (all 10 domains)

# 2. Ingest (Option B: manual)
# make normalize-all && make load-all
# make inventory-pipeline

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
# Option A: Unified pipeline orchestrator (recommended)
make pipeline-refresh            # Detects changed files, reloads only deltas, refreshes affected MVs

# Option B: Manual per-dataset reload
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

Configure notification channels in `config/platform/notification_config.yaml`:

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
make test-all      # Backend + frontend (2273+ backend tests, 741+ frontend tests, <3s)
```

**Test structure:**
```
tests/
├── conftest.py              # Shared fixtures (sample DataFrames)
├── unit/
│   ├── test_metrics.py      # WAPE, bias, accuracy %
│   ├── test_constants.py    # LAG_RANGE, ROLLING_WINDOWS, thresholds
│   ├── test_domain_specs.py # All 10 domains (parametrized)
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

## Database Maintenance & Optimization

Routine maintenance procedures to keep PostgreSQL healthy as data grows.

### Quick Reference

```bash
make db-health                          # Full health report (size, cache hit, bloat, seq scans)
make db-analyze                         # Update planner statistics (run after bulk loads)
make db-optimize                        # ANALYZE + identify unused indexes (dry-run)
make db-drop-unused-indexes             # Show unused indexes (dry-run)
make db-drop-unused-indexes EXECUTE=1   # Actually drop unused indexes
make db-retention                       # Show data retention actions (dry-run)
make db-retention EXECUTE=1             # Apply retention policies (drop old partitions, delete old rows)
make db-maintain                        # Routine: ANALYZE + health report
```

### Recommended Schedule

| Task | Frequency | Command |
|---|---|---|
| ANALYZE (planner stats) | After every bulk load, or weekly | `make db-analyze` |
| Health check | Weekly | `make db-health` |
| Drop unused indexes | Monthly | `make db-drop-unused-indexes EXECUTE=1` |
| Retention policy | Monthly | `make db-retention EXECUTE=1` |
| Full maintenance | Weekly | `make db-maintain` |

### PostgreSQL Configuration (docker-compose.yml)

Tuned for a 16 GB host with SSD storage:

| Parameter | Value | Why |
|---|---|---|
| `shared_buffers` | 4 GB | 25% of RAM — primary buffer cache |
| `effective_cache_size` | 12 GB | 75% of RAM — planner hint for OS cache |
| `work_mem` | 128 MB | Reduce temp file spills for sorts/joins |
| `maintenance_work_mem` | 1 GB | Faster VACUUM, REINDEX, CREATE INDEX |
| `effective_io_concurrency` | 200 | SSD-optimized parallel I/O |
| `max_parallel_workers_per_gather` | 4 | Parallel query execution |
| `wal_buffers` | 64 MB | Better write throughput |
| `log_min_duration_statement` | 5000 ms | Log slow queries (> 5s) |
| `default_statistics_target` | 200 | More accurate planner stats |

### Index Strategy

- **BRIN indexes** on large partitioned tables (`fact_customer_demand_monthly`, `fact_inventory_snapshot`) — 100-1000x smaller than B-tree for date-ordered data
- **Partial indexes** for hot query paths (open exceptions, urgent signals, pending approvals)
- **Unique indexes on MVs** enable `REFRESH MATERIALIZED VIEW CONCURRENTLY` (zero read downtime)
- **Unused index cleanup**: `scripts/db/drop_unused_indexes.py` identifies indexes with zero planner scans. Preserves primary keys and unique constraints.

### Materialized View Refresh

All MV refreshes now use `CONCURRENTLY` (via unique indexes from migration 119), providing zero-downtime refreshes. The tiered refresh order prevents dependency issues:

1. **Tier 1**: `agg_sales_monthly`, `agg_forecast_monthly`, `agg_inventory_monthly`
2. **Tier 2**: `mv_inventory_forecast_monthly`, `mv_fill_rate_monthly`, `mv_intramonth_stockout`
3. **Tier 3**: `mv_supplier_performance`, `mv_supplier_po_performance`, `agg_accuracy_by_dim`, `agg_dfu_coverage`
4. **Tier 4**: `mv_inventory_health_score`, `mv_control_tower_kpis`

### Data Retention Policies

Configured in `config/platform/db_maintenance_config.yaml`:

| Table | Retention | Method |
|---|---|---|
| `fact_customer_demand_monthly` | 36 months | Drop old partitions |
| `fact_inventory_snapshot` | 24 months | Drop old partitions |
| `job_history` | 3 months | DELETE + VACUUM |
| `ai_call_log` | 2 months | DELETE + VACUUM |
| `fact_audit_log` | 12 months | DELETE + VACUUM |
| `fact_query_performance` | 1 month | DELETE + VACUUM |

### Maintenance Scripts

| Script | Purpose |
|---|---|
| `scripts/db/drop_unused_indexes.py` | Identify and drop indexes with zero planner scans |
| `scripts/db/db_maintenance.py analyze` | Run ANALYZE on all tables |
| `scripts/db/db_maintenance.py health` | Comprehensive health report |
| `scripts/db/db_maintenance.py retention` | Apply retention policies |

---

## Database Cleanup & Fresh Recreate

Full wipe-and-reload procedure: clears non-config data, derived outputs, job/perf history, and experiment history while preserving configuration masters, schedules, and platform settings. Then reloads from `data/input/`, runs the ML pipeline through champion selection, and refreshes baseline planning outputs.

> **When to use:** Starting fresh with new input data, recovering from a corrupted pipeline state, or resetting after major schema changes.

### Quick Start (Make Targets)

**One-command recipes** — pick the level you need:

```bash
make fresh-all              # Full reset: truncate + clean + load + ML + champion + baseline planning (~4-6 hours)
make fresh-champion         # Load + features + backtests + champion (no truncate, ~3-4 hours)
make fresh-backtest         # Load + features + backtests (no champion, ~3 hours)
make fresh-features         # Load + clustering + seasonality + variability + LT (~1 hour)
make fresh-load             # Normalize + load + refresh MVs only (~5 min)
```

**Individual step targets** — run these manually if you need granular control:

| Target | What it does | Time |
|---|---|---|
| `make db-truncate-data` | Truncate non-config data, history, and experiment tables in one transaction while preserving configuration masters | < 1 min |
| `make clean-artifacts` | Remove stale clean CSVs, backtest/tuning outputs, clustering artifacts, champion files, and perf reports | < 1 sec |
| `make normalize-all` | Normalize all 10 input CSVs → `data/staged/*_clean.csv` | ~1 min |
| `make load-all` | Load all 10 domains into Postgres (dimensions first, facts second) | ~1 min |
| `make refresh-mvs-tiered` | Refresh all 13 MVs in 4-tier dependency order | ~30 sec |
| `make cluster-all` | Feature engineering + KMeans training + label + update dim_sku.ml_cluster | ~10 min |
| `make seasonality-all` | Seasonality detection + update dim_sku | ~5 min |
| `make variability-all` | Demand variability computation → dim_sku | ~2 min |
| `make lt-profile-all` | Lead time profiles → dim_item_lead_time_profile | ~2 min |
| `make backtest-lgbm` | LGBM per-cluster backtest (10 timeframes) | ~30 min |
| `make backtest-catboost` | CatBoost per-cluster backtest | ~40 min |
| `make backtest-xgboost` | XGBoost per-cluster backtest | ~20 min |
| `make backtest-chronos` | Chronos T5 foundation model backtest | ~2.5 hours |
| `make backtest-bolt` | Chronos Bolt foundation model backtest | ~12 min |
| `make backtest-chronos2` | Chronos 2 foundation model backtest | ~5.5 hours |
| `make backtest-chronos2e` | Chronos 2 Enriched backtest (31 covariates) | ~6 hours |
| `make backtest-seasonal-naive` | Seasonal Naive baseline backtest | ~5 min |
| `make backtest-rolling-mean` | Rolling Mean baseline backtest | ~5 min |
| `make backtest-mstl` | MSTL decomposition backtest | ~15 min |
| `make backtest-nhits` | N-HiTS deep learning backtest | ~1 hour |
| `make backtest-nbeats` | N-BEATS deep learning backtest | ~1 hour |
| `make backtest-baselines` | Seasonal Naive + Rolling Mean together | ~10 min |
| `make backtest-all` | All tree + foundation backtests sequentially | ~12 hours |
| `make backtest-load-all` | Load all backtest predictions into DB | ~5 min |
| `make backtest-load-all-bulk` | Load all predictions with single index cycle (~4x faster) | ~1.5 min |
| `make backtest-load-bulk` | Load 4 core models (lgbm, catboost, xgboost, chronos) in bulk | ~1 min |
| `make backtest-load-main-only` | Load specific models to main table only (skip archive) | varies |
| `make backtest-load-archive-only` | Load specific models to archive only (skip main table) | varies |
| `make refresh-accuracy-mvs` | Refresh 4 accuracy MVs (after backtest load) | ~10 sec |
| `make champion-all` | Train meta-learner + simulate strategies + select champion | ~15 min |
| `make policy-all` | Refresh policy assignments while preserving manual overrides | ~1 min |
| `make ss-all` | Recompute safety stock targets | ~2 min |
| `make eoq-all` | Recompute EOQ targets | ~1 min |
| `make health-all` | Refresh inventory health score after SS/EOQ refresh | ~1 min |

**Dependency chain:**

```
fresh-all
├── db-truncate-data              (truncate non-config data/history/experiments)
├── clean-artifacts               (remove stale files)
└── fresh-champion
    └── fresh-backtest
        └── fresh-features
            └── fresh-load
                ├── normalize-all     (CSV → clean CSV)
                ├── load-all          (clean CSV → Postgres)
                └── refresh-mvs-tiered (13 MVs, tier-ordered)
            ├── cluster-all           (clustering pipeline)
            ├── seasonality-all       (seasonality detection)
            ├── variability-all       (demand variability)
            └── lt-profile-all        (lead time profiles)
        ├── backtest-all              (LGBM + CatBoost + XGBoost + Chronos + Bolt + Chronos2 + Chronos2e)
        ├── backtest-load-all         (load predictions → DB)
        └── refresh-accuracy-mvs      (accuracy MVs)
    └── champion-all                  (meta-learner + simulate + select)
├── seed-baselines                   (seed baseline forecasts)
├── policy-all                       (refresh DFU policy assignments)
├── ss-all                           (recompute safety stock)
├── eoq-all                          (recompute EOQ)
└── health-all                       (refresh inventory health score)
```

> **Note:** `fresh-all` covers data loading, ML pipeline, and baseline inventory planning only. To fully populate the application (production forecasts, demand planning, operations), run `setup-all` or the remaining phases below.

### Full Application Setup (`setup-all`)

`setup-all` runs all 6 phases end-to-end. Use this for a complete environment build from scratch.

```bash
make setup-all    # Phases 1-6: data → features → backtests → inv planning → demand planning → ops
```

**Phase breakdown:**

| Phase | Target | What it does | Depends on |
|---|---|---|---|
| 1 | `setup-data` | Normalize + load all 10 domains (parallel pipeline) | Input CSVs |
| 2 | `setup-features` | Clustering, seasonality, variability, lead time, ABC-XYZ, demand signals | Phase 1 |
| 3 | `setup-backtest` | All backtests (tree + foundation) + load + accuracy refresh + champion selection + seed baselines | Phase 2 |
| 4 | `setup-inv-planning` | EOQ, policies, safety stock, exceptions, fill rate, health, supplier perf, investment, intramonth, control tower, rebalancing | Phase 1 |
| 5 | `setup-demand-planning` | Production forecasts, projections, POs, quantiles, consensus, planned orders, replenishment plan, bias, blended, service level, lead time, echelon | Phase 3 |
| 6 | `setup-ops` | S&OP, events, financial plan, storyboard, scenarios, DQ | Phase 1 |

**Dependency chain:**

```
setup-all
├── setup-backtest (Phase 3)
│   └── setup-features (Phase 2)
│       └── setup-data (Phase 1)
│           └── run_pipeline.py --mode full --parallel
│       ├── cluster-all
│       ├── seasonality-all
│       ├── variability-all
│       ├── lt-profile-all
│       ├── abc-xyz-all
│       └── demand-signals-all
│   ├── backtest-all           (LGBM + CatBoost + XGBoost + Chronos + Bolt + Chronos2 + Chronos2e)
│   ├── backtest-load-all      (predictions → DB)
│   ├── accuracy-slice-refresh (accuracy MVs)
│   ├── champion-all           (meta-learner + simulate + select)
│   └── seed-baselines         (baseline forecasts)
├── setup-inv-planning (Phase 4)
│   ├── eoq-all                ├── supplier-perf-all
│   ├── policy-all             ├── investment-all
│   ├── ss-all                 ├── intramonth-all
│   ├── exceptions-generate    ├── control-tower-all
│   ├── fill-rate-all          └── rebalancing-all
│   └── health-all
├── setup-demand-planning (Phase 5)
│   ├── forecast-prod-all      ├── planned-orders-all
│   ├── projection-all         ├── replplan-all
│   ├── po-all                 ├── bias-all
│   ├── quantile-all           ├── blended-all
│   ├── consensus-all          ├── service-level-all
│   ├── lead-time-all          └── echelon-all
└── setup-ops (Phase 6)
    ├── sop-all                ├── storyboard-all
    ├── events-all             ├── scenarios-all
    ├── financial-plan-all     └── dq-all
```

**After `fresh-all`, run the remaining phases:**

```bash
make setup-demand-planning    # Phase 5: production forecasts + demand planning
make setup-ops                # Phase 6: S&OP, events, storyboard, DQ
```

### Preserved Tables (Untouched)

These tables are not explicitly truncated by `db-truncate-data` because they hold durable configuration, master data, schedules, or platform settings that should survive a fresh reload:

| Category | Tables |
|---|---|
| Config / Policy | `dim_replenishment_policy`, `fact_dfu_policy_assignment`, `fact_service_level_targets` |
| Master Data / Planning Inputs | `dim_supplier`, `dim_item_supplier`, `dim_item_cost`, `fact_budget_periods`, `dim_echelon_network`, `dim_external_signal_source`, `dim_transfer_lane` |
| Platform | `dim_user`, `dim_dq_check_catalog`, `dim_notification_channel`, `dim_webhook_registration`, `dim_erp_integration`, `dim_report_template`, `fact_report_schedule`, `job_schedule` |
| MLflow | All `experiments`, `runs`, `metrics`, `params`, `tags`, `logged_models`, `model_versions`, `registered_models`, `datasets`, `inputs`, `trace_*` tables |
| System | `alembic_version` |

### Step 1: Truncate Data Tables

Run as a single SQL transaction. Ordered by FK dependency (children before parents). This reset clears transactional facts, derived outputs, job/perf history, and experiment history, while leaving configuration masters in place.

> **Note:** `fact_inventory_snapshot` is monthly RANGE-partitioned (2025-01 through 2026-03 + default). TRUNCATE on the parent cascades to all partitions automatically.

```bash
docker compose exec -T postgres psql -U demand -d demand_mvp -v ON_ERROR_STOP=1 <<'EOSQL'
BEGIN;

-- Group 1: AI / Analytics
TRUNCATE TABLE ai_recommendation_outcomes CASCADE;
TRUNCATE TABLE ai_insights CASCADE;
TRUNCATE TABLE ai_planning_memos CASCADE;
TRUNCATE TABLE ai_call_log CASCADE;

-- Group 2: Exceptions / Decisions
TRUNCATE TABLE planner_decisions CASCADE;
TRUNCATE TABLE exception_queue CASCADE;

-- Group 3: S&OP (children → parent)
TRUNCATE TABLE fact_sop_approved_plan CASCADE;
TRUNCATE TABLE fact_sop_gaps CASCADE;
TRUNCATE TABLE fact_sop_supply_constraints CASCADE;
TRUNCATE TABLE fact_sop_demand_review CASCADE;
TRUNCATE TABLE fact_sop_cycles CASCADE;

-- Group 4: Events (children → parent)
TRUNCATE TABLE fact_event_conflicts CASCADE;
TRUNCATE TABLE fact_event_performance CASCADE;
TRUNCATE TABLE fact_event_adjusted_forecast CASCADE;
TRUNCATE TABLE fact_event_calendar CASCADE;

-- Group 5: Scenarios (child → parent)
TRUNCATE TABLE fact_scenario_results CASCADE;
TRUNCATE TABLE fact_supply_scenarios CASCADE;

-- Group 6: Procurement (children → parent)
TRUNCATE TABLE fact_po_approval_log CASCADE;
TRUNCATE TABLE fact_po_receipts CASCADE;
TRUNCATE TABLE fact_open_purchase_orders CASCADE;
TRUNCATE TABLE fact_purchase_orders CASCADE;

-- Group 7: Consensus / Overrides
TRUNCATE TABLE fact_consensus_plan CASCADE;
TRUNCATE TABLE fact_forecast_overrides CASCADE;

-- Group 8: Rebalancing (child → parent)
TRUNCATE TABLE fact_rebalancing_transfer CASCADE;
TRUNCATE TABLE fact_rebalancing_plan CASCADE;

-- Group 9: Forecasting / Backtesting
TRUNCATE TABLE backtest_lag_archive CASCADE;
TRUNCATE TABLE fact_external_forecast_monthly CASCADE;
TRUNCATE TABLE fact_candidate_forecast CASCADE;
TRUNCATE TABLE model_promotion_log CASCADE;
TRUNCATE TABLE fact_production_forecast CASCADE;
TRUNCATE TABLE fact_blended_demand_plan CASCADE;
TRUNCATE TABLE fact_demand_plan CASCADE;
TRUNCATE TABLE fact_demand_plan_weekly CASCADE;
TRUNCATE TABLE fact_bias_corrections CASCADE;
TRUNCATE TABLE fact_bias_correction_history CASCADE;

-- Group 10: Inventory (parent CASCADE → all 15 partitions + default)
TRUNCATE TABLE fact_inventory_snapshot CASCADE;
TRUNCATE TABLE fact_inventory_projection CASCADE;

-- Group 11: Sales
TRUNCATE TABLE fact_sales_monthly CASCADE;
TRUNCATE TABLE fact_sales_monthly_original CASCADE;

-- Group 11b: Customer Demand (parent CASCADE → all partitions + default)
TRUNCATE TABLE fact_customer_demand_monthly CASCADE;

-- Group 12: Inventory Planning
TRUNCATE TABLE fact_ss_simulation_results CASCADE;
TRUNCATE TABLE fact_safety_stock_targets CASCADE;
TRUNCATE TABLE fact_eoq_targets CASCADE;
TRUNCATE TABLE fact_demand_signals CASCADE;
TRUNCATE TABLE fact_replenishment_plan CASCADE;
TRUNCATE TABLE fact_replenishment_exceptions CASCADE;
TRUNCATE TABLE fact_planned_orders CASCADE;
TRUNCATE TABLE fact_plan_versions CASCADE;
TRUNCATE TABLE fact_financial_inventory_plan CASCADE;
TRUNCATE TABLE fact_inventory_investment_plan CASCADE;
TRUNCATE TABLE fact_efficient_frontier CASCADE;

-- Group 13: Echelon
TRUNCATE TABLE fact_echelon_reorder_points CASCADE;
TRUNCATE TABLE fact_echelon_ss_targets CASCADE;

-- Group 14: Service Level / Lead Time
TRUNCATE TABLE fact_service_level_performance CASCADE;
TRUNCATE TABLE fact_lead_time_actuals CASCADE;
TRUNCATE TABLE fact_lt_review_triggers CASCADE;

-- Group 15: External Signals
TRUNCATE TABLE fact_external_signal CASCADE;

-- Group 16: DQ / Collaboration / Audit / Notifications / Reports
TRUNCATE TABLE fact_dq_corrections CASCADE;
TRUNCATE TABLE fact_dq_check_results CASCADE;
TRUNCATE TABLE fact_annotation CASCADE;
TRUNCATE TABLE fact_shared_view CASCADE;
TRUNCATE TABLE fact_intervention_metrics CASCADE;
TRUNCATE TABLE fact_notification_log CASCADE;
TRUNCATE TABLE fact_webhook_delivery CASCADE;
TRUNCATE TABLE fact_report_delivery CASCADE;
TRUNCATE TABLE fact_audit_log CASCADE;
TRUNCATE TABLE fact_query_performance CASCADE;

-- Group 17: Infrastructure
TRUNCATE TABLE audit_load_batch CASCADE;
TRUNCATE TABLE job_history CASCADE;
TRUNCATE TABLE perf_suggestion CASCADE;
TRUNCATE TABLE perf_query CASCADE;
TRUNCATE TABLE perf_section CASCADE;
TRUNCATE TABLE perf_run CASCADE;

-- Group 18: Tuning / Model Experiment History
TRUNCATE TABLE tuning_chat_message CASCADE;
TRUNCATE TABLE tuning_chat_session CASCADE;
TRUNCATE TABLE tuning_promotion_log CASCADE;
TRUNCATE TABLE lgbm_tuning_lag_cluster CASCADE;
TRUNCATE TABLE lgbm_tuning_lag CASCADE;
TRUNCATE TABLE lgbm_tuning_comparison CASCADE;
TRUNCATE TABLE lgbm_tuning_month CASCADE;
TRUNCATE TABLE lgbm_tuning_cluster CASCADE;
TRUNCATE TABLE lgbm_tuning_timeframe CASCADE;
TRUNCATE TABLE lgbm_tuning_run CASCADE;

-- Group 19: Cluster / Champion Experiment History
TRUNCATE TABLE cluster_experiment_comparison CASCADE;
TRUNCATE TABLE cluster_experiment CASCADE;
TRUNCATE TABLE champion_experiment CASCADE;

-- Group 20: Dimensions (facts already cleared)
TRUNCATE TABLE dim_sku CASCADE;
TRUNCATE TABLE dim_item CASCADE;
TRUNCATE TABLE dim_location CASCADE;
TRUNCATE TABLE dim_customer CASCADE;
TRUNCATE TABLE dim_time CASCADE;
TRUNCATE TABLE dim_sourcing CASCADE;

-- Group 21: Profiles / Decomposition
TRUNCATE TABLE dim_item_lead_time_profile CASCADE;
TRUNCATE TABLE dim_lead_time_profile CASCADE;
TRUNCATE TABLE mv_demand_decomposition CASCADE;

COMMIT;
EOSQL
```

### Step 2: Clean Intermediate Files

Remove stale artifacts so the pipeline regenerates everything from scratch:

```bash
rm -f data/staged/*_clean.csv data/staged/inventory_clean.csv
rm -rf data/backtest/lgbm_cluster/ data/backtest/catboost_cluster/ data/backtest/xgboost_cluster/
rm -rf data/backtest/chronos/ data/backtest/chronos_bolt/ data/backtest/chronos2/ data/backtest/chronos2_enriched/
rm -rf data/backtest/seasonal_naive/ data/backtest/rolling_mean/ data/backtest/mstl/
rm -rf data/backtest/nhits/ data/backtest/nbeats/
rm -rf data/backtest/logs/ data/backtest/tuning_archive/ data/tuning/ data/perf_reports/
rm -rf data/clustering/ data/champion/ data/models/
rm -f data/staged/seasonality_results.csv data/staged/clustering_features.csv
```

### Step 3: Normalize Input CSVs

```bash
~/.local/bin/uv run python scripts/etl/normalize_dataset_csv.py --dataset item
~/.local/bin/uv run python scripts/etl/normalize_dataset_csv.py --dataset location
~/.local/bin/uv run python scripts/etl/normalize_dataset_csv.py --dataset customer
~/.local/bin/uv run python scripts/etl/normalize_dataset_csv.py --dataset time
~/.local/bin/uv run python scripts/etl/normalize_dataset_csv.py --dataset sku
~/.local/bin/uv run python scripts/etl/normalize_dataset_csv.py --dataset sales
~/.local/bin/uv run python scripts/etl/normalize_dataset_csv.py --dataset forecast
~/.local/bin/uv run python scripts/etl/normalize_inventory_csv.py
~/.local/bin/uv run python scripts/etl/normalize_dataset_csv.py --dataset sourcing
~/.local/bin/uv run python scripts/etl/normalize_dataset_csv.py --dataset purchase_order
```

### Step 4: Load Into Postgres (Dimensions First, Then Facts)

```bash
# Wave 1: Dimensions
~/.local/bin/uv run python scripts/etl/load_dataset_postgres.py --dataset item
~/.local/bin/uv run python scripts/etl/load_dataset_postgres.py --dataset location
~/.local/bin/uv run python scripts/etl/load_dataset_postgres.py --dataset customer
~/.local/bin/uv run python scripts/etl/load_dataset_postgres.py --dataset time
~/.local/bin/uv run python scripts/etl/load_dataset_postgres.py --dataset sku
~/.local/bin/uv run python scripts/etl/load_dataset_postgres.py --dataset sourcing

# Wave 2: Facts
~/.local/bin/uv run python scripts/etl/load_dataset_postgres.py --dataset sales
~/.local/bin/uv run python scripts/etl/load_dataset_postgres.py --dataset forecast
~/.local/bin/uv run python scripts/etl/load_dataset_postgres.py --dataset inventory
~/.local/bin/uv run python scripts/etl/load_dataset_postgres.py --dataset purchase_order
```

### Step 5: Refresh Materialized Views (Tier-Ordered)

MV dependencies require a specific refresh order. Some MVs (DQ, sensing, projections) depend on data from later pipeline steps — they will populate when those steps run.

```bash
docker compose exec -T postgres psql -U demand -d demand_mvp -c "
  -- Tier 1: Base aggregates (no MV dependencies)
  REFRESH MATERIALIZED VIEW agg_sales_monthly;
  REFRESH MATERIALIZED VIEW agg_forecast_monthly;
  REFRESH MATERIALIZED VIEW agg_inventory_monthly;

  -- Tier 2: Depend on tier 1 aggregates
  REFRESH MATERIALIZED VIEW mv_inventory_forecast_monthly;
  REFRESH MATERIALIZED VIEW mv_fill_rate_monthly;
  REFRESH MATERIALIZED VIEW mv_intramonth_stockout;
  REFRESH MATERIALIZED VIEW mv_supplier_performance;
  REFRESH MATERIALIZED VIEW mv_supplier_po_performance;
  REFRESH MATERIALIZED VIEW mv_po_lead_time_analysis;
  REFRESH MATERIALIZED VIEW agg_accuracy_by_dim;
  REFRESH MATERIALIZED VIEW agg_dfu_coverage;

  -- Tier 3: Depend on tier 2 (mv_inventory_forecast_monthly)
  REFRESH MATERIALIZED VIEW mv_inventory_health_score;

  -- Tier 4: Depend on tier 3 (mv_inventory_health_score + mv_fill_rate + mv_intramonth)
  REFRESH MATERIALIZED VIEW mv_control_tower_kpis;
"
```

### Step 6: Clustering & Feature Engineering

```bash
~/.local/bin/uv run python scripts/ml/run_cluster_pipeline.py  # unified: features -> train -> label -> promote
```

### Step 7: SKU Feature Engineering (Seasonality + Variability)

The unified pipeline computes seasonality, variability, and lifecycle features in a single pass.
The legacy scripts (`detect_seasonality.py`, `update_seasonality_profiles.py`, `compute_demand_variability.py`, `generate_clustering_features.py`, `train_clustering_model.py`, `detect_drift.py`) have been removed -- use `make features-compute` (or `scripts/ml/compute_sku_features.py`) and `scripts/ml/run_cluster_pipeline.py` instead.

```bash
~/.local/bin/uv run python scripts/ml/compute_sku_features.py   # or: make features-compute
```

### Step 8: Lead Time Profiles

```bash
~/.local/bin/uv run python scripts/inventory/compute_lead_time_variability.py
```

### Step 9: Run Backtests (Tree + Foundation + Statistical + Deep Learning)

Sequential execution (safe for laptops). For parallel, append `&` to each and `wait` at the end.

```bash
# Tree models
~/.local/bin/uv run python scripts/ml/run_backtest.py
~/.local/bin/uv run python scripts/ml/run_backtest_catboost.py
~/.local/bin/uv run python scripts/ml/run_backtest_xgboost.py

# Foundation models
~/.local/bin/uv run python -m scripts.ml.run_backtest_chronos
~/.local/bin/uv run python -m scripts.ml.run_backtest_chronos_bolt
~/.local/bin/uv run python -m scripts.ml.run_backtest_chronos2
~/.local/bin/uv run python -m scripts.ml.run_backtest_chronos2_enriched

# Statistical baselines
~/.local/bin/uv run python scripts/ml/run_backtest.py --model seasonal_naive
~/.local/bin/uv run python scripts/ml/run_backtest.py --model rolling_mean
~/.local/bin/uv run python scripts/ml/run_backtest_mstl.py

# Deep learning models
~/.local/bin/uv run python scripts/ml/run_backtest_dl.py --model nhits
~/.local/bin/uv run python scripts/ml/run_backtest_dl.py --model nbeats
```

### Step 10: Load Backtest Predictions

```bash
# Standard load (per-model index cycle):
~/.local/bin/uv run python scripts/etl/load_backtest_forecasts.py --all --replace

# Bulk load (~4x faster — single index drop/recreate across all models):
~/.local/bin/uv run python scripts/etl/load_backtest_forecasts.py --all --replace --bulk

# Load specific models only:
~/.local/bin/uv run python scripts/etl/load_backtest_forecasts.py \
  --models lgbm_cluster catboost_cluster xgboost_cluster chronos --replace --bulk
```

### Step 11: Refresh Accuracy MVs

These MVs depend on backtest data loaded in Step 10:

```bash
docker compose exec -T postgres psql -U demand -d demand_mvp -c "
  REFRESH MATERIALIZED VIEW agg_accuracy_by_dim;
  REFRESH MATERIALIZED VIEW agg_accuracy_lag_archive;
  REFRESH MATERIALIZED VIEW agg_dfu_coverage;
  REFRESH MATERIALIZED VIEW agg_dfu_coverage_lag_archive;
"
```

### Step 12: Champion Model Selection

```bash
~/.local/bin/uv run python scripts/ml/train_meta_learner.py --config config/forecasting/forecast_pipeline_config.yaml
~/.local/bin/uv run python scripts/ml/simulate_champion_strategies.py --config config/forecasting/forecast_pipeline_config.yaml
~/.local/bin/uv run python scripts/ml/run_champion_selection.py --config config/forecasting/forecast_pipeline_config.yaml
```

### Step 13: Baseline Planning Refresh

`fresh-all` finishes by rebuilding the baseline planning outputs that are derived from the refreshed dataset while preserving user-managed configuration tables:

```bash
make policy-all
make ss-all
make eoq-all
make health-all
```

### Validation

After completing the pipeline, verify key table counts:

```bash
docker compose exec -T postgres psql -U demand -d demand_mvp -c "
  SELECT 'dim_item' AS tbl, COUNT(*) FROM dim_item
  UNION ALL SELECT 'dim_location', COUNT(*) FROM dim_location
  UNION ALL SELECT 'dim_sku', COUNT(*) FROM dim_sku
  UNION ALL SELECT 'fact_sales_monthly', COUNT(*) FROM fact_sales_monthly
  UNION ALL SELECT 'fact_external_forecast_monthly', COUNT(*) FROM fact_external_forecast_monthly
  UNION ALL SELECT 'fact_inventory_snapshot', COUNT(*) FROM fact_inventory_snapshot
  UNION ALL SELECT 'backtest_lag_archive', COUNT(*) FROM backtest_lag_archive
  UNION ALL SELECT 'fact_safety_stock_targets', COUNT(*) FROM fact_safety_stock_targets
  UNION ALL SELECT 'fact_eoq_targets', COUNT(*) FROM fact_eoq_targets
  UNION ALL SELECT 'champion_rows', COUNT(*) FROM fact_external_forecast_monthly WHERE model_id = 'champion'
  ORDER BY tbl;
"
```

---

## Data Cleanup

### Remove Backtest Model Predictions

```bash
make backtest-list                                 # Row counts per model_id
make backtest-clean MODELS="--dry-run lgbm_cluster" # Preview before deleting
make backtest-clean MODELS="lgbm_cluster catboost_cluster"  # Delete specific tree models
make backtest-clean MODELS="chronos chronos_bolt chronos2 chronos2_enriched"  # Delete all foundation models
make backtest-clean MODELS="seasonal_naive rolling_mean mstl"  # Delete statistical baselines
make backtest-clean MODELS="nhits nbeats"          # Delete deep learning models
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
| Inventory normalize fails | Verify 14 CSVs in `data/input/`: `ls data/input/Inventory_Snapshot_*.csv \| wc -l` (expect 14); files must be UTF-8 CSV with columns `exec_date,item,loc,lead_time,tot_oh,tot_oh_oo,mtd_sls` |
| Inventory load fails | Apply DDL first: `make db-apply-inventory`; verify `ls -lh data/staged/inventory_clean.csv` |
| Inventory tab shows no data | Verify load: `docker compose exec -T postgres psql -U demand -d demand_mvp -c "SELECT COUNT(*) FROM fact_inventory_snapshot"`; refresh view: `make refresh-agg-inventory` |

### ML Pipelines

| Problem | Fix |
|---|---|
| Clustering fails | Load sales first: `make load-sales`; verify MLflow: `docker ps \| grep mlflow`; check: `ls -lh data/staged/clustering_features.csv` |
| Cluster assignments not updating | Preview with `--dry-run`; verify DFU key format matches database; check Postgres connection |
| Backtest fails | Run clustering first: `make cluster-all`; load sales: `make load-sales`; install deps: `uv sync` |
| Champion selection finds no DFUs | Load backtest predictions: `make backtest-load`; lower `min_dfu_rows` in `config/forecasting/forecast_pipeline_config.yaml` champion section; verify models exist: `SELECT DISTINCT model_id FROM fact_external_forecast_monthly` |
| Chat endpoint errors | Set `OPENAI_API_KEY` in `.env`; run `make generate-embeddings`; check API logs for rate limit errors |
| AI Planner errors | Set `ANTHROPIC_API_KEY` in `.env`; verify insight schema exists: `make ai-insights-schema`; check API logs for rate limit or tool dispatch errors |

### Exception Queue

| Problem | Fix |
|---|---|
| Exception queue is empty | Safety stock targets must exist first. Run `make ss-all` to compute safety stock targets, then `make exceptions-generate` to detect exceptions. Dependency chain: safety stock targets -> exception generation |
| `make exceptions-generate` finds no exceptions | Verify `fact_safety_stock_targets` has rows: `SELECT COUNT(*) FROM fact_safety_stock_targets`; check thresholds in `config/operations/exception_config.yaml` |

### Inventory Rebalancing

| Problem | Fix |
|---|---|
| `mv_network_balance` refresh fails | Ensure dependent tables are populated first: `fact_rebalancing_plan`, `dim_transfer_lane`; run `make rebalancing-refresh` after data is loaded |
| Rebalancing compute finds no candidates | Verify safety stock targets exist (`SELECT COUNT(*) FROM fact_safety_stock_targets`); verify inventory snapshots are loaded; check `config/inventory/rebalancing_config.yaml` thresholds are not too restrictive |
| Dry-run shows transfers but compute writes nothing | Confirm you are running `make rebalancing-compute` (not `make rebalancing-compute-dry`); check DB connectivity and table permissions |

### Platform Services (Specs 08-01 through 08-10)

| Problem | Fix |
|---|---|
| JWT auth returns 401 on all requests | Verify `JWT_SECRET` is set in `.env`; check token expiry; re-authenticate to get a fresh token |
| `bcrypt` or `PyJWT` import error | Run `make init` or `uv sync` to install new dependencies |
| Data quality checks return empty results | Run `make db-apply-sql` to ensure DDL 062-070 are applied; verify check catalog has entries via `GET /data-quality/catalog` |
| Cache not working (stale data) | Check `CACHE_BACKEND` env var; for Redis, verify connectivity: `redis-cli ping`; InMemory cache clears on restart |
| Notifications not sending | Verify channel is `enabled: true` in `config/platform/notification_config.yaml`; check SMTP credentials / Slack webhook URL |
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
Phase 10 (Start Services)
    └─► Cache layer active (Redis or in-memory fallback)
    └─► Query performance tracking active (fact_query_performance)
    └─► Auth/RBAC middleware enforced on mutation endpoints
```
