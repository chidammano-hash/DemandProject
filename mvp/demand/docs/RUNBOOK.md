# Demand Unified Runbook

## 1) Initialize
```bash
cd mvp/demand
make init
```

## 2) Start infra
```bash
make up
```

## 3) Ingest datasets
```bash
make normalize-all
make load-all
```

Sales fact only:
```bash
make normalize-sales
make load-sales
```

Forecast fact only:
```bash
make normalize-forecast
make load-forecast              # TRUNCATE all + reload external (exec-lag row → main via JOIN, all lags → archive untouched)
make load-forecast-replace      # Replace external only (preserves backtest/champion/ceiling rows)
make load-forecast-replace-no-archive  # Replace external only, skip archive load (fast, no 45M archive INSERT)
```

Inventory fact only:
```bash
make normalize-inventory    # Merge 14 monthly CSVs into single clean CSV
make load-inventory         # Load into Postgres + refresh agg view
```

Or use the full inventory pipeline:
```bash
make db-apply-inventory     # Create table + indexes + materialized view (one-time)
make inventory-pipeline     # normalize + load + refresh aggregates
```

## 3b) Setup chatbot (requires OPENAI_API_KEY in .env)
```bash
make generate-embeddings
```
This parses all domain specs, generates OpenAI embeddings for schema metadata, and stores them in the `chat_embeddings` pgvector table. Re-run after adding new datasets or changing schema.

## 3c) Run DFU clustering (optional, for LGBM model support)
```bash
make cluster-all
```

Or run steps individually:
```bash
make cluster-features  # Generate time series + attribute features
make cluster-train     # Train KMeans model with optimal K selection
make cluster-label     # Assign business labels to clusters
make cluster-update    # Update dim_dfu.cluster_assignment in database
```

This groups DFUs by historical demand patterns for improved global LGBM model performance. Results are logged to MLflow experiment `dfu_clustering`. Cluster assignments can be filtered via `/domains/dfu/page` using the `cluster_assignment` filter, or viewed via `/domains/dfu/clusters` endpoint.

## 3c-1) Hyperparameter tuning (optional, Feature 41)

Two modes are available. Choose based on your goal:

### Mode A — Production scoring (tune once on full history)

Tune LGBM, CatBoost, and/or XGBoost parameters using Bayesian optimisation (Optuna). Uses walk-forward CV with causal masking across the full sales history. Best for production forecasting — tune once, apply indefinitely.

```bash
make tune-lgbm      # ~20–40 min → data/tuning/best_params_lgbm.json
make tune-catboost  # ~30–60 min → data/tuning/best_params_catboost.json
make tune-xgboost   # ~25–50 min → data/tuning/best_params_xgboost.json

# Or all three sequentially
make tune-all
```

The JSON files contain `best_params`, `best_n_estimators`, per-cluster WAPEs, and CV fold metrics. Pass them to backtest scripts via `--params-file` in step 3d.

> **Warning:** Using `--params-file` with backtesting introduces temporal data leakage — the tuner sees all history including future timeframes. Use Mode B for honest backtest accuracy evaluation.

### Mode B — Honest backtesting (per-timeframe causal tuning, PL-002 fix)

Run backtests with inline per-timeframe tuning: each of the 10 expanding timeframes tunes on only the data available up to its training cutoff. No future leakage into backtest accuracy metrics. ~2–3× slower than untuned backtests (600 model fits vs. 250).

```bash
make backtest-lgbm-cluster-tuned       # LGBM — 10 timeframes × 20 trials × 3 folds
make backtest-catboost-cluster-tuned   # CatBoost — same
make backtest-xgboost-cluster-tuned    # XGBoost — same
```

Then load and compare as usual:
```bash
make backtest-load-all
make champion-select
```

## 3d) Run backtesting (optional, requires clustering)

> **Architecture note:** All tree-based backtest scripts (LGBM, CatBoost, XGBoost) share a common framework in `common/`. Each script contains only model-specific training functions (~280 lines) and delegates orchestration to `common/backtest_framework.py` via `run_tree_backtest()`. Shared modules: `backtest_framework.py` (orchestrator), `feature_engineering.py` (lag/rolling features), `metrics.py` (WAPE/accuracy), `mlflow_utils.py` (experiment logging), `db.py` (connection params), `constants.py` (thresholds), `tuning.py` (CV splits, WAPE, param loading, `tune_for_timeframe()`, `TRAIN_FOLD_FNS` registry). `run_tree_backtest()` accepts an optional `inline_tuner_fn` parameter — when provided, each timeframe performs per-timeframe causal tuning using only historically available data (PL-002 fix). Each backtest writes to `data/backtest/<model_id>/` — multiple models can be run without overwriting each other. Prophet and NeuralProphet use shared utilities (`generate_timeframes`, `load_backtest_data`, `postprocess_predictions`, `save_backtest_output`, `log_backtest_run`) but orchestrate their own per-DFU fitting loops. StatsForecast uses the same shared utilities with vectorized batch fitting (no per-DFU loop).

### LGBM

Global model (one LGBM for all DFUs, `ml_cluster` as feature):
```bash
make backtest-lgbm                  # Global LGBM backtest (10 timeframes)
make backtest-load MODEL=lgbm_global  # Load predictions into Postgres
```

Per-cluster model (separate LGBM per cluster):
```bash
make backtest-lgbm-cluster          # Per-cluster LGBM backtest (default params)
make backtest-load MODEL=lgbm_cluster

# With globally tuned params (production scoring mode):
make backtest-lgbm-cluster ARGS="--params-file data/tuning/best_params_lgbm.json"
make backtest-load MODEL=lgbm_cluster

# With per-timeframe inline tuning (honest backtesting, no data leakage — PL-002):
make backtest-lgbm-cluster-tuned
make backtest-load MODEL=lgbm_cluster
```

Or run global + load in one shot:
```bash
make backtest-all           # backtest-lgbm + backtest-load (lgbm_global)
```

### CatBoost

Global model:
```bash
make backtest-catboost                    # Global CatBoost backtest (10 timeframes)
make backtest-load MODEL=catboost_global
```

Per-cluster model:
```bash
make backtest-catboost-cluster            # Per-cluster CatBoost backtest (default params)
make backtest-load MODEL=catboost_cluster

# With globally tuned params (production scoring mode):
make backtest-catboost-cluster ARGS="--params-file data/tuning/best_params_catboost.json"
make backtest-load MODEL=catboost_cluster

# With per-timeframe inline tuning (honest backtesting, no data leakage — PL-002):
make backtest-catboost-cluster-tuned
make backtest-load MODEL=catboost_cluster
```

### XGBoost

Global model:
```bash
make backtest-xgboost                    # Global XGBoost backtest (10 timeframes)
make backtest-load MODEL=xgboost_global
```

Per-cluster model:
```bash
make backtest-xgboost-cluster            # Per-cluster XGBoost backtest (default params)
make backtest-load MODEL=xgboost_cluster

# With globally tuned params (production scoring mode):
make backtest-xgboost-cluster ARGS="--params-file data/tuning/best_params_xgboost.json"
make backtest-load MODEL=xgboost_cluster

# With per-timeframe inline tuning (honest backtesting, no data leakage — PL-002):
make backtest-xgboost-cluster-tuned
make backtest-load MODEL=xgboost_cluster
```

### Prophet

Prophet fits individual time series models per DFU (unlike global tree models). Three strategies:

Global (per-DFU fits across all DFUs):
```bash
make backtest-prophet                    # Global Prophet backtest (per-DFU fits)
make backtest-load MODEL=prophet_global  # Load predictions into Postgres
```

Per-cluster (only clustered DFUs):
```bash
make backtest-prophet-cluster              # Per-cluster Prophet backtest
make backtest-load MODEL=prophet_cluster
```

Pooled (aggregate by cluster → fit → disaggregate proportionally):
```bash
make backtest-prophet-pooled              # Pooled cluster Prophet backtest
make backtest-load MODEL=prophet_pooled
```

### PatchTST (Deep Learning)

PatchTST is a Transformer-based model using patched time series input. Supports Apple MPS GPU acceleration.

Global model:
```bash
make backtest-patchtst                    # Global PatchTST backtest (Apple MPS GPU)
make backtest-load MODEL=patchtst_global
```

Per-cluster model:
```bash
make backtest-patchtst-cluster              # Per-cluster PatchTST backtest
make backtest-load MODEL=patchtst_cluster
```

Transfer learning (global base → per-cluster fine-tune):
```bash
make backtest-patchtst-transfer               # Transfer learning PatchTST backtest
make backtest-load MODEL=patchtst_transfer
```

### DeepAR (Deep Learning)

DeepAR is an LSTM-based autoregressive probabilistic model. Produces point forecasts and prediction intervals.

Global model:
```bash
make backtest-deepar                    # Global DeepAR backtest
make backtest-load MODEL=deepar_global
```

Per-cluster model:
```bash
make backtest-deepar-cluster              # Per-cluster DeepAR backtest
make backtest-load MODEL=deepar_cluster
```

Transfer learning:
```bash
make backtest-deepar-transfer               # Transfer learning DeepAR backtest
make backtest-load MODEL=deepar_transfer
```

### StatsForecast (Fast Statistical Models)

StatsForecast uses vectorized AutoARIMA + AutoETS models — ~100x faster than Prophet for large-scale backtesting.

Global (batch all DFUs at once):
```bash
make backtest-statsforecast                         # Global StatsForecast backtest (AutoARIMA+AutoETS)
make backtest-load MODEL=statsforecast_global
```

Per-cluster (only clustered DFUs):
```bash
make backtest-statsforecast-cluster                   # Per-cluster StatsForecast backtest
make backtest-load MODEL=statsforecast_cluster
```

Pooled (aggregate by cluster → fit → disaggregate):
```bash
make backtest-statsforecast-pooled                   # Pooled cluster StatsForecast backtest
make backtest-load MODEL=statsforecast_pooled
```

### NeuralProphet (PyTorch-based Prophet)

NeuralProphet is a PyTorch-based Prophet successor with Apple MPS GPU acceleration.

Global (per-DFU fits with GPU):
```bash
make backtest-neuralprophet                         # Global NeuralProphet backtest (PyTorch GPU)
make backtest-load MODEL=neuralprophet_global
```

Per-cluster (only clustered DFUs):
```bash
make backtest-neuralprophet-cluster                   # Per-cluster NeuralProphet backtest
make backtest-load MODEL=neuralprophet_cluster
```

Pooled (aggregate by cluster → fit → disaggregate):
```bash
make backtest-neuralprophet-pooled                   # Pooled cluster NeuralProphet backtest
make backtest-load MODEL=neuralprophet_pooled
```

### Transfer Learning (all frameworks)

Transfer learning trains a global base model (no `ml_cluster`), then fine-tunes per cluster with warm-start. Small clusters and unassigned DFUs fall back to the base model (never zeroed).

```bash
make backtest-lgbm-transfer                   # LGBM transfer backtest
make backtest-load MODEL=lgbm_transfer

make backtest-catboost-transfer               # CatBoost transfer backtest
make backtest-load MODEL=catboost_transfer

make backtest-xgboost-transfer                # XGBoost transfer backtest
make backtest-load MODEL=xgboost_transfer
```

Transfer model IDs: `lgbm_transfer`, `catboost_transfer`, `xgboost_transfer`

### Backtest output

Each backtest run writes to a **model-scoped subdirectory**:
- `data/backtest/<model_id>/backtest_predictions.csv` — execution-lag only (loaded into `fact_external_forecast_monthly`)
- `data/backtest/<model_id>/backtest_predictions_all_lags.csv` — lag 0–4 archive (loaded into `backtest_lag_archive`)

For example:
```
data/backtest/lgbm_cluster/backtest_predictions.csv
data/backtest/catboost_cluster/backtest_predictions.csv
data/backtest/xgboost_cluster/backtest_predictions.csv
```

Multiple backtests can be run back-to-back without overwriting each other. Load all at once:

```bash
make backtest-lgbm-cluster
make backtest-catboost-cluster
make backtest-xgboost-cluster
make backtest-load-all          # scans data/backtest/*/ and loads each model
```

Or load one model at a time:
```bash
make backtest-load MODEL=lgbm_cluster
```

`--replace` is built into `backtest-load`. It only deletes rows matching the `model_id` in the CSV, so running different models does **not** affect each other's results in Postgres.

Predictions are stored in `fact_external_forecast_monthly` with model_id values:
- LGBM: `lgbm_global` / `lgbm_cluster` / `lgbm_transfer`
- CatBoost: `catboost_global` / `catboost_cluster` / `catboost_transfer`
- XGBoost: `xgboost_global` / `xgboost_cluster` / `xgboost_transfer`
- Prophet: `prophet_global` / `prophet_cluster` / `prophet_pooled`
- PatchTST: `patchtst_global` / `patchtst_cluster` / `patchtst_transfer`
- DeepAR: `deepar_global` / `deepar_cluster` / `deepar_transfer`
- StatsForecast: `statsforecast_global` / `statsforecast_cluster` / `statsforecast_pooled`
- NeuralProphet: `neuralprophet_global` / `neuralprophet_cluster` / `neuralprophet_pooled`

All-lag predictions are archived in `backtest_lag_archive` for accuracy reporting at any horizon, with each row's original `lag` preserved as `execution_lag` (staging data is never mutated). Results appear automatically in the forecast model selector UI and accuracy KPIs.

`make backtest-load` also automatically refreshes the accuracy slice views (`agg_accuracy_by_dim`, `agg_accuracy_lag_archive`) after loading — no additional step needed.

Verify archive data:
```bash
docker exec demand-mvp-postgres psql -U demand -d demand_mvp \
  -c "SELECT model_id, lag, COUNT(*) FROM backtest_lag_archive GROUP BY 1,2 ORDER BY 1,2"
```

## 3d-cleanup) Clean up backtest model predictions (feature23)

To remove model predictions that are no longer needed from the database:

### List model row counts
```bash
make backtest-list
```

### Preview what would be deleted (dry run)
```bash
make backtest-clean MODELS="--dry-run lgbm_global deepar_global"
```

### Delete specific models
```bash
make backtest-clean MODELS="lgbm_global deepar_global"
```

### Delete ALL non-external backtest models
```bash
make backtest-clean MODELS="--all-backtest"
```

This removes rows from both `fact_external_forecast_monthly` and `backtest_lag_archive` for the specified model_ids, then refreshes all 5 dependent materialized views (`agg_forecast_monthly`, `agg_accuracy_by_dim`, `agg_dfu_coverage`, `agg_accuracy_lag_archive`, `agg_dfu_coverage_lag_archive`).

**Safety notes:**
- `--all-backtest` never deletes `model_id='external'` (source-system forecasts are protected)
- Always use `--dry-run` first to preview row counts before deleting
- After cleanup, re-run `make champion-select` if champion/ceiling rows are affected

## 3d-2) Clean up forecasts by date range

To remove forecast data for specific time periods:

### List row counts by model and month
```bash
make forecast-clean-list
```

### Preview what would be deleted (dry run)
```bash
make forecast-clean ARGS="--before 2025-04-01 --model external --dry-run"
```

### Delete all external model data before April 2025
```bash
make forecast-clean ARGS="--before 2025-04-01 --model external"
```

### Delete all models between Jan-Jun 2024
```bash
make forecast-clean ARGS="--between 2024-01-01 2024-07-01"
```

### Delete specific months
```bash
make forecast-clean ARGS="--months 2024-03 2024-06 2024-09"
```

### Delete a single month for one model
```bash
make forecast-clean ARGS="--months 2025-01 --model external"
```

### Delete from archive only, using fcstdate column
```bash
make forecast-clean ARGS="--before 2025-01-01 --date-column fcstdate --archive-only"
```

This removes matching rows from `fact_external_forecast_monthly` and/or `backtest_lag_archive`, then refreshes all 5 dependent materialized views.

**Date formats accepted:** `YYYY-MM-DD`, `YYYY-MM`, `MM/DD/YYYY` (all normalized to month-start)

**Safety notes:**
- Always use `--dry-run` first to preview row counts before deleting
- Use `--model external` carefully — this removes source-system forecasts
- After cleanup, re-run `make champion-select` if champion/ceiling rows are affected

## 3e) Multi-dimensional accuracy comparison (feature10)

After running `make backtest-load`, the accuracy slice views are automatically populated. To view accuracy sliced by DFU attributes:

1. Open the Forecast domain in the UI.
2. Click **Accuracy Comparison** (collapsible card below the main analytics section).
3. Select a **Slice by** dimension (e.g., Cluster, Supplier, ABC Volume, Region).
4. Optionally select a **Lag Filter** (Execution Lag, or specific lag 0–4).
5. Optionally filter **Models** (comma-separated, e.g., `lgbm_global,external`).

The panel shows:
- **Model comparison table**: side-by-side Accuracy %, WAPE, Bias per model for each bucket. Best model highlighted in teal, high-bias cells in red.
- **Lag curve chart**: accuracy degradation from lag 0 → lag 4, one line per model.

To refresh the views manually (e.g., after `load-forecast` without backtest):
```bash
make accuracy-slice-refresh
```

To verify the slice endpoint:
```bash
make accuracy-slice-check
```

API endpoints:
- `GET /forecast/accuracy/slice?group_by=cluster_assignment&models=lgbm_global,external`
- `GET /forecast/accuracy/lag-curve?models=lgbm_global,lgbm_cluster,external`

## 3f) Champion model selection (feature15)

After loading backtest predictions for multiple models, run champion selection to identify the best model per DFU per month.

### Selection Strategies

The system supports 5 configurable strategies via `config/model_competition.yaml`. All strategies enforce **strict causality** — selection for month T uses ONLY data from months < T (no data leak).

| Strategy | Key Idea | Default |
|---|---|---|
| `expanding` | Cumulative WAPE, all prior months equal weight | **Default** |
| `rolling` | Last N months only (`window_months`, default 6) | |
| `decay` | Exponential decay — recent months weighted more (`decay_factor`, default 0.9) | |
| `ensemble` | Blend top-K models by inverse-WAPE weights (`top_k`, default 3) | |
| `meta_learner` | ML classifier predicts best model from DFU features + performance stats | |

Strategy registry lives in `common/champion_strategies.py`. Each strategy operates on a pandas DataFrame (testable without DB).

### Via CLI

```bash
# Run with default strategy (from config YAML)
make champion-select

# Override strategy on command line
cd mvp/demand && .venv/bin/python -m scripts.run_champion_selection --strategy rolling

# Train meta-learner classifier (required before using meta_learner strategy)
make champion-train-meta

# Simulate all strategies and compare accuracy vs ceiling
make champion-simulate

# Full pipeline: train meta-learner + simulate + select
make champion-all
```

### Via UI
1. Open the Forecast domain in the UI.
2. Go to the **Accuracy Comparison** section.
3. Scroll to the **Champion Selection** panel.
4. Check/uncheck models to include in the competition.
5. Select metric (WAPE or Accuracy %) and lag mode (Execution Lag or fixed 0–4).
6. Select strategy (expanding, rolling, decay, ensemble, meta_learner).
7. Click **Save Config** to persist changes to YAML.
8. Click **Run Competition** to execute champion selection + ceiling computation.
9. Results show: DFUs evaluated, champion accuracy/WAPE, ceiling accuracy/WAPE (oracle), gap-to-ceiling, and model wins breakdown for both champion and ceiling.

### Via API
```bash
# Get current config + available models
curl http://localhost:8000/competition/config

# Update config (with strategy)
curl -X PUT http://localhost:8000/competition/config \
  -H "Content-Type: application/json" \
  -d '{"metric": "wape", "lag": "execution", "min_dfu_rows": 3, "models": ["lgbm_cluster", "catboost_cluster", "xgboost_cluster"], "strategy": "rolling", "strategy_params": {"window_months": 6}}'

# Run champion selection
curl -X POST http://localhost:8000/competition/run

# Get last run summary
curl http://localhost:8000/competition/summary
```

### Output
- Champion rows appear in `fact_external_forecast_monthly` with `model_id='champion'`
- Ceiling rows appear in `fact_external_forecast_monthly` with `model_id='ceiling'`
- Summary saved to `data/champion/champion_summary.json` (includes both champion and ceiling metrics)
- Materialized views refreshed automatically — champion + ceiling appear in all accuracy comparisons
- Running again is idempotent (old champion and ceiling rows replaced)
- Simulation results saved to `data/champion/simulation_results.json`

### Config file
`config/model_competition.yaml` controls which models compete, the selection metric, lag mode, minimum DFU rows, strategy, and strategy parameters. Editable from the UI or directly on disk.

```yaml
competition:
  metric: accuracy_pct
  lag: execution
  min_dfu_rows: 3
  models: [lgbm_cluster, catboost_cluster, xgboost_cluster, neuralprophet_cluster, statsforecast_global]
  strategy: expanding          # expanding | rolling | decay | ensemble | meta_learner
  strategy_params:
    window_months: 6           # rolling strategy window
    decay_factor: 0.90         # decay strategy weighting
    top_k: 3                   # ensemble: blend top-K models
    performance_window: 6      # meta-learner feature window
  meta_learner:
    model_type: random_forest  # random_forest | xgboost
    n_estimators: 200
    max_depth: 15
    test_months: 3
    performance_window: 6
```

### Key files
| File | Purpose |
|---|---|
| `common/champion_strategies.py` | 5 strategy functions + registry + accuracy helper + leak guards |
| `scripts/run_champion_selection.py` | CLI: per-DFU champion selection via configurable strategy |
| `scripts/train_meta_learner.py` | Train meta-learner classifier (ceiling labels as ground truth) |
| `scripts/simulate_champion_strategies.py` | Diagnostic: run all strategies, compare accuracy vs ceiling |
| `config/model_competition.yaml` | Strategy + model + metric configuration |

## 3g) Apply performance indexes (recommended for large tables)

After loading data, apply GIN trigram indexes for fast Data Explorer filtering on large tables:
```bash
make db-apply-sql
```

This creates GIN `gin_trgm_ops` indexes on fact table text columns (`model_id`, `dmdunit`, `loc`, `dmdgroup`). One-time operation; takes 5–15 minutes on 66M+ row tables. Enables indexed `ILIKE` substring search instead of full table scans.

## 3h) Apply job scheduler schema (Feature 39)

Create the `job_history` and `job_schedule` tables used by the Job Scheduler/Monitor:
```bash
make db-apply-jobs
```

This creates:
- `job_history` table with scheduling columns (`scheduled_cron`, `retry_count`, `max_retries`, `pipeline_id`, `pipeline_step`, `triggered_by`) and indexes on `status`, `job_type`, `submitted_at`
- `job_schedule` table for persistent recurring schedule records

Required for the Jobs tab to function. One-time operation.

The job scheduler is powered by **APScheduler 3.11** (`BackgroundScheduler` + `ThreadPoolExecutor`). It enables submitting, scheduling, and monitoring long-running operations from the UI. Jobs are tracked in Postgres with status, progress, params, results, retry counts, and pipeline metadata.

**Core API endpoints:**
- `GET /jobs/types` — List available job types (7 types across 4 groups)
- `GET /jobs/stats` — Aggregate statistics for dashboard KPIs
- `POST /jobs` — Submit a new job (returns HTTP 202)
- `GET /jobs` — List jobs with optional filters and pagination
- `GET /jobs/active` — Currently running/queued jobs
- `GET /jobs/{id}` — Job detail with full result/error
- `POST /jobs/{id}/cancel` — Cancel a running job
- `DELETE /jobs/{id}` — Remove from history

**Scheduling API endpoints:**
- `POST /jobs/schedule` — Create recurring schedule (cron expression or interval minutes)
- `GET /jobs/schedules` — List active recurring schedules
- `DELETE /jobs/schedules/{id}` — Remove a recurring schedule

**Pipeline API endpoint:**
- `POST /jobs/pipeline` — Submit a chained job pipeline (sequential execution)

**UI:** Jobs tab (keyboard shortcut `9`) — automation dashboard with KPI cards, grouped job type cards, "Run Now" and schedule buttons, live active job monitoring with animated progress bars, schedule dialog with presets (hourly, 6h, daily 2AM, weekly Mon 2AM), recurring schedules section, and paginated expandable job history.

**Note:** Clustering What-If scenarios from ClustersTab use the "Schedule Scenario Job" button which delegates to the job system. Active job count is shown as a badge on the sidebar Jobs nav item.

## 3i) Run tests

Run the full test suite to verify everything works:

### Backend tests (pytest)
```bash
make test              # All backend tests (~0.7s)
make test-unit         # Unit tests only (common/ modules)
make test-api          # API endpoint tests only (mock DB, no infra needed)
make test-cov          # With coverage report (api + common modules)
```

Backend tests require no running infrastructure — DB connections are mocked. Dependencies are installed via `make init` (`uv sync`).

### Frontend tests (Vitest + React Testing Library)
```bash
make ui-test           # All frontend tests (218 tests, ~1.5s)
```

Frontend tests require `make ui-init` to install npm dependencies (includes vitest, @testing-library/react, @testing-library/user-event).

### All tests
```bash
make test-all          # Backend + frontend (485+ total tests, <3s)
```

### Test structure
```
tests/
├── conftest.py            # Shared fixtures (sample DataFrames)
├── unit/
│   ├── test_metrics.py    # WAPE, bias, accuracy %
│   ├── test_constants.py  # LAG_RANGE, ROLLING_WINDOWS, thresholds
│   ├── test_domain_specs.py  # All 8 domains (parametrized)
│   ├── test_backtest_framework.py  # Timeframe generation
│   ├── test_mlflow_utils.py  # MLflow logging
│   ├── test_db.py         # DB connection parameters
│   └── test_load_dataset_postgres.py  # Forecast execution-lag loading (JOIN-based, no staging mutation) + archive
└── api/
    ├── conftest.py        # Mock pool + async httpx client
    ├── test_health.py     # Health endpoint
    ├── test_domains.py    # Domain CRUD
    ├── test_forecast_accuracy.py  # Accuracy endpoints
    ├── test_dfu_analysis.py  # DFU analysis
    ├── test_competition.py  # Champion selection
    ├── test_clusters.py   # Cluster endpoints
    ├── test_inventory.py  # Inventory endpoints
    ├── test_distinct.py   # Distinct values endpoint
    ├── test_dashboard.py  # Dashboard endpoints (kpis, alerts, top-movers, heatmap)
    └── test_jobs.py       # Job scheduler endpoints (16 tests: types, submit, list, cancel, delete, stats, schedules, pipeline)

frontend/src/
├── hooks/__tests__/       # useTheme, useUrlState, useKeyboardShortcuts, useSidebar, useGlobalFilters
├── lib/__tests__/         # formatters, export
├── api/__tests__/         # TanStack Query keys
├── context/__tests__/     # JobNotificationContext, ScenarioNotificationContext
├── components/__tests__/  # Skeleton, KeyboardShortcutHelp, EChartContainer, AppSidebar, ThemeSelector, GlobalFilterBar, WidgetGrid, AlertPanel, TopMovers, HeatmapGrid
└── tabs/__tests__/        # ExplorerTab, AccuracyTab, DfuAnalysisTab, ClustersTab, MarketIntelTab, ChatPanel, InventoryTab, WhatIfScenarios, DashboardTab, JobsTab
```

**Important:** Run `make test-all` after any code changes to catch regressions. Every new feature must include corresponding tests (see `docs/design-specs/feature31.md`).

## 4) Start API + UI
```bash
make api
```

In another terminal:
```bash
make ui-init
make ui
```

Open:
- UI: `http://127.0.0.1:5173`
- API: `http://127.0.0.1:8000`

Notes:
- `make ui-init` requires internet access for npm package download.
- UI analytics is enabled only for `sales` and `forecast` (dimensions are table-only).
- `sales` and `forecast` analytics include Item (`dmdunit`) and Location (`loc`) filters.
- Use item/location filters (exact match) to focus charts and KPIs on one item-location pair.
- Item/Location filters show autocomplete suggestions as you type.
- Column-level filters support two modes: plain text for substring search, prefix `=` for exact match.
- Column filters on text columns show typeahead suggestions as you type.
- Large tables show approximate row counts (`100,000+`) when filtered.
- A chemistry-themed loading overlay (periodic table element tile) appears during queries.
- Trend chart supports multiple measures via `Trend Measures` checkboxes.
- Forecast domain has a **Model selector** dropdown to filter by `model_id` (e.g., `external`, `arima`).
- **Chat panel** (below analytics grid) lets you ask natural language questions. Requires `OPENAI_API_KEY`.
- **Inventory tab** shows inventory KPI cards (Total On-Hand, Total On-Order, Avg Lead Time), monthly trend chart, paginated position table, and item detail drill-down. Keyboard shortcut: `6`.

## 5) Validate
```bash
make check-api
make check-db
```

Optional Iceberg path:
```bash
make spark-item
make spark-location
make spark-customer
make spark-time
make spark-dfu
make spark-sales
make spark-forecast
make trino-check-item
make trino-check-location
make trino-check-customer
make trino-check-time
make trino-check-dfu
make trino-check-sales
make trino-check-forecast
```

## 6) Stop
```bash
make down
```

## Troubleshooting

### Tests failing
- **Backend `ModuleNotFoundError`**: Run `make init` to install dev dependencies (pytest, httpx, pytest-asyncio, pytest-cov, pytest-mock)
- **Frontend tests fail to start**: Run `make ui-init` to install npm dev dependencies (vitest, @testing-library/react, @testing-library/user-event)
- **API tests fail with DB errors**: API tests mock the DB pool — ensure `tests/api/conftest.py` is present with the mock fixtures
- **`localStorage.clear is not a function`**: useTheme tests require a custom localStorage mock for jsdom — see `frontend/src/hooks/__tests__/useTheme.test.ts`

## Troubleshooting (continued)
- `make up` fails on bucket creation:
  - rerun `make minio-bucket`
- API returns DB errors:
  - verify `.env` DB values and `make up` status
- Spark fails:
  - run `make normalize-all` first
  - inspect `demand-mvp-spark` and `demand-mvp-iceberg-rest` logs
- pgvector extension not found:
  - ensure `docker-compose.yml` uses `pgvector/pgvector:pg16` (not `postgres:16`)
  - run `make down && docker volume rm demand_pg_data && make up` to rebuild
- Chat endpoint errors:
  - verify `OPENAI_API_KEY` is set in `.env`
  - verify embeddings exist: `make generate-embeddings`
  - check API logs for OpenAI rate limit or connection errors
- Forecast load fails with "missing data for column model_id":
  - re-normalize forecast: `make normalize-forecast && make load-forecast`
- **MLflow not running (Connection refused on port 5003)**:
  - MLflow is a Docker Compose service and only runs when the stack is up.
  - Start the full stack: `make up` (this starts Postgres, MinIO, MLflow, Iceberg REST, Spark, Trino).
  - Check that the MLflow container is up: `docker ps | grep mlflow` (expect `demand-mvp-mlflow`).
  - MLflow UI: `http://localhost:5003` (or the port in `MLFLOW_HOST_PORT` in `.env`).
  - Clustering still completes if MLflow is down; it skips logging and saves outputs to disk.
- Clustering fails:
  - Ensure sales data is loaded: `make load-sales`
  - Check minimum history requirement (default: 12 months)
  - Verify MLflow is running (optional): `docker ps | grep mlflow`
  - Check feature matrix output: `ls -lh data/clustering_features.csv`
  - Review cluster output: `ls -lh data/clustering/`
- Backtest fails:
  - Ensure clustering has been run first: `make cluster-all`
  - Ensure sales data is loaded: `make load-sales`
  - Install dependencies: `uv sync` (installs lightgbm, catboost, xgboost)
  - Check output: `ls -lh data/backtest/`
- Cluster assignments not updating:
  - Use `--dry-run` flag to preview changes: `make cluster-update` (with dry-run in script)
  - Verify DFU keys match: check `dfu_ck` format in assignments vs database
  - Check PostgreSQL connection: verify `.env` DB values
- Inventory normalize fails:
  - Ensure all 14 CSV files exist in `datafiles/`: `ls datafiles/Inventory_Snapshot_*.csv | wc -l` (should be 14)
  - File format: comma-separated with columns `exec_date,item,loc,lead_time,tot_oh,tot_oh_oo,mtd_sls`
  - Check for encoding issues: files should be UTF-8
- Inventory load fails:
  - Apply DDL first: `make db-apply-inventory` (creates table + indexes + materialized view)
  - Verify normalized CSV exists: `ls -lh data/inventory_clean.csv`
  - Check PostgreSQL connection: verify `.env` DB values
- Inventory tab shows no data:
  - Verify data was loaded: `docker exec demand-mvp-postgres psql -U demand -d demand_mvp -c "SELECT COUNT(*) FROM fact_inventory_snapshot"`
  - Verify aggregate view: `docker exec demand-mvp-postgres psql -U demand -d demand_mvp -c "SELECT COUNT(*) FROM agg_inventory_monthly"`
  - Refresh aggregate view: `make refresh-agg-inventory`
- Jobs tab shows no data / API calls return HTML:
  - The Vite dev server must proxy `/jobs` requests to the FastAPI backend
  - Verify `frontend/vite.config.ts` has a `/jobs` proxy entry pointing to `http://127.0.0.1:8000`
  - Restart the Vite dev server (`make ui`) after adding the proxy entry
  - Verify backend is working directly: `curl http://localhost:8000/jobs/stats`
  - If the backend returns valid JSON but the UI shows nothing, it's a missing proxy issue
- Champion selection finds no qualifying DFUs:
  - Ensure backtest predictions are loaded: `make backtest-load`
  - Check `min_dfu_rows` in `config/model_competition.yaml` (default 3); lower if DFUs have few forecast rows
  - Verify models listed in config exist in `fact_external_forecast_monthly`: `SELECT DISTINCT model_id FROM fact_external_forecast_monthly`
  - Check that `basefcst_pref` and `tothist_dmd` are not NULL for the configured models
