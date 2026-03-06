# Demand Studio — Runbook

All commands run from `mvp/demand/` unless noted.

---

## 1. Setup

### 1.1 Initialize Python environment
```bash
make init          # Create .venv, install uv, sync all dependencies
```

### 1.2 Start infrastructure
```bash
make up            # Postgres, MinIO, MLflow, Iceberg REST, Spark, Trino
make down          # Stop all services
```

### 1.3 Apply database schemas (one-time per environment)
```bash
make db-apply-sql          # Core DDL: all tables, indexes, materialized views
make db-apply-inventory    # Inventory snapshot table + indexes + agg view
make db-apply-chat         # pgvector extension + chat_embeddings table
make db-apply-jobs         # job_history + job_schedule tables + indexes
make db-apply-inv-backtest # mv_inventory_forecast_monthly bridge view (Feature 37)
make eoq-schema            # fact_eoq_targets table (IPfeature4)
make policy-schema         # dim_replenishment_policy + fact_dfu_policy_assignment (IPfeature5)
make health-schema         # fact_safety_stock_targets stub + mv_inventory_health_score view (IPfeature6)
make exceptions-schema     # fact_replenishment_exceptions table + indexes (IPfeature7)
make fill-rate-schema      # mv_fill_rate_monthly materialized view (IPfeature8)
make demand-signals-schema # fact_demand_signals table (IPfeature9)
make sim-schema            # fact_ss_simulation_results table (IPfeature10)
make abc-xyz-schema        # XYZ classification columns on dim_dfu (IPfeature11)
make supplier-perf-schema  # mv_supplier_performance materialized view (IPfeature12)
make investment-schema     # fact_inventory_investment_plan + fact_efficient_frontier (IPfeature13)
make intramonth-schema     # mv_intramonth_stockout materialized view (IPfeature14)
make control-tower-schema  # mv_control_tower_kpis materialized view (IPfeature15)
make ai-insights-schema    # ai_insights + ai_planning_memos tables (IPAIfeature1)
```

> Run once when setting up a new environment. Safe to re-run (uses `IF NOT EXISTS`).

---

## 2. Data Ingestion

### 2.1 All datasets (recommended starting point)
```bash
make normalize-all     # Normalize all 8 datasets (CSV → clean CSV)
make load-all          # Load all datasets into Postgres + refresh materialized views
```

### 2.2 Individual datasets

**Sales:**
```bash
make normalize-sales
make load-sales
```

**Forecast:**
```bash
make normalize-forecast
make load-forecast                      # TRUNCATE all + reload external (exec-lag → main, all lags → archive)
make load-forecast-replace              # Replace external only (preserves backtest/champion/ceiling)
make load-forecast-replace-no-archive   # Replace external only, skip 45M-row archive INSERT (fast)
```

**Inventory** (14 monthly snapshot CSVs):
```bash
make normalize-inventory    # Merge 14 monthly CSVs → single clean CSV
make load-inventory         # Load into Postgres + refresh agg view
make inventory-pipeline     # normalize + load + refresh (all-in-one)
```

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

**AI Planning Agent** (IPAIfeature1 — requires `ANTHROPIC_API_KEY` in `.env`):
```bash
make ai-insights-schema     # Apply DDL: ai_insights + ai_planning_memos tables (one-time)
make ai-insights-scan       # Run portfolio scan (requires anthropic API key + data loaded)
make ai-insights-dfu ITEM=<item_no> LOC=<loc>   # Analyze a single DFU
make ai-insights-all        # ai-insights-schema + ai-insights-scan (full pipeline)
```

### 2.3 Chatbot embeddings (requires `OPENAI_API_KEY` in `.env`)
```bash
make generate-embeddings
```
Parses all domain specs, generates OpenAI embeddings for schema metadata, and stores them in the `chat_embeddings` pgvector table. Re-run after adding new datasets or changing schema.

---

## 3. ML Pipelines

Pipelines run in this order: **clustering → tuning (optional) → backtesting → champion selection.**

### 3.1 DFU Clustering

Groups DFUs by historical demand patterns. Required before running per-cluster backtest strategies.

**Full pipeline (recommended):**
```bash
make cluster-all
```

**Individual steps:**
```bash
make cluster-features  # Generate time series + attribute features
make cluster-train     # Train KMeans with optimal K selection (logged to MLflow: dfu_clustering)
make cluster-label     # Assign business labels to clusters
make cluster-update    # Write cluster_assignment to dim_dfu in Postgres
```

Cluster assignments are filterable in the Data Explorer via the `cluster_assignment` column and viewable via `/domains/dfu/clusters`.

### 3.2 Hyperparameter Tuning (optional, Feature 41)

Two modes — choose based on your goal:

| Mode | When to use | How to activate (Feature 44) |
|---|---|---|
| **Global** — tune once on full history, apply via `params_file` | Production scoring, fastest path | `make tune-lgbm` → set `params_file: data/tuning/best_params_lgbm.json` in `config/algorithm_config.yaml` |
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

Set `tune_inline: true` in `config/algorithm_config.yaml` for the relevant algorithm section, then run `make backtest-lgbm` (or catboost/xgboost). This is equivalent to the old `--tune-inline` flag — each of the 10 timeframes (~20 trials × 3 folds, ~2–3× slower) tunes on only data available up to its training cutoff — no future leakage into backtest accuracy metrics.

### 3.3 Backtesting

Each backtest run writes to a **model-scoped subdirectory** so multiple models can run back-to-back without overwriting each other:
- `data/backtest/<model_id>/backtest_predictions.csv` — execution-lag predictions (→ `fact_external_forecast_monthly`)
- `data/backtest/<model_id>/backtest_predictions_all_lags.csv` — lag 0–4 archive (→ `backtest_lag_archive`)

**Architecture (Feature 44):** Tree-based models (LGBM, CatBoost, XGBoost) share `common/backtest_framework.py` via `run_tree_backtest()` — per-cluster strategy only. Each script provides only `train_and_predict_per_cluster()`. Algorithm options (recursive, shap_select, tune_inline, params_file, hyperparameters) are read from `config/algorithm_config.yaml`. Shared modules: `feature_engineering.py`, `metrics.py`, `mlflow_utils.py`, `db.py`, `constants.py`, `tuning.py`, `shap_selector.py`.

---

#### Configuration (Feature 44)

Before running any backtest, edit `config/algorithm_config.yaml` to set options for each algorithm:

```yaml
lgbm:
  recursive: false       # Set true for recursive multi-step inference (Feature 43)
  shap_select: false     # Set true for per-timeframe SHAP feature selection (Feature 42)
  shap_threshold: 0.95   # Cumulative SHAP importance threshold
  shap_top_n: null       # Exact top-N features (overrides threshold)
  shap_sample_size: 500
  tune_inline: false     # Set true for per-timeframe causal tuning (PL-002)
  params_file: null      # Set to data/tuning/best_params_lgbm.json for pre-tuned params
  # Default hyperparameters used when params_file is null and tune_inline is false
  n_estimators: 300
  learning_rate: 0.05
  # ... (see full file for all keys)
```

Same structure for `catboost:` and `xgboost:` sections.

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

#### Run all three

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

---

#### Loading predictions

```bash
make backtest-load MODEL=<model_id>   # Load one model
make backtest-load-all                # Scan data/backtest/*/ and load all models
```

`--replace` is built into `backtest-load` — it only deletes rows for the loaded `model_id`, leaving all other models untouched. Accuracy materialized views are refreshed automatically after every load.

**Available model IDs (Feature 44 — per-cluster only):**

| Framework | model_ids |
|---|---|
| LGBM | `lgbm_cluster` |
| CatBoost | `catboost_cluster` |
| XGBoost | `xgboost_cluster` |

Verify archive data:
```bash
docker exec demand-mvp-postgres psql -U demand -d demand_mvp \
  -c "SELECT model_id, lag, COUNT(*) FROM backtest_lag_archive GROUP BY 1,2 ORDER BY 1,2"
```

### 3.4 Champion Model Selection

Identifies the best model per DFU per month. All strategies enforce **strict exec-lag-aware causality** — selection for month T uses only data from months where `startdate < fcstdate`.

**Strategies:**

| Strategy | Key Idea |
|---|---|
| `expanding` *(default)* | Cumulative WAPE over all prior months, equal weight |
| `rolling` | Last N months only (`window_months`, default 6) |
| `decay` | Exponential decay — recent months weighted more (`decay_factor`, default 0.9) |
| `ensemble` | Blend top-K models by inverse-WAPE weights (`top_k`, default 3) |
| `meta_learner` | ML classifier trained on DFU features + performance stats |

**CLI:**
```bash
make champion-select          # Run with strategy from config/model_competition.yaml
make champion-train-meta      # Train meta-learner (required before using meta_learner strategy)
make champion-simulate        # Simulate all strategies, compare accuracy vs ceiling
make champion-all             # train-meta + simulate + select (full pipeline)

# Override strategy:
.venv/bin/python -m scripts.run_champion_selection --strategy rolling
```

**Config** (`config/model_competition.yaml`):
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

**Output:**
- `model_id='champion'` rows in `fact_external_forecast_monthly`
- `model_id='ceiling'` rows (oracle best-with-hindsight) in `fact_external_forecast_monthly`
- `data/champion/champion_summary.json` — champion + ceiling metrics
- `data/champion/simulation_results.json` — strategy comparison
- All accuracy materialized views refreshed automatically; running again is idempotent

**API:**
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

## 4. Launch Services

```bash
make api           # FastAPI backend on :8000
```

In a second terminal:
```bash
make ui-init       # Install npm dependencies (once, requires internet)
make ui            # React dev server on :5173
```

Open in browser:
- **UI:** `http://127.0.0.1:5173`
- **API:** `http://127.0.0.1:8000`
- **MLflow:** `http://localhost:5003`

> **Vite proxy:** Every new API path prefix must be added to `frontend/vite.config.ts` — otherwise the frontend receives HTML instead of JSON. Restart `make ui` after changes.

---

## 5. Validation

```bash
make check-db      # Table row counts in Postgres
make check-api     # Curl health + sample endpoints
make check-all     # DB + API + Trino (full)
```

**Optional Iceberg/Spark path** (only needed if using Trino query engine):
```bash
make spark-all     # Publish all datasets to Iceberg (MinIO)
make trino-check-item && make trino-check-sales  # Spot-check via Trino
```

---

## 6. Testing

```bash
make test          # All backend tests (~0.7s, no infra needed — DB is mocked)
make test-unit     # Unit tests only (common/ modules)
make test-api      # API endpoint tests only
make test-cov      # Backend tests with coverage report
make ui-test       # All frontend tests (Vitest + RTL, ~1.5s)
make test-all      # Backend + frontend (879 backend tests, 265 frontend tests, <3s)
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

---

## 7. Data Cleanup

### 7.1 Remove backtest model predictions

```bash
make backtest-list                                 # Row counts per model_id
make backtest-clean MODELS="--dry-run lgbm_cluster" # Preview before deleting
make backtest-clean MODELS="lgbm_cluster catboost_cluster"  # Delete specific models
make backtest-clean MODELS="--all-backtest"        # Delete all non-external backtest models
```

Removes rows from `fact_external_forecast_monthly` and `backtest_lag_archive`, then refreshes all 5 dependent materialized views. `--all-backtest` never deletes `model_id='external'`.

### 7.2 Remove forecasts by date range

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

## 8. Troubleshooting

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
| `make up` fails on bucket creation | Re-run `make minio-bucket` |
| API returns DB errors | Verify `.env` DB values and `make up` status |
| pgvector extension not found | Ensure `docker-compose.yml` uses `pgvector/pgvector:pg16`; rebuild: `make down && docker volume rm demand_pg_data && make up` |
| MLflow connection refused (port 5003) | Run `make up`; verify: `docker ps \| grep mlflow`; MLflow only runs when stack is up |
| Spark fails | Run `make normalize-all` first; inspect `demand-mvp-spark` and `demand-mvp-iceberg-rest` logs |

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

### Frontend / API

| Problem | Fix |
|---|---|
| Jobs tab returns HTML instead of JSON | Add `/jobs` proxy to `frontend/vite.config.ts`; restart `make ui`; verify backend: `curl http://localhost:8000/jobs/stats` |
| New API route returns HTML in UI | Add the path prefix to Vite proxy config in `frontend/vite.config.ts`; restart `make ui` |
