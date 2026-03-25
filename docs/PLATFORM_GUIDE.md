# Supply Chain Command Center

A full-stack supply chain analytics platform for demand planning and inventory optimization. Ingests historical sales and forecast data, stores it in PostgreSQL, runs ML-based forecasting pipelines, and serves an interactive React dashboard for planners and analysts.

---

## Core Architecture

| Layer | Technology |
|---|---|
| Backend API | Python + FastAPI + Uvicorn (61 mounted routers) |
| Frontend | React + Vite + TypeScript + Tailwind CSS + shadcn/ui |
| Charts | Recharts + ECharts |
| Database | PostgreSQL 16 (pgvector for embeddings) |
| ML Pipeline | scikit-learn, LightGBM, CatBoost, XGBoost, pandas |
| ML Tracking | MLflow |
| Job Scheduler | APScheduler 3.11 (BackgroundScheduler + ThreadPoolExecutor) |
| AI Agent | Claude (Anthropic) via tool_use API |
| AI / Chatbot | OpenAI GPT-4o (NL-to-SQL, market intelligence) |
| E2E Testing | Playwright |
| Build | Make, uv, Docker Compose (2 services: Postgres + MLflow) |

Data flow: raw CSVs -> normalize scripts -> PostgreSQL -> FastAPI (:8000) -> React UI (:5173)

---

## Directory Structure

```
DemandProject/
├── api/                         FastAPI backend (main.py + routers/)
│   └── routers/                 63 router files (61 mounted)
├── common/                      29 shared Python modules
├── scripts/                     Data pipeline & ML scripts (ETL, clustering, backtesting)
├── frontend/                    React + TypeScript UI
│   ├── src/tabs/                21 tab components + sub-panels
│   ├── src/components/          Shared UI components
│   ├── src/hooks/               Custom React hooks
│   ├── src/api/queries/         30 domain query modules
│   └── e2e/                     Playwright E2E tests
├── tests/                       Backend test suite (pytest: unit/ + api/)
├── sql/                         72 DDL migration files
├── config/                      YAML configs (all tunable parameters externalized)
├── docs/specs/                  Design specs (8 domains, 53 files)
├── Makefile                     All dev commands
├── docker-compose.yml           2-service infra (Postgres + MLflow)
├── CLAUDE.md                    Full project specification
└── data/input/                   Source CSVs (gitignored, ~15GB)
```

---

## Datasets

### Dimensions
- `dim_item` — from `data/input/itemdata.csv`
- `dim_location` — from `data/input/locationdata.csv`
- `dim_customer` — from `data/input/customerdata.csv`
- `dim_time` — auto-generated 2020-2035
- `dim_sku` — from `data/input/sku.txt`
- `dim_sourcing` — from `data/input/sourcing.csv` (~1.05M rows), maps item-location to supply sources (supplier-plant)

### Facts
- `fact_sales_monthly` — from `data/input/dfu_lvl2_hist.txt`, only `TYPE=1` rows
- `fact_external_forecast_monthly` — from `data/input/dfu_stat_fcst.txt`, 12-month rolling window, execution lag from `dim_sku`
- `fact_inventory_snapshot` — monthly range-partitioned, from 14 monthly CSVs (`Inventory_Snapshot_YYYY_MM.csv`, ~198M rows)
- `fact_purchase_orders` — from `data/input/purchase_orders.csv` (~5.64M rows), both open and closed POs with lead time tracking

### Forecast Loading Details
- `model_id` identifies forecasting algorithm (default `'external'`); uniqueness: `(forecast_ck, model_id)`
- **12-month filter:** only `startdate` within the last 12 months from planning date is loaded
- **Execution lag:** all external forecasts are assumed to be at execution lag; source file's `lag` and `execution_lag` fields are ignored — both are overwritten from `dim_sku` (unmatched DFUs default to 0)
- **Phase ordering:** 12-month filter first, archive loads from untouched staging, THEN staging is mutated, THEN all rows enter main table (no lag filter)
- `--replace` flag: replaces only `model_id='external'` rows, preserving backtest/champion/ceiling data
- `--skip-archive` flag: skips archive load for faster reloads

---

## Quick Start

```bash
cd DemandProject

make init              # Create .venv, install uv, sync dependencies
make up                # Start Docker services (Postgres, MLflow)
make db-apply-sql      # Apply DDL schemas
make normalize-all     # Normalize all 10 source domains
make load-all          # Load into Postgres + refresh materialized views

make api               # Start FastAPI on :8000
make ui-init           # Install npm deps
make ui                # Start React dev server on :5173
```

Open UI at `http://127.0.0.1:5173`

### Full Pipeline Setup

```bash
# Option 1: Everything (data + ML + planning + ops, ~4-6 hours)
make init && make up
make setup-all
make api &             # Start API on :8000
make ui                # Start UI on :5173

# Option 2: Data only (~30 min)
make setup-data        # normalize + load all 10 domains

# Option 3: Data + Inventory Planning, no ML (~1 hour)
make setup-planning    # data + inv planning features
```

### Setup Dependency Chain

```
setup-all
├── setup-backtest
│   ├── setup-features
│   │   ├── setup-data (normalize-all + load-all)
│   │   ├── cluster-all, seasonality-all, variability-all
│   │   ├── lt-profile-all, abc-xyz-all, demand-signals-all
│   │   └── (10 domains: item, location, customer, time, sku, sales, forecast, inventory, sourcing, purchase_order)
│   ├── backtest-all (LGBM + CatBoost + XGBoost)
│   ├── backtest-load-all + accuracy-slice-refresh
│   └── champion-all (meta-learner + simulate + select)
├── setup-inv-planning (SS, EOQ, policies, exceptions, health, fill rate, supplier, investment, rebalancing)
├── setup-demand-planning (production forecasts, projections, POs, quantile, consensus, planned orders, replenishment, bias, blended, service level, lead time, echelon)
└── setup-ops (S&OP, events, financial plan, storyboard, scenarios, DQ)
```

### Optional Pipelines

```bash
make inventory-pipeline       # Normalize 15 monthly inventory CSVs + load + refresh
make pipeline-full            # Full reload all domains + refresh MVs
make pipeline-refresh         # Incremental: detect changes, reload only deltas
make pipeline-inventory       # Full reload inventory only
make pipeline-inventory-refresh  # Incremental inventory refresh only
make setup-data               # Normalize + load all 10 domains (~30 min)
make setup-features           # Data + clustering + seasonality + variability + lead time + ABC-XYZ + demand signals
make setup-backtest           # Features + backtests + champion selection
make setup-inv-planning       # Inventory planning features (SS, EOQ, policies, exceptions, health, etc.)
make setup-demand-planning    # Demand planning features (production forecasts, projections, consensus, etc.)
make setup-ops                # Operations features (S&OP, events, financial plan, storyboard, DQ)
make setup-planning           # Data + inventory planning (no ML)
make setup-all                # Everything: data + ML + planning + ops (~4-6 hours)
make cluster-all              # Full clustering pipeline (features -> train -> label -> update)
make backtest-all             # Run LGBM + CatBoost + XGBoost backtests
make champion-all             # Train meta-learner + simulate strategies + select champions
make seasonality-all          # Detect seasonality patterns + write to dim_sku
make forecast-prod-all        # Generate production forecasts from champion models
make ai-insights-all          # Run AI portfolio scan + write insights
make dq-all                   # Data quality schema + checks
make sop-all                  # S&OP cycle setup
make rebalancing-all          # Inventory rebalancing computation
```

---

## Feature Summary

### 1. Demand Forecasting & Accuracy

Three tree-based backtest models (LightGBM, CatBoost, XGBoost) with configurable `cluster_strategy` (per_cluster or global) via `config/algorithm_config.yaml`. `ml_cluster` is always a hard feature. Expanding-window backtesting across 10 timeframes (A-J), storing lag 0-4 predictions in an archive table. Champion model selection picks the best model per DFU per month using 5 strategies (expanding, rolling, decay, ensemble, meta-learner) with exec-lag-aware causal safeguards. Production forecast inference generates versioned forward-looking forecasts with P10/P90 confidence intervals. Advanced options: recursive multi-step forecasting, SHAP-based feature selection, Bayesian hyperparameter tuning (Optuna), per-timeframe inline causal tuning.

**Hyperparameter Tuning:** Two modes -- (1) Production scoring: tune once on full history (`make tune-lgbm`), apply via `params_file` in algorithm config. (2) Honest backtesting: set `tune_inline: true` for per-timeframe causal tuning with no future leakage.

**SHAP Feature Selection:** Enable with `shap_select: true` in algorithm config. Per-timeframe SHAP computation selects features covering 95% cumulative importance. 4 read-only API endpoints under `/forecast/shap/` plus on-demand per-DFU SHAP endpoint. UI panels in Accuracy tab and Item Analysis tab.

**Recursive Forecasting:** Enable with `recursive: true`. Each predict month writes predictions back into the feature grid, giving `qty_lag_1` a real signal instead of zero for subsequent months.

**LGBM Tuning Tracker:** Systematic experiment tracking for LGBM hyperparameter optimization. Records each backtest run's config, accuracy, WAPE, and bias into `lgbm_tuning_run` with per-timeframe, per-cluster, and per-month breakdowns. Pairwise A/B comparisons compute deltas and verdicts (improved/degraded/neutral). Four ways to run experiments: (1) `make lgbm-auto-tune RUNS=N` — batch campaign with 13 predefined strategies, (2) manual single runs via `make backtest-lgbm` + `compare_backtest_runs.py --register-latest`, (3) AI Tuning Advisor chat in the UI, (4) sampled fast backtests (~3 min) via `/lgbm-tuning/sampled/run`. Per-cluster adaptive profiles (`config/cluster_tuning_profiles.yaml`) auto-apply different params for sparse, volatile, stable, and seasonal clusters. Current best: Run 8 at 71.70% accuracy (+236 bps over 69.34% baseline). See `docs/specs/02-forecasting/10-lgbm-tuning.md` for full instructions.

**AI Tuning Chat:** Interactive AI-powered chat panel within the LGBM Tuning tab. An agentic advisor (OpenAI/Anthropic, 7 tools, 20-turn loop) reviews previous runs, identifies cluster/timeframe patterns, recommends parameter changes via structured cards, and (with user confirmation) triggers new backtest runs. Results flow back into the chat for iterative tuning. DB-backed sessions (`tuning_chat_session`, `tuning_chat_message`) with 6 API endpoints under `/lgbm-tuning/chat/`. Safety: max 1 concurrent run, 5-minute cooldown, user confirmation required.

**Tuning Analysis Panels:** The LGBM Tuning tab includes 4 analysis sub-tabs: Cluster EDA (cluster demand profiles, error concentration, seasonality heatmap), Feature Lab (SHAP importance, stability, correlation, per-cluster importance), Accuracy Budget (waterfall decomposition, ABC targets, monthly trend, model comparison), and Sampled Backtest (stratified DFU sampling for fast ~3-min iteration runs). API prefixes: `/cluster-eda`, `/feature-lab`, `/accuracy-budget`, `/lgbm-tuning/sampled`.

**Unified Model Tuning Studio (Feature 46):** A production-grade, UI-driven hyperparameter tuning platform for LightGBM, CatBoost, and XGBoost. Users configure experiment parameters directly in the browser using an Experiment Builder with templates (production baseline, expert recommendations, custom), launch backtest runs as resilient jobs (subprocess isolation, PID tracking, log streaming), monitor real-time logs in the Jobs tab, compare results with execution-lag filtering (lags 0-4) including per-lag/per-cluster/per-month breakdowns, parameter diffs, and feature diffs, then promote winners to the champion pipeline via a confirmation modal. A unified API router (`/model-tuning/{model}/`) replaces the split between `lgbm-tuning.py` and `model-tuning.py` with a parametrized prefix supporting all three model types. 14 endpoints: list experiments (paginated + filtered), experiment detail, per-lag accuracy, per-cluster accuracy, per-month accuracy, experiment logs (offset-based streaming), pairwise comparison, templates, promoted run, promotion log, create experiment, promote, cancel, delete. See `docs/specs/02-forecasting/11-unified-model-tuning-v2.md` for full spec.

### 2. DFU Clustering & Segmentation

KMeans clustering groups ~112K DFUs by demand patterns using **14 core features across 6 dimensions** (volume, trend, seasonality, periodicity, intermittency, lifecycle). Feature engineering includes FFT periodicity strength, OLS seasonal R-squared, Croston ADI, scale-invariant trend slope, IQR, CAGR, recency ratio, and YoY correlation (36-month window). Optimal K via combined Silhouette + Calinski-Harabasz scoring with 5% minimum cluster size constraint (k_range [5,18]). Priority-ordered taxonomy labeling produces compound labels like `high_volume_seasonal_growing`. What-If scenario engine runs trial clusterings with custom parameters without touching production. Seasonality detection computes strength, profile, and peak/trough months per DFU. ABC-XYZ classification cross-segments DFUs by revenue volume x demand variability into a 3x3 policy matrix.

### 3. Inventory Planning (15 Sub-features, 34 Panels)

Two-column layout with 8 color-coded sidebar groups and 5 role-based view presets (All Panels, Daily Ops, Weekly Review, Monthly Planning, Executive) with progressive disclosure:

**Insights group** (7 new panels from expert recommendations):
- **Unified Action Feed**: priority-ranked items aggregating exceptions, signals, PO risks, and stockouts with severity color-coding and auto-refresh
- **Network Heatmap**: location × category DOS grid with 5 color tiers (red/orange/yellow/green/blue)
- **Segment Dashboard**: ABC-XYZ segment deep-dive with policy distribution, top exceptions, and recommended actions
- **Planning Scorecard**: effectiveness metrics with health score hero KPI, current/prior/trend/sparkline rows
- **Cash Flow Timeline**: monthly stacked BarChart for PO Committed/Planned Orders/SS Investment breakdown
- **Service Level Waterfall**: decomposing CSL into Base Forecast + SS Buffer + LT Buffer + Sensing contributions
- **Budget Optimizer**: constrained optimization with budget input and allocation detail table

**Core features** (15 sub-features across 7 groups):
- **Safety Stock Engine**: Z-score service level targets with Monte Carlo simulation, cost-benefit analysis, supplier risk adjustment, sparklines
- **EOQ Cycle Stock**: Economic Order Quantity with sensitivity analysis and MOQ guardrails
- **Replenishment Policies**: 4 policy types with auto-assignment by ABC-XYZ segment and impact preview
- **Exception Queue**: 6 exception types with severity scoring, 7-day dedup, root cause analysis, and AI annotation badges
- **Fill Rate Analytics**: order fulfillment metrics by item-location-month with trend sparklines
- **Demand Signals**: short-horizon sensing from sales velocity and inventory movement with AI tags
- **Intramonth Stockout Detection**: within-month events before end-of-month snapshot
- **Supplier Performance**: delivery reliability KPIs from receipt data with portfolio risk section
- **Capital Investment Optimization**: efficient frontier for budget-vs-service-level trade-offs
- **Portfolio Health Score**: 4-component 100-point composite score per DFU
- **Inventory Rebalancing**: cross-location transfer optimization (greedy + LP solvers) with proactive rebalancing opportunities
- **Replenishment Plan**: forward 12-month plan with CI bands from champion models
- **Blended Demand**: alpha-weighted sensing + statistical blend
- **Multi-Echelon Safety Stock**: cascade risk severity badges across echelons
- **Inventory Projection**: forward projection of inventory positions

### 4. AI Planning Agent

Proactive exception work-queue (not a chatbot) powered by Claude (`claude-opus-4-6`) via `tool_use` API. 10 tools: 9 read-only SQL lookups + `create_insight`. Generates structured insight cards with severity badge, summary, recommendation, financial impact, and causal reasoning chain. Circuit-breaker guarded (MAX_TURNS=40, TOKEN_BUDGET=100K). Async portfolio scans write to `ai_insights` table; planning memos summarize the portfolio narrative. Observability via `ai_call_log` table.

### 5. Operations & Planning

- **S&OP Cycle**: 6-stage machine (demand_review -> supply_review -> pre_sop -> executive_sop -> approved -> closed) with gap analysis and approval workflow
- **Control Tower**: cross-dimensional KPI command center with alerts and top-critical items
- **Storyboard**: exception-driven planner workflow with causal chain cards and decision logging
- **Market Intelligence**: Google Search + GPT-4o narrative briefing per item/location
- **NL-to-SQL Chatbot**: pgvector-powered schema retrieval + GPT-4o SQL generation (read-only)
- **Financial Planning**: inventory value, carrying cost, budget utilization
- **Event Calendar**: promotion and event planning with approval status
- **Scenario Planning**: disruption what-if scenarios with financial impact results

### 6. Performance Profiling

Centralized profiling for scripts, API endpoints, and full pipelines. Instruments code with `@profile_function` decorators and `profiled_section()` context managers, auto-tracks all DB queries via `wrap_connection()`, and monitors memory with `tracemalloc`. A rule-based suggestion engine detects 8 anti-patterns (slow queries, N+1, unbatched inserts, memory spikes, sequential processing, query dominance) and generates actionable recommendations.

```bash
make perf-report                         # Generate summary report from last run
make perf-script SCRIPT=compute_safety_stock  # Profile a specific pipeline script
make perf-api                            # Profile API endpoint latencies
make perf-pipeline                       # Profile end-to-end pipeline
```

**Production safety:** All profiled DB connections use `SET default_transaction_read_only = true` and always `ROLLBACK` -- zero side effects on the target database. Reports are written to `data/perf_reports/` (gitignored).

Configuration: `config/perf_config.yaml` (thresholds for all 8 suggestion rules). Spec: [docs/specs/01-foundation/05-performance-profiling.md](docs/specs/01-foundation/05-performance-profiling.md).

### 7. Platform Services

- **Data Quality**: DQEngine with 12 check types, statistical auto-fix, Self-Heal UI
- **RBAC**: JWT authentication, 4 roles (admin/planner/analyst/viewer), session management
- **Notifications**: Slack, Teams, Email, PagerDuty dispatch with user preferences
- **Collaboration**: threaded annotations on DFUs, forecasts, and exceptions
- **External Signals**: weather, economic indicators, promotional calendars integration
- **FVA & ROI**: forecast value-add waterfall, intervention tracking, financial impact measurement
- **SQL Runner**: ad-hoc read-only SQL execution from the UI with schema browser, query history, and CSV export
- **Reporting**: scheduled PDF/Excel export with configurable templates and distribution
- **API Governance**: per-role rate limiting, API versioning, usage analytics
- **Webhooks**: HMAC-SHA256 signed outbound event delivery with retry and dead-letter queue
- **Caching**: multi-tier (in-memory LRU + optional Redis) with automatic invalidation

### 8. Job Automation

APScheduler-powered engine with 8 job types across 5 groups (clustering, backtest, seasonality, champion, tuning). Per-group FIFO concurrency, cron/interval scheduling, job pipelines (sequential chaining), retry with exponential backoff. Resilient execution: subprocesses survive API restarts via `Popen(start_new_session=True)` with PID tracking, real kill via SIGTERM to process group, PID-aware startup recovery (re-adopt live processes, fail dead ones), and persistent execution log streaming to DB. Jobs tab with live progress bars, elapsed timers, persistent log panels, Kill button (2-step confirm), and cross-tab completion alerts.

### 9. UI Platform

12-tab sidebar navigation across 5 sections (Tower, Operations, Supply, Demand, System). Single "General" (Supply Chain Command Center) theme with light/dark modes. Global filter bar (brand, category, item, location, market, channel) synced across tabs via URL state. Keyboard shortcuts (1-9 tabs, `[` sidebar, `d` dark mode, `?` help). Virtualized data grid with CSV export. TanStack Query caching with lazy-loaded tab components and per-tab error boundaries. Item Analysis tab merges DFU Analysis + Inventory with 7 toggleable panels (clickable forecast lines, per-DFU SHAP panel).

---

## Data Scale

- **~198M rows** in `fact_inventory_snapshot` (15 monthly partitions, range-partitioned by `snapshot_date`)
- **~112K DFUs** across item x location combinations
- **45M+ rows** in the backtest lag archive
- **80 tables** across 10 domains: 6 dimensions + 4 facts + materialized views + planning tables
- **35+ materialized views** for O(1) KPI queries

---

## Performance Defaults

- Fact indexes on `(item_id, loc, startdate/fcstdate)` via `sql/008_perf_indexes_and_agg.sql`
- Trigram (`pg_trgm`) indexes for common `ILIKE` search fields
- Monthly materialized views: `agg_sales_monthly`, `agg_forecast_monthly`
- Accuracy slice views: `agg_accuracy_by_dim`, `agg_accuracy_lag_archive` for O(1) aggregate KPIs
- Inventory aggregate: `agg_inventory_monthly` for monthly trend aggregation
- Auto-refresh: `load-sales`, `load-forecast`, `load-inventory`, `load-all` refresh `agg_*` views; `backtest-load` refreshes accuracy slice views

---

## Testing

Full-stack automated testing (2,380 backend / 837 frontend):

```bash
cd DemandProject

make test              # All backend pytest tests (~0.7s, fully mocked DB)
make test-unit         # Unit tests only (common/ modules)
make test-api          # API endpoint tests only
make test-cov          # With coverage report

make ui-test           # All frontend tests (Vitest + React Testing Library)
make test-all          # Backend + frontend

make e2e-install       # Install Playwright browsers (one-time)
make e2e               # Run E2E smoke tests (8 suites: navigation, dashboard, accuracy, etc.)
make e2e-ui            # Playwright interactive UI mode
```

Every feature ships with tests; every removed feature removes its tests.

---

## Key Paths

| Purpose | Path |
|---|---|
| Project spec | `CLAUDE.md` |
| API entry point | `api/main.py` |
| API routers (62 files, 60 mounted) | `api/routers/` |
| Shared Python modules (29) | `common/` |
| Shared SQL helpers | `common/sql_helpers.py` |
| Domain config | `common/domain_specs.py` |
| YAML configs | `config/` |
| Pipeline scripts | `scripts/` |
| DDL migrations (80) | `sql/` |
| Frontend app | `frontend/src/App.tsx` |
| Tab components | `frontend/src/tabs/` |
| API query modules | `frontend/src/api/queries/` |
| Backend tests | `tests/` |
| Frontend tests | `frontend/src/**/__tests__/` |
| E2E tests | `frontend/e2e/tests/` |
| Design specs | `docs/specs/` (8 domains, 54 files) |
| Makefile | `Makefile` |

## Key Documentation

- [CLAUDE.md](../../CLAUDE.md) -- Complete project specification (tech stack, commands, conventions)
- [docs/specs/](../../docs/specs/) -- Feature design specifications (8 domains, 53 files)
- [ARCHITECTURE.md](ARCHITECTURE.md) -- Architecture, data flow, database schema, API routing
- [RUNBOOK.md](RUNBOOK.md) -- Setup, workflow, troubleshooting guide
