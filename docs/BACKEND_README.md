# Supply Chain Command Center

A full-stack supply chain analytics platform for demand planning and inventory optimization. Ingests historical sales and forecast data, stores it in PostgreSQL, runs ML-based forecasting pipelines, and serves an interactive React dashboard for planners and analysts.

---

## Core Architecture

| Layer | Technology |
|---|---|
| Backend API | Python + FastAPI + Uvicorn (56 mounted routers) |
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
│   └── routers/                 58 router files (56 mounted)
├── common/                      28 shared Python modules
├── scripts/                     Data pipeline & ML scripts (ETL, clustering, backtesting)
├── frontend/                    React + TypeScript UI
│   ├── src/tabs/                21 tab components + sub-panels
│   ├── src/components/          Shared UI components
│   ├── src/hooks/               Custom React hooks
│   ├── src/api/queries/         24 domain query modules
│   └── e2e/                     Playwright E2E tests
├── tests/                       Backend test suite (pytest: unit/ + api/)
├── sql/                         86 DDL migration files
├── config/                      YAML configs (all tunable parameters externalized)
├── docs/specs/                  Design specs (8 domains, 52 files)
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
- `dim_dfu` — from `data/input/dfu.txt`

### Facts
- `fact_sales_monthly` — from `data/input/dfu_lvl2_hist.txt`, only `TYPE=1` rows
- `fact_external_forecast_monthly` — from `data/input/dfu_stat_fcst.txt`, dual-path loading with execution-lag filtering
- `fact_inventory_snapshot` — from 14 monthly CSVs (`Inventory_Snapshot_YYYY_MM.csv`, ~190M rows)

### Forecast Loading Details
- `lag = month_diff(startdate, fcstdate)` with allowed range 0..4
- `model_id` identifies forecasting algorithm (default `'external'`); uniqueness: `(forecast_ck, model_id)`
- **Phase ordering:** archive loads FIRST from untouched staging (all lags 0-4), THEN staging is mutated from `dim_dfu`, THEN main table receives execution-lag rows only
- `--replace` flag: replaces only `model_id='external'` rows, preserving backtest/champion/ceiling data
- `--skip-archive` flag: skips the 45M-row archive load for faster reloads

---

## Quick Start

```bash
cd DemandProject

make init              # Create .venv, install uv, sync dependencies
make up                # Start Docker services (Postgres, MLflow)
make db-apply-sql      # Apply DDL schemas
make normalize-all     # Normalize source CSVs
make load-all          # Load into Postgres + refresh materialized views

make api               # Start FastAPI on :8000
make ui-init           # Install npm deps
make ui                # Start React dev server on :5173
```

Open UI at `http://127.0.0.1:5173`

### Optional Pipelines

```bash
make inventory-pipeline       # Normalize 14 monthly inventory CSVs + load + refresh
make cluster-all              # Full clustering pipeline (features -> train -> label -> update)
make backtest-all             # Run LGBM + CatBoost + XGBoost backtests
make champion-all             # Train meta-learner + simulate strategies + select champions
make seasonality-all          # Detect seasonality patterns + write to dim_dfu
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

### 6. Platform Services

- **Data Quality**: DQEngine with 12 check types, statistical auto-fix, Self-Heal UI, Medallion pipeline (Bronze/Silver/Gold)
- **RBAC**: JWT authentication, 4 roles (admin/planner/analyst/viewer), session management
- **Notifications**: Slack, Teams, Email, PagerDuty dispatch with user preferences
- **Collaboration**: threaded annotations on DFUs, forecasts, and exceptions
- **External Signals**: weather, economic indicators, promotional calendars integration
- **FVA & ROI**: forecast value-add waterfall, intervention tracking, financial impact measurement
- **Reporting**: scheduled PDF/Excel export with configurable templates and distribution
- **API Governance**: per-role rate limiting, API versioning, usage analytics
- **Webhooks**: HMAC-SHA256 signed outbound event delivery with retry and dead-letter queue
- **Caching**: multi-tier (in-memory LRU + optional Redis) with automatic invalidation

### 7. Job Automation

APScheduler-powered engine with 7 job types across 4 groups (clustering, backtest, seasonality, champion). Per-group FIFO concurrency, cron/interval scheduling, job pipelines (sequential chaining), retry with exponential backoff. Jobs tab with live progress bars, elapsed timers, and cross-tab completion alerts.

### 8. UI Platform

12-tab sidebar navigation across 5 sections (Tower, Operations, Supply, Demand, System). Single "General" (Supply Chain Command Center) theme with light/dark modes. Global filter bar (brand, category, item, location, market, channel) synced across tabs via URL state. Keyboard shortcuts (1-9 tabs, `[` sidebar, `d` dark mode, `?` help). Virtualized data grid with CSV export. TanStack Query caching with lazy-loaded tab components and per-tab error boundaries. Item Analysis tab merges DFU Analysis + Inventory with 7 toggleable panels (clickable forecast lines, per-DFU SHAP panel).

---

## Data Scale

- **~190M rows** in `fact_inventory_snapshot` (14 monthly snapshots)
- **~112K DFUs** across item x location combinations
- **45M+ rows** in the backtest lag archive
- **8 domain tables**: 5 dimensions + 3 facts (sales, forecast, inventory)
- **35+ materialized views and fact tables** for O(1) KPI queries

---

## Performance Defaults

- Fact indexes on `(dmdunit, loc, startdate/fcstdate)` via `sql/008_perf_indexes_and_agg.sql`
- Trigram (`pg_trgm`) indexes for common `ILIKE` search fields
- Monthly materialized views: `agg_sales_monthly`, `agg_forecast_monthly`
- Accuracy slice views: `agg_accuracy_by_dim`, `agg_accuracy_lag_archive` for O(1) aggregate KPIs
- Inventory aggregate: `agg_inventory_monthly` for monthly trend aggregation
- Auto-refresh: `load-sales`, `load-forecast`, `load-inventory`, `load-all` refresh `agg_*` views; `backtest-load` refreshes accuracy slice views

---

## Testing

Full-stack automated testing (2,213 backend / 730 frontend):

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
| API routers (58 files, 56 mounted) | `api/routers/` |
| Shared Python modules (28) | `common/` |
| Shared SQL helpers | `common/sql_helpers.py` |
| Domain config | `common/domain_specs.py` |
| YAML configs | `config/` |
| Pipeline scripts | `scripts/` |
| DDL migrations (86) | `sql/` |
| Frontend app | `frontend/src/App.tsx` |
| Tab components | `frontend/src/tabs/` |
| API query modules | `frontend/src/api/queries/` |
| Backend tests | `tests/` |
| Frontend tests | `frontend/src/**/__tests__/` |
| E2E tests | `frontend/e2e/tests/` |
| Design specs | `docs/specs/` (8 domains, 52 files) |
| Makefile | `Makefile` |

## Key Documentation

- [CLAUDE.md](../../CLAUDE.md) -- Complete project specification (tech stack, commands, conventions)
- [docs/specs/](../../docs/specs/) -- Feature design specifications (8 domains, 52 files)
- [ARCHITECTURE.md](ARCHITECTURE.md) -- Architecture, data flow, database schema, API routing
- [OPERATIONS.md](OPERATIONS.md) -- Setup, workflow, troubleshooting guide
- [CLAUDE_SKILLS_GUIDE.md](CLAUDE_SKILLS_GUIDE.md) -- Claude Code skills developer guide
