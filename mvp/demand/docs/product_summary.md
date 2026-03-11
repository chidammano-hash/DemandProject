# Demand Studio — Product Summary

## What It Is

**Demand Studio** is a full-stack supply chain analytics platform built for demand planning and inventory optimization. It ingests historical sales and forecast data, stores it in PostgreSQL, runs ML-based forecasting pipelines, and serves an interactive React dashboard for planners and analysts.

---

## Core Architecture

| Layer | Technology |
|---|---|
| Backend API | Python + FastAPI (30 modular routers) |
| Frontend | React + Vite + TypeScript + Tailwind + shadcn/ui |
| Database | PostgreSQL 16 (~190M+ rows inventory snapshots) |
| ML Pipeline | scikit-learn, LightGBM, CatBoost, XGBoost, pandas |
| ML Tracking | MLflow |
| Job Scheduler | APScheduler 3.11 |
| AI Agent | Claude (Anthropic) via tool_use API |

Data flows: raw CSVs → normalize scripts → PostgreSQL → FastAPI → React UI at `:5173`

---

## Feature Areas

### 1. Demand Forecasting & Accuracy
- Imports external forecasts and computes accuracy metrics: **WAPE, Bias, MAPE** at multiple dimensional slices (item, location, brand, channel)
- **Three tree-based backtest models**: LightGBM, CatBoost, XGBoost — configurable `cluster_strategy` (per_cluster or global) via `algorithm_config.yaml`; `ml_cluster` always a hard feature
- **Expanding-window backtesting** across 10 timeframes (A–J), storing all lag 0–4 predictions in an archive table
- **Champion model selection**: 5 strategies (expanding, rolling, decay, ensemble, meta-learner) — picks the best model per DFU per month with causal safeguards
- **Production forecast inference**: full pipeline generates versioned forward-looking forecasts from champion models
- Advanced options: recursive multi-step forecasting, SHAP-based feature selection, Bayesian hyperparameter tuning (Optuna), per-timeframe inline causal tuning

### 2. DFU Clustering & Segmentation
- **KMeans clustering** groups ~112K Demand Forecast Units (DFUs) by demand patterns using **14 core features across 6 dimensions** (volume, trend, seasonality, periodicity, intermittency, lifecycle) into labeled segments
- Feature engineering: FFT periodicity strength, OLS seasonal R-squared, Croston ADI, scale-invariant trend slope, IQR, CAGR, recency ratio, YoY correlation (36-month window)
- Optimal K via **combined Silhouette + Calinski-Harabasz scoring** (0.5*sil + 0.5*CH) with hard **5% minimum cluster size** constraint (k_range [5,18])
- **Priority-ordered taxonomy labeling**: Intermittency -> Periodicity -> Seasonality -> Trend -> Volatility -> Volume (5 tiers); compound labels like `high_volume_seasonal_growing`
- **What-If scenario engine**: run trial clusterings with custom parameters without touching production; promote winning scenarios; background execution with progress tracking
- **Seasonality detection**: computes strength, profile label, peak/trough months per DFU and stores them in `dim_dfu`
- **ABC-XYZ classification**: cross-segments DFUs by revenue volume (ABC) × demand variability (XYZ) into a 3×3 policy matrix

### 3. Inventory Planning (14 sub-features)
- **Safety Stock Engine**: Z-score service level targets with Monte Carlo simulation (configurable confidence curves)
- **EOQ Cycle Stock**: Economic Order Quantity with sensitivity analysis and MOQ guardrails
- **Replenishment Policies**: 4 policy types (ROP, periodic, min-max, custom) with auto-assignment rules by ABC-XYZ segment
- **Exception Queue**: 6 exception types (DOS breach, excess stock, stockout risk, etc.) with severity scoring and deduplication
- **Fill Rate Analytics**: order fulfillment metrics aggregated by item-location-month
- **Demand Signals**: short-horizon demand sensing from sales velocity and inventory movement
- **Intramonth Stockout Detection**: within-month stockout events before end-of-month
- **Supplier Performance**: delivery reliability KPIs from receipt data
- **Capital Investment Optimization**: efficient frontier computation for budget-vs-service-level trade-offs
- **Portfolio Health Score**: 4-component 100-point health score per DFU

### 4. AI Planning Agent
- A **proactive exception work-queue** (not a chatbot) powered by Claude (`claude-opus-4-6`) via the `tool_use` API
- **10 tools**: 9 read-only SQL lookups (forecast performance, inventory trend, EOQ context, similar DFUs, stockout history, etc.) + `create_insight` (writes to DB)
- Generates structured insight cards: severity badge, 1-sentence summary, specific recommendation, financial impact estimate, causal reasoning chain
- Circuit-breaker guarded: MAX_TURNS=40, TOKEN_BUDGET=100K per run
- Async portfolio scans write insights to `ai_insights` table; planning memos summarize the portfolio narrative
- Observability: per-turn token usage and latency logged to `ai_call_log` table

### 5. Operations & Planning Tabs
- **S&OP Cycle**: stage machine with approval workflow (Sensing → Consensus → Executive → Approved Plan)
- **Control Tower**: cross-dimensional KPI command center — active alerts, top-critical items, trend aggregates
- **Storyboard**: exception-driven planner workflow with causal chain cards and decision logging
- **Market Intelligence**: Google Search + GPT-4o narrative briefing per item/location pair
- **NL→SQL Chatbot**: pgvector-powered schema retrieval + GPT-4o SQL generation (read-only, 5s timeout)

### 6. Automation & Jobs
- **APScheduler-powered job engine**: 7 job types across 4 groups (clustering, backtest, seasonality, champion)
- Per-group FIFO concurrency: one active job per group, others queue automatically
- Scheduling: cron/interval presets, job pipelines (sequential chaining), retry with exponential backoff
- **Jobs tab**: live progress bars, elapsed timers, schedule dialog, expandable history, cross-tab completion alerts

### 7. UI Platform
- **14-tab sidebar** navigation: Dashboard, Data Explorer, Accuracy, DFU Analysis, Clusters, Market Intel, Inventory, Inv. Backtest, Inv. Planning, Control Tower, AI Planner, Storyboard, S&OP, Jobs
- **Inventory Planning tab**: two-column layout — fixed 220px grouped sidebar navigation (7 color-coded groups: Daily Operations, Optimize, Analytics, Planning, Sensing, Strategic, Supply) on the left, scrollable panel body with a fixed header bar on the right; 26 panel components unchanged
- **Global filter bar**: brand, category, item, location, market, channel — synced across tabs via URL state
- Light/dark mode, keyboard shortcuts (1–9 tab switch, `[` sidebar, `d` dark mode, `?` help), virtualized data grid with CSV export
- TanStack Query caching (stale-while-revalidate), lazy-loaded tab components with per-tab error boundaries

---

## Data Scale

- **~190M rows** in `fact_inventory_snapshot` (14 monthly snapshots)
- **~112K DFUs** across item × location combinations
- **45M+ rows** in the backtest lag archive
- **8 domain tables**: 5 dimensions (item, location, customer, time, DFU) + 3 facts (sales, forecast, inventory)
- **30+ materialized views** for O(1) KPI query performance

---

## Testing

- **1,552 backend tests** (pytest, fully mocked DB — no infrastructure needed, ~0.7s)
- **442 frontend tests** (Vitest + React Testing Library)
- Every feature ships with tests; every removed feature removes its tests
