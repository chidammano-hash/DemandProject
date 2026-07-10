# Supply Chain Command Center — Design Specifications

> A complete reference for every feature in the platform, organized by business domain and ordered by dependency.

## How to Read These Specs

**New to the platform?** Start with [01-foundation/01-infrastructure](01-foundation/01-infrastructure.md) for the tech stack, then [01-foundation/02-data-models](01-foundation/02-data-models.md) for the data dictionary. From there, follow the domain that interests you.

**Looking for a specific feature?** Use the index below. Each spec is self-contained with a Problem, Solution, and How It Works section.

**Building a new feature?** Check Dependencies at the bottom of the relevant spec to understand what must exist first.

---

## Domain Map

```
┌─────────────────────────────────────────────────────────┐
│  01 Foundation     Data models, quality, infrastructure  │
│        ↓                                                │
│  02 Forecasting    Predict future demand                │
│        ↓                                                │
│  03 Demand Intel   Classify & understand patterns       │
│        ↓                                                │
│  04 Inventory      Convert forecasts to stock decisions │
│        ↓                                                │
│  05 Operations     S&OP, finance, events, scenarios     │
│        ↓                                                │
│  06 AI Platform    Automated insights & control tower   │
│        ↓                                                │
│  07 UX             UI, theming, jobs, testing           │
│        ↓                                                │
│  08 Integration    Auth, notifications, webhooks, APIs  │
└─────────────────────────────────────────────────────────┘
```

---

## 01 — Foundation

The data layer everything else depends on.

| # | Spec | Summary |
|---|------|---------|
| 01 | [Infrastructure](01-foundation/01-infrastructure.md) | Tech stack, Docker services, implemented features index |
| 02 | [Data Models](01-foundation/02-data-models.md) | 5 dimension + 5 fact tables, materialized views, ERD |
| 03 | [Data Quality](01-foundation/03-data-quality.md) | 12 check types, Self-Heal auto-fix, scoring dashboard |
| 04 | [Planning Date](01-foundation/04-planning-date.md) | Configurable date replacing `date.today()` for reproducibility |
| 05 | [Performance Profiling](01-foundation/05-performance-profiling.md) | Centralized profiling, suggestion engine, production-safe |
| 06 | [Execution Lag](01-foundation/06-execution-lag.md) | DFU-level planning horizon, forecast lag vs execution lag, lag filter semantics |
| 07 | [Customer Demand Fact](01-foundation/07-customer-demand-fact.md) | Customer-level demand fact table, monthly range-partitioned by startdate |
| 09 | [Databricks + Lakebase Migration](01-foundation/09-databricks-lakebase-migration.md) | **Proposed** — port to Databricks: Lakebase (Postgres) DB, Delta-synced source tables, synced-vs-native split, token auth |

---

## 02 — Forecasting

Predict future demand using ML models, then select the best forecast per item.

| # | Spec | Summary |
|---|------|---------|
| 01 | [Accuracy KPIs](02-forecasting/01-accuracy-kpis.md) | WAPE, bias, accuracy% — how we measure forecast quality |
| 02 | [Multi-Model](02-forecasting/02-multi-model.md) | model_id column lets multiple algorithms coexist |
| 03 | [Backtest Framework](02-forecasting/03-backtest-framework.md) | Expanding-window evaluation across 10 timeframes |
| 04 | [Tree Models](02-forecasting/04-tree-models.md) | LGBM + CatBoost + XGBoost implementations |
| 05 | [Advanced Backtest](02-forecasting/05-advanced-backtest.md) | Hyperparameter tuning, SHAP selection, recursive forecasting |
| 07 | [Champion Selection](02-forecasting/07-champion-selection.md) | Pick the best model per item per month (8 strategies) |
| 08 | [Production Forecast](02-forecasting/08-production-forecast.md) | Generate real forward-looking predictions from champion models |
| 09 | [Bias Correction](02-forecasting/09-bias-correction.md) | Detect and correct systematic over/under-forecasting |
| 10 | [Forecast CI Bands](02-forecasting/10-forecast-ci-bands.md) | Confidence intervals showing the range of likely outcomes |
| 11 | [Unified Model Tuning Studio](02-forecasting/11-unified-model-tuning-v2.md) | UI-driven experiment launch, comparison, and promotion for LGBM/CatBoost/XGBoost; also covers the systematic A/B testing and run registry for tree model hyperparameter experiments |
| 12 | [Dual Promotion](02-forecasting/12-dual-promotion.md) | Two-stage promotion: config → results for tuning experiments |
| 13 | [Production Baseline Seeding](02-forecasting/13-production-baseline-seeding.md) | Auto-seed production baselines from completed backtests |
| 14 | [Execution Lag Filters](02-forecasting/14-execution-lag-filters.md) | Lag filter bar semantics for Algorithm and Champion experiment tabs |
| 15 | [Expert Panel: Algorithm Selection](02-forecasting/15-expert-panel-algorithm-selection.md) | 31-expert panel tests 30+ algorithms (statistical, tree, deep learning, foundation models) across demand segments; routes each DFU to its best-fit algorithm via affinity matrix optimization |
| 16 | [Expert System Backtest](02-forecasting/16-expert-system-backtest.md) | Segment→algorithm routing backtest (`expsys_accuracy` router) |
| 17 | [External ML Forecast Loading](02-forecasting/17-ext-ml-forecast-load.md) | ETL to load externally-generated ML forecasts into the platform |
| 18 | [Chronos 2 Enriched Foundation Model](02-forecasting/18-chronos-foundation-models.md) | The surviving Chronos 2 Enriched foundation model (31 covariates) — architecture, covariates, configuration, performance benchmarks |
| 19 | [Forecast Pipeline Config](02-forecasting/19-forecast-pipeline-config.md) | Master config consolidation — algorithm roster with lifecycle flags, backtest/tuning/champion/production settings in one file |
| 22 | [Expert Panel Flow](02-forecasting/22-expert-panel-flow.md) | Mermaid process flow diagram for the advanced expert panel algorithm selection pipeline |
| 23 | [LGBM Accuracy Tuning](02-forecasting/23-lgbm-accuracy-tuning.md) | Systematic LGBM accuracy improvement (59% -> 68%): data fixes, per-cluster SHAP, MAE objective, tuning profiles, intermittent routing, per-cluster Bayesian tuning pipeline |
| 24 | [Candidate Forecast & Promotion](02-forecasting/24-candidate-forecast-promotion.md) | Production forecast promotion workflow (`fact_production_forecast_staging` → `fact_production_forecast`); champion promotions route per-DFU via the promoted champion experiment's winners CSV |
| 26 | [Forecast Pipeline Operational Reference](02-forecasting/26-forecast-pipeline-operational-reference.md) | Comprehensive operational reference: 7-stage quick workflow + per-stage detail, dependency DAG, configuration reference, database reference, experimentation workflows, expert panel testing, and gap analysis |
| 27 | [AI Champion Forecast](02-forecasting/27-ai-champion-forecast.md) | Interactive, single-DFU AI adjuster: from the Item Analysis tab an LLM (Ollama/Google/Anthropic/OpenAI) nudges the promoted champion forecast forward and writes a new `model_id='ai_champion'` (preview→save, no batch, no grading) |
| 28 | [Feature Selection Pipeline](02-forecasting/28-feature-selection-pipeline.md) | Multi-stage per-timeframe feature selection (duplicate / near-zero-variance / correlation / cumulative SHAP) |
| 29 | [Consensus Plan & Overrides](02-forecasting/29-consensus-plan-overrides.md) | Planner override queue, consensus merge, decision-ledger audit on approve |
| 30 | [Champion Strategy Sweep](02-forecasting/30-champion-strategy-sweep.md) | Tournament grid over champion-selection strategies (`champion_sweep`, sql/192); winners feed champion experiments |
| 32 | [Lag-Decomposed Accuracy Leaderboard](02-forecasting/32-lag-decomposed-accuracy-leaderboard.md) | Per-lag model rankings from `agg_accuracy_lag_archive` |
| 33 | [Forecast Snapshot Archive & Live FVA](02-forecasting/33-forecast-snapshot-archive-fva.md) | Monthly as-of archive of the champion plus three frozen, WAPE-ranked contenders (`record_month`, lags 0-5), separately gated staging cleanup, live snapshot FVA |
| 34 | [Forecast Release Readiness](02-forecasting/34-forecast-release-readiness.md) | Common-cohort quality, lineage, freshness, coverage, and archive readiness contract for planner use |

**Reading order:** 01-03 (foundations) → 04-05 (engine) → 07 (selection) → 08-10 (production) → 11-14 (tuning studio) → 15 (expert panel) → 18 (foundation model) → 19 (pipeline config) → 22 (expert panel flow) → 23 (LGBM accuracy tuning) → 26 (operational reference)

---

## 03 — Demand Intelligence

Understand demand patterns — descriptive analytics that inform forecasting and planning.

| # | Spec | Summary |
|---|------|---------|
| 01 | [SKU Clustering & Experimentation Studio](03-demand-intelligence/01-sku-clustering.md) | Group SKUs by demand behavior (14 features, KMeans, What-If), experiment lifecycle (create, run, compare, promote), cluster-aware algorithm tuning |
| 02 | [SKU Feature Engineering](03-demand-intelligence/02-sku-feature-engineering.md) | 34 time-series features (volume, trend, seasonality, periodicity, intermittency, lifecycle, statistical), derived classifications, `dim_sku` persistence, API + UI explorer |
| 03 | [Blended Demand](03-demand-intelligence/03-blended-demand.md) | Alpha-weighted blend of statistical forecast + demand signals |
| 05 | [Champion Experimentation Studio](03-demand-intelligence/05-champion-experimentation-studio.md) | Experiment lifecycle for 8 champion selection strategies (expanding, rolling, decay, ensemble, meta_learner, hybrid_warmup, adaptive_ensemble, ensemble_rolling) with 2-stage promotion |
| 06 | [Demand History Workbench](03-demand-intelligence/06-demand-history-workbench.md) | 5-endpoint customer demand analysis API — reference panel, proportional decomposition, hierarchical drill-down, cross-reference matrix |
| 07 | [Customer Analytics](03-demand-intelligence/07-customer-analytics.md) | Five-view customer intelligence workspace with demand, service, behavior visuals, and grounded GPT Q&A |

---

## 04 — Inventory Planning

Convert demand forecasts into inventory decisions.

| # | Spec | Summary |
|---|------|---------|
| 01 | [Inventory Snapshot](04-inventory/01-inventory-snapshot.md) | 190M rows of daily inventory, monthly aggregation |
| 02 | [Demand Variability](04-inventory/02-demand-variability.md) | CV, MAD, volatility profiles + lead time variability |
| 03 | [Safety Stock](04-inventory/03-safety-stock.md) | Z-score buffers by ABC class + Monte Carlo simulation |
| 04 | [Replenishment](04-inventory/04-replenishment.md) | EOQ, 4 policy types, health scores |
| 05 | [Exception Queue](04-inventory/05-exception-queue.md) | 6 exception types, severity ranking, planner work queue |
| 06 | [Analytics](04-inventory/06-analytics.md) | Fill rate, demand signals, intramonth stockout detection |
| 07 | [ABC-XYZ & Supplier](04-inventory/07-abc-xyz-supplier.md) | 9-cell segmentation matrix + supplier reliability |
| 08 | [Investment](04-inventory/08-investment.md) | Efficient frontier optimization for inventory budget |
| 09 | [Multi-Echelon](04-inventory/09-multi-echelon.md) | 2-echelon safety stock with risk pooling |
| 10 | [Replenishment Plan](04-inventory/10-replenishment-plan.md) | Forward order schedule from CI bands + policies |
| 11 | [Rebalancing](04-inventory/11-rebalancing.md) | Cross-location transfer optimization |
| 12 | [Service-Level Unification](04-inventory/12-service-level-unification.md) | Single source of truth for SL targets across SS, fill rate, S&OP |
| 13 | [Integrated Targets](04-inventory/13-integrated-targets.md) | Unified SS / ROP / EOQ targets per DFU for insights feed |
| 14 | [Algorithm Comparison](04-inventory/14-algorithm-comparison.md) | Side-by-side inventory algorithm backtest comparison |

**Reading order:** 01 (data) → 02 (analysis) → 03-04 (targets) → 05-06 (monitoring) → 07 (segmentation) → 08-09 (optimization) → 10-11 (execution) → 12-14 (SL unification + targets)

---

## 05 — Operations

Cross-functional planning processes that align demand, supply, and finance.

| # | Spec | Summary |
|---|------|---------|
| 01 | [S&OP Cycle](05-operations/01-sop-cycle.md) | Six-stage monthly planning process with approval workflow |
| 02 | [Financial Planning](05-operations/02-financial-planning.md) | Inventory value, carrying cost, budget utilization |
| 03 | [Event Calendar](05-operations/03-event-calendar.md) | Promotions and events that adjust demand forecasts |
| 04 | [Scenario Planning](05-operations/04-scenario-planning.md) | What-if disruption simulation with financial impact |
| 05 | [Working Capital Analytics](05-operations/05-working-capital-analytics.md) | DIO, DSO, DPO, cash-to-cash cycle, inventory turns |

---

## 06 — AI & Decision Support

Automated intelligence that surfaces exceptions and recommends actions.

| # | Spec | Summary |
|---|------|---------|
| 01 | [AI Planning Agent](06-ai-platform/01-ai-planning-agent.md) | Claude-powered proactive exception work queue (not a chatbot) |
| 02 | [Market Intel](06-ai-platform/02-market-intel.md) | Google search + GPT-4o market briefings for item-location pairs |
| 03 | [Control Tower](06-ai-platform/03-control-tower.md) | Single pane of glass for supply chain health KPIs |
| 04 | [Storyboard](06-ai-platform/04-storyboard.md) | Exception cards with causal chains and decision logging |
| 05 | [Decision Ledger + Policy](06-ai-platform/05-decision-ledger-and-policy.md) | Hash-chained append-only AI decision ledger (**Partial** — ledger shipped; policy engine not yet wired) |
| 06 | [Forecast Explain API](06-ai-platform/06-explain-api.md) | SHAP-based forecast explanation per DFU |
| 07 | [SKU Chatbot](06-ai-platform/07-sku-chatbot.md) | Conversational per-SKU assistant on the Claude Agent SDK; tiered Haiku/Sonnet/Opus routing; standalone tab + Item Analysis side chat; best-effort persistence (sql/196). Subscription auth in Claude Code, API key/Bedrock/Vertex standalone (**Partial** — Phases 1+3 shipped; needs `uv sync --extra agent` for the live path) |
| 08 | [AI Integration Scan Orchestrator](06-ai-platform/08-integration-scan-orchestrator.md) | AI-assisted Scan Now planner that asks clarifying questions and returns the safest load sequence |
| 09 | [AI Operations Workbench](06-ai-platform/09-ai-operations-workbench.md) | Unified Workflows command center that scans integration, clustering, model, forecast, archive, and inventory readiness; AI verifies a system-safe sequence and asks only decision-changing questions |

---

## 07 — User Experience

The React UI, automation, and testing infrastructure.

| # | Spec | Summary |
|---|------|---------|
| 01 | [Data Explorer](07-user-experience/01-data-explorer.md) | Browse 60M+ rows with type-aware filters and CSV export |
| 02 | [UI Architecture](07-user-experience/02-ui-architecture.md) | React + Vite + TanStack Query shell, sidebar, global filters |
| 03 | [Theming](07-user-experience/03-theming.md) | Supply Chain Command Center theme with light/dark modes |
| 04 | [Job Scheduler](07-user-experience/04-job-scheduler.md) | APScheduler engine: 7 job types, queuing, cron scheduling |
| 05 | [Testing](07-user-experience/05-testing.md) | pytest + Vitest + Playwright testing pyramid |
| 06 | [Backtest Cleanup](07-user-experience/06-backtest-cleanup.md) | Model and date-range forecast deletion utilities |
| 07 | [Developer Tools](07-user-experience/07-developer-tools.md) | Claude Code skills, agents, and slash commands installed in `.claude/` |

---

## 08 — Integration

Connect Supply Chain Command Center to the rest of the enterprise.

| # | Spec | Summary |
|---|------|---------|
| 01 | [Integration Architecture](08-integration/01-integration-architecture.md) | 4 integration vectors: notifications, REST, cloud, ERP |
| 02 | [RBAC](08-integration/02-rbac.md) | Role-based access control (admin, planner, analyst, viewer) |
| 03 | [Caching](08-integration/03-caching.md) | In-memory LRU cache with TTL and namespace isolation |
| 04 | [Notifications](08-integration/04-notifications.md) | Multi-channel dispatch: email, Slack, Teams, PagerDuty |
| 05 | [Collaboration](08-integration/05-collaboration.md) | Annotation threads and shared views |
| 06 | [External Signals](08-integration/06-external-signals.md) | Ingest POS data, weather, economic indicators |
| 07 | [FVA](08-integration/07-fva.md) | Forecast Value Add — measure if human overrides help |
| 08 | [Reporting](08-integration/08-reporting.md) | Scheduled report generation and delivery |
| 09 | [API Governance](08-integration/09-api-governance.md) | Rate limiting, versioning, usage tracking |
| 10 | [Webhooks](08-integration/10-webhooks.md) | Event-driven outbound notifications with retry |

---

## Glossary

| Term | Definition |
|------|-----------|
| **SKU** | An item + location combination (the atomic unit of planning). Also referred to as DFU (Demand Forecast Unit) in legacy contexts. |
| **WAPE** | Weighted Absolute Percentage Error — `SUM(\|F-A\|) / \|SUM(A)\|` |
| **Bias** | `(SUM(Forecast) / SUM(Actual)) - 1` — positive means over-forecasting |
| **DOS** | Days of Supply — how many days current inventory will last at current demand |
| **WOC** | Weeks of Cover — DOS expressed in weeks |
| **EOQ** | Economic Order Quantity — the order size that minimizes total inventory cost |
| **ROP** | Reorder Point — the inventory level that triggers a new order |
| **ABC** | Volume segmentation — A (top 80% revenue), B (next 15%), C (bottom 5%) |
| **XYZ** | Variability segmentation — X (stable), Y (moderate), Z (erratic) |
| **S&OP** | Sales & Operations Planning — monthly cross-functional alignment process |
| **FVA** | Forecast Value Add — measures whether manual adjustments improve accuracy |
| **SHAP** | SHapley Additive exPlanations — which features drove a model's prediction |
| **CI** | Confidence Interval — range of likely forecast outcomes (e.g., 80% CI) |
| **ml_cluster** | The cluster label assigned to an SKU by the KMeans pipeline |
| **model_id** | Identifies the forecasting algorithm (e.g., `external`, `lgbm_cluster`, `champion`) |
| **execution_lag** | Months between when a forecast is made and when it's evaluated |
