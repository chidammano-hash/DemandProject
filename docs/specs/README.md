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
| 06 | [Algorithm Config](02-forecasting/06-algorithm-config.md) | One YAML file controls all model behavior |
| 07 | [Champion Selection](02-forecasting/07-champion-selection.md) | Pick the best model per item per month (8 strategies) |
| 08 | [Production Forecast](02-forecasting/08-production-forecast.md) | Generate real forward-looking predictions from champion models |
| 09 | [Bias Correction](02-forecasting/09-bias-correction.md) | Detect and correct systematic over/under-forecasting |
| 10 | [Forecast CI Bands](02-forecasting/10-forecast-ci-bands.md) | Confidence intervals showing the range of likely outcomes |
| 11 | [Unified Model Tuning Studio](02-forecasting/11-unified-model-tuning-v2.md) | UI-driven experiment launch, comparison, and promotion for LGBM/CatBoost/XGBoost |

| 12 | [Dual Promotion](02-forecasting/12-dual-promotion.md) | Two-stage promotion: config → results for tuning experiments |
| 13 | [Production Baseline Seeding](02-forecasting/13-production-baseline-seeding.md) | Auto-seed production baselines from completed backtests |
| 14 | [Execution Lag Filters](02-forecasting/14-execution-lag-filters.md) | Lag filter bar semantics for Algorithm and Champion experiment tabs |
| 15 | [Expert Panel: Algorithm Selection](02-forecasting/15-expert-panel-algorithm-selection.md) | 31-expert panel tests 30+ algorithms (statistical, tree, deep learning, foundation models) across demand segments; routes each DFU to its best-fit algorithm via affinity matrix optimization |
| 18 | [Chronos Foundation Models](02-forecasting/18-chronos-foundation-models.md) | Four Chronos variants (T5, Bolt, Chronos 2, Chronos 2 Enriched) — architecture, covariates, configuration, performance benchmarks |
| 19 | [Forecast Pipeline Config](02-forecasting/19-forecast-pipeline-config.md) | Master config consolidation — algorithm roster with lifecycle flags, backtest/tuning/champion/production settings in one file |
| 20 | [Bolt Hierarchical](02-forecasting/20-bolt-hierarchical.md) | Customer-level bottom-up Chronos Bolt with top-down reconciliation — uses true demand from `fact_customer_demand_monthly` to correct stockout bias |
| 21 | [Customer-Enriched Features](02-forecasting/21-customer-enriched-features.md) | 34 customer-derived features (concentration, churn, OOS, channel mix, customer attribute mix) for tree model enrichment |

**Reading order:** 01-03 (foundations) → 04-06 (engine) → 07 (selection) → 08-10 (production) → 11-14 (tuning studio) → 15 (expert panel) → 18 (foundation models) → 19 (pipeline config) → 20-21 (customer-enriched)

---

## 03 — Demand Intelligence

Understand demand patterns — descriptive analytics that inform forecasting and planning.

| # | Spec | Summary |
|---|------|---------|
| 01 | [DFU Clustering](03-demand-intelligence/01-dfu-clustering.md) | Group items by demand behavior (14 features, KMeans, What-If) |
| 02 | [Seasonality](03-demand-intelligence/02-seasonality.md) | Detect peak/trough months and seasonal strength per item |
| 03 | [Blended Demand](03-demand-intelligence/03-blended-demand.md) | Alpha-weighted blend of statistical forecast + demand signals |
| 04 | [Cluster Experimentation Studio](03-demand-intelligence/04-cluster-experimentation-studio.md) | Experiment lifecycle for testing segmentation configs (create, run, compare, promote) with cluster-aware algorithm tuning |
| 05 | [Champion Experimentation Studio](03-demand-intelligence/05-champion-experimentation-studio.md) | Experiment lifecycle for 8 champion selection strategies (expanding, rolling, decay, ensemble, meta_learner, hybrid_warmup, adaptive_ensemble, ensemble_rolling) with 2-stage promotion |

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

**Reading order:** 01 (data) → 02 (analysis) → 03-04 (targets) → 05-06 (monitoring) → 07 (segmentation) → 08-09 (optimization) → 10-11 (execution)

---

## 05 — Operations

Cross-functional planning processes that align demand, supply, and finance.

| # | Spec | Summary |
|---|------|---------|
| 01 | [S&OP Cycle](05-operations/01-sop-cycle.md) | Six-stage monthly planning process with approval workflow |
| 02 | [Financial Planning](05-operations/02-financial-planning.md) | Inventory value, carrying cost, budget utilization |
| 03 | [Event Calendar](05-operations/03-event-calendar.md) | Promotions and events that adjust demand forecasts |
| 04 | [Scenario Planning](05-operations/04-scenario-planning.md) | What-if disruption simulation with financial impact |

---

## 06 — AI & Decision Support

Automated intelligence that surfaces exceptions and recommends actions.

| # | Spec | Summary |
|---|------|---------|
| 01 | [AI Planning Agent](06-ai-platform/01-ai-planning-agent.md) | Claude-powered proactive exception work queue (not a chatbot) |
| 02 | [Chatbot & Market Intel](06-ai-platform/02-chatbot-market-intel.md) | Ask questions in English + Google/GPT market briefings |
| 03 | [Control Tower](06-ai-platform/03-control-tower.md) | Single pane of glass for supply chain health KPIs |
| 04 | [Storyboard](06-ai-platform/04-storyboard.md) | Exception cards with causal chains and decision logging |

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
| **DFU** | Demand Forecast Unit — an item + location combination (the atomic unit of planning) |
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
| **ml_cluster** | The cluster label assigned to a DFU by the KMeans pipeline |
| **model_id** | Identifies the forecasting algorithm (e.g., `external`, `lgbm_cluster`, `champion`) |
| **execution_lag** | Months between when a forecast is made and when it's evaluated |
