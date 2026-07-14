# Infrastructure & Platform Overview

> Supply Chain Command Center runs on PostgreSQL, FastAPI, and React — all on a single machine with Docker Compose — to deliver demand forecasting analytics across 500M+ rows of sales, forecast, and inventory data.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A (platform-wide) |
| **Key Files** | `docker-compose.yml`, `api/main.py`, `common/core/domain_specs.py`, `Makefile` |

---

## Problem

Supply chain teams need a single platform that combines sales history, ML forecasts, inventory snapshots, and planning workflows. Without it, teams juggle spreadsheets, disconnected BI tools, and manual data pipelines — leading to stale data, inconsistent metrics, and slow decision cycles.

## Solution

Supply Chain Command Center is a full-stack analytics platform that ingests CSV data, stores it in PostgreSQL with materialized views for fast queries, serves it through a FastAPI backend, and renders it in a React UI. MLflow tracks all model experiments. Everything runs locally via Docker Compose — no cloud infrastructure required.

## How It Works

1. Raw CSV files are normalized into clean CSVs (written to `data/staged/`) by Python scripts.
2. Clean CSVs are loaded into PostgreSQL from `data/staged/` via `psycopg` bulk copy.
3. Materialized views pre-aggregate data for O(1) KPI queries.
4. FastAPI serves analytics endpoints to the React frontend.
5. ML training and scoring jobs log parameters and metrics to MLflow.
6. The React UI provides interactive dashboards, data exploration, and planning tools.

## Tech Stack

| Layer | Technology |
|---|---|
| Database | PostgreSQL 16 (via `pgvector/pgvector:pg16`) |
| ML Tracking | MLflow v3.0.0 |
| API | Python + FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| DB Driver | psycopg v3 (sync `ConnectionPool` min=2 / max=20, plus async `AsyncConnectionPool` sibling and optional read-replica pools both sync + async) |
| Frontend | React + Vite + TypeScript |
| Styling | Tailwind CSS + shadcn/ui |
| Charts | Recharts + ECharts |
| ML / Clustering | scikit-learn, pandas, scipy |
| Job Scheduling | APScheduler 3.11 (in-process); pg-queue scaffold (`common/services/pg_queue.py`, `sql/183_create_job_queue.sql`) for cross-process work |
| Cache | Redis with in-memory fallback (`common/services/cache.py` `RedisBackend` -> `MemoryBackend`); `cached_async` decorator + `get_or_compute` single-flight |
| E2E Testing | Playwright |
| Python Packaging | uv |
| Build | Make |
| Containers | Docker Compose |

## Docker Services

| Service | Image | Port |
|---|---|---|
| postgres | pgvector/pgvector:pg16 | 5440 |
| mlflow | ghcr.io/mlflow/mlflow:v3.0.0 | 5003 |
| redis | redis:7-alpine | 6379 |
| api | build: . (project root Dockerfile) | 8000 |

Postgres tuning: `shared_buffers=512MB`, `work_mem=64MB`, `effective_cache_size=1536MB`.

## Architecture

The API layer uses domain-organized routers under `api/routers/{core,forecasting,inventory,operations,platform,intelligence}/` mounted in `api/main.py` (~80 routers, ~330-line shell). The `domains.py` router is mounted last because it has a catch-all `{domain}` path parameter. Optional API key auth is enabled when the `API_KEY` env var is set.

The central schema registry is `DomainSpec` in `common/core/domain_specs.py`, covering all 11 domains (item, location, customer, time, sku, sales, forecast, customer_demand, inventory, sourcing, purchase_order). Scripts and API endpoints are generic — they operate on any domain via `--dataset <name>` or `/domains/{domain}/*`.

## Implemented Features

| Feature | Spec | Domain |
|---|---|---|
| Data Models (dimensions + facts) | [02-data-models.md](02-data-models.md) | Foundation |
| Data Quality Engine (12 check types) | [03-data-quality.md](03-data-quality.md) | Foundation |
| Planning Date Configuration | [04-planning-date.md](04-planning-date.md) | Foundation |
| Accuracy KPIs (WAPE, bias, accuracy%) | [../02-forecasting/01-accuracy-kpis.md](../02-forecasting/01-accuracy-kpis.md) | Forecasting |
| Multi-Model Forecast Support | [../02-forecasting/02-multi-model.md](../02-forecasting/02-multi-model.md) | Forecasting |
| Backtest Framework (timeframes A-J) | [../02-forecasting/03-backtest-framework.md](../02-forecasting/03-backtest-framework.md) | Forecasting |
| Champion Model Selection (5 strategies) | [../02-forecasting/07-champion-selection.md](../02-forecasting/07-champion-selection.md) | Forecasting |
| Advanced Backtest (tuning, SHAP, recursive) | [../02-forecasting/05-advanced-backtest.md](../02-forecasting/05-advanced-backtest.md) | Forecasting |
| Forecast Pipeline Config (YAML-driven algorithm roster) | [../02-forecasting/19-forecast-pipeline-config.md](../02-forecasting/19-forecast-pipeline-config.md) | Forecasting |
| Forecast Release Control (immutable run manifests, transactional archive/promotion checksums, plus post-release common-cohort readiness) | [../02-forecasting/34-forecast-release-readiness.md](../02-forecasting/34-forecast-release-readiness.md) | Forecasting / Operations |
| Production Forecast Pipeline | [../02-forecasting/08-production-forecast.md](../02-forecasting/08-production-forecast.md) | Forecasting |
| Customer-Level Forecasting + Bottom-Up Blend (Croston/SBA-only resumable customer runs; exact completed-load/profile-refresh lineage; 18-month demand-to-sales normalization; six-origin common-cohort backtest; governed 50/50 customer/champion staging and promotion gate) | [../02-forecasting/35-customer-level-forecasting.md](../02-forecasting/35-customer-level-forecasting.md) | Forecasting |
| Bias Correction Engine | [../02-forecasting/09-bias-correction.md](../02-forecasting/09-bias-correction.md) | Forecasting |
| SKU Clustering + What-If Scenarios | [../03-demand-intelligence/01-sku-clustering.md](../03-demand-intelligence/01-sku-clustering.md) | Demand Intelligence |
| SKU Feature Engineering (Seasonality + Variability) | [../03-demand-intelligence/02-sku-feature-engineering.md](../03-demand-intelligence/02-sku-feature-engineering.md) | Demand Intelligence |
| Blended Demand Forecast | [../03-demand-intelligence/03-blended-demand.md](../03-demand-intelligence/03-blended-demand.md) | Forecasting |
| Inventory Snapshots + Backtest | [../04-inventory/01-inventory-snapshot.md](../04-inventory/01-inventory-snapshot.md) | Inventory |
| Lead Time Variability | [../04-inventory/02-demand-variability.md](../04-inventory/02-demand-variability.md) | Inventory |
| Safety Stock Engine + Simulation | [../04-inventory/03-safety-stock.md](../04-inventory/03-safety-stock.md) | Inventory |
| EOQ + Replenishment Policies + Health Score | [../04-inventory/04-replenishment.md](../04-inventory/04-replenishment.md) | Inventory |
| Exception Queue | [../04-inventory/05-exception-queue.md](../04-inventory/05-exception-queue.md) | Inventory |
| Fill Rate + Demand Signals + Intramonth | [../04-inventory/06-analytics.md](../04-inventory/06-analytics.md) | Inventory |
| ABC-XYZ + Supplier Performance | [../04-inventory/07-abc-xyz-supplier.md](../04-inventory/07-abc-xyz-supplier.md) | Inventory |
| Investment Optimization | [../04-inventory/08-investment.md](../04-inventory/08-investment.md) | Inventory |
| Multi-Echelon Safety Stock | [../04-inventory/09-multi-echelon.md](../04-inventory/09-multi-echelon.md) | Inventory |
| Replenishment Plan | [../04-inventory/10-replenishment-plan.md](../04-inventory/10-replenishment-plan.md) | Inventory |
| Inventory Reduction Opportunities + Quantile Shadow Target | [../04-inventory/15-reduction-opportunities.md](../04-inventory/15-reduction-opportunities.md) | Inventory |
| Inventory Rebalancing | [../04-inventory/11-rebalancing.md](../04-inventory/11-rebalancing.md) | Inventory |
| S&OP Cycle Management | [../05-operations/01-sop-cycle.md](../05-operations/01-sop-cycle.md) | Operations |
| Financial Planning | [../05-operations/02-financial-planning.md](../05-operations/02-financial-planning.md) | Operations |
| Event Calendar | [../05-operations/03-event-calendar.md](../05-operations/03-event-calendar.md) | Operations |
| Scenario Planning | [../05-operations/04-scenario-planning.md](../05-operations/04-scenario-planning.md) | Operations |
| AI Planning Agent | [../06-ai-platform/01-ai-planning-agent.md](../06-ai-platform/01-ai-planning-agent.md) | AI |
| SKU Chatbot (Claude Agent SDK, tiered Haiku/Sonnet/Opus, Item Analysis side chat) | [../06-ai-platform/07-sku-chatbot.md](../06-ai-platform/07-sku-chatbot.md) | AI |
| Grounded Planning Copilot (owner scope, evidence hashes, explicit provider boundary) | [../06-ai-platform/10-grounded-planning-copilot.md](../06-ai-platform/10-grounded-planning-copilot.md) | AI |
| Chatbot + Market Intelligence | [../06-ai-platform/02-market-intel.md](../06-ai-platform/02-market-intel.md) | AI |
| Control Tower | [../06-ai-platform/03-control-tower.md](../06-ai-platform/03-control-tower.md) | AI |
| Storyboard (Exception Workflow) | [../06-ai-platform/04-storyboard.md](../06-ai-platform/04-storyboard.md) | AI |
| UI Architecture + Theming | [../07-user-experience/02-ui-architecture.md](../07-user-experience/02-ui-architecture.md) | UI |
| Job Scheduler (APScheduler) | [../07-user-experience/04-job-scheduler.md](../07-user-experience/04-job-scheduler.md) | UI |
| Testing Strategy (pytest + Vitest + Playwright) | [../07-user-experience/05-testing.md](../07-user-experience/05-testing.md) | UI |
| Performance Profiling (decorator, suggestions, production-safe) | [05-performance-profiling.md](05-performance-profiling.md) | Foundation |
| Unified Pipeline Orchestrator (full/incremental modes) | — | Integration |
| RBAC + User Management | [../08-integration/02-rbac.md](../08-integration/02-rbac.md) | Integration |
| Notifications + Webhooks | [../08-integration/04-notifications.md](../08-integration/04-notifications.md) | Integration |
| FVA + ROI Measurement | [../08-integration/07-fva.md](../08-integration/07-fva.md) | Integration |
| Demand History Workbench (5 endpoints) | [../03-demand-intelligence/06-demand-history-workbench.md](../03-demand-intelligence/06-demand-history-workbench.md) | Demand Intelligence |
| Customer Analytics progressive workspace + grounded GPT assistant | [../03-demand-intelligence/07-customer-analytics.md](../03-demand-intelligence/07-customer-analytics.md) | Demand Intelligence / AI |
| AI Operations Workbench (system-safe readiness scan + unified Workflows UI) | [../06-ai-platform/09-ai-operations-workbench.md](../06-ai-platform/09-ai-operations-workbench.md) | Operations / AI |
| Mechanical lint gates for 7 unenforced rules (`scripts/ai_checks/check_unenforced_rules.sh`) | — | Foundation |
| Central MV-refresh service (`common/core/mv_refresh.py` — table→MV dependency map, DDL-diff tested, nightly `refresh_all_mvs` safety net) | — | Foundation |
| Workflow orchestration (named pipelines, boot-restored schedules, cross-stage staleness lineage + readiness checks, monthly Period Roll scoring/archive control) | [../07-user-experience/04-job-scheduler.md](../07-user-experience/04-job-scheduler.md) | Foundation |
| Async router pilot (`customer_analytics` + `inv_planning_insights` GETs use `AsyncConnectionPool` via `get_async_conn` / `get_async_read_only_conn`) | — | Foundation |
| pg-queue scaffold (`sql/183_create_job_queue.sql`, `common/services/pg_queue.py` — `enqueue_job`, `claim_next_job`, exponential-backoff requeue) | — | Foundation |
| Read-replica scaffold (`READ_REPLICA_URL` opt-in, parsed by `common/core/db.py` `get_read_replica_params()`; `api/pool.py` builds replica pools when present; `api/core.py` `get_read_only_conn` / `get_async_read_only_conn` route reads when available) | — | Foundation |
| Enforced hot-analytics read pattern (`@cached_sync` + `get_read_only_conn` on Dashboard and Forecast Accuracy GETs; checked by `scripts/ai_checks/check_unenforced_rules.sh`) | [../08-integration/03-caching.md](../08-integration/03-caching.md) | Foundation |
| Weekly partitioning DDL prep (`sql/184`, `sql/185` — REVIEW-BEFORE-RUN; `scripts/db/auto_create_partitions.py` extended with `interval="week"` ISO week math) | — | Foundation |
| Streaming ETL helpers (`common/core/sql_helpers.py` — `stream_query_in_chunks`, `read_sql_chunked`, `DEFAULT_CHUNK_SIZE`) | [02-data-models.md](02-data-models.md) | Foundation |
| ORJSON default response class (`api/main.py` uses `ORJSONResponse` for faster JSON serialization) | — | Foundation |

## Quick Start

```bash
make up          # Start Docker services (Postgres, MLflow)
make api         # FastAPI on :8000
make ui          # React dev server on :5173
make check-all   # Verify DB + API health
```

## Dependencies

- Docker and Docker Compose
- Python 3.11+ with `uv` package manager
- Node.js 18+ for frontend

## See Also

- [Data Models](02-data-models.md) — tables, keys, materialized views
- [Data Quality](03-data-quality.md) — automated validation across all domains
- [Planning Date](04-planning-date.md) — configurable date for frozen data environments
