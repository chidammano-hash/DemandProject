# Infrastructure & Platform Overview

> Supply Chain Command Center runs on PostgreSQL, FastAPI, and React — all on a single machine with Docker Compose — to deliver demand forecasting analytics across 500M+ rows of sales, forecast, and inventory data.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A (platform-wide) |
| **Key Files** | `docker-compose.yml`, `api/main.py`, `common/domain_specs.py`, `Makefile` |

---

## Problem

Supply chain teams need a single platform that combines sales history, ML forecasts, inventory snapshots, and planning workflows. Without it, teams juggle spreadsheets, disconnected BI tools, and manual data pipelines — leading to stale data, inconsistent metrics, and slow decision cycles.

## Solution

Supply Chain Command Center is a full-stack analytics platform that ingests CSV data, stores it in PostgreSQL with materialized views for fast queries, serves it through a FastAPI backend, and renders it in a React UI. MLflow tracks all model experiments. Everything runs locally via Docker Compose — no cloud infrastructure required.

## How It Works

1. Raw CSV files are normalized into clean CSVs by Python scripts.
2. Clean CSVs are loaded into PostgreSQL via `psycopg` bulk copy.
3. Materialized views pre-aggregate data for O(1) KPI queries.
4. FastAPI serves analytics endpoints to the React frontend.
5. ML training and scoring jobs log parameters and metrics to MLflow.
6. The React UI provides interactive dashboards, data exploration, and planning tools.

## Tech Stack

| Layer | Technology |
|---|---|
| Database | PostgreSQL 16 (via `pgvector/pgvector:pg16`) |
| ML Tracking | MLflow v2.16.2 |
| API | Python + FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| DB Driver | psycopg v3 (connection pool: min=2, max=10) |
| Frontend | React + Vite + TypeScript |
| Styling | Tailwind CSS + shadcn/ui |
| Charts | Recharts + ECharts |
| ML / Clustering | scikit-learn, pandas, scipy |
| Job Scheduling | APScheduler 3.11 |
| E2E Testing | Playwright |
| Python Packaging | uv |
| Build | Make |
| Containers | Docker Compose |

## Docker Services

| Service | Image | Port |
|---|---|---|
| postgres | pgvector/pgvector:pg16 | 5440 |
| mlflow | ghcr.io/mlflow/mlflow:v2.16.2 | 5003 |

Postgres tuning: `shared_buffers=512MB`, `work_mem=64MB`, `effective_cache_size=1536MB`.

## Architecture

The API layer uses 54 modular routers mounted in `api/main.py`. The `domains.py` router is mounted last because it has a catch-all `{domain}` path parameter. All route handlers live in router modules — `main.py` is a ~149-line shell. Optional API key auth is enabled when the `API_KEY` env var is set.

The central schema registry is `DomainSpec` in `common/domain_specs.py`, covering all 8 domains (item, location, customer, time, dfu, sales, forecast, inventory). Scripts and API endpoints are generic — they operate on any domain via `--dataset <name>` or `/domains/{domain}/*`.

## Implemented Features

| Feature | Spec | Domain |
|---|---|---|
| Data Models (dimensions + facts) | [02-data-models.md](02-data-models.md) | Foundation |
| Data Quality Engine (12 check types) | [03-data-quality.md](03-data-quality.md) | Foundation |
| Planning Date Configuration | [04-planning-date.md](04-planning-date.md) | Foundation |
| Accuracy KPIs (WAPE, bias, accuracy%) | [../02-forecasting/01-accuracy-kpis.md](../02-forecasting/01-accuracy-kpis.md) | Forecasting |
| Multi-Model Forecast Support | [../02-forecasting/02-multi-model.md](../02-forecasting/02-multi-model.md) | Forecasting |
| Backtest Framework (timeframes A-J) | [../02-forecasting/03-backtest-framework.md](../02-forecasting/03-backtest-framework.md) | Forecasting |
| Tree Model Implementations (LGBM, CatBoost, XGBoost) | [../02-forecasting/04-tree-models.md](../02-forecasting/04-tree-models.md) | Forecasting |
| Champion Model Selection (5 strategies) | [../02-forecasting/05-champion-selection.md](../02-forecasting/05-champion-selection.md) | Forecasting |
| Advanced Backtest (tuning, SHAP, recursive) | [../02-forecasting/06-advanced-backtest.md](../02-forecasting/06-advanced-backtest.md) | Forecasting |
| Algorithm Config (YAML-driven) | [../02-forecasting/07-algorithm-config.md](../02-forecasting/07-algorithm-config.md) | Forecasting |
| Production Forecast Pipeline | [../02-forecasting/08-production-forecast.md](../02-forecasting/08-production-forecast.md) | Forecasting |
| Bias Correction Engine | [../02-forecasting/09-bias-correction.md](../02-forecasting/09-bias-correction.md) | Forecasting |
| DFU Clustering + What-If Scenarios | [../02-forecasting/10-clustering.md](../02-forecasting/10-clustering.md) | Forecasting |
| Seasonality Detection | [../02-forecasting/11-seasonality.md](../02-forecasting/11-seasonality.md) | Forecasting |
| Blended Demand Forecast | [../02-forecasting/12-blended-demand.md](../02-forecasting/12-blended-demand.md) | Forecasting |
| Inventory Snapshots + Backtest | [../03-inventory/01-inventory-snapshot.md](../03-inventory/01-inventory-snapshot.md) | Inventory |
| Demand Variability + Lead Time | [../03-inventory/02-variability.md](../03-inventory/02-variability.md) | Inventory |
| Safety Stock Engine + Simulation | [../03-inventory/03-safety-stock.md](../03-inventory/03-safety-stock.md) | Inventory |
| EOQ + Replenishment Policies + Health Score | [../03-inventory/04-replenishment.md](../03-inventory/04-replenishment.md) | Inventory |
| Exception Queue | [../03-inventory/05-exceptions.md](../03-inventory/05-exceptions.md) | Inventory |
| Fill Rate + Demand Signals + Intramonth | [../03-inventory/06-analytics.md](../03-inventory/06-analytics.md) | Inventory |
| ABC-XYZ + Supplier Performance | [../03-inventory/07-abc-xyz-supplier.md](../03-inventory/07-abc-xyz-supplier.md) | Inventory |
| Investment Optimization | [../03-inventory/08-investment.md](../03-inventory/08-investment.md) | Inventory |
| Multi-Echelon Safety Stock | [../03-inventory/10-echelon.md](../03-inventory/10-echelon.md) | Inventory |
| Replenishment Plan | [../03-inventory/11-replenishment-plan.md](../03-inventory/11-replenishment-plan.md) | Inventory |
| Inventory Rebalancing | [../03-inventory/12-rebalancing.md](../03-inventory/12-rebalancing.md) | Inventory |
| S&OP Cycle Management | [../04-operations/01-sop.md](../04-operations/01-sop.md) | Operations |
| Financial Planning | [../04-operations/02-financial.md](../04-operations/02-financial.md) | Operations |
| Event Calendar | [../04-operations/03-events.md](../04-operations/03-events.md) | Operations |
| Scenario Planning | [../04-operations/04-scenarios.md](../04-operations/04-scenarios.md) | Operations |
| AI Planning Agent | [../05-ai/01-ai-planner.md](../05-ai/01-ai-planner.md) | AI |
| Chatbot + Market Intelligence | [../05-ai/02-chatbot.md](../05-ai/02-chatbot.md) | AI |
| Control Tower | [../05-ai/03-control-tower.md](../05-ai/03-control-tower.md) | AI |
| Storyboard (Exception Workflow) | [../05-ai/04-storyboard.md](../05-ai/04-storyboard.md) | AI |
| UI Architecture + Theming | [../06-ui/01-ui-architecture.md](../06-ui/01-ui-architecture.md) | UI |
| Job Scheduler (APScheduler) | [../06-ui/02-job-scheduler.md](../06-ui/02-job-scheduler.md) | UI |
| Testing Strategy (pytest + Vitest + Playwright) | [../06-ui/03-testing.md](../06-ui/03-testing.md) | UI |
| Medallion Data Pipeline (Bronze/Silver/Gold) | [../07-integration/01-medallion.md](../07-integration/01-medallion.md) | Integration |
| Medallion Pipeline Refactoring (sql_helpers, dedup, SQL safety, DDL gaps) | [../07-integration/01-medallion.md](../07-integration/01-medallion.md) | Integration |
| RBAC + User Management | [../07-integration/02-rbac.md](../07-integration/02-rbac.md) | Integration |
| Notifications + Webhooks | [../07-integration/03-notifications.md](../07-integration/03-notifications.md) | Integration |
| FVA + ROI Measurement | [../07-integration/04-fva.md](../07-integration/04-fva.md) | Integration |

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
