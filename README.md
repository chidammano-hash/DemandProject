# Demand Studio

Unified demand forecasting analytics platform. Ingests sales and forecast data, stores it in PostgreSQL and Apache Iceberg, and serves a React UI for interactive analytics.

## Quick Start

```bash
cd mvp/demand

make init          # Create .venv, install dependencies
make up            # Start Docker services (Postgres, MinIO, Spark, Trino, MLflow)
make db-apply-sql  # Apply DDL schemas
make normalize-all # Normalize source CSVs
make load-all      # Load into Postgres + refresh materialized views

make api           # Start FastAPI on :8000
make ui-init       # Install frontend dependencies
make ui            # Start React dev server on :5173
```

## Tech Stack

| Layer | Technology |
|---|---|
| API | Python + FastAPI + Uvicorn |
| Frontend | React + Vite + TypeScript + Tailwind CSS + shadcn/ui |
| Charts | Recharts + ECharts |
| Database | PostgreSQL 16 |
| Lakehouse | Apache Iceberg via MinIO + Iceberg REST |
| ML | scikit-learn, LightGBM, CatBoost, XGBoost, Prophet, StatsForecast, NeuralProphet |
| ML Tracking | MLflow |
| Build | Make, uv, Docker Compose |

## Directory Structure

```
DemandProject/
├── mvp/demand/              Main application
│   ├── api/                 FastAPI backend (endpoints + routers)
│   ├── common/              Shared Python modules (domain specs, metrics, feature engineering)
│   ├── scripts/             Data pipeline & ML scripts (ETL, clustering, backtesting)
│   ├── frontend/            React + TypeScript UI
│   ├── tests/               Backend test suite (pytest)
│   ├── sql/                 DDL schemas (numbered migrations)
│   ├── config/              YAML configs (clustering, model competition, seasonality)
│   ├── infra/               Spark & Trino configs
│   ├── Makefile             All dev commands
│   └── docker-compose.yml   7-service infra cluster
├── docs/                    Design specs & strategic docs
│   └── design-specs/        35 feature specifications
├── CLAUDE.md                Full project specification
└── datafiles/               Source CSVs (gitignored, ~15GB)
```

## Testing

```bash
cd mvp/demand

make test          # Backend pytest (111+ tests)
make ui-test       # Frontend vitest (86+ tests)
make test-all      # Run all tests
make test-cov      # Backend with coverage report
```

## Key Documentation

- [CLAUDE.md](CLAUDE.md) - Complete project specification (tech stack, commands, conventions)
- [docs/design-specs/](docs/design-specs/) - Feature design specifications (feature1-35)
- [docs/REFACTORING_RECOMMENDATIONS.md](docs/REFACTORING_RECOMMENDATIONS.md) - Codebase improvement roadmap
- [mvp/demand/docs/ARCHITECTURE.md](mvp/demand/docs/ARCHITECTURE.md) - Architecture details
- [mvp/demand/docs/RUNBOOK.md](mvp/demand/docs/RUNBOOK.md) - Setup & troubleshooting guide
