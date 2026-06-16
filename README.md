# Supply Chain Command Center

Unified supply chain planning and execution platform with ML-powered forecasting, inventory optimization, and interactive analytics.

## Project Structure

```
api/              # FastAPI backend (56 routers)
common/           # Shared Python modules (28 modules)
scripts/          # ETL, ML, and computation pipelines (58 scripts)
config/           # YAML configuration (41 files)
sql/              # PostgreSQL DDL migrations (86 files)
tests/            # Backend tests (2200+ pytest)
frontend/         # React + Vite + TypeScript UI (730+ vitest)
docs/             # Architecture docs + 52 design specs
data/             # Generated ML artifacts (gitignored)
data/input/       # Source CSV data (gitignored)
```

## Quick Start

```bash
# Setup
uv sync                    # Install Python dependencies
docker compose up -d       # Start Postgres, MLflow, Redis

# Run
make api                   # FastAPI on :8000
make ui                    # React dev server on :5173

# Test
make test-all              # Backend + frontend tests
```

## Tech Stack

**Backend:** Python, FastAPI, psycopg3, Pydantic v2, scikit-learn, MLflow
**Frontend:** React, TypeScript, Vite, Tailwind CSS, shadcn/ui, Recharts, ECharts
**Infrastructure:** PostgreSQL 16, Redis, Docker Compose, GitHub Actions

See [CLAUDE.md](CLAUDE.md) for detailed architecture and development rules.
