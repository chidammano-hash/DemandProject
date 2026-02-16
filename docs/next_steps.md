# Next Steps — Demand Studio Technology Roadmap

## Current Architecture

| Component | Status | Scale Limit |
|---|---|---|
| PostgreSQL 16 (OLTP) | Tuned, connection-pooled, trigram-indexed | ~50M rows per fact table, ~50 concurrent users |
| Iceberg + MinIO (Lakehouse) | Working, Spark writes, Trino reads | 100M+ rows, immutable time-travel snapshots |
| Trino (OLAP) | Benchmarking-ready | Complex cross-table analytics at scale |
| MLflow | Running, Postgres-backed | Experiment tracking ready for ML models |
| FastAPI + React | Pooled, debounced, parallel fetches | Sync-only; needs async for high concurrency |

---

## Phase 1: Multiple Forecast Algorithms

**Goal:** Run multiple models (ARIMA, Prophet, ML) and compare accuracy side-by-side.

**Tech needed:** None new — use existing stack.

**Steps:**
1. Add `model_id` / `model_name` column to `fact_external_forecast_monthly`
2. Each algorithm writes output rows with its own `model_id`
3. Update `agg_forecast_monthly` materialized view to group by `model_id`
4. Add model selector dropdown to the UI (filter KPIs + trend by model)
5. Use MLflow (already running) to track hyperparams, metrics, and artifacts per model run
6. Add a `make train-<model>` target that writes scored output back to Postgres
7. KPI cards show accuracy/WAPE/MAPE per model for direct comparison

**Files to modify:**
- `sql/007_create_fact_external_forecast_monthly.sql` — add `model_id` column
- `sql/008_perf_indexes_and_agg.sql` — update agg view to include `model_id`
- `common/domain_specs.py` — add `model_id` to forecast spec
- `api/main.py` — add model filter to analytics endpoint
- `frontend/src/App.tsx` — add model selector dropdown
- `scripts/` — add model training/scoring scripts

---

## Phase 2: Chatbot / Natural Language Queries

**Goal:** Users ask questions in plain English ("What's the accuracy for item 100320 in Q4?") and get answers.

**Tech needed:** pgvector (Postgres extension) + LLM API (Claude or OpenAI).

**Steps:**
1. `CREATE EXTENSION vector` in Postgres
2. Create `embeddings` table: store vector embeddings of column descriptions, sample queries, and schema metadata
3. Add a `/chat` POST endpoint in FastAPI:
   - Embed user question using LLM embedding API
   - Vector similarity search for relevant schema context
   - Send context + question to Claude API
   - Return natural language answer (or generated SQL)
4. Add a chat panel to the React UI
5. Add `anthropic` or `openai` SDK to `pyproject.toml`

**Files to create/modify:**
- `sql/009_create_embeddings.sql` — embeddings table with vector column
- `scripts/generate_embeddings.py` — embed schema metadata + sample queries
- `api/main.py` — add `/chat` endpoint
- `frontend/src/App.tsx` — add chat panel component
- `pyproject.toml` — add `anthropic` SDK dependency

---

## Phase 3: Inventory Snapshot Analysis

**Goal:** Track inventory positions over time, support point-in-time queries and what-if analysis.

**Tech needed:** None new — use Postgres temporal patterns.

**Steps:**
1. Design `dim_inventory_snapshot` with SCD Type 2 pattern:
   - `item_no`, `location_id`, `snapshot_date`, `qty_on_hand`, `qty_allocated`, `qty_available`
   - `valid_from` / `valid_to` (daterange) for temporal queries
2. Add DDL: `sql/009_create_dim_inventory_snapshot.sql` (or 010 if embeddings takes 009)
3. Add `inventory` to `common/domain_specs.py` as a new domain
4. Extend normalize + load scripts to handle inventory CSV
5. Add temporal query endpoint: `GET /domains/inventory/as-of?date=2025-01-15`
6. UI: add date picker for point-in-time inventory view
7. Materialized view for inventory KPIs (days of supply, stockout risk)

**Files to create/modify:**
- `sql/` — new DDL for inventory snapshot table + indexes
- `common/domain_specs.py` — add `inventory` domain spec
- `scripts/normalize_dataset_csv.py` — handle inventory source files
- `scripts/load_dataset_postgres.py` — load inventory data (upsert pattern instead of truncate)
- `api/main.py` — add temporal query endpoint
- `frontend/src/App.tsx` — add inventory domain + date picker

---

## Phase 4: Agentic AI Workflows

**Goal:** Autonomous agents that detect anomalies, investigate root causes, and recommend actions.

**Tech needed:** Redis + Celery (task queue) + LLM orchestration (LangGraph or Claude tool_use).

**Steps:**
1. Add Redis to `docker-compose.yml` (session store + task queue broker)
2. Add Celery workers for async task execution:
   - `make worker` target to start Celery
   - Tasks: `run_forecast`, `compare_models`, `check_inventory`, `generate_report`
3. Define agent "tools" as FastAPI endpoints the agent can call:
   - `/domains/{domain}/analytics` (already exists)
   - `/chat` (from Phase 2)
   - `/forecast/run` (trigger model scoring)
   - `/inventory/alert` (stockout detection)
4. Add agent orchestration:
   - Claude tool_use for simple single-turn agents
   - LangGraph for multi-step workflows (detect → investigate → act)
5. Agent memory:
   - Short-term: Redis (conversation context, tool results)
   - Long-term: Postgres table `agent_sessions` (decisions, outcomes)
6. Add `/agent/run` endpoint: accepts a goal, returns a plan + results

**Files to create/modify:**
- `docker-compose.yml` — add Redis service
- `pyproject.toml` — add `celery`, `redis`, `langgraph` (or `anthropic` for tool_use)
- `api/tasks.py` — Celery task definitions
- `api/agent.py` — agent orchestration logic
- `api/main.py` — add `/agent/*` endpoints
- `frontend/src/App.tsx` — add agent panel (goal input, progress, results)

---

## When to Add Heavier Infrastructure

Only add these if you hit a specific scale wall:

| Technology | Trigger | What It Replaces |
|---|---|---|
| **ClickHouse** | >500M rows in a fact table, Postgres aggregations >5s even with mat views | Postgres for OLAP queries (keep Postgres for OLTP) |
| **Apache Kafka + Flink** | Need real-time streaming (live inventory updates, streaming forecast scores) | Batch COPY pipeline; enables CDC and event-driven architecture |
| **Pinecone / Qdrant** | >5M vector embeddings, or need multi-modal (image+text) search | pgvector (keep pgvector for small-medium embedding sets) |
| **AsyncPG + uvloop** | >100 concurrent users or agent endpoints that call external LLMs | psycopg sync pool; needed for non-blocking LLM API calls |
| **Kubernetes** | >10 services, need auto-scaling, or multi-team deployment | Docker Compose (keep Compose for local dev) |

---

## What NOT to Do

- Don't migrate away from Postgres — it handles OLTP, vectors, temporal queries, and ML metadata in one place
- Don't add Kafka/Flink until you have real-time ingestion needs — batch COPY is fine for daily/weekly data
- Don't add ClickHouse until Postgres aggregations are provably slow — materialized views + indexing handles tens of millions of rows
- Don't pick a vector-only DB before trying pgvector — pgvector handles millions of vectors with zero new infra
- Don't over-engineer the agent framework — start with Claude tool_use before adding LangGraph

---

## Summary

```
Current: PostgreSQL + Iceberg + Trino + MLflow
              (you are here)

Phase 1: + model_id in forecast table ──→ Multi-algorithm comparison
Phase 2: + pgvector + Claude API ───────→ Chatbot / NL queries
Phase 3: + SCD Type 2 pattern ─────────→ Inventory snapshots
Phase 4: + Redis + Celery + Agents ────→ Agentic AI workflows

Only if needed:
  + ClickHouse ──────→ 500M+ row sub-second OLAP
  + Kafka/Flink ─────→ Real-time streaming ingestion
  + Pinecone/Qdrant ─→ 10M+ vector embeddings
  + Kubernetes ──────→ Multi-service auto-scaling
```

PostgreSQL with extensions (pgvector, pg_trgm, temporal ranges) covers ~80% of this roadmap without adding infrastructure complexity.
