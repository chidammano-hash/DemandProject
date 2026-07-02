# Databricks + Lakebase Migration

> Port the platform to Databricks with **Lakebase** (managed PostgreSQL) as the operational database and **Delta Lake** as the source-of-record for input data, synced into Lakebase. The application code is ~unchanged; the work concentrates in data-ingestion topology, connection auth, and deployment plumbing.

| | |
|---|---|
| **Status** | Proposed — not yet implemented |
| **Target platform** | Databricks (Lakebase Postgres + Unity Catalog + Delta Lake + Databricks Apps / Workflows) |
| **Key Files (touched by the move)** | `common/core/db.py`, `api/pool.py`, `common/core/paths.py`, `common/core/domain_specs.py`, `config/.../etl_config.yaml`, `config/ai/sku_chat_config.yaml`, `Makefile`, `scripts/etl/load_*.py`, `sql/139,137,092,090,008,170` (extensions) |
| **New artifact (this spec)** | `scripts/db/apply_sql_lakebase.py` — host-side, token-aware DDL applier (`make db-apply-sql-lakebase`) |
| **Key Files (UNCHANGED)** | every router, every `%s` query, Pydantic models, `common/ml/**`, `frontend/**`, `read_sql_chunked`, all 33 MVs |

---

## Problem

The platform is delivered today as Docker Compose: a self-hosted PostgreSQL 16, a host/Docker FastAPI, a Vite/Nginx frontend, Redis, and a `normalize → COPY` ETL that lands external CSVs into Postgres fact/dim tables. Several customers run their lakehouse on **Databricks** and want the platform to:

1. Use **Lakebase** (Databricks' managed, Postgres-compatible OLTP database) instead of a self-hosted Postgres, to remove the DB from their operational burden.
2. Treat **Delta Lake** (Unity Catalog) as the system-of-record for **input** data — sales, external forecasts, inventory snapshots, customer demand, master data — and **sync** those tables into Lakebase rather than re-ingesting CSVs.
3. Keep the application code essentially as-is.

The question this spec answers: **what actually has to change, what does not, and how hard is it.**

---

## Solution

**Lakebase speaks the Postgres wire protocol.** psycopg3, every parameterized (`%s`) query, the three connection pools, all 33 materialized views, the read-replica abstraction, `read_sql_chunked`/`stream_query_in_chunks`, the ML training scripts (pandas/scikit-learn/LightGBM), and the React frontend are **byte-for-byte unchanged**. The socket on the other end simply happens to be Lakebase.

The move is tractable because the codebase **already separates** two classes of table, which is exactly the line Lakebase forces:

- **Source tables** — populated from external feeds via `DomainSpec` + `etl_config.yaml` `domain_order`. These become **Lakebase synced tables** fed from Delta (continuous or scheduled). **Synced tables are READ-ONLY on the Postgres side.**
- **App-written tables** — forecasts, experiments, champion selections, AI logs, chat. These stay **native Lakebase tables** that the app keeps writing to via psycopg/`COPY`, unchanged.

Three abstractions already in the codebase make this nearly mechanical:

| Abstraction | File | Why it helps |
|---|---|---|
| Single DB-cred source of truth | `common/core/db.py` `get_db_params` → `api/pool.py` `_build_conninfo` | The token-auth change lands in **one** place, not scattered across routers |
| Read-replica switch | `get_read_replica_params` (`READ_REPLICA_URL`) | Point at a Lakebase replica endpoint; the 7 opted-in CA endpoints route to it with no code change |
| Agent auth-mode switch | `common/ai/sku_chat/auth.py` (`auth.mode`: auto/api_key/bedrock/vertex) | The SKU Chat agent flips from Claude-Code subscription auth to **Bedrock** with a config edit |

---

## How It Works

### 1. Table classification — synced (read-only) vs native (writable)

This is the central design decision. Each table is assigned to exactly one path.

**Synced from Delta → read-only in Lakebase** (today's `domain_order` load targets):

```
dim_item              dim_location          dim_customer
fact_sales_monthly    fact_external_forecast_monthly
fact_inventory_snapshot                     fact_customer_demand_monthly
dim_sourcing          fact_purchase_orders
```

`dim_time` is auto-generated (2020–2035) — keep it native, or materialize it as a Delta table.

**Native Lakebase tables — the app keeps writing** (unchanged psycopg/`COPY`):

```
fact_candidate_forecast   fact_production_forecast   fact_ai_champion_forecast
backtest_run              champion_*                 cluster_experiment*
cluster_tuning_profile*   ai_insights   ai_call_log  ai_decision_ledger
exception_queue           sku_chat_session/message/call_log
sku_chat_pending_adjustment   audit_load_batch       ...
```

The generative-pipeline COPYs (`scripts/forecasting/generate_production_forecasts.py`, `scripts/ml/run_backtest.py`, `scripts/ml/run_champion_selection.py`, `scripts/etl/load_backtest_forecasts.py`) target **native** tables and require **no change**. Only the **source-load** scripts (`scripts/etl/load_dataset_postgres.py`, `load_customer_demand_postgres.py`, `load_ext_ml_forecasts.py`) are retired in favour of Delta sync.

**Schema bring-up.** The DDL is the `sql/*.sql` migrations applied in sorted (numeric-prefix) order — there is no separate `schema.sql`. `make db-apply-sql` does this **inside the Docker container**, so it cannot reach Lakebase. Use the host-side applier instead:

```bash
export POSTGRES_HOST=<instance>.database.cloud.databricks.com POSTGRES_PORT=5432
export POSTGRES_DB=databricks_postgres POSTGRES_USER=<identity> PGSSLMODE=require
export POSTGRES_PASSWORD="$(databricks auth token …)"   # fresh OAuth token (one-shot apply fits the TTL)
make db-apply-sql-lakebase                               # → python -m scripts.db.apply_sql_lakebase
```

`scripts/db/apply_sql_lakebase.py` connects over the network (token auth via env or `DATABASE_URL`), applies every file in order, and **tolerates** role-restricted extension files (`sql/170_enable_pg_stat_statements.sql`) instead of aborting (`--strict` to make all failures fatal; `--dry-run`/`--start-from`/`--only` for control). It creates the source tables as **native** first; converting them to synced tables (next section) is a follow-up.

### 2. ⚠️ The write-after-sync gotcha (`dim_sku`, `customer_features_monthly`)

Two tables are **seeded from source AND mutated by the app** — they appear in both lists:

- **`dim_sku`** — loaded from CSV, then `common/ml/sku_features/` writes computed feature columns into it. Clustering, and every full-grain FVA/accuracy join (`item_id AND customer_group AND loc`), depend on it.
- **`customer_features_monthly`** — loaded as source and also written by `scripts/ml/generate_customer_features.py`.

A read-only synced table **cannot** be both source and write target. Resolution per table:

- **(a) Keep native, seed from a synced raw table.** Sync `dim_sku_raw` from Delta; keep `dim_sku` native; the SKU-features pipeline writes computed columns into the native `dim_sku` as it does today. **Recommended for `dim_sku`** — it is on the hot path for clustering and accuracy.
- **(b) Sidecar split.** Sync the raw columns; move computed columns to a native sidecar table joined at query time.

Get this decision right **first** — a naive "all dims are synced" rule breaks clustering and the accuracy/FVA joins silently.

### 3. Connection auth — static password → rotating token

Today `get_db_params()` returns static `POSTGRES_*` env values and `api/pool.py` builds a conninfo string once at pool creation. Lakebase authenticates Databricks identities with **short-lived OAuth tokens** (rotated, typically ~1 h), not a fixed password.

Required change, confined to two files:

1. A **credential provider** that fetches/refreshes a Lakebase token (Databricks SDK / workspace identity).
2. `get_db_params()` returns the current token as `password`; `api/pool.py` `_create_pool()` supplies a `psycopg` **connection-factory / `configure` hook** (or a pool `reconnect` path) that injects a fresh token on every (re)connect, so long-lived pooled connections survive token rotation.

This is the only change that touches the DB hot path and the only one warranting a dedicated test (token-expiry → reconnect).

### 4. Postgres extension compatibility

The schema uses four extensions — verify each against Lakebase's allowlist before cutover:

| Extension | Used by | Note |
|---|---|---|
| `vector` (pgvector) | `sql/139_create_rag_chunk.sql` | Supported on Lakebase. `sql/139` already documents a `REAL[]` fallback if a version is too old. |
| `pg_trgm` | `sql/008`, `sql/017` | Standard contrib; expected available. |
| `pgcrypto` | `sql/090`, `sql/092`, `sql/137` | Standard contrib; expected available. |
| `pg_stat_statements` | `sql/170` | Often **managed/restricted** on hosted Postgres — profiling may read it differently or require a platform toggle. |

### 5. Connection-limit retune

The multi-pool invariant still holds, but against **Lakebase's** ceiling, not `docker-compose.yml`'s `max_connections=200`:

```
GUNICORN_WORKERS × (POOL_MAX_SIZE + ASYNC_POOL_MAX_SIZE) + overhead ≤ Lakebase max_connections
```

Lakebase favours its **built-in PgBouncer-style pooling**; consider routing pools through the pooled endpoint and lowering per-pool sizes. Update the `make deploy-check` preflight to the new ceiling (it trips at >85%).

### 6. Artifact storage → Unity Catalog Volumes

`data/` (gitignored) holds model `.pkl`s (`data/models/`), staged intermediates (`data/staged/`), and `data/champion/dfu_assignments.csv`. On Databricks this maps to a **Unity Catalog Volume** (or DBFS). `common/core/paths.py` already centralizes `PROJECT_ROOT/DATA_DIR/...` — point `DATA_DIR` at the mounted volume via env; no scattered path edits.

### 7. Supporting services

| Service | Today | On Databricks |
|---|---|---|
| **Cache** | Redis 7, in-memory fallback | External managed Redis (Azure Cache / ElastiCache) **or** the existing in-memory fallback (loses cross-worker single-flight stampede protection). Config: `config/.../cache_config.yaml`. |
| **MLflow** | self-hosted tracking | Databricks **managed MLflow** — tracking URI `databricks`. Near drop-in. |
| **Scheduling** | APScheduler (in-proc) + `pg_queue` (Postgres-table queue) | Both work on Lakebase as-is. Optionally move multi-hour jobs to **Databricks Workflows**; keep short recurring jobs on APScheduler. |
| **API hosting** | host/Docker uvicorn + Nginx | **Databricks App** hosting FastAPI with a native Lakebase connection; frontend served as static assets. |
| **Pipelines / ML** | Make targets on host | **Databricks Jobs/Workflows** (single-node clusters — code is pandas/sklearn, no Spark rewrite needed). |

### 8. SKU Chat agent auth

The agent (`common/ai/sku_chat/`) runs `auth.mode: auto` (inherits the Claude Code subscription). A Databricks App has no Claude Code session, so set `auth.mode: bedrock` (or `vertex`/`api_key`) in `config/ai/sku_chat_config.yaml` and provide credentials. **The switch already exists** in `auth.py` — no code change.

### 9. MV refresh after sync

The 33 MVs depend on the now-**synced** base tables. Run `make refresh-mvs-tiered` **after** each Delta→Lakebase sync (dependent-after-source ordering preserved). `REFRESH MATERIALIZED VIEW CONCURRENTLY` still needs its unique index. Wire the tiered refresh into the sync schedule (Databricks Workflow task downstream of the sync).

---

## Migration Plan (phased)

1. **Compat spike (days).** Point the local app at a Lakebase instance; `CREATE EXTENSION` the four extensions; bring up the schema with `make db-apply-sql-lakebase` (§1); run `make test-all` + `make health`. Flushes out §4 immediately.
2. **Connection auth (§3).** Implement the token-refresh credential provider behind `get_db_params`; add a token-expiry → reconnect test. *Only hot-path code change.*
3. **Table classification (§1–§2).** Define synced vs native; stand up Delta sync for the 9 source tables; resolve `dim_sku` / `customer_features_monthly` (recommend: keep native, seed from synced raw).
4. **Infra (§5–§7, §9).** `DATA_DIR` → Volume; MLflow URI; Redis decision; pool retune + `deploy-check`; MV-after-sync hook; Databricks App + Workflows.
5. **Agent (§8).** `auth.mode: bedrock`; smoke-test `/sku-chat/stream`.

---

## Effort & Risk

| Area | Effort | Risk |
|---|---|---|
| App / SQL / ML / frontend / MVs | **None** (unchanged) | — |
| Connection auth (token rotation) | Small code, fiddly | Medium — long-lived pooled conns must survive rotation |
| Source-load → Delta-sync flip | Medium | Low — clean synced/native split already exists |
| `dim_sku` write-after-sync | Small | **Highest** — breaks clustering + accuracy joins if mis-modelled |
| Extension compatibility | Low (verify) | Low–medium — `pg_stat_statements` may be restricted |
| Connection-limit retune | Low | Low |
| Artifact storage → Volumes | Low–medium | Low |
| Redis / MLflow / scheduling / hosting | Low–medium | Low |
| SKU Chat agent auth | Trivial (config) | Low |

**Overall:** a focused ~2–4 week effort for an engineer fluent on both sides, dominated by data-pipeline rewiring and platform plumbing — **not** by touching business logic. The single biggest correctness risk is the `dim_sku` write-after-sync conflict (§2); decide it first.

---

## What Does NOT Change

- Any router, any `%s` query, Pydantic v2 schemas.
- `common/ml/**` (clustering, sku_features, backtest, champion, expert panel), `read_sql_chunked`.
- All 33 materialized views and the tiered-refresh ordering.
- The entire `frontend/**` (React/Vite/Tailwind) — same API contract.
- Champion candidate→production promotion, the generative-pipeline COPYs.
- The read-replica abstraction (just set `READ_REPLICA_URL` to a Lakebase replica).

---

## Open Questions / Verify Before Cutover

- Lakebase **extension allowlist** — confirm `vector`, `pg_trgm`, `pgcrypto`; confirm whether `pg_stat_statements` is exposed (affects `common/services/perf_profiler.py` query stats).
- Lakebase **token TTL** and the recommended psycopg reconnect pattern (connection-factory vs pool `configure`).
- Lakebase **`max_connections`** for the chosen compute size → final pool sizing + `deploy-check` ceiling.
- Delta→Lakebase **sync mode** per source table (continuous vs scheduled) and its lag vs the MV refresh cadence.
- Whether `customer_features_monthly` follows the `dim_sku` "native, seed from raw" pattern or a sidecar split.

---

## Dependencies

- **[01-infrastructure](01-infrastructure.md)** — tech stack, Docker services, pool sizing this spec re-targets.
- **[02-data-models](02-data-models.md)** — the dim/fact tables classified as synced vs native.
- **[07-customer-demand-fact](07-customer-demand-fact.md)** — `fact_customer_demand_monthly` partitioning under sync.
- **[06-ai-platform/07-sku-chatbot](../06-ai-platform/07-sku-chatbot.md)** — the agent `auth.mode` flip to Bedrock.
- **CLAUDE.md** — the connection-pool invariant and read-replica rules this spec re-applies to Lakebase.
