# 01 — Initial Setup & Environment Bootstrap

This section takes a fresh clone of the **Supply Chain Command Center** repo to a fully working local dev environment (FastAPI backend on `:8000`, React/Vite UI on `:5173`, Postgres + MLflow + Redis in Docker). Follow the steps in order.

Repo root for all paths and commands: `/Users/manoharchidambaram/projects/DemandProject`.

---

## 1. Prerequisites

Install the following on the host machine **before** cloning. Versions match `pyproject.toml` and `frontend/package.json`.

| Tool | Required version | Install (macOS) | Notes |
|---|---|---|---|
| Python | `>=3.11, <3.14` | `brew install python@3.11` | Pinned in `pyproject.toml` `requires-python` |
| `uv` | latest | `brew install uv` | Python packaging + venv manager — must be on `PATH` |
| Node.js | `>=18 LTS` (20+ recommended) | `brew install node` | For Vite + Playwright |
| npm | bundled with Node | — | For `frontend/` deps |
| Docker | latest | Docker Desktop | Runs Postgres 16, MLflow, Redis |
| Docker Compose | v2 (bundled) | — | `docker compose` (not `docker-compose`) |
| GNU Make | 3.81+ (macOS default OK) | preinstalled / `brew install make` | All commands in `Makefile` |
| `psql` client | 16.x | `brew install libpq && brew link --force libpq` | Optional — most DB ops go through the container |

Quick sanity check:

```bash
python3 --version          # 3.11.x – 3.13.x
uv --version
node --version             # v18+ / v20+
docker --version
docker compose version
make --version
```

---

## 2. Clone & First-Time Setup

```bash
git clone <repo-url> DemandProject
cd DemandProject

# Backend: creates .venv via uv, installs all deps from pyproject.toml + uv.lock
make init

# Frontend: installs node_modules under frontend/
make ui-init
```

What `make init` does (see `Makefile:300-309`):

1. Copies `.env.example` -> `.env` if `.env` is missing.
2. Verifies `uv` is on `PATH` (else falls back to `make init-pip`).
3. Runs `uv venv` then `uv sync` against `uv.lock`.

What `make ui-init` does: `cd frontend && npm install` (`Makefile:452-453`).

**Fallback if `uv` is unavailable:** `make init-pip` creates a plain `.venv` and installs the minimum FastAPI stack via `pip` (no ML deps).

---

## 3. Environment Variables

Edit `/Users/manoharchidambaram/projects/DemandProject/.env` (created by `make init` from `.env.example`). All variables are read at process start by the API, scripts, and `docker-compose.yml`.

### 3.1 Required for local dev

| Variable | Default (dev) | Read by | Purpose |
|---|---|---|---|
| `POSTGRES_HOST` | `localhost` | `common/core/db.py`, `api/pool.py` | Postgres host. In the API container (`docker-compose.yml`) this is overridden to `postgres`. |
| `POSTGRES_PORT` | `5440` | `common/core/db.py` | Host-side mapped Postgres port (container internal port stays `5432`). |
| `POSTGRES_DB` | `demand_mvp` | `common/core/db.py` | Database name (also set in `docker-compose.yml`). |
| `POSTGRES_USER` | `demand` | `common/core/db.py` | DB user (matches container init). |
| `POSTGRES_PASSWORD` | `demand` (dev) / `changeme` (template) | `common/core/db.py` | DB password. Change for any non-local deployment. |
| `POSTGRES_HOST_PORT` | `5440` | `docker-compose.yml` | Host port mapping for the Postgres container. |
| `MLFLOW_HOST_PORT` | `5003` | `docker-compose.yml` | Host port for the MLflow UI. |
| `REDIS_HOST_PORT` | `6379` | `docker-compose.yml` | Host port for Redis. |

### 3.2 Auth / API security

| Variable | Default | Read by | Purpose |
|---|---|---|---|
| `API_KEY` | unset (auth disabled) | `common/auth.py` (`require_api_key`) | Static API key for write endpoints. When unset, `require_api_key` allows anonymous access — fine for local dev. **Required in production.** |
| `JWT_SECRET` | insecure default + warning logged | `common/auth.py` | HMAC secret for JWT-based auth. Generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. **Required in production.** |

### 3.3 LLM / external integrations

| Variable | Default | Read by | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | unset | `api/llm.py` | Enables AI Planner, chat, embeddings. Without it, AI features return errors. |
| `ANTHROPIC_API_KEY` | unset | `api/llm.py` | Optional Anthropic client (Claude) for AI features. |
| `GOOGLE_API_KEY` | unset | market intelligence router | Google Custom Search for market signals. |
| `GOOGLE_CX` | unset | market intelligence router | Google CSE id paired with `GOOGLE_API_KEY`. |
| `MLFLOW_TRACKING_URI` | `http://localhost:5003` (host) / `http://mlflow:5000` (container) | `scripts/ml/*` | MLflow run tracking. |

### 3.4 Runtime tuning

| Variable | Default | Read by | Purpose |
|---|---|---|---|
| `PLANNING_DATE` | unset (uses config) | `common/core/planning_date.py` | Override "today" for all date-sensitive code (`YYYY-MM-DD`). Useful for replaying a fixed planning date. |
| `USE_SYSTEM_DATE` | unset (`false`) | `common/core/planning_date.py` | When `true`, ignore `config/planning_config.yaml` and use the OS clock. |
| `DEMAND_GPU` | `auto` | backtest scripts in `scripts/ml/` | GPU acceleration mode: `on` / `off` / `auto`. Falls back gracefully if `cupy` / `numba` are not installed. |
| `POOL_MIN_SIZE` | `5` | `api/pool.py` | psycopg3 connection pool floor. |
| `POOL_MAX_SIZE` | `12` | `api/pool.py` | Sync primary pool ceiling. The API runs THREE independent pools per worker (sync / async / read) — see the multi-pool invariant below. |
| `ASYNC_POOL_MIN_SIZE` | falls back to `POOL_MIN_SIZE` (`5`) | `api/pool.py` | Floor for the async pool used by the async pilot routers (`customer_analytics`, `inv_planning_insights`). |
| `ASYNC_POOL_MAX_SIZE` | `20` (independent default) | `api/pool.py` | Async primary pool ceiling — sized larger than the sync pool because it carries the Customer-Analytics fan-out. Independent of `POOL_MAX_SIZE`. |
| `READ_REPLICA_URL` | unset (replica disabled — fall back to primary) | `common/core/db.py`, `api/pool.py` | Optional Postgres replica URL (`postgres://user:pass@host:port/dbname`). When set, `get_read_only_conn()` / `get_async_read_only_conn()` route to the replica; `_read_replica_configured()` is the gate. Caller must be lag-tolerant. |
| `READ_POOL_MIN_SIZE` / `READ_POOL_MAX_SIZE` | `READ_POOL_MAX_SIZE` final default `12` (chain: `READ_POOL_*` → `ASYNC_POOL_*` → `POOL_*`) | `api/pool.py` | Per-pool overrides for the read-replica pool (only created when `READ_REPLICA_URL` is set). Counts against the REPLICA's `max_connections`, not the primary's. |

**Multi-pool connection invariant.** Total backend connections against the PRIMARY are `GUNICORN_WORKERS × (POOL_MAX_SIZE + ASYNC_POOL_MAX_SIZE) + overhead`; keep that `≤` Postgres `max_connections` (dev `max_connections=200` in `docker-compose.yml`). The read-replica pool (`GUNICORN_WORKERS × READ_POOL_MAX_SIZE`) is bounded by the replica's own ceiling. With the defaults: `4 × (12 + 20) = 128 ≤ 200`. The `make deploy-check` preflight gate computes this exact formula (adding the read pool only when `READ_REPLICA_URL` is set) and fails if the total exceeds 85% of `max_connections`. Override `POSTGRES_MAX_CONNECTIONS` in `.env` if you raise the DB ceiling so the gate stays accurate.
| `PG_STATEMENT_TIMEOUT_MS` | `30000` | `api/pool.py` | `statement_timeout` applied per backend connection. |
| `REDIS_URL` | `redis://localhost:6379/0` (host) / `redis://redis:6379/0` (container) | `common/services/cache.py` | Redis connection string. Backs `cached_async` (single-flight de-dup); `reset_cache` flushes the live backend. |

After editing `.env`, reload your shell or restart any running processes — the API picks up `.env` only at startup.

---

## 4. Docker Services

```bash
make up            # Start postgres + mlflow + redis (and api if you want it containerized) + apply DDL
make logs          # Tail container logs
make down          # Stop all containers (volumes preserved)
```

`make up` runs `docker compose up -d` then `make db-apply-sql` (`Makefile:316-318`).

### 4.1 Containers in `docker-compose.yml`

| Service | Image | Host port | Container port | Volume | Purpose |
|---|---|---|---|---|---|
| `postgres` | `pgvector/pgvector:pg16` | `${POSTGRES_HOST_PORT:-5440}` | `5432` | `pg_data` | Primary DB (PG16 + pgvector). Tuned for OLAP — see `docker-compose.yml:8-25`. |
| `mlflow` | `ghcr.io/mlflow/mlflow:v3.0.0` | `${MLFLOW_HOST_PORT:-5003}` | `5000` | `mlflow_data` | ML experiment tracking. Backend store = same Postgres. UI: `http://localhost:5003`. |
| `redis` | `redis:7-alpine` | `${REDIS_HOST_PORT:-6379}` | `6379` | `redis_data` | API cache + future job queue. Capped at 256MB LRU. |
| `api` | built from project `Dockerfile` | `${API_HOST_PORT:-8000}` | `8000` | — | Optional containerized API (gunicorn + uvicorn). For dev you usually run `make api` on the host instead. |

Healthchecks: `pg_isready` on Postgres, `redis-cli ping` on Redis. The `api` and `mlflow` services depend on `postgres` being healthy.

Postgres has a **5GB** `shm_size` and large `shared_buffers` — make sure Docker Desktop has at least 8 GB of RAM allocated.

---

## 5. Database Schema Setup

```bash
make db-apply-sql
```

What it does (`Makefile:321-331`):

1. Waits for `pg_isready` inside the `postgres` container.
2. Iterates `sql/*.sql` in **lexical order** and pipes each through `psql -v ON_ERROR_STOP=1`.
3. Runs a one-off `ALTER TABLE` to relax `dim_customer.customer_name NOT NULL`.
4. Prints the count of applied files (currently **143** files in `sql/` — the `170-185` range covers the recent perf work: `pg_stat_statements`, partition cleanup, customer-analytics MVs, `pg_queue`, weekly-partition cutover prep).

### 5.1 DDL apply order (lexical)

Files are applied in the order returned by `ls sql/*.sql | sort`. The numeric prefix encodes the dependency chain:

| Range | Group | Examples |
|---|---|---|
| `001-007` | Core dims + facts | `dim_item`, `dim_location`, `dim_customer`, `dim_time`, `dim_sku`, `fact_sales_monthly`, `fact_external_forecast_monthly` |
| `008-013` | Indexes + accuracy views | `perf_indexes_and_agg`, `backtest_lag_archive`, `accuracy_slice_views` |
| `015-035` | Inventory + planning tables | `fact_inventory_snapshot`, `lead_time_profile`, `eoq_targets`, `replenishment_policy`, `safety_stock_targets`, `fill_rate_monthly`, `control_tower_kpis` |
| `036-058` | AI insights + S&OP + scenarios | `ai_insights`, `storyboard`, `production_forecast`, `replenishment_plan`, `sop_module`, `event_planning`, `supply_scenarios` |
| `062-073` | RBAC + DQ + collaboration + transfer network | `users_rbac`, `data_quality`, `notification_log`, `webhook_registrations`, `transfer_network`, `rebalancing_plan` |
| `080-099` | Medallion + perf profiling + tuning | `medallion_infrastructure`, `partition_inventory_snapshot`, `dim_sourcing`, `fact_purchase_orders`, `lgbm_tuning`, `tuning_chat`, `unified_model_tuning` |
| `100-130` | Promotion + experiments + customer demand + integrated targets | `results_promotion`, `cluster_experiments`, `champion_experiments`, `fact_customer_demand_monthly`, `candidate_forecast_and_promotion`, `inventory_algorithm_comparison`, `integrated_targets` |
| `170-185` | Perf work + customer-analytics MVs + pg-queue + weekly-partition cutover | `pg_stat_statements`, `drop_empty_future_partitions`, `drop_unused_indexes`, `mv_customer_filter_options`, `mv_customer_activity_geo` extension, `mv_ca_segment_trends`, `mv_ca_demand_at_risk`, `mv_ca_order_patterns`, `pg_queue`, weekly-partition cutover prep for `fact_inventory_snapshot` and `fact_customer_demand_monthly` |

A subset of the early DDL files (`001-007`, `009`) are also auto-applied on **first** Postgres container boot via `docker-entrypoint-initdb.d` mounts (`docker-compose.yml:31-38`). `make db-apply-sql` is still required to bring the schema fully up to date — the initdb mounts only run on an empty data volume.

### 5.2 Targeted re-applies (rarely needed)

```bash
make db-apply-inventory     # 017_create_fact_inventory_snapshot.sql
make db-apply-inv-backtest  # 019_inventory_forecast_view.sql
make db-apply-jobs          # 020 + 021 job_history DDL
```

### 5.3 Feature-specific schema targets (Inventory Planning, AI, forecast, platform)

Apply all DDL in order. Safe to re-run (`IF NOT EXISTS` guards on every statement). `make db-apply-sql` covers the majority of tables; the `make *-schema` commands below add feature-specific tables on top.

```bash
# Inventory Planning (IPfeature3–15)
make ss-schema                 # fact_safety_stock_targets + indexes
make eoq-schema                # fact_eoq_targets
make policy-schema             # dim_replenishment_policy + fact_dfu_policy_assignment
make health-schema             # mv_inventory_health_score materialized view
make exceptions-schema         # fact_replenishment_exceptions
make fill-rate-schema          # mv_fill_rate_monthly
make demand-signals-schema     # fact_demand_signals
make sim-schema                # fact_ss_simulation_results
make abc-xyz-schema            # XYZ classification columns on dim_sku
make supplier-perf-schema      # mv_supplier_performance
make investment-schema         # fact_inventory_investment_plan + fact_efficient_frontier
make intramonth-schema         # mv_intramonth_stockout
make control-tower-schema      # mv_control_tower_kpis

# Inventory Rebalancing
make rebalancing-schema        # mv_network_balance + fact_rebalancing_recommendations

# AI Planning Agent
make ai-insights-schema        # ai_insights + ai_planning_memos + ai_call_log + ai_recommendation_outcomes

# Production Forecast (F1.1)
make forecast-prod-schema      # fact_production_forecast (source_model_id included in base DDL)
                               # + fact_candidate_forecast (staging) + model_promotion_log (audit trail)
                               # DDL: sql/121_candidate_forecast_and_promotion.sql

# Forward-Looking Replenishment Plan (CI Bands + Repl. Plan)
make replplan-schema           # fact_replenishment_plan

# Storyboard
make storyboard-schema         # fact_storyboard_exceptions

# Data Quality (Spec 08-01)
make dq-schema                 # dim_dq_check_catalog + fact_dq_check_results + mv_dq_dashboard (sql/063)

# FVA Tracking (Spec 08-07)
make fva-schema                # fact_fva_tracking (sql/068)
```

> **Tip:** `make db-apply-sql` covers the majority of tables (including DDL 062-070 for auth, data quality, cache, notifications, webhooks, reports, rate limiting). The remaining `make *-schema` commands add feature-specific tables on top.

> **Note (stale RUNBOOK targets):** Older RUNBOOK revisions referenced `make auth-schema` (sql/062), `make cache-perf-schema` (sql/064), `make notification-schema` (sql/065), `make collaboration-schema` (sql/066), `make external-signals-schema` (sql/067), and `make report-schema` (sql/069). These standalone Make targets no longer exist in the current `Makefile`; their DDL (sql/062–070) is applied by `make db-apply-sql` directly.

### 5.4 AI Champion forward adjuster schema (Spec 02-27)

Apply the DDL manually:

```bash
psql "$DATABASE_URL" -f sql/189_drop_ai_fva_backtest.sql      # removes the old backtest store (idempotent)
psql "$DATABASE_URL" -f sql/190_create_ai_champion_forecast.sql
```

The AI Champion adjuster is **interactive** — run it from the **Item Analysis**
tab ("AI Champion" panel → pick a provider → **AI Adjust** → **Save**). There is
no batch Make target. See [09-ai-intelligence §9.11](09-ai-intelligence.md) for
the provider/key matrix.

### 5.5 Auth & RBAC setup

Run after schema setup. Seeds default admin user and configures JWT-based authentication.

```bash
# Auth config lives in config/platform/auth_config.yaml (JWT secret, token TTL, role hierarchy)
# common/auth.py provides: CurrentUser, get_current_user, require_role dependencies
# api/routers/platform/auth_router.py provides: POST /auth/login, POST /auth/refresh
# api/routers/platform/users.py provides: CRUD for dim_user (admin-only)

# No Make target needed — auth is auto-initialized when API starts.
# All mutation endpoints use require_role() for RBAC enforcement.
# Audit log entries written to fact_audit_log on every state-changing request.
```

---

## 6. Sanity Verification

Run after `make up` + `make db-apply-sql`. With no data loaded yet, row counts will be zero — that is expected.

```bash
make health           # alias for check-all → check-db + check-api
make check-all        # DB row counts + API endpoint pings
make check-db         # Just the row-count summary across all dim/fact tables
make check-api        # Just curl probes against /health, /items, /locations, ...
make audit-routers    # Router count vs main.py mount count + Vite proxy gaps
make ai-sync-check    # Verify Claude/Codex shared-guidance scripts are wired up
```

`make check-db` prints estimated row counts for `dim_item`, `dim_location`, `dim_customer`, `dim_time`, `dim_sku`, all major facts, `backtest_lag_archive`, `champion_experiment`, `cluster_experiment`, `lgbm_tuning_run`, and `job_history` (`Makefile:514-531`).

`make check-api` requires the API to be running on `:8000` — start it first with `make api`.

`make audit-routers` runs `scripts/tools/audit_routes.py` to flag any `app.include_router()` whose URL prefix is missing from `frontend/vite.config.ts` (the #1 cause of "HTML instead of JSON" errors in the UI).

---

## 7. Common First-Run Problems

### 7.1 `uv: command not found` when running `make`
`make` invokes `uv run …`; if `uv` isn't on the Make-level `PATH` (different from your interactive shell), every Python target fails.

**Fix:**
```bash
brew install uv
# or, if installed via pipx/cargo, ensure ~/.local/bin or the install dir is on PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && exec zsh
```
As a last resort: `make init-pip` (no `uv`, but no ML deps either).

### 7.2 Port already in use (5440 / 5003 / 6379 / 8000 / 5173)
```bash
lsof -iTCP -sTCP:LISTEN -P | grep -E ':(5440|5003|6379|8000|5173)'
```
Either stop the conflicting process or override the host port in `.env` (`POSTGRES_HOST_PORT`, `MLFLOW_HOST_PORT`, `REDIS_HOST_PORT`, `API_HOST_PORT`) and re-run `make up`. Vite (`:5173`) is changed via `make ui` flags or `frontend/vite.config.ts`.

### 7.3 `make db-apply-sql` hangs at `pg_isready`
The Postgres container is still starting (5GB shared memory init takes ~30 s on first boot) or it crashed.
```bash
docker compose ps
docker compose logs postgres | tail -50
```
If RAM is the issue, raise Docker Desktop's memory allocation to >= 8 GB.

### 7.4 Pool exhaustion (`pool timeout` / 503s under load)
`POOL_MAX_SIZE` defaults to 12 (sync primary pool, `api/pool.py`). Symptoms: API requests time out after ~10 s with `pool timeout exceeded`. Increase the cap and restart the API:
```bash
echo 'POOL_MAX_SIZE=18' >> .env
# Restart make api / containerized api
```
Stay within the multi-pool invariant: `gunicorn_workers × (POOL_MAX_SIZE + ASYNC_POOL_MAX_SIZE) ≤ max_connections` (dev `max_connections=200`). `make deploy-check` enforces it (fails at >85% of the ceiling), so raise `POSTGRES_MAX_CONNECTIONS` in `.env` and `max_connections` in `docker-compose.yml` together if you need more headroom.

### 7.5 Frontend gets HTML instead of JSON
A new API prefix was added without a Vite proxy entry. The dev server falls back to `index.html`.

**Fix:**
```bash
make audit-routers       # lists missing prefixes
# Then add the prefix to frontend/vite.config.ts proxy block.
```

### 7.6 `JWT_SECRET is using insecure default — set JWT_SECRET env var in production` warning
Expected in dev (`common/auth.py:64`). Set a real secret only when deploying.

### 7.7 `make init` fails on fresh macOS
Likely missing Xcode CLT (needed by some wheel builds): `xcode-select --install`. Then re-run `make init`.

### 7.8 Permission denied writing to `data/`
`data/` is gitignored and writable by the user that ran `make init`. If it was created by Docker as root, fix ownership:
```bash
sudo chown -R "$USER" data/
```

---

## 8. Start Dev

### 8.1 One-shot: bring up everything
```bash
make dev
```
Equivalent to `make up && make api && make ui` (`Makefile:14`). Note: `make api` runs in the foreground; in practice run the three targets in three terminals.

### 8.2 Recommended: three terminals

**Terminal 1 — infra:**
```bash
make up              # postgres + mlflow + redis (idempotent)
```

**Terminal 2 — API (FastAPI on http://localhost:8000):**
```bash
make api             # uv run uvicorn api.main:app --reload --port 8000
```

**Terminal 3 — UI (Vite on http://localhost:5173):**
```bash
make ui              # cd frontend && npm run dev -- --host --port 5173
```

Vite proxies all known API prefixes (see `frontend/vite.config.ts`) to `:8000`, so the UI talks to the live FastAPI process.

### 8.3 Verify the stack
- API docs: `http://localhost:8000/docs`
- API health: `curl http://localhost:8000/health`
- UI: `http://localhost:5173`
- MLflow UI: `http://localhost:5003`

### 8.4 Stop everything
```bash
# Ctrl-C the api + ui terminals, then:
make down            # docker compose down (volumes preserved)
```

---

## 9. Full First-Time Run (New Environment)

Sections 1–8 bring up an empty stack. To populate data, ML artifacts, inventory planning, demand planning, and ops in dependency order, use either the orchestrated `setup-*` targets (Option A) or the manual sequence (Option B).

### 9.1 Option A: Automated Setup Targets (Recommended)

Use the orchestrated `setup-*` targets that handle dependency ordering automatically:

```bash
# 0. Environment + schema
make init && make up && make ui-init
make db-apply-sql
make db-apply-inventory db-apply-inv-backtest db-apply-jobs

# 1. Full setup — data + ML + inventory + demand + ops (everything)
make setup-all

# 2. Start services
make api   # terminal 1
make ui    # terminal 2
```

**Available setup targets:**

| Target | What it does |
|---|---|
| `make setup-data` | Normalize + load all 10 domains into Postgres |
| `make setup-planning` | Data load + inventory planning (no ML — fastest path to a working UI) |
| `make setup-all` | Full pipeline: data + features + backtests + champion + inv planning + demand planning + ops |

**Intermediate targets** (called by `setup-all` in dependency order):

| Target | Phase |
|---|---|
| `make setup-features` | Clustering, seasonality, variability, lead time, ABC-XYZ, demand signals |
| `make setup-backtest` | All backtests + champion selection (depends on setup-features) |
| `make setup-inv-planning` | Safety stock, EOQ, policies, exceptions, health, rebalancing, control tower |
| `make setup-demand-planning` | Production forecasts, projections, orders, replenishment, consensus |
| `make setup-ops` | S&OP, events, financial plan, storyboard, DQ |

### 9.2 Option B: Manual Step-by-Step

```bash
# 0. Setup
make init && make up && make ui-init

# 1. Schema (one-time)
make db-apply-sql
make db-apply-inventory db-apply-inv-backtest db-apply-jobs
make ss-schema eoq-schema policy-schema health-schema exceptions-schema
make fill-rate-schema demand-signals-schema sim-schema abc-xyz-schema
make supplier-perf-schema investment-schema intramonth-schema control-tower-schema
make rebalancing-schema
make ai-insights-schema storyboard-schema forecast-prod-schema
make dq-schema fva-schema
# 2. Ingest (Option A: unified pipeline — recommended)
make pipeline-full               # Normalize + load + refresh MVs (all 10 domains)

# 2. Ingest (Option B: manual)
# make normalize-all && make load-all
# make inventory-pipeline

# 2b. Data Quality
make dq-run

# 3. Inventory Planning
make ss-compute eoq-compute policy-assign health-refresh
make exceptions-generate fill-rate-refresh variability-compute lt-profile-compute
make demand-signals-compute sim-run abc-xyz-classify
make supplier-perf-refresh investment-plan intramonth-refresh
make rebalancing-refresh rebalancing-compute
make control-tower-refresh

# 4. Clustering + Seasonality
make cluster-all && make seasonality-all

# 5. Backtesting
make backtest-all && make backtest-load-all

# 6. Champion selection
make champion-select

# 7. Production forecasts
make forecast-generate

# 8. AI insights
make ai-insights-scan

# 9. Storyboard
make storyboard-generate

# 9b. Data Quality (also runs automatically every 4h)
make dq-run

# 10. Start services
make api   # terminal 1
make ui    # terminal 2
```

> **Note:** The manual sequence above is migrated verbatim from the legacy RUNBOOK. A few step targets it references — `variability-compute`, `lt-profile-compute`, and `seasonality-all` — are not present as standalone targets in the current `Makefile` (their work is folded into `make features-compute` / `make setup-features`). If a target is not found, use the orchestrated `make setup-*` path in 9.1.

---

## 10. Incremental Refresh (New Data Arrives)

When new monthly data files are added:

```bash
# Option A: Unified pipeline orchestrator (recommended)
make pipeline-refresh            # Detects changed files, reloads only deltas, refreshes affected MVs

# Option B: Manual per-dataset reload
make load-forecast-replace       # New external forecast (preserves ML rows)
make inventory-pipeline          # New inventory snapshots

# Validate ingested data
make dq-run                      # Run data quality checks on refreshed data

# Re-compute dependent views
make health-refresh fill-rate-refresh intramonth-refresh
make demand-signals-compute
make rebalancing-refresh rebalancing-compute
make control-tower-refresh

# Re-run backtests (if model needs refreshing)
make backtest-all && make backtest-load-all
make champion-select

# Regenerate production forecasts
make forecast-generate

# Refresh AI insights
make ai-insights-scan

# Notifications & webhooks fire automatically on pipeline events
# (DQ failures, new AI insights, exception alerts, forecast generation)
```

---

## 11. Next Steps

- Load data: `make normalize-all && make load-all` (covered in Section 02 — Data Pipeline).
- Run tests: `make test-all` (backend pytest + frontend vitest).
- Full reset from CSV to fully populated app: `make fresh-all` (long-running).
