# 10 — Frontend, Testing, Performance & Routine Maintenance

This section covers day-to-day operation of the **React frontend** (Vite + TypeScript), regenerating the OpenAPI-derived TypeScript types, running the full test suite (backend pytest, frontend vitest, Playwright E2E), profiling script and API performance, and the routine maintenance chores that keep the app healthy after route or schema changes.

Repo root for all paths and commands: `/Users/manoharchidambaram/projects/DemandProject`.

---

## 1. Frontend Dev Server

The frontend lives at `frontend/` and is a Vite-powered React 18 + TypeScript SPA. All UI work happens against a hot-reloading dev server on `:5173`, which proxies API calls to the FastAPI backend on `:8000`.

### 1.1 First-time install

```bash
make ui-init           # cd frontend && npm install
```

Run this:

- After a fresh clone.
- After pulling a branch that changed `frontend/package.json` or `package-lock.json`.
- If `node_modules/` was deleted or `npm` reports missing modules.

### 1.2 Start the dev server

```bash
make ui                # cd frontend && (install if missing) && vite --host --port 5173
```

The `make ui` target is self-healing: if `node_modules/.bin/vite` is missing it runs `npm install` first, so you can skip `make ui-init` on a clean checkout.

### 1.3 Hot reload behavior

| Change | Reload type |
|---|---|
| `*.tsx` / `*.ts` source edit | HMR — module swap, component state preserved where possible |
| Tailwind class change | HMR — instant style update |
| `tailwind.config.js`, `postcss.config.js` | Full reload |
| `vite.config.ts` | **Restart Vite** — Vite watches but proxy / build changes need a fresh `make ui` |
| `src/api/generated/schema.ts` regen | HMR (just types) |
| New file added under `src/` | HMR picks it up automatically |

### 1.4 Common dev URLs

| URL | What it is |
|---|---|
| `http://localhost:5173/` | App root (defaults to first sidebar tab) |
| `http://localhost:5173/?tab=invPlanning` | Open a specific tab via query param |
| `http://localhost:8000/docs` | FastAPI Swagger UI (the source for `gen:types`) |
| `http://localhost:8000/openapi.json` | Raw OpenAPI schema |

---

## 2. Generating TypeScript Types from OpenAPI

The frontend keeps a generated `src/api/generated/schema.ts` aligned with the FastAPI Pydantic models. Regenerate it whenever a backend response shape changes.

### 2.1 Procedure

```bash
# 1. Start the API in another shell
make api                                       # FastAPI on :8000

# 2. From the frontend directory
cd frontend
npm run gen:types
# runs: openapi-typescript http://127.0.0.1:8000/openapi.json -o src/api/generated/schema.ts
```

### 2.2 When to regen

Regenerate `schema.ts` after **any** of these:

- New / changed Pydantic response model in `api/`.
- New router added under `api/routers/<domain>/`.
- Field added/removed/renamed on an existing endpoint.
- Status code or error envelope change.

### 2.3 Verify

After regen, run:

```bash
cd frontend && npx tsc --noEmit              # surface type drift
make ui-test                                  # ensure mocks still match
```

If `tsc` errors appear in callers, update the call sites — do **not** hand-edit `schema.ts`.

---

## 3. Vite Proxy & Route Audit

The Vite dev server proxies certain path prefixes to the FastAPI backend. Without a proxy entry, the SPA fallback returns `index.html` (HTML) instead of JSON, and you get the classic *“Unexpected token < in JSON at position 0”* in the browser console.

### 3.1 Where it lives

- `frontend/vite.config.ts` — array `API_PATH_PREFIXES` (one entry per prefix). All entries proxy to `http://127.0.0.1:8000`.
- `frontend/nginx.conf` — same prefix list inside the `location ~ ^/(...)/ ` regex used by the production Nginx container. **Both must stay in sync.**

### 3.2 Adding a new API prefix

When you mount a new router under a new top-level path (e.g. `/foo`):

1. Add `"/foo",` to `API_PATH_PREFIXES` in `frontend/vite.config.ts`.
2. Add `foo` to the alternation in `frontend/nginx.conf` (`location ~ ^/(...|foo|...)/ `).
3. Restart `make ui` so Vite re-reads the config.
4. Run the audit (next).

### 3.3 Audit the routes

```bash
make audit-routers
```

This counts router files vs `app.include_router()` calls and runs `scripts/tools/audit_routes.py`, which compares FastAPI mounts in `api/main.py` against `vite.config.ts`. Run it after **every** route change.

---

## 4. Running Tests

### 4.1 Test command matrix

| Command | Scope | Approx. time | Notes |
|---|---|---|---|
| `make test` | All backend pytest (`tests/`) | ~0.7 s | DB fully mocked via `make_pool()` |
| `make test-unit` | `tests/unit/` only | <0.5 s | Pure-Python unit tests |
| `make test-api` | `tests/api/` only | ~0.5 s | httpx AsyncClient + ASGITransport |
| `make test-cov` | Backend + coverage report | ~1 s | Term-missing coverage on `api/`, `common/` |
| `make ui-test` | Frontend vitest | ~1.5 s | Runs `cd frontend && npx vitest run` |
| `make test-quick` | `test` + `ui-test` | ~2 s | Fast pre-commit gate |
| `make test-all` | `test` + `ui-test` | ~2 s | Identical to `test-quick` today |
| `make e2e` | Playwright E2E | ~30–90 s | **Requires API on :8000** |
| `make e2e-install` | One-time Playwright setup | ~30 s | Installs Chromium |
| `make e2e-ui` | Playwright in UI mode | interactive | Visual debugger |
| `make e2e-headed` | E2E with visible browser | ~60 s | Useful for debugging selectors |
| `make e2e-report` | Open last HTML report | instant | After a failing run |

### 4.2 PATH workaround if `make` fails

Some shells do not have `uv` or `node` on `PATH` when Make is invoked. In that case run the underlying commands directly from the project root:

```bash
# Backend (project root)
~/.local/bin/uv run pytest tests/ -q

# Frontend (frontend/)
PATH="/opt/homebrew/bin:$PATH" /opt/homebrew/bin/node node_modules/.bin/vitest run --reporter=dot
```

### 4.3 E2E prerequisites

Playwright tests assume:

- API is up on `:8000` (`make api`).
- The Vite dev server is auto-started by `e2e/playwright.config.ts` (`webServer.command = "npm run dev"` on `:5173`, with `reuseExistingServer = !CI`). If you already have `make ui` running locally, Playwright reuses it.
- Chromium is installed (`make e2e-install`).

Failure artifacts land in `frontend/e2e/test-results/` (screenshots + video on retain-on-failure) and `frontend/e2e/playwright-report/` (HTML).

### 4.4 Validation checks

Beyond the test suite, these targets validate that the running stack is healthy:

```bash
make check-db    # Table row counts in Postgres
make check-api   # curl API health + sample endpoints
make check-all   # DB + API (full)
make test-all    # Full test suite (backend + frontend)

# E2E smoke tests (requires API + UI running)
make e2e-install # Install Playwright browsers (one-time)
make e2e         # Run all E2E smoke tests (headless)
```

---

## 5. Test Patterns

### 5.1 Backend (pytest)

- **Mock pool factory**: import `from tests.api.conftest import make_pool as _make_pool`. Patch `api.core._get_pool` with a `make_pool(...)` call. The `mock_pool` fixture wraps `make_pool`. `make_pool` accepts an optional `description=` arg to set `cursor.description` (column names) when the route reads it. Async (`get_async_read_only_conn` / `get_async_pool`) routes use `make_async_pool` and must patch `api.core._get_async_pool`.
- **Rate limiter**: an autouse fixture resets the global rate limiter between tests, so per-test request counts start clean.
- **Multi-fetchall endpoints**: set `cursor.fetchall.side_effect = [list1, list2, ...]` (one list per `cursor.fetchall()` invocation in the route).
- **Single-call endpoints**: set `cursor.fetchall.return_value = [...]`.
- **API client**: inline `httpx.AsyncClient(transport=ASGITransport(app=app))` inside the test — no live server needed.
- **Mock row tuples MUST match the SQL column count and order** of the query. A wrong arity is the most common test failure after a router change.
- **Bigger fixtures** (LLM, OpenAI, Anthropic): `resp.usage` needs integer fields (`total_tokens`, `prompt_tokens`, `completion_tokens`).

### 5.2 Frontend (vitest + React Testing Library)

- **Wrapper**: `TestQueryWrapper` from `frontend/src/tabs/__tests__/test-utils.tsx` (provides `QueryClientProvider`, `ThemeProvider`, `GlobalFilterProvider`).
- **API mocks**: `vi.mock("../api/queries")` for the barrel module. Tests must export every key/fetcher the component imports — see CLAUDE.md MEMORY notes for known traps (`InvPlanningTab.test.tsx` needs `insightKeys`, `STALE_INSIGHTS`, evolution keys, sourcing/PO fetchers, etc.).
- **Charts**: Recharts is the default engine and renders in jsdom without mocking. Only the 8 heavy customer-analytics panels use ECharts (via `ModularReactECharts` in `frontend/src/components/echarts-modular.tsx`); for those, mock `echarts-for-react` (e.g. `vi.mock("echarts-for-react", () => ({ default: () => null }))`) since jsdom cannot render canvas/SVG-heavy charts. `EChartContainer` has been removed — do not reference it. `ForecastTrendChart` is now a Recharts component.
- **Virtualized rows**: mock `@tanstack/react-virtual` so virtualized lists render synchronously.
- **Theme**: use `useThemeContext()` / `useChartColors()` — never accept a `theme` prop from `App.tsx`.

### 5.3 E2E (Playwright)

- Use the fixtures in `frontend/e2e/fixtures/base.ts` — `appPage`, `navigateToTab(page, tabKey)`, `clickNavItem(page, label)`, `getContentArea(page)`.
- **Selectors**: semantic only — `getByRole`, `getByText`, `getByTitle`. Never CSS classes.
- New sidebar tabs go in `e2e/tests/navigation.spec.ts` with an entry in `SIDEBAR_LABELS`.

---

## 6. Lint, Format, Type-Check

### 6.1 Backend

| Command | What it does |
|---|---|
| `make lint` | `uv run ruff check api/ common/ scripts/ --fix` |
| `make format` | `uv run ruff format api/ common/ scripts/` |
| `make type-check` | `uv run mypy api/ common/ --ignore-missing-imports` |

The PostToolUse hook auto-runs `ruff check` on every edited Python file, and the PreToolUse hook blocks `git commit` if `ruff` or `pytest` fails. See `.claude/settings.json`.

### 6.2 Frontend

Run from `frontend/`:

| npm script | What it does |
|---|---|
| `npm run lint` | `eslint src/ --ext .ts,.tsx` |
| `npm run lint:fix` | ESLint with `--fix` |
| `npm run format` | `prettier --write src/` |
| `npm run format:check` | Prettier check (no write) |
| `npm run type-check` | `tsc --noEmit` (full project type check) |
| `npm run build` | `tsc -b && vite build` (production bundle) |

> The `deploy-frontend` Make target intentionally skips `tsc -b` and runs only `npx vite build` because the `restructure` branch carries ~30 pre-existing type errors in unrelated files (model-tuning, lgbm-tuning, settings, storyboard, types/index.ts). For local development always prefer `npm run build` so you see the full type errors.

---

## 7. Performance Profiling

Performance work is centralised in `common/services/perf_profiler.py` and `scripts/ops/run_perf_analysis.py`, with thresholds in `config/platform/perf_config.yaml`.

### 7.1 Make targets

| Command | Mode | Notes |
|---|---|---|
| `make perf-report` | Full system report | Read-only, safe for production |
| `make perf-script SCRIPT=<name>` | Profile a single script | Read-only — wraps DB in `default_transaction_read_only` + always rolls back |
| `make perf-script-full SCRIPT=<name>` | Same with REAL writes | **Staging only** — pass `--no-readonly` |
| `make perf-api` | API endpoint analysis | Read-only |
| `make perf-pipeline` | ETL pipeline analysis | Read-only |
| `make perf-clean` | Truncate `perf_*` history tables | Destructive — clears the profiling DB log |
| `make scale-test` | Run the scale-test suite under `tests/scale/` | Synthetic 100K rows by default; nightly knob `SCALE=10000000` |

### 7.2 Available script presets

`config/platform/perf_config.yaml` declares which scripts can be profiled and what args to pass. Examples currently registered:

```
compute_safety_stock        compute_eoq               compute_replenishment_plan
compute_sku_features        compute_inventory_projection
compute_demand_signals      compute_blended_forecast  compute_investment_plan
compute_rebalancing         compute_echelon_targets   compute_lead_time_variability
assign_replenishment_policies   classify_abc_xyz      compute_bias_corrections
compute_service_level_actuals   compute_financial_plan
generate_planned_orders     generate_replenishment_exceptions
generate_production_forecasts   generate_storyboard_exceptions
generate_consensus_plan     apply_event_adjustments
run_ss_simulation           run_sop_cycle             populate_dq_checks
fix_dq_issues
```

> Note: `compute_demand_variability` and `detect_seasonality` were unified into `compute_sku_features` on 2026-05-06 (the standalone scripts were deleted; redirect any preset to `compute_sku_features`).

To add a new preset, append it under `script_presets:` in `config/platform/perf_config.yaml` with optional `args:` / `callable:` / `callable_kwargs:` keys.

### 7.3 Suggestion thresholds (`config/platform/perf_config.yaml`)

| Threshold | Default | Triggers a suggestion when… |
|---|---|---|
| `query_slow_ms` | 5000 | A single query exceeds 5 s |
| `function_slow_s` | 10 | A profiled function exceeds 10 s |
| `memory_spike_mb` | 1024 | Peak memory exceeds 1 GiB |
| `memory_delta_mb` | 200 | A single section allocates >200 MiB |
| `n_plus_1_min_count` | 10 | ≥10 repeated near-identical queries |
| `unbatched_insert_min` | 5 | ≥5 individual `INSERT` statements (suggest `executemany`) |
| `sequential_child_min_s` | 2 | Sequential children each ≥2 s (suggest parallelism) |
| `total_query_time_pct` | 0.5 | DB time exceeds 50 % of wall time |

JSON reports land in `data/perf_reports/` (gitignored).

### 7.4 Instrumenting new scripts

Wrap major stages in `profiled_section()` rather than raw `time.time()`:

```python
from common.services.perf_profiler import profiled_section

def main():
    with profiled_section("load_inputs"):
        ...
    with profiled_section("compute"):
        ...
    with profiled_section("write_outputs"):
        ...
```

Keep section names short and stage-shaped — the report engine groups suggestions by section.

### 7.5 Production safety guarantees

`wrap_connection()` sets `default_transaction_read_only = true` and **always** issues `ROLLBACK`. Both `perf-report` and `perf-script` are safe to run against production. Only `perf-script-full --no-readonly` writes — restrict that to staging.

### 7.6 Recent perf work (sql/170-185)

The `170-185` SQL range and the supporting code paths cover this session's perf work. These are additive — the legacy paths still operate.

| Concern | What changed | Where |
|---|---|---|
| Boot-time perf | `pg_stat_statements` enabled; unused indexes dropped; empty future partitions cleaned | `sql/170`, `sql/171`, `sql/172` |
| Customer Analytics MV migration | Heavy ad-hoc aggregates replaced by MVs (segment trends, demand-at-risk, order patterns, filter options, activity geo) | `sql/173`, `sql/174`, `sql/180`-`sql/182` |
| Async pool pilot | `customer_analytics` and `inv_planning_insights` routers use `get_async_conn()` / `get_async_read_only_conn()` against the new `AsyncConnectionPool` | `api/pool.py`, `api/core.py` |
| Read-replica routing | Opt-in via `READ_REPLICA_URL`; read-only handlers route to the replica when set | `api/pool.py`, `common/core/db.py` |
| pg-queue scaffold | `refresh_intramonth` migrated off APScheduler onto pg-queue; see Section 8.7.3 for cutover recipe | `common/services/pg_queue.py`, `scripts/ops/pg_queue_worker.py`, `sql/183` |
| Cache + single-flight | `common/services/cache.py` `cached_async` decorator wraps hot endpoints with single-flight de-dup; `reset_cache` flushes the live backend (in-memory + Redis) | `common/services/cache.py` |
| Streaming ETL | `stream_query_in_chunks`, `read_sql_chunked` for bounded-memory loads | `common/core/sql_helpers.py` (see Section 02 §2.3) |
| Weekly partition cutover prep | DDL prepared; `auto_create_partitions.py` extended for weekly intervals | `sql/184`, `sql/185`, `scripts/db/auto_create_partitions.py` |
| Frontend chart consolidation | Recharts is the default engine; `EChartContainer` removed and `ForecastTrendChart` ported to Recharts. ECharts retained only for the 8 heavy customer-analytics panels via `ModularReactECharts` (tree-shaken `echarts-modular`). `LazyPanel.tsx` IntersectionObserver wrapper added; `HeatmapGrid` extended with compact mode + clickable headers | `frontend/src/components/echarts-modular.tsx`, `frontend/src/components/LazyPanel.tsx`, `frontend/src/components/HeatmapGrid` |
| Tab splits + dead-file cleanup | Large tabs split into subpanel dirs (`command-center/`, `data-quality/`, `aggregate-analysis/`, `storyboard/`, `forecast/`, `item-analysis/`). Removed dead frontend files: `AlertPanel`, `ItemDetailPanel`, `PipelineConfigPanel`, `expsys`, `useTabVisibility` | `frontend/src/tabs/` |
| Scale tests | New `tests/scale/` directory with `make scale-test` (synthetic 100K rows by default; `SCALE=10000000` for nightly) | `tests/scale/`, `tests/unit/test_pg_queue.py`, `tests/unit/test_auto_create_partitions.py` |

---

## 8. Adding a New Tab or Component

Follow this checklist whenever you add a new sidebar tab, panel, or shared component (mirrors the CLAUDE.md *Feature Integration Checklist*).

### 8.1 Backend prep

- [ ] Router placed in the correct `api/routers/<domain>/` subdirectory (never at the flat root).
- [ ] Mounted in `api/main.py` **before** `domains.py` (which is mounted last).
- [ ] Auth-guarded write endpoints (`dependencies=[Depends(require_api_key)]`).
- [ ] Backend test in `tests/api/test_<feature>.py`.

### 8.2 Frontend wiring

- [ ] Query module created under `frontend/src/api/queries/<feature>.ts`.
- [ ] Vite proxy entry added to `API_PATH_PREFIXES` in `frontend/vite.config.ts` (and the matching alternation in `frontend/nginx.conf`).
- [ ] Tab component placed in `frontend/src/tabs/<NewTab>.tsx` (sub-panels go in `frontend/src/tabs/<feature>/`).
- [ ] Theme uses `useThemeContext()` — **no `theme` prop**.
- [ ] Sidebar entry added to the navigation registry.

### 8.3 Tests

- [ ] Co-located component test in `frontend/src/tabs/__tests__/<NewTab>.test.tsx` using `TestQueryWrapper`.
- [ ] If new sidebar tab: add label to `SIDEBAR_LABELS` and assertion to `frontend/e2e/tests/navigation.spec.ts`.
- [ ] Run `make test-all` — it must pass before opening a PR.

### 8.4 Verify

- [ ] `make audit-routers` reports zero drift.
- [ ] `cd frontend && npm run gen:types` if backend response shapes changed.
- [ ] Manual smoke: open the new tab in `make ui` — no “HTML instead of JSON” errors in the browser console.

---

## 8a. Starting Services & Platform Configuration

### 8a.1 API middleware stack

The FastAPI backend includes platform middleware applied in `api/main.py`:

- **GZip compression** — responses > 1KB are gzip-compressed
- **CORS** — allows `localhost:5173` (dev) origins
- **Cache layer** — `common/services/cache.py` provides `@cached` decorator with TTL-based caching.
  Two backends: Redis (when `REDIS_URL` env var set) or in-memory fallback.
  Config in `config/platform/cache_config.yaml`. Cache invalidation on write endpoints.
- **Query performance tracking** — endpoint latency + DB query counts logged to
  `fact_query_performance` for API governance and observability.

### 8a.2 Start services

```bash
# Terminal 1 — FastAPI backend
make api                     # FastAPI on :8000

# Terminal 2 — React frontend
make ui                      # Vite dev server on :5173
```

Open in browser:
- **UI:** `http://localhost:5173`
- **API:** `http://localhost:8000`
- **MLflow:** `http://localhost:5003`

> **Vite proxy:** Every new API path prefix must be added to `frontend/vite.config.ts` — otherwise the frontend receives HTML instead of JSON. Restart `make ui` after changes.

### 8a.3 Authentication & authorization (Spec 08-02)

Set `JWT_SECRET` in `.env` to enable JWT-based auth:

```bash
# .env
JWT_SECRET=your-secret-key-here    # Required for token signing/verification
JWT_ALGORITHM=HS256                # Default algorithm (optional, defaults to HS256)
JWT_EXPIRY_MINUTES=60              # Token lifetime (optional, defaults to 60)
```

**Default dev mode:** When `JWT_SECRET` is not set, the API runs in anonymous admin mode — all requests are treated as authenticated with admin privileges. Set `JWT_SECRET` in any non-local environment.

Passwords are hashed with bcrypt (12 rounds). Plaintext passwords are never stored.

**New Python dependencies** (added to `pyproject.toml`):
- `bcrypt` — password hashing for user accounts
- `PyJWT` — JWT token creation and validation

### 8a.4 Cache management (Spec 08-03)

Default backend is **InMemory** (no external dependencies). For production, configure Redis:

```bash
# .env (optional — InMemory is the default)
CACHE_BACKEND=redis               # "memory" (default) or "redis"
CACHE_REDIS_URL=redis://localhost:6379/0
CACHE_DEFAULT_TTL=300             # Default TTL in seconds (optional)
```

InMemory cache is cleared on process restart. Redis cache persists across restarts and is shared across workers.

### 8a.5 Notifications (Spec 08-04)

Configure notification channels in `config/platform/notification_config.yaml`:

```yaml
channels:
  email:
    enabled: false
    smtp_host: smtp.example.com
    smtp_port: 587
    from_address: alerts@example.com
  slack:
    enabled: false
    webhook_url: https://hooks.slack.com/services/...
  webhook:
    enabled: false
    url: https://your-endpoint.com/notify
```

Channels are disabled by default. Enable and configure as needed. Notification preferences are per-user and managed via the API.

### 8a.6 Webhooks (Spec 08-10)

Webhooks use **HMAC-SHA256 signing** for payload verification. Each webhook subscription has its own signing secret.

```bash
# Create a webhook subscription via API
curl -X POST http://localhost:8000/webhooks/subscriptions \
  -H "Content-Type: application/json" \
  -d '{"url": "https://your-endpoint.com/hook", "events": ["insight.created", "exception.generated"]}'
```

Failed deliveries are retried with exponential backoff (3 attempts by default). Delivery history is available via `GET /webhooks/deliveries`.

### 8a.7 Reports (Spec 08-08)

Report templates are managed via API. Schedules use the existing job engine:

```bash
# List available report templates
curl http://localhost:8000/reports/templates

# Schedule a recurring report
curl -X POST http://localhost:8000/reports/schedule \
  -H "Content-Type: application/json" \
  -d '{"template_id": "portfolio_summary", "schedule": "0 8 * * 1", "recipients": ["team@example.com"]}'
```

Templates define the data queries, layout, and output format (PDF/CSV/Excel).

### 8a.8 Rate limiting (Spec 08-09)

Configure rate limits in `config/platform/auth_config.yaml`'s `governance:` block
(`api_governance_config.yaml` was merged into it - see that file's own comment). The middleware
is sliding-window, not token-bucket, applies only to POST/PUT/DELETE (GETs are unlimited), and
only ever reads the `standard` tier (`api/main.py` calls `get_tier_limit("standard")` -
`free`/`premium` are defined in config but unused):

```yaml
governance:
  rate_limit_tiers:                      # Per-tier request quotas; only "standard" is read at runtime
    standard:
      requests_per_minute: 300           # Enforced by IP, POST/PUT/DELETE only
```

There are no per-endpoint overrides, no `X-RateLimit-*` response headers, and no per-user limiting
today - limiting is per-IP only. `deprecation.sunset_header` and `versioning.current_version` are
config-only fields with no enforcement code reading them yet.

When rate limited, the API returns HTTP 429 with `Retry-After` header. Limits are applied per authenticated user when `per_user: true`; otherwise globally.

---

## 9. Build & Deploy

### 9.1 Production bundle

The frontend container is a Nginx-fronted, multi-stage build defined in `frontend/Dockerfile`:

```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

Stage 1 installs deps with `npm ci`, runs `tsc -b && vite build`, and produces `dist/`. Stage 2 serves the static bundle from Nginx.

### 9.2 Nginx behaviour (`frontend/nginx.conf`)

- **SPA fallback**: `location /` → `try_files $uri $uri/ /index.html;` so client-side routing works.
- **API reverse proxy**: a `location ~ ^/(...)/ ` regex (with the same path prefix list as `vite.config.ts`) forwards to `http://api:8000` inside the Docker network. **A missing prefix here means HTML in the browser instead of JSON in production.**
- **Compression**: `gzip on; gzip_proxied any;` so JSON responses proxied from FastAPI are gzipped.
- **Security headers**: `X-Frame-Options`, `X-Content-Type-Options`, `Referrer-Policy`, and a CSP that allows `connect-src 'self' http://api:8000`.

### 9.3 Deploy via `make deploy`

`make deploy` runs the seven-stage `deploy-check → deploy-pydeps → deploy-redis → deploy-sql → deploy-frontend → deploy-api → deploy-smoke` chain. The frontend stage runs `cd frontend && npx vite build` (skipping `tsc -b` — see §6.2 note) and the smoke stage curls `/customer-analytics/kpis` and `/health` to confirm the cluster is live.

---

## 10. Routine Maintenance

### 10.1 After a route change

1. `make audit-routers` — confirm `main.py` mount count matches router file count and `vite.config.ts` proxy is in sync.
2. If a new prefix was added: also update `frontend/nginx.conf`.
3. Restart `make ui` so Vite picks up the new proxy entry.

### 10.2 After a Pydantic / response model change

1. Make sure `make api` is running on `:8000`.
2. `cd frontend && npm run gen:types` to regenerate `src/api/generated/schema.ts`.
3. `cd frontend && npx tsc --noEmit` to surface drift in callers.
4. `make ui-test` — fix any mock mismatches.
5. Commit `schema.ts` together with the source change so reviewers see the contract delta.

### 10.3 After an SQL / schema change

1. Apply DDL via `make db-apply-sql`.
2. Update mock row tuples in any affected `tests/api/test_<feature>.py` so column counts match.
3. `make test` to confirm no drift.
4. If a fact/MV row count changed materially, run `make perf-report` to spot any new query bottleneck.

### 10.4 After adding a new pipeline script

1. `make perf-script SCRIPT=<your_script>` (after registering it in `config/platform/perf_config.yaml` `script_presets:`).
2. Review suggestions for N+1 queries, unbatched inserts, memory spikes.
3. Wrap stages in `profiled_section()` if you have not already.

### 10.5 Periodic hygiene

| Cadence | Task |
|---|---|
| After every PR | `make test-all`, `make audit-routers` |
| Weekly | `make perf-report`, review `data/perf_reports/` for new regressions |
| Weekly | `make lint`, `make format`, `cd frontend && npm run lint` |
| When `node_modules/` feels stale | `cd frontend && rm -rf node_modules && npm ci` |
| When deps drift | `uv sync` (Python) and `cd frontend && npm install` (Node) |
| Before a release | `make e2e` against a staging API on `:8000` |
| Profiling table bloat | `make perf-clean` (truncates `perf_run`, `perf_section`, `perf_query`, `perf_suggestion`) |

### 10.6 Things to never do

- Never hand-edit `frontend/src/api/generated/schema.ts` — always regen.
- Never add a router file to the flat `api/routers/` root — use the correct domain subdirectory.
- Never pass `theme` as a prop — use `useThemeContext()`.
- Never commit `node_modules/`, `__pycache__/`, `dist/`, or `frontend/e2e/test-results/`.
- Never run `make perf-script-full` against production.

---

## 11. AI UX Hardening Loop (`/ux-loop`)

An AI-driven, persona-based critique→fix loop that drives the **live** app with Playwright,
finds UX/UI defects from screenshots + code, fixes them under strict test-first TDD, and
re-validates against live endpoints. Each cycle runs three agents: a **Demand Planner**
(acceptance-tests the planner workflows), a **Usability/Simplification** reviewer, and a
**Technical Fixer** (red→green TDD). Use it to push the UI toward best-in-class between releases.

### 11.1 Run it

Inside Claude Code, invoke the slash command (defined in `.claude/commands/ux-loop.md`):

```
/ux-loop 10        # run 10 cycles (the number is the cycle count)
/ux-loop           # defaults to 5 cycles
```

It stops early if **2 consecutive cycles** find no new P0/P1/P2 issues (the app is clean).
To force exactly N cycles, raise `dryStop` above N (see the command file).

### 11.2 What one cycle does

| Phase | Agent / step | Output |
|---|---|---|
| Capture | Playwright over 14 planner tabs → screenshots + visible text + console errors | `usertestinputs/cycleN/screens/`, `capture-digest.md` |
| Critique | Demand Planner + Usability reviewer (read screenshots + code) | `testinputN.md`, `usabilityN.md` |
| Fix | Technical Fixer — failing test → implement → green → refactor → live re-verify | `cycleN/fixes-applied.md` (with red→green evidence) |
| Ledger | Append issue → FIXED/DEFERRED | `usertestinputs/LEDGER.md` |

### 11.3 Prerequisite — host API must serve current code (CRITICAL)

The Docker API (`demandproject-api-1`) ships a **static image with no `--reload`**, so fixes
won't be live for the next cycle's scan unless a **host** uvicorn owns `:8000`. The command
sets this up automatically; to do it manually:

```bash
docker stop demandproject-api-1                                   # free :8000
~/.local/bin/uv run uvicorn api.main:app --reload --port 8000     # host API, hot-reload
curl -s -m3 http://localhost:8000/health                          # expect status: ok
```

Postgres/Redis stay in Docker and are reachable from the host via the defaults in
`common/core/db.py` (Postgres `localhost:5440`, db `demand_mvp`, user/pass `demand`/`demand`).
To revert: `docker start demandproject-api-1`. Vite must also be up on `:5173` (`make ui`).

### 11.4 Capture-only (no AI)

For just the Playwright capture + screenshots, without the critique/fix agents:

```bash
cd frontend && node ../usertestinputs/_harness/capture.mjs /abs/path/to/outdir
# writes <outdir>/screens/*.png, capture-digest.md, capture-dump.json
```

### 11.5 Outputs, review & hygiene

- The harness scripts live in `usertestinputs/_harness/` (`capture.mjs`, `improvement-loop.mjs`).
- Findings, `LEDGER.md`, and `SUMMARY.md` are kept; screenshots (`**/screens/`) and raw
  `capture-dump.json` are gitignored (`usertestinputs/.gitignore`, ~20MB of PNGs).
- The loop **edits the working tree but does not commit** — review with `git diff`, run
  `make test-all`, then commit when satisfied.

---

## 12. Quick Reference

```bash
# Dev loop
make ui-init                         # one-time npm install
make ui                              # Vite dev server :5173
make api                             # FastAPI :8000 (separate shell)

# Type sync
cd frontend && npm run gen:types     # regen schema.ts (API must be up)

# Tests
make test                            # backend (~0.7s)
make ui-test                         # frontend (~1.5s)
make test-all                        # both
make e2e                             # Playwright (needs API on :8000)

# Quality
make lint                            # ruff --fix
make format                          # ruff format
make type-check                      # mypy
cd frontend && npm run type-check    # tsc --noEmit

# Performance
make perf-report                     # full system, read-only
make perf-script SCRIPT=<name>       # one script, read-only
make perf-api                        # API endpoints
make perf-pipeline                   # ETL pipelines

# Maintenance
make audit-routers                   # router/proxy drift check

# AI UX hardening (Claude Code slash command)
/ux-loop 10                          # 10 live-app critique→TDD-fix→validate cycles
```
