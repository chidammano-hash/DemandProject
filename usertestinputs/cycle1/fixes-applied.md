# Cycle 1 — Fixes Applied

_Engineer pass on branch `restructure`. Date: 2026-06-14._

## Environment note (read first)

The dockerized API (`demandproject-api-1`) runs **gunicorn with no source volume
mount and no `--reload`**, so host-side edits do NOT hot-reload into that
container. All live verification below was therefore run against a **local
uvicorn process on `127.0.0.1:8001`** started from the working tree, pointed at
the same Postgres (`localhost:5440`, db `demand_mvp`, user `demand`). The
docker container still serves the old code on `:8000`; restart/redeploy it to
pick up these fixes (`docker compose restart api` or rebuild).

---

## F1.1 — Action Feed shows 0 actions despite 6,142 open critical exceptions  [P0] — FIXED

**What was wrong (two bugs):**
1. Source 1 of `get_action_feed` selected a non-existent column `created_at`
   from `fact_replenishment_exceptions` (the table has `exception_date` /
   `load_ts`). The query raised `UndefinedColumn`.
2. All three source queries shared **one cursor on one transaction**. When
   Source 1 raised, the transaction aborted, so Sources 2 (planned orders) and
   3 (stockout-risk MV) failed with "current transaction is aborted" — the feed
   silently returned `{actions:[], total:0}` (HTTP 200).

**Fix** — `api/routers/inventory/inv_planning_insights.py` (`get_action_feed`):
- Source 1 SELECT `created_at` → `exception_date` (verified against the live
  DDL, not guessed).
- Wrapped **each source** in `async with conn.transaction()` (a psycopg3
  SAVEPOINT) so a failure in one source rolls back only that savepoint and the
  remaining sources still execute. `fetchall()` is pulled inside the savepoint;
  row mapping happens after, unchanged.

**Verification (local API):**
- Before: `curl /inv-planning/action-feed` → `{"actions":[],"summary":{"total":0,...}}`; log showed `column "created_at" does not exist` + 2× `transaction is aborted`.
- After: `curl /inv-planning/action-feed?limit=20` → 20 actions, `summary={total:20, critical:20, financial_at_risk:3598.89}`; **zero** errors in the API log.
- DB cross-check: `fact_replenishment_exceptions WHERE status='open'` = 6,142 rows; top action item_ids/values reconcile with the exceptions list.

**Tests added** — `tests/api/test_action_feed.py`:
- `test_action_feed_returns_open_exceptions` — happy path, summary counts.
- `test_action_feed_source_isolation` — Source 1 raises `UndefinedColumn`,
  asserts Source 2's planned order still lands in the feed (regression guard
  for the transaction-isolation fix).
- conftest: `make_async_pool` now mocks `conn.transaction()` as a no-op async
  context manager so the SAVEPOINT pattern is exercisable in tests.

**Acceptance criterion:** MET — feed returns open critical exceptions ranked by
urgency + financial impact; non-zero Total/Critical; no column / aborted-txn
warnings; each source runs on its own statement.

---

## F1.2 — Command Center health KPIs blank + 500 (missing MV)  [P0] — FIXED (graceful) + migration applied

**What was wrong:** `GET /control-tower/kpis` queries `mv_control_tower_kpis`,
whose migration (`sql/035_create_control_tower_kpis.sql`) had never been
applied — `to_regclass` was NULL → `UndefinedTable` → HTTP 500. The handler
only caught `ObjectNotInPrerequisiteState` (empty MV), **not** `UndefinedTable`
(missing MV).

**Fix:**
- `api/routers/operations/control_tower.py` (`get_control_tower_kpis`): the
  catch now handles `(ObjectNotInPrerequisiteState, UndefinedTable)`, degrading
  to the existing neutral empty payload + `warning` instead of 500.
- Applied the missing migration `sql/035_create_control_tower_kpis.sql` to
  `demand_mvp` so the object exists (`to_regclass('mv_control_tower_kpis')` now
  resolves). The MV is created `WITH NO DATA` (its source
  `mv_inventory_health_score` is itself unpopulated), so the endpoint now hits
  the already-graceful `ObjectNotInPrerequisiteState` path.

**Verification (local API):**
- Before: `curl /control-tower/kpis` → HTTP 500 (`relation "mv_control_tower_kpis" does not exist`).
- After: HTTP 200, payload `health.total_dfus=0` + `warning: "mv_control_tower_kpis not yet refreshed..."`. No 500, no traceback.

**Tests added** — `tests/api/test_control_tower.py::test_control_tower_kpis_handles_missing_mv`
(asserts `UndefinedTable` → 200 + warning).

**Acceptance criterion:** PARTIALLY MET. The 500 and the false "Portfolio looks
healthy!" lie are gone — the screen now degrades honestly (empty + warning).
Rendering **real** numbers requires populating the upstream MVs (operational
step: `make refresh-mvs-tiered` after the inventory/health pipeline runs). That
is data/pipeline state, not a code defect, so it is intentionally left to the
operator.

---

## F1.3 — fill-rate / inventory trend endpoints 500 on unpopulated MVs  [P1] — FIXED (graceful)

**What was wrong:** `/control-tower/trend` already degraded gracefully, but the
sibling trend endpoints 500'd on unpopulated MVs because their `with get_conn()`
blocks had no exception handling.

**Fix** — same `(ObjectNotInPrerequisiteState, UndefinedTable)` catch →
`{<data>: [], warning: "...refresh-mvs-tiered"}`, matching `/control-tower/trend`:
- `api/routers/operations/fill_rate.py` — `get_fill_rate_trend`.
- `api/routers/inventory/inventory_main.py` — `inventory_trend` (added
  `logging` + `psycopg` imports; extracted the SS/EOQ/policy row-mapping into a
  `_set_inv_params` helper to keep the try-block tidy).
- `api/routers/inventory/inv_backtest.py` — `inv_backtest_trend` (added
  `logging` + `psycopg` imports).

**Verification (local API):**
- Before: `/fill-rate/trend`, `/inventory/trend`, `/inventory-backtest/trend` → HTTP 500.
- After: all → HTTP 200 with `{months|trend: [], warning: "Upstream materialized view not yet refreshed..."}`. Zero tracebacks.

**Tests added:**
- `tests/api/test_fill_rate.py::test_fill_rate_trend_handles_unpopulated_mv`
- `tests/api/test_inventory.py::test_inventory_trend_handles_unpopulated_mv`
- `tests/api/test_inventory_backtest.py::test_inventory_backtest_trend_handles_unpopulated_mv`

**Acceptance criterion:** MET for the degradation half — these endpoints now
return `{data:[], warning}` instead of 500. Returning populated trends still
requires `make refresh-mvs-tiered` (operational).

---

## F1.4 — Item Analysis side-panel 500s  [P1] — FIXED (root 500 isolated)

**What was wrong:** The console 500s on the Item Analysis screen traced to
`GET /inventory/kpis` (the "always-fetched" DFU attributes call). It runs 3
queries: Query 1 reads `fact_inventory_snapshot` (a base table, fine); Queries
2 & 3 read `agg_inventory_monthly` (an unpopulated MV) and 500'd the whole
endpoint. The SHAP panel was already graceful (its sku-shap 404 → cluster-level
fallback) and `/data-quality/corrections`, lt-profile, shap-summary/timeframes/
clusters all returned 200 — so `inventory/kpis` was the real defect.

**Fix** — `api/routers/inventory/inventory_main.py` (`inventory_kpis`): run
Queries 2 & 3 inside a `conn.transaction()` SAVEPOINT; on
`(ObjectNotInPrerequisiteState, UndefinedTable)` keep the Query-1 snapshot
totals and degrade the MV-derived KPIs (DOS, turns, lead-time coverage) to
neutral `null`. No 500.

**Verification (local API):**
- Before: `/inventory/kpis?item=133716&location=1401-BULK` → HTTP 500 (`agg_inventory_monthly has not been populated`).
- After: HTTP 200 → `{total_on_hand:69.83, total_on_order:154.0, dos:null, inventory_turns:null, last_snapshot_date:"2026-04-30", ...}`. Snapshot totals render; MV-derived KPIs are honest nulls.
- Full item-analysis endpoint sweep for the DFU now returns 200 (or graceful 404-fallback for sku-shap); **zero** tracebacks in the log.

**Tests added** — `tests/api/test_inventory.py::test_inventory_kpis_degrades_when_agg_mv_unpopulated`
(Query 1 succeeds, Queries 2/3 raise; asserts snapshot totals survive and
MV-derived KPIs are null, no 500).

**Acceptance criterion:** MET — the SHAP/forecast view works and the secondary
panels no longer throw a 500 toast; they show neutral/empty values until the
inventory MV is refreshed.

---

## F1.7 — Clusters tab "setState during render" warning  [P2] — FIXED

**What was wrong:** `ClustersTab` called `onDomainChange("sku")` **in the
component render body** (top-level), which updates the parent `App` state during
ClustersTab's render → React "Cannot update a component (App) while rendering a
different component (ClustersTab)" warning.

**Fix** — `frontend/src/tabs/ClustersTab.tsx`: moved the domain-sync into a
`useEffect(() => { if (domain !== "sku") onDomainChange("sku"); }, [domain, onDomainChange])`.

**Verification:** `ClustersTab.test.tsx` — 8/8 pass; the only remaining console
warning is a pre-existing unrelated `validateDOMNesting (<button> in <button>)`.
The "setState in render" / "Cannot update a component while rendering" warning
is gone.

**Acceptance criterion:** MET — no setState-in-render warning on mount; the
empty state ("Run the clustering pipeline") remains as-is.

---

## Deferred (not fixed this cycle)

- **F1.2 / F1.3 data population** — rendering *real* Command Center numbers and
  populated trends requires running the inventory/health/fill-rate pipelines and
  `make refresh-mvs-tiered`. This is operational state, not a code bug; the code
  now degrades honestly. Left to the operator.
- **F1.5 — negative accuracy heatmap (BEER −263%)** [P1, presentation]. Not a
  crash; a UI-floor/annotation change on the accuracy heatmap + cluster table.
  Deferred to keep this cycle focused on the P0/P1 *errors*; needs a frontend
  design decision (floor at 0% + "low base" note vs. legend explainer).
- **F1.6 — DQ "Run Checks Now" button wiring** [P2]. Requires a new
  `POST /data-quality/run` trigger button + write-endpoint plumbing; the empty
  state is already good. Deferred (net-new feature, not a defect).
- **F1.8 — FVA Champion "No data"** [P2]. Genuinely empty
  (`fact_candidate_forecast` = 0 rows); needs a champion backtest run, not a
  code change.
- **F1.9 — S&OP in-app "New Cycle" button** [P3]. Net-new feature.
- **F1.10** — harness artifact, not a product defect.

## Risk / notes

- **psycopg3 SAVEPOINT semantics:** the per-source `conn.transaction()` blocks
  are nested savepoints inside the pool connection's transaction; a failing
  source rolls back to its savepoint only. This is the intended psycopg3
  pattern and is exercised by the new isolation tests.
- **No data invented.** Every fix either corrects a query, applies an existing
  migration, or degrades to an honest empty/neutral state with an actionable
  `warning`. No placeholder/fake numbers were introduced.
- **Lint:** ruff error counts on all 5 edited production routers are identical
  to HEAD (no new violations introduced). Pre-existing style debt in those
  large files was left untouched.
- **Pre-existing test failures unrelated to this work:** `tests/api/test_fva.py`
  (3 failures, `IndexError` in `api/routers/forecasting/fva.py:130`) fail
  identically on clean HEAD; and `tests/unit/test_backtest_chronos.py` fails to
  collect (`No module named 'torch'`). Both predate this cycle and were not
  touched.
- **Deployment:** the docker API must be restarted/redeployed to serve these
  fixes (it has no source mount). Verified locally on :8001 in the meantime.

## Test summary

- Backend (touched suites): `test_action_feed`, `test_control_tower`,
  `test_fill_rate`, `test_inventory`, `test_inventory_backtest`,
  `test_daily_briefing` → **70 passed**.
- Full API suite: **1239 passed**, 3 pre-existing FVA failures (unrelated).
- Frontend: `ClustersTab.test.tsx` 8 passed; `InvPlanningTab` + `ControlTowerTab`
  37 passed.
