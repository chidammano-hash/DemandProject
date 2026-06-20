# Scaling Assessment — DemandProject at 50× Data

**Date:** 2026-06-20
**Scope:** Evaluate the current stack (backend / database / frontend) for production scale,
where the current database holds **~1/50th** of production volume.
**Method:** Read-only review by three senior specialist perspectives (backend/data-access,
PostgreSQL/OLAP, frontend analytical-UI), grounded in `file:line` / table-level evidence and
real row counts from the running database. No code or data was modified.

---

## Verdict

**Keep the stack — evolve it, don't rewrite it.** Postgres 16 + FastAPI/psycopg3 + React/Vite are
the right choices for this workload, and roughly half the scaling work is already done (biggest
table partitioned, MVs refresh `CONCURRENTLY`, Redis single-flight cache, read replica + pg-queue +
chunked fact reads + lazy tabs + a virtualization pattern all exist).

The gap is **not architectural**. It is a handful of **"small-result-set" assumptions** that are
invisible at 1/50th data and become cliffs at 50×. There are **three P0 issues, one per layer, all
sharing that one root cause**, and none require new infrastructure — each leverages a primitive
already in the repo.

---

## Current scale → 50× projection (measured)

Engine: PostgreSQL 16 (`pgvector/pgvector:pg16`), single node, DB `demand_mvp`. Current size **36 GB**
→ projected **~1.8 TB at 50×**.

| Table | Rows now | On-disk | Partitioned? | ~50× rows | ~50× size |
|---|---|---|---|---|---|
| `fact_customer_demand_monthly` (46 monthly partitions) | **39.2 M** | 14 GB | ✅ range/month | ~2.0 B | ~700 GB |
| `fact_purchase_orders` | 6.0 M | 2.66 GB | ❌ | ~300 M | ~133 GB |
| `fact_lead_time_actuals` | 5.6 M | 1.34 GB | ❌ | ~278 M | ~67 GB |
| `fact_production_forecast_staging` | 4.7 M (7.8% dead) | 1.42 GB | ❌ | ~236 M | ~71 GB |
| `backtest_lag_archive` | 4.8 M | 1.41 GB | ❌ | ~241 M | ~70 GB |
| `agg_accuracy_lag_archive` (MV) | 2.8 M | 955 MB | ❌ (MV) | ~142 M | ~48 GB |
| `mv_ca_item_state` (MV) | 2.4 M | 501 MB | ❌ (MV) | ~121 M | ~25 GB |
| `fact_external_forecast_monthly` | 1.8 M (11.7% dead, 12 mo) | 794 MB | ❌ | ~89 M | ~40 GB |
| `dim_customer` | 1.0 M | 516 MB | n/a | ~50 M | ~26 GB |
| `dim_sku` | 324 K | 209 MB | n/a | ~16 M | ~10 GB |
| `fact_sales_monthly` | 305 K | 80 MB | ❌ | ~15 M | ~4 GB |

`fact_inventory_snapshot` and `fact_candidate_forecast` were empty in this snapshot; their 50× cost
is inferred from grain, not measured.

---

## The single root cause

At 1/50th data every result set is small — so unbounded API responses, unvirtualized tables, and
full-heap scans all *look* fine. At 50× the same code paths return tens of thousands of rows, scan
hundreds of millions, and mount thousands of DOM nodes. **The fix is to bound cardinality at every
layer:** cap + paginate at the API, partition + index at the DB, virtualize + downsample at the UI.

---

## Prioritized roadmap

### P0 — hard failures / cliffs (do first)

| # | Change | Layer | Effort | Impact |
|---|---|---|---|---|
| 1 ⭐ | **Fix the connection-pool math** (see backend C-1) so `workers × (sync+async+read) < max_connections`; make the preflight gate count all three pools; reconcile docs (20) vs code (50). | Backend | S | H |
| 2 ⭐ | **Range-partition the 4–5 unpartitioned hot fact tables** (PO, lead-time, staging, archive, external_forecast) via the existing `auto_create_partitions.py`. | DB | M | H |
| 3 ⭐ | **Make `limit`/`topN` mandatory + server-aggregate + virtualize** the four flagship surfaces (Accuracy Slice, Workbench, SQL Runner, Affinity). | FE+BE | M | H |
| 4 | **Move the PCA cluster scatter from recharts (SVG) to canvas echarts** + server-side point cap. | Frontend | S | H |
| 5 | **Add the `dim_sku (item_id, customer_group, loc)` composite covering index.** | DB | S | H |
| 6 | **Migrate multi-hour jobs (backtests, champion sweep, inventory pipeline, MV refresh) to pg-queue.** | Backend | M | H |

The single most urgent item is **#1** — it is a hard failure (`FATAL: too many connections`) that
triggers the moment concurrency rises, and the code default silently contradicts the documented gate.

### P1 — degradations (do next)

- **Cache the heavy accuracy/dashboard endpoints** — `@cached_async`/`@cached_sync` exist and are
  proven on Customer Analytics but used nowhere else. *(S · H)*
- **Convert hot read-only accuracy/dashboard handlers to `async def` + `get_async_read_only_conn()`**
  so they hit the replica and stop consuming anyio threadpool tokens. *(M · M)*
- **Incremental / per-partition refresh for the two big CA MVs** — full `REFRESH CONCURRENTLY` over a
  2 B-row base won't fit the nightly window; re-aggregate only the changed month. *(L · H)*
- **LTTB downsampling before recharts** (Item Analysis, Decomposition) and **move the Item Analysis
  4-map merge off the main thread** (repo has zero web workers). *(M · M)*
- **Drop duplicate `dim_sku` indexes; raise `max_wal_size` to 8–16 GB** for bulk loads. *(S · M)*

### P2 — hardening

- Replica-lag monitoring (`pg_last_wal_replay_lsn`) when the replica is enabled.
- Retention via `DROP PARTITION` instead of DELETE (once tables are partitioned).
- Gate `ClustersTab`'s unconditional 3 s poll on an active scenario.
- Schedule MV refresh on a fixed off-peak cadence with a stale-MV cache fallback.
- Drop the 4 `gin_trgm_ops` indexes on `fact_external_forecast_monthly` unless fuzzy search is used.

---

## Detailed findings

### Backend / data-access

**Strengths:** chunked fact reads (`read_sql_chunked`/`stream_query_in_chunks`,
`common/core/sql_helpers.py`, `DEFAULT_CHUNK_SIZE=50_000`) adopted in the 4 heavy ML/forecast scripts;
Redis single-flight stampede protection (`common/services/cache.py:220-279`) with graceful in-memory
fallback; MV fast-path for CA (`/kpis` 10.8 s → 63 ms); `statement_timeout=30000` + fail-fast pool
acquire (`api/pool.py`); async + read-replica plumbing that falls back bit-identically.

| Sev | Issue | Evidence | Why it breaks at 50× |
|---|---|---|---|
| **Crit** | Connection-pool math overcommits Postgres | `api/pool.py:99,171,213,292` (defaults `50`); `Dockerfile:26` (`GUNICORN_WORKERS=4`); `docker-compose.yml:14` (`max_connections=100`); gate `Makefile:51-60` assumes `20` and ignores async/read pools | 4 workers × (sync+async+read) backends ≫ 100 → `FATAL: too many connections`, cascading acquire timeouts. Documented invariant is false. |
| **Crit** | Multi-hour ML jobs on APScheduler's 4-thread pool, not pg-queue | `common/services/job_scheduler.py:29` (`ThreadPoolExecutor(max_workers=4)`); `scripts/ops/pg_queue_worker.py:61-63` (only `refresh_intramonth`); `job_scheduler.py:31-34` (`coalesce=True`) | One ~30 h foundation backtest pins 1 of 4 threads for a day; 4 such jobs starve all scheduled work; missed MV refreshes silently dropped → stale dashboards. Violates the "no multi-hour jobs on APScheduler" rule. |
| **High** | Heavy accuracy/dashboard endpoints not cached | decorators `cache.py:414-462`; 0 `@cached_*` in `api/routers/forecasting/` or `core/`; accuracy/dashboard handlers sync `def` | Every hit re-runs a full MV/fact aggregation; sub-second → multi-second at 50×; repeated loads saturate the threadpool. Fix infra already exists. |
| **High** | Per-DFU + decomposition/slice endpoints return unbounded rows | `production_forecast.py:365-419` (no LIMIT); `accuracy.py:860-986`, `:125`; `inventory/demand_history.py:162` | History depth × model count both grow at 50× → payloads balloon; JSON + `row_to_dict` on a sync thread. `/error-contributors` (`accuracy.py:987`) shows the capped pattern to copy. |
| **Med** | Sync handlers in hot routers consume anyio threadpool tokens | `api/main.py:44-56` (100 tokens shared); accuracy/dashboard sync `def` | At 50× latencies, request bursts exhaust tokens and queue, even though async + replica plumbing exists. |
| **Med** | MVs refresh on the primary, on load, no scheduled cadence | `Makefile:1677-1703`; `scripts/etl/load.py:177-220` | Full tiered refresh is multi-minute at 50× and competes with live load; a skipped load silently regresses CA/accuracy endpoints to 10 s+ with no fallback cache. |

### Database (PostgreSQL / OLAP)

**Strengths:** biggest table (`fact_customer_demand_monthly`, 39 M) is monthly range-partitioned with a
registry-driven auto-partition manager (`scripts/db/auto_create_partitions.py`); all 8 hot MVs carry
unique indexes → `REFRESH … CONCURRENTLY` works (`sql/119`); Postgres genuinely OLAP-tuned
(`shared_buffers=4G`, `work_mem=128M`, `maintenance_work_mem=1G`, parallel query, `random_page_cost=1.1`,
JIT, `wal_level=replica`); BRIN on append-mostly `fact_purchase_orders`; conservative replica fallback.

| Sev | Issue | Evidence | Why it breaks at 50× |
|---|---|---|---|
| **Crit** | 4 heaviest non-demand fact tables not partitioned | `pg_partitioned_table` = 2 parents; `auto_create_partitions.py:94-109`. PO (→300 M), lead-time (→278 M), staging (→236 M), archive (→241 M) flat heaps | Full-heap autovacuum/ANALYZE/index-rebuild/DELETE-retention; no pruning; vacuum/refresh windows blow past the batch window. |
| **Crit** | `fact_external_forecast_monthly` unpartitioned AND reloaded in place (11.7% dead at 12 mo) | `sql/007` (no PARTITION); `pg_stat_user_tables` 1.77 M live / 207 K dead | Replace-load churns the whole heap; at 89 M rows the load + autovacuum dominates the window. It's the accuracy/WAPE source table. |
| **High** | `dim_sku` has NO `(item_id, customer_group, loc)` composite index | `pg_indexes` (only single-col `item_id`, `loc`); join `accuracy.py:228-229, 340-341, 545-546` | At 16 M dim rows the planner hash-joins the full dimension or single-col scan + filter; accuracy/FVA + MV refreshes degrade sharply. |
| **High** | Big CA MVs refresh by full recompute | `pg_depend` (both depend on `fact_customer_demand_monthly`); `Makefile:1689-1702` whole-MV `REFRESH CONCURRENTLY` | Full refresh = scan all 2 B rows + sort/agg + diff-merge before swap; multi-hour, doubles disk transiently, serializes against the next refresh. |
| **Med** | Duplicate/unused `dim_sku` indexes | `pg_stat_user_indexes`: `_abc_xyz` ≡ `_abc_xyz_segment`, `_xyz` ≡ `_xyz_class` (identical, `idx_scan=0`) | ~36 GB of dead indexes at 50× → write amplification + longer vacuum, zero read benefit. `make db-drop-unused-indexes` exists. |
| **Med** | `max_wal_size=1 GB` too small for bulk loads | `pg_settings` | Multi-hundred-M-row COPY + MV/index builds force checkpoint thrashing; remedy is 8–32 GB. |
| **Med** | Replication lag at write-heavy 50× | `api/pool.py:125-140`; write-bursty loads | Replica behind a 2 B-row reload + MV churn can lag minutes; CA reads tolerate it, but monitor + alert. |

**Engine verdict:** stay on single-node row-store Postgres 16 to ~50× **if P0/P1 land** — the constraint
is vacuum/refresh/retention windows on flat heaps, not raw latency, and partitioning fixes exactly that.
Defer Citus/columnar (`pg_mooncake`/DuckDB offload) until proven necessary by full-history ad-hoc scans
over the 2 B-row demand table; if ever needed, the pragmatic move is a read-optimized columnar *copy* of
that one table, not a wholesale migration.

### Frontend (analytical UI)

**Strengths:** every tab `React.lazy()` (`App.tsx:35-56`); LazyPanel/IntersectionObserver on all 9
below-fold CA panels (`CustomerAnalyticsTab.tsx:321-400`) and 31 InvPlanning panels; echarts-modular
boundary respected (canvas for heavy CA charts, recharts elsewhere) with manual vendor chunking
(`vite.config.ts:90-108`); sane React Query defaults (`staleTime 120s`, 4xx-aware retry, mostly
activity-gated polling); a correct `useVirtualizer` reference (`CustomerRanking.tsx:19-72`);
server-side pagination on inventory/supply panels (`PAGE=50`).

| Sev | Issue | Evidence | Why it breaks at 50× |
|---|---|---|---|
| **Crit** | Accuracy Slice: unbounded fetch + non-virtualized render + per-row `flatMap` | fetch no limit `accuracy.ts:30-48`; `sliceData.map(...)` `SliceTablePanel.tsx:263`; `flatMap` `AggregateAnalysisTab.tsx:304-305` | Slice grain explodes ~50×; tens of thousands of `<tr>` + a main-thread `flatMap` on every change → multi-second lock on a flagship tab. |
| **Crit** | PCA cluster scatter on recharts (SVG), one `<circle>` per SKU | `ScenarioCharts.tsx:23-26,221-259` | ~50× points → 10k–50k SVG nodes; ~1–2 s render + O(n) hover hit-test. Exactly the canvas case. |
| **Crit** | Demand History Workbench fetches full grain (limit/offset `undefined`) | `WorkbenchPanel.tsx:309-312`; params unused `demand-history.ts:157-170` | At `item_loc_customer` grain returns the entire DFU universe (100k+) as one payload + renders every row → freeze. |
| **Crit** | SQL Runner sends no `max_rows`, renders all rows unvirtualized | `SqlRunnerTab.tsx:280-281`; `sql-runner.ts:41-46`; `:426` | `SELECT *` at 50× returns up to the server cap; full array to DOM → jank / OOM risk. |
| **High** | Explorer table non-virtualized; page size up to 500 | `ExplorerTable.tsx:124` | 500 × N cells re-rendered per page/sort/filter — janky (data itself is paginated). |
| **High** | Customer-Item Affinity: unbounded fetch + client-side downsample | `customer-analytics.ts (~507)`; `CustomerItemAffinity.tsx:47-66` | Quadratic matrix; client sorts/filters millions of cells before echarts draws. topN belongs server-side. |
| **High** | Item Analysis 4-source merge on main thread + undownsampled recharts | `ItemAnalysisTab.tsx:301-393` | Merge grows with history×models; recharts renders 300+ pts × 15 series with no LTTB. |
| **Med** | Demand History Matrix double-`map` grid + O(n²) max scan, not virtualized | `MatrixPanel.tsx:166,41-49` | item×location cells grow ~50× → thousands rendered + scanned on main thread. |
| **Low/Med** | `ClustersTab` polls every 3 s unconditionally | `ClustersTab.tsx:93` | Fixed poll (not gated on a running job) = steady heap + API churn at 50× payloads. |

**Principle:** push aggregation/pagination server-side; virtualize every unbounded table (reuse
`CustomerRanking`'s `useVirtualizer`); canvas for high-cardinality charts; workers (or server pre-merge)
for heavy client compute.

---

## Assumptions & caveats

- **50× = data volume.** No concurrency multiplier was assumed. If users/concurrency also scale, P0 #1
  and the caching/async items rise in urgency.
- **Doc/code drift to reconcile (part of P0 #1):** `CLAUDE.md` / `MEMORY.md` state `POOL_MAX_SIZE=20`,
  but `api/pool.py` defaults to `50` and adds independent async + read pools the preflight gate never
  counts. The documented invariant is currently false.
- Two hot fact tables (`fact_inventory_snapshot`, `fact_candidate_forecast`) were empty at review time;
  their 50× cost is inferred from grain.

---

## Suggested sequencing

1. **Week 1 (quick, isolated, high-impact):** P0 #1 connection-pool fix, #5 `dim_sku` composite index,
   #4 PCA-scatter canvas swap. All small and independently shippable, test-first.
2. **Weeks 2–3:** P0 #2 partitioning (backfill migration + register in auto-partition manager) and P0 #3
   API caps + virtualization on the four flagship surfaces.
3. **Weeks 3–4:** P0 #6 job migration to pg-queue, then the P1 caching/async/MV-incremental work.
4. **Ongoing:** P2 hardening + observability (replica lag, MV freshness, pool saturation metrics).
