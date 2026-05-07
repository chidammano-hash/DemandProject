# Section 8 — Operations: S&OP, Control Tower, Events, Storyboard, DQ & Jobs

This section describes how to operate the cross-functional planning and execution
modules that sit on top of the demand and inventory cores: Sales & Operations
Planning (S&OP), the Inventory Control Tower, Promotion & Event Planning, the
Planner Storyboard, the Data Quality (DQ) engine, the APScheduler-based job
engine, and the notification + webhook fan-out.

All Make targets, scripts, and routers referenced below were verified against
the current `restructure` branch.

---

## 8.1 Module Map

| Module | API router | UI tab | Primary Make target |
|---|---|---|---|
| S&OP | `api/routers/operations/sop.py` (mounted at `/sop`) | `frontend/src/tabs/SopTab.tsx` | `make sop-all` |
| Control Tower | `api/routers/operations/control_tower.py` (mounted at `/control-tower`) | `frontend/src/tabs/ControlTowerTab.tsx` | `make control-tower-all` |
| Promotion / Event Planning | `api/routers/operations/events.py` (mounted at `/events`) | `EventCalendarPanel` inside `InvPlanningTab` | `make events-all` |
| Planner Storyboard | `api/routers/intelligence/storyboard.py` (mounted at `/storyboard`) | `frontend/src/tabs/StoryboardTab.tsx` | `make storyboard-all` |
| Data Quality | `api/routers/platform/data_quality.py` + `common/engines/dq_engine.py` | DQ surfaced inside ControlTower / Storyboard | `make dq-all` |
| Job Engine | `api/routers/core/jobs.py` | `frontend/src/tabs/JobsTab.tsx` | n/a (runtime) |
| Notifications | `api/routers/platform/notifications.py` + `common/services/notification_engine.py` | `SettingsTab` channel config | n/a (runtime) |
| Webhooks | `api/routers/platform/webhooks.py` + `common/services/webhook_dispatcher.py` | `SettingsTab` registrations | n/a (runtime) |

The umbrella target that bootstraps every operational module from a clean
database is:

```bash
make setup-ops
# expands to: sop-all events-all financial-plan-all storyboard-all scenarios-all dq-all
```

`setup-ops` is the final phase of `make setup-all` and assumes
`setup-data`, `setup-features`, `setup-backtest`, and `setup-inv-planning`
have already completed (Control Tower MVs are produced by `setup-inv-planning`).

---

## 8.2 S&OP Module

### 8.2.1 What it computes

`fact_sop_cycles` drives a six-stage workflow:

```
demand_review → supply_review → pre_sop → executive_sop → approved → closed
```

(See `STAGE_ORDER` in `api/routers/operations/sop.py`.) A cycle materialises the
following downstream tables:

| Table | Purpose |
|---|---|
| `fact_sop_cycles` | one row per planning cycle (typically one per month) |
| `fact_sop_demand_review` | snapshot of demand at the demand_review stage |
| `fact_sop_supply_constraints` | supply-side commitments captured during supply_review |
| `fact_sop_gaps` | demand-vs-supply gap analysis surfaced in pre_sop |
| `fact_sop_approved_plan` | locked, executive-approved monthly plan |
| `fact_sop_decisions` | audit trail of stage transitions and approvals (see `sql/147`) |

### 8.2.2 Operating commands

```bash
make sop-schema                              # Apply sql/056_create_sop_module.sql
make sop-seed                                # Create the cycle for the current month
make sop-all                                 # sop-schema + sop-seed
make sop-create CYCLE_MONTH=2026-05-01       # Create a specific monthly cycle
make sop-populate CYCLE_ID=<uuid>            # Populate demand snapshot
make sop-advance CYCLE_ID=<uuid>             # Advance to next stage
```

All four `sop-*` action commands shell out to `scripts/run_sop_cycle.py`.

### 8.2.3 API surface

| Endpoint | Purpose | Auth |
|---|---|---|
| `GET /sop/cycles` | Paginated list of cycles | no |
| `GET /sop/cycles/{cycle_id}` | Full cycle detail (stages, gaps, approved plan) | no |
| `POST /sop/cycles/{cycle_id}/advance` | Move to the next stage | yes |
| `POST /sop/cycles/{cycle_id}/approve` | Lock approved demand into `fact_sop_approved_plan` | yes |
| `GET /sop/cycles/{cycle_id}/gaps` | Demand vs supply gap report | no |
| `GET /sop/approved-plan` | Read the active approved plan | no |

Write endpoints are guarded by `require_api_key`; the `SopTab` UI passes the
configured API key automatically.

---

## 8.3 Control Tower

The Control Tower is a **read-only** aggregator. It does not compute anything
of its own — it stitches together five upstream MVs into a single nested KPI
payload.

### 8.3.1 Materialized views consumed

| MV | Source target | Refreshes via |
|---|---|---|
| `mv_inventory_health_score` | `make health-all` | `scripts/refresh_health_scores.py` |
| `mv_fill_rate_monthly` | `make fill-rate-all` | dedicated refresh target |
| `mv_supplier_performance` | `make supplier-perf-all` | dedicated refresh target |
| `mv_intramonth_stockout` | `make intramonth-all` | `scripts/refresh_intramonth_stockout.py` |
| `mv_control_tower_kpis` | `make control-tower-all` | direct REFRESH |

`make refresh-mvs-tiered` refreshes them in dependency order; always use that
target rather than refreshing individual MVs ad-hoc.

### 8.3.2 KPI payload shape

`GET /control-tower/kpis` returns a strict five-key nested object — this exact
shape is what frontend tests mock and what the `ControlTowerTab` UI expects:

```json
{
  "computed_at": "2026-04-28T03:14:00Z",
  "health":         { "total_skus": ..., "healthy_count": ..., "avg_health_score": ..., "below_ss_count": ..., "avg_portfolio_dos": ... },
  "exceptions":     { "open_exceptions_total": ..., "critical_exceptions": ..., "high_exceptions": ..., "recommended_order_value": ... },
  "fill_rate":      { "portfolio_fill_rate_3m": ..., "total_shortage_qty_3m": ... },
  "demand_signals": { "urgent_demand_signals": ..., "projected_stockouts_today": ... },
  "intramonth":     { "items_with_stockout_this_month": ..., "extended_stockouts_this_month": ... }
}
```

If `mv_control_tower_kpis` has never been refreshed, the endpoint returns the
shape above with zero/null fields plus a `warning` instructing the operator to
run `make refresh-mvs-tiered`. Do **not** strip the warning — the UI surfaces
it to planners.

When writing tests for any tab that consumes the Control Tower payload, mock
the full nested structure (see `frontend/src/tabs/__tests__/ControlTowerTab.test.tsx`
for the canonical mock).

### 8.3.3 UI tab

`ControlTowerTab.tsx` renders four banner KPI cards (health, fill rate,
exceptions, intramonth), a portfolio health table, and a 3-month trend chart.
Cache hint is 120s on the API side; the UI uses `react-query` with the
default stale time, so a hard refresh on the tab simply re-reads the cached
MV and is safe at any time.

---

## 8.4 Events (Promotion & Event Planning)

### 8.4.1 What triggers an event

Events live in `fact_event_calendar` and are created either through the UI
(`EventCalendarPanel` inside the `InvPlanningTab` "Events" sub-tab) or via the
REST API. An event has:

- `event_type` (e.g. promo, holiday, NPI launch)
- `event_start` / `event_end` window
- `uplift_pct` (multiplier applied to baseline forecast)
- `ramp_weeks` (linear ramp-in)
- `pantry_loading_pct` + `pantry_loading_weeks` (post-event pull-back)
- `priority` and `status` (`draft` → `approved`)

### 8.4.2 How adjustments are applied

Once an event is approved, `make events-apply` runs
`scripts/apply_event_adjustments.py`, which writes adjusted forecast rows
keyed by event and exposes them through the impact-preview endpoint.
`events-apply-dry` does the same calculation without writing.

| Endpoint | Purpose | Auth |
|---|---|---|
| `GET /events/calendar` | List events (filters: type, status, date range) | no |
| `POST /events/calendar` | Create a draft event | yes |
| `GET /events/calendar/{event_id}` | Detail view | no |
| `PUT /events/calendar/{event_id}/approve` | Approve and queue for adjustment | yes |
| `GET /events/impact-preview` | Demand adjustment preview | no |
| `GET /events/performance` | Post-event lift accuracy | no |

### 8.4.3 Operating commands

```bash
make events-schema       # Apply sql/057_create_event_planning.sql
make events-apply        # Apply approved event adjustments to the forecast
make events-apply-dry    # Preview without writes
make events-all          # events-schema + events-apply
```

---

## 8.5 Planner Storyboard

The Storyboard is the **AI-driven narrative layer** — it converts numeric
exceptions into human-readable headlines, ranks them by severity and financial
impact, and tracks every planner decision.

### 8.5.1 Auto-generated insights

`scripts/generate_storyboard_exceptions.py` (invoked by `make storyboard-generate`)
calls `common/engines/exception_engine.run_exception_detection` which produces
exceptions with these attributes:

- `exception_type` (bias, dos_breach, fill_rate_drop, signal_spike, …)
- `severity` (0–1 score) → mapped to band via `derive_severity_band`
  (`critical ≥ 0.75`, `high ≥ 0.50`, `medium ≥ 0.25`, else `low`)
- `financial_impact` (USD) — used for sorting
- `headline` (human-readable summary, AI-assisted)
- `supporting_data` (JSON blob of the metrics that triggered detection)
- `root_cause_key` (deterministic hash, groups repeat occurrences)
- `sla_due_at` (response deadline derived from `compute_sla_due_at`:
  critical=4h, high=24h, medium=72h, low=168h)

### 8.5.2 API surface

| Endpoint | Purpose | Auth |
|---|---|---|
| `GET /storyboard/exceptions` | Paginated list with filters | no |
| `GET /storyboard/exceptions/summary` | KPI counts by type/status, top items | no |
| `GET /storyboard/exceptions/{id}` | Detail + decision history | no |
| `PUT /storyboard/exceptions/{id}/status` | Update status (open/investigating/resolved/dismissed) | yes |
| `POST /storyboard/exceptions/{id}/decide` | Record a planner decision | yes |
| `GET /storyboard/decisions` | Recent planner decisions log | no |
| `POST /storyboard/generate` | Manually trigger a generation cycle | yes |

### 8.5.3 Operating commands

```bash
make storyboard-schema         # Apply sql/038_create_storyboard.sql
make storyboard-generate       # Run a fresh detection cycle for current month
make storyboard-generate-dry   # Detect but do not insert
make storyboard-all            # storyboard-schema + storyboard-generate
```

The `StoryboardTab.tsx` UI groups exceptions by `root_cause_key` so repeat
issues collapse into a single card with an occurrence count.

---

## 8.6 Data Quality (DQ) Engine

### 8.6.1 What checks run

`common/engines/dq_engine.py` exposes a `DQEngine` class that loads check
definitions from `config/platform/data_quality_config.yaml` and persists them in
`fact_dq_check_definitions`. Four primitive check types are implemented:

| Check type | Implementation | Failure condition |
|---|---|---|
| `completeness` | `_check_completeness` | NULL pct of column > `max_null_pct` |
| `row_count` | `_check_row_count` | Table row count < `min_rows` |
| `uniqueness` | `_check_uniqueness` | Any duplicate group on `key_columns` |
| `range` | `_check_range` | Any value outside `[min_val, max_val]` |

Each run writes a row to `fact_dq_check_results` with `status`
(`pass` / `warn` / `fail` / `skip`), `metric_value`, and a JSON `details` blob.
Aggregated domain health scores (`dim_dq_domain_score`) feed the Control Tower
"data quality" tile.

### 8.6.2 Operating commands

```bash
make dq-schema     # Apply sql/063_create_data_quality.sql
make dq-populate   # Seed check definitions from config/platform/data_quality_config.yaml
make dq-run        # Execute all enabled checks and persist results
make dq-all        # dq-schema + dq-populate + dq-run
```

`make dq-run` is the safe everyday command — it only reads source tables and
writes to `fact_dq_check_results`. Re-run it after any `pipeline-refresh` to
keep the Control Tower DQ tile current.

### 8.6.3 DQ fix scripts

When a check fails, the operator-facing remediation script is:

```bash
uv run python scripts/fix_dq_issues.py --check <check_name>
```

This script is **interactive and idempotent** — it prints the proposed fix,
asks for confirmation, and only then applies it. Common remediations include
backfilling NULL `customer_no` from upstream joins, deduplicating duplicate
`forecast_ck` rows from re-loads, and clamping negative `demand_qty`.

### 8.6.4 Common false positives

| Symptom | Likely cause | Action |
|---|---|---|
| `row_count` fails on `fact_inventory_snapshot` | A new monthly partition has not yet been loaded | Run `make load-inventory` for the missing month before re-running `dq-run` |
| `completeness` fails on `customer_no` for a recent month | Customer demand load skipped a partition | `make load-customer-demand-month MONTH=YYYY-MM` |
| `range` warns 0% but flags `outliers=0` | Empty source table — produces a `warn`, not a `fail` | Safe to ignore until the table is populated |
| `uniqueness` fails on `forecast_ck` after a re-run | Duplicate insert path — investigate before fixing | Run `scripts/fix_dq_issues.py --check forecast_uniqueness` |

---

## 8.7 Job Scheduling (APScheduler)

### 8.7.1 Architecture

- `common/services/job_scheduler.py` is the thin APScheduler factory.
  `make_scheduler()` returns a started `BackgroundScheduler` with:
  - `ThreadPoolExecutor(max_workers=4)`
  - `job_defaults = {coalesce: True, max_instances: 1, misfire_grace_time: 3600}`
  - `timezone = UTC`
- `common/services/job_registry.py` exposes the public `JobManager` singleton
  (accessed via `JobManager.instance()`) and `JOB_TYPE_REGISTRY` — the
  catalogue of every runnable job type. Job implementations are imported from
  `common/services/job_state.py` (`_run_*` callables).
- `api/main.py` lifespan handler **pre-warms** `JobManager.instance()` on
  startup so APScheduler is live before the first `/jobs` request:

  ```python
  from common.services.job_registry import JobManager
  JobManager.instance()
  logger.info("APScheduler BackgroundScheduler started on startup")
  ```

  Scheduler init is best-effort — if it fails, the API still serves read
  endpoints but `/jobs` writes return errors. Check the lifespan log line
  on every restart.
- Jobs persist state in PostgreSQL (`fact_job_run`, `fact_job_schedule`,
  `fact_job_pipeline`) so restarts do not lose history.

### 8.7.2 Concurrency model

- One active job per **group** (e.g. `clustering`, `backtest`, `forecast`,
  `inventory`).
- Up to 4 jobs across all groups (the ThreadPoolExecutor size).
- Recurring jobs are coalesced — if two firings happen during a long-running
  job, only one will run when the slot becomes free.

### 8.7.3 pg-queue scaffold (cutover surface alongside APScheduler)

`common/services/pg_queue.py` is a Postgres-backed job queue (DDL: `sql/183_create_pg_queue.sql`)
introduced as the cutover surface for long-running, restart-survivable, multi-instance-safe
recurring jobs. It does **not** replace APScheduler — both run side by side.

- **Pilot job:** `refresh_intramonth` has been migrated off APScheduler onto pg-queue.
  All other recurring jobs continue to run via the APScheduler thread pool.
- **Worker:** `scripts/ops/pg_queue_worker.py` is a long-running worker process. Run via
  `make pg-queue-worker`.
- **Scheduling:** a thin enqueueing entry-point (`make pg-queue-enqueue-recurring`)
  drops a single row into `job_queue` per cycle — typically driven by cron or a
  lightweight APScheduler job whose only side-effect is the enqueue. The actual
  work runs whenever the worker claims it via `FOR UPDATE SKIP LOCKED`.
- **Diagnostics:** `make pg-queue-depth` shows queue depth grouped by status.
- **Migration recipe:** see `docs/RUNBOOK.md` for the full APScheduler -> pg-queue
  cutover steps. The `refresh_intramonth` migration is the reference recipe.

| Make target | Purpose |
|---|---|
| `make pg-queue-worker` | Run a worker (long-running; one per host is enough for the pilot) |
| `make pg-queue-enqueue-recurring` | Drop a single `refresh_intramonth` row into `job_queue` (cron entry-point) |
| `make pg-queue-depth` | Diagnostic — current depth grouped by status |

### 8.7.4 UI surface

`JobsTab.tsx` exposes six panels: Active Jobs, Job History, Job Groups,
Schedules, Pipeline Builder, Champion Config. Important behaviour:

> **Clustering, features, backtest, champion, and forecast are
> hidden from the Jobs tab.** `JobsTab.tsx` passes
> `hiddenGroups={["clustering", "features", "backtest", "champion", "forecast"]}`
> and any clustering-specific config panels are removed —
> these pipelines are managed exclusively via the **Cluster** and
> **Champion Experiments** tabs. Do not re-expose them in JobsTab.

Recurring jobs (e.g. nightly DQ scan) are managed through `SchedulesPanel.tsx`
and persist as cron expressions in `fact_job_schedule`.

---

## 8.8 Notifications & Webhooks

### 8.8.1 Notifications

`common/services/notification_engine.py` is a multi-channel dispatcher with
adapters for Slack, Microsoft Teams, Email (SMTP), and PagerDuty. Channel
config lives in `config/platform/notification_config.yaml`.

Notifications are rate-limited in-process by `event_key` — duplicate events
within the cooldown window (default 300s) are suppressed. This is per-process
and resets on API restart.

| Endpoint | Purpose | Auth |
|---|---|---|
| `POST /notifications/test` | Send a test notification on a given channel | yes |

### 8.8.2 Webhooks

`common/services/webhook_dispatcher.py` provides an HMAC-SHA256 signed
delivery layer. Each registered webhook receives every event whose
`event_type` is in its `event_types` array. Signed payload structure:

```
POST /<your-url>
X-DS-Signature: sha256=<hex>
X-DS-Event:     <event_type>
Content-Type:   application/json

{ "event_type": "...", "data": {...}, "timestamp": 1714291200.123 }
```

Retries: 3 attempts with exponential backoff (`backoff_base=2.0` ⇒ 2s, 4s, 8s).

| Endpoint | Purpose | Auth |
|---|---|---|
| `POST /webhooks/register` | Register a URL + secret + event_types | yes |
| `GET /webhooks` | List active registrations | yes |
| `DELETE /webhooks/{webhook_id}` | Deactivate a registration | yes |
| `POST /webhooks/{webhook_id}/test` | Fire a test event | yes |

### 8.8.3 Events that fire

The following events are dispatched by the platform (by `WebhookEngine.dispatch_event`):

| event_type | Source |
|---|---|
| `sop.cycle.advanced` | `POST /sop/cycles/{id}/advance` |
| `sop.plan.approved` | `POST /sop/cycles/{id}/approve` |
| `event.calendar.approved` | `PUT /events/calendar/{id}/approve` |
| `storyboard.exception.created` | `scripts/generate_storyboard_exceptions.py` |
| `storyboard.exception.resolved` | `PUT /storyboard/exceptions/{id}/status` |
| `dq.check.failed` | `DQEngine.run_all_checks` (per failed check) |
| `job.completed`, `job.failed` | APScheduler hooks in `JobManager` |

Subscribers should be idempotent on `event_type + data` since the engine may
retry on transient delivery failures.

---

## 8.9 Operational Scripts

`scripts/ops/` per the file-placement policy in `CLAUDE.md` is the canonical
home for operations scripts. The current state of the repository is mid-migration:
most ops scripts still live at `scripts/` root and only `run_perf_analysis.py`
has been moved. New ops scripts must go in `scripts/ops/`.

| Script | Purpose | Make target |
|---|---|---|
| `scripts/run_sop_cycle.py` | Create / advance / populate-demand for an S&OP cycle | `sop-create`, `sop-advance`, `sop-populate`, `sop-seed` |
| `scripts/refresh_health_scores.py` | Recompute `mv_inventory_health_score` | `health-refresh` |
| `scripts/refresh_intramonth_stockout.py` | Recompute `mv_intramonth_stockout` | `intramonth-refresh` |
| `scripts/apply_event_adjustments.py` | Apply approved event uplifts to forecast | `events-apply` (`--dry-run` for preview) |
| `scripts/generate_storyboard_exceptions.py` | Run the exception engine for the current month | `storyboard-generate` (`-dry` variant available) |
| `scripts/populate_dq_checks.py` | Seed `fact_dq_check_definitions` from YAML | `dq-populate` |
| `scripts/fix_dq_issues.py` | Interactive remediation of known DQ failures | (manual: `uv run python scripts/fix_dq_issues.py`) |
| `scripts/ops/run_perf_analysis.py` | Read-only performance profiling | `perf-report` family |

---

## 8.10 Operating Cadence

The recommended cadence balances freshness against compute cost. Adjust to
local SLAs by editing `fact_job_schedule` rows from the `SchedulesPanel`.

| Activity | Cadence | How |
|---|---|---|
| S&OP cycle creation | Monthly (1st business day) | `make sop-create CYCLE_MONTH=YYYY-MM-01` or scheduled job |
| S&OP demand snapshot | Weekly during the cycle | `make sop-populate CYCLE_ID=<uuid>` |
| Event adjustment apply | After every event approval + nightly | `make events-apply` (cron the second invocation) |
| Storyboard exception scan | Daily (e.g. 04:00 UTC) | Schedule `generate_storyboard` job in `SchedulesPanel` |
| Health score refresh | Daily | `make health-refresh` |
| Intramonth stockout refresh | Hourly during business hours | `make intramonth-refresh` |
| Control Tower KPI MV | Daily (after health + fill_rate refresh) | `make control-tower-refresh` (or `refresh-mvs-tiered`) |
| Fill rate MV | Daily | `make fill-rate-refresh` |
| DQ scan | Daily after every ETL refresh | `make dq-run` |
| DQ schema reseed | Only when YAML changes | `make dq-populate` |
| Webhook delivery retry sweep | Continuous (in-process) | n/a |

The full daily refresh chain in execution order:

```bash
make pipeline-refresh           # Detect deltas, reload changed domains
make refresh-mvs-tiered         # Refresh aggregates, then derived MVs
make storyboard-generate        # Detect new exceptions
make dq-run                     # Score data quality
```

---

## 8.11 Verification & Troubleshooting

### 8.11.1 Smoke checks

```bash
make health        # alias for check-all — DB row counts + API /health
make check-all     # explicit form
make audit-routers # confirm every mounted router has a Vite proxy entry
```

`make health` hits `GET /health` on `localhost:8000`; if the API is not
running it prints a clear failure message. `make check-all` additionally
verifies row counts in the canonical fact tables.

### 8.11.2 Common issues

**Scheduler not starting on API boot.**
Check the API startup log for the line `APScheduler BackgroundScheduler
started on startup`. If you see `Scheduler not started on startup: ...`
instead, the lifespan handler in `api/main.py` caught an exception while
calling `JobManager.instance()`. Most often this is missing
`POSTGRES_PASSWORD` or a `fact_job_run` schema migration that has not been
applied (`make db-apply-jobs`). The API will still serve read endpoints in
this state but `/jobs` writes will fail.

**Control Tower returns the empty payload + warning.**
The MV exists but is unpopulated. Run `make refresh-mvs-tiered` (which also
refreshes the upstream MVs in dependency order). If the warning persists,
confirm the upstream MVs (`mv_inventory_health_score`, `mv_fill_rate_monthly`,
`mv_intramonth_stockout`) all exist via `\d+` in `psql`.

**Storyboard exceptions never appear.**
Verify three things in order: (1) `make exceptions-generate` has run for the
current month, (2) `mv_control_tower_kpis.open_exceptions_total` is non-zero,
(3) the storyboard job is enabled in `SchedulesPanel`. Then re-run
`make storyboard-generate` and check `fact_storyboard_exceptions`.

**S&OP `advance` returns 400.**
The cycle is already at the final stage (`closed`). Check
`SELECT current_stage FROM fact_sop_cycles WHERE cycle_id = '...'`. Closed
cycles must be re-created for the next month — there is no rewind.

**Event adjustments not flowing to forecast.**
Status must be `approved`, not `draft`. Verify with
`GET /events/calendar?status=approved`, then re-run `make events-apply`.

**DQ check fails immediately after a fresh load.**
A common false positive is a `range` check on a column whose source partition
has not loaded yet. See §8.6.4 above. If the failure is genuine, run
`scripts/fix_dq_issues.py --check <name>` for an interactive fix.

**Webhook delivery returning `failed` after 3 attempts.**
Subscriber URL is unreachable or returning non-2xx. Check
`fact_webhook_delivery_log` for the last error message. The dispatcher does
not retry beyond the initial 3 attempts — re-fire manually via
`POST /webhooks/{webhook_id}/test` after the subscriber is fixed.

**Notifications duplicated across processes.**
The 300s rate-limiter is in-process. If you run multiple API workers
(e.g. gunicorn `-w 4`), each worker maintains its own deduplication map.
Move duplicate suppression upstream (Slack channel filters) if this matters.

---

## 8.12 Quick Reference

```bash
# One-shot bootstrap of every operational module
make setup-ops

# Daily refresh
make pipeline-refresh && make refresh-mvs-tiered && \
  make storyboard-generate && make dq-run

# Verify
make health
curl -s http://localhost:8000/control-tower/kpis | jq '.computed_at, .health.total_skus'
curl -s http://localhost:8000/storyboard/exceptions/summary | jq '.totals'
```
