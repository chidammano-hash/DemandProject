# Job Scheduler

> An APScheduler-powered background job engine with per-group concurrency control, FIFO queueing, cron/interval scheduling, job pipelines, and a frontend automation dashboard with live progress monitoring.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | JobsTab |
| **Key Files** | `JobsTab.tsx`, `api/routers/core/jobs.py`, `common/services/job_registry.py`, `common/services/job_scheduler.py`, `sql/020_create_job_history.sql`, `common/services/pg_queue.py`, `scripts/ops/pg_queue_worker.py`, `sql/183_create_pg_queue.sql` |

---

## Problem

The platform has dozens of compute-intensive pipelines: clustering, backtesting (three models), seasonality detection, champion selection, production forecast generation, and more. Running these manually via Make targets is fine for development but not for production operations. Planners need to trigger jobs from the UI, monitor progress, schedule recurring runs, and chain dependent pipelines -- without SSH access or terminal knowledge.

---

## Solution

A `JobManager` singleton wraps APScheduler 3.11's `BackgroundScheduler` with a `ThreadPoolExecutor(max_workers=4)`. Seven job types across four groups enforce per-group concurrency (one active job per group). When a group is busy, new jobs are queued in FIFO order and auto-dispatched when the active job completes. A REST API exposes 12 endpoints for job CRUD, scheduling, pipelines, and statistics. The frontend renders a real-time automation dashboard with KPI cards, grouped job cards, schedules, and expandable history.

---

## How It Works

### Job Types and Groups

| Group | Job Types | Color |
|---|---|---|
| clustering | `cluster_pipeline`, `cluster_scenario` | Blue |
| backtest | `backtest_lgbm`, `backtest_catboost`, `backtest_xgboost` | Violet |
| seasonality | `seasonality_pipeline` | Emerald |
| champion | `champion_selection` | Amber |
| forecast | `generate_production_forecast` | Teal |
| tuning | `tuning_backtest` | Violet |

### Concurrency Model

Each group allows exactly one active job. Additional submissions to a busy group enter a FIFO pending queue:

1. Job submitted to group `backtest`.
2. If no active job in `backtest`, run immediately.
3. If `backtest_lgbm` is already running, queue the new job with `status=queued`.
4. When the active job completes, `_dispatch_next()` dequeues the next job in the group.

### Job Lifecycle

| Status | Meaning |
|---|---|
| `queued` | Waiting for group slot |
| `running` | Executing in background thread |
| `completed` | Finished successfully |
| `failed` | Terminated with error (error message stored) |
| `cancelled` | User-cancelled via API |

### Scheduling

Jobs can be scheduled with cron expressions or fixed intervals:

| Schedule Type | Example | Use Case |
|---|---|---|
| Cron | `0 2 * * *` (daily at 2 AM) | Nightly clustering refresh |
| Interval | Every 6 hours | Periodic demand signal computation |

Schedules are stored in the `job_schedule` table and managed by APScheduler's persistent job store.

### Pipelines

A pipeline chains multiple jobs sequentially. Each step runs only if the previous step succeeded:

Example: `cluster_pipeline` -> `backtest_lgbm` -> `champion_selection`

If any step fails, the pipeline stops and records the failure point.

### Resilient Execution (Survive Restarts)

All subprocess-based jobs run via `subprocess.Popen()` with `start_new_session=True`, placing each child in its own process group. This means stopping the API does **not** kill running jobs. The subprocess PID is stored in `job_history.pid` for kill and recovery.

**Kill mechanism:** The Kill button sends `SIGTERM` to the entire process group via `os.killpg(os.getpgid(pid), signal.SIGTERM)`. A `cancel_event` (`threading.Event`) is also threaded through all callables and checked between subprocess output lines, enabling cooperative cancellation for multi-step jobs.

**Persistent logs:** Subprocess stdout is streamed to the `job_history.log` column in real-time (flushed every 20 lines or 5 seconds). The frontend polls `GET /jobs/{id}/logs` for live log display during execution and on-demand viewing in job history.

### Recovery

On application restart, `recover_stale_jobs()` performs PID-aware recovery:
- If PID is alive (`os.kill(pid, 0)` succeeds) → re-adopt the process via a monitoring thread that polls PID status
- If PID is dead or missing → mark as `failed`
- Re-enqueue previously `queued` jobs for dispatch.

### Thread Safety

| Lock | Protects |
|---|---|
| `_state_lock` (threading.Lock) | `_active_jobs`, `_pending_queues`, `_cancel_flags` |
| `_init_lock` (threading.Lock) | Double-checked locking for `_ensure_init()` singleton |

---

## Data Model

| Table | Purpose | Key Columns |
|---|---|---|
| `job_history` | Persistent job execution log | `job_id`, `job_type`, `group_name`, `status`, `params` (JSONB), `result` (JSONB), `error`, `pid` (INTEGER), `log` (TEXT), `started_at`, `completed_at`, `duration_seconds` |
| `job_schedule` | Recurring schedule definitions | `schedule_id`, `job_type`, `cron_expression`, `interval_seconds`, `enabled`, `last_run_at`, `next_run_at` |

---

## Long-Running Jobs: pg-queue

APScheduler hosts every job inside the API process's own `ThreadPoolExecutor`, which is fine for a job that finishes in minutes but becomes a liability for one that runs for hours: the API process has to stay up for the job's entire duration, and only one process can host that executor, so there's no way to add worker capacity horizontally. `common/services/pg_queue.py` is a minimal Postgres-backed job queue (Item 22 pilot) that decouples *scheduling* from *execution* for exactly that class of job. It is not a replacement for APScheduler - it is the cutover surface for long-running, restart-survivable, multi-instance-safe jobs, sitting alongside the `JobManager` documented above.

### The Queue Table

`job_queue` (`sql/183_create_pg_queue.sql`) holds one row per job. `status` moves through `pending` -> `claimed` -> `running` -> `completed`/`failed`, and workers claim the next due row with `SELECT ... FOR UPDATE SKIP LOCKED` - Postgres's canonical primitive for letting multiple workers race for work without blocking each other.

| Column | Purpose |
|---|---|
| `status` | `pending` \| `claimed` \| `running` \| `completed` \| `failed` |
| `priority` | Lower runs first (default 100) |
| `run_at` | Job becomes eligible to be claimed at/after this time |
| `attempts` / `max_attempts` | Retry bookkeeping (default `max_attempts=3`) |
| `claimed_by` | `<hostname>:<pid>` of the claiming worker, overridable via `PG_QUEUE_WORKER_ID` |
| `result` / `last_error` | JSONB result on success, truncated error text on failure |

A failed job is re-enqueued with exponential backoff (`requeue_failed_with_backoff`: 60s, 120s, 240s... capped at 1 hour) until `attempts` reaches `max_attempts`, at which point it is a dead-letter.

### The Worker

`scripts/ops/pg_queue_worker.py` is a long-lived polling process that runs independently of the API:

```
uv run python scripts/ops/pg_queue_worker.py [--types TYPE ...] [--poll-interval SECONDS]
```

It polls `job_queue` (default every 5 seconds), claims the next due job, and dispatches it to a handler from its `HANDLERS` table. Handlers delegate to the same runner functions in `common/services/job_state.py` that APScheduler calls, so a job behaves identically regardless of which system runs it. `SIGTERM`/`SIGINT` trigger a graceful shutdown that lets the in-flight job finish before exiting. One worker process handles one job at a time; run multiple worker instances for parallelism, since `FOR UPDATE SKIP LOCKED` guarantees no two workers claim the same row.

Pilot scope: today only `refresh_intramonth` (the intramonth stockout materialized-view refresh) is routed through pg-queue - its *recurring daily schedule* was cut over from APScheduler's persistent job store; the job type itself remains registered in APScheduler too, so it can still be triggered manually from the Jobs tab. A cron entry-point drops one row into `job_queue` per day; the long-lived worker claims and runs it whenever it is free. That decoupling matters because the refresh previously tied up an APScheduler thread for 7-20 hours.

| Command | What It Does |
|---|---|
| `make pg-queue-worker` | Run a pg-queue worker (long-running; handles `refresh_intramonth`) |
| `make pg-queue-enqueue-recurring` | Enqueue the recurring `refresh_intramonth` job (cron entry-point) |
| `make pg-queue-depth` | Show queue depth grouped by status (diagnostic) |

### Decision Rule: pg-queue vs. APScheduler

APScheduler (the `JobManager` documented above) stays the default for new jobs - it is where the Jobs tab dashboard, live progress bars, log streaming, and per-group concurrency all live. Route a job to pg-queue instead when it needs any of the following, which APScheduler does not provide:

| Requirement | Why pg-queue |
|---|---|
| Runs for hours, not minutes | The API process no longer has to stay alive for the job's whole duration |
| Must survive an API restart/redeploy cleanly | Work lives in `job_queue`, not in-memory state PID-recovered by `recover_stale_jobs()` |
| Needs multiple worker processes for throughput | `FOR UPDATE SKIP LOCKED` is safe for concurrent claims; APScheduler's executor is single-process |

`_run_etl_pipeline`'s docstring in `common/services/job_state.py` states the rule directly: a `full` reload of large fact tables "can exceed the APScheduler comfort window," and for very large datasets should route to the pg-queue worker instead, while the incremental `refresh` mode stays on APScheduler because it is short. In short - if a new job is a quick, UI-triggered, monitorable task, it belongs on APScheduler; if it is a multi-hour batch job that must keep running whether or not the API process does, it belongs on pg-queue.

---

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/jobs/run` | Submit a job for immediate execution |
| GET | `/jobs/active` | List currently running and queued jobs |
| GET | `/jobs/{job_id}` | Job detail with params, result, error |
| POST | `/jobs/{job_id}/cancel` | Cancel a running or queued job |
| POST | `/jobs/schedule` | Create a recurring schedule |
| GET | `/jobs/schedules` | List all schedules with next run time |
| DELETE | `/jobs/schedules/{id}` | Remove a schedule |
| POST | `/jobs/pipeline` | Submit a sequential job pipeline |
| GET | `/jobs/stats` | Dashboard KPIs: total, active, success rate, avg duration |
| GET | `/jobs/{job_id}/logs` | Persistent execution log (supports `offset` for incremental polling) |
| GET | `/jobs/history` | Paginated job history with filtering |

Route ordering note: literal paths (`/jobs/schedules`, `/jobs/pipeline`, `/jobs/stats`) are defined before the parameterized `{job_id}` path to avoid conflicts.

---

## Frontend

### Dashboard Layout

| Zone | Component | Poll Interval |
|---|---|---|
| KPI cards | Total Jobs, Active Now, Success Rate, Avg Duration | 5 seconds |
| Job group cards | Grouped by type with category colors and "Run Now" buttons | 2 seconds |
| Active jobs | Animated progress bars, elapsed timers, persistent log panel, Kill button (2-step confirm) | 2 seconds |
| Schedules | Cron badge display with enable/disable toggle | 10 seconds |
| History | Expandable rows with params, results, errors, "View Execution Log" button | 10 seconds |

### Cross-Tab Notifications

`JobNotificationContext` tracks running and completed jobs across all tabs. The sidebar shows an active job count badge next to the Jobs nav item. Dashboard injects completion alert banners. ClustersTab integrates a "Schedule Scenario Job" button.

---

## Dependencies

| Dependency | Reason |
|---|---|
| `apscheduler>=3.10` | Background scheduler engine |
| `tzlocal>=5.0` | Timezone support for cron expressions |
| PostgreSQL `job_history` table | Persistent execution log |

---

## See Also

- `02-forecasting/10-sku-clustering.md` -- clustering scenarios dispatched as jobs
- `02-forecasting/08-production-forecast.md` -- forecast generation as a scheduled job
- `07-user-experience/02-ui-architecture.md` -- sidebar badge and notification context
