# Job Scheduler

> An APScheduler-powered background job engine with per-group concurrency control, FIFO queueing, cron/interval scheduling, job pipelines, and a frontend automation dashboard with live progress monitoring.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | JobsTab |
| **Key Files** | `JobsTab.tsx`, `api/routers/core/jobs.py`, `common/services/job_registry.py`, `common/services/job_scheduler.py`, `sql/020_create_job_history.sql` |

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
