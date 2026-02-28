# Feature 39: Job Scheduler/Monitor with APScheduler

## Objective

Provide a production-grade job execution, scheduling, and monitoring system powered by **APScheduler 3.11** that enables:
- Submitting, monitoring, and reviewing long-running operations from the UI
- Cron/interval scheduling for recurring automation
- Job pipelines (sequential chaining of multi-step workflows)
- Per-group concurrency control
- Retry logic with exponential backoff
- Foundation for agentic AI automation

## Architecture

### Backend — APScheduler Engine

**Technology:** APScheduler 3.11 (`BackgroundScheduler` with `ThreadPoolExecutor(max_workers=4)`)

**Why APScheduler:**
- Production-grade scheduler used in thousands of Python projects
- Thread-safe `BackgroundScheduler` runs in the same process as FastAPI
- Supports cron triggers, interval triggers, and one-shot date triggers
- `ThreadPoolExecutor` provides managed thread pool for concurrent job execution
- Extensible — future upgrade path to `AsyncIOScheduler` or distributed backends (Redis, MongoDB)

#### Database Schema

**`job_history` table** (DDL: `sql/020_create_job_history.sql` + `sql/021_alter_job_history_scheduling.sql`):

| Column | Type | Default | Purpose |
|--------|------|---------|---------|
| `job_id` | TEXT PK | — | UUID-style ID (e.g., `job_20260227_120000_abc12345`) |
| `job_type` | TEXT NOT NULL | — | Registry type (e.g., `cluster_scenario`) |
| `job_label` | TEXT NOT NULL | — | Human-readable description |
| `status` | TEXT NOT NULL | — | `queued` → `running` → `completed`/`failed`/`cancelled` |
| `params` | JSONB | — | Job-specific parameters |
| `result` | JSONB | — | Output summary/metrics |
| `error` | TEXT | — | Error message if failed |
| `submitted_at` | TIMESTAMPTZ | — | When submitted |
| `started_at` | TIMESTAMPTZ | — | When execution began |
| `completed_at` | TIMESTAMPTZ | — | When finished |
| `progress_pct` | SMALLINT | — | 0–100 progress |
| `progress_msg` | TEXT | — | Current step description |
| `scheduled_cron` | TEXT | NULL | Cron expression (if recurring) |
| `retry_count` | SMALLINT | 0 | Current retry attempt |
| `max_retries` | SMALLINT | 0 | Maximum retries allowed |
| `pipeline_id` | TEXT | NULL | Pipeline identifier (if part of pipeline) |
| `pipeline_step` | SMALLINT | NULL | Step number within pipeline |
| `triggered_by` | TEXT | 'manual' | Origin: `manual`, `api`, `schedule`, `pipeline`, `scenario` |

Indexes: `status`, `job_type`, `submitted_at DESC`, `pipeline_id`.

**`job_schedule` table** (DDL: `sql/021_alter_job_history_scheduling.sql`):

| Column | Type | Default | Purpose |
|--------|------|---------|---------|
| `schedule_id` | TEXT PK | — | Schedule identifier |
| `job_type` | TEXT NOT NULL | — | Registry type to submit |
| `job_label` | TEXT NOT NULL | — | Description |
| `cron_expr` | TEXT | NULL | Cron expression (e.g., `0 2 * * *`) |
| `interval_min` | INTEGER | NULL | Interval in minutes (alternative to cron) |
| `params` | JSONB | '{}' | Parameters for each run |
| `enabled` | BOOLEAN | TRUE | Whether schedule is active |
| `created_at` | TIMESTAMPTZ | NOW() | When created |
| `last_run_at` | TIMESTAMPTZ | NULL | When last execution occurred |
| `next_run_at` | TIMESTAMPTZ | NULL | Next scheduled execution time |
| `run_count` | INTEGER | 0 | Total number of executions |

Indexes: `enabled`.

#### Job Engine: `common/job_registry.py`

**`JOB_TYPE_REGISTRY`** — dictionary mapping 7 job types across 4 concurrency groups:

| Type ID | Label | Group | Callable |
|---------|-------|-------|----------|
| `cluster_scenario` | Clustering What-If | clustering | `scripts/run_clustering_scenario.py` (inline import) |
| `cluster_pipeline` | Full Clustering Pipeline | clustering | `uv run` subprocess chain (features → train → label → update) |
| `seasonality_pipeline` | Seasonality Detection | seasonality | `uv run` subprocess chain (detect → update) |
| `backtest_lgbm` | LGBM Backtest | backtest | `uv run python scripts/run_backtest.py` |
| `backtest_catboost` | CatBoost Backtest | backtest | `uv run python scripts/run_backtest_catboost.py` |
| `backtest_xgboost` | XGBoost Backtest | backtest | `uv run python scripts/run_backtest_xgboost.py` |
| `champion_select` | Champion Selection | champion | `uv run python scripts/run_champion_selection.py` |

**`JobManager`** class (singleton):
- `submit_job(job_type, params, label, triggered_by, max_retries)` — creates DB record, dispatches via `scheduler.add_job()`
- `start_job_in_background(job_id)` — backward-compatible no-op (APScheduler handles execution)
- `get_status(job_id)` — reads from `job_history` table
- `list_jobs(status, type, limit, offset)` — paginated query
- `get_active_jobs()` — all running/queued jobs
- `cancel_job(job_id)` — sets status to cancelled, removes from APScheduler
- `delete_job(job_id)` — removes completed/failed/cancelled jobs
- `get_types()` — list registered types with metadata
- `get_stats()` — aggregate statistics via SQL `COUNT(*) FILTER` (total, active, completed, failed, avg duration, last 24h)
- `schedule_recurring(job_type, params, label, cron, interval_minutes)` — creates persistent schedule via APScheduler `CronTrigger` or `IntervalTrigger`, persists to `job_schedule` table
- `list_schedules()` — reads from `job_schedule` table
- `remove_schedule(schedule_id)` — removes from APScheduler and DB
- `submit_pipeline(steps, label, triggered_by)` — submits chained job sequence, stores remaining steps in params
- `recover_stale_jobs()` — on startup, marks leftover `running`/`queued` jobs as `failed`
- `_execute_job()` — job execution wrapper with retry logic and exponential backoff
- `_scheduled_execution()` — callback for recurring schedule triggers, checks group busy before submitting
- `_trigger_next_pipeline_step()` — chains pipeline steps on completion

**Concurrency:** Per-group `threading.Lock()`. One active job per group; 409 Conflict if group is busy. Groups: `clustering`, `backtest`, `seasonality`, `champion`.

**Retry Logic:** Configurable `max_retries` per job with exponential backoff (`min(2^attempt, 60)` seconds delay). Retry count tracked in DB.

**Execution:** Job callables wrap existing scripts via `subprocess.run()`. Progress callback updates DB during execution. APScheduler `add_job()` dispatches to managed thread pool. DB uses direct `psycopg.connect()` with `autocommit=True` (not the API pool).

#### Router: `api/routers/jobs.py`

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/jobs/types` | GET | No | List available job types |
| `/jobs/stats` | GET | No | Aggregate statistics for dashboard KPIs |
| `/jobs` | POST | Yes | Submit new job → HTTP 202 |
| `/jobs` | GET | No | List jobs (filters: status, type; pagination) |
| `/jobs/active` | GET | No | Currently running/queued jobs |
| `/jobs/schedule` | POST | Yes | Create recurring schedule (cron/interval) |
| `/jobs/schedules` | GET | No | List active recurring schedules |
| `/jobs/schedules/{id}` | DELETE | Yes | Remove recurring schedule |
| `/jobs/pipeline` | POST | Yes | Submit chained job pipeline → HTTP 202 |
| `/jobs/{job_id}` | GET | No | Single job status + result |
| `/jobs/{job_id}/cancel` | POST | Yes | Cancel running job |
| `/jobs/{job_id}` | DELETE | Yes | Remove from history |

**Route ordering:** Specific literal paths (`/jobs/schedules`, `/jobs/pipeline`, `/jobs/schedule`) MUST be registered before parameterized `{job_id}` paths to avoid FastAPI matching "schedules" as a job_id.

Mutation endpoints use `Depends(require_api_key)` from `api/auth.py`.

### Frontend

#### Vite Proxy Configuration

**CRITICAL:** The Vite dev server must proxy `/jobs` requests to the backend. Without this, all job API calls hit Vite directly and return HTML instead of JSON, causing the Jobs tab to show no data.

```typescript
// frontend/vite.config.ts → server.proxy
"/jobs": {
  target: "http://127.0.0.1:8000",
  changeOrigin: true,
},
```

#### Types: `types/jobs.ts`

- `JobStatus`: `queued | running | completed | failed | cancelled`
- `JobType`: type metadata (type_id, label, description, group, params_schema)
- `Job`: full job record with scheduling fields
- `JobStats`: aggregate statistics (total, active, completed, failed, avg_duration_seconds, last_24h)
- `JobSchedule`: recurring schedule record (schedule_id, job_type, cron_expr, interval_min, enabled, run_count, next_run_at)
- `JobGroup`: `clustering | backtest | seasonality | champion`
- `GROUP_CONFIG`: visual config per group (color, bgColor, borderColor, iconBg)

#### Query Layer: `api/queries.ts`

Query keys: `jobTypes()`, `jobs(params)`, `jobDetail(id)`, `activeJobs()`, `jobStats()`, `jobSchedules()`

Fetch functions: `fetchJobTypes`, `fetchJobs`, `fetchJobDetail`, `fetchActiveJobs`, `fetchJobStats`, `fetchJobSchedules`, `submitJob`, `cancelJob`, `deleteJob`, `createSchedule`, `deleteSchedule`, `submitPipeline`

Polling intervals:
- Active jobs: 2s refetch
- Job stats: 5s refetch
- Job history: 10s refetch
- Job schedules: 1min stale time

#### JobsTab: `tabs/JobsTab.tsx`

Professional automation dashboard with 5 sections:

1. **KPI Cards** — Total Jobs, Active Now, Success Rate, Avg Duration. Polled every 5s via `fetchJobStats`.

2. **Submit New Job** — Cards grouped by category (Clustering, Backtesting, Seasonality, Champion). Each card has "Run Now" and schedule (calendar icon) buttons. Group-specific colors: blue (clustering), violet (backtest), emerald (seasonality), amber (champion). Group icons: Network (clustering), TrendingUp (backtest), Activity (seasonality), Trophy (champion).

3. **Active Jobs** — Real-time cards for `queued`/`running` jobs. Animated gradient progress bar, live elapsed timer via `setInterval(1000)`, cancel button. Polled every 2s.

4. **Schedules** — Active recurring schedules with cron expression badges. Delete button per schedule.

5. **Job History** — Paginated table with color-coded status badges (queued=yellow, running=blue, completed=emerald, failed=red, cancelled=gray), expandable rows showing job ID, progress, params, error/result. Filters by type and status. Polled every 10s.

**Schedule Dialog:** Modal with preset buttons (hourly, 6h, daily 2AM, weekly Mon 2AM), custom cron/interval input, interval minutes field.

**Header:** "APScheduler Engine" badge, job type count, active count.

#### JobNotificationContext: `context/JobNotificationContext.tsx`

Cross-tab notification context tracking:
- `activeJobs: Map<string, ActiveJob>` — currently running jobs
- `recentCompletions: CompletedJob[]` — recently finished jobs for dashboard alerts
- `activeJobCount: number` — badge count
- Methods: `startJob()`, `completeJob()`, `failJob()`, `dismissCompletion()`

#### Navigation Integration

- **AppSidebar.tsx**: Jobs nav item with active job count badge (blue circle with count)
- **useUrlState.ts**: `"jobs"` in `VALID_TABS`
- **useKeyboardShortcuts.ts**: `"9": "jobs"` in `TAB_MAP`
- **App.tsx**: Lazy import `JobsTab`, `JobNotificationProvider` wrapping app
- **ClustersTab.tsx**: "Schedule Scenario Job" button, job ID badge, "View in Jobs" link

#### Dashboard Alert Integration

- `AlertType` union includes `"job_complete"`
- `AlertPanel.tsx` maps `job_complete` to `PlayCircle` icon
- `DashboardTab.tsx` injects job completion/failure alerts from `JobNotificationContext`

## Files

### Created

| File | Purpose |
|------|---------|
| `sql/020_create_job_history.sql` | Base job_history DDL |
| `sql/021_alter_job_history_scheduling.sql` | Scheduling columns + job_schedule table |
| `common/job_registry.py` | APScheduler-powered job engine |
| `api/routers/jobs.py` | 12 REST API endpoints |
| `frontend/src/types/jobs.ts` | TypeScript types + GROUP_CONFIG |
| `frontend/src/tabs/JobsTab.tsx` | Automation dashboard UI |
| `frontend/src/context/JobNotificationContext.tsx` | Cross-tab notification context |
| `frontend/src/context/__tests__/JobNotificationContext.test.tsx` | Context tests |
| `tests/api/test_jobs.py` | 16 backend API tests |
| `frontend/src/tabs/__tests__/JobsTab.test.tsx` | 7 frontend tests |

### Modified

| File | Change |
|------|--------|
| `api/main.py` | Mount jobs router |
| `Makefile` | Add `db-apply-jobs` target (applies both 020 + 021 SQL files) |
| `frontend/vite.config.ts` | Add `/jobs` proxy rule to forward API calls to backend |
| `frontend/src/api/queries.ts` | Add job query keys + fetch functions |
| `frontend/src/App.tsx` | Add JobsTab + JobNotificationProvider |
| `frontend/src/components/AppSidebar.tsx` | Add Jobs nav item + active badge |
| `frontend/src/hooks/useUrlState.ts` | Add "jobs" to VALID_TABS |
| `frontend/src/hooks/useKeyboardShortcuts.ts` | Add "9": "jobs" |
| `frontend/src/types/theme.ts` | Add "job_complete" to AlertType |
| `frontend/src/components/AlertPanel.tsx` | Add job_complete icon |
| `frontend/src/tabs/DashboardTab.tsx` | Add job completion alerts |
| `frontend/src/tabs/ClustersTab.tsx` | Job scheduling integration ("Schedule Scenario Job") |

## Testing

### Backend (16 tests in `tests/api/test_jobs.py`)

| Test | Endpoint | Assertion |
|------|----------|-----------|
| `test_list_job_types` | GET /jobs/types | 200, 2 types |
| `test_submit_job_returns_202` | POST /jobs | 202, job_id |
| `test_submit_job_invalid_type` | POST /jobs (bad type) | 422 |
| `test_submit_job_conflict` | POST /jobs (group busy) | 409 |
| `test_list_jobs` | GET /jobs | 200, paginated |
| `test_get_job_detail_not_found` | GET /jobs/{id} | 404 |
| `test_get_job_detail_found` | GET /jobs/{id} | 200, full record |
| `test_cancel_job` | POST /jobs/{id}/cancel | 200 |
| `test_cancel_job_not_found` | POST /jobs/{id}/cancel | 404 |
| `test_delete_job` | DELETE /jobs/{id} | 200 |
| `test_active_jobs` | GET /jobs/active | 200 |
| `test_job_stats` | GET /jobs/stats | 200, aggregates |
| `test_schedule_recurring_job` | POST /jobs/schedule | 201, schedule_id |
| `test_schedule_missing_trigger` | POST /jobs/schedule | 422 |
| `test_list_schedules` | GET /jobs/schedules | 200, schedule list |
| `test_submit_pipeline` | POST /jobs/pipeline | 202, pipeline_id |

### Frontend (7 tests in `tabs/__tests__/JobsTab.test.tsx`)

- Renders without crashing
- Renders available job type cards
- Shows job group headers (Clustering, Backtesting)
- Shows empty history message
- Renders header description
- Shows KPI cards with stats
- Shows APScheduler engine badge

### Frontend (11 tests in `components/__tests__/AppSidebar.test.tsx`)

- All wrapped with `JobNotificationProvider` for badge support

## Dependencies

- `apscheduler>=3.10` (installed: 3.11.2)
- `tzlocal>=5.0` (installed: 5.3.1, APScheduler dependency)

## Make Targets

```bash
make db-apply-jobs   # Apply job_history + job_schedule DDL to Postgres (both 020 + 021 SQL files)
```

## Troubleshooting

### Jobs tab shows no data
- **Cause:** Vite dev server proxy missing `/jobs` rule — API requests hit Vite instead of FastAPI backend
- **Fix:** Add `/jobs` proxy entry to `frontend/vite.config.ts`, restart Vite dev server (`make ui`)
- **Verify:** `curl http://localhost:8000/jobs/active` should return JSON (not HTML)

### Jobs fail to insert into database
- **Cause:** `sql/021_alter_job_history_scheduling.sql` not applied — missing `triggered_by`, `pipeline_id`, etc. columns
- **Fix:** Run `make db-apply-jobs` (applies both 020 + 021 SQL files)
- **Verify:** `SELECT column_name FROM information_schema.columns WHERE table_name='job_history'` should show 18 columns

### Route conflict (GET /jobs/schedules returns 404)
- **Cause:** FastAPI matches `{job_id}="schedules"` before the literal `/jobs/schedules` route
- **Fix:** Ensure literal routes (`/jobs/schedules`, `/jobs/pipeline`) are registered before `{job_id}` routes in `api/routers/jobs.py`

## Agentic AI Foundation

This job system provides the groundwork for agentic AI automation:

1. **Scheduled Automation:** Cron/interval triggers enable hands-off recurring workflows (nightly backtests, weekly clustering refresh)
2. **Pipeline Orchestration:** Sequential job chaining enables multi-step workflows (cluster → backtest → champion select)
3. **API-First Design:** All operations available via REST API — AI agents can submit, monitor, and react to job results
4. **Extensible Type Registry:** New job types added by registering a callable + metadata in `JOB_TYPE_REGISTRY`
5. **Retry Resilience:** Automatic retry with exponential backoff ensures transient failures don't break automation
6. **Cross-Tab Notifications:** Real-time status propagation via React context for responsive UI

## Future Extensions

- Additional job types for simulation scenarios
- Job dependency DAG execution (parallel steps)
- WebSocket-based real-time progress updates (replacing polling)
- Job result artifact storage (charts, reports, model artifacts)
- APScheduler distributed backend (Redis/MongoDB for multi-worker deployments)

---

## Implementation Corrections

### JobStats Response
Additional fields not in spec:
- `cancelled` (count of cancelled jobs)
- `last_24h` is a nested object: `{submitted, completed, failed}` (not a single number)

### JobTypeDef
- `default_max_retries: int = 0` field not documented

### Cancellation Mechanism
- `_cancel_flags: dict[str, threading.Event]` for signal propagation

### APScheduler Configuration
- `coalesce: True`, `max_instances: 1`, `misfire_grace_time: 3600`, `timezone: "UTC"`

### Lazy Initialization
- `_ensure_init()` method — scheduler doesn't start until first method call
- Router uses `_get_manager()` with global `_manager` for lazy init

### Job Callable Details
- `_run_subprocess()` shared helper captures stdout/stderr, raises `RuntimeError` on non-zero exit
- `_run_cluster_pipeline`: 4 steps at 25%/50%/75%/95%, accepts `time_window_months`/`k_range`
- `_run_seasonality`: 2 steps at 40%/90%, accepts `config` path
- `_run_backtest`: accepts `cluster_strategy` (default `"global"`)

### Recovery
- `recover_stale_jobs()` auto-called in `_ensure_init()` — marks leftover `running`/`queued` as `failed` with "Interrupted by server restart"

### Pipeline Internals
- `__pipeline_remaining`, `__pipeline_label`, `__pipeline_step` keys in params
- Step label format: `[Pipeline N/M] step_label`

### Response Formats (not fully documented in spec)
- POST `/jobs`: `{"job_id": ..., "status": "queued"}` (not "running")
- POST `/jobs/schedule`: HTTP 201 with `{"schedule_id": ..., "status": "active"}`
- DELETE endpoints: `{"deleted": true}`
- Cancel: `{"job_id": ..., "status": "cancelled"}`
- Pipeline: includes `"steps"` count in response

### TypeScript Types (not in spec)
- `GroupConfig`: label, color, bgColor, borderColor, iconBg
- `JobListPayload`: jobs, total, limit, offset
- `ActiveJobsPayload`: jobs
- `JobSchedulesPayload`: schedules

### Frontend Query Functions
- `fetchJobs({status?, job_type?, limit?, offset?})`
- `submitJob(jobType, params, label?)`
- `createSchedule(jobType, params, label?, cron?, intervalMinutes?)`
- `fetchScenarioHistory(limit?)` — returns last N completed cluster_scenario jobs
- `fetchJobDetail(jobId)` — returns full job record with result data

---

## Enhancements (Post-Initial Implementation)

### View Results Navigation (JobsTab → ClustersTab)
- "View Results" button (BarChart3 icon) in JobsTab history rows for completed `cluster_scenario` jobs
- Clicking navigates to ClustersTab via `?scenario_job=<job_id>` URL parameter
- ClustersTab auto-loads the scenario result from DB on mount, expands What-If section, renders ScenarioCharts
- URL param is immediately cleared after consumption to prevent stale state on refresh
- `getScenarioJobParam()` / `setScenarioJobParam()` exported from `useUrlState.ts`
- `App.tsx` wires `handleNavigateToScenario` callback through `JobsTab → setScenarioJobParam → handleTabSwitch("clusters")`

### Scenario Queueing (replaces 409 rejection)
- **Previous behavior:** `submit_job()` raised `RuntimeError` when group was busy → API returned HTTP 409 Conflict
- **New behavior:** `submit_job()` creates the job in DB with `status="queued"` and appends to `_pending_queues[group]` FIFO queue
- When the running job completes, `_dispatch_next(group)` pops the first queued job and dispatches it to APScheduler
- `_pending_queues: dict[str, list[tuple]]` — per-group in-memory queue
- `_dispatch_next(group)` — updates `_active_jobs`, `_cancel_flags`, schedules via APScheduler
- API responses now include `"status": "queued"` or `"status": "running"` depending on whether the job started immediately
- Frontend ClustersTab shows "Queued..." button text and a distinct info banner when scenario is queued
- Queued → running transition detected via polling; banner updates automatically
- All 409 error handlers removed from `jobs.py` (submit, pipeline) and `clusters.py` (scenario)

### Past Scenarios History (ClustersTab)
- "Past Scenarios" section in ClustersTab What-If panel showing last 10 completed cluster_scenario runs
- Accordion pattern: clicking an entry expands it to show profile table + ScenarioCharts inline
- Each entry displays: job label, optimal K, DFU count, runtime, timestamp
- "Promote" button on each expanded entry triggers the existing promote flow
- Data fetched via `fetchScenarioHistory(10)` with 30s stale time, refetches when What-If is expanded
- Only one entry expanded at a time (expanding one collapses the previous)
