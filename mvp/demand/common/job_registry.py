"""Job scheduler engine powered by APScheduler (Feature 39).

Production-grade job scheduling and execution with:
- APScheduler BackgroundScheduler with managed ThreadPoolExecutor
- PostgreSQL persistence for job state, results, and history
- Per-group concurrency control (one active job per group)
- Cron/interval scheduling for recurring automation
- Job pipeline support (sequential chaining)
- Configurable retry logic with exponential backoff
- Real-time progress tracking
- Statistics and monitoring API

This module provides the groundwork for agentic AI automation:
any AI agent can submit, schedule, and monitor jobs via the REST API.

Public API (unchanged — all external imports still work):
    from common.job_registry import JobManager, JOB_TYPE_REGISTRY
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Sub-module imports
# ---------------------------------------------------------------------------
from common.job_state import (
    JobTypeDef,
    _get_conn,
    _row_to_dict,
    _run_backtest_catboost,
    _run_backtest_lgbm,
    _run_backtest_xgboost,
    _run_champion_select,
    _run_cluster_pipeline,
    _run_cluster_scenario,
    _run_generate_ai_insights,
    _run_seasonality,
)
from common.job_scheduler import make_scheduler, make_trigger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

JOB_TYPE_REGISTRY: dict[str, JobTypeDef] = {
    "cluster_scenario": JobTypeDef(
        type_id="cluster_scenario",
        label="Clustering What-If",
        description="Run a trial clustering pipeline with custom parameters",
        group="clustering",
        callable=_run_cluster_scenario,
        params_schema={"feature_params": {}, "model_params": {}, "label_params": {}},
    ),
    "cluster_pipeline": JobTypeDef(
        type_id="cluster_pipeline",
        label="Full Clustering Pipeline",
        description="Generate features, train model, label clusters, update DFU table",
        group="clustering",
        callable=_run_cluster_pipeline,
        params_schema={"time_window_months": 24, "k_range": [3, 12]},
    ),
    "seasonality_pipeline": JobTypeDef(
        type_id="seasonality_pipeline",
        label="Seasonality Detection",
        description="Detect seasonality patterns and update DFU profiles",
        group="seasonality",
        callable=_run_seasonality,
        params_schema={"config": "config/seasonality_config.yaml"},
    ),
    "backtest_lgbm": JobTypeDef(
        type_id="backtest_lgbm",
        label="LGBM Backtest",
        description="Run LightGBM backtest with expanding window timeframes",
        group="backtest",
        callable=_run_backtest_lgbm,
        params_schema={"cluster_strategy": "global"},
    ),
    "backtest_catboost": JobTypeDef(
        type_id="backtest_catboost",
        label="CatBoost Backtest",
        description="Run CatBoost backtest with expanding window timeframes",
        group="backtest",
        callable=_run_backtest_catboost,
        params_schema={"cluster_strategy": "global"},
    ),
    "backtest_xgboost": JobTypeDef(
        type_id="backtest_xgboost",
        label="XGBoost Backtest",
        description="Run XGBoost backtest with expanding window timeframes",
        group="backtest",
        callable=_run_backtest_xgboost,
        params_schema={"cluster_strategy": "global"},
    ),
    "champion_select": JobTypeDef(
        type_id="champion_select",
        label="Champion Selection",
        description="Select per-DFU champion model via rolling WAPE comparison",
        group="champion",
        callable=_run_champion_select,
        params_schema={},
    ),
    "generate_ai_insights": JobTypeDef(
        type_id="generate_ai_insights",
        label="AI Insight Generation",
        description="Scan portfolio for planning exceptions and generate AI insights",
        group="ai",
        callable=_run_generate_ai_insights,
        params_schema={},
    ),
}


# ---------------------------------------------------------------------------
# JobManager — singleton, powered by APScheduler
# ---------------------------------------------------------------------------


class JobManager:
    """Production-grade job manager powered by APScheduler.

    Features:
    - Managed ThreadPoolExecutor via APScheduler BackgroundScheduler
    - Immediate and scheduled (cron/interval) job execution
    - Per-group concurrency control
    - PostgreSQL persistence for all job state
    - Retry logic with exponential backoff
    - Job pipeline (chain) support
    - Real-time statistics
    """

    _instance: JobManager | None = None
    _init_lock = threading.Lock()

    def __new__(cls) -> JobManager:
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    cls._instance = inst
        return cls._instance

    def _ensure_init(self) -> None:
        """Lazily initialise APScheduler and internal state.

        Thread-safe: uses _init_lock to prevent double initialisation.
        """
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self._scheduler = make_scheduler()
            self._state_lock = threading.Lock()  # guards _active_jobs, _pending_queues, _cancel_flags
            self._active_jobs: dict[str, str] = {}  # job_id -> group
            self._cancel_flags: dict[str, threading.Event] = {}
            self._pending_queues: dict[str, list[tuple[str, Any, dict, int, str | None]]] = {}  # group -> [(job_id, type_def, params, max_retries, pipeline_id)]
            self._initialized = True
            logger.info("JobManager initialised: APScheduler BackgroundScheduler started (4 workers)")

            # Recover stale jobs on startup
            recovered = self.recover_stale_jobs()
            if recovered:
                logger.info("Recovered %d stale jobs on startup", recovered)

    # ---- helpers ----

    @staticmethod
    def _generate_id() -> str:
        return f"job_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def _is_group_busy(self, group: str) -> bool:
        """Check if a group has an active job. Must be called under _state_lock."""
        return any(g == group for g in self._active_jobs.values())

    # ---- DB operations ----

    @staticmethod
    def _db_insert(job_id: str, job_type: str, label: str, params: dict,
                   triggered_by: str = "manual", pipeline_id: str | None = None,
                   pipeline_step: int | None = None, max_retries: int = 0) -> None:
        with _get_conn() as conn:
            conn.execute(
                """INSERT INTO job_history
                   (job_id, job_type, job_label, status, params, submitted_at,
                    triggered_by, pipeline_id, pipeline_step, max_retries)
                   VALUES (%s, %s, %s, 'queued', %s, NOW(), %s, %s, %s, %s)""",
                (job_id, job_type, label, json.dumps(params),
                 triggered_by, pipeline_id, pipeline_step, max_retries),
            )

    @staticmethod
    def _db_update_status(job_id: str, status: str, **kwargs: Any) -> None:
        sets = ["status = %s"]
        vals: list[Any] = [status]
        for col in ("started_at", "completed_at", "progress_pct", "progress_msg",
                     "error", "retry_count"):
            if col in kwargs:
                sets.append(f"{col} = %s")
                vals.append(kwargs[col])
        if "result" in kwargs:
            sets.append("result = %s")
            vals.append(json.dumps(kwargs["result"]) if kwargs["result"] is not None else None)
        vals.append(job_id)
        with _get_conn() as conn:
            conn.execute(f"UPDATE job_history SET {', '.join(sets)} WHERE job_id = %s", vals)

    @staticmethod
    def _db_get(job_id: str) -> dict[str, Any] | None:
        cols = ("job_id", "job_type", "job_label", "status", "params", "result",
                "error", "submitted_at", "started_at", "completed_at",
                "progress_pct", "progress_msg")
        with _get_conn() as conn:
            row = conn.execute(
                f"SELECT {', '.join(cols)} FROM job_history WHERE job_id = %s",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_dict(cols, row)

    @staticmethod
    def _db_list(
        status: str | None = None,
        job_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        where: list[str] = []
        params: list[Any] = []
        if status:
            where.append("status = %s")
            params.append(status)
        if job_type:
            where.append("job_type = %s")
            params.append(job_type)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        cols = ("job_id", "job_type", "job_label", "status", "params", "result",
                "error", "submitted_at", "started_at", "completed_at",
                "progress_pct", "progress_msg")

        with _get_conn() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM job_history {where_sql}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"""SELECT {', '.join(cols)}
                    FROM job_history {where_sql}
                    ORDER BY submitted_at DESC LIMIT %s OFFSET %s""",
                [*params, limit, offset],
            ).fetchall()

        jobs = [_row_to_dict(cols, r) for r in rows]
        return jobs, int(total)

    @staticmethod
    def _db_delete(job_id: str) -> bool:
        with _get_conn() as conn:
            result = conn.execute(
                "DELETE FROM job_history WHERE job_id = %s AND status IN ('completed', 'failed', 'cancelled')",
                (job_id,),
            )
            return result.rowcount > 0

    @staticmethod
    def _db_stats() -> dict[str, Any]:
        """Aggregate statistics for the job dashboard."""
        with _get_conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE status IN ('running', 'queued')) AS active,
                    COUNT(*) FILTER (WHERE status = 'completed') AS completed,
                    COUNT(*) FILTER (WHERE status = 'failed') AS failed,
                    COUNT(*) FILTER (WHERE status = 'cancelled') AS cancelled,
                    COALESCE(AVG(EXTRACT(EPOCH FROM (completed_at - started_at)))
                             FILTER (WHERE completed_at IS NOT NULL AND started_at IS NOT NULL), 0)
                        AS avg_duration_seconds,
                    COUNT(*) FILTER (WHERE submitted_at > NOW() - INTERVAL '24 hours') AS last_24h_submitted,
                    COUNT(*) FILTER (WHERE status = 'completed'
                                     AND completed_at > NOW() - INTERVAL '24 hours') AS last_24h_completed,
                    COUNT(*) FILTER (WHERE status = 'failed'
                                     AND completed_at > NOW() - INTERVAL '24 hours') AS last_24h_failed
                FROM job_history
            """).fetchone()
        return {
            "total": row[0],
            "active": row[1],
            "completed": row[2],
            "failed": row[3],
            "cancelled": row[4],
            "avg_duration_seconds": round(row[5], 1),
            "last_24h": {
                "submitted": row[6],
                "completed": row[7],
                "failed": row[8],
            },
        }

    # ---- public API ----

    def submit_job(
        self,
        job_type: str,
        params: dict[str, Any] | None = None,
        label: str | None = None,
        triggered_by: str = "manual",
        max_retries: int = 0,
        pipeline_id: str | None = None,
        pipeline_step: int | None = None,
    ) -> str:
        """Submit a job for immediate background execution. Returns job_id.

        The job is dispatched to APScheduler's managed thread pool for execution.
        Per-group concurrency control ensures only one job runs per group at a time.
        """
        self._ensure_init()

        if job_type not in JOB_TYPE_REGISTRY:
            raise ValueError(f"Unknown job type: {job_type}")

        type_def = JOB_TYPE_REGISTRY[job_type]

        job_id = self._generate_id()
        job_label = label or type_def.label
        job_params = params or {}

        with self._state_lock:
            if self._is_group_busy(type_def.group):
                # Queue the job instead of rejecting
                self._db_insert(
                    job_id, job_type, job_label, job_params,
                    triggered_by=triggered_by,
                    pipeline_id=pipeline_id,
                    pipeline_step=pipeline_step,
                    max_retries=max_retries,
                )
                self._db_update_status(job_id, "queued", progress_msg="Waiting for group to be free")
                queue = self._pending_queues.setdefault(type_def.group, [])
                queue.append((job_id, type_def, job_params, max_retries, pipeline_id))
                logger.info("Queued job %s (%s) — group '%s' is busy", job_id, job_type, type_def.group)
                return job_id

            # Track as active (inside lock to prevent race with _dispatch_next)
            self._active_jobs[job_id] = type_def.group
            self._cancel_flags[job_id] = threading.Event()

        # DB insert and APScheduler dispatch outside lock (no contention on I/O)
        self._db_insert(
            job_id, job_type, job_label, job_params,
            triggered_by=triggered_by,
            pipeline_id=pipeline_id,
            pipeline_step=pipeline_step,
            max_retries=max_retries,
        )

        self._scheduler.add_job(
            self._execute_job,
            args=[job_id, type_def, job_params, max_retries, pipeline_id],
            id=job_id,
            name=f"{type_def.label}: {job_label}",
            replace_existing=True,
        )
        logger.info("Submitted job %s (%s) via APScheduler", job_id, job_type)
        return job_id

    def start_job_in_background(self, job_id: str) -> None:
        """Start executing a previously submitted job.

        Backward-compatible method for code that calls submit_job() then
        start_job_in_background() separately (e.g. clusters router).
        With APScheduler, submit_job() already dispatches execution, so
        this is a no-op if the job is already scheduled.
        """
        self._ensure_init()
        # APScheduler already scheduled the job in submit_job()
        # This method exists for backward compatibility
        pass

    def schedule_recurring(
        self,
        job_type: str,
        params: dict[str, Any] | None = None,
        label: str | None = None,
        cron: str | None = None,
        interval_minutes: int | None = None,
    ) -> str:
        """Schedule a recurring job via cron expression or interval.

        Returns a schedule_id that can be used to remove the schedule.
        """
        self._ensure_init()

        if job_type not in JOB_TYPE_REGISTRY:
            raise ValueError(f"Unknown job type: {job_type}")
        if not cron and not interval_minutes:
            raise ValueError("Must specify either cron or interval_minutes")

        type_def = JOB_TYPE_REGISTRY[job_type]
        schedule_id = f"sched_{uuid.uuid4().hex[:8]}"
        job_label = label or f"Scheduled: {type_def.label}"
        job_params = params or {}

        trigger = make_trigger(cron, interval_minutes)

        # Add to APScheduler
        self._scheduler.add_job(
            self._scheduled_execution,
            trigger=trigger,
            args=[job_type, job_params, job_label],
            id=schedule_id,
            name=job_label,
            replace_existing=True,
        )

        # Persist schedule in DB
        next_run = self._scheduler.get_job(schedule_id)
        next_run_at = next_run.next_run_time if next_run else None
        try:
            with _get_conn() as conn:
                conn.execute(
                    """INSERT INTO job_schedule
                       (schedule_id, job_type, job_label, cron_expr, interval_min, params, next_run_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (schedule_id) DO UPDATE SET
                       cron_expr = EXCLUDED.cron_expr, interval_min = EXCLUDED.interval_min,
                       params = EXCLUDED.params, enabled = TRUE, next_run_at = EXCLUDED.next_run_at""",
                    (schedule_id, job_type, job_label, cron, interval_minutes,
                     json.dumps(job_params), next_run_at),
                )
        except Exception:
            logger.warning("job_schedule table may not exist yet; schedule only in memory")

        logger.info("Scheduled recurring %s as %s (cron=%s, interval=%s min)",
                     job_type, schedule_id, cron, interval_minutes)
        return schedule_id

    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a recurring schedule."""
        self._ensure_init()
        try:
            self._scheduler.remove_job(schedule_id)
        except Exception:
            pass
        try:
            with _get_conn() as conn:
                res = conn.execute(
                    "DELETE FROM job_schedule WHERE schedule_id = %s", (schedule_id,))
                return res.rowcount > 0
        except Exception:
            return False

    def list_schedules(self) -> list[dict[str, Any]]:
        """List all active recurring schedules."""
        self._ensure_init()
        try:
            with _get_conn() as conn:
                rows = conn.execute(
                    """SELECT schedule_id, job_type, job_label, cron_expr, interval_min,
                              params, enabled, created_at, last_run_at, next_run_at, run_count
                       FROM job_schedule WHERE enabled = TRUE
                       ORDER BY created_at DESC"""
                ).fetchall()
            schedules = []
            for r in rows:
                schedules.append({
                    "schedule_id": r[0], "job_type": r[1], "job_label": r[2],
                    "cron_expr": r[3], "interval_min": r[4],
                    "params": r[5] if isinstance(r[5], dict) else json.loads(r[5] or "{}"),
                    "enabled": r[6],
                    "created_at": r[7].isoformat() if r[7] else None,
                    "last_run_at": r[8].isoformat() if r[8] else None,
                    "next_run_at": r[9].isoformat() if r[9] else None,
                    "run_count": r[10] or 0,
                })
            return schedules
        except Exception:
            # table may not exist yet
            return []

    def submit_pipeline(
        self,
        steps: list[dict[str, Any]],
        label: str = "Pipeline",
        triggered_by: str = "manual",
    ) -> str:
        """Submit a pipeline (chain) of jobs to run sequentially.

        Each step: {"job_type": str, "params": dict, "label": str (optional)}
        Returns pipeline_id. Jobs are submitted one at a time as each completes.
        """
        self._ensure_init()
        pipeline_id = f"pipe_{uuid.uuid4().hex[:8]}"

        # Submit the first step; subsequent steps triggered on completion
        if not steps:
            raise ValueError("Pipeline must have at least one step")

        first = steps[0]
        job_type = first["job_type"]
        params = first.get("params", {})
        step_label = first.get("label", JOB_TYPE_REGISTRY[job_type].label)

        # Store remaining steps in the first job's params
        params["__pipeline_remaining"] = steps[1:]
        params["__pipeline_label"] = label

        self.submit_job(
            job_type=job_type,
            params=params,
            label=f"[{label} 1/{len(steps)}] {step_label}",
            triggered_by=triggered_by,
            pipeline_id=pipeline_id,
            pipeline_step=1,
        )

        logger.info("Submitted pipeline %s with %d steps", pipeline_id, len(steps))
        return pipeline_id

    def get_status(self, job_id: str) -> dict[str, Any] | None:
        """Get current job status from DB."""
        return self._db_get(job_id)

    def list_jobs(
        self,
        status: str | None = None,
        job_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List jobs with optional filters."""
        return self._db_list(status=status, job_type=job_type, limit=limit, offset=offset)

    def get_active_jobs(self) -> list[dict[str, Any]]:
        """Get all currently running/queued jobs."""
        jobs, _ = self._db_list(status="running", limit=100)
        queued, _ = self._db_list(status="queued", limit=100)
        return jobs + queued

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate job statistics for dashboard."""
        return self._db_stats()

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        job = self._db_get(job_id)
        if job is None:
            return False
        if job["status"] not in ("queued", "running"):
            return False
        with self._state_lock:
            cancel_event = self._cancel_flags.get(job_id)
            if cancel_event:
                cancel_event.set()
            self._active_jobs.pop(job_id, None)
            self._cancel_flags.pop(job_id, None)
        # Remove from APScheduler if still pending
        try:
            self._scheduler.remove_job(job_id)
        except Exception:
            pass
        self._db_update_status(
            job_id, "cancelled",
            completed_at=datetime.now(timezone.utc),
            progress_msg="Cancelled by user",
        )
        return True

    def delete_job(self, job_id: str) -> bool:
        """Delete a completed/failed/cancelled job from history."""
        return self._db_delete(job_id)

    def recover_stale_jobs(self) -> int:
        """On startup: fail stale 'running' jobs, re-enqueue 'queued' jobs."""
        recovered = 0

        # 1. Mark running jobs as failed (they were interrupted mid-execution)
        with _get_conn() as conn:
            result = conn.execute(
                """UPDATE job_history SET status = 'failed',
                          error = 'Interrupted by server restart',
                          completed_at = NOW()
                   WHERE status = 'running'"""
            )
            recovered += result.rowcount

        # 2. Re-enqueue queued jobs into memory (in submission order)
        try:
            with _get_conn() as conn:
                rows = conn.execute(
                    """SELECT job_id, job_type, params, max_retries, pipeline_id
                       FROM job_history
                       WHERE status = 'queued'
                       ORDER BY submitted_at ASC"""
                ).fetchall()
            for row in rows:
                job_id, job_type, params_raw, max_retries, pipeline_id = row
                type_def = JOB_TYPE_REGISTRY.get(job_type)
                if not type_def:
                    # Unknown job type — mark as failed
                    self._db_update_status(
                        job_id, "failed",
                        error=f"Unknown job type '{job_type}' on restart",
                        completed_at=datetime.now(timezone.utc),
                    )
                    recovered += 1
                    continue
                params = params_raw if isinstance(params_raw, dict) else json.loads(params_raw or "{}")
                with self._state_lock:
                    queue = self._pending_queues.setdefault(type_def.group, [])
                    queue.append((job_id, type_def, params, max_retries or 0, pipeline_id))
                recovered += 1
                logger.info("Re-enqueued job %s (%s) from DB on startup", job_id, job_type)
        except Exception:
            logger.exception("Failed to re-enqueue queued jobs on startup")

        return recovered

    def get_types(self) -> list[dict[str, Any]]:
        """List all registered job types with metadata."""
        return [
            {
                "type_id": t.type_id,
                "label": t.label,
                "description": t.description,
                "group": t.group,
                "params_schema": t.params_schema,
            }
            for t in JOB_TYPE_REGISTRY.values()
        ]

    # ---- internal execution ----

    def _execute_job(
        self,
        job_id: str,
        type_def: JobTypeDef,
        params: dict[str, Any],
        max_retries: int = 0,
        pipeline_id: str | None = None,
    ) -> None:
        """Execute a job within APScheduler's thread pool."""
        retry_count = 0

        while True:
            try:
                self._db_update_status(
                    job_id, "running",
                    started_at=datetime.now(timezone.utc),
                    progress_pct=0,
                    progress_msg="Starting",
                    retry_count=retry_count,
                )

                def progress_cb(pct: int | None = None, msg: str | None = None):
                    updates: dict[str, Any] = {}
                    if pct is not None:
                        updates["progress_pct"] = pct
                    if msg is not None:
                        updates["progress_msg"] = msg
                    if updates:
                        self._db_update_status(job_id, "running", **updates)

                # Strip pipeline metadata before passing to callable
                clean_params = {k: v for k, v in params.items() if not k.startswith("__pipeline")}
                result = type_def.callable(clean_params, progress_cb)

                self._db_update_status(
                    job_id, "completed",
                    completed_at=datetime.now(timezone.utc),
                    progress_pct=100,
                    progress_msg="Done",
                    result=result,
                )
                logger.info("Job %s completed successfully", job_id)

                # If part of a pipeline, trigger next step
                remaining = params.get("__pipeline_remaining", [])
                if remaining and pipeline_id:
                    self._trigger_next_pipeline_step(pipeline_id, remaining, params)

                break  # success

            except Exception as exc:
                retry_count += 1
                if retry_count <= max_retries:
                    wait = min(2 ** retry_count, 60)
                    logger.warning("Job %s failed (attempt %d/%d), retrying in %ds: %s",
                                   job_id, retry_count, max_retries + 1, wait, exc)
                    self._db_update_status(
                        job_id, "running",
                        progress_msg=f"Retry {retry_count}/{max_retries} in {wait}s",
                        retry_count=retry_count,
                    )
                    time.sleep(wait)
                    continue

                logger.exception("Job %s failed after %d attempts", job_id, retry_count)
                self._db_update_status(
                    job_id, "failed",
                    completed_at=datetime.now(timezone.utc),
                    error=str(exc),
                    progress_msg="Failed",
                    retry_count=retry_count,
                )
                break

            finally:
                with self._state_lock:
                    was_active = job_id in self._active_jobs
                    self._active_jobs.pop(job_id, None)
                    self._cancel_flags.pop(job_id, None)
                # Auto-dispatch next queued job (outside lock to avoid deadlock)
                if was_active:
                    self._dispatch_next(type_def.group)

    def _dispatch_next(self, group: str) -> None:
        """Pop the next queued job for *group* and dispatch it to APScheduler."""
        with self._state_lock:
            queue = self._pending_queues.get(group)
            if not queue:
                return

            job_id, type_def, params, max_retries, pipeline_id = queue.pop(0)
            remaining = len(queue)

            # Track as active (inside lock)
            self._active_jobs[job_id] = group
            self._cancel_flags[job_id] = threading.Event()

        # APScheduler dispatch outside lock (I/O operation)
        logger.info("Auto-dispatching queued job %s for group '%s' (%d remaining in queue)",
                     job_id, type_def.type_id, remaining)
        self._scheduler.add_job(
            self._execute_job,
            args=[job_id, type_def, params, max_retries, pipeline_id],
            id=job_id,
            name=f"{type_def.label}: {job_id}",
            replace_existing=True,
        )

    def _scheduled_execution(self, job_type: str, params: dict, label: str) -> None:
        """Execute a scheduled recurring job by submitting a new one-off instance."""
        type_def = JOB_TYPE_REGISTRY.get(job_type)
        if not type_def:
            logger.error("Scheduled job type %s not in registry", job_type)
            return
        with self._state_lock:
            busy = self._is_group_busy(type_def.group)
        if busy:
            logger.warning("Skipping scheduled %s — group '%s' is busy", job_type, type_def.group)
            return
        try:
            self.submit_job(job_type, params, label, triggered_by="schedule")
        except Exception as exc:
            logger.exception("Failed to submit scheduled job %s: %s", job_type, exc)

    def _trigger_next_pipeline_step(
        self, pipeline_id: str, remaining: list[dict], original_params: dict
    ) -> None:
        """Submit the next step in a pipeline."""
        if not remaining:
            return
        next_step = remaining[0]
        future_steps = remaining[1:]
        job_type = next_step["job_type"]
        params = next_step.get("params", {})
        step_label = next_step.get("label", JOB_TYPE_REGISTRY[job_type].label)
        total_steps = len(future_steps) + 1 + (original_params.get("__pipeline_step", 1))
        step_num = total_steps - len(future_steps)

        if future_steps:
            params["__pipeline_remaining"] = future_steps
            params["__pipeline_label"] = original_params.get("__pipeline_label", "Pipeline")

        try:
            self.submit_job(
                job_type=job_type,
                params=params,
                label=f"[Pipeline {step_num}/{total_steps}] {step_label}",
                triggered_by="pipeline",
                pipeline_id=pipeline_id,
                pipeline_step=step_num,
            )
        except Exception as exc:
            logger.exception("Pipeline %s step %d failed to submit: %s", pipeline_id, step_num, exc)
