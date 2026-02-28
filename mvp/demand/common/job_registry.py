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
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import psycopg
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB helpers (lightweight — avoid importing api.core at module level so
# that this module can be used from tests without starting the full API)
# ---------------------------------------------------------------------------

def _get_conn():
    """Open a single psycopg connection using environment variables."""
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5440")),
        dbname=os.getenv("POSTGRES_DB", "demand_mvp"),
        user=os.getenv("POSTGRES_USER", "demand"),
        password=os.getenv("POSTGRES_PASSWORD", "demand"),
        autocommit=True,
    )


# ---------------------------------------------------------------------------
# Job type definition
# ---------------------------------------------------------------------------

@dataclass
class JobTypeDef:
    """Metadata for a registered job type."""
    type_id: str
    label: str
    description: str
    group: str  # concurrency group — one active job per group
    callable: Callable[..., dict[str, Any]]  # (params, progress_cb) -> result dict
    params_schema: dict[str, Any] = field(default_factory=dict)
    default_max_retries: int = 0


# ---------------------------------------------------------------------------
# Job type callables — thin wrappers around existing scripts
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
_UV = "uv"


def _run_subprocess(cmd: list[str], progress_cb: Callable | None = None, step_msg: str = "") -> str:
    """Run a subprocess command, returning stdout. Raises on failure."""
    if progress_cb and step_msg:
        progress_cb(msg=step_msg)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_SCRIPTS_DIR.parent))
    if result.returncode != 0:
        error_msg = (result.stderr or result.stdout or "Unknown error").strip()
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{error_msg}")
    return result.stdout


def _run_cluster_scenario(params: dict[str, Any], progress_cb: Callable | None = None) -> dict[str, Any]:
    """Run a what-if clustering scenario (delegates to run_clustering_scenario.py)."""
    from scripts.run_clustering_scenario import run_scenario, generate_scenario_id, get_scenario_result

    scenario_id = params.get("scenario_id") or generate_scenario_id()
    if progress_cb:
        progress_cb(pct=5, msg="Starting clustering scenario")

    run_scenario(
        scenario_id=scenario_id,
        feature_params=params.get("feature_params"),
        model_params=params.get("model_params"),
        label_params=params.get("label_params"),
        relabel_only=params.get("relabel_only", False),
        previous_scenario_id=params.get("previous_scenario_id"),
    )

    result = get_scenario_result(scenario_id) or {}
    return {"scenario_id": scenario_id, **result}


def _run_cluster_pipeline(params: dict[str, Any], progress_cb: Callable | None = None) -> dict[str, Any]:
    """Run the full clustering pipeline: features -> train -> label -> update."""
    tw = params.get("time_window_months", 24)
    k_range = params.get("k_range", [3, 12])
    steps = [
        (25, "Generating clustering features",
         [_UV, "run", "python", "scripts/generate_clustering_features.py", "--time-window", str(tw)]),
        (50, "Training clustering model",
         [_UV, "run", "python", "scripts/train_clustering_model.py",
          "--k-range", str(k_range[0]), str(k_range[1]), "--skip-gap"]),
        (75, "Labeling clusters",
         [_UV, "run", "python", "scripts/label_clusters.py"]),
        (95, "Updating DFU assignments",
         [_UV, "run", "python", "scripts/update_cluster_assignments.py"]),
    ]
    outputs = []
    for pct, msg, cmd in steps:
        if progress_cb:
            progress_cb(pct=pct, msg=msg)
        out = _run_subprocess(cmd)
        outputs.append(out)
    return {"steps_completed": len(steps), "output_summary": "Pipeline completed successfully"}


def _run_seasonality(params: dict[str, Any], progress_cb: Callable | None = None) -> dict[str, Any]:
    """Run the seasonality detection + update pipeline."""
    config = params.get("config", "config/seasonality_config.yaml")
    steps = [
        (40, "Detecting seasonality patterns",
         [_UV, "run", "python", "scripts/detect_seasonality.py", "--config", config]),
        (90, "Updating seasonality profiles",
         [_UV, "run", "python", "scripts/update_seasonality_profiles.py", "--config", config]),
    ]
    for pct, msg, cmd in steps:
        if progress_cb:
            progress_cb(pct=pct, msg=msg)
        _run_subprocess(cmd)
    return {"steps_completed": len(steps), "output_summary": "Seasonality pipeline completed"}


def _run_backtest(model: str, params: dict[str, Any], progress_cb: Callable | None = None) -> dict[str, Any]:
    """Run a backtest for a given model type."""
    script_map = {
        "lgbm": "scripts/run_backtest.py",
        "catboost": "scripts/run_backtest_catboost.py",
        "xgboost": "scripts/run_backtest_xgboost.py",
    }
    script = script_map.get(model)
    if not script:
        raise ValueError(f"Unknown backtest model: {model}")
    strategy = params.get("cluster_strategy", "global")
    if progress_cb:
        progress_cb(pct=10, msg=f"Running {model.upper()} backtest ({strategy})")
    cmd = [_UV, "run", "python", script, "--cluster-strategy", strategy]
    output = _run_subprocess(cmd)
    return {"model": model, "strategy": strategy, "output_summary": output[:500] if output else "Completed"}


def _run_backtest_lgbm(params: dict[str, Any], progress_cb: Callable | None = None) -> dict[str, Any]:
    return _run_backtest("lgbm", params, progress_cb)


def _run_backtest_catboost(params: dict[str, Any], progress_cb: Callable | None = None) -> dict[str, Any]:
    return _run_backtest("catboost", params, progress_cb)


def _run_backtest_xgboost(params: dict[str, Any], progress_cb: Callable | None = None) -> dict[str, Any]:
    return _run_backtest("xgboost", params, progress_cb)


def _run_champion_select(params: dict[str, Any], progress_cb: Callable | None = None) -> dict[str, Any]:
    """Run champion model selection."""
    if progress_cb:
        progress_cb(pct=10, msg="Running champion selection")
    cmd = [_UV, "run", "python", "scripts/run_champion_selection.py"]
    output = _run_subprocess(cmd)
    return {"output_summary": output[:500] if output else "Champion selection completed"}


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
        """Lazily initialise APScheduler and internal state."""
        if self._initialized:
            return
        executors = {
            "default": ThreadPoolExecutor(max_workers=4),
        }
        job_defaults = {
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": 3600,
        }
        self._scheduler = BackgroundScheduler(
            executors=executors,
            job_defaults=job_defaults,
            timezone="UTC",
        )
        self._scheduler.start()
        self._group_locks: dict[str, threading.Lock] = {}
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

    def _get_group_lock(self, group: str) -> threading.Lock:
        if group not in self._group_locks:
            self._group_locks[group] = threading.Lock()
        return self._group_locks[group]

    def _is_group_busy(self, group: str) -> bool:
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

        if self._is_group_busy(type_def.group):
            # Queue the job instead of rejecting
            self._db_insert(
                job_id, job_type, job_label, job_params,
                triggered_by=triggered_by,
                pipeline_id=pipeline_id,
                pipeline_step=pipeline_step,
                max_retries=max_retries,
            )
            # Mark as queued in DB
            self._db_update_status(job_id, "queued", progress_msg="Waiting for group to be free")
            queue = self._pending_queues.setdefault(type_def.group, [])
            queue.append((job_id, type_def, job_params, max_retries, pipeline_id))
            logger.info("Queued job %s (%s) — group '%s' is busy", job_id, job_type, type_def.group)
            return job_id

        # Insert into DB
        self._db_insert(
            job_id, job_type, job_label, job_params,
            triggered_by=triggered_by,
            pipeline_id=pipeline_id,
            pipeline_step=pipeline_step,
            max_retries=max_retries,
        )

        # Track as active
        self._active_jobs[job_id] = type_def.group
        self._cancel_flags[job_id] = threading.Event()

        # Dispatch to APScheduler's thread pool
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

        if cron:
            trigger = CronTrigger.from_crontab(cron)
        else:
            trigger = IntervalTrigger(minutes=interval_minutes)

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
        cancel_event = self._cancel_flags.get(job_id)
        if cancel_event:
            cancel_event.set()
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
        self._active_jobs.pop(job_id, None)
        self._cancel_flags.pop(job_id, None)
        return True

    def delete_job(self, job_id: str) -> bool:
        """Delete a completed/failed/cancelled job from history."""
        return self._db_delete(job_id)

    def recover_stale_jobs(self) -> int:
        """Mark any leftover 'running'/'queued' jobs as failed (after restart)."""
        with _get_conn() as conn:
            result = conn.execute(
                """UPDATE job_history SET status = 'failed',
                          error = 'Interrupted by server restart',
                          completed_at = NOW()
                   WHERE status IN ('running', 'queued')"""
            )
            return result.rowcount

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
                if job_id in self._active_jobs:
                    self._active_jobs.pop(job_id, None)
                    self._cancel_flags.pop(job_id, None)
                    # Auto-dispatch next queued job for this group
                    self._dispatch_next(type_def.group)

    def _dispatch_next(self, group: str) -> None:
        """Pop the next queued job for *group* and dispatch it to APScheduler."""
        queue = self._pending_queues.get(group)
        if not queue:
            return

        job_id, type_def, params, max_retries, pipeline_id = queue.pop(0)
        logger.info("Auto-dispatching queued job %s for group '%s' (%d remaining in queue)",
                     job_id, type_def.type_id, len(queue))

        # Track as active
        self._active_jobs[job_id] = group
        self._cancel_flags[job_id] = threading.Event()

        # Dispatch to APScheduler's thread pool
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
        if self._is_group_busy(type_def.group):
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


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _row_to_dict(cols: tuple[str, ...], row: tuple) -> dict[str, Any]:
    """Convert a DB row tuple to a dictionary with proper JSON/datetime handling."""
    d: dict[str, Any] = {}
    for i, col in enumerate(cols):
        val = row[i]
        if col in ("params", "result"):
            if isinstance(val, dict):
                d[col] = val
            elif val:
                d[col] = json.loads(val)
            else:
                d[col] = {} if col == "params" else None
        elif col in ("submitted_at", "started_at", "completed_at"):
            d[col] = val.isoformat() if val else None
        elif col == "progress_pct":
            d[col] = val or 0
        else:
            d[col] = val
    return d
