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
    from common.services.job_registry import JobManager, JOB_TYPE_REGISTRY
"""
from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Sub-module imports
# ---------------------------------------------------------------------------
from common.services.job_state import (
    MODEL_OUTPUT_DIRS,
    JobTypeDef,
    _get_conn,
    _row_to_dict,
    get_job_log,
    get_job_pid,
    _run_assign_policies,
    _run_backtest_catboost,
    _run_backtest_chronos2_enriched,
    _run_backtest_lgbm,
    _run_backtest_mstl,
    _run_backtest_nbeats,
    _run_backtest_nhits,
    _run_backtest_rolling_mean,
    _run_backtest_rolling_median,
    _run_backtest_seasonal_naive,
    _run_backtest_xgboost,
    _run_champion_experiment,
    _run_champion_results_load,
    _run_champion_select,
    _run_champion_sweep,
    _run_classify_abc_xyz,
    _run_cluster_pipeline,
    _run_cluster_scenario,
    _run_compare_inventory_algorithms,
    _run_compute_demand_signals,
    _run_compute_eoq,
    _run_compute_investment,
    _run_compute_replenishment_plan,
    _run_compute_safety_stock,
    _run_inventory_backtest,
    _run_inventory_planning_pipeline,
    _run_compute_sku_features,
    _run_compute_variability,
    _run_data_quality,
    _run_etl_pipeline,
    _run_generate_ai_insights,
    _run_load_domain,
    _run_generate_exceptions,
    _run_generate_production_forecast,
    _run_train_production_model,
    _run_generate_storyboard,
    _run_load_backtest_model,
    _run_load_backtest_results,
    _run_model_tuning_experiment,
    _run_refresh_customer_analytics,
    _run_refresh_forecast_views,
    _run_refresh_health_scores,
    _run_refresh_intramonth,
    _run_seasonality,
    _run_ss_simulation,
    _run_tuning_backtest,
)
from common.services.job_scheduler import make_scheduler, make_trigger

try:
    from apscheduler.jobstores.base import JobLookupError
except ImportError:
    JobLookupError = KeyError  # type: ignore[assignment,misc]

import psycopg

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
        description="Generate features, train model, label clusters, update SKU table",
        group="clustering",
        callable=_run_cluster_pipeline,
        params_schema={"time_window_months": 24, "k_range": [3, 12]},
    ),
    "compute_sku_features": JobTypeDef(
        type_id="compute_sku_features",
        label="Compute SKU Features",
        description="Compute all time-series features (volume, trend, seasonality, variability, lifecycle) for all SKUs",
        group="features",
        callable=_run_compute_sku_features,
        params_schema={"time_window_months": 36},
    ),
    "seasonality_pipeline": JobTypeDef(
        type_id="seasonality_pipeline",
        label="Seasonality Detection",
        description="Detect seasonality patterns and update SKU profiles (delegates to compute_sku_features)",
        group="features",
        callable=_run_seasonality,
        params_schema={"time_window_months": 36},
    ),
    # -- Tree models --
    "backtest_lgbm": JobTypeDef(
        type_id="backtest_lgbm",
        label="LightGBM Backtest",
        description="Run LightGBM gradient boosting backtest",
        group="backtest",
        callable=_run_backtest_lgbm,
        params_schema={},
    ),
    "backtest_catboost": JobTypeDef(
        type_id="backtest_catboost",
        label="CatBoost Backtest",
        description="Run CatBoost gradient boosting backtest",
        group="backtest",
        callable=_run_backtest_catboost,
        params_schema={},
    ),
    "backtest_xgboost": JobTypeDef(
        type_id="backtest_xgboost",
        label="XGBoost Backtest",
        description="Run XGBoost gradient boosting backtest",
        group="backtest",
        callable=_run_backtest_xgboost,
        params_schema={},
    ),
    # -- Foundation models --
    "backtest_chronos2_enriched": JobTypeDef(
        type_id="backtest_chronos2_enriched",
        label="Chronos 2 Enriched Backtest",
        description="Run Chronos 2 with 31 covariates backtest (~6h)",
        group="backtest",
        callable=_run_backtest_chronos2_enriched,
        params_schema={},
    ),
    # -- Statistical baselines --
    "backtest_seasonal_naive": JobTypeDef(
        type_id="backtest_seasonal_naive",
        label="Seasonal Naive Backtest",
        description="Run Seasonal Naive statistical baseline backtest (~5 min)",
        group="backtest",
        callable=_run_backtest_seasonal_naive,
        params_schema={},
    ),
    "backtest_rolling_mean": JobTypeDef(
        type_id="backtest_rolling_mean",
        label="Rolling Mean Backtest",
        description="Run Rolling Mean statistical baseline backtest (~5 min)",
        group="backtest",
        callable=_run_backtest_rolling_mean,
        params_schema={},
    ),
    "backtest_rolling_median": JobTypeDef(
        type_id="backtest_rolling_median",
        label="Rolling Median Backtest",
        description="Run Rolling Median statistical baseline backtest (~5 min)",
        group="backtest",
        callable=_run_backtest_rolling_median,
        params_schema={},
    ),
    "backtest_mstl": JobTypeDef(
        type_id="backtest_mstl",
        label="MSTL Backtest",
        description="Run MSTL decomposition backtest (~15 min)",
        group="backtest",
        callable=_run_backtest_mstl,
        params_schema={},
    ),
    # -- Deep learning --
    "backtest_nhits": JobTypeDef(
        type_id="backtest_nhits",
        label="N-HiTS Backtest",
        description="Run N-HiTS deep learning backtest (~1h)",
        group="backtest",
        callable=_run_backtest_nhits,
        params_schema={},
    ),
    "backtest_nbeats": JobTypeDef(
        type_id="backtest_nbeats",
        label="N-BEATS Backtest",
        description="Run N-BEATS deep learning backtest (~1h)",
        group="backtest",
        callable=_run_backtest_nbeats,
        params_schema={},
    ),
    "champion_select": JobTypeDef(
        type_id="champion_select",
        label="Champion Selection",
        description="Select per-DFU champion model via rolling WAPE comparison",
        group="champion",
        callable=_run_champion_select,
        params_schema={},
    ),
    "champion_experiment": JobTypeDef(
        type_id="champion_experiment",
        label="Champion Experiment",
        description="Run champion selection strategy experiment with configurable params",
        group="champion",
        callable=_run_champion_experiment,
        params_schema={"experiment_id": 0},
    ),
    "champion_results_load": JobTypeDef(
        type_id="champion_results_load",
        label="Load Champion Results",
        description="Run champion selection and load results into forecast tables",
        group="champion",
        callable=_run_champion_results_load,
        params_schema={"experiment_id": 0},
    ),
    "champion_sweep": JobTypeDef(
        type_id="champion_sweep",
        label="Champion Sweep",
        description="Fan out a grid of champion configs, rank globally + per segment, recommend a winner",
        group="champion",
        callable=_run_champion_sweep,
        params_schema={"sweep_id": 0},
    ),
    "generate_ai_insights": JobTypeDef(
        type_id="generate_ai_insights",
        label="AI Insight Generation",
        description="Scan portfolio for planning exceptions and generate AI insights",
        group="ai",
        callable=_run_generate_ai_insights,
        params_schema={},
    ),
    "train_production_model": JobTypeDef(
        type_id="train_production_model",
        label="Train Production Model",
        description="Train model on full history for production forecasting",
        group="forecast",
        callable=_run_train_production_model,
        params_schema={"model_id": "", "all_models": False},
    ),
    "generate_production_forecast": JobTypeDef(
        type_id="generate_production_forecast",
        label="Production Forecast",
        description="Generate future-period production forecasts using champion ML models",
        group="forecast",
        callable=_run_generate_production_forecast,
        params_schema={"horizon": 12, "model_id": None, "confidence_intervals": None},
    ),
    "compute_replenishment_plan": JobTypeDef(
        type_id="compute_replenishment_plan",
        label="Replenishment Plan",
        description="Forward-looking replenishment plan from production forecast CI bands",
        group="replenishment",
        callable=_run_compute_replenishment_plan,
        params_schema={},
    ),
    # ── Storyboard (ai group) ────────────────────────────────────────────────
    "generate_storyboard": JobTypeDef(
        type_id="generate_storyboard",
        label="Storyboard Exceptions",
        description="Detect planning exceptions and generate storyboard cards",
        group="ai",
        callable=_run_generate_storyboard,
        params_schema={},
    ),
    # ── Inventory Planning (inventory group) ────────────────────────────────
    "inventory_planning_pipeline": JobTypeDef(
        type_id="inventory_planning_pipeline",
        label="Inventory Planning Pipeline",
        description="End-to-end: SS → EOQ → Repl Plan → Planned Orders → Exceptions",
        group="inventory",
        callable=_run_inventory_planning_pipeline,
        params_schema={"steps": None},
    ),
    "inventory_backtest": JobTypeDef(
        type_id="inventory_backtest",
        label="Inventory Backtest",
        description="Simulate inventory outcomes using historical forecast predictions",
        group="inventory",
        callable=_run_inventory_backtest,
        params_schema={"models": None, "months": 12},
    ),
    "compute_safety_stock": JobTypeDef(
        type_id="compute_safety_stock",
        label="Safety Stock",
        description="Compute Z-score safety stock targets for all DFUs",
        group="inventory",
        callable=_run_compute_safety_stock,
        params_schema={"forecast_source": "historical", "model_id": None},
    ),
    "compute_eoq": JobTypeDef(
        type_id="compute_eoq",
        label="EOQ Targets",
        description="Compute economic order quantity cycle stock targets",
        group="inventory",
        callable=_run_compute_eoq,
        params_schema={},
    ),
    "assign_policies": JobTypeDef(
        type_id="assign_policies",
        label="Policy Assignment",
        description="Upsert replenishment policies and auto-assign DFUs by segment",
        group="inventory",
        callable=_run_assign_policies,
        params_schema={},
    ),
    "generate_exceptions": JobTypeDef(
        type_id="generate_exceptions",
        label="Exception Detection",
        description="Detect replenishment exceptions and write to the exception queue",
        group="inventory",
        callable=_run_generate_exceptions,
        params_schema={},
    ),
    "classify_abc_xyz": JobTypeDef(
        type_id="classify_abc_xyz",
        label="ABC-XYZ Classification",
        description="Classify DFUs by volume (ABC) and variability (XYZ) and update dim_sku",
        group="inventory",
        callable=_run_classify_abc_xyz,
        params_schema={},
    ),
    "compute_variability": JobTypeDef(
        type_id="compute_variability",
        label="Demand Variability",
        description="Compute demand CV, MAD, and volatility profiles per DFU (delegates to compute_sku_features)",
        group="features",
        callable=_run_compute_variability,
        params_schema={},
    ),
    "compute_demand_signals": JobTypeDef(
        type_id="compute_demand_signals",
        label="Demand Signals",
        description="Compute short-horizon demand signals from sales velocity and inventory movement",
        group="inventory",
        callable=_run_compute_demand_signals,
        params_schema={},
    ),
    "compute_investment": JobTypeDef(
        type_id="compute_investment",
        label="Investment Plan",
        description="Compute efficient frontier and capital investment allocation across DFUs",
        group="inventory",
        callable=_run_compute_investment,
        params_schema={},
    ),
    "refresh_health_scores": JobTypeDef(
        type_id="refresh_health_scores",
        label="Health Scores",
        description="Refresh the inventory health score materialized view",
        group="inventory",
        callable=_run_refresh_health_scores,
        params_schema={},
    ),
    "refresh_intramonth": JobTypeDef(
        type_id="refresh_intramonth",
        label="Intramonth Stockout",
        description="Refresh the intramonth stockout detection materialized view",
        group="inventory",
        callable=_run_refresh_intramonth,
        params_schema={},
    ),
    "refresh_forecast_views": JobTypeDef(
        type_id="refresh_forecast_views",
        label="Refresh Forecast Views",
        description="Refresh customer-analytics + forecast MVs concurrently (background)",
        group="forecast",
        callable=_run_refresh_forecast_views,
        params_schema={"mvs": None},
    ),
    "refresh_customer_analytics": JobTypeDef(
        type_id="refresh_customer_analytics",
        label="Recalculate Customer Analytics",
        description="Refresh the 6 customer-analytics MVs concurrently (background)",
        group="forecast",
        callable=_run_refresh_customer_analytics,
        params_schema={},
    ),
    "run_ss_simulation": JobTypeDef(
        type_id="run_ss_simulation",
        label="SS Simulation",
        description="Run Monte Carlo safety stock simulation for service level distributions",
        group="inventory",
        callable=_run_ss_simulation,
        params_schema={},
    ),
    # ── AI Tuning (tuning group) ──────────────────────────────────────────────
    "tuning_backtest": JobTypeDef(
        type_id="tuning_backtest",
        label="AI Tuning Backtest",
        description="Run LGBM backtest with tuning chat overrides and register results",
        group="tuning",
        callable=_run_tuning_backtest,
        params_schema={"run_id": 0, "session_id": "", "overrides": {}, "strategy_label": ""},
    ),
    "model_tuning_run": JobTypeDef(
        type_id="model_tuning_run",
        label="Model Tuning Experiment",
        description="Run backtest with custom hyperparameters and register results",
        group="tuning",  # overridden to tuning_{model} at submit time
        callable=_run_model_tuning_experiment,
        params_schema={
            "run_id": 0,
            "model": "",
            "config_path": "",
            "run_label": "",
        },
    ),
    "load_backtest_results": JobTypeDef(
        type_id="load_backtest_results",
        label="Load Backtest Results",
        description="Load backtest predictions into DB and refresh materialized views",
        group="tuning",  # overridden to tuning_{model} at submit time
        callable=_run_load_backtest_results,
        params_schema={"run_id": 0, "model": ""},
    ),
    "backtest_load_model": JobTypeDef(
        type_id="backtest_load_model",
        label="Load Backtest Results",
        description="Load backtest predictions into DB for a specific model",
        group="backtest_load",
        callable=_run_load_backtest_model,
        params_schema={"model_id": "", "run_id": None},
    ),
    # ── Inventory comparison (inventory group) ─────────────────────────────────
    "compare_inventory_algorithms": JobTypeDef(
        type_id="compare_inventory_algorithms",
        label="Algorithm Inventory Comparison",
        description="Compare SS/EOQ/ROP across forecast algorithms",
        group="inventory",
        callable=_run_compare_inventory_algorithms,
        params_schema={"models": None},
    ),
    # ── Platform (platform group) ─────────────────────────────────────────────
    "data_quality": JobTypeDef(
        type_id="data_quality",
        label="Data Quality Checks",
        description="Run all configured data quality checks and record results",
        group="platform",
        callable=_run_data_quality,
        params_schema={"domain": None},
    ),
    # ── Ingestion (etl group) ────────────────────────────────────────────────
    "etl_pipeline": JobTypeDef(
        type_id="etl_pipeline",
        label="Data Ingestion Pipeline",
        description="Run the data-ingestion pipeline: full reload or incremental "
                    "refresh (change-detected) across all or selected domains",
        group="etl",  # one ingestion run at a time
        callable=_run_etl_pipeline,
        params_schema={"mode": "refresh", "domains": None, "parallel": False},
    ),
    "load_domain": JobTypeDef(
        type_id="load_domain",
        label="Load Domain",
        description="Load a single domain via the unified ETL engine "
                    "(onetime / delta / file); records row metrics in job_history",
        group="etl",  # shares the single-ingestion-at-a-time group with etl_pipeline
        callable=_run_load_domain,
        params_schema={"domain": None, "mode": "delta", "slice": None,
                       "file": None, "reindex": False},
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
        # Append log entry when progress_msg is provided
        if "progress_msg" in kwargs and kwargs["progress_msg"]:
            log_entry = json.dumps({
                "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "pct": kwargs.get("progress_pct", 0),
                "msg": kwargs["progress_msg"],
            })
            sets.append("logs = COALESCE(logs, '[]'::jsonb) || %s::jsonb")
            vals.append(log_entry)
        vals.append(job_id)
        with _get_conn() as conn:
            conn.execute(f"UPDATE job_history SET {', '.join(sets)} WHERE job_id = %s", vals)

    @staticmethod
    def _db_get(job_id: str) -> dict[str, Any] | None:
        cols = ("job_id", "job_type", "job_label", "status", "params", "result",
                "error", "submitted_at", "started_at", "completed_at",
                "progress_pct", "progress_msg", "logs", "pid")
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
                "progress_pct", "progress_msg", "logs", "pid")

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
        group_override: str | None = None,
    ) -> str:
        """Submit a job for immediate background execution. Returns job_id.

        The job is dispatched to APScheduler's managed thread pool for execution.
        Per-group concurrency control ensures only one job runs per group at a time.

        Args:
            group_override: If provided, use this group instead of the job type's
                default group. Allows e.g. per-model concurrency for tuning jobs.
        """
        self._ensure_init()

        if job_type not in JOB_TYPE_REGISTRY:
            raise ValueError(f"Unknown job type: {job_type}")

        type_def = JOB_TYPE_REGISTRY[job_type]
        effective_group = group_override or type_def.group

        job_id = self._generate_id()
        job_label = label or type_def.label
        job_params = params or {}

        with self._state_lock:
            if self._is_group_busy(effective_group):
                # Queue the job instead of rejecting
                self._db_insert(
                    job_id, job_type, job_label, job_params,
                    triggered_by=triggered_by,
                    pipeline_id=pipeline_id,
                    pipeline_step=pipeline_step,
                    max_retries=max_retries,
                )
                self._db_update_status(job_id, "queued", progress_msg="Waiting for group to be free")
                queue = self._pending_queues.setdefault(effective_group, [])
                queue.append((job_id, type_def, job_params, max_retries, pipeline_id))
                logger.info("Queued job %s (%s) — group '%s' is busy", job_id, job_type, effective_group)
                return job_id

            # Track as active (inside lock to prevent race with _dispatch_next)
            self._active_jobs[job_id] = effective_group
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

        Backward-compatible no-op: submit_job() already dispatches via APScheduler.
        Retained for callers that call submit_job() then start_job_in_background().
        """
        self._ensure_init()

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
        except (KeyError, JobLookupError):
            logger.debug("Schedule %s not found in APScheduler", schedule_id)
        except Exception:
            logger.exception("Failed to remove APScheduler job %s", schedule_id)
        try:
            with _get_conn() as conn:
                res = conn.execute(
                    "DELETE FROM job_schedule WHERE schedule_id = %s", (schedule_id,))
                return res.rowcount > 0
        except psycopg.Error:
            logger.exception("Failed to delete schedule %s from DB", schedule_id)
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
            logger.warning("Failed to list schedules — table may not exist yet")
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

    def get_job_logs(self, job_id: str) -> str:
        """Get the persistent execution log for a job."""
        return get_job_log(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running job. Removes from pending queue or kills subprocess."""
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
            # Remove from pending queues (queued jobs that haven't dispatched yet)
            for group, queue in self._pending_queues.items():
                before = len(queue)
                self._pending_queues[group] = [
                    entry for entry in queue if entry[0] != job_id
                ]
                if len(self._pending_queues[group]) < before:
                    logger.info("Removed queued job %s from pending queue '%s'", job_id, group)
                    break
        # Remove from APScheduler if still pending
        try:
            self._scheduler.remove_job(job_id)
        except (KeyError, JobLookupError):
            pass  # job not in APScheduler — already running or completed
        except Exception:
            logger.exception("Failed to remove APScheduler job %s during cancel", job_id)
        # Kill the subprocess by PID as safety net
        self._kill_process(job_id)
        self._db_update_status(
            job_id, "cancelled",
            completed_at=datetime.now(timezone.utc),
            progress_msg="Cancelled by user",
        )
        return True

    @staticmethod
    def _kill_process(job_id: str) -> None:
        """Kill the subprocess process group by PID stored in job_history."""
        pid = get_job_pid(job_id)
        if not pid:
            return
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            logger.info("Sent SIGTERM to process group of PID %d for job %s", pid, job_id)
        except ProcessLookupError:
            pass  # already dead
        except OSError as exc:
            logger.warning("Failed to kill PID %d for job %s: %s", pid, job_id, exc)

    def delete_job(self, job_id: str) -> bool:
        """Delete a completed/failed/cancelled job from history."""
        return self._db_delete(job_id)

    @staticmethod
    def purge_history(
        *,
        older_than_hours: int | None = None,
        status: str | None = None,
        job_type: str | None = None,
    ) -> int:
        """Bulk-delete terminal jobs (completed/failed/cancelled).

        Filters are AND-combined; ALL filters are optional. Running and queued
        jobs are NEVER touched. Returns the number of rows deleted.
        """
        clauses = ["status IN ('completed', 'failed', 'cancelled')"]
        params: list[Any] = []
        if status:
            clauses.append("status = %s")
            params.append(status)
        if job_type:
            clauses.append("job_type = %s")
            params.append(job_type)
        if older_than_hours is not None and older_than_hours > 0:
            clauses.append("submitted_at < NOW() - (%s * INTERVAL '1 hour')")
            params.append(older_than_hours)
        sql = f"DELETE FROM job_history WHERE {' AND '.join(clauses)}"
        with _get_conn() as conn:
            result = conn.execute(sql, params)
            return int(result.rowcount or 0)

    def recover_stale_jobs(self) -> int:
        """On startup: recover running jobs (PID-aware) and re-enqueue queued jobs.

        For running jobs:
        - If PID is alive → re-adopt via a monitoring thread
        - If PID is dead or missing → mark as failed
        """
        recovered = 0

        # 1. Handle running jobs — PID-aware recovery
        try:
            with _get_conn() as conn:
                running_rows = conn.execute(
                    """SELECT job_id, job_type, pid, params, max_retries, pipeline_id
                       FROM job_history
                       WHERE status = 'running'
                       ORDER BY submitted_at ASC"""
                ).fetchall()
            for row in running_rows:
                job_id, job_type, pid, params_raw, max_retries, pipeline_id = row
                if pid and self._is_pid_alive(pid):
                    # Process is still running — re-adopt it
                    type_def = JOB_TYPE_REGISTRY.get(job_type)
                    if type_def:
                        self._readopt_job(job_id, type_def, pid)
                        logger.info("Re-adopted running job %s (PID %d) on startup", job_id, pid)
                    else:
                        self._db_update_status(
                            job_id, "failed",
                            error=f"Unknown job type '{job_type}' on restart",
                            completed_at=datetime.now(timezone.utc),
                        )
                else:
                    # PID is dead or missing — mark as failed
                    self._db_update_status(
                        job_id, "failed",
                        error="Interrupted by server restart (process not found)",
                        completed_at=datetime.now(timezone.utc),
                    )
                recovered += 1
        except Exception:
            logger.exception("Failed to recover running jobs on startup")
            # Fallback: mark all running as failed
            try:
                with _get_conn() as conn:
                    result = conn.execute(
                        """UPDATE job_history SET status = 'failed',
                                  error = 'Interrupted by server restart',
                                  completed_at = NOW()
                           WHERE status = 'running'"""
                    )
                    recovered += result.rowcount
            except Exception:
                logger.exception("Fallback recovery also failed")

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
                    self._db_update_status(
                        job_id, "failed",
                        error=f"Unknown job type '{job_type}' on restart",
                        completed_at=datetime.now(timezone.utc),
                    )
                    recovered += 1
                    continue
                params = params_raw if isinstance(params_raw, dict) else json.loads(params_raw or "{}")
                # Use per-model group for tuning jobs (matches submit-time group_override)
                effective_group = type_def.group
                if job_type == "model_tuning_run" and params.get("model"):
                    effective_group = f"tuning_{params['model']}"
                with self._state_lock:
                    queue = self._pending_queues.setdefault(effective_group, [])
                    queue.append((job_id, type_def, params, max_retries or 0, pipeline_id))
                recovered += 1
                logger.info("Re-enqueued job %s (%s) from DB on startup", job_id, job_type)

            # Dispatch first queued job for each group that has no active job
            groups_with_queued: set[str] = set()
            with self._state_lock:
                groups_with_queued = set(self._pending_queues.keys())
            for group in groups_with_queued:
                with self._state_lock:
                    busy = self._is_group_busy(group)
                if not busy:
                    self._dispatch_next(group)
        except Exception:
            logger.exception("Failed to re-enqueue queued jobs on startup")

        return recovered

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        """Check if a process with the given PID is still running."""
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # process exists but we can't signal it

    def _readopt_job(self, job_id: str, type_def: JobTypeDef, pid: int) -> None:
        """Start a monitoring thread that polls a re-adopted process until it exits.

        Since the re-adopted process is NOT a child of this API process,
        we cannot use proc.wait(). Instead we poll os.kill(pid, 0).
        """
        with self._state_lock:
            self._active_jobs[job_id] = type_def.group
            self._cancel_flags[job_id] = threading.Event()

        def _monitor():
            try:
                while self._is_pid_alive(pid):
                    cancel_event = self._cancel_flags.get(job_id)
                    if cancel_event and cancel_event.is_set():
                        self._kill_process(job_id)
                        self._db_update_status(
                            job_id, "cancelled",
                            completed_at=datetime.now(timezone.utc),
                            progress_msg="Cancelled by user (re-adopted job)",
                        )
                        return
                    time.sleep(2)
                # Process exited — check if results were written by the subprocess
                # If the subprocess wrote its own result, status may already be completed
                job = self._db_get(job_id)
                if job and job["status"] == "running":
                    # Subprocess exited but didn't update status — assume success
                    # For tuning jobs, run post-completion registration
                    if type_def.type_id == "model_tuning_run":
                        self._finalize_tuning_run(job_id)
                    self._db_update_status(
                        job_id, "completed",
                        completed_at=datetime.now(timezone.utc),
                        progress_msg="Completed (re-adopted)",
                    )
            except Exception:
                logger.exception("Monitor thread for re-adopted job %s failed", job_id)
                try:
                    self._db_update_status(
                        job_id, "failed",
                        error="Monitor thread failed",
                        completed_at=datetime.now(timezone.utc),
                    )
                except Exception:
                    pass
            finally:
                with self._state_lock:
                    self._active_jobs.pop(job_id, None)
                    self._cancel_flags.pop(job_id, None)
                self._dispatch_next(type_def.group)

        t = threading.Thread(target=_monitor, name=f"readopt-{job_id}", daemon=True)
        t.start()

    def _finalize_tuning_run(self, job_id: str) -> None:
        """Post-completion registration for re-adopted tuning jobs.

        When a tuning subprocess completes after an API restart, the normal
        callback (complete_run + register_timeframes + register_cluster_month_breakdowns)
        never fires. This method reads the backtest output and registers results.
        """
        job = self._db_get(job_id)
        if not job:
            return
        params = job.get("params") or {}
        if isinstance(params, str):
            params = json.loads(params)
        run_id = params.get("run_id")
        model = params.get("model")
        if not run_id or not model:
            return

        output_dir = MODEL_OUTPUT_DIRS.get(model)
        if not output_dir:
            return

        from common.core.paths import DATA_DIR
        meta_path = DATA_DIR / "backtest" / output_dir / "backtest_metadata.json"
        pred_path = DATA_DIR / "backtest" / output_dir / "backtest_predictions.csv"

        try:
            from common.ml.tuning_tracker import complete_run, register_timeframes, register_cluster_month_breakdowns
            complete_run(run_id, meta_path)
            register_timeframes(run_id, meta_path)
            if pred_path.exists():
                register_cluster_month_breakdowns(run_id, pred_path)
            logger.info("Finalized re-adopted tuning run %d (%s) from backtest output", run_id, model)
        except Exception:
            logger.exception("Failed to finalize re-adopted tuning run %d (%s)", run_id, model)

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
                cancel_event = self._cancel_flags.get(job_id)
                result = type_def.callable(
                    clean_params, progress_cb,
                    cancel_event=cancel_event, job_id=job_id,
                )

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
                    # Use the stored group (may differ from type_def.group if group_override was used)
                    active_group = self._active_jobs.get(job_id, type_def.group)
                    was_active = job_id in self._active_jobs
                    self._active_jobs.pop(job_id, None)
                    self._cancel_flags.pop(job_id, None)
                # Auto-dispatch next queued job (outside lock to avoid deadlock)
                if was_active:
                    self._dispatch_next(active_group)

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
