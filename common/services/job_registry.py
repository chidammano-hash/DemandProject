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
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from common.services.champion_refresh import run_governed_champion_refresh
from common.services.job_scheduler import make_scheduler, make_trigger

# ---------------------------------------------------------------------------
# Sub-module imports
# ---------------------------------------------------------------------------
from common.services.job_state import (
    MODEL_OUTPUT_DIRS,
    JobCancelledError,
    JobTypeDef,
    _clear_pid,
    _finalize_champion_results_lineage,
    _get_conn,
    _reserve_backtest_run,
    _run_archive_forecast_snapshot,
    _run_assign_policies,
    _run_backtest_chronos2_enriched,
    _run_backtest_lgbm,
    _run_backtest_mstl,
    _run_backtest_nbeats,
    _run_backtest_nhits,
    _run_champion_experiment,
    _run_champion_results_load,
    _run_champion_sweep,
    _run_classify_abc_xyz,
    _run_cleanup_forecast_staging,
    _run_cluster_pipeline,
    _run_cluster_scenario,
    _run_compare_inventory_algorithms,
    _run_compute_demand_signals,
    _run_compute_eoq,
    _run_compute_investment,
    _run_compute_replenishment_plan,
    _run_compute_safety_stock,
    _run_compute_sku_features,
    _run_compute_variability,
    _run_data_quality,
    _run_etl_pipeline,
    _run_generate_ai_insights,
    _run_generate_exceptions,
    _run_generate_production_forecast,
    _run_generate_storyboard,
    _run_inventory_backtest,
    _run_inventory_planning_pipeline,
    _run_load_backtest_model,
    _run_load_backtest_results,
    _run_load_domain,
    _run_model_tuning_experiment,
    _run_period_roll,
    _run_prepare_forecast_snapshot_contenders,
    _run_refresh_all_mvs,
    _run_refresh_customer_analytics,
    _run_refresh_forecast_snapshot_kpis,
    _run_refresh_forecast_views,
    _run_refresh_health_scores,
    _run_refresh_intramonth,
    _run_sampled_backtest,
    _run_seasonality,
    _run_ss_simulation,
    _run_train_production_model,
    _run_tune_stale_clusters,
    _run_tuning_backtest,
    _serialize_job_row,
    _store_attempt_result,
    _validate_attempt_result,
    bind_job_attempt,
    get_job_log,
    get_job_pid,
    get_job_process_identity,
    load_attempt_result,
    process_identity_matches,
    reconcile_backtest_run,
    reconcile_cluster_pipeline_experiment,
    reset_job_attempt,
    verify_backtest_artifact_identity,
    verify_backtest_tracking_identity,
    verify_cluster_pipeline_completion,
)

try:
    from apscheduler.jobstores.base import JobLookupError
except ImportError:
    JobLookupError = KeyError  # type: ignore[assignment,misc]

import psycopg

logger = logging.getLogger(__name__)

_BACKTEST_JOB_MODELS = {
    "backtest_lgbm": "lgbm_cluster",
    "backtest_chronos2_enriched": "chronos2_enriched",
    "backtest_mstl": "mstl",
    "backtest_nhits": "nhits",
    "backtest_nbeats": "nbeats",
}
_PROCESS_TERMINATION_TIMEOUT = 5.0
_PROCESS_TERMINATION_POLL_INTERVAL = 0.1
_RECOVERY_LEASE_SECONDS = 120


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
    # -- LightGBM --
    "backtest_lgbm": JobTypeDef(
        type_id="backtest_lgbm",
        label="LightGBM Backtest",
        description="Run LightGBM gradient boosting backtest",
        group="backtest",
        callable=_run_backtest_lgbm,
        params_schema={},
    ),
    # -- Foundation models --
    "backtest_chronos2_enriched": JobTypeDef(
        type_id="backtest_chronos2_enriched",
        label="Chronos 2 Enriched Backtest",
        description="Run Chronos 2E with 30 covariates (~6h)",
        group="backtest",
        callable=_run_backtest_chronos2_enriched,
        params_schema={},
    ),
    # -- Statistical model --
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
        label="Governed Champion Refresh",
        description=(
            "Create a five-model champion experiment and atomically promote its audited results"
        ),
        group="champion",
        callable=run_governed_champion_refresh,
        params_schema={},
    ),
    "governed_champion_refresh": JobTypeDef(
        type_id="governed_champion_refresh",
        label="Governed Champion Refresh",
        description=(
            "Create a five-model champion experiment and atomically promote its audited results"
        ),
        group="champion",
        callable=run_governed_champion_refresh,
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
        label="Train Production Models",
        description="Final-refit LightGBM, N-HiTS, and N-BEATS on full history",
        group="forecast",
        callable=_run_train_production_model,
        params_schema={"model_id": None, "all_models": True},
    ),
    "generate_production_forecast": JobTypeDef(
        type_id="generate_production_forecast",
        label="Production Forecast",
        description="Generate future-period production forecasts using champion ML models",
        group="forecast",
        callable=_run_generate_production_forecast,
        params_schema={
            "horizon": None,
            "model_id": None,
            "run_id": None,
            "generation_purpose": "release_candidate",
            "confidence_intervals": None,
        },
    ),
    "prepare_forecast_snapshot_contenders": JobTypeDef(
        type_id="prepare_forecast_snapshot_contenders",
        label="Prepare Forecast Snapshot Contenders",
        description="Freeze the top three backtest-ranked contenders and generate six snapshot lags",
        group="forecast",
        callable=_run_prepare_forecast_snapshot_contenders,
        params_schema={"record_month": None, "dry_run": False, "from_existing_staging": False},
    ),
    "archive_forecast_snapshot": JobTypeDef(
        type_id="archive_forecast_snapshot",
        label="Archive Forecast Snapshot",
        description="Archive the promoted champion plus three frozen contenders at lags 0 through 5",
        group="forecast",
        callable=_run_archive_forecast_snapshot,
        params_schema={"record_month": None, "dry_run": False, "overwrite": False},
    ),
    "refresh_forecast_snapshot_kpis": JobTypeDef(
        type_id="refresh_forecast_snapshot_kpis",
        label="Calculate Snapshot KPIs",
        description="Score newly closed live snapshot lags after monthly actuals load",
        group="forecast",
        callable=_run_refresh_forecast_snapshot_kpis,
        params_schema={},
    ),
    "period_roll": JobTypeDef(
        type_id="period_roll",
        label="Period Roll",
        description="Score the prior archived month, archive the current month, and clean reconciled staging",
        group="forecast",
        callable=_run_period_roll,
        params_schema={"record_month": None},
    ),
    "cleanup_forecast_staging": JobTypeDef(
        type_id="cleanup_forecast_staging",
        label="Clean Forecast Staging",
        description="Delete staging only after the bounded archive is reconciled",
        group="forecast",
        callable=_run_cleanup_forecast_staging,
        params_schema={"generation": None, "dry_run": True},
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
    "refresh_all_mvs": JobTypeDef(
        type_id="refresh_all_mvs",
        label="Refresh All Materialized Views",
        description="Refresh every materialized view in dependency order "
        "(staleness safety net; set skip_heavy to exclude "
        "mv_intramonth_stockout)",
        group="platform",
        callable=_run_refresh_all_mvs,
        params_schema={"skip_heavy": True},
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
    "tune_stale_clusters": JobTypeDef(
        type_id="tune_stale_clusters",
        label="Re-tune Stale Cluster Profiles",
        description="Re-tune per-cluster hyperparameters flagged stale by a "
        "clustering promotion (tune_cluster_hyperparams.py --stale-only)",
        group="tuning",
        callable=_run_tune_stale_clusters,
        params_schema={"model": "lgbm", "trials": None},
    ),
    "sampled_backtest": JobTypeDef(
        type_id="sampled_backtest",
        label="Sampled Backtest",
        description="Run a sampled-SKU LGBM backtest for fast hyperparameter iteration",
        group="tuning_lgbm",  # shares the per-model group with model_tuning_run(lgbm)
        callable=_run_sampled_backtest,
        params_schema={"run_id": 0, "sku_file": "", "param_overrides": {}},
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
        params_schema={
            "domain": None,
            "mode": "delta",
            "slice": None,
            "file": None,
            "reindex": False,
        },
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
            self._state_lock = (
                threading.Lock()
            )  # guards _active_jobs, _pending_queues, _cancel_flags
            self._active_jobs: dict[str, str] = {}  # job_id -> group
            self._cancel_flags: dict[str, threading.Event] = {}
            self._pending_queues: dict[
                str,
                list[tuple[str, Any, dict, int, int, str | None]],
            ] = {}  # group -> [(job_id, type_def, params, max_retries, retry_count, pipeline_id)]
            self._worker_id = f"job-manager-{os.getpid()}-{uuid.uuid4().hex}"
            self._initialized = True
            logger.info(
                "JobManager initialised: APScheduler BackgroundScheduler started (4 workers)"
            )

            # Recover stale jobs on startup
            recovered = self.recover_stale_jobs()
            if recovered:
                logger.info("Recovered %d stale jobs on startup", recovered)

            # Re-register persisted recurring schedules. They previously lived
            # only in APScheduler memory, so an API restart silently killed
            # them while list_schedules() kept returning the DB rows.
            restored = self.restore_schedules()
            if restored:
                logger.info("Restored %d recurring schedule(s) from job_schedule", restored)

            # Guarantee config-declared default schedules (e.g. the nightly
            # refresh_all_mvs staleness safety net) exist.
            self.ensure_default_schedules()

    def start(self) -> None:
        """Initialize recovery, queues, recurring schedules, and APScheduler."""
        self._ensure_init()

    def shutdown(self) -> None:
        """Stop this process's scheduler without waiting for long-running jobs."""
        if not self._initialized:
            return
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        self._initialized = False

    # ---- helpers ----

    @staticmethod
    def _generate_id() -> str:
        return f"job_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def _is_group_busy(self, group: str) -> bool:
        """Check if a group has an active job. Must be called under _state_lock."""
        return any(g == group for g in self._active_jobs.values())

    @staticmethod
    def _materialize_durable_params(
        job_id: str,
        type_def: JobTypeDef,
        params: dict[str, Any],
    ) -> None:
        """Persist identities needed to resume forecast work after an API restart."""
        model_id = _BACKTEST_JOB_MODELS.get(type_def.type_id)
        if model_id and params.get("backtest_run_id") is None:
            tracking_model = str(params.get("model_id") or model_id)
            if tracking_model == "lgbm":
                tracking_model = "lgbm_cluster"
            params["backtest_run_id"] = _reserve_backtest_run(
                tracking_model,
                job_id,
                get_conn=_get_conn,
            )
        if type_def.type_id == "generate_production_forecast" and not params.get("run_id"):
            params["run_id"] = str(uuid.uuid4())

    # ---- DB operations ----

    @staticmethod
    def _db_insert(
        job_id: str,
        job_type: str,
        label: str,
        params: dict,
        triggered_by: str = "manual",
        pipeline_id: str | None = None,
        pipeline_step: int | None = None,
        max_retries: int = 0,
        execution_group: str | None = None,
    ) -> None:
        with _get_conn() as conn:
            conn.execute(
                """INSERT INTO job_history
                   (job_id, job_type, job_label, status, params, submitted_at,
                    triggered_by, pipeline_id, pipeline_step, max_retries,
                    execution_group)
                   VALUES (%s, %s, %s, 'queued', %s, NOW(), %s, %s, %s, %s,
                           %s)""",
                (
                    job_id,
                    job_type,
                    label,
                    json.dumps(params),
                    triggered_by,
                    pipeline_id,
                    pipeline_step,
                    max_retries,
                    execution_group,
                ),
            )

    @staticmethod
    def _db_update_status(
        job_id: str,
        status: str,
        *,
        expected_status: str | None = None,
        expected_attempt_token: str | None = None,
        **kwargs: Any,
    ) -> bool:
        sets = ["status = %s"]
        vals: list[Any] = [status]
        for col in (
            "started_at",
            "completed_at",
            "progress_pct",
            "progress_msg",
            "error",
            "retry_count",
            "attempt_token",
            "attempt_result",
            "attempt_failure_recorded",
            "recovery_quarantine_reason",
            "recovery_lease_owner",
            "recovery_lease_until",
        ):
            if col in kwargs:
                placeholder = "%s::jsonb" if col == "attempt_result" else "%s"
                sets.append(f"{col} = {placeholder}")
                value = kwargs[col]
                if col == "attempt_result" and value is not None:
                    value = json.dumps(value)
                vals.append(value)
        if "result" in kwargs:
            sets.append("result = %s")
            vals.append(json.dumps(kwargs["result"]) if kwargs["result"] is not None else None)
        if "attempt_callable_completion" in kwargs:
            completion = kwargs["attempt_callable_completion"]
            if completion is None:
                sets.append(
                    "params = COALESCE(params, '{}'::jsonb) - '__attempt_callable_completion'"
                )
            else:
                sets.append(
                    "params = jsonb_set(COALESCE(params, '{}'::jsonb), "
                    "'{__attempt_callable_completion}', %s::jsonb, true)"
                )
                vals.append(json.dumps(completion))
        # Append log entry when progress_msg is provided
        if kwargs.get("progress_msg"):
            log_entry = json.dumps(
                {
                    "ts": datetime.now(UTC).strftime("%H:%M:%S"),
                    "pct": kwargs.get("progress_pct", 0),
                    "msg": kwargs["progress_msg"],
                }
            )
            sets.append("logs = COALESCE(logs, '[]'::jsonb) || %s::jsonb")
            vals.append(log_entry)
        where_sql = "job_id = %s"
        vals.append(job_id)
        if expected_status is not None:
            where_sql += " AND status = %s"
            vals.append(expected_status)
        if expected_attempt_token is not None:
            where_sql += " AND attempt_token = %s"
            vals.append(expected_attempt_token)
        with _get_conn() as conn:
            result = conn.execute(
                f"UPDATE job_history SET {', '.join(sets)} WHERE {where_sql}",
                vals,
            )
        return int(result.rowcount or 0) > 0

    @staticmethod
    def _db_get(job_id: str) -> dict[str, Any] | None:
        cols = (
            "job_id",
            "job_type",
            "job_label",
            "status",
            "params",
            "result",
            "error",
            "submitted_at",
            "started_at",
            "completed_at",
            "progress_pct",
            "progress_msg",
            "logs",
            "pid",
            "pipeline_id",
            "pipeline_step",
            "retry_count",
            "max_retries",
            "execution_group",
            "attempt_token",
            "attempt_result",
            "attempt_failure_recorded",
            "recovery_quarantine_reason",
        )
        with _get_conn() as conn:
            row = conn.execute(
                f"SELECT {', '.join(cols)} FROM job_history WHERE job_id = %s",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return _serialize_job_row(cols, row)

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
        cols = (
            "job_id",
            "job_type",
            "job_label",
            "status",
            "params",
            "result",
            "error",
            "submitted_at",
            "started_at",
            "completed_at",
            "progress_pct",
            "progress_msg",
            "logs",
            "pid",
            "pipeline_id",
            "pipeline_step",
            "retry_count",
            "max_retries",
            "execution_group",
            "attempt_token",
            "attempt_result",
            "attempt_failure_recorded",
            "recovery_quarantine_reason",
        )

        with _get_conn() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM job_history {where_sql}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"""SELECT {", ".join(cols)}
                    FROM job_history {where_sql}
                    ORDER BY submitted_at DESC LIMIT %s OFFSET %s""",
                [*params, limit, offset],
            ).fetchall()

        jobs = [_serialize_job_row(cols, r) for r in rows]
        return jobs, int(total)

    @staticmethod
    def _db_delete(job_id: str) -> tuple[str | None, str | None] | None:
        with _get_conn() as conn:
            row = conn.execute(
                """DELETE FROM job_history
                   WHERE job_id = %s
                     AND status IN ('completed', 'failed', 'cancelled')
                   RETURNING execution_group, recovery_quarantine_reason""",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return (
            str(row[0]) if row[0] else None,
            str(row[1]) if row[1] else None,
        )

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
        # Scheduler callbacks may outlive their submitter. Keep an owned copy so
        # later caller mutations cannot diverge execution from the params that
        # were persisted in job_history.
        job_params = deepcopy(params) if params is not None else {}
        self._materialize_durable_params(
            job_id,
            type_def,
            job_params,
        )

        with self._state_lock:
            if self._is_group_busy(effective_group):
                # Queue the job instead of rejecting
                self._db_insert(
                    job_id,
                    job_type,
                    job_label,
                    job_params,
                    triggered_by=triggered_by,
                    pipeline_id=pipeline_id,
                    pipeline_step=pipeline_step,
                    max_retries=max_retries,
                    execution_group=effective_group,
                )
                self._db_update_status(
                    job_id, "queued", progress_msg="Waiting for group to be free"
                )
                queue = self._pending_queues.setdefault(effective_group, [])
                queue.append((job_id, type_def, job_params, max_retries, 0, pipeline_id))
                logger.info(
                    "Queued job %s (%s) — group '%s' is busy", job_id, job_type, effective_group
                )
                return job_id

            # Track as active (inside lock to prevent race with _dispatch_next)
            self._active_jobs[job_id] = effective_group
            self._cancel_flags[job_id] = threading.Event()

        # DB insert and APScheduler dispatch outside lock (no contention on I/O)
        self._db_insert(
            job_id,
            job_type,
            job_label,
            job_params,
            triggered_by=triggered_by,
            pipeline_id=pipeline_id,
            pipeline_step=pipeline_step,
            max_retries=max_retries,
            execution_group=effective_group,
        )

        self._scheduler.add_job(
            self._execute_job,
            args=[
                job_id,
                type_def,
                job_params,
                max_retries,
                pipeline_id,
                0,
                effective_group,
            ],
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
        schedule_id: str | None = None,
    ) -> str:
        """Schedule a recurring job via cron expression or interval.

        Returns a schedule_id that can be used to remove the schedule.
        Pass an explicit ``schedule_id`` for deterministic schedules (config
        defaults) so repeated calls upsert instead of duplicating.
        """
        self._ensure_init()

        if job_type not in JOB_TYPE_REGISTRY:
            raise ValueError(f"Unknown job type: {job_type}")
        if not cron and not interval_minutes:
            raise ValueError("Must specify either cron or interval_minutes")

        type_def = JOB_TYPE_REGISTRY[job_type]
        schedule_id = schedule_id or f"sched_{uuid.uuid4().hex[:8]}"
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
                    (
                        schedule_id,
                        job_type,
                        job_label,
                        cron,
                        interval_minutes,
                        json.dumps(job_params),
                        next_run_at,
                    ),
                )
        except Exception:
            logger.warning("job_schedule table may not exist yet; schedule only in memory")

        logger.info(
            "Scheduled recurring %s as %s (cron=%s, interval=%s min)",
            job_type,
            schedule_id,
            cron,
            interval_minutes,
        )
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
                    "DELETE FROM job_schedule WHERE schedule_id = %s", (schedule_id,)
                )
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
                schedules.append(
                    {
                        "schedule_id": r[0],
                        "job_type": r[1],
                        "job_label": r[2],
                        "cron_expr": r[3],
                        "interval_min": r[4],
                        "params": r[5] if isinstance(r[5], dict) else json.loads(r[5] or "{}"),
                        "enabled": r[6],
                        "created_at": r[7].isoformat() if r[7] else None,
                        "last_run_at": r[8].isoformat() if r[8] else None,
                        "next_run_at": r[9].isoformat() if r[9] else None,
                        "run_count": r[10] or 0,
                    }
                )
            return schedules
        except Exception:
            logger.warning("Failed to list schedules — table may not exist yet")
            return []

    def restore_schedules(self) -> int:
        """Re-register enabled ``job_schedule`` rows into APScheduler.

        Called from ``_ensure_init`` on startup. Without this, recurring
        schedules created via ``POST /jobs/schedule`` stopped firing after an
        API restart even though ``list_schedules()`` still returned them.
        """
        try:
            with _get_conn() as conn:
                rows = conn.execute(
                    """SELECT schedule_id, job_type, job_label, cron_expr, interval_min, params
                       FROM job_schedule WHERE enabled = TRUE"""
                ).fetchall()
        except psycopg.Error:
            logger.warning("job_schedule table unavailable — no schedules restored")
            return 0

        restored = 0
        for row in rows:
            try:
                schedule_id, job_type, job_label, cron_expr, interval_min, params_raw = row
                if job_type not in JOB_TYPE_REGISTRY:
                    logger.warning(
                        "Skipping schedule %s: unknown job type %s", schedule_id, job_type
                    )
                    continue
                params = (
                    params_raw if isinstance(params_raw, dict) else json.loads(params_raw or "{}")
                )
                self._scheduler.add_job(
                    self._scheduled_execution,
                    trigger=make_trigger(cron_expr, interval_min),
                    args=[job_type, params, job_label],
                    id=schedule_id,
                    name=job_label,
                    replace_existing=True,
                )
                restored += 1
            except (ValueError, TypeError, KeyError):
                logger.exception("Failed to restore schedule row %r", row)
        return restored

    def ensure_default_schedules(self) -> None:
        """Create config-declared default schedules that are not registered yet.

        Reads ``default_schedules`` from ``config/platform/jobs_config.yaml``.
        Each entry gets the deterministic id ``sched_default_<name>``, so this
        is idempotent across restarts. Disable an entry in the config to stop
        it — deleting the schedule row alone lasts only until the next boot.
        """
        try:
            from common.core.utils import load_config

            cfg = load_config("jobs_config.yaml") or {}
        except (OSError, ValueError):
            logger.exception("jobs_config.yaml unreadable — skipping default schedules")
            return

        for entry in cfg.get("default_schedules") or []:
            if not isinstance(entry, dict) or not entry.get("enabled", True):
                continue
            job_type = entry.get("job_type")
            name = entry.get("name") or job_type
            if job_type not in JOB_TYPE_REGISTRY:
                logger.warning("Skipping default schedule %r: unknown job type %r", name, job_type)
                continue
            schedule_id = f"sched_default_{name}"
            if self._scheduler.get_job(schedule_id) is not None:
                continue  # already restored from job_schedule
            try:
                self.schedule_recurring(
                    job_type,
                    params=entry.get("params") or {},
                    label=entry.get("label"),
                    cron=entry.get("cron"),
                    interval_minutes=entry.get("interval_minutes"),
                    schedule_id=schedule_id,
                )
                logger.info("Registered default schedule %s (%s)", schedule_id, job_type)
            except (ValueError, TypeError):
                logger.exception("Failed to register default schedule %r", name)

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
        if not steps:
            raise ValueError("Pipeline must have at least one step")

        # Validate the complete chain before submitting step one. Otherwise an
        # invalid later step strands a partially executed pipeline. The deep
        # copy also keeps internal pipeline metadata out of the caller's dicts.
        pipeline_steps = deepcopy(steps)
        for step_number, step in enumerate(pipeline_steps, start=1):
            if not isinstance(step, dict):
                raise ValueError(f"Pipeline step {step_number} must be an object")
            step_job_type = step.get("job_type")
            if not isinstance(step_job_type, str) or step_job_type not in JOB_TYPE_REGISTRY:
                raise ValueError(
                    f"Unknown job type at pipeline step {step_number}: {step_job_type}"
                )
            step_params = step.get("params")
            if step_params is None:
                step["params"] = {}
            elif not isinstance(step_params, dict):
                raise ValueError(f"Pipeline step {step_number} params must be an object")

        pipeline_id = f"pipe_{uuid.uuid4().hex[:8]}"
        first = pipeline_steps[0]
        job_type = first["job_type"]
        params = first["params"]
        step_label = first.get("label", JOB_TYPE_REGISTRY[job_type].label)

        # Store remaining steps in the first job's params
        params["__pipeline_remaining"] = pipeline_steps[1:]
        params["__pipeline_label"] = label
        params["__pipeline_step"] = 1
        params["__pipeline_total_steps"] = len(pipeline_steps)

        self.submit_job(
            job_type=job_type,
            params=params,
            label=f"[{label} 1/{len(pipeline_steps)}] {step_label}",
            triggered_by=triggered_by,
            pipeline_id=pipeline_id,
            pipeline_step=1,
        )

        logger.info("Submitted pipeline %s with %d steps", pipeline_id, len(pipeline_steps))
        return pipeline_id

    def submit_named_pipeline(
        self,
        steps: list[dict[str, Any]],
        label: str,
        triggered_by: str = "manual",
    ) -> tuple[str, bool]:
        """Return the active named pipeline or atomically create one.

        A transaction-scoped advisory lock serializes submissions for one
        preset name across API workers. A completed intermediate step still
        counts as active because restart recovery may be between persisting
        that completion and creating its successor.
        """
        self._ensure_init()
        with _get_conn() as conn:
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"named-pipeline:{label}",),
            )
            row = conn.execute(
                """
                WITH matching AS (
                    SELECT pipeline_id,
                           status,
                           pipeline_step,
                           params,
                           submitted_at,
                           completed_at,
                           ROW_NUMBER() OVER (
                               PARTITION BY pipeline_id
                               ORDER BY pipeline_step DESC, submitted_at DESC
                           ) AS latest_in_pipeline
                    FROM job_history
                    WHERE pipeline_id IS NOT NULL
                      AND params ->> '__pipeline_label' = %s
                )
                SELECT pipeline_id
                FROM matching
                WHERE latest_in_pipeline = 1
                  AND (
                      status IN ('queued', 'running')
                      OR (
                          status = 'completed'
                          AND COALESCE(pipeline_step, 0) < COALESCE(
                              (params ->> '__pipeline_total_steps')::integer,
                              0
                          )
                          AND completed_at > NOW() - INTERVAL '5 minutes'
                      )
                  )
                ORDER BY submitted_at DESC
                LIMIT 1
                """,
                (label,),
            ).fetchone()
            if row is not None:
                return str(row[0]), False

            pipeline_id = self.submit_pipeline(
                steps=steps,
                label=label,
                triggered_by=triggered_by,
            )
            return pipeline_id, True

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
            # Remove from pending queues (queued jobs that haven't dispatched yet)
            for group, queue in self._pending_queues.items():
                before = len(queue)
                self._pending_queues[group] = [entry for entry in queue if entry[0] != job_id]
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
        # A running job becomes terminal only after its recorded subprocess is
        # verified dead. Jobs without a stored PID are cooperatively cancelled
        # by the worker, which owns the final durable status transition.
        terminated = self._kill_process(job_id)
        if not terminated:
            logger.error("Cancellation could not verify process exit for job %s", job_id)
            return False
        if job["status"] == "running" and not job.get("pid"):
            logger.info("Cancellation requested for in-process job %s", job_id)
            return True
        type_def = JOB_TYPE_REGISTRY.get(str(job.get("job_type") or ""))
        self._reconcile_job_terminal_state(
            job_id,
            type_def,
            "cancelled",
        )
        cancelled = self._db_update_status(
            job_id,
            "cancelled",
            expected_status=job["status"],
            expected_attempt_token=job.get("attempt_token"),
            completed_at=datetime.now(UTC),
            progress_msg="Cancelled by user",
        )
        if job["status"] == "queued" and cancelled:
            with self._state_lock:
                active_group = self._active_jobs.pop(job_id, None)
                self._cancel_flags.pop(job_id, None)
            if active_group:
                self._dispatch_next(active_group)
        return True

    @staticmethod
    def _kill_process(job_id: str) -> bool:
        """Terminate a subprocess group and confirm its recorded PID exited."""
        pid = get_job_pid(job_id)
        if not pid:
            return True
        expected_identity = get_job_process_identity(job_id)
        identity_match = process_identity_matches(pid, expected_identity)
        if identity_match is False:
            logger.warning(
                "PID %d for job %s was reused; treating the recorded child as exited",
                pid,
                job_id,
            )
            return True
        if identity_match is None:
            logger.error(
                "Refusing to signal unverifiable PID %d for job %s",
                pid,
                job_id,
            )
            return False
        try:
            process_group = os.getpgid(pid)
            os.killpg(process_group, signal.SIGTERM)
            logger.info("Sent SIGTERM to process group of PID %d for job %s", pid, job_id)
        except ProcessLookupError:
            return True
        except OSError as exc:
            logger.warning("Failed to kill PID %d for job %s: %s", pid, job_id, exc)
            return False

        deadline = time.monotonic() + _PROCESS_TERMINATION_TIMEOUT
        while time.monotonic() < deadline:
            if not JobManager._is_pid_alive(pid):
                return True
            identity_match = process_identity_matches(pid, expected_identity)
            if identity_match is False:
                return True
            if identity_match is None:
                logger.error(
                    "Stopped cancellation after PID %d identity became unverifiable",
                    pid,
                )
                return False
            time.sleep(_PROCESS_TERMINATION_POLL_INTERVAL)

        if process_identity_matches(pid, expected_identity) is not True:
            logger.error("Refusing to escalate unverifiable PID %d", pid)
            return False
        try:
            os.killpg(process_group, signal.SIGKILL)
            logger.warning("Escalated to SIGKILL for PID %d (job %s)", pid, job_id)
        except ProcessLookupError:
            return True
        except OSError as exc:
            logger.warning("Failed to SIGKILL PID %d for job %s: %s", pid, job_id, exc)
            return False

        deadline = time.monotonic() + _PROCESS_TERMINATION_TIMEOUT
        while time.monotonic() < deadline:
            if not JobManager._is_pid_alive(pid):
                return True
            identity_match = process_identity_matches(pid, expected_identity)
            if identity_match is False:
                return True
            if identity_match is None:
                return False
            time.sleep(_PROCESS_TERMINATION_POLL_INTERVAL)
        return (
            not JobManager._is_pid_alive(pid)
            or process_identity_matches(pid, expected_identity) is False
        )

    def delete_job(self, job_id: str) -> bool:
        """Delete one terminal job and release an acknowledged quarantine."""
        deleted = self._db_delete(job_id)
        if deleted is None:
            return False
        execution_group, quarantine_reason = deleted
        if quarantine_reason is not None:
            self._release_group(f"quarantine:{job_id}")
            if execution_group is not None:
                self._dispatch_next(execution_group)
        return True

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
        clauses = [
            "status IN ('completed', 'failed', 'cancelled')",
            "recovery_quarantine_reason IS NULL",
        ]
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

    def _recovery_owner(self) -> str:
        """Return the stable identity used for this manager's DB leases."""
        owner = getattr(self, "_worker_id", None)
        if isinstance(owner, str) and owner:
            return owner
        owner = f"job-manager-{os.getpid()}-{uuid.uuid4().hex}"
        self._worker_id = owner
        return owner

    @staticmethod
    def _persisted_execution_group(
        type_def: JobTypeDef | None,
        execution_group: object,
    ) -> str | None:
        if isinstance(execution_group, str) and execution_group:
            return execution_group
        return type_def.group if type_def is not None else None

    def _hydrate_group(self, job_id: str, execution_group: str) -> None:
        """Block one exact group before any queued rows are considered."""
        with self._state_lock:
            self._active_jobs[job_id] = execution_group
            self._cancel_flags.setdefault(job_id, threading.Event())

    def _release_group(self, job_id: str) -> None:
        with self._state_lock:
            self._active_jobs.pop(job_id, None)
            self._cancel_flags.pop(job_id, None)

    @staticmethod
    def _reconcile_job_terminal_state(
        job_id: str,
        type_def: JobTypeDef | None,
        terminal_status: str,
    ) -> None:
        """Reconcile domain tracking rows before the job becomes terminal."""
        if type_def is None:
            return
        if type_def.type_id == "cluster_pipeline":
            reconcile_cluster_pipeline_experiment(job_id, terminal_status)
            return
        if type_def.type_id in {
            "backtest_lgbm",
            "backtest_chronos2_enriched",
            "backtest_mstl",
            "backtest_nhits",
            "backtest_nbeats",
        }:
            reconcile_backtest_run(job_id, terminal_status)

    def _quarantine_recovered_job(
        self,
        job_id: str,
        reason: str,
        *,
        attempt_token: str | None,
        type_def: JobTypeDef | None = None,
    ) -> None:
        """Fail closed while leaving the persisted group blocked for review."""
        self._reconcile_job_terminal_state(job_id, type_def, "failed")
        updated = self._db_update_status(
            job_id,
            "failed",
            expected_status="running",
            expected_attempt_token=attempt_token,
            completed_at=datetime.now(UTC),
            error=f"Recovery quarantined: {reason}",
            progress_msg="Recovery quarantined",
            recovery_quarantine_reason=reason,
        )
        if updated:
            logger.error("Quarantined recovered job %s: %s", job_id, reason)

    def _lease_queued_jobs(self) -> list[tuple[Any, ...]]:
        """Atomically lease queued rows so startup workers cannot duplicate them."""
        owner = self._recovery_owner()
        with _get_conn() as conn:
            return conn.execute(
                """WITH lease_candidates AS (
                       SELECT queued.job_id
                       FROM job_history queued
                       WHERE queued.status = 'queued'
                         AND (
                           queued.recovery_lease_until IS NULL
                           OR queued.recovery_lease_until < NOW()
                         )
                         AND NOT EXISTS (
                           SELECT 1
                           FROM job_history blocker
                           WHERE blocker.execution_group = queued.execution_group
                             AND (
                               blocker.status = 'running'
                               OR blocker.recovery_quarantine_reason IS NOT NULL
                             )
                         )
                       ORDER BY queued.submitted_at ASC
                       FOR UPDATE SKIP LOCKED
                   )
                   UPDATE job_history queued
                   SET recovery_lease_owner = %s,
                       recovery_lease_until = NOW() + (%s * INTERVAL '1 second')
                   FROM lease_candidates candidate
                   WHERE queued.job_id = candidate.job_id
                   RETURNING queued.job_id, queued.job_type, queued.params,
                             queued.max_retries, queued.retry_count,
                             queued.pipeline_id, queued.execution_group""",
                (owner, _RECOVERY_LEASE_SECONDS),
            ).fetchall()

    def _lease_running_jobs(self) -> list[tuple[Any, ...]]:
        """Lease running rows so only one API worker performs finalization."""
        owner = self._recovery_owner()
        with _get_conn() as conn:
            return conn.execute(
                """WITH lease_candidates AS (
                       SELECT running.job_id
                       FROM job_history running
                       WHERE running.status = 'running'
                         AND (
                           running.recovery_lease_until IS NULL
                           OR running.recovery_lease_until < NOW()
                         )
                       ORDER BY running.submitted_at ASC
                       FOR UPDATE SKIP LOCKED
                   )
                   UPDATE job_history running
                   SET recovery_lease_owner = %s,
                       recovery_lease_until = NOW() + (%s * INTERVAL '1 second')
                   FROM lease_candidates candidate
                   WHERE running.job_id = candidate.job_id
                   RETURNING running.job_id, running.job_type, running.pid,
                             running.params, running.max_retries,
                             running.pipeline_id, running.retry_count,
                             running.execution_group, running.attempt_token,
                             running.attempt_result,
                             running.attempt_failure_recorded""",
                (owner, _RECOVERY_LEASE_SECONDS),
            ).fetchall()

    def _renew_recovery_lease(
        self,
        job_id: str,
        attempt_token: str | None,
    ) -> bool:
        """Extend this manager's exact running-attempt recovery lease."""
        with _get_conn() as conn:
            result = conn.execute(
                """UPDATE job_history
                   SET recovery_lease_until =
                       NOW() + (%s * INTERVAL '1 second')
                   WHERE job_id = %s
                     AND status = 'running'
                     AND recovery_lease_owner = %s
                     AND attempt_token = %s""",
                (
                    _RECOVERY_LEASE_SECONDS,
                    job_id,
                    self._recovery_owner(),
                    attempt_token,
                ),
            )
        return int(result.rowcount or 0) == 1

    @staticmethod
    def _exact_attempt_result(
        job_id: str,
        attempt_token: str | None,
        params: dict[str, Any],
        persisted_result: object,
    ) -> dict[str, Any] | None:
        expected_digest = params.get("__attempt_command_digest")
        if not isinstance(expected_digest, str):
            return None
        if attempt_token:
            persisted = _validate_attempt_result(
                persisted_result,
                attempt_token,
                expected_digest,
            )
            if persisted is not None:
                return persisted
        return load_attempt_result(
            job_id,
            attempt_token,
            expected_command_digest=expected_digest,
        )

    def recover_stale_jobs(self) -> int:
        """Lease and reconcile durable jobs on startup.

        For running jobs:
        - hydrate every exact persisted execution group before queued dispatch;
        - re-adopt only a live PID with the persisted OS identity;
        - reconcile exit only from an exact token + command wrapper result;
        - quarantine ambiguous attempts instead of inferring success.

        Jobs previously marked failed specifically because of a server restart
        are also leased and re-queued. Queued and running leases prevent two API
        workers from dispatching or finalizing the same durable row.
        """
        recovered = 0

        # 1. Hydrate every durable quarantine before inspecting queues. A
        # quarantined attempt deliberately blocks its exact execution group
        # until an operator resolves or deletes the failed history row.
        try:
            with _get_conn() as conn:
                quarantine_rows = conn.execute(
                    """SELECT job_id, execution_group
                       FROM job_history
                       WHERE recovery_quarantine_reason IS NOT NULL"""
                ).fetchall()
            for job_id, execution_group in quarantine_rows:
                if isinstance(execution_group, str) and execution_group:
                    self._hydrate_group(f"quarantine:{job_id}", execution_group)
            with _get_conn() as conn:
                active_group_rows = conn.execute(
                    """SELECT job_id, job_type, execution_group
                       FROM job_history
                       WHERE status = 'running'"""
                ).fetchall()
            for job_id, job_type, persisted_group in active_group_rows:
                group = self._persisted_execution_group(
                    JOB_TYPE_REGISTRY.get(job_type),
                    persisted_group,
                )
                if group is not None:
                    self._hydrate_group(str(job_id), group)
        except (psycopg.Error, TypeError, ValueError):
            logger.exception("Failed to hydrate active and quarantined job groups")

        # 2. Hydrate every running group before any queued lease is acquired,
        # then re-adopt only exact process identities. A missing/reused PID is
        # reconciled exclusively from the wrapper's exact-token exit record.
        try:
            running_rows = self._lease_running_jobs()
            for row in running_rows:
                (
                    job_id,
                    job_type,
                    pid,
                    params_raw,
                    max_retries,
                    pipeline_id,
                    retry_count,
                    persisted_group,
                    attempt_token,
                    attempt_result,
                    attempt_failure_recorded,
                ) = row
                params = (
                    params_raw if isinstance(params_raw, dict) else json.loads(params_raw or "{}")
                )
                type_def = JOB_TYPE_REGISTRY.get(job_type)
                execution_group = self._persisted_execution_group(
                    type_def,
                    persisted_group,
                )
                if execution_group is None:
                    self._db_update_status(
                        job_id,
                        "failed",
                        expected_status="running",
                        expected_attempt_token=attempt_token,
                        error=f"Unknown job type '{job_type}' on restart",
                        completed_at=datetime.now(UTC),
                        recovery_quarantine_reason="Missing execution group",
                    )
                    recovered += 1
                    continue
                self._hydrate_group(str(job_id), execution_group)
                if type_def is None:
                    self._quarantine_recovered_job(
                        str(job_id),
                        f"Unknown job type '{job_type}'",
                        attempt_token=attempt_token,
                    )
                    recovered += 1
                    continue

                if pid and self._is_pid_alive(pid):
                    expected_identity = params.get("__process_identity")
                    identity_match = process_identity_matches(
                        int(pid),
                        expected_identity if isinstance(expected_identity, dict) else None,
                    )
                    if identity_match is True:
                        self._readopt_job(
                            str(job_id),
                            type_def,
                            int(pid),
                            pipeline_id=pipeline_id,
                            execution_group=execution_group,
                            attempt_token=attempt_token,
                            recovery_leased=True,
                            process_identity=(
                                expected_identity if isinstance(expected_identity, dict) else None
                            ),
                        )
                        logger.info(
                            "Re-adopted running job %s (PID %d) on startup",
                            job_id,
                            pid,
                        )
                        recovered += 1
                        continue
                    if identity_match is None:
                        self._quarantine_recovered_job(
                            str(job_id),
                            f"PID {pid} identity is unverifiable",
                            attempt_token=attempt_token,
                            type_def=type_def,
                        )
                        recovered += 1
                        continue

                exact_result = self._exact_attempt_result(
                    str(job_id),
                    attempt_token,
                    params,
                    attempt_result,
                )
                job = {
                    "status": "running",
                    "params": params,
                    "pipeline_id": pipeline_id,
                    "retry_count": retry_count or 0,
                    "max_retries": max_retries or 0,
                    "execution_group": execution_group,
                    "attempt_token": attempt_token,
                    "attempt_result": exact_result,
                    "attempt_failure_recorded": bool(attempt_failure_recorded),
                }
                if exact_result is None:
                    self._quarantine_recovered_job(
                        str(job_id),
                        "Process exited without an exact attempt result",
                        attempt_token=attempt_token,
                        type_def=type_def,
                    )
                else:
                    release_group = self._reconcile_recovered_attempt(
                        str(job_id),
                        type_def,
                        job,
                        execution_group=execution_group,
                    )
                    if release_group:
                        self._release_group(str(job_id))
                recovered += 1
        except (psycopg.Error, RuntimeError, TypeError, ValueError, OSError):
            logger.exception("Failed to recover running jobs on startup")

        # 3. Recover rows marked failed by older server versions, which treated
        # a restart as terminal instead of durable/retryable.
        try:
            with _get_conn() as conn:
                result = conn.execute(
                    """UPDATE job_history SET status = 'queued', pid = NULL,
                              started_at = NULL, completed_at = NULL,
                              error = NULL, progress_pct = 0,
                              progress_msg = 'Re-queued after server restart'
                       WHERE status = 'failed'
                         AND error LIKE 'Interrupted by server restart%'"""
                )
                recovered += int(result.rowcount or 0)
        except Exception:
            logger.exception("Failed to re-queue restart-interrupted jobs")

        # 4. Lease and enqueue durable rows. The DB lease deduplicates API
        # workers; the in-memory ID set also makes repeated local recovery safe.
        try:
            rows = self._lease_queued_jobs()
            queued_ids = {
                queued_job_id
                for queue in self._pending_queues.values()
                for queued_job_id, *_rest in queue
            }
            for row in rows:
                (
                    job_id,
                    job_type,
                    params_raw,
                    max_retries,
                    retry_count,
                    pipeline_id,
                    persisted_group,
                ) = row
                if job_id in queued_ids:
                    continue
                type_def = JOB_TYPE_REGISTRY.get(job_type)
                if not type_def:
                    self._db_update_status(
                        job_id,
                        "failed",
                        error=f"Unknown job type '{job_type}' on restart",
                        completed_at=datetime.now(UTC),
                    )
                    recovered += 1
                    continue
                params = (
                    params_raw if isinstance(params_raw, dict) else json.loads(params_raw or "{}")
                )
                effective_group = self._persisted_execution_group(
                    type_def,
                    persisted_group,
                )
                if effective_group is None:
                    self._db_update_status(
                        job_id,
                        "failed",
                        error="Queued job is missing its execution group",
                        completed_at=datetime.now(UTC),
                        recovery_quarantine_reason="Missing execution group",
                    )
                    recovered += 1
                    continue
                with self._state_lock:
                    queue = self._pending_queues.setdefault(effective_group, [])
                    queue.append(
                        (
                            job_id,
                            type_def,
                            params,
                            max_retries or 0,
                            retry_count or 0,
                            pipeline_id,
                        )
                    )
                    queued_ids.add(job_id)
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

        # 5. Heal the crash window where a step reached completed but the API
        # exited before it durably submitted the successor. The per-step
        # advisory lock in _advance_pipeline_step_once serializes this with a
        # concurrently finishing worker and with other recovering API workers.
        try:
            with _get_conn() as conn:
                completed_pipeline_rows = conn.execute(
                    """SELECT completed.job_id,
                              completed.pipeline_id,
                              completed.params
                       FROM job_history completed
                       WHERE completed.status = 'completed'
                         AND completed.pipeline_id IS NOT NULL
                         AND completed.pipeline_step IS NOT NULL
                         AND jsonb_typeof(
                               completed.params -> '__pipeline_remaining'
                             ) = 'array'
                         AND jsonb_array_length(
                               completed.params -> '__pipeline_remaining'
                             ) > 0
                         AND NOT EXISTS (
                             SELECT 1
                             FROM job_history successor
                             WHERE successor.pipeline_id = completed.pipeline_id
                               AND successor.pipeline_step = completed.pipeline_step + 1
                         )
                       ORDER BY completed.completed_at ASC"""
                ).fetchall()
            for job_id, pipeline_id, params_raw in completed_pipeline_rows:
                params = (
                    params_raw if isinstance(params_raw, dict) else json.loads(params_raw or "{}")
                )
                remaining = params.get("__pipeline_remaining") or []
                if isinstance(remaining, list):
                    self._advance_pipeline_step_once(
                        str(job_id),
                        str(pipeline_id),
                        remaining,
                        params,
                    )
        except (psycopg.Error, TypeError, ValueError):
            logger.exception("Failed to reconcile completed pipeline steps")

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

    def _readopt_job(
        self,
        job_id: str,
        type_def: JobTypeDef,
        pid: int,
        *,
        pipeline_id: str | None = None,
        execution_group: str | None = None,
        attempt_token: str | None = None,
        recovery_leased: bool = False,
        process_identity: dict[str, str] | None = None,
    ) -> None:
        """Start a monitoring thread that polls a re-adopted process until it exits.

        Since the re-adopted process is NOT a child of this API process,
        we cannot use proc.wait(). Instead we poll os.kill(pid, 0).
        """
        exact_group = execution_group or type_def.group
        self._hydrate_group(job_id, exact_group)

        def _monitor():
            release_group = True
            try:
                while self._is_pid_alive(pid):
                    if recovery_leased and not self._renew_recovery_lease(
                        job_id,
                        attempt_token,
                    ):
                        self._quarantine_recovered_job(
                            job_id,
                            "Recovery lease was lost",
                            attempt_token=attempt_token,
                            type_def=type_def,
                        )
                        release_group = False
                        return
                    if process_identity is not None:
                        identity_match = process_identity_matches(
                            pid,
                            process_identity,
                        )
                        if identity_match is False:
                            logger.info(
                                "Recovered process for job %s exited before PID %d was reused",
                                job_id,
                                pid,
                            )
                            break
                        if identity_match is None:
                            logger.error(
                                "Waiting because recovered PID %d identity is unverifiable",
                                pid,
                            )
                            time.sleep(2)
                            continue
                    cancel_event = self._cancel_flags.get(job_id)
                    if cancel_event and cancel_event.is_set():
                        if self._kill_process(job_id):
                            self._db_update_status(
                                job_id,
                                "cancelled",
                                expected_status="running",
                                expected_attempt_token=attempt_token,
                                completed_at=datetime.now(UTC),
                                progress_msg="Cancelled by user (re-adopted job)",
                            )
                            return
                        logger.error(
                            "Cancellation is waiting for recovered PID %d (job %s)",
                            pid,
                            job_id,
                        )
                    time.sleep(2)
                cancel_event = self._cancel_flags.get(job_id)
                if cancel_event and cancel_event.is_set():
                    self._db_update_status(
                        job_id,
                        "cancelled",
                        expected_status="running",
                        expected_attempt_token=attempt_token,
                        completed_at=datetime.now(UTC),
                        progress_msg="Cancelled by user (re-adopted job)",
                    )
                    return
                job = self._db_get(job_id)
                if job and job["status"] == "running":
                    job.setdefault("attempt_token", attempt_token)
                    job.setdefault("execution_group", exact_group)
                    job.setdefault("pipeline_id", pipeline_id)
                    release_group = self._reconcile_recovered_attempt(
                        job_id,
                        type_def,
                        job,
                        execution_group=exact_group,
                    )
            except Exception:  # noqa: BLE001,RUF100 — persist unexpected finalizer failures.
                logger.exception("Monitor thread for re-adopted job %s failed", job_id)
                self._quarantine_recovered_job(
                    job_id,
                    "Monitor thread failed",
                    attempt_token=attempt_token,
                    type_def=type_def,
                )
                release_group = False
            finally:
                if release_group:
                    self._release_group(job_id)
                    self._dispatch_next(exact_group)

        t = threading.Thread(target=_monitor, name=f"readopt-{job_id}", daemon=True)
        t.start()

    def _reconcile_recovered_attempt(
        self,
        job_id: str,
        type_def: JobTypeDef,
        job: dict[str, Any],
        *,
        execution_group: str,
    ) -> bool:
        """Finalize, retry, or quarantine one exact recovered child attempt.

        Returns true only when the execution group may be released.
        """
        params = job.get("params") or {}
        if isinstance(params, str):
            params = json.loads(params)
        if not isinstance(params, dict):
            self._quarantine_recovered_job(
                job_id,
                "Persisted job parameters are invalid",
                attempt_token=job.get("attempt_token"),
                type_def=type_def,
            )
            return False
        attempt_token = job.get("attempt_token")
        if not isinstance(attempt_token, str) or not attempt_token:
            self._quarantine_recovered_job(
                job_id,
                "Running job has no attempt token",
                attempt_token=None,
                type_def=type_def,
            )
            return False
        expected_digest = params.get("__attempt_command_digest")
        if not isinstance(expected_digest, str) or not expected_digest:
            self._quarantine_recovered_job(
                job_id,
                "Running job has no command digest",
                attempt_token=attempt_token,
                type_def=type_def,
            )
            return False
        persisted_result = _validate_attempt_result(
            job.get("attempt_result"),
            attempt_token,
            expected_digest,
        )
        attempt_result = persisted_result or load_attempt_result(
            job_id,
            attempt_token,
            expected_command_digest=expected_digest,
        )
        if attempt_result is None:
            self._quarantine_recovered_job(
                job_id,
                "Process exited without an exact attempt result",
                attempt_token=attempt_token,
                type_def=type_def,
            )
            return False
        if persisted_result is None and not _store_attempt_result(
            job_id,
            attempt_token,
            attempt_result,
        ):
            self._quarantine_recovered_job(
                job_id,
                "Exact attempt result could not be persisted",
                attempt_token=attempt_token,
                type_def=type_def,
            )
            return False

        exit_code = int(attempt_result["exit_code"])
        retry_count = int(job.get("retry_count") or 0)
        max_retries = int(job.get("max_retries") or 0)
        if exit_code != 0:
            failure_recorded = bool(job.get("attempt_failure_recorded"))
            next_retry_count = retry_count if failure_recorded else retry_count + 1
            if next_retry_count <= max_retries:
                self._reconcile_job_terminal_state(
                    job_id,
                    type_def,
                    "failed",
                )
                # Clear the exited process while the old token still owns the
                # row; the requeue CAS intentionally clears that token.
                _clear_pid(job_id, attempt_token)
                requeued = self._db_update_status(
                    job_id,
                    "queued",
                    expected_status="running",
                    expected_attempt_token=attempt_token,
                    started_at=None,
                    completed_at=None,
                    progress_pct=0,
                    progress_msg=(f"Recovered attempt exited {exit_code}; retry queued"),
                    error=None,
                    retry_count=next_retry_count,
                    attempt_token=None,
                    attempt_result=None,
                    attempt_failure_recorded=False,
                    attempt_callable_completion=None,
                    recovery_lease_owner=None,
                    recovery_lease_until=None,
                )
                if requeued:
                    logger.warning(
                        "Re-queued recovered job %s after exit code %d",
                        job_id,
                        exit_code,
                    )
                    return True
                return False
            self._reconcile_job_terminal_state(
                job_id,
                type_def,
                "failed",
            )
            failed = self._db_update_status(
                job_id,
                "failed",
                expected_status="running",
                expected_attempt_token=attempt_token,
                completed_at=datetime.now(UTC),
                error=f"Recovered subprocess exited with code {exit_code}",
                progress_msg="Failed",
                retry_count=next_retry_count,
            )
            if failed:
                _clear_pid(job_id, attempt_token)
            return failed

        callable_completion = params.get("__attempt_callable_completion")
        callable_completed = (
            isinstance(callable_completion, dict)
            and callable_completion.get("attempt_token") == attempt_token
        )
        try:
            if not callable_completed:
                if type_def.type_id == "model_tuning_run":
                    self._finalize_tuning_run(job_id)
                self._finalize_recovered_job(job_id, type_def, job)
        except Exception:  # noqa: BLE001,RUF100 — keep finalization failures quarantined.
            logger.exception("Recovered finalization failed for job %s", job_id)
            self._quarantine_recovered_job(
                job_id,
                "Durable finalization failed",
                attempt_token=attempt_token,
                type_def=type_def,
            )
            return False

        completed = self._db_update_status(
            job_id,
            "completed",
            expected_status="running",
            expected_attempt_token=attempt_token,
            completed_at=datetime.now(UTC),
            progress_pct=100,
            progress_msg="Completed (re-adopted)",
            attempt_result=attempt_result,
        )
        if not completed:
            return False
        _clear_pid(job_id, attempt_token)
        remaining = params.get("__pipeline_remaining", [])
        recovered_pipeline_id = job.get("pipeline_id")
        if remaining and recovered_pipeline_id:
            self._advance_pipeline_step_once(
                job_id,
                str(recovered_pipeline_id),
                remaining,
                params,
            )
        logger.info(
            "Completed recovered job %s in execution group %s",
            job_id,
            execution_group,
        )
        return True

    @staticmethod
    def _finalize_recovered_job(
        job_id: str,
        type_def: JobTypeDef,
        job: dict[str, Any],
    ) -> None:
        """Run completion work lost when the original API process exited."""
        params = job.get("params") or {}
        if isinstance(params, str):
            params = json.loads(params)
        if type_def.type_id == "generate_production_forecast":
            run_id = params.get("run_id")
            if not run_id:
                raise RuntimeError("Recovered generation is missing run_id")
            with _get_conn() as conn:
                row = conn.execute(
                    """SELECT run_status, row_count, artifact_checksum
                       FROM forecast_generation_run
                       WHERE run_id = %s::uuid""",
                    (str(run_id),),
                ).fetchone()
            if row is None:
                raise RuntimeError("Recovered generation has no durable manifest")
            status, row_count, artifact_checksum = row
            if status != "ready" or int(row_count or 0) <= 0 or not artifact_checksum:
                raise RuntimeError("Recovered generation manifest is not a complete ready run")
            return

        if type_def.type_id == "champion_results_load":
            experiment_id = params.get("experiment_id")
            if not experiment_id:
                raise RuntimeError("Recovered champion results load is missing experiment_id")
            from common.core.paths import DATA_DIR

            winners_csv = DATA_DIR / "champion" / f"experiment_{int(experiment_id)}_winners.csv"
            if not winners_csv.exists():
                raise RuntimeError("Recovered champion results load has no winners artifact")
            _finalize_champion_results_lineage(
                int(experiment_id),
                job_id,
                winners_csv,
            )
            return

        if type_def.type_id == "cluster_pipeline":
            verify_cluster_pipeline_completion(
                job_id,
                require_promoted=bool(params.get("auto_promote", True)),
            )
            return

        if type_def.type_id in {"champion_select", "governed_champion_refresh"}:
            experiment_id = params.get("experiment_id")
            if not experiment_id:
                raise RuntimeError("Recovered governed champion refresh is missing experiment_id")
            from common.services.champion_refresh import (
                finalize_governed_champion_refresh,
                refresh_spec_from_payload,
            )

            expected_spec = refresh_spec_from_payload(params.get("governed_spec"))
            result = finalize_governed_champion_refresh(
                int(experiment_id),
                job_id=job_id,
                expected_spec=expected_spec,
            )
            with _get_conn() as conn:
                conn.execute(
                    "UPDATE job_history SET result = %s WHERE job_id = %s",
                    (json.dumps(result), job_id),
                )
            return

        backtest_types = {
            "backtest_lgbm",
            "backtest_chronos2_enriched",
            "backtest_mstl",
            "backtest_nhits",
            "backtest_nbeats",
        }
        if type_def.type_id not in backtest_types:
            parent_postwork_types = {
                "tuning_backtest",
                "load_backtest_results",
                "backtest_load_model",
            }
            if type_def.type_id in parent_postwork_types:
                raise RuntimeError(f"Job type {type_def.type_id} has no durable recovery finalizer")
            return

        from common.services.job_state import (
            _auto_load_backtest,
            _mark_backtest_run_failed,
            _update_backtest_run_on_completion,
        )

        run_id = params.get("backtest_run_id")
        if not run_id:
            raise RuntimeError("Recovered backtest is missing backtest_run_id")
        model = str(params.get("model_id") or type_def.type_id.removeprefix("backtest_"))
        try:
            verify_backtest_artifact_identity(model, int(run_id), job_id)
            verify_backtest_tracking_identity(model, int(run_id), job_id)
            _auto_load_backtest(model, int(run_id), job_id=job_id)
        except Exception:
            _mark_backtest_run_failed(int(run_id))
            raise
        _update_backtest_run_on_completion(int(run_id), model)

    def _finalize_tuning_run(self, job_id: str) -> None:
        """Post-completion registration for re-adopted tuning jobs.

        When a tuning subprocess completes after an API restart, the normal
        callback (complete_run + register_timeframes + register_cluster_month_breakdowns)
        never fires. This method reads the backtest output and registers results.
        """
        job = self._db_get(job_id)
        if not job:
            raise RuntimeError(f"Re-adopted tuning job {job_id} no longer exists")
        params = job.get("params") or {}
        if isinstance(params, str):
            params = json.loads(params)
        run_id = params.get("run_id")
        model = params.get("model")
        if not run_id or not model:
            raise RuntimeError("Re-adopted tuning job is missing run_id or model")

        output_dir = MODEL_OUTPUT_DIRS.get(model)
        if not output_dir:
            raise RuntimeError(f"Re-adopted tuning job has unsupported model {model!r}")

        from common.core.paths import DATA_DIR

        meta_path = DATA_DIR / "backtest" / output_dir / "backtest_metadata.json"
        pred_path = DATA_DIR / "backtest" / output_dir / "backtest_predictions.csv"
        all_lags_path = DATA_DIR / "backtest" / output_dir / "backtest_predictions_all_lags.csv"

        try:
            from common.ml.tuning_tracker import (
                complete_run,
                register_cluster_month_breakdowns,
                register_lag_breakdowns,
                register_timeframes,
            )

            complete_run(run_id, meta_path)
            register_timeframes(run_id, meta_path)
            register_lag_breakdowns(run_id, all_lags_path)
            if pred_path.exists():
                register_cluster_month_breakdowns(run_id, pred_path)
            logger.info(
                "Finalized re-adopted tuning run %d (%s) from backtest output", run_id, model
            )
        except Exception:
            logger.exception("Failed to finalize re-adopted tuning run %d (%s)", run_id, model)
            raise

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
        retry_count: int = 0,
        execution_group: str | None = None,
    ) -> None:
        """Execute a job within APScheduler's thread pool."""
        cancel_event = self._cancel_flags.get(job_id)
        first_attempt = True
        previous_attempt_token: str | None = None
        exact_group = execution_group or self._active_jobs.get(
            job_id,
            type_def.group,
        )

        try:
            while True:
                attempt_token = uuid.uuid4().hex
                try:
                    started = self._db_update_status(
                        job_id,
                        "running",
                        expected_status="queued" if first_attempt else "running",
                        expected_attempt_token=previous_attempt_token,
                        started_at=datetime.now(UTC),
                        progress_pct=0,
                        progress_msg="Starting",
                        retry_count=retry_count,
                        attempt_token=attempt_token,
                        attempt_result=None,
                        attempt_failure_recorded=False,
                        attempt_callable_completion=None,
                        recovery_lease_owner=None,
                        recovery_lease_until=None,
                        recovery_quarantine_reason=None,
                    )
                    if started is False:
                        logger.info(
                            "Job %s was terminal before worker attempt %d started",
                            job_id,
                            retry_count + 1,
                        )
                        break
                    first_attempt = False
                    previous_attempt_token = attempt_token

                    def progress_cb(
                        pct: int | None = None,
                        msg: str | None = None,
                        *,
                        _attempt_token: str = attempt_token,
                    ) -> None:
                        if cancel_event and cancel_event.is_set():
                            raise JobCancelledError("Job cancelled by user")
                        updates: dict[str, Any] = {}
                        if pct is not None:
                            updates["progress_pct"] = pct
                        if msg is not None:
                            updates["progress_msg"] = msg
                        if updates:
                            self._db_update_status(
                                job_id,
                                "running",
                                expected_status="running",
                                expected_attempt_token=_attempt_token,
                                **updates,
                            )

                    # Strip durable orchestration metadata before invoking the
                    # domain callable; only user/job parameters cross this seam.
                    clean_params = {
                        key: value for key, value in params.items() if not key.startswith("__")
                    }
                    context_token = bind_job_attempt(attempt_token)
                    try:
                        result = type_def.callable(
                            clean_params,
                            progress_cb,
                            cancel_event=cancel_event,
                            job_id=job_id,
                        )
                    finally:
                        reset_job_attempt(context_token)
                    if cancel_event and cancel_event.is_set():
                        raise JobCancelledError("Job cancelled by user")

                    callable_persisted = self._db_update_status(
                        job_id,
                        "running",
                        expected_status="running",
                        expected_attempt_token=attempt_token,
                        result=result,
                        attempt_callable_completion={
                            "attempt_token": attempt_token,
                        },
                    )
                    if not callable_persisted:
                        logger.error(
                            "Job %s finished but its exact callable completion "
                            "could not be persisted",
                            job_id,
                        )
                        break

                    completed = self._db_update_status(
                        job_id,
                        "completed",
                        expected_status="running",
                        expected_attempt_token=attempt_token,
                        completed_at=datetime.now(UTC),
                        progress_pct=100,
                        progress_msg="Done",
                    )
                    if not completed:
                        logger.info(
                            "Job %s finished after its durable status became terminal",
                            job_id,
                        )
                        break
                    logger.info("Job %s completed successfully", job_id)

                    # If part of a pipeline, trigger next step.
                    remaining = params.get("__pipeline_remaining", [])
                    if remaining and pipeline_id:
                        self._advance_pipeline_step_once(
                            job_id,
                            pipeline_id,
                            remaining,
                            params,
                        )
                    break

                except JobCancelledError:
                    self._reconcile_job_terminal_state(
                        job_id,
                        type_def,
                        "cancelled",
                    )
                    self._db_update_status(
                        job_id,
                        "cancelled",
                        expected_status="running",
                        expected_attempt_token=attempt_token,
                        completed_at=datetime.now(UTC),
                        progress_msg="Cancelled by user",
                        retry_count=retry_count,
                    )
                    break

                except Exception as exc:  # noqa: BLE001,RUF100 — persist worker failures.
                    if cancel_event and cancel_event.is_set():
                        self._reconcile_job_terminal_state(
                            job_id,
                            type_def,
                            "cancelled",
                        )
                        self._db_update_status(
                            job_id,
                            "cancelled",
                            expected_status="running",
                            expected_attempt_token=attempt_token,
                            completed_at=datetime.now(UTC),
                            progress_msg="Cancelled by user",
                            retry_count=retry_count,
                        )
                        break

                    retry_count += 1
                    if retry_count <= max_retries:
                        self._reconcile_job_terminal_state(
                            job_id,
                            type_def,
                            "failed",
                        )
                        delay = min(2**retry_count, 60)
                        logger.warning(
                            "Job %s failed (attempt %d/%d), retrying in %ds: %s",
                            job_id,
                            retry_count,
                            max_retries + 1,
                            delay,
                            exc,
                        )
                        self._db_update_status(
                            job_id,
                            "running",
                            expected_status="running",
                            expected_attempt_token=attempt_token,
                            progress_msg=(f"Retry {retry_count}/{max_retries} in {delay}s"),
                            retry_count=retry_count,
                            attempt_failure_recorded=True,
                        )
                        if cancel_event:
                            if cancel_event.wait(delay):
                                self._reconcile_job_terminal_state(
                                    job_id,
                                    type_def,
                                    "cancelled",
                                )
                                self._db_update_status(
                                    job_id,
                                    "cancelled",
                                    expected_status="running",
                                    expected_attempt_token=attempt_token,
                                    completed_at=datetime.now(UTC),
                                    progress_msg="Cancelled by user",
                                    retry_count=retry_count,
                                )
                                break
                        else:
                            time.sleep(delay)
                        continue

                    logger.exception(
                        "Job %s failed after %d attempts",
                        job_id,
                        retry_count,
                    )
                    self._reconcile_job_terminal_state(
                        job_id,
                        type_def,
                        "failed",
                    )
                    self._db_update_status(
                        job_id,
                        "failed",
                        expected_status="running",
                        expected_attempt_token=attempt_token,
                        completed_at=datetime.now(UTC),
                        error=str(exc),
                        progress_msg="Failed",
                        retry_count=retry_count,
                        attempt_failure_recorded=True,
                    )
                    break
        finally:
            with self._state_lock:
                # Use the stored group when submit_job applied group_override.
                active_group = self._active_jobs.get(job_id, exact_group)
                was_active = job_id in self._active_jobs
                self._active_jobs.pop(job_id, None)
                self._cancel_flags.pop(job_id, None)
            if was_active:
                self._dispatch_next(active_group)

    def _dispatch_next(self, group: str) -> None:
        """Pop the next queued job for *group* and dispatch it to APScheduler."""
        with self._state_lock:
            queue = self._pending_queues.get(group)
            if not queue:
                return

            (
                job_id,
                type_def,
                params,
                max_retries,
                retry_count,
                pipeline_id,
            ) = queue.pop(0)
            remaining = len(queue)

            # Track as active (inside lock)
            self._active_jobs[job_id] = group
            self._cancel_flags[job_id] = threading.Event()

        # APScheduler dispatch outside lock (I/O operation)
        logger.info(
            "Auto-dispatching queued job %s for group '%s' (%d remaining in queue)",
            job_id,
            type_def.type_id,
            remaining,
        )
        self._scheduler.add_job(
            self._execute_job,
            args=[
                job_id,
                type_def,
                params,
                max_retries,
                pipeline_id,
                retry_count,
                group,
            ],
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

    def _advance_pipeline_step_once(
        self,
        completed_job_id: str,
        pipeline_id: str,
        remaining: list[dict],
        original_params: dict,
    ) -> bool:
        """Submit one missing successor under a cross-process step lock."""
        if not remaining:
            return False
        try:
            current_step = int(original_params.get("__pipeline_step", 1))
            next_step = current_step + 1
            with _get_conn() as conn:
                conn.execute(
                    "SELECT pg_advisory_lock(hashtext(%s), %s)",
                    (pipeline_id, next_step),
                )
                try:
                    claim_token = uuid.uuid4().hex
                    claimed = conn.execute(
                        """UPDATE job_history completed
                           SET params = jsonb_set(
                               COALESCE(completed.params, '{}'::jsonb),
                               '{__pipeline_advance_claim}',
                               to_jsonb(%s::text),
                               true
                           )
                           WHERE completed.job_id = %s
                             AND completed.status = 'completed'
                             AND completed.pipeline_id = %s
                             AND completed.pipeline_step = %s
                             AND NOT EXISTS (
                                 SELECT 1
                                 FROM job_history successor
                                 WHERE successor.pipeline_id = %s
                                   AND successor.pipeline_step = %s
                             )
                           RETURNING 1""",
                        (
                            claim_token,
                            completed_job_id,
                            pipeline_id,
                            current_step,
                            pipeline_id,
                            next_step,
                        ),
                    ).fetchone()
                    if claimed is None:
                        successor = conn.execute(
                            """SELECT 1
                               FROM job_history
                               WHERE pipeline_id = %s AND pipeline_step = %s
                               LIMIT 1""",
                            (pipeline_id, next_step),
                        ).fetchone()
                        return successor is not None
                    submitted = self._trigger_next_pipeline_step(
                        pipeline_id,
                        remaining,
                        original_params,
                    )
                    return submitted is not None
                finally:
                    conn.execute(
                        "SELECT pg_advisory_unlock(hashtext(%s), %s)",
                        (pipeline_id, next_step),
                    )
                    logger.debug(
                        "Released pipeline continuation lock for %s after %s",
                        pipeline_id,
                        completed_job_id,
                    )
        except (psycopg.Error, RuntimeError, TypeError, ValueError):
            logger.exception(
                "Failed to advance pipeline %s after job %s",
                pipeline_id,
                completed_job_id,
            )
            return False

    def _trigger_next_pipeline_step(
        self, pipeline_id: str, remaining: list[dict], original_params: dict
    ) -> str | None:
        """Submit the next step in a pipeline."""
        if not remaining:
            return None
        next_step = deepcopy(remaining[0])
        future_steps = deepcopy(remaining[1:])
        job_type = next_step["job_type"]
        params = next_step.get("params") or {}
        step_label = next_step.get("label", JOB_TYPE_REGISTRY[job_type].label)
        current_step = int(original_params.get("__pipeline_step", 1))
        total_steps = int(
            original_params.get("__pipeline_total_steps", current_step + len(remaining))
        )
        step_num = current_step + 1
        pipeline_label = str(original_params.get("__pipeline_label", "Pipeline"))
        params["__pipeline_step"] = step_num
        params["__pipeline_total_steps"] = total_steps
        params["__pipeline_remaining"] = future_steps
        params["__pipeline_label"] = pipeline_label

        try:
            return self.submit_job(
                job_type=job_type,
                params=params,
                label=f"[{pipeline_label} {step_num}/{total_steps}] {step_label}",
                triggered_by="pipeline",
                pipeline_id=pipeline_id,
                pipeline_step=step_num,
            )
        except Exception as exc:
            logger.exception("Pipeline %s step %d failed to submit: %s", pipeline_id, step_num, exc)
            return None
