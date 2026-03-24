"""Pure state management for the job engine (no APScheduler, no psycopg at module level).

Contains:
- DB connection helper (_get_conn)
- JobTypeDef dataclass
- Job callable wrappers (_run_*)
- _row_to_dict helper
- _SCRIPTS_DIR / _UV constants

Deliberately free of APScheduler and psycopg imports at the module level so
that this module can be imported from tests without starting the full API or
requiring a running scheduler.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _get_conn():
    """Open a single psycopg connection using environment variables."""
    import psycopg  # imported here to keep module-level imports APScheduler-free

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

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
_UV = "uv"


_SUBPROCESS_TIMEOUT = 7200  # 2 hours — prevents hung jobs from blocking the executor thread
_LOG_FLUSH_INTERVAL = 5  # seconds between DB log flushes
_LOG_FLUSH_LINES = 20  # flush after this many buffered lines


# ---------------------------------------------------------------------------
# PID + log DB helpers (used by _run_subprocess for resilient jobs)
# ---------------------------------------------------------------------------


def _store_pid(job_id: str | None, pid: int) -> None:
    """Store the subprocess PID in job_history for kill/recovery."""
    if not job_id:
        return
    try:
        with _get_conn() as conn:
            conn.execute("UPDATE job_history SET pid = %s WHERE job_id = %s", (pid, job_id))
    except Exception:
        logger.warning("Failed to store PID %d for job %s", pid, job_id)


def _clear_pid(job_id: str | None) -> None:
    """Clear the PID column after subprocess exits."""
    if not job_id:
        return
    try:
        with _get_conn() as conn:
            conn.execute("UPDATE job_history SET pid = NULL WHERE job_id = %s", (job_id,))
    except Exception:
        logger.warning("Failed to clear PID for job %s", job_id)


def _append_log(job_id: str | None, text: str) -> None:
    """Append text to the persistent log column in job_history."""
    if not job_id or not text:
        return
    try:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE job_history SET log = COALESCE(log, '') || %s WHERE job_id = %s",
                (text, job_id),
            )
    except Exception:
        logger.warning("Failed to append log for job %s", job_id)


def get_job_log(job_id: str) -> str:
    """Read the persistent log for a job. Returns empty string if not found."""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(log, '') FROM job_history WHERE job_id = %s", (job_id,)
            ).fetchone()
        return row[0] if row else ""
    except Exception:
        logger.warning("Failed to read log for job %s", job_id)
        return ""


def get_job_pid(job_id: str) -> int | None:
    """Read the PID for a running job. Returns None if not found or cleared."""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT pid FROM job_history WHERE job_id = %s", (job_id,)
            ).fetchone()
        return row[0] if row and row[0] else None
    except Exception:
        logger.warning("Failed to read PID for job %s", job_id)
        return None


def _run_subprocess(
    cmd: list[str],
    progress_cb: Callable | None = None,
    step_msg: str = "",
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> str:
    """Run a subprocess command with PID tracking, cancellation, and log streaming.

    - Subprocess runs in its own process group (start_new_session=True) so it
      survives API restarts.
    - PID is stored in job_history for real kill and startup recovery.
    - stdout is streamed to DB log column periodically.
    - cancel_event is checked between reads; on cancel, the process group is killed.

    Returns the full stdout as a string. Raises on failure, timeout, or cancel.
    """
    if progress_cb and step_msg:
        progress_cb(msg=step_msg)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(_SCRIPTS_DIR.parent),
        start_new_session=True,
    )

    # Store PID for kill/recovery
    _store_pid(job_id, proc.pid)

    stdout_lines: list[str] = []
    log_buffer: list[str] = []
    last_flush = time.time()

    try:
        assert proc.stdout is not None
        start = time.time()

        for line in proc.stdout:
            stripped = line.rstrip("\n")
            stdout_lines.append(stripped)
            log_buffer.append(line)

            if progress_cb and stripped:
                progress_cb(msg=stripped)

            # Check cancellation
            if cancel_event and cancel_event.is_set():
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=5)
                raise RuntimeError("Job cancelled by user")

            # Check timeout
            if time.time() - start > _SUBPROCESS_TIMEOUT:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10)
                raise RuntimeError("Subprocess timed out")

            # Flush log buffer to DB periodically
            if log_buffer and (
                len(log_buffer) >= _LOG_FLUSH_LINES
                or time.time() - last_flush > _LOG_FLUSH_INTERVAL
            ):
                _append_log(job_id, "".join(log_buffer))
                log_buffer.clear()
                last_flush = time.time()

        proc.wait(timeout=30)

        # Final flush
        if log_buffer:
            _append_log(job_id, "".join(log_buffer))
            log_buffer.clear()

        if proc.returncode != 0:
            error_msg = "\n".join(stdout_lines[-20:]) or "Unknown error"
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{error_msg.strip()}")

        return "\n".join(stdout_lines)
    finally:
        _clear_pid(job_id)


def _run_cluster_scenario(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
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


def _run_cluster_pipeline(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run the full clustering pipeline: features -> train -> label -> update."""
    tw = params.get("time_window_months", 24)
    k_range = params.get("k_range", [3, 12])
    steps = [
        (25, "Generating clustering features",
         [_UV, "run", "python", "scripts/generate_clustering_features.py", "--time-window", str(tw)]),
        (50, "Training clustering model",
         [_UV, "run", "python", "scripts/train_clustering_model.py",
          "--k-range", str(k_range[0]), str(k_range[1])]),
        (75, "Labeling clusters",
         [_UV, "run", "python", "scripts/label_clusters.py"]),
        (95, "Updating DFU assignments",
         [_UV, "run", "python", "scripts/update_cluster_assignments.py"]),
    ]
    outputs = []
    for pct, msg, cmd in steps:
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Job cancelled by user")
        if progress_cb:
            progress_cb(pct=pct, msg=msg)
        out = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
        outputs.append(out)
    return {"steps_completed": len(steps), "output_log": "Pipeline completed successfully"}


def _run_seasonality(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run the seasonality detection + update pipeline."""
    config = params.get("config", "config/seasonality_config.yaml")
    steps = [
        (40, "Detecting seasonality patterns",
         [_UV, "run", "python", "scripts/detect_seasonality.py", "--config", config]),
        (90, "Updating seasonality profiles",
         [_UV, "run", "python", "scripts/update_seasonality_profiles.py", "--config", config]),
    ]
    for pct, msg, cmd in steps:
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Job cancelled by user")
        if progress_cb:
            progress_cb(pct=pct, msg=msg)
        _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"steps_completed": len(steps), "output_log": "Seasonality pipeline completed"}


def _run_backtest(
    model: str,
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
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
    config_path = params.get("config")
    if progress_cb:
        progress_cb(pct=0, msg=f"Running {model.upper()} backtest ({strategy})")
    cmd = [_UV, "run", "python", script, "--cluster-strategy", strategy]
    if config_path:
        cmd.extend(["--config", config_path])
    output = _run_subprocess(cmd, progress_cb, cancel_event=cancel_event, job_id=job_id)
    if progress_cb:
        progress_cb(pct=100, msg=f"{model.upper()} backtest completed")
    return {"model": model, "strategy": strategy, "output_log": output if output else "Completed"}


def _run_backtest_lgbm(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    return _run_backtest("lgbm", params, progress_cb, cancel_event=cancel_event, job_id=job_id)


def _run_backtest_catboost(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    return _run_backtest("catboost", params, progress_cb, cancel_event=cancel_event, job_id=job_id)


def _run_backtest_xgboost(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    return _run_backtest("xgboost", params, progress_cb, cancel_event=cancel_event, job_id=job_id)


def _run_champion_select(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run champion model selection."""
    if progress_cb:
        progress_cb(pct=10, msg="Running champion selection")
    cmd = [_UV, "run", "python", "scripts/run_champion_selection.py"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "Champion selection completed"}


def _run_generate_production_forecast(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run the production forecast generation pipeline (F1.1)."""
    horizon = params.get("horizon", 12)
    if progress_cb:
        progress_cb(pct=5, msg=f"Starting production forecast generation (horizon={horizon})")
    cmd = [_UV, "run", "python", "scripts/generate_production_forecasts.py", "--horizon", str(horizon)]
    output = _run_subprocess(cmd, progress_cb, "Generating production forecasts",
                             cancel_event=cancel_event, job_id=job_id)
    if progress_cb:
        progress_cb(pct=100, msg="Production forecast generation complete")
    return {"horizon": horizon, "output_log": output if output else "Production forecast generation completed"}


def _run_compute_replenishment_plan(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run the forward-looking replenishment plan computation (CI Bands + Repl. Plan)."""
    if progress_cb:
        progress_cb(pct=10, msg="Starting replenishment plan computation")
    cmd = [_UV, "run", "python", "scripts/compute_replenishment_plan.py"]
    output = _run_subprocess(cmd, progress_cb, "Computing replenishment plan from production forecast",
                             cancel_event=cancel_event, job_id=job_id)
    if progress_cb:
        progress_cb(pct=100, msg="Replenishment plan computation complete")
    return {"output_log": output if output else "Replenishment plan computation completed"}


def _run_generate_ai_insights(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run AI Planning Agent portfolio scan to generate insights."""
    if progress_cb:
        progress_cb(pct=5, msg="Starting AI insights generation")
    cmd = [_UV, "run", "python", "scripts/generate_ai_insights.py", "--portfolio"]
    output = _run_subprocess(cmd, progress_cb, "Scanning portfolio for exceptions",
                             cancel_event=cancel_event, job_id=job_id)
    if progress_cb:
        progress_cb(pct=100, msg="AI insights generation complete")
    return {"output_log": output if output else "AI insights generation completed"}


def _run_generate_storyboard(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Generate storyboard exceptions for all DFUs."""
    if progress_cb:
        progress_cb(pct=10, msg="Generating storyboard exceptions")
    cmd = [_UV, "run", "python", "scripts/generate_storyboard_exceptions.py"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "Storyboard exceptions generated"}


def _run_compute_safety_stock(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Compute safety stock targets for all DFUs."""
    if progress_cb:
        progress_cb(pct=10, msg="Computing safety stock targets")
    cmd = [_UV, "run", "python", "scripts/compute_safety_stock.py", "--config", "config/safety_stock_config.yaml"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "Safety stock computation completed"}


def _run_compute_eoq(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Compute EOQ cycle stock targets."""
    if progress_cb:
        progress_cb(pct=10, msg="Computing EOQ targets")
    cmd = [_UV, "run", "python", "scripts/compute_eoq.py", "--config", "config/eoq_config.yaml"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "EOQ computation completed"}


def _run_assign_policies(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Upsert replenishment policies and auto-assign DFUs by segment."""
    if progress_cb:
        progress_cb(pct=10, msg="Assigning replenishment policies")
    cmd = [_UV, "run", "python", "scripts/assign_replenishment_policies.py",
           "--config", "config/replenishment_policy_config.yaml"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "Policy assignment completed"}


def _run_generate_exceptions(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Detect replenishment exceptions and write to queue."""
    if progress_cb:
        progress_cb(pct=10, msg="Detecting replenishment exceptions")
    cmd = [_UV, "run", "python", "scripts/generate_replenishment_exceptions.py"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "Exception detection completed"}


def _run_classify_abc_xyz(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run ABC-XYZ classification and write to dim_sku."""
    if progress_cb:
        progress_cb(pct=10, msg="Running ABC-XYZ classification")
    cmd = [_UV, "run", "python", "scripts/classify_abc_xyz.py"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "ABC-XYZ classification completed"}


def _run_compute_variability(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Compute demand variability (CV, MAD) per DFU."""
    if progress_cb:
        progress_cb(pct=10, msg="Computing demand variability")
    cmd = [_UV, "run", "python", "scripts/compute_demand_variability.py",
           "--config", "config/variability_config.yaml"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "Demand variability computation completed"}


def _run_compute_demand_signals(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Compute short-horizon demand signals from sales velocity."""
    if progress_cb:
        progress_cb(pct=10, msg="Computing demand signals")
    cmd = [_UV, "run", "python", "scripts/compute_demand_signals.py"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "Demand signals computation completed"}


def _run_compute_investment(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Compute efficient frontier and capital investment allocation."""
    if progress_cb:
        progress_cb(pct=10, msg="Computing investment plan")
    cmd = [_UV, "run", "python", "scripts/compute_investment_plan.py"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "Investment plan computation completed"}


def _run_refresh_health_scores(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Refresh the inventory health score materialized view."""
    if progress_cb:
        progress_cb(pct=10, msg="Refreshing inventory health scores")
    cmd = [_UV, "run", "python", "scripts/refresh_health_scores.py"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "Health scores refreshed"}


def _run_refresh_intramonth(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Refresh the intramonth stockout materialized view."""
    if progress_cb:
        progress_cb(pct=10, msg="Refreshing intramonth stockout view")
    cmd = [_UV, "run", "python", "scripts/refresh_intramonth_stockout.py"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "Intramonth stockout view refreshed"}


def _run_ss_simulation(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run Monte Carlo safety stock simulation."""
    if progress_cb:
        progress_cb(pct=10, msg="Running Monte Carlo SS simulation")
    cmd = [_UV, "run", "python", "scripts/run_ss_simulation.py"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "SS simulation completed"}


def _run_data_quality(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run all data quality checks via DQEngine."""
    from common.dq_engine import DQEngine

    if progress_cb:
        progress_cb(pct=10, msg="Running data quality checks")
    domain = params.get("domain")
    engine = DQEngine()
    results = engine.run_all_checks(domain=domain)
    total = len(results)
    passed = sum(1 for r in results if r.get("status") == "pass")
    failed = sum(1 for r in results if r.get("status") == "fail")
    if progress_cb:
        progress_cb(pct=100, msg="Data quality checks complete")
    return {
        "total_checks": total,
        "passed": passed,
        "failed": failed,
        "output_log": f"Data quality checks completed: {passed}/{total} passed, {failed} failed",
    }


def _run_tuning_backtest(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run a tuning chat backtest: build temp config, run backtest, register results, insert chat message."""
    import copy
    import tempfile

    import yaml

    ROOT = Path(__file__).resolve().parents[2]

    run_id = params["run_id"]
    session_id = params["session_id"]
    overrides = params.get("overrides", {})
    strategy_label = params.get("strategy_label", "chat_experiment")

    if progress_cb:
        progress_cb(pct=5, msg=f"Starting tuning backtest #{run_id} ({strategy_label})")

    # 1. Build temp config with overrides
    algo_path = ROOT / "config" / "algorithm_config.yaml"
    with open(algo_path) as f:
        base_config = yaml.safe_load(f)

    cfg = copy.deepcopy(base_config)
    cfg["algorithms"]["lgbm"].update(overrides)

    tmp_dir = Path(tempfile.mkdtemp(prefix="tuning_chat_"))
    tmp_path = tmp_dir / f"algorithm_config_{strategy_label}.yaml"
    with open(tmp_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # 2. Run backtest via _run_subprocess (gets PID tracking + cancel + log streaming)
    cmd = [
        _UV, "run", "python", str(ROOT / "scripts" / "run_backtest.py"),
        "--model", "lgbm",
        "--config", str(tmp_path),
    ]
    start = time.time()
    output = _run_subprocess(cmd, progress_cb, "Running LGBM backtest",
                             cancel_event=cancel_event, job_id=job_id)
    duration = time.time() - start

    # 3. Complete run via tracker
    from common.ml.tuning_tracker import (
        complete_run,
        register_cluster_month_breakdowns,
        register_timeframes,
    )

    meta_path = ROOT / "data" / "backtest" / "lgbm_cluster" / "backtest_metadata.json"
    complete_run(run_id, meta_path)
    register_timeframes(run_id, meta_path)

    predictions_path = ROOT / "data" / "backtest" / "lgbm_cluster" / "backtest_predictions.csv"
    if predictions_path.exists():
        register_cluster_month_breakdowns(run_id, predictions_path)

    # 4. Insert run_completed chat message
    _insert_tuning_chat_message(session_id, run_id, "run_completed", None)

    if progress_cb:
        progress_cb(pct=100, msg=f"Tuning run #{run_id} completed in {duration:.0f}s")
    return {"run_id": run_id, "strategy_label": strategy_label, "duration_seconds": round(duration),
            "output_log": output[:5000] if output else "Tuning backtest completed"}


def _insert_tuning_chat_message(
    session_id: str, run_id: int, msg_type: str, error: str | None,
) -> None:
    """Insert a run_completed or run_failed chat message (called from background callable)."""
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                if msg_type == "run_completed":
                    cur.execute(
                        "SELECT accuracy_pct, wape, bias, n_predictions, n_dfus "
                        "FROM lgbm_tuning_run WHERE run_id = %s",
                        (run_id,),
                    )
                    row = cur.fetchone()
                    metadata: dict[str, Any] = {"run_id": run_id}
                    if row:
                        metadata.update({
                            "accuracy_pct": row[0], "wape": row[1], "bias": row[2],
                            "n_predictions": row[3], "n_dfus": row[4],
                        })
                    content = (
                        f"Run #{run_id} completed — "
                        f"accuracy {row[0]:.2f}%, WAPE {row[1]:.2f}, bias {row[2]:.4f}"
                        if row and row[0] is not None
                        else f"Run #{run_id} completed"
                    )
                else:
                    metadata = {"run_id": run_id, "error": error}
                    content = f"Run #{run_id} failed: {error or 'unknown error'}"

                cur.execute(
                    """INSERT INTO tuning_chat_message
                        (session_id, role, content, message_type, metadata)
                    VALUES (%s::uuid, 'system', %s, %s, %s)""",
                    (session_id, content, msg_type,
                     json.dumps(metadata, default=str)),
                )
    except Exception:
        logger.warning("Failed to insert %s message for run %d", msg_type, run_id)


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
        elif col == "logs":
            if isinstance(val, list):
                d[col] = val
            elif val:
                d[col] = json.loads(val)
            else:
                d[col] = []
        elif col in ("submitted_at", "started_at", "completed_at"):
            d[col] = val.isoformat() if val else None
        elif col == "progress_pct":
            d[col] = val or 0
        elif col == "pid":
            d[col] = val  # int or None
        else:
            d[col] = val
    return d
