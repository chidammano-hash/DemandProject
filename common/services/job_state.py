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
    """Open a single psycopg connection using get_db_params()."""
    import psycopg  # imported here to keep module-level imports APScheduler-free
    from common.db import get_db_params

    return psycopg.connect(**get_db_params(), autocommit=True)


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

# Model name → backtest output directory mapping (shared across job callables)
MODEL_OUTPUT_DIRS: dict[str, str] = {
    "lgbm": "lgbm_cluster",
    "catboost": "catboost_cluster",
    "xgboost": "xgboost_cluster",
}


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

    # Force line-buffered stdout so log streaming is real-time, not block-buffered
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(_SCRIPTS_DIR.parent),
        start_new_session=True,
        env=env,
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
    """Run a what-if clustering scenario (delegates to run_clustering_scenario.py).

    When params contains ``experiment_id``, the cluster_experiment row is updated
    to ``status='running'`` before starting, and the experiment_id is forwarded to
    ``run_scenario()`` so it can write results on completion/failure.
    """
    from scripts.ml.run_clustering_scenario import run_scenario, generate_scenario_id, get_scenario_result

    scenario_id = params.get("scenario_id") or generate_scenario_id()
    experiment_id: int | None = params.get("experiment_id")

    if progress_cb:
        progress_cb(pct=5, msg="Starting clustering scenario")

    # If this is a cluster experiment, mark it as running
    if experiment_id is not None:
        try:
            with _get_conn() as conn:
                conn.execute(
                    "UPDATE cluster_experiment SET status = 'running', started_at = NOW() "
                    "WHERE experiment_id = %s",
                    (experiment_id,),
                )
        except Exception:
            logger.warning("Failed to update cluster_experiment %d to running", experiment_id)

    try:
        run_scenario(
            scenario_id=scenario_id,
            feature_params=params.get("feature_params"),
            model_params=params.get("model_params"),
            label_params=params.get("label_params"),
            relabel_only=params.get("relabel_only", False),
            previous_scenario_id=params.get("previous_scenario_id"),
            experiment_id=experiment_id,
        )
    except Exception:
        # run_scenario already handles updating experiment status to 'failed'
        # internally, but if it raises before that (unlikely), mark failed here
        if experiment_id is not None:
            try:
                with _get_conn() as conn:
                    conn.execute(
                        "UPDATE cluster_experiment SET status = 'failed', completed_at = NOW() "
                        "WHERE experiment_id = %s AND status = 'running'",
                        (experiment_id,),
                    )
            except Exception:
                logger.warning("Failed to mark cluster_experiment %d as failed", experiment_id)
        raise

    result = get_scenario_result(scenario_id) or {}
    return {"scenario_id": scenario_id, **result}


def _run_cluster_pipeline(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run the full clustering pipeline via the unified experiment system."""
    from scripts.ml.run_cluster_pipeline import run_unified_pipeline

    if progress_cb:
        progress_cb(pct=10, msg="Starting unified clustering pipeline")

    result = run_unified_pipeline(
        feature_params=params.get("feature_params"),
        model_params=params.get("model_params"),
        label_params=params.get("label_params"),
        label=params.get("label", "Job Pipeline Run"),
        auto_promote=params.get("auto_promote", True),
    )

    if progress_cb:
        progress_cb(pct=100, msg="Pipeline completed")

    return result


def _run_seasonality(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Legacy seasonality pipeline — delegates to unified SKU features."""
    return _run_compute_sku_features(params, progress_cb, cancel_event, job_id)


def _update_backtest_run_on_completion(run_id: int, model: str) -> None:
    """Update a backtest_run row with results from the completed backtest metadata."""
    import json
    from pathlib import Path

    # Model ID mapping (backtest key → output directory model_id)
    _MODEL_TO_DIR = {
        "lgbm": "lgbm_cluster", "catboost": "catboost_cluster", "xgboost": "xgboost_cluster",
        "chronos": "chronos", "chronos_bolt": "chronos_bolt", "chronos2": "chronos2",
        "chronos2_enriched": "chronos2_enriched", "bolt_hierarchical": "bolt_hierarchical",
        "mstl": "mstl", "nbeats": "nbeats", "nhits": "nhits",
        "seasonal_naive": "seasonal_naive", "rolling_mean": "rolling_mean",
        "lgbm_cust_enriched": "lgbm_cust_enriched", "catboost_cust_enriched": "catboost_cust_enriched",
        "xgboost_cust_enriched": "xgboost_cust_enriched",
    }
    model_dir = _MODEL_TO_DIR.get(model, model)
    meta_path = Path("data/backtest") / model_dir / "backtest_metadata.json"

    try:
        with _get_conn() as conn:
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                acc = meta.get("accuracy_at_execution_lag", {})
                conn.execute(
                    """UPDATE backtest_run SET
                        status = 'completed', completed_at = NOW(),
                        accuracy_pct = %s, wape = %s, bias = %s,
                        n_predictions = %s, n_dfus = %s,
                        metadata = %s::jsonb
                    WHERE id = %s""",
                    (
                        acc.get("accuracy_pct"), acc.get("wape"), acc.get("bias"),
                        meta.get("n_predictions"), meta.get("n_dfus"),
                        json.dumps(meta), run_id,
                    ),
                )
            else:
                conn.execute(
                    "UPDATE backtest_run SET status = 'completed', completed_at = NOW() WHERE id = %s",
                    (run_id,),
                )
    except Exception:
        logger.warning("Failed to update backtest_run %d after completion", run_id)


def _run_backtest(
    model: str,
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run a backtest for a given model type.

    Supports tree models (direct script), foundation models (module invocation),
    deep learning (--model flag), and statistical baselines.
    """
    # Tree models: direct script invocation
    tree_scripts = {
        "lgbm": "scripts/run_backtest.py",
        "catboost": "scripts/run_backtest_catboost.py",
        "xgboost": "scripts/run_backtest_xgboost.py",
    }
    # Foundation models: python -m invocation
    foundation_modules = {
        "chronos": "scripts.ml.run_backtest_chronos",
        "chronos_bolt": "scripts.ml.run_backtest_chronos_bolt",
        "chronos2": "scripts.ml.run_backtest_chronos2",
        "chronos2_enriched": "scripts.ml.run_backtest_chronos2_enriched",
    }
    # Special scripts: direct file path
    special_scripts = {
        "bolt_hierarchical": "scripts/run_backtest_bolt_hierarchical.py",
        "mstl": "scripts/run_backtest_mstl.py",
        "seasonal_naive": ("scripts/run_backtest.py", ["--model", "seasonal_naive"]),
        "rolling_mean": ("scripts/run_backtest.py", ["--model", "rolling_mean"]),
        "nhits": ("scripts/run_backtest_dl.py", ["--model", "nhits"]),
        "nbeats": ("scripts/run_backtest_dl.py", ["--model", "nbeats"]),
    }

    backtest_run_id = params.get("backtest_run_id")
    if progress_cb:
        progress_cb(pct=0, msg=f"Running {model} backtest")

    # Mark as running
    if backtest_run_id:
        try:
            with _get_conn() as conn:
                conn.execute(
                    "UPDATE backtest_run SET status = 'running', started_at = NOW() WHERE id = %s",
                    (backtest_run_id,),
                )
        except Exception:
            logger.warning("Failed to mark backtest_run %d as running", backtest_run_id)

    if model in tree_scripts:
        cmd = [_UV, "run", "python", tree_scripts[model]]
    elif model in foundation_modules:
        cmd = [_UV, "run", "python", "-m", foundation_modules[model]]
    elif model in special_scripts:
        entry = special_scripts[model]
        if isinstance(entry, tuple):
            script, extra_args = entry
            cmd = [_UV, "run", "python", script] + extra_args
        else:
            cmd = [_UV, "run", "python", entry]
    else:
        raise ValueError(f"Unknown backtest model: {model}")

    config_path = params.get("config")
    if config_path:
        cmd.extend(["--config", config_path])
    if params.get("resume"):
        cmd.append("--resume")

    try:
        output = _run_subprocess(cmd, progress_cb, cancel_event=cancel_event, job_id=job_id)
    except Exception:
        if backtest_run_id:
            try:
                with _get_conn() as conn:
                    conn.execute(
                        "UPDATE backtest_run SET status = 'failed', completed_at = NOW() WHERE id = %s",
                        (backtest_run_id,),
                    )
            except Exception:
                pass
        raise

    if progress_cb:
        progress_cb(pct=100, msg=f"{model} backtest completed")

    # Update backtest_run tracking row with results
    if backtest_run_id:
        _update_backtest_run_on_completion(backtest_run_id, model)

    return {"model": model, "output_log": output if output else "Completed"}


def _make_backtest_runner(model: str):
    """Factory to create a backtest runner for a specific model."""
    def runner(
        params: dict[str, Any],
        progress_cb: Callable | None = None,
        cancel_event: Event | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        return _run_backtest(model, params, progress_cb, cancel_event=cancel_event, job_id=job_id)
    runner.__name__ = f"_run_backtest_{model}"
    return runner


_run_backtest_lgbm = _make_backtest_runner("lgbm")
_run_backtest_catboost = _make_backtest_runner("catboost")
_run_backtest_xgboost = _make_backtest_runner("xgboost")
_run_backtest_chronos = _make_backtest_runner("chronos")
_run_backtest_chronos_bolt = _make_backtest_runner("chronos_bolt")
_run_backtest_chronos2 = _make_backtest_runner("chronos2")
_run_backtest_chronos2_enriched = _make_backtest_runner("chronos2_enriched")
_run_backtest_bolt_hierarchical = _make_backtest_runner("bolt_hierarchical")
_run_backtest_mstl = _make_backtest_runner("mstl")
_run_backtest_seasonal_naive = _make_backtest_runner("seasonal_naive")
_run_backtest_rolling_mean = _make_backtest_runner("rolling_mean")
_run_backtest_nhits = _make_backtest_runner("nhits")
_run_backtest_nbeats = _make_backtest_runner("nbeats")


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


def _run_champion_experiment(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run a champion selection strategy experiment."""
    experiment_id = params["experiment_id"]
    if progress_cb:
        progress_cb(pct=5, msg=f"Starting champion experiment #{experiment_id}")

    # Store job_id on experiment record
    try:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE champion_experiment SET job_id = %s WHERE experiment_id = %s",
                (job_id, experiment_id),
            )
    except Exception:
        logger.warning("Failed to store job_id on champion experiment %d", experiment_id)

    cmd = [_UV, "run", "python", "scripts/run_champion_experiment.py",
           "--experiment-id", str(experiment_id)]
    output = _run_subprocess(cmd, progress_cb, "Running champion experiment",
                             cancel_event=cancel_event, job_id=job_id)
    if progress_cb:
        progress_cb(pct=100, msg=f"Champion experiment #{experiment_id} completed")
    return {"experiment_id": experiment_id, "output_log": output or "Champion experiment completed"}


def _run_champion_results_load(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Load champion results into DB, using cached experiment winners when available.

    If a cached winners CSV exists from the experiment run
    (data/champion/experiment_{id}_winners.csv), the script loads those
    pre-computed winners directly instead of recomputing from scratch.
    Falls back to full computation if no cached file is found.
    """
    experiment_id = params["experiment_id"]
    if progress_cb:
        progress_cb(pct=5, msg="Loading champion results into forecast tables")

    # Check for cached winners CSV from experiment run
    winners_csv = _SCRIPTS_DIR.parent / "data" / "champion" / f"experiment_{experiment_id}_winners.csv"
    cmd = [_UV, "run", "python", "scripts/run_champion_selection.py"]
    if winners_csv.exists():
        cmd.extend(["--load-winners-from", str(winners_csv)])
        logger.info("Using cached winners from experiment %d: %s", experiment_id, winners_csv)
        if progress_cb:
            progress_cb(pct=10, msg="Found cached winners — loading directly (skipping recomputation)")
    else:
        logger.info("No cached winners for experiment %d, running full computation", experiment_id)

    output = _run_subprocess(cmd, progress_cb, "Loading champion results",
                             cancel_event=cancel_event, job_id=job_id)

    # Mark results promoted — clear flag on ALL other experiments first
    # (only one experiment's results can be active in the forecast table at a time)
    try:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE champion_experiment SET is_results_promoted = FALSE "
                "WHERE experiment_id != %s AND is_results_promoted = TRUE",
                (experiment_id,),
            )
            conn.execute(
                "UPDATE champion_experiment SET is_results_promoted = TRUE, "
                "results_promoted_at = NOW(), results_promote_job_id = %s "
                "WHERE experiment_id = %s",
                (job_id, experiment_id),
            )
    except Exception:
        logger.warning("Failed to mark champion experiment %d results as promoted", experiment_id)

    if progress_cb:
        progress_cb(pct=100, msg="Champion results loaded successfully")
    return {"experiment_id": experiment_id, "output_log": output or "Champion results loaded"}


def _run_train_production_model(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Train a production model on full history for forecasting.

    Invokes ``scripts/ml/train_production_models.py`` as a subprocess.
    Supports training a single model (--model) or all tree models (--all).
    """
    ROOT = Path(__file__).resolve().parents[2]

    model_id = params.get("model_id", "")
    all_models = params.get("all_models", False)

    if progress_cb:
        progress_cb(pct=0, msg=f"Starting production training: {model_id or 'all models'}")

    cmd = [_UV, "run", "python", str(ROOT / "scripts" / "ml" / "train_production_models.py")]
    if all_models:
        cmd.append("--all")
    elif model_id:
        cmd.extend(["--model", model_id])

    start = time.time()
    if progress_cb:
        progress_cb(pct=5, msg=f"Training {'all models' if all_models else model_id}")

    _run_subprocess(
        cmd, progress_cb, f"Training production model: {model_id or 'all'}",
        cancel_event=cancel_event, job_id=job_id,
    )
    duration = time.time() - start

    if progress_cb:
        progress_cb(pct=100, msg=f"Training completed in {duration:.0f}s")

    return {
        "model_id": model_id,
        "all_models": all_models,
        "duration_s": round(duration, 1),
        "status": "trained",
    }


def _run_generate_production_forecast(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run the production forecast generation pipeline (F1.1)."""
    horizon = params.get("horizon", 24)
    model_id = params.get("model_id")
    if progress_cb:
        progress_cb(pct=5, msg=f"Starting production forecast generation (horizon={horizon})")
    cmd = [_UV, "run", "python", "scripts/generate_production_forecasts.py", "--horizon", str(horizon)]
    if model_id:
        cmd.extend(["--model-id", str(model_id)])
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


def _run_inventory_backtest(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run inventory backtest simulation."""
    cmd = [_UV, "run", "python", "scripts/run_inventory_backtest.py"]
    if params.get("models"):
        cmd.extend(["--models", params["models"]])
    if params.get("months"):
        cmd.extend(["--months", str(params["months"])])
    output = _run_subprocess(
        cmd, progress_cb, "Inventory backtest",
        cancel_event=cancel_event, job_id=job_id,
    )
    return {"output_log": output or "Inventory backtest completed"}


def _run_inventory_planning_pipeline(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run the end-to-end inventory planning pipeline."""
    steps = params.get("steps")
    cmd = [_UV, "run", "python", "scripts/run_inventory_planning_pipeline.py"]
    if steps:
        cmd.extend(["--steps", steps])
    if progress_cb:
        progress_cb(pct=0, msg="Starting inventory planning pipeline")
    output = _run_subprocess(
        cmd, progress_cb, "Inventory planning pipeline",
        cancel_event=cancel_event, job_id=job_id,
    )
    if progress_cb:
        progress_cb(pct=100, msg="Inventory planning pipeline complete")
    return {"output_log": output or "Pipeline completed"}


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
    if params.get("forecast_source"):
        cmd.extend(["--forecast-source", params["forecast_source"]])
    if params.get("model_id"):
        cmd.extend(["--model-id", params["model_id"]])
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


def _run_compare_inventory_algorithms(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Compare SS/EOQ/ROP across forecast algorithms."""
    if progress_cb:
        progress_cb(pct=10, msg="Comparing inventory algorithms")
    cmd = [_UV, "run", "python", "scripts/compare_inventory_algorithms.py"]
    models = params.get("models")
    if models:
        cmd.extend(["--models", models])
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output or "Algorithm comparison completed"}


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
           "--config", "config/forecast_domain_config.yaml"]
    output = _run_subprocess(cmd, cancel_event=cancel_event, job_id=job_id)
    return {"output_log": output if output else "Demand variability computation completed"}


def _run_compute_sku_features(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Compute all time-series features (volume, trend, seasonality, variability, lifecycle) for all SKUs."""
    from scripts.ml.compute_sku_features import run_pipeline

    if progress_cb:
        progress_cb(pct=10, msg="Computing SKU features")
    result = run_pipeline(time_window_months=params.get("time_window_months", 36))
    if progress_cb:
        progress_cb(pct=100, msg="Complete")
    return result


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

    # 1. Build temp config with overrides (pipeline config format)
    from common.utils import get_pipeline_config_path
    pipeline_path = get_pipeline_config_path()
    with open(pipeline_path) as f:
        base_config = yaml.safe_load(f)

    cfg = copy.deepcopy(base_config)
    # Apply overrides to lgbm_cluster params in pipeline config
    lgbm_entry = cfg.get("algorithms", {}).get("lgbm_cluster", {})
    if "params" in lgbm_entry:
        lgbm_entry["params"].update(overrides)
    else:
        lgbm_entry.update(overrides)

    tmp_dir = Path(tempfile.mkdtemp(prefix="tuning_chat_"))
    tmp_path = tmp_dir / f"pipeline_config_{strategy_label}.yaml"
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


def _run_model_tuning_experiment(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run a model tuning experiment: update run status, run backtest, register results.

    Supports lgbm, catboost, and xgboost models. The caller provides a pre-built
    temp config file and the run_id of the lgbm_tuning_run record.
    """
    ROOT = Path(__file__).resolve().parents[2]

    run_id = params["run_id"]
    model = params["model"]
    config_path = params["config_path"]
    run_label = params.get("run_label", "tuning_experiment")

    if model not in MODEL_OUTPUT_DIRS:
        raise ValueError(f"Unknown model type: {model}")

    if progress_cb:
        progress_cb(pct=0, msg=f"Starting {model.upper()} tuning experiment #{run_id} ({run_label})")

    # 1. Update lgbm_tuning_run: set job_id, status=running, started_at
    try:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE lgbm_tuning_run SET job_id = %s, status = 'running', started_at = NOW() "
                "WHERE run_id = %s",
                (job_id, run_id),
            )
    except Exception:
        logger.warning("Failed to update run %d status to running", run_id)

    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Job cancelled by user")

    # 2. Build backtest command — optionally include cluster override
    cmd = [
        _UV, "run", "python", str(ROOT / "scripts" / "run_backtest.py"),
        "--model", model,
        "--config", config_path,
    ]

    # If a cluster experiment is referenced, look up its artifacts and add --cluster-override
    cluster_experiment_id: int | None = params.get("cluster_experiment_id")
    if cluster_experiment_id is not None:
        try:
            with _get_conn() as conn:
                row = conn.execute(
                    "SELECT artifacts_path, scenario_id, status FROM cluster_experiment "
                    "WHERE experiment_id = %s",
                    (cluster_experiment_id,),
                ).fetchone()
            if row is None:
                raise ValueError(
                    f"Cluster experiment {cluster_experiment_id} not found"
                )
            ce_artifacts_path, ce_scenario_id, ce_status = row
            if ce_status != "completed":
                raise ValueError(
                    f"Cluster experiment {cluster_experiment_id} is not completed "
                    f"(status={ce_status})"
                )
            if not ce_artifacts_path:
                raise ValueError(
                    f"Cluster experiment {cluster_experiment_id} has no artifacts_path"
                )
            safe_path = Path(ce_artifacts_path).resolve()
            allowed_base = Path(__file__).resolve().parents[2] / "data"
            if not str(safe_path).startswith(str(allowed_base)):
                raise ValueError(
                    f"artifacts_path {ce_artifacts_path!r} is outside the allowed data directory"
                )
            cluster_override_csv = str(safe_path / "cluster_labels.csv")
            cmd.extend(["--cluster-override", cluster_override_csv])
            if progress_cb:
                progress_cb(
                    pct=3,
                    msg=f"Using clusters from experiment #{cluster_experiment_id} "
                        f"(scenario {ce_scenario_id})",
                )
        except ValueError:
            raise
        except Exception:
            logger.warning(
                "Failed to look up cluster experiment %d — proceeding with production clusters",
                cluster_experiment_id,
            )

    start = time.time()
    try:
        if progress_cb:
            progress_cb(pct=5, msg=f"Running {model.upper()} backtest")
        output = _run_subprocess(cmd, progress_cb, f"Running {model.upper()} backtest",
                                 cancel_event=cancel_event, job_id=job_id)
        duration = time.time() - start
    except (RuntimeError, OSError) as exc:
        # 5. On failure: update lgbm_tuning_run with status=failed
        duration = time.time() - start
        error_msg = str(exc)[:2000]
        try:
            from common.ml.tuning_tracker import fail_run
            fail_run(run_id, error_msg)
        except ImportError:
            logger.warning("tuning_tracker not available — marking run %d failed via direct SQL", run_id)
            try:
                with _get_conn() as conn:
                    conn.execute(
                        "UPDATE lgbm_tuning_run SET status = 'failed', completed_at = NOW(), "
                        "notes = COALESCE(notes || E'\\n', '') || %s WHERE run_id = %s",
                        (error_msg, run_id),
                    )
            except Exception:
                logger.warning("Failed to mark run %d as failed in DB", run_id)
        # 6. Clean up temp config file
        _cleanup_temp_config(config_path)
        raise

    # 3. On success: complete run via tracker
    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Job cancelled by user")

    if progress_cb:
        progress_cb(pct=90, msg="Registering results")

    from common.ml.tuning_tracker import (
        complete_run,
        register_cluster_month_breakdowns,
        register_timeframes,
    )

    output_dir_name = MODEL_OUTPUT_DIRS[model]
    meta_path = ROOT / "data" / "backtest" / output_dir_name / "backtest_metadata.json"
    complete_run(run_id, meta_path)

    if progress_cb:
        progress_cb(pct=93, msg="Registering timeframe breakdowns")
    register_timeframes(run_id, meta_path)

    predictions_path = ROOT / "data" / "backtest" / output_dir_name / "backtest_predictions.csv"
    if predictions_path.exists():
        if progress_cb:
            progress_cb(pct=96, msg="Registering cluster/month breakdowns")
        register_cluster_month_breakdowns(run_id, predictions_path)

    # 6. Clean up temp config file
    _cleanup_temp_config(config_path)

    if progress_cb:
        progress_cb(pct=100, msg=f"Tuning experiment #{run_id} completed in {duration:.0f}s")
    return {
        "run_id": run_id,
        "model": model,
        "run_label": run_label,
        "duration_seconds": round(duration),
        "output_log": output[:5000] if output else "Model tuning experiment completed",
    }


def _run_load_backtest_results(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Load backtest predictions into DB and refresh materialized views.

    Invokes ``scripts/load_backtest_forecasts.py --model <model_id> --replace``
    as a subprocess. On success, marks the tuning run as results-promoted.
    """
    ROOT = Path(__file__).resolve().parents[2]

    run_id = params["run_id"]
    model = params["model"]
    model_id = MODEL_OUTPUT_DIRS.get(model, params.get("model_id", ""))

    if progress_cb:
        progress_cb(pct=0, msg=f"Starting results load for {model.upper()}")

    pred_path = ROOT / "data" / "backtest" / model_id / "backtest_predictions.csv"
    if not pred_path.exists():
        raise RuntimeError(f"Prediction file not found: {pred_path}")

    cmd = [
        _UV, "run", "python",
        str(ROOT / "scripts" / "load_backtest_forecasts.py"),
        "--model", model_id, "--replace",
    ]
    start = time.time()
    if progress_cb:
        progress_cb(pct=5, msg=f"Loading {model.upper()} predictions into database")

    output = _run_subprocess(
        cmd, progress_cb, f"Loading {model.upper()} results",
        cancel_event=cancel_event, job_id=job_id,
    )
    duration = time.time() - start

    # Mark tuning run as results-promoted
    if progress_cb:
        progress_cb(pct=95, msg="Updating promotion status")
    try:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE lgbm_tuning_run "
                "SET is_results_promoted = TRUE, results_promoted_at = NOW() "
                "WHERE run_id = %s",
                (run_id,),
            )
            conn.execute(
                "INSERT INTO tuning_promotion_log "
                "(run_id, model_id, promoted_by, params_written, promotion_type) "
                "VALUES (%s, %s, %s, %s::jsonb, %s)",
                (run_id, model_id, "manual", "{}", "results"),
            )
    except Exception:
        logger.warning("Failed to update results promotion status for run %d", run_id)

    if progress_cb:
        progress_cb(pct=100, msg=f"Results loaded in {duration:.0f}s")

    return {
        "run_id": run_id,
        "model": model,
        "duration_seconds": round(duration),
        "status": "loaded",
    }


def _run_load_backtest_model(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Load backtest predictions for a specific model into Postgres.

    Invokes ``scripts/load_backtest_forecasts.py --model <model_id> --replace``
    as a subprocess.  If ``run_id`` is provided, marks the corresponding
    ``backtest_run`` row as loaded on success.
    """
    ROOT = Path(__file__).resolve().parents[2]

    model_id = params["model_id"]
    run_id = params.get("run_id")

    if progress_cb:
        progress_cb(pct=0, msg=f"Starting results load for {model_id}")

    pred_path = ROOT / "data" / "backtest" / model_id / "backtest_predictions.csv"
    if not pred_path.exists():
        raise RuntimeError(f"Prediction file not found: {pred_path}")

    cmd = [
        _UV, "run", "python",
        str(ROOT / "scripts" / "load_backtest_forecasts.py"),
        "--model", model_id, "--replace",
    ]
    start = time.time()
    if progress_cb:
        progress_cb(pct=5, msg=f"Loading {model_id} predictions into database")

    _run_subprocess(
        cmd, progress_cb, f"Loading {model_id} results",
        cancel_event=cancel_event, job_id=job_id,
    )
    duration = time.time() - start

    # Mark backtest_run as loaded if run_id provided
    if run_id is not None:
        if progress_cb:
            progress_cb(pct=95, msg="Updating load status in backtest_run")
        try:
            with _get_conn() as conn:
                conn.execute(
                    "UPDATE backtest_run "
                    "SET is_loaded_to_db = TRUE, loaded_at = NOW() "
                    "WHERE id = %s",
                    (run_id,),
                )
        except Exception:
            logger.warning("Failed to update backtest_run %s as loaded", run_id)

    if progress_cb:
        progress_cb(pct=100, msg=f"Results loaded in {duration:.0f}s")

    return {
        "model_id": model_id,
        "run_id": run_id,
        "duration_seconds": round(duration),
        "status": "loaded",
    }


def _cleanup_temp_config(config_path: str) -> None:
    """Remove a temporary config file and its parent directory if empty."""
    try:
        p = Path(config_path)
        if p.exists():
            p.unlink()
        parent = p.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        logger.warning("Failed to clean up temp config: %s", config_path)


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
