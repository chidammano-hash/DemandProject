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
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
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


def _run_generate_ai_insights(params: dict[str, Any], progress_cb: Callable | None = None) -> dict[str, Any]:
    """Run AI Planning Agent portfolio scan to generate insights."""
    if progress_cb:
        progress_cb(pct=5, msg="Starting AI insights generation")
    cmd = [_UV, "run", "python", "scripts/generate_ai_insights.py", "--portfolio"]
    output = _run_subprocess(cmd, progress_cb, "Scanning portfolio for exceptions")
    if progress_cb:
        progress_cb(pct=100, msg="AI insights generation complete")
    return {"output_summary": output[:500] if output else "AI insights generation completed"}


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
