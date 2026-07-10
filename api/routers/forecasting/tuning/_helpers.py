"""Shared helpers, constants, and Pydantic models for the unified model-tuning router.

This module is the single source of truth for the constants and helper functions
that the split sub-routers (list, detail, create, lag, cluster, month, logs,
compare, promote, cancel_delete, templates, promotions) all depend on.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException
from pydantic import BaseModel, Field

from api.core import get_conn
from common.core.sql_helpers import parse_db_json as _parse_json
from common.core.utils import get_pipeline_config_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_MODELS = {"lgbm"}

MODEL_ID_MAP: dict[str, str] = {
    "lgbm": "lgbm_cluster",
}

# Model-specific hyperparameter keys recognized during promotion
_MODEL_PARAM_KEYS: dict[str, set[str]] = {
    "lgbm": {
        "n_estimators", "learning_rate", "num_leaves", "min_child_samples",
        "max_depth", "min_gain_to_split", "subsample", "bagging_freq",
        "colsample_bytree", "feature_fraction_bynode", "reg_lambda", "reg_alpha",
        "path_smooth", "max_bin",
    },
}

# Config keys for comparison (non-hyperparameter settings from metadata)
_CONFIG_KEYS = [
    "cluster_strategy", "recursive", "shap_select", "shap_threshold",
    "shap_top_n", "shap_sample_size", "tune_inline", "params_source",
    "cluster_source", "cluster_experiment_id",
]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CreateExperimentBody(BaseModel):
    """Request body for POST /{model}/experiments.

    The ``config`` dict supports these keys:
    - cluster_strategy, recursive, shap_select, shap_threshold, shap_sample_size
    - cluster_source: "production" (default) or "experimental"
    - cluster_experiment_id: int — FK to cluster_experiment (required when
      cluster_source is "experimental")
    """
    run_label: str = Field(min_length=1, max_length=200)
    notes: str | None = None
    template: str | None = None
    params: dict[str, Any] | None = None
    config: dict[str, Any] | None = None


class UpdateExperimentBody(BaseModel):
    """Request body for PATCH updates to an experiment run."""
    status: str | None = Field(
        default=None,
        pattern=r"^(queued|running|completed|failed|cancelled)$",
    )
    accuracy_pct: float | None = None
    wape: float | None = None
    bias: float | None = None
    notes: str | None = None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_model(model: str) -> str:
    """Validate and return the model string. Raises 400 on unknown model."""
    if model not in VALID_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model}'. Must be one of: {', '.join(sorted(VALID_MODELS))}",
        )
    return model


def _model_id(model: str) -> str:
    """Return the DB model_id for a validated model name."""
    return MODEL_ID_MAP[model]




def _list_row_to_dict(r: tuple) -> dict[str, Any]:
    """Convert a list-query row tuple (22 columns) to a response dict."""
    return {
        "run_id": r[0],
        "run_label": r[1],
        "model_id": r[2],
        "started_at": str(r[3]) if r[3] else None,
        "completed_at": str(r[4]) if r[4] else None,
        "status": r[5],
        "accuracy_pct": float(r[6]) if r[6] is not None else None,
        "wape": float(r[7]) if r[7] is not None else None,
        "bias": float(r[8]) if r[8] is not None else None,
        "n_predictions": int(r[9]) if r[9] is not None else None,
        "n_dfus": int(r[10]) if r[10] is not None else None,
        "notes": r[11],
        "is_promoted": bool(r[12]),
        "promoted_at": str(r[13]) if r[13] else None,
        "job_id": r[14],
        "template_id": r[15],
        "is_results_promoted": bool(r[16]),
        "results_promoted_at": str(r[17]) if r[17] else None,
        "results_promote_job_id": r[18],
        "cluster_source": r[19] or "production",
        "cluster_experiment_id": int(r[20]) if r[20] is not None else None,
        "cluster_experiment_label": r[21],
    }


def _detail_row_to_dict(r: tuple) -> dict[str, Any]:
    """Convert a detail-query row tuple (24 columns) to a response dict."""
    return {
        "run_id": r[0],
        "run_label": r[1],
        "model_id": r[2],
        "started_at": str(r[3]) if r[3] else None,
        "completed_at": str(r[4]) if r[4] else None,
        "status": r[5],
        "params": _parse_json(r[6]),
        "feature_count": r[7],
        "features": _parse_json(r[8]),
        "accuracy_pct": float(r[9]) if r[9] is not None else None,
        "wape": float(r[10]) if r[10] is not None else None,
        "bias": float(r[11]) if r[11] is not None else None,
        "n_predictions": int(r[12]) if r[12] is not None else None,
        "n_dfus": int(r[13]) if r[13] is not None else None,
        "metadata": _parse_json(r[14]),
        "notes": r[15],
        "backup_path": r[16],
        "job_id": r[17],
        "template_id": r[18],
        "is_promoted": bool(r[19]),
        "promoted_at": str(r[20]) if r[20] else None,
        "is_results_promoted": bool(r[21]),
        "results_promoted_at": str(r[22]) if r[22] else None,
        "results_promote_job_id": r[23],
    }


def _compare_row_to_dict(r: tuple) -> dict[str, Any]:
    """Convert a compare-query row tuple (16 columns) to a response dict."""
    d: dict[str, Any] = {
        "run_id": r[0],
        "run_label": r[1],
        "model_id": r[2],
        "accuracy_pct": float(r[3]) if r[3] is not None else None,
        "wape": float(r[4]) if r[4] is not None else None,
        "bias": float(r[5]) if r[5] is not None else None,
        "n_predictions": int(r[6]) if r[6] is not None else None,
        "n_dfus": int(r[7]) if r[7] is not None else None,
        "status": r[8],
        "params": _parse_json(r[9]),
        "features": _parse_json(r[10]),
        "feature_count": int(r[11]) if r[11] is not None else None,
        "metadata": _parse_json(r[12]),
    }
    # Cluster source fields (columns 13-15, present in compare queries)
    if len(r) > 13:
        d["cluster_source"] = r[13] or "production"
        d["cluster_experiment_id"] = int(r[14]) if r[14] is not None else None
        d["cluster_experiment_label"] = r[15] if len(r) > 15 else None
    return d


def _verify_run_ownership(run_id: int, model: str) -> None:
    """Verify that a run exists and belongs to the specified model. Raises 404 if not."""
    import psycopg

    mid = _model_id(model)
    sql = "SELECT 1 FROM lgbm_tuning_run WHERE run_id = %s AND model_id = %s"
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, [run_id, mid])
            if cur.fetchone() is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Experiment {run_id} not found for model {model}",
                )
    except HTTPException:
        raise
    except psycopg.Error:
        logger.exception("Failed to verify run ownership for %d/%s", run_id, model)
        raise HTTPException(status_code=500, detail="Failed to verify experiment")


def _build_temp_config(
    model: str,
    params: dict[str, Any] | None,
    config: dict[str, Any] | None,
) -> Path:
    """Create a temporary forecast_pipeline_config.yaml with experiment overrides.

    Returns the path to the temp config file.
    """
    try:
        with open(get_pipeline_config_path()) as f:
            cfg = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        logger.exception("Failed to read forecast_pipeline_config.yaml for temp config")
        raise HTTPException(status_code=500, detail="Failed to read config")

    pipeline_key = MODEL_ID_MAP[model]
    entry = cfg.get("algorithms", {}).get(pipeline_key, {})
    params_section = entry.setdefault("params", {})

    # Apply hyperparameter overrides into the params sub-dict
    if params:
        for key, value in params.items():
            params_section[key] = value

    # Apply config overrides (cluster_strategy, recursive, etc.) at entry level
    if config:
        for key, value in config.items():
            entry[key] = value

    cfg.setdefault("algorithms", {})[pipeline_key] = entry

    # Write to a temp directory
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"tuning_{model}_"))
    tmp_path = tmp_dir / "forecast_pipeline_config.yaml"
    with open(tmp_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return tmp_path
