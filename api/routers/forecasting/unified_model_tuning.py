"""Unified Model Tuning router — handles LGBM, CatBoost, and XGBoost experiments.

All 3 model types share a single parametrized router at /{model}/*.
Supports experiment lifecycle (create, launch, cancel, delete), comparison with
execution-lag filtering, promotion to production, and template loading.
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel, Field

from api.core import get_conn, set_cache

logger = logging.getLogger(__name__)

router = APIRouter(tags=["model-tuning"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_MODELS = {"lgbm", "catboost", "xgboost"}

MODEL_ID_MAP: dict[str, str] = {
    "lgbm": "lgbm_cluster",
    "catboost": "catboost_cluster",
    "xgboost": "xgboost_cluster",
}

_ALGO_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "algorithm_config.yaml"

# CatBoost synonym pairs — only one of each pair can be set at a time.
# When both appear, keep the first (CatBoost-native) name and drop the alias.
_CATBOOST_SYNONYM_PAIRS: list[tuple[str, str]] = [
    ("l2_leaf_reg", "reg_lambda"),
    ("iterations", "num_boost_round"),
    ("iterations", "n_estimators"),
    ("iterations", "num_trees"),
    ("learning_rate", "eta"),
    ("depth", "max_depth"),
    ("random_seed", "random_state"),
]

# Model-specific hyperparameter keys recognized during promotion
_MODEL_PARAM_KEYS: dict[str, set[str]] = {
    "lgbm": {
        "n_estimators", "learning_rate", "num_leaves", "min_child_samples",
        "max_depth", "min_gain_to_split", "subsample", "bagging_freq",
        "colsample_bytree", "feature_fraction_bynode", "reg_lambda", "reg_alpha",
        "path_smooth", "max_bin",
    },
    "catboost": {
        "iterations", "learning_rate", "depth", "l2_leaf_reg",
        "border_count", "bagging_temperature", "random_strength",
        "min_data_in_leaf", "grow_policy", "max_bin", "max_leaves",
        "max_ctr_complexity", "subsample", "colsample_bylevel",
        "bootstrap_type", "model_size_reg", "leaf_estimation_method",
        "boost_from_average", "leaf_estimation_iterations", "score_function",
        "posterior_sampling", "langevin", "diffusion_temperature",
    },
    "xgboost": {
        "n_estimators", "learning_rate", "max_depth", "min_child_weight",
        "subsample", "colsample_bytree", "colsample_bylevel",
        "reg_lambda", "reg_alpha", "gamma", "max_bin",
        "grow_policy", "tree_method", "max_leaves",
        "booster", "rate_drop", "skip_drop", "colsample_bynode",
    },
}

# algorithm_config.yaml section name per model
_ALGO_SECTION: dict[str, str] = {
    "lgbm": "lgbm",
    "catboost": "catboost",
    "xgboost": "xgboost",
}

# Config keys for comparison (non-hyperparameter settings from metadata)
_CONFIG_KEYS = [
    "cluster_strategy", "recursive", "shap_select", "shap_threshold",
    "shap_top_n", "shap_sample_size", "tune_inline", "params_source",
    "cluster_source", "cluster_experiment_id",
]

# Script path per model for backtest launch
_BACKTEST_SCRIPT: dict[str, str] = {
    "lgbm": "scripts/run_backtest.py",
    "catboost": "scripts/run_backtest.py",
    "xgboost": "scripts/run_backtest.py",
}


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


def _parse_json(val: Any) -> Any:
    """Parse JSON from a DB value that may be a string, dict, list, or None."""
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    return json.loads(val)


def _run_row_to_dict(r: tuple) -> dict[str, Any]:
    """Convert a run row tuple (16 columns) to a response dict."""
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


# ---------------------------------------------------------------------------
# 1. GET /{model}/experiments — List experiments
# ---------------------------------------------------------------------------

@router.get("/{model}/experiments")
def list_experiments(
    model: str,
    response: FastAPIResponse,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    status: str = Query(default="", max_length=20),
    exec_lag: int | None = Query(default=None, ge=0, le=4),
):
    """List tuning experiments for a model, newest first."""
    _validate_model(model)
    set_cache(response, max_age=30)

    mid = _model_id(model)
    offset = (page - 1) * page_size

    parts: list[str] = ["r.model_id = %s"]
    params: list[Any] = [mid]
    if status.strip():
        parts.append("r.status = %s")
        params.append(status.strip())

    where_sql = f"WHERE {' AND '.join(parts)}"

    # Count total for pagination
    count_sql = f"SELECT count(*) FROM lgbm_tuning_run r {where_sql}"

    sql = f"""
        SELECT r.run_id, r.run_label, r.model_id, r.started_at, r.completed_at,
               r.status, r.accuracy_pct, r.wape, r.bias, r.n_predictions, r.n_dfus, r.notes,
               r.is_promoted, r.promoted_at, r.job_id, r.template_id,
               r.is_results_promoted, r.results_promoted_at, r.results_promote_job_id,
               r.cluster_source, r.cluster_experiment_id,
               ce.label AS cluster_experiment_label
        FROM lgbm_tuning_run r
        LEFT JOIN cluster_experiment ce ON ce.experiment_id = r.cluster_experiment_id
        {where_sql}
        ORDER BY r.started_at DESC
        LIMIT %s OFFSET %s
    """

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(count_sql, list(params))
            total = cur.fetchone()[0]

            cur.execute(sql, [*params, page_size, offset])
            rows = cur.fetchall()

            # If exec_lag is specified, fetch lag-level metrics for these run_ids
            lag_metrics: dict[int, dict[str, Any]] = {}
            if exec_lag is not None and rows:
                run_ids = [r[0] for r in rows]
                placeholders = ", ".join(["%s"] * len(run_ids))
                lag_sql = f"""
                    SELECT run_id, accuracy_pct, wape, bias, n_predictions
                    FROM lgbm_tuning_lag
                    WHERE run_id IN ({placeholders}) AND exec_lag = %s
                """
                cur.execute(lag_sql, [*run_ids, exec_lag])
                for lr in cur.fetchall():
                    lag_metrics[lr[0]] = {
                        "accuracy_pct": float(lr[1]) if lr[1] is not None else None,
                        "wape": float(lr[2]) if lr[2] is not None else None,
                        "bias": float(lr[3]) if lr[3] is not None else None,
                        "n_predictions": int(lr[4]) if lr[4] is not None else None,
                    }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to list %s tuning experiments", model)
        raise HTTPException(status_code=500, detail=f"Failed to list {model} tuning experiments")

    experiments = []
    for r in rows:
        run_id = r[0]
        entry: dict[str, Any] = {
            "run_id": run_id,
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
        # Override with lag-specific metrics when filtering by exec_lag
        if exec_lag is not None and run_id in lag_metrics:
            lm = lag_metrics[run_id]
            entry["accuracy_pct"] = lm["accuracy_pct"]
            entry["wape"] = lm["wape"]
            entry["bias"] = lm["bias"]
            if lm["n_predictions"] is not None:
                entry["n_predictions"] = lm["n_predictions"]
            entry["exec_lag_filter"] = exec_lag

        experiments.append(entry)

    return {
        "experiments": experiments,
        "total": total,
        "page": page,
        "page_size": page_size,
        "model": model,
    }


# ---------------------------------------------------------------------------
# 2. GET /{model}/experiments/{run_id} — Single experiment detail
# ---------------------------------------------------------------------------

@router.get("/{model}/experiments/{run_id}")
def get_experiment(model: str, run_id: int, response: FastAPIResponse):
    """Get full detail for a single experiment, including timeframe breakdowns."""
    _validate_model(model)
    set_cache(response, max_age=30)
    mid = _model_id(model)

    run_sql = """
        SELECT run_id, run_label, model_id, started_at, completed_at,
               status, params, feature_count, features,
               accuracy_pct, wape, bias, n_predictions, n_dfus,
               metadata, notes, backup_path, job_id, template_id,
               is_promoted, promoted_at,
               is_results_promoted, results_promoted_at, results_promote_job_id
        FROM lgbm_tuning_run
        WHERE run_id = %s AND model_id = %s
    """
    tf_sql = """
        SELECT id, run_id, timeframe, train_end, predict_start, predict_end,
               n_predictions, accuracy_pct, wape, bias
        FROM lgbm_tuning_timeframe
        WHERE run_id = %s
        ORDER BY timeframe
    """

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(run_sql, [run_id, mid])
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Experiment not found")

            cur.execute(tf_sql, [run_id])
            tf_rows = cur.fetchall()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to get experiment %d for %s", run_id, model)
        raise HTTPException(status_code=500, detail="Failed to fetch experiment")

    run = {
        "run_id": row[0],
        "run_label": row[1],
        "model_id": row[2],
        "started_at": str(row[3]) if row[3] else None,
        "completed_at": str(row[4]) if row[4] else None,
        "status": row[5],
        "params": _parse_json(row[6]),
        "feature_count": row[7],
        "features": _parse_json(row[8]),
        "accuracy_pct": float(row[9]) if row[9] is not None else None,
        "wape": float(row[10]) if row[10] is not None else None,
        "bias": float(row[11]) if row[11] is not None else None,
        "n_predictions": int(row[12]) if row[12] is not None else None,
        "n_dfus": int(row[13]) if row[13] is not None else None,
        "metadata": _parse_json(row[14]),
        "notes": row[15],
        "backup_path": row[16],
        "job_id": row[17],
        "template_id": row[18],
        "is_promoted": bool(row[19]),
        "promoted_at": str(row[20]) if row[20] else None,
        "is_results_promoted": bool(row[21]),
        "results_promoted_at": str(row[22]) if row[22] else None,
        "results_promote_job_id": row[23],
    }

    timeframes = []
    for tf in tf_rows:
        timeframes.append({
            "id": tf[0],
            "run_id": tf[1],
            "timeframe": tf[2],
            "train_end": str(tf[3]) if tf[3] else None,
            "predict_start": str(tf[4]) if tf[4] else None,
            "predict_end": str(tf[5]) if tf[5] else None,
            "n_predictions": int(tf[6]) if tf[6] is not None else None,
            "accuracy_pct": float(tf[7]) if tf[7] is not None else None,
            "wape": float(tf[8]) if tf[8] is not None else None,
            "bias": float(tf[9]) if tf[9] is not None else None,
        })

    return {**run, "model": model, "timeframes": timeframes}


# ---------------------------------------------------------------------------
# 3. POST /{model}/experiments — Create + launch experiment
# ---------------------------------------------------------------------------

@router.post("/{model}/experiments", status_code=201)
def create_experiment(model: str, body: CreateExperimentBody):
    """Create a new tuning experiment and launch it as a background job."""
    _validate_model(model)
    mid = _model_id(model)

    params_json = json.dumps(body.params) if body.params else None

    # Extract cluster source settings from config
    cfg = body.config or {}
    cluster_source = cfg.get("cluster_source", "production")
    cluster_experiment_id = cfg.get("cluster_experiment_id")

    if cluster_source not in ("production", "experimental"):
        raise HTTPException(
            status_code=400,
            detail="cluster_source must be 'production' or 'experimental'",
        )
    if cluster_source == "experimental" and cluster_experiment_id is None:
        raise HTTPException(
            status_code=400,
            detail="cluster_experiment_id is required when cluster_source is 'experimental'",
        )

    # Validate cluster experiment if experimental
    cluster_override_path: str | None = None
    if cluster_source == "experimental":
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT artifacts_path, scenario_id, label FROM cluster_experiment "
                    "WHERE experiment_id = %s AND status = 'completed'",
                    [cluster_experiment_id],
                )
                ce_row = cur.fetchone()
        except HTTPException:
            raise
        except Exception:
            logger.exception("Failed to validate cluster_experiment_id %s", cluster_experiment_id)
            raise HTTPException(
                status_code=500,
                detail="Failed to validate cluster experiment",
            )

        if ce_row is None:
            raise HTTPException(
                status_code=400,
                detail=f"Cluster experiment {cluster_experiment_id} not found or not completed",
            )
        artifacts_path = ce_row[0]
        cluster_override_path = f"{artifacts_path}/cluster_labels.csv"

    config_json = json.dumps(body.config) if body.config else None

    # Insert the run record with status='queued'
    insert_sql = """
        INSERT INTO lgbm_tuning_run
            (run_label, model_id, params, notes, status, template_id, metadata,
             cluster_source, cluster_experiment_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING run_id
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(insert_sql, [
                body.run_label,
                mid,
                params_json,
                body.notes,
                "queued",
                body.template,
                config_json,
                cluster_source,
                cluster_experiment_id,
            ])
            row = cur.fetchone()
            conn.commit()
    except Exception:
        logger.exception("Failed to create %s experiment", model)
        raise HTTPException(status_code=500, detail=f"Failed to create {model} experiment")

    new_run_id = row[0]

    # Build temporary algorithm_config.yaml with overrides
    effective_config = dict(cfg)
    if cluster_override_path:
        effective_config["cluster_override_path"] = cluster_override_path
    config_path = _build_temp_config(model, body.params, effective_config)

    # Submit job via JobManager
    job_id: str | None = None
    try:
        from common.services.job_registry import JobManager
        mgr = JobManager()
        job_id = mgr.submit_job(
            job_type="model_tuning_run",
            params={
                "run_id": new_run_id,
                "model": model,
                "config_path": str(config_path),
                "run_label": body.run_label,
            },
            label=f"{model.upper()} Tuning — {body.run_label}",
            triggered_by="manual",
            group_override=f"tuning_{model}",
        )
        # Record job_id on the run
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE lgbm_tuning_run SET job_id = %s WHERE run_id = %s",
                [job_id, new_run_id],
            )
            conn.commit()
    except ValueError as exc:
        logger.warning("Job submission failed for run %d: %s", new_run_id, exc)
    except Exception:
        logger.exception("Failed to submit job for %s experiment %d", model, new_run_id)

    return {
        "run_id": new_run_id,
        "job_id": job_id,
        "status": "queued",
        "model": model,
        "run_label": body.run_label,
        "started_at": None,
        "message": (
            f"Experiment queued. Track progress in Jobs tab or via "
            f"GET /{model}/experiments/{new_run_id}/logs"
        ),
    }


def _sanitize_catboost_synonyms(section: dict[str, Any]) -> dict[str, Any]:
    """Remove CatBoost synonym conflicts — keep the native name, drop the alias."""
    for native, alias in _CATBOOST_SYNONYM_PAIRS:
        if native in section and alias in section:
            logger.info(
                "CatBoost synonym conflict: both '%s' and '%s' present — keeping '%s'",
                native, alias, native,
            )
            del section[alias]
    return section


def _build_temp_config(
    model: str,
    params: dict[str, Any] | None,
    config: dict[str, Any] | None,
) -> Path:
    """Create a temporary algorithm_config.yaml with experiment overrides.

    Returns the path to the temp config file.
    """
    try:
        with open(_ALGO_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        logger.exception("Failed to read algorithm_config.yaml for temp config")
        raise HTTPException(status_code=500, detail=f"Failed to read config: {exc}")

    algo_section = _ALGO_SECTION[model]

    # Apply hyperparameter overrides
    if params:
        section = cfg.get("algorithms", {}).get(algo_section, {})
        for key, value in params.items():
            section[key] = value
        cfg.setdefault("algorithms", {})[algo_section] = section

    # Apply config overrides (cluster_strategy, recursive, etc.)
    if config:
        section = cfg.get("algorithms", {}).get(algo_section, {})
        for key, value in config.items():
            section[key] = value
        cfg.setdefault("algorithms", {})[algo_section] = section

    # Strip CatBoost synonym conflicts before writing
    if model == "catboost":
        section = cfg.get("algorithms", {}).get(algo_section, {})
        cfg["algorithms"][algo_section] = _sanitize_catboost_synonyms(section)

    # Write to a temp directory
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"tuning_{model}_"))
    tmp_path = tmp_dir / "algorithm_config.yaml"
    with open(tmp_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return tmp_path


# ---------------------------------------------------------------------------
# 4. GET /{model}/experiments/{run_id}/lags — Per-lag accuracy
# ---------------------------------------------------------------------------

@router.get("/{model}/experiments/{run_id}/lags")
def get_experiment_lags(model: str, run_id: int, response: FastAPIResponse):
    """Get per-execution-lag accuracy breakdown (5 rows: lag 0-4)."""
    _validate_model(model)
    set_cache(response, max_age=60)

    # Verify run exists and belongs to this model
    _verify_run_ownership(run_id, model)

    sql = """
        SELECT exec_lag, n_predictions, n_dfus, accuracy_pct, wape, bias
        FROM lgbm_tuning_lag
        WHERE run_id = %s
        ORDER BY exec_lag
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, [run_id])
            rows = cur.fetchall()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to get lag data for run %d", run_id)
        raise HTTPException(status_code=500, detail="Failed to fetch lag data")

    lags = [
        {
            "exec_lag": r[0],
            "n_predictions": int(r[1]) if r[1] is not None else 0,
            "n_dfus": int(r[2]) if r[2] is not None else 0,
            "accuracy_pct": float(r[3]) if r[3] is not None else None,
            "wape": float(r[4]) if r[4] is not None else None,
            "bias": float(r[5]) if r[5] is not None else None,
        }
        for r in rows
    ]

    return {"run_id": run_id, "model": model, "lags": lags}


# ---------------------------------------------------------------------------
# 5. GET /{model}/experiments/{run_id}/clusters — Per-cluster
# ---------------------------------------------------------------------------

@router.get("/{model}/experiments/{run_id}/clusters")
def get_experiment_clusters(
    model: str,
    run_id: int,
    response: FastAPIResponse,
    exec_lag: int | None = Query(default=None, ge=0, le=4),
):
    """Get per-cluster accuracy breakdowns, optionally filtered by execution lag."""
    _validate_model(model)
    set_cache(response, max_age=60)
    _verify_run_ownership(run_id, model)

    if exec_lag is not None:
        # Use lag-cluster table for lag-specific breakdown
        sql = """
            SELECT cluster_type, cluster_value, n_predictions,
                   accuracy_pct, wape, bias
            FROM lgbm_tuning_lag_cluster
            WHERE run_id = %s AND exec_lag = %s
            ORDER BY cluster_type, accuracy_pct DESC NULLS LAST
        """
        query_params: list[Any] = [run_id, exec_lag]
    else:
        sql = """
            SELECT cluster_type, cluster_value, n_predictions, n_dfus,
                   accuracy_pct, wape, bias
            FROM lgbm_tuning_cluster
            WHERE run_id = %s
            ORDER BY cluster_type, accuracy_pct DESC NULLS LAST
        """
        query_params = [run_id]

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, query_params)
            rows = cur.fetchall()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to get cluster data for run %d", run_id)
        raise HTTPException(status_code=500, detail="Failed to fetch cluster data")

    clusters: dict[str, list[dict[str, Any]]] = {"ml_cluster": [], "business_cluster": []}

    if exec_lag is not None:
        # lag_cluster table: 6 columns (no n_dfus)
        for r in rows:
            entry = {
                "cluster_value": r[1],
                "n_predictions": int(r[2]) if r[2] is not None else 0,
                "accuracy_pct": float(r[3]) if r[3] is not None else None,
                "wape": float(r[4]) if r[4] is not None else None,
                "bias": float(r[5]) if r[5] is not None else None,
            }
            ct = r[0]
            if ct in clusters:
                clusters[ct].append(entry)
    else:
        # cluster table: 7 columns (with n_dfus)
        for r in rows:
            entry = {
                "cluster_value": r[1],
                "n_predictions": int(r[2]) if r[2] is not None else 0,
                "n_dfus": int(r[3]) if r[3] is not None else 0,
                "accuracy_pct": float(r[4]) if r[4] is not None else None,
                "wape": float(r[5]) if r[5] is not None else None,
                "bias": float(r[6]) if r[6] is not None else None,
            }
            ct = r[0]
            if ct in clusters:
                clusters[ct].append(entry)

    result: dict[str, Any] = {"run_id": run_id, "model": model, "clusters": clusters}
    if exec_lag is not None:
        result["exec_lag"] = exec_lag
    return result


# ---------------------------------------------------------------------------
# 6. GET /{model}/experiments/{run_id}/months — Per-month
# ---------------------------------------------------------------------------

@router.get("/{model}/experiments/{run_id}/months")
def get_experiment_months(model: str, run_id: int, response: FastAPIResponse):
    """Get per-month accuracy breakdowns for a single experiment."""
    _validate_model(model)
    set_cache(response, max_age=60)
    _verify_run_ownership(run_id, model)

    sql = """
        SELECT month_start, n_predictions, n_dfus,
               accuracy_pct, wape, bias
        FROM lgbm_tuning_month
        WHERE run_id = %s
        ORDER BY month_start
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, [run_id])
            rows = cur.fetchall()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to get month data for run %d", run_id)
        raise HTTPException(status_code=500, detail="Failed to fetch month data")

    months = [
        {
            "month_start": str(r[0]),
            "n_predictions": int(r[1]) if r[1] is not None else 0,
            "n_dfus": int(r[2]) if r[2] is not None else 0,
            "accuracy_pct": float(r[3]) if r[3] is not None else None,
            "wape": float(r[4]) if r[4] is not None else None,
            "bias": float(r[5]) if r[5] is not None else None,
        }
        for r in rows
    ]
    return {"run_id": run_id, "model": model, "months": months}


# ---------------------------------------------------------------------------
# 7. GET /{model}/experiments/{run_id}/logs — Incremental logs
# ---------------------------------------------------------------------------

@router.get("/{model}/experiments/{run_id}/logs")
def get_experiment_logs(
    model: str,
    run_id: int,
    response: FastAPIResponse,
    offset: int = Query(default=0, ge=0),
):
    """Get incremental log text for an experiment (offset-based streaming)."""
    _validate_model(model)
    set_cache(response, max_age=5)
    _verify_run_ownership(run_id, model)

    # Fetch job_id from the run, then read logs from job_history
    run_sql = """
        SELECT job_id, status FROM lgbm_tuning_run WHERE run_id = %s
    """
    log_sql = """
        SELECT log FROM job_history WHERE job_id = %s
    """

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(run_sql, [run_id])
            run_row = cur.fetchone()
            if run_row is None:
                raise HTTPException(status_code=404, detail="Experiment not found")

            job_id = run_row[0]
            run_status = run_row[1]

            log_text = ""
            if job_id:
                cur.execute(log_sql, [job_id])
                log_row = cur.fetchone()
                if log_row and log_row[0]:
                    log_text = log_row[0]
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to get logs for run %d", run_id)
        raise HTTPException(status_code=500, detail="Failed to fetch logs")

    # Apply offset — return text from character position `offset` onward
    if offset > 0 and offset < len(log_text):
        log_text = log_text[offset:]
    elif offset >= len(log_text):
        log_text = ""

    total_length = offset + len(log_text)

    return {
        "run_id": run_id,
        "model": model,
        "log": log_text,
        "offset": offset,
        "next_offset": total_length,
        "status": run_status,
        "has_more": run_status in ("queued", "running"),
    }


# ---------------------------------------------------------------------------
# 8. GET /{model}/compare — Compare two experiments
# ---------------------------------------------------------------------------

@router.get("/{model}/compare")
def compare_experiments(
    model: str,
    response: FastAPIResponse,
    baseline_id: int = Query(ge=1),
    candidate_id: int = Query(ge=1),
    exec_lag: int | None = Query(default=None, ge=0, le=4),
):
    """Compare two tuning runs and return delta metrics, with optional exec_lag filtering."""
    _validate_model(model)
    set_cache(response, max_age=60)

    if baseline_id == candidate_id:
        raise HTTPException(status_code=400, detail="Baseline and candidate must be different runs")

    run_sql = """
        SELECT r.run_id, r.run_label, r.model_id, r.accuracy_pct, r.wape, r.bias,
               r.n_predictions, r.n_dfus, r.status, r.params, r.features, r.feature_count,
               r.metadata, r.cluster_source, r.cluster_experiment_id,
               ce.label AS cluster_experiment_label
        FROM lgbm_tuning_run r
        LEFT JOIN cluster_experiment ce ON ce.experiment_id = r.cluster_experiment_id
        WHERE r.run_id = %s
    """

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(run_sql, [baseline_id])
            baseline = cur.fetchone()
            if baseline is None:
                raise HTTPException(status_code=404, detail="Baseline run not found")

            cur.execute(run_sql, [candidate_id])
            candidate = cur.fetchone()
            if candidate is None:
                raise HTTPException(status_code=404, detail="Candidate run not found")

            # Check for existing comparison
            cur.execute(
                "SELECT id FROM lgbm_tuning_comparison "
                "WHERE baseline_run_id = %s AND candidate_run_id = %s",
                [baseline_id, candidate_id],
            )
            existing = cur.fetchone()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to compare runs %d vs %d", baseline_id, candidate_id)
        raise HTTPException(status_code=500, detail="Failed to compare experiments")

    b = _run_row_to_dict(baseline)
    c = _run_row_to_dict(candidate)

    # If exec_lag specified, override portfolio metrics with lag-specific metrics
    if exec_lag is not None:
        _apply_lag_metrics(b, baseline_id, exec_lag)
        _apply_lag_metrics(c, candidate_id, exec_lag)

    delta_acc = None
    delta_wape = None
    delta_bias = None
    verdict = "neutral"
    if b["accuracy_pct"] is not None and c["accuracy_pct"] is not None:
        delta_acc = round(c["accuracy_pct"] - b["accuracy_pct"], 2)
        if delta_acc >= 0.05:
            verdict = "improved"
        elif delta_acc <= -0.05:
            verdict = "degraded"
    if b["wape"] is not None and c["wape"] is not None:
        delta_wape = round(c["wape"] - b["wape"], 2)
    if b["bias"] is not None and c["bias"] is not None:
        delta_bias = round(c["bias"] - b["bias"], 4)

    # Fetch per-lag comparison (always, regardless of exec_lag filter)
    per_lag = _build_per_lag_comparison(baseline_id, candidate_id)

    # Fetch cluster and month breakdowns
    cluster_sql = """
        SELECT cluster_type, cluster_value, n_predictions, n_dfus,
               accuracy_pct, wape, bias
        FROM lgbm_tuning_cluster
        WHERE run_id = %s
        ORDER BY cluster_type, cluster_value
    """
    month_sql = """
        SELECT month_start, n_predictions, n_dfus,
               accuracy_pct, wape, bias
        FROM lgbm_tuning_month
        WHERE run_id = %s
        ORDER BY month_start
    """

    try:
        with get_conn() as conn2, conn2.cursor() as cur2:
            cur2.execute(cluster_sql, [baseline_id])
            base_clusters = _parse_cluster_rows(cur2.fetchall())
            cur2.execute(cluster_sql, [candidate_id])
            cand_clusters = _parse_cluster_rows(cur2.fetchall())
            cur2.execute(month_sql, [baseline_id])
            base_months = _parse_month_rows(cur2.fetchall())
            cur2.execute(month_sql, [candidate_id])
            cand_months = _parse_month_rows(cur2.fetchall())
    except Exception:
        logger.exception("Failed to fetch breakdowns for comparison")
        raise HTTPException(status_code=500, detail="Failed to fetch comparison breakdowns")

    # Build per-cluster comparison
    per_cluster: dict[str, list[dict[str, Any]]] = {}
    for ct in ("ml_cluster", "business_cluster"):
        b_map = {r["cluster_value"]: r for r in base_clusters if r["cluster_type"] == ct}
        c_map = {r["cluster_value"]: r for r in cand_clusters if r["cluster_type"] == ct}
        all_vals = sorted(set(b_map.keys()) | set(c_map.keys()))
        items = []
        for val in all_vals:
            br = b_map.get(val, {})
            cr = c_map.get(val, {})
            b_a = br.get("accuracy_pct")
            c_a = cr.get("accuracy_pct")
            items.append({
                "cluster": val,
                "baseline_accuracy": b_a,
                "candidate_accuracy": c_a,
                "delta_accuracy": round(c_a - b_a, 2) if b_a is not None and c_a is not None else None,
                "baseline_wape": br.get("wape"),
                "candidate_wape": cr.get("wape"),
                "baseline_n_dfus": br.get("n_dfus"),
                "candidate_n_dfus": cr.get("n_dfus"),
            })
        per_cluster[ct] = items

    # Build per-month comparison
    b_month_map = {r["month_start"]: r for r in base_months}
    c_month_map = {r["month_start"]: r for r in cand_months}
    all_months = sorted(set(b_month_map.keys()) | set(c_month_map.keys()))
    per_month = []
    for m in all_months:
        br = b_month_map.get(m, {})
        cr = c_month_map.get(m, {})
        b_a = br.get("accuracy_pct")
        c_a = cr.get("accuracy_pct")
        per_month.append({
            "month": m,
            "baseline_accuracy": b_a,
            "candidate_accuracy": c_a,
            "delta_accuracy": round(c_a - b_a, 2) if b_a is not None and c_a is not None else None,
            "baseline_wape": br.get("wape"),
            "candidate_wape": cr.get("wape"),
        })

    # Parameter comparison
    param_diffs: list[dict[str, Any]] = []
    param_common: list[dict[str, Any]] = []
    b_params = b.get("params") or {}
    c_params = c.get("params") or {}
    all_keys = sorted(set(b_params.keys()) | set(c_params.keys()))
    for key in all_keys:
        bv = b_params.get(key)
        cv = c_params.get(key)
        if bv != cv:
            param_diffs.append({"param": key, "baseline": bv, "candidate": cv})
        else:
            param_common.append({"param": key, "value": bv})

    # Feature diff
    b_features = set(b.get("features") or [])
    c_features = set(c.get("features") or [])
    feature_diffs = {
        "baseline_count": b.get("feature_count") or len(b_features),
        "candidate_count": c.get("feature_count") or len(c_features),
        "added": sorted(c_features - b_features),
        "removed": sorted(b_features - c_features),
        "common_count": len(b_features & c_features),
    }

    # Config diff — check metadata dict first, then fall back to top-level keys
    # (cluster_source / cluster_experiment_id are stored as direct columns)
    config_diffs: list[dict[str, Any]] = []
    config_common: list[dict[str, Any]] = []
    b_meta = b.get("metadata") or {}
    c_meta = c.get("metadata") or {}
    for key in _CONFIG_KEYS:
        bv = b_meta.get(key) if b_meta.get(key) is not None else b.get(key)
        cv = c_meta.get(key) if c_meta.get(key) is not None else c.get(key)
        if bv is None and cv is None:
            continue
        if bv != cv:
            config_diffs.append({"setting": key, "baseline": bv, "candidate": cv})
        else:
            config_common.append({"setting": key, "value": bv})

    result: dict[str, Any] = {
        "model": model,
        "baseline": b,
        "candidate": c,
        "delta_accuracy": delta_acc,
        "delta_wape": delta_wape,
        "delta_bias": delta_bias,
        "verdict": verdict,
        "existing_comparison_id": existing[0] if existing else None,
        "per_lag": per_lag,
        "param_diffs": param_diffs,
        "param_common": param_common,
        "feature_diffs": feature_diffs,
        "config_diffs": config_diffs,
        "config_common": config_common,
        "per_cluster": per_cluster,
        "per_month": per_month,
        "baseline_has_breakdowns": len(base_clusters) > 0 or len(base_months) > 0,
        "candidate_has_breakdowns": len(cand_clusters) > 0 or len(cand_months) > 0,
    }
    if exec_lag is not None:
        result["exec_lag_filter"] = exec_lag

    return result


def _apply_lag_metrics(run_dict: dict[str, Any], run_id: int, exec_lag: int) -> None:
    """Override portfolio-level accuracy/wape/bias with lag-specific values in-place."""
    sql = """
        SELECT accuracy_pct, wape, bias, n_predictions
        FROM lgbm_tuning_lag
        WHERE run_id = %s AND exec_lag = %s
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, [run_id, exec_lag])
            row = cur.fetchone()
    except Exception:
        logger.warning("Failed to fetch lag %d metrics for run %d", exec_lag, run_id)
        return

    if row is not None:
        run_dict["accuracy_pct"] = float(row[0]) if row[0] is not None else None
        run_dict["wape"] = float(row[1]) if row[1] is not None else None
        run_dict["bias"] = float(row[2]) if row[2] is not None else None
        if row[3] is not None:
            run_dict["n_predictions"] = int(row[3])


def _build_per_lag_comparison(baseline_id: int, candidate_id: int) -> list[dict[str, Any]]:
    """Build per-lag accuracy comparison array for both runs."""
    sql = """
        SELECT exec_lag, accuracy_pct, wape, bias
        FROM lgbm_tuning_lag
        WHERE run_id = %s
        ORDER BY exec_lag
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, [baseline_id])
            b_rows = cur.fetchall()
            cur.execute(sql, [candidate_id])
            c_rows = cur.fetchall()
    except Exception:
        logger.warning("Failed to fetch per-lag data for comparison")
        return []

    b_map = {r[0]: r for r in b_rows}
    c_map = {r[0]: r for r in c_rows}

    per_lag = []
    for lag in range(5):
        br = b_map.get(lag)
        cr = c_map.get(lag)
        b_acc = float(br[1]) if br and br[1] is not None else None
        c_acc = float(cr[1]) if cr and cr[1] is not None else None
        b_wape = float(br[2]) if br and br[2] is not None else None
        c_wape = float(cr[2]) if cr and cr[2] is not None else None
        b_bias = float(br[3]) if br and br[3] is not None else None
        c_bias = float(cr[3]) if cr and cr[3] is not None else None

        per_lag.append({
            "exec_lag": lag,
            "baseline_acc": b_acc,
            "candidate_acc": c_acc,
            "delta_acc": round(c_acc - b_acc, 2) if b_acc is not None and c_acc is not None else None,
            "baseline_wape": b_wape,
            "candidate_wape": c_wape,
            "delta_wape": round(c_wape - b_wape, 2) if b_wape is not None and c_wape is not None else None,
            "baseline_bias": b_bias,
            "candidate_bias": c_bias,
            "delta_bias": round(c_bias - b_bias, 4) if b_bias is not None and c_bias is not None else None,
        })

    return per_lag


def _parse_cluster_rows(rows: list[tuple]) -> list[dict[str, Any]]:
    """Parse cluster result rows into dicts."""
    return [
        {
            "cluster_type": r[0],
            "cluster_value": r[1],
            "n_predictions": int(r[2]) if r[2] is not None else 0,
            "n_dfus": int(r[3]) if r[3] is not None else 0,
            "accuracy_pct": float(r[4]) if r[4] is not None else None,
            "wape": float(r[5]) if r[5] is not None else None,
            "bias": float(r[6]) if r[6] is not None else None,
        }
        for r in rows
    ]


def _parse_month_rows(rows: list[tuple]) -> list[dict[str, Any]]:
    """Parse month result rows into dicts."""
    return [
        {
            "month_start": str(r[0]),
            "n_predictions": int(r[1]) if r[1] is not None else 0,
            "n_dfus": int(r[2]) if r[2] is not None else 0,
            "accuracy_pct": float(r[3]) if r[3] is not None else None,
            "wape": float(r[4]) if r[4] is not None else None,
            "bias": float(r[5]) if r[5] is not None else None,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# 9. POST /{model}/experiments/{run_id}/promote — Promote
# ---------------------------------------------------------------------------

@router.post("/{model}/experiments/{run_id}/promote")
def promote_experiment(model: str, run_id: int):
    """Promote a completed experiment to production — writes params to algorithm_config.yaml."""
    _validate_model(model)
    mid = _model_id(model)
    algo_section = _ALGO_SECTION[model]
    param_keys = _MODEL_PARAM_KEYS[model]

    # 1. Fetch run
    fetch_sql = """
        SELECT run_id, run_label, status, params, accuracy_pct, wape, bias
        FROM lgbm_tuning_run
        WHERE run_id = %s AND model_id = %s
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(fetch_sql, [run_id, mid])
            row = cur.fetchone()
    except Exception:
        logger.exception("Failed to fetch run %d for promotion", run_id)
        raise HTTPException(status_code=500, detail="Failed to fetch experiment for promotion")

    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    run_status = row[2]
    if run_status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot promote experiment with status '{run_status}' — only completed runs",
        )

    params_raw = row[3]
    if params_raw is None:
        raise HTTPException(status_code=400, detail="Experiment has no params to promote")

    run_params = params_raw if isinstance(params_raw, dict) else json.loads(params_raw)

    # 2. Filter to known param keys for this model
    overrides = {k: v for k, v in run_params.items() if k in param_keys}
    if not overrides:
        raise HTTPException(
            status_code=400,
            detail=f"Experiment params contain no recognized {model} hyperparameters",
        )

    # 3. Backup and write to algorithm_config.yaml
    try:
        with open(_ALGO_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)

        # Create backup
        backup_path = _ALGO_CONFIG_PATH.with_suffix(f".yaml.bak.{run_id}")
        shutil.copy2(_ALGO_CONFIG_PATH, backup_path)

        section = cfg["algorithms"][algo_section]
        old_params = {k: section.get(k) for k in overrides}
        for key, value in overrides.items():
            section[key] = value

        with open(_ALGO_CONFIG_PATH, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    except (OSError, KeyError, yaml.YAMLError) as exc:
        logger.exception("Failed to write algorithm_config.yaml during %s promote", model)
        raise HTTPException(status_code=500, detail=f"Failed to update config: {exc}")

    # 4. Atomically update DB: clear previous, set new, log to promotion table
    try:
        with get_conn() as conn, conn.cursor() as cur:
            # Get previous promoted run_id for audit trail
            cur.execute(
                "SELECT run_id FROM lgbm_tuning_run "
                "WHERE is_promoted = TRUE AND model_id = %s AND run_id != %s",
                [mid, run_id],
            )
            prev_row = cur.fetchone()
            previous_run_id = prev_row[0] if prev_row else None

            # Clear previous promotion
            cur.execute(
                "UPDATE lgbm_tuning_run SET is_promoted = FALSE, promoted_at = NULL "
                "WHERE is_promoted = TRUE AND model_id = %s AND run_id != %s",
                [mid, run_id],
            )

            # Set new promotion
            cur.execute(
                "UPDATE lgbm_tuning_run SET is_promoted = TRUE, promoted_at = NOW() "
                "WHERE run_id = %s",
                [run_id],
            )

            # Insert promotion log
            cur.execute(
                """INSERT INTO tuning_promotion_log
                    (run_id, model_id, promoted_by, previous_run_id,
                     params_written, accuracy_pct, wape, bias)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    run_id,
                    mid,
                    "manual",
                    previous_run_id,
                    json.dumps(overrides),
                    row[4],  # accuracy_pct
                    row[5],  # wape
                    row[6],  # bias
                ],
            )

            conn.commit()
    except Exception:
        logger.exception("Failed to update DB during %s promote of run %d", model, run_id)
        raise HTTPException(status_code=500, detail="Failed to update promotion state")

    logger.info(
        "Promoted %s run #%d to production. Overrides: %s",
        model, run_id, overrides,
    )

    return {
        "promoted": True,
        "run_id": run_id,
        "model": model,
        "run_label": row[1],
        "accuracy_pct": float(row[4]) if row[4] is not None else None,
        "params_written": overrides,
        "old_params": old_params,
        "previous_run_id": previous_run_id,
        "backup_path": str(backup_path),
    }


# ---------------------------------------------------------------------------
# 10. GET /{model}/promoted — Currently promoted run
# ---------------------------------------------------------------------------

@router.get("/{model}/promoted")
def get_promoted(model: str, response: FastAPIResponse):
    """Return the currently promoted experiment for this model (if any)."""
    _validate_model(model)
    set_cache(response, max_age=30)
    mid = _model_id(model)

    sql = """
        SELECT run_id, run_label, model_id, accuracy_pct, wape, bias,
               promoted_at, params
        FROM lgbm_tuning_run
        WHERE is_promoted = TRUE AND model_id = %s
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, [mid])
            row = cur.fetchone()
    except Exception:
        logger.exception("Failed to get promoted run for %s", model)
        raise HTTPException(status_code=500, detail="Failed to fetch promoted run")

    if row is None:
        return {"model": model, "promoted": None}

    return {
        "model": model,
        "promoted": {
            "run_id": row[0],
            "run_label": row[1],
            "model_id": row[2],
            "accuracy_pct": float(row[3]) if row[3] is not None else None,
            "wape": float(row[4]) if row[4] is not None else None,
            "bias": float(row[5]) if row[5] is not None else None,
            "promoted_at": str(row[6]) if row[6] else None,
            "params": row[7] if isinstance(row[7], dict) else json.loads(row[7]) if row[7] else None,
        },
    }


# ---------------------------------------------------------------------------
# 11. POST /{model}/experiments/{run_id}/promote-results — Load predictions into DB
# ---------------------------------------------------------------------------

@router.post("/{model}/experiments/{run_id}/promote-results")
def promote_results(model: str, run_id: int):
    """Load backtest predictions into fact_external_forecast_monthly + backtest_lag_archive.

    Submits an async job to run load_backtest_forecasts.py --model <model_id> --replace.
    After loading, refreshes 5 materialized views so accuracy screens reflect new data.
    """
    _validate_model(model)
    mid = _model_id(model)

    # Verify run exists, belongs to this model, and is completed
    sql = """
        SELECT status, is_results_promoted, results_promote_job_id
        FROM lgbm_tuning_run
        WHERE run_id = %s AND model_id = %s
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, [run_id, mid])
            row = cur.fetchone()
    except Exception:
        logger.exception("Failed to fetch run %d for results promotion", run_id)
        raise HTTPException(status_code=500, detail="Failed to fetch experiment")

    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    current_status, is_results_promoted, existing_job_id = row

    if current_status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot promote results for experiment with status '{current_status}' — must be completed",
        )

    # Check prediction files exist
    output_dir = MODEL_ID_MAP[model]
    pred_path = _ALGO_CONFIG_PATH.parent.parent / "data" / "backtest" / output_dir / "backtest_predictions.csv"
    if not pred_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Prediction file not found at {pred_path}. Re-run the experiment to regenerate.",
        )

    # Submit async job
    try:
        from common.services.job_registry import JobManager
        mgr = JobManager()
        job_id = mgr.submit_job(
            job_type="load_backtest_results",
            params={"run_id": run_id, "model": model},
            label=f"Load {model.upper()} results — Run #{run_id}",
            triggered_by="manual",
            group_override=f"tuning_{model}",
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception:
        logger.exception("Failed to submit results load job for run %d", run_id)
        raise HTTPException(status_code=500, detail="Failed to submit load job")

    # Store job_id on the run record
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE lgbm_tuning_run SET results_promote_job_id = %s WHERE run_id = %s",
                [job_id, run_id],
            )
            conn.commit()
    except Exception:
        logger.warning("Failed to store results_promote_job_id on run %d", run_id)

    return {
        "job_id": job_id,
        "run_id": run_id,
        "model": model,
        "message": f"Results loading started for {model.upper()} run #{run_id}",
    }


# ---------------------------------------------------------------------------
# 11b. GET /{model}/experiments/{run_id}/promote-results/status
# ---------------------------------------------------------------------------

@router.get("/{model}/experiments/{run_id}/promote-results/status")
def promote_results_status(model: str, run_id: int):
    """Check the status of a results promotion job."""
    _validate_model(model)
    mid = _model_id(model)

    sql = """
        SELECT is_results_promoted, results_promoted_at, results_promote_job_id
        FROM lgbm_tuning_run
        WHERE run_id = %s AND model_id = %s
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, [run_id, mid])
            row = cur.fetchone()
    except Exception:
        logger.exception("Failed to fetch results promotion status for run %d", run_id)
        raise HTTPException(status_code=500, detail="Failed to fetch status")

    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    is_promoted, promoted_at, job_id = row

    if is_promoted:
        return {
            "status": "completed",
            "is_results_promoted": True,
            "results_promoted_at": str(promoted_at) if promoted_at else None,
        }

    if not job_id:
        return {"status": "not_started", "is_results_promoted": False}

    # Look up job status
    try:
        from common.services.job_registry import JobManager
        mgr = JobManager()
        job = mgr.get_job(job_id)
        if job:
            return {
                "status": job["status"],
                "is_results_promoted": False,
                "progress_pct": job.get("progress_pct", 0),
                "progress_msg": job.get("progress_msg", ""),
                "error": job.get("error"),
            }
    except Exception:
        logger.warning("Failed to look up job %s", job_id)

    return {"status": "unknown", "is_results_promoted": False}


# ---------------------------------------------------------------------------
# 12. POST /{model}/experiments/{run_id}/cancel — Cancel
# ---------------------------------------------------------------------------

@router.post("/{model}/experiments/{run_id}/cancel")
def cancel_experiment(model: str, run_id: int):
    """Cancel a running or queued experiment."""
    _validate_model(model)
    mid = _model_id(model)

    # Fetch current status and job_id
    sql = """
        SELECT status, job_id FROM lgbm_tuning_run
        WHERE run_id = %s AND model_id = %s
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, [run_id, mid])
            row = cur.fetchone()
    except Exception:
        logger.exception("Failed to fetch run %d for cancel", run_id)
        raise HTTPException(status_code=500, detail="Failed to fetch experiment")

    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    current_status = row[0]
    job_id = row[1]

    if current_status not in ("queued", "running"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel experiment with status '{current_status}' — only queued/running",
        )

    # Attempt to cancel via JobManager if job_id exists
    if job_id:
        try:
            from common.services.job_registry import JobManager
            mgr = JobManager()
            mgr.cancel_job(job_id)
        except Exception:
            logger.warning("Failed to cancel job %s via JobManager", job_id)

    # Update run status to cancelled
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE lgbm_tuning_run SET status = %s, completed_at = NOW() "
                "WHERE run_id = %s",
                ["cancelled", run_id],
            )
            conn.commit()
    except Exception:
        logger.exception("Failed to update status to cancelled for run %d", run_id)
        raise HTTPException(status_code=500, detail="Failed to cancel experiment")

    logger.info("Cancelled %s experiment %d (job_id=%s)", model, run_id, job_id)

    return {
        "cancelled": True,
        "run_id": run_id,
        "model": model,
        "previous_status": current_status,
    }


# ---------------------------------------------------------------------------
# 12. DELETE /{model}/experiments/{run_id} — Delete
# ---------------------------------------------------------------------------

@router.delete("/{model}/experiments/{run_id}")
def delete_experiment(model: str, run_id: int):
    """Delete a completed, failed, or cancelled experiment."""
    _validate_model(model)
    mid = _model_id(model)

    # Verify status allows deletion
    sql = """
        SELECT status, is_promoted FROM lgbm_tuning_run
        WHERE run_id = %s AND model_id = %s
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, [run_id, mid])
            row = cur.fetchone()
    except Exception:
        logger.exception("Failed to fetch run %d for deletion", run_id)
        raise HTTPException(status_code=500, detail="Failed to fetch experiment")

    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    current_status = row[0]
    is_promoted = bool(row[1])

    if current_status in ("queued", "running"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete experiment with status '{current_status}' — cancel it first",
        )

    if is_promoted:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete the currently promoted experiment — demote it first",
        )

    # Delete the run (CASCADE will clean up timeframe, cluster, month, lag rows)
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM lgbm_tuning_run WHERE run_id = %s", [run_id])
            conn.commit()
    except Exception:
        logger.exception("Failed to delete run %d", run_id)
        raise HTTPException(status_code=500, detail="Failed to delete experiment")

    logger.info("Deleted %s experiment %d", model, run_id)

    return {"deleted": True, "run_id": run_id, "model": model}


# ---------------------------------------------------------------------------
# 13. GET /{model}/templates — Experiment templates
# ---------------------------------------------------------------------------

@router.get("/{model}/templates")
def get_templates(model: str, response: FastAPIResponse):
    """Load experiment templates from config/tuning_templates.yaml."""
    _validate_model(model)
    set_cache(response, max_age=300)

    try:
        from common.utils import load_config
        tmpl_cfg = load_config("tuning_templates")
    except (FileNotFoundError, OSError) as exc:
        logger.exception("Failed to load tuning_templates.yaml")
        raise HTTPException(status_code=500, detail=f"Failed to load templates: {exc}")

    model_templates = tmpl_cfg.get("templates", {}).get(model, [])

    # For templates with source='algorithm_config', load live params
    enriched: list[dict[str, Any]] = []
    live_params: dict[str, Any] | None = None

    for tmpl in model_templates:
        entry = dict(tmpl)
        if entry.get("source") == "algorithm_config":
            # Lazy-load live params from algorithm_config.yaml
            if live_params is None:
                live_params = _load_live_params(model)
            entry["params"] = live_params
        enriched.append(entry)

    return {"model": model, "templates": enriched}


def _load_live_params(model: str) -> dict[str, Any]:
    """Load the current production params from algorithm_config.yaml for a model."""
    try:
        with open(_ALGO_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        section = cfg.get("algorithms", {}).get(_ALGO_SECTION[model], {})
        # Filter to known param keys
        param_keys = _MODEL_PARAM_KEYS[model]
        return {k: v for k, v in section.items() if k in param_keys}
    except (OSError, yaml.YAMLError):
        logger.warning("Failed to load live params from algorithm_config.yaml for %s", model)
        return {}


# ---------------------------------------------------------------------------
# 14. GET /{model}/promotions — Promotion audit trail
# ---------------------------------------------------------------------------

@router.get("/{model}/promotions")
def list_promotions(
    model: str,
    response: FastAPIResponse,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List promotion audit trail for this model."""
    _validate_model(model)
    set_cache(response, max_age=30)
    mid = _model_id(model)

    sql = """
        SELECT p.id, p.run_id, p.model_id, p.promoted_at, p.promoted_by,
               p.previous_run_id, p.params_written, p.accuracy_pct, p.wape, p.bias,
               p.notes, r.run_label
        FROM tuning_promotion_log p
        LEFT JOIN lgbm_tuning_run r ON r.run_id = p.run_id
        WHERE p.model_id = %s
        ORDER BY p.promoted_at DESC
        LIMIT %s OFFSET %s
    """

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, [mid, limit, offset])
            rows = cur.fetchall()
    except Exception:
        logger.exception("Failed to list promotions for %s", model)
        raise HTTPException(status_code=500, detail="Failed to fetch promotion history")

    promotions = []
    for r in rows:
        promotions.append({
            "id": r[0],
            "run_id": r[1],
            "model_id": r[2],
            "promoted_at": str(r[3]) if r[3] else None,
            "promoted_by": r[4],
            "previous_run_id": r[5],
            "params_written": _parse_json(r[6]),
            "accuracy_pct": float(r[7]) if r[7] is not None else None,
            "wape": float(r[8]) if r[8] is not None else None,
            "bias": float(r[9]) if r[9] is not None else None,
            "notes": r[10],
            "run_label": r[11],
        })

    return {"model": model, "promotions": promotions}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _verify_run_ownership(run_id: int, model: str) -> None:
    """Verify that a run exists and belongs to the specified model. Raises 404 if not."""
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
    except Exception:
        logger.exception("Failed to verify run ownership for %d/%s", run_id, model)
        raise HTTPException(status_code=500, detail="Failed to verify experiment")
