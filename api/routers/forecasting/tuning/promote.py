"""Promote a tuning experiment + view the currently promoted run."""
from __future__ import annotations

import json
import logging
import shutil

import psycopg
import yaml
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response as FastAPIResponse

from api.auth import require_api_key
from api.core import get_conn, set_cache
from common.core.utils import get_pipeline_config_path, reset_config

from ._helpers import (
    MODEL_ID_MAP,
    _MODEL_PARAM_KEYS,
    _model_id,
    _validate_model,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["model-tuning"])


@router.post("/{model}/experiments/{run_id}/promote", dependencies=[Depends(require_api_key)])
def promote_experiment(model: str, run_id: int):
    """Promote a completed experiment to production — writes params to forecast_pipeline_config.yaml."""
    _validate_model(model)
    mid = _model_id(model)
    algo_section = MODEL_ID_MAP[model]
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
    except psycopg.Error:
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

    # 3. Backup and write to forecast_pipeline_config.yaml (algorithms.<model_id>.params)
    try:
        with open(get_pipeline_config_path()) as f:
            cfg = yaml.safe_load(f)

        # Create backup
        backup_path = get_pipeline_config_path().with_suffix(f".yaml.bak.{run_id}")
        shutil.copy2(get_pipeline_config_path(), backup_path)

        entry = cfg["algorithms"][algo_section]
        params_section = entry.setdefault("params", {})
        old_params = {k: params_section.get(k) for k in overrides}
        for key, value in overrides.items():
            params_section[key] = value

        with open(get_pipeline_config_path(), "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        reset_config("forecast_pipeline_config.yaml")
    except (OSError, KeyError, yaml.YAMLError):
        logger.exception("Failed to write forecast_pipeline_config.yaml during %s promote", model)
        raise HTTPException(status_code=500, detail="Failed to update config")

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
    except psycopg.Error:
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
    except psycopg.Error:
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
