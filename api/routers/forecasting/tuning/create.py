"""POST /{model}/experiments — create + launch experiment."""
from __future__ import annotations

import json
import logging

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_api_key
from api.core import get_conn

from . import _helpers
from ._helpers import (
    CreateExperimentBody,
    _model_id,
    _validate_model,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["model-tuning"])


@router.post("/{model}/experiments", status_code=201, dependencies=[Depends(require_api_key)])
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
        except psycopg.Error:
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
    except psycopg.Error:
        logger.exception("Failed to create %s experiment", model)
        raise HTTPException(status_code=500, detail=f"Failed to create {model} experiment")

    new_run_id = row[0]

    # Build temporary algorithm_config.yaml with overrides.
    # NOTE: tests patch ``_helpers._build_temp_config``; calling via the
    # module attribute (rather than a direct import) keeps the patch effective.
    effective_config = dict(cfg)
    if cluster_override_path:
        effective_config["cluster_override_path"] = cluster_override_path
    config_path = _helpers._build_temp_config(model, body.params, effective_config)

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
    except psycopg.Error:
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


