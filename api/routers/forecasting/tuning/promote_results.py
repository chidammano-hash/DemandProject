"""Promote backtest predictions of a tuning run into the fact tables."""
from __future__ import annotations

import logging

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_api_key
from api.core import get_conn
from common.core.utils import get_pipeline_config_path

from ._helpers import MODEL_ID_MAP, _model_id, _validate_model

logger = logging.getLogger(__name__)

router = APIRouter(tags=["model-tuning"])


@router.post("/{model}/experiments/{run_id}/promote-results", dependencies=[Depends(require_api_key)])
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
    except psycopg.Error:
        logger.exception("Failed to fetch run %d for results promotion", run_id)
        raise HTTPException(status_code=500, detail="Failed to fetch experiment")

    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    current_status, _is_results_promoted, _existing_job_id = row

    if current_status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot promote results for experiment with status '{current_status}' — must be completed",
        )

    # Check prediction files exist
    output_dir = MODEL_ID_MAP[model]
    pred_path = get_pipeline_config_path().parent.parent / "data" / "backtest" / output_dir / "backtest_predictions.csv"
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
    except Exception:  # noqa: BLE001 — JobManager.submit_job may raise undocumented exceptions
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
    except psycopg.Error:
        logger.warning("Failed to store results_promote_job_id on run %d", run_id)

    return {
        "job_id": job_id,
        "run_id": run_id,
        "model": model,
        "message": f"Results loading started for {model.upper()} run #{run_id}",
    }


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
    except psycopg.Error:
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
    except Exception:  # noqa: BLE001 — JobManager.get_job may raise undocumented exceptions
        logger.warning("Failed to look up job %s", job_id)

    return {"status": "unknown", "is_results_promoted": False}
