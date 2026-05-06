"""Cancel a queued/running tuning experiment + delete a finished one."""
from __future__ import annotations

import logging

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_api_key
from api.core import get_conn

from ._helpers import _model_id, _validate_model

logger = logging.getLogger(__name__)

router = APIRouter(tags=["model-tuning"])


@router.post("/{model}/experiments/{run_id}/cancel", dependencies=[Depends(require_api_key)])
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
    except psycopg.Error:
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
        except Exception:  # noqa: BLE001 — JobManager.cancel_job may raise undocumented exceptions
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
    except psycopg.Error:
        logger.exception("Failed to update status to cancelled for run %d", run_id)
        raise HTTPException(status_code=500, detail="Failed to cancel experiment")

    logger.info("Cancelled %s experiment %d (job_id=%s)", model, run_id, job_id)

    return {
        "cancelled": True,
        "run_id": run_id,
        "model": model,
        "previous_status": current_status,
    }


@router.delete("/{model}/experiments/{run_id}", dependencies=[Depends(require_api_key)])
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
    except psycopg.Error:
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
    except psycopg.Error:
        logger.exception("Failed to delete run %d", run_id)
        raise HTTPException(status_code=500, detail="Failed to delete experiment")

    logger.info("Deleted %s experiment %d", model, run_id)

    return {"deleted": True, "run_id": run_id, "model": model}
