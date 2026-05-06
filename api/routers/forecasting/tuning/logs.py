"""GET /{model}/experiments/{run_id}/logs — incremental log streaming."""
from __future__ import annotations

import logging

import psycopg
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache

from ._helpers import _validate_model, _verify_run_ownership

logger = logging.getLogger(__name__)

router = APIRouter(tags=["model-tuning"])


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
    except psycopg.Error:
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
