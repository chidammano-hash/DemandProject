"""GET /{model}/experiments/{run_id}/months — per-month accuracy."""
from __future__ import annotations

import logging

import psycopg
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache

from ._helpers import _validate_model, _verify_run_ownership

logger = logging.getLogger(__name__)

router = APIRouter(tags=["model-tuning"])


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
    except psycopg.Error:
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
