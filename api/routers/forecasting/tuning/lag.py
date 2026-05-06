"""GET /{model}/experiments/{run_id}/lags — per-execution-lag accuracy."""
from __future__ import annotations

import logging

import psycopg
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache

from ._helpers import _validate_model, _verify_run_ownership

logger = logging.getLogger(__name__)

router = APIRouter(tags=["model-tuning"])


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
    except psycopg.Error:
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
