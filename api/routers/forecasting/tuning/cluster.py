"""GET /{model}/experiments/{run_id}/clusters — per-cluster accuracy."""
from __future__ import annotations

import logging
from typing import Any

import psycopg
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache

from ._helpers import _validate_model, _verify_run_ownership

logger = logging.getLogger(__name__)

router = APIRouter(tags=["model-tuning"])


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
    except psycopg.Error:
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
