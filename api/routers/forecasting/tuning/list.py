"""GET /{model}/experiments — list tuning experiments."""
from __future__ import annotations

import logging
from typing import Any

import psycopg
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache

from ._helpers import _model_id, _validate_model

logger = logging.getLogger(__name__)

router = APIRouter(tags=["model-tuning"])


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
               ce.label AS cluster_experiment_label,
               jh.error AS job_error
        FROM lgbm_tuning_run r
        LEFT JOIN cluster_experiment ce ON ce.experiment_id = r.cluster_experiment_id
        LEFT JOIN job_history jh ON jh.job_id = r.job_id
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
    except psycopg.Error:
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
            "error": r[22],
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
