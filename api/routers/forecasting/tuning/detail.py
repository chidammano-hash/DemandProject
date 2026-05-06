"""GET /{model}/experiments/{run_id} — single experiment detail."""
from __future__ import annotations

import logging

import psycopg
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache

from ._helpers import _model_id, _parse_json, _validate_model

logger = logging.getLogger(__name__)

router = APIRouter(tags=["model-tuning"])


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
    except psycopg.Error:
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
