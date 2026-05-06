"""GET /{model}/promotions — promotion audit trail."""
from __future__ import annotations

import logging

import psycopg
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache

from ._helpers import _model_id, _parse_json, _validate_model

logger = logging.getLogger(__name__)

router = APIRouter(tags=["model-tuning"])


@router.get("/{model}/promotions")
def list_promotions(
    model: str,
    response: FastAPIResponse,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List promotion audit trail for this model."""
    _validate_model(model)
    set_cache(response, max_age=30)
    mid = _model_id(model)

    sql = """
        SELECT p.id, p.run_id, p.model_id, p.promoted_at, p.promoted_by,
               p.previous_run_id, p.params_written, p.accuracy_pct, p.wape, p.bias,
               p.notes, r.run_label
        FROM tuning_promotion_log p
        LEFT JOIN lgbm_tuning_run r ON r.run_id = p.run_id
        WHERE p.model_id = %s
        ORDER BY p.promoted_at DESC
        LIMIT %s OFFSET %s
    """

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, [mid, limit, offset])
            rows = cur.fetchall()
    except psycopg.Error:
        logger.exception("Failed to list promotions for %s", model)
        raise HTTPException(status_code=500, detail="Failed to fetch promotion history")

    promotions = []
    for r in rows:
        promotions.append({
            "id": r[0],
            "run_id": r[1],
            "model_id": r[2],
            "promoted_at": str(r[3]) if r[3] else None,
            "promoted_by": r[4],
            "previous_run_id": r[5],
            "params_written": _parse_json(r[6]),
            "accuracy_pct": float(r[7]) if r[7] is not None else None,
            "wape": float(r[8]) if r[8] is not None else None,
            "bias": float(r[9]) if r[9] is not None else None,
            "notes": r[10],
            "run_label": r[11],
        })

    return {"model": model, "promotions": promotions}
