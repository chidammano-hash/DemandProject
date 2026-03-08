"""
F3.1 — Forecast Bias Correction Engine API endpoints.

Endpoints:
    GET /forecast/bias-corrections          — DFU-level corrections
    GET /forecast/bias-corrections/summary  — Plan-level KPIs
    GET /forecast/bias-corrections/flagged  — Items flagged for review
    GET /forecast/bias-corrections/history  — Segment trend history
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.core import get_conn

router = APIRouter(tags=["bias-corrections"])


@router.get("/forecast/bias-corrections")
async def get_bias_corrections(
    item_no: str | None = None,
    loc: str | None = None,
    plan_month: str | None = None,
    segment_type: str | None = None,
    page: int = 1,
    page_size: int = 50,
):
    """List bias correction rows for a given plan month / item / location."""
    page_size = max(1, min(page_size, 200))
    offset = (max(1, page) - 1) * page_size

    conditions = ["1=1"]
    params: list = []
    if item_no:
        conditions.append("item_no = %s"); params.append(item_no)
    if loc:
        conditions.append("loc = %s"); params.append(loc)
    if plan_month:
        conditions.append("plan_month = %s"); params.append(plan_month)
    if segment_type:
        conditions.append("segment_type = %s"); params.append(segment_type)
    where = " AND ".join(conditions)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM fact_bias_corrections WHERE {where}",
                params,
            )
            total = cur.fetchone()[0]
            cur.execute(
                f"""
                SELECT item_no, loc, plan_month, segment_type, segment_value,
                       rolling_bias_3m, correction_factor, correction_was_clipped,
                       correction_pct, flagged_for_review, correction_applied,
                       months_of_data, computed_at
                FROM fact_bias_corrections
                WHERE {where}
                ORDER BY plan_month DESC, flagged_for_review DESC, item_no
                LIMIT %s OFFSET %s
                """,
                params + [page_size, offset],
            )
            rows = cur.fetchall()

    cols = [
        "item_no", "loc", "plan_month", "segment_type", "segment_value",
        "rolling_bias_3m", "correction_factor", "correction_was_clipped",
        "correction_pct", "flagged_for_review", "correction_applied",
        "months_of_data", "computed_at",
    ]
    items = []
    for r in rows:
        d = dict(zip(cols, r))
        d["plan_month"] = d["plan_month"].isoformat() if d["plan_month"] else None
        d["computed_at"] = d["computed_at"].isoformat() if d["computed_at"] else None
        items.append(d)

    return {"total": total, "page": page, "corrections": items}


@router.get("/forecast/bias-corrections/summary")
async def get_bias_corrections_summary(
    plan_month: str | None = None,
):
    """Portfolio-level bias correction KPIs."""
    cond = "WHERE plan_month = %s" if plan_month else ""
    params = [plan_month] if plan_month else []

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    COUNT(*)                                                    AS total_corrections,
                    COUNT(DISTINCT item_no || '@' || loc)                       AS dfu_count,
                    SUM(CASE WHEN flagged_for_review THEN 1 ELSE 0 END)         AS flagged_count,
                    SUM(CASE WHEN correction_was_clipped THEN 1 ELSE 0 END)     AS clipped_count,
                    AVG(rolling_bias_3m)                                        AS avg_rolling_bias,
                    AVG(correction_factor)                                      AS avg_correction_factor,
                    MAX(computed_at)                                            AS last_computed_at
                FROM fact_bias_corrections
                {cond}
                """,
                params,
            )
            row = cur.fetchone()

    if not row:
        return {"total_corrections": 0, "dfu_count": 0}

    return {
        "total_corrections": row[0] or 0,
        "dfu_count": row[1] or 0,
        "flagged_count": row[2] or 0,
        "clipped_count": row[3] or 0,
        "avg_rolling_bias": float(row[4]) if row[4] is not None else None,
        "avg_correction_factor": float(row[5]) if row[5] is not None else None,
        "last_computed_at": row[6].isoformat() if row[6] else None,
        "plan_month": plan_month,
    }


@router.get("/forecast/bias-corrections/flagged")
async def get_flagged_bias_corrections(
    plan_month: str | None = None,
    page: int = 1,
    page_size: int = 50,
):
    """Return items flagged for human review (correction > 20%)."""
    page_size = max(1, min(page_size, 200))
    offset = (max(1, page) - 1) * page_size

    conditions = ["flagged_for_review = TRUE"]
    params: list = []
    if plan_month:
        conditions.append("plan_month = %s")
        params.append(plan_month)
    where = " AND ".join(conditions)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM fact_bias_corrections WHERE {where}", params)
            total = cur.fetchone()[0]
            cur.execute(
                f"""
                SELECT item_no, loc, plan_month, segment_type, rolling_bias_3m,
                       correction_factor_raw, correction_factor, correction_was_clipped,
                       months_of_data
                FROM fact_bias_corrections
                WHERE {where}
                ORDER BY ABS(rolling_bias_3m) DESC
                LIMIT %s OFFSET %s
                """,
                params + [page_size, offset],
            )
            rows = cur.fetchall()

    cols = [
        "item_no", "loc", "plan_month", "segment_type", "rolling_bias_3m",
        "correction_factor_raw", "correction_factor", "correction_was_clipped",
        "months_of_data",
    ]
    items = [dict(zip(cols, r)) for r in rows]
    for d in items:
        if d.get("plan_month"):
            d["plan_month"] = d["plan_month"].isoformat()

    return {"total": total, "page": page, "flagged": items}


@router.get("/forecast/bias-corrections/history")
async def get_bias_correction_history(
    segment_type: str = "cluster",
    segment_value: str | None = None,
    months: int = 6,
):
    """Trend history of correction factors for a segment."""
    conditions = ["segment_type = %s"]
    params: list = [segment_type]
    if segment_value:
        conditions.append("segment_value = %s")
        params.append(segment_value)
    where = " AND ".join(conditions)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT segment_value, computation_month, rolling_bias_3m,
                       correction_factor, dfu_count_in_segment,
                       avg_raw_wape, avg_corrected_wape, correction_improved_accuracy
                FROM fact_bias_correction_history
                WHERE {where}
                ORDER BY computation_month DESC
                LIMIT %s
                """,
                params + [months * 20],
            )
            rows = cur.fetchall()

    cols = [
        "segment_value", "computation_month", "rolling_bias_3m", "correction_factor",
        "dfu_count_in_segment", "avg_raw_wape", "avg_corrected_wape",
        "correction_improved_accuracy",
    ]
    history = [dict(zip(cols, r)) for r in rows]
    for d in history:
        if d.get("computation_month"):
            d["computation_month"] = d["computation_month"].isoformat()

    return {"segment_type": segment_type, "segment_value": segment_value, "history": history}
