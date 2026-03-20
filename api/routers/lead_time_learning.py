"""
F3.3 — Supplier Lead Time Learning API endpoints.

Endpoints:
    GET  /supply/supplier-lead-times          — Supplier LT profile + trend
    GET  /supply/supplier-lead-times/summary  — Portfolio health by supplier
    GET  /supply/lead-time-alerts             — Open SS review triggers
    POST /supply/lead-time-review/{trigger_id}/acknowledge  — Dismiss trigger (auth)
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from api.core import get_conn
from api.auth import require_api_key

router = APIRouter(tags=["lead-time-learning"])


@router.get("/supply/supplier-lead-times")
async def get_supplier_lead_times(
    supplier_id: str | None = None,
    item_category: str | None = None,
    page: int = 1,
    page_size: int = 50,
):
    """Return supplier lead time profiles with reliability statistics."""
    page_size = max(1, min(page_size, 200))
    offset = (max(1, page) - 1) * page_size

    conditions = ["1=1"]
    params: list = []
    if supplier_id:
        conditions.append("supplier_id = %s"); params.append(supplier_id)
    if item_category:
        conditions.append("item_category = %s"); params.append(item_category)
    where = " AND ".join(conditions)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM dim_lead_time_profile WHERE {where}", params
            )
            total = cur.fetchone()[0] or 0
            cur.execute(
                f"""
                SELECT supplier_id, item_category, loc,
                       mean_lt_days, stddev_lt_days, p50_lt_days, p90_lt_days,
                       on_time_delivery_rate, sample_size, flagged_for_ss_review,
                       updated_at
                FROM dim_lead_time_profile
                WHERE {where}
                ORDER BY on_time_delivery_rate ASC
                LIMIT %s OFFSET %s
                """,
                params + [page_size, offset],
            )
            rows = cur.fetchall()

    cols = [
        "supplier_id", "item_category", "loc",
        "mean_lt_days", "stddev_lt_days", "p50_lt_days", "p90_lt_days",
        "on_time_delivery_rate", "sample_size", "flagged_for_ss_review",
        "updated_at",
    ]
    items = []
    for r in rows:
        d = dict(zip(cols, r))
        if d.get("updated_at"):
            d["updated_at"] = d["updated_at"].isoformat()
        items.append(d)

    return {"total": total, "page": page, "profiles": items}


@router.get("/supply/supplier-lead-times/summary")
async def get_supplier_lt_summary():
    """Portfolio aggregation: OTDR distribution, top degraded suppliers."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(DISTINCT supplier_id)                             AS supplier_count,
                    AVG(on_time_delivery_rate)                             AS avg_otdr,
                    AVG(mean_lt_days)                                      AS avg_mean_lt,
                    SUM(CASE WHEN on_time_delivery_rate < 0.80 THEN 1 ELSE 0 END) AS poor_suppliers,
                    SUM(CASE WHEN flagged_for_ss_review THEN 1 ELSE 0 END) AS flagged_suppliers
                FROM dim_lead_time_profile
            """)
            row = cur.fetchone()

    if not row:
        return {"supplier_count": 0}

    return {
        "supplier_count": row[0] or 0,
        "avg_otdr": float(row[1]) if row[1] is not None else None,
        "avg_mean_lt_days": float(row[2]) if row[2] is not None else None,
        "poor_suppliers": row[3] or 0,
        "flagged_suppliers": row[4] or 0,
    }


@router.get("/supply/lead-time-alerts")
async def get_lead_time_alerts(
    review_status: str = "open",
    page: int = 1,
    page_size: int = 50,
):
    """Open safety stock review triggers from lead time degradation."""
    page_size = max(1, min(page_size, 100))
    offset = (max(1, page) - 1) * page_size

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM fact_lt_review_triggers WHERE review_status = %s",
                (review_status,),
            )
            total = cur.fetchone()[0] or 0
            cur.execute(
                """
                SELECT id, supplier_id, trigger_type, old_mean_lt_days, new_mean_lt_days,
                       old_stddev_lt_days, new_stddev_lt_days,
                       affected_dfu_count, review_status, triggered_at
                FROM fact_lt_review_triggers
                WHERE review_status = %s
                ORDER BY triggered_at DESC
                LIMIT %s OFFSET %s
                """,
                (review_status, page_size, offset),
            )
            rows = cur.fetchall()

    cols = [
        "id", "supplier_id", "trigger_type", "old_mean_lt_days", "new_mean_lt_days",
        "old_stddev_lt_days", "new_stddev_lt_days",
        "affected_dfu_count", "review_status", "triggered_at",
    ]
    items = []
    for r in rows:
        d = dict(zip(cols, r))
        if d.get("triggered_at"):
            d["triggered_at"] = d["triggered_at"].isoformat()
        items.append(d)

    return {"total": total, "review_status": review_status, "alerts": items}


@router.post("/supply/lead-time-review/{trigger_id}/acknowledge")
async def acknowledge_lt_trigger(trigger_id: int, request: Request):
    """Acknowledge (close) a lead time review trigger (auth required)."""
    await require_api_key(x_api_key=request.headers.get("x-api-key"))

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE fact_lt_review_triggers
                SET review_status = 'acknowledged', acknowledged_at = NOW()
                WHERE id = %s AND review_status = 'open'
                RETURNING id
                """,
                (trigger_id,),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(404, f"Trigger {trigger_id} not found or already acknowledged")

    return {"id": trigger_id, "review_status": "acknowledged"}
