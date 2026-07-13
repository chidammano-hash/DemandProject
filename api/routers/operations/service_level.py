"""
F3.2 — Service Level Actuals vs. Targets Tracking API endpoints.

Endpoints:
    GET  /analytics/service-level/summary       — Portfolio SL KPIs
    GET  /analytics/service-level/detail        — DFU-level performance
    GET  /analytics/service-level/chronic-misses — Streak-based alerts
    PUT  /analytics/service-level/targets       — Create/update SL targets (auth)
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from api.core import get_conn
from api.auth import require_api_key

router = APIRouter(tags=["service-level"])


class _TargetUpsert(BaseModel):
    abc_class: str
    item_id: str | None = None
    loc: str | None = None
    target_fill_rate: float
    effective_from: str | None = None


@router.get("/analytics/service-level/summary")
async def get_sl_summary(period: str | None = None):
    """Portfolio-level service level KPIs: avg fill rate, miss rate, by ABC class."""
    cond = "WHERE perf_month = %s" if period else ""
    params = [period] if period else []

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    COUNT(*)                                                   AS total_dfus,
                    AVG(actual_fill_rate)                                      AS avg_fill_rate,
                    AVG(target_fill_rate)                                      AS avg_target,
                    AVG(gap)                                                   AS avg_gap,
                    SUM(CASE WHEN gap < 0 THEN 1 ELSE 0 END)                  AS miss_count,
                    SUM(CASE WHEN flagged_for_review THEN 1 ELSE 0 END)        AS flagged_count,
                    MAX(perf_month)                                            AS latest_month
                FROM fact_service_level_performance
                {cond}
                """,
                params,
            )
            row = cur.fetchone()

    if not row:
        return {"total_dfus": 0, "avg_fill_rate": None}

    return {
        "total_dfus": row[0] or 0,
        "avg_fill_rate": float(row[1]) if row[1] is not None else None,
        "avg_target": float(row[2]) if row[2] is not None else None,
        "avg_gap": float(row[3]) if row[3] is not None else None,
        "miss_count": row[4] or 0,
        "flagged_count": row[5] or 0,
        "latest_month": row[6].isoformat() if row[6] else None,
        "period": period,
    }


@router.get("/analytics/service-level/detail")
async def get_sl_detail(
    item_id: str | None = None,
    loc: str | None = None,
    abc_class: str | None = None,
    page: int = 1,
    page_size: int = 50,
):
    """DFU-level service level performance."""
    page_size = max(1, min(page_size, 200))
    offset = (max(1, page) - 1) * page_size

    conditions = ["1=1"]
    params: list = []
    if item_id:
        conditions.append("item_id = %s")
        params.append(item_id)
    if loc:
        conditions.append("loc = %s")
        params.append(loc)
    if abc_class:
        conditions.append("abc_class = %s")
        params.append(abc_class)
    where = " AND ".join(conditions)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM fact_service_level_performance WHERE {where}",
                params,
            )
            total = cur.fetchone()[0]
            cur.execute(
                f"""
                SELECT item_id, loc, perf_month, abc_class,
                       actual_fill_rate, target_fill_rate, gap, gap_direction,
                       stockout_events, miss_streak_months, primary_miss_reason,
                       flagged_for_review
                FROM fact_service_level_performance
                WHERE {where}
                ORDER BY perf_month DESC, gap ASC
                LIMIT %s OFFSET %s
                """,
                params + [page_size, offset],
            )
            rows = cur.fetchall()

    cols = [
        "item_id", "loc", "perf_month", "abc_class",
        "actual_fill_rate", "target_fill_rate", "gap", "gap_direction",
        "stockout_events", "miss_streak_months", "primary_miss_reason",
        "flagged_for_review",
    ]
    items = []
    for r in rows:
        d = dict(zip(cols, r))
        if d.get("perf_month"):
            d["perf_month"] = d["perf_month"].isoformat()
        items.append(d)

    return {"total": total, "page": page, "performance": items}


@router.get("/analytics/service-level/chronic-misses")
async def get_chronic_misses(
    min_streak: int = 3,
    abc_class: str | None = None,
    page: int = 1,
    page_size: int = 50,
):
    """DFUs missing SL targets for N consecutive months."""
    page_size = max(1, min(page_size, 200))
    offset = (max(1, page) - 1) * page_size

    conditions = ["miss_streak_months >= %s"]
    params: list = [min_streak]
    if abc_class:
        conditions.append("abc_class = %s")
        params.append(abc_class)
    where = " AND ".join(conditions)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(DISTINCT item_id || '@' || loc) FROM fact_service_level_performance WHERE {where}",
                params,
            )
            total = cur.fetchone()[0]
            cur.execute(
                f"""
                SELECT item_id, loc, perf_month, abc_class,
                       actual_fill_rate, target_fill_rate, gap,
                       miss_streak_months, primary_miss_reason
                FROM fact_service_level_performance
                WHERE {where}
                ORDER BY miss_streak_months DESC, gap ASC
                LIMIT %s OFFSET %s
                """,
                params + [page_size, offset],
            )
            rows = cur.fetchall()

    cols = [
        "item_id", "loc", "perf_month", "abc_class",
        "actual_fill_rate", "target_fill_rate", "gap",
        "miss_streak_months", "primary_miss_reason",
    ]
    items = []
    for r in rows:
        d = dict(zip(cols, r))
        if d.get("perf_month"):
            d["perf_month"] = d["perf_month"].isoformat()
        items.append(d)

    return {"total": total, "min_streak": min_streak, "chronic_misses": items}


@router.put("/analytics/service-level/targets")
async def upsert_sl_target(body: _TargetUpsert, request: Request):
    """Create or update a service level target (auth required)."""
    await require_api_key(
        x_api_key=request.headers.get("x-api-key"),
        authorization=request.headers.get("authorization"),
    )

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO fact_service_level_targets
                    (abc_class, item_id, loc, target_fill_rate, effective_from)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (abc_class, COALESCE(item_id, ''), COALESCE(loc, ''))
                DO UPDATE SET
                    target_fill_rate = EXCLUDED.target_fill_rate,
                    effective_from   = EXCLUDED.effective_from
                """,
                (
                    body.abc_class,
                    body.item_id,
                    body.loc,
                    body.target_fill_rate,
                    body.effective_from,
                ),
            )
        conn.commit()

    return {
        "status": "ok",
        "abc_class": body.abc_class,
        "item_id": body.item_id,
        "loc": body.loc,
        "target_fill_rate": body.target_fill_rate,
    }
