"""
F3.5 — Network / Multi-Echelon Planning API endpoints.

Endpoints:
    GET /supply/echelon/targets         — Echelon safety stock targets
    GET /supply/echelon/network         — Network structure (DC → stores)
    GET /supply/echelon/summary         — Portfolio network KPIs
    GET /supply/echelon/reorder-points  — Echelon reorder point recommendations
"""
from __future__ import annotations

from fastapi import APIRouter

from api.core import get_conn

router = APIRouter(tags=["echelon-planning"])


@router.get("/supply/echelon/network")
async def get_echelon_network():
    """Return the multi-echelon location hierarchy (parent → child links)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT parent_loc, child_loc, echelon_level, link_type,
                       replenishment_lead_time_days, is_active
                FROM dim_echelon_network
                WHERE is_active = TRUE
                ORDER BY echelon_level, parent_loc, child_loc
            """)
            rows = cur.fetchall()

    cols = [
        "parent_loc", "child_loc", "echelon_level", "link_type",
        "replenishment_lead_time_days", "is_active",
    ]
    return {"nodes": [dict(zip(cols, r)) for r in rows]}


@router.get("/supply/echelon/targets")
async def get_echelon_targets(
    item_id: str | None = None,
    loc: str | None = None,
    echelon_level: int | None = None,
    page: int = 1,
    page_size: int = 50,
):
    """Return network-optimised safety stock targets per echelon."""
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
    if echelon_level is not None:
        conditions.append("echelon_level = %s")
        params.append(echelon_level)
    where = " AND ".join(conditions)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM fact_echelon_ss_targets WHERE {where}", params
            )
            total = cur.fetchone()[0] or 0
            cur.execute(
                f"""
                SELECT item_id, loc, echelon_level,
                       echelon_ss_qty, standalone_ss_qty, pooling_benefit_pct,
                       service_level_target, computed_at
                FROM fact_echelon_ss_targets
                WHERE {where}
                ORDER BY echelon_level, item_id, loc
                LIMIT %s OFFSET %s
                """,
                params + [page_size, offset],
            )
            rows = cur.fetchall()

    cols = [
        "item_id", "loc", "echelon_level",
        "echelon_ss_qty", "standalone_ss_qty", "pooling_benefit_pct",
        "service_level_target", "computed_at",
    ]
    items = []
    for r in rows:
        d = dict(zip(cols, r))
        if d.get("computed_at"):
            d["computed_at"] = d["computed_at"].isoformat()
        items.append(d)

    return {"total": total, "page": page, "targets": items}


@router.get("/supply/echelon/summary")
async def get_echelon_summary():
    """Portfolio pooling benefit and echelon coverage KPIs."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(DISTINCT item_id || '@' || loc)      AS total_sku_locs,
                    AVG(pooling_benefit_pct)                   AS avg_pooling_benefit_pct,
                    SUM(standalone_ss_qty - echelon_ss_qty)    AS total_units_saved,
                    COUNT(DISTINCT echelon_level)              AS echelon_depth,
                    MAX(computed_at)                           AS last_computed_at
                FROM fact_echelon_ss_targets
            """)
            row = cur.fetchone()

    if not row or not row[0]:
        return {"total_sku_locs": 0, "avg_pooling_benefit_pct": None}

    return {
        "total_sku_locs": row[0],
        "avg_pooling_benefit_pct": float(row[1]) if row[1] is not None else None,
        "total_units_saved": float(row[2]) if row[2] is not None else None,
        "echelon_depth": row[3],
        "last_computed_at": row[4].isoformat() if row[4] else None,
    }


@router.get("/supply/echelon/reorder-points")
async def get_echelon_reorder_points(
    item_id: str | None = None,
    loc: str | None = None,
    page: int = 1,
    page_size: int = 50,
):
    """Echelon reorder points (ROP) with cascade risk flag."""
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
    where = " AND ".join(conditions)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM fact_echelon_reorder_points WHERE {where}", params
            )
            total = cur.fetchone()[0] or 0
            cur.execute(
                f"""
                SELECT item_id, loc, echelon_level, reorder_point_qty,
                       echelon_ss_qty, demand_during_lt_qty,
                       cascade_risk_flag, computed_at
                FROM fact_echelon_reorder_points
                WHERE {where}
                ORDER BY cascade_risk_flag DESC, echelon_level
                LIMIT %s OFFSET %s
                """,
                params + [page_size, offset],
            )
            rows = cur.fetchall()

    cols = [
        "item_id", "loc", "echelon_level", "reorder_point_qty",
        "echelon_ss_qty", "demand_during_lt_qty",
        "cascade_risk_flag", "computed_at",
    ]
    items = []
    for r in rows:
        d = dict(zip(cols, r))
        if d.get("computed_at"):
            d["computed_at"] = d["computed_at"].isoformat()
        items.append(d)

    return {"total": total, "page": page, "reorder_points": items}
