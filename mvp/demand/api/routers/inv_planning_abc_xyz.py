"""Inventory Planning — IPfeature11: ABC-XYZ Policy Matrix endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache

router = APIRouter(tags=["inv-planning"])


@router.get("/inv-planning/abc-xyz/matrix")
def get_abc_xyz_matrix(
    response: FastAPIResponse,
) -> dict:
    """9-cell ABC-XYZ matrix with DFU counts and avg service level per cell."""
    set_cache(response, max_age=3600)
    sql = """
        SELECT
            abc_vol, xyz_class,
            COUNT(*)                     AS dfu_count,
            AVG(abc_xyz_service_level)   AS avg_service_level,
            AVG(abc_xyz_dos_min)         AS avg_dos_min,
            AVG(abc_xyz_dos_max)         AS avg_dos_max
        FROM dim_dfu
        WHERE abc_vol IS NOT NULL AND xyz_class IS NOT NULL
        GROUP BY abc_vol, xyz_class
        ORDER BY abc_vol, xyz_class
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [])
            rows = cur.fetchall()

    cells = [
        {
            "abc_vol":           r[0],
            "xyz_class":         r[1],
            "segment":           (r[0] or "") + (r[1] or ""),
            "dfu_count":         int(r[2]),
            "avg_service_level": float(r[3]) if r[3] is not None else None,
            "avg_dos_min":       float(r[4]) if r[4] is not None else None,
            "avg_dos_max":       float(r[5]) if r[5] is not None else None,
        }
        for r in rows
    ]
    return {"cells": cells, "total_classified": sum(c["dfu_count"] for c in cells)}


@router.get("/inv-planning/abc-xyz/summary")
def get_abc_xyz_summary(
    response: FastAPIResponse,
) -> dict:
    """Portfolio-level ABC-XYZ classification summary."""
    set_cache(response, max_age=3600)
    sql = """
        SELECT
            COUNT(*)                                  AS total_dfus,
            COUNT(*) FILTER (WHERE xyz_class IS NOT NULL) AS classified_dfus,
            COUNT(*) FILTER (WHERE xyz_class = 'X')   AS x_count,
            COUNT(*) FILTER (WHERE xyz_class = 'Y')   AS y_count,
            COUNT(*) FILTER (WHERE xyz_class = 'Z')   AS z_count,
            AVG(demand_cv)                            AS avg_demand_cv,
            AVG(intermittency_ratio)                  AS avg_intermittency_ratio
        FROM dim_dfu
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [])
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]

    result = dict(zip(cols, row)) if row else {}
    return {k: (float(v) if isinstance(v, (int, float)) and v is not None else v)
            for k, v in result.items()}


@router.get("/inv-planning/abc-xyz/detail")
def get_abc_xyz_detail(
    response: FastAPIResponse,
    abc_vol: Optional[str] = Query(None, max_length=10),
    xyz_class: Optional[str] = Query(None, max_length=5),
    segment: Optional[str] = Query(None, max_length=10),
    item: Optional[str] = Query(None, max_length=120),
    location: Optional[str] = Query(None, max_length=120),
    brand: Optional[str] = Query(None, max_length=120),
    category: Optional[str] = Query(None, max_length=120),
    market: Optional[str] = Query(None, max_length=120),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict:
    """Paginated list of DFUs with their ABC-XYZ classification."""
    set_cache(response, max_age=600)

    where_clauses: list[str] = []
    params: list = []

    if abc_vol:
        params.append(abc_vol.upper())
        where_clauses.append("abc_vol = %s")
    if xyz_class:
        params.append(xyz_class.upper())
        where_clauses.append("xyz_class = %s")
    if segment:
        params.append(segment.upper())
        where_clauses.append("abc_xyz_segment = %s")
    if item:
        params.append(f"%{item}%")
        where_clauses.append("dmdunit ILIKE %s")
    if location:
        params.append(f"%{location}%")
        where_clauses.append("loc ILIKE %s")
    if brand:
        params.append(brand.split(","))
        where_clauses.append("EXISTS (SELECT 1 FROM dim_item di WHERE di.item_no = t.dmdunit AND di.brand_name = ANY(%s))")
    if category:
        params.append(category.split(","))
        where_clauses.append('EXISTS (SELECT 1 FROM dim_item di WHERE di.item_no = t.dmdunit AND di.class_ = ANY(%s))')
    if market:
        params.append(market.split(","))
        where_clauses.append("EXISTS (SELECT 1 FROM dim_location dl WHERE dl.loc = t.loc AND dl.state_id = ANY(%s))")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    count_sql = f"SELECT COUNT(*) FROM dim_dfu t {where_sql}"
    params.append(limit)
    params.append(offset)
    data_sql = f"""
        SELECT dmdunit, dmdgroup, loc,
               abc_vol, xyz_class, abc_xyz_segment,
               demand_cv, intermittency_ratio,
               abc_xyz_dos_min, abc_xyz_dos_max, abc_xyz_service_level
        FROM dim_dfu t
        {where_sql}
        ORDER BY abc_xyz_segment NULLS LAST, dmdunit
        LIMIT %s OFFSET %s
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, params[:-2])
            total = cur.fetchone()[0] or 0
            cur.execute(data_sql, params)
            rows = cur.fetchall()

    return {
        "total": int(total),
        "rows": [
            {
                "dmdunit":               r[0],
                "dmdgroup":              r[1],
                "loc":                   r[2],
                "abc_vol":               r[3],
                "xyz_class":             r[4],
                "abc_xyz_segment":       r[5],
                "demand_cv":             float(r[6]) if r[6] is not None else None,
                "intermittency_ratio":   float(r[7]) if r[7] is not None else None,
                "abc_xyz_dos_min":       float(r[8]) if r[8] is not None else None,
                "abc_xyz_dos_max":       float(r[9]) if r[9] is not None else None,
                "abc_xyz_service_level": float(r[10]) if r[10] is not None else None,
            }
            for r in rows
        ],
    }
