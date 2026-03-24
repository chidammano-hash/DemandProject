"""Sourcing endpoints — item-location supply source mapping."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache, qident

router = APIRouter(prefix="/sourcing", tags=["sourcing"])


@router.get("/rows")
def sourcing_rows(
    response: FastAPIResponse,
    item: str = Query(default="", max_length=120),
    loc: str = Query(default="", max_length=120),
    supplier: str = Query(default="", max_length=120),
    transit_mode: str = Query(default="", max_length=60),
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    sort_by: str = Query(default="item_id", max_length=60),
    sort_dir: str = Query(default="asc", max_length=4),
):
    """Paginated sourcing rows with optional filters."""
    set_cache(response, max_age=120)

    allowed_sort = {"item_id", "loc", "source_cd", "supplier_id", "plant_id", "transit_mode", "site_id"}
    order_col = sort_by if sort_by in allowed_sort else "item_id"
    order_dir = "DESC" if sort_dir.lower() == "desc" else "ASC"

    where_parts: list[str] = []
    params: list[Any] = []
    if item.strip():
        where_parts.append("item_id ILIKE %s")
        params.append(f"%{item.strip()}%")
    if loc.strip():
        where_parts.append("loc ILIKE %s")
        params.append(f"%{loc.strip()}%")
    if supplier.strip():
        where_parts.append("(supplier_id ILIKE %s OR source_cd ILIKE %s)")
        params.extend([f"%{supplier.strip()}%", f"%{supplier.strip()}%"])
    if transit_mode.strip():
        where_parts.append("transit_mode ILIKE %s")
        params.append(f"%{transit_mode.strip()}%")

    where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM dim_sourcing {where_sql}", params)
        total = cur.fetchone()[0]

        cur.execute(
            f"""SELECT sourcing_ck, site_id, item_id, loc, source_cd,
                       transit_mode, supplier_id, plant_id
                FROM dim_sourcing {where_sql}
                ORDER BY {qident(order_col)} {order_dir}
                LIMIT %s OFFSET %s""",
            [*params, limit, offset],
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    return {"total": total, "rows": rows}


@router.get("/search")
def sourcing_search(
    response: FastAPIResponse,
    q: str = Query(min_length=1, max_length=200),
    limit: int = Query(default=20, ge=1, le=200),
):
    """Full-text search across sourcing fields."""
    set_cache(response, max_age=60)
    pattern = f"%{q.strip()}%"
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT sourcing_ck, site_id, item_id, loc, source_cd,
                      transit_mode, supplier_id, plant_id
               FROM dim_sourcing
               WHERE item_id ILIKE %s OR loc ILIKE %s
                  OR source_cd ILIKE %s OR supplier_id ILIKE %s
                  OR transit_mode ILIKE %s
               LIMIT %s""",
            [pattern, pattern, pattern, pattern, pattern, limit],
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return {"rows": rows}


@router.get("/by-item/{item_id}")
def sourcing_by_item(item_id: str, response: FastAPIResponse):
    """All supply sources for a given item."""
    set_cache(response, max_age=120)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT sourcing_ck, site_id, item_id, loc, source_cd,
                      transit_mode, supplier_id, plant_id
               FROM dim_sourcing WHERE item_id = %s
               ORDER BY loc""",
            [item_id],
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return {"rows": rows, "total": len(rows)}


@router.get("/by-supplier/{supplier_id}")
def sourcing_by_supplier(supplier_id: str, response: FastAPIResponse):
    """All items sourced from a given supplier."""
    set_cache(response, max_age=120)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT sourcing_ck, site_id, item_id, loc, source_cd,
                      transit_mode, supplier_id, plant_id
               FROM dim_sourcing WHERE supplier_id = %s
               ORDER BY item_id, loc""",
            [supplier_id],
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return {"rows": rows, "total": len(rows)}


@router.get("/network")
def sourcing_network(response: FastAPIResponse):
    """Supply network summary: supplier counts, transit mode distribution, single-source risk."""
    set_cache(response, max_age=300)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM dim_sourcing")
        total_rows = cur.fetchone()[0]

        cur.execute("SELECT count(DISTINCT supplier_id) FROM dim_sourcing WHERE supplier_id IS NOT NULL")
        supplier_count = cur.fetchone()[0]

        cur.execute("SELECT count(DISTINCT item_id || '_' || loc) FROM dim_sourcing")
        item_loc_count = cur.fetchone()[0]

        cur.execute(
            """SELECT transit_mode, count(*) AS cnt
               FROM dim_sourcing
               GROUP BY transit_mode
               ORDER BY cnt DESC"""
        )
        transit_modes = [{"transit_mode": r[0], "count": r[1]} for r in cur.fetchall()]

        cur.execute(
            """SELECT count(*) FROM (
                   SELECT item_id, loc
                   FROM dim_sourcing
                   GROUP BY item_id, loc
                   HAVING count(DISTINCT source_cd) = 1
               ) single"""
        )
        single_source = cur.fetchone()[0]

    return {
        "total_rows": total_rows,
        "supplier_count": supplier_count,
        "item_location_count": item_loc_count,
        "single_source_count": single_source,
        "multi_source_count": item_loc_count - single_source,
        "transit_modes": transit_modes,
    }
