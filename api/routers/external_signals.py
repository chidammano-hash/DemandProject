"""External Demand Signals & Decomposition endpoints (Spec 08-06)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import require_api_key
from api.core import get_conn
from common.auth import CurrentUser, get_current_user, require_role

router = APIRouter(prefix="/demand-signals/external", tags=["demand-signals"])


@router.get("")
async def list_external_signals(
    item_no: str = Query("", description="Filter by item"),
    loc: str = Query("", description="Filter by location"),
    days: int = Query(90, ge=1, le=365),
    limit: int = Query(100, ge=1, le=1000),
):
    """List external demand signals."""
    where = ["signal_date >= now() - interval '%s days'" % days]
    params: list = []
    if item_no:
        where.append("item_no ILIKE %s")
        params.append(f"%{item_no}%")
    if loc:
        where.append("loc ILIKE %s")
        params.append(f"%{loc}%")

    where_sql = "WHERE " + " AND ".join(where)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT s.signal_id, src.name AS source_name, s.signal_date,
                       s.item_no, s.loc, s.signal_type, s.signal_value,
                       s.confidence, s.created_at
                FROM fact_external_signal s
                JOIN dim_external_signal_source src ON src.source_id = s.source_id
                {where_sql}
                ORDER BY s.signal_date DESC LIMIT %s""",
            [*params, limit],
        )
        rows = cur.fetchall()

    return {
        "signals": [
            {
                "signal_id": r[0], "source_name": r[1],
                "signal_date": r[2].isoformat() if r[2] else None,
                "item_no": r[3], "loc": r[4],
                "signal_type": r[5],
                "signal_value": float(r[6]) if r[6] is not None else None,
                "confidence": float(r[7]) if r[7] is not None else None,
                "created_at": r[8].isoformat() if r[8] else None,
            }
            for r in rows
        ]
    }


@router.get("/sources")
async def list_signal_sources():
    """List configured external signal sources."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT source_id, name, source_type, enabled,
                      refresh_interval_hours, last_refresh_at
               FROM dim_external_signal_source
               ORDER BY name"""
        )
        rows = cur.fetchall()

    return {
        "sources": [
            {
                "source_id": r[0], "name": r[1], "source_type": r[2],
                "enabled": r[3], "refresh_interval_hours": r[4],
                "last_refresh_at": r[5].isoformat() if r[5] else None,
            }
            for r in rows
        ]
    }


@router.post("/sources/{source_id}/refresh")
async def refresh_source(
    source_id: int,
    user: CurrentUser = Depends(require_role("manager")),
    api_key: str = Depends(require_api_key),
):
    """Trigger a manual refresh of an external signal source."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT source_id, name, source_type FROM dim_external_signal_source WHERE source_id = %s",
            (source_id,),
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Source not found")

    # In full implementation, this would trigger the signal ingestion script
    return {"source_id": source_id, "name": row[1], "status": "refresh_queued"}


@router.get("/decomposition")
async def demand_decomposition(
    item_no: str = Query(..., description="Item number"),
    loc: str = Query(..., description="Location"),
):
    """Get demand decomposition (base, trend, seasonal, promotional, external)."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT month, base_demand, trend_component, seasonal_component,
                      promotional_uplift, external_signal_effect, residual
               FROM mv_demand_decomposition
               WHERE item_no = %s AND loc = %s
               ORDER BY month""",
            (item_no, loc),
        )
        rows = cur.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="No decomposition data found")

    return {
        "item_no": item_no,
        "loc": loc,
        "decomposition": [
            {
                "month": r[0].isoformat() if r[0] else None,
                "base_demand": float(r[1]) if r[1] is not None else None,
                "trend": float(r[2]) if r[2] is not None else None,
                "seasonal": float(r[3]) if r[3] is not None else None,
                "promotional": float(r[4]) if r[4] is not None else None,
                "external": float(r[5]) if r[5] is not None else None,
                "residual": float(r[6]) if r[6] is not None else None,
            }
            for r in rows
        ],
    }
