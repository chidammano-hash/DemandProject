"""
F4.3 — Promotion & Event Planning API endpoints.

Endpoints:
    GET  /events/calendar                   — Event list with filters
    POST /events/calendar                   — Create new event (auth)
    GET  /events/calendar/{event_id}        — Event detail
    PUT  /events/calendar/{event_id}/approve — Approve event (auth)
    GET  /events/impact-preview             — Demand adjustment preview
    GET  /events/performance                — Post-event lift accuracy
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

from api.core import get_conn
from api.auth import require_api_key

router = APIRouter(tags=["events"])


class _EventCreate(BaseModel):
    event_type: str
    event_name: str
    event_start: str
    event_end: str
    uplift_pct: float = 0.0
    ramp_weeks: int = 1
    pantry_loading_pct: float = 0.0
    pantry_loading_weeks: int = 0
    priority: int = 5
    status: str = "draft"
    target_items: Optional[list] = None
    target_locations: Optional[list] = None
    target_categories: Optional[list] = None


@router.get("/events/calendar")
async def get_event_calendar(
    event_type: str | None = None,
    status: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    page: int = 1,
    page_size: int = 50,
):
    """List events from the event calendar with optional filters."""
    page_size = max(1, min(page_size, 200))
    offset = (max(1, page) - 1) * page_size

    conditions = ["1=1"]
    params: list = []
    if event_type:
        conditions.append("event_type = %s")
        params.append(event_type)
    if status:
        conditions.append("status = %s")
        params.append(status)
    if from_date:
        conditions.append("event_end >= %s")
        params.append(from_date)
    if to_date:
        conditions.append("event_start <= %s")
        params.append(to_date)
    where = " AND ".join(conditions)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM fact_event_calendar WHERE {where}", params
            )
            total = cur.fetchone()[0]
            cur.execute(
                f"""
                SELECT event_id, event_type, event_name, event_start, event_end,
                       uplift_pct, ramp_weeks, pantry_loading_pct,
                       priority, status, conflict_resolution, created_at
                FROM fact_event_calendar
                WHERE {where}
                ORDER BY event_start DESC
                LIMIT %s OFFSET %s
                """,
                params + [page_size, offset],
            )
            rows = cur.fetchall()

    cols = [
        "event_id", "event_type", "event_name", "event_start", "event_end",
        "uplift_pct", "ramp_weeks", "pantry_loading_pct",
        "priority", "status", "conflict_resolution", "created_at",
    ]
    events = []
    for r in rows:
        d = dict(zip(cols, r))
        for field in ("event_start", "event_end", "created_at"):
            if d.get(field) and hasattr(d[field], "isoformat"):
                d[field] = d[field].isoformat()
        events.append(d)

    return {"total": total, "page": page, "events": events}


@router.post("/events/calendar", status_code=201)
async def create_event(body: _EventCreate, request: Request):
    """Create a new promotion / event (auth required)."""
    await require_api_key(x_api_key=request.headers.get("x-api-key"))

    import json as _json
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO fact_event_calendar (
                    event_type, event_name, event_start, event_end,
                    uplift_pct, ramp_weeks, pantry_loading_pct, pantry_loading_weeks,
                    priority, status, target_items, target_locations, target_categories
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING event_id
                """,
                (
                    body.event_type, body.event_name, body.event_start, body.event_end,
                    body.uplift_pct, body.ramp_weeks, body.pantry_loading_pct,
                    body.pantry_loading_weeks, body.priority, body.status,
                    _json.dumps(body.target_items or []),
                    _json.dumps(body.target_locations or []),
                    _json.dumps(body.target_categories or []),
                ),
            )
            event_id = cur.fetchone()[0]
        conn.commit()

    return {"event_id": event_id, "status": body.status}


@router.get("/events/calendar/{event_id}")
async def get_event(event_id: int):
    """Event detail."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT event_id, event_type, event_name, event_start, event_end,
                       uplift_pct, ramp_weeks, pantry_loading_pct, pantry_loading_weeks,
                       priority, status, conflict_resolution, target_items,
                       target_locations, target_categories, created_at
                FROM fact_event_calendar WHERE event_id = %s
                """,
                (event_id,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(404, f"Event {event_id} not found")

    cols = [
        "event_id", "event_type", "event_name", "event_start", "event_end",
        "uplift_pct", "ramp_weeks", "pantry_loading_pct", "pantry_loading_weeks",
        "priority", "status", "conflict_resolution", "target_items",
        "target_locations", "target_categories", "created_at",
    ]
    d = dict(zip(cols, row))
    for field in ("event_start", "event_end", "created_at"):
        if d.get(field) and hasattr(d[field], "isoformat"):
            d[field] = d[field].isoformat()
    return d


@router.put("/events/calendar/{event_id}/approve")
async def approve_event(event_id: int, request: Request):
    """Approve an event (auth required)."""
    await require_api_key(x_api_key=request.headers.get("x-api-key"))

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE fact_event_calendar SET status = 'approved' WHERE event_id = %s RETURNING event_id",
                (event_id,),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(404, f"Event {event_id} not found")

    return {"event_id": event_id, "status": "approved"}


@router.get("/events/impact-preview")
async def get_event_impact(
    event_id: int | None = None,
    item_id: str | None = None,
    loc: str | None = None,
):
    """Demand adjustment preview for an event."""
    conditions = ["1=1"]
    params: list = []
    if event_id:
        conditions.append("event_id = %s")
        params.append(event_id)
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
                f"""
                SELECT item_id, loc, plan_month, event_id, base_forecast_qty,
                       event_adjustment_qty, post_promo_dip_qty, adjusted_forecast_qty,
                       adjustment_type, order_deadline
                FROM fact_event_adjusted_forecast
                WHERE {where}
                ORDER BY plan_month
                """,
                params,
            )
            rows = cur.fetchall()

    cols = [
        "item_id", "loc", "plan_month", "event_id", "base_forecast_qty",
        "event_adjustment_qty", "post_promo_dip_qty", "adjusted_forecast_qty",
        "adjustment_type", "order_deadline",
    ]
    adjustments = []
    for r in rows:
        d = dict(zip(cols, r))
        for field in ("plan_month", "order_deadline"):
            if d.get(field) and hasattr(d[field], "isoformat"):
                d[field] = d[field].isoformat()
        adjustments.append(d)

    return {
        "event_id": event_id,
        "item_id": item_id,
        "loc": loc,
        "adjustments": adjustments,
    }


@router.get("/events/performance")
async def get_event_performance(
    event_id: int | None = None,
    min_lift_accuracy: float | None = None,
    page: int = 1,
    page_size: int = 50,
):
    """Post-event lift accuracy and calibration factors."""
    page_size = max(1, min(page_size, 200))
    offset = (max(1, page) - 1) * page_size

    conditions = ["1=1"]
    params: list = []
    if event_id:
        conditions.append("event_id = %s")
        params.append(event_id)
    if min_lift_accuracy is not None:
        conditions.append("lift_accuracy_pct >= %s")
        params.append(min_lift_accuracy)
    where = " AND ".join(conditions)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM fact_event_performance WHERE {where}", params
            )
            total = cur.fetchone()[0]
            cur.execute(
                f"""
                SELECT event_id, item_id, loc, plan_month,
                       forecasted_lift_qty, actual_lift_qty, lift_accuracy_pct,
                       uplift_calibration_factor
                FROM fact_event_performance
                WHERE {where}
                ORDER BY event_id, plan_month
                LIMIT %s OFFSET %s
                """,
                params + [page_size, offset],
            )
            rows = cur.fetchall()

    cols = [
        "event_id", "item_id", "loc", "plan_month",
        "forecasted_lift_qty", "actual_lift_qty", "lift_accuracy_pct",
        "uplift_calibration_factor",
    ]
    items = []
    for r in rows:
        d = dict(zip(cols, r))
        if d.get("plan_month") and hasattr(d["plan_month"], "isoformat"):
            d["plan_month"] = d["plan_month"].isoformat()
        items.append(d)

    return {"total": total, "page": page, "performance": items}
