"""Reporting & Distribution endpoints (Spec 08-08)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.core import get_conn
from common.auth import CurrentUser, get_current_user, require_role

router = APIRouter(prefix="/reports", tags=["reports"])


class CreateScheduleRequest(BaseModel):
    template_id: int
    recipients: list[str] = []
    cron: str = "0 8 * * 1"  # Weekly Monday 8am
    format: str = "pdf"


@router.get("/templates")
async def list_templates(user: CurrentUser = Depends(get_current_user)):
    """List available report templates."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT template_id, name, report_type, is_system, created_at
               FROM dim_report_template
               ORDER BY is_system DESC, name"""
        )
        rows = cur.fetchall()

    return {
        "templates": [
            {"template_id": r[0], "name": r[1], "report_type": r[2],
             "is_system": r[3], "created_at": r[4].isoformat() if r[4] else None}
            for r in rows
        ]
    }


@router.post("/generate")
async def generate_report(
    template_id: int = Query(...),
    format: str = Query("pdf", pattern="^(pdf|csv|html)$"),
    user: CurrentUser = Depends(require_role("planner")),
):
    """Generate a report ad-hoc (planner+ only). Returns report metadata."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT template_id, name, report_type, query_config FROM dim_report_template WHERE template_id = %s",
            (template_id,),
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Template not found")

    # In a full implementation, this would invoke the report engine
    return {
        "status": "queued",
        "template_id": row[0],
        "name": row[1],
        "format": format,
        "message": "Report generation queued. Check /reports/deliveries for status.",
    }


@router.get("/schedules")
async def list_schedules(user: CurrentUser = Depends(get_current_user)):
    """List report schedules."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT s.schedule_id, t.name, t.report_type, s.recipients,
                      s.cron, s.format, s.enabled, s.last_run_at, s.next_run_at
               FROM fact_report_schedule s
               JOIN dim_report_template t ON t.template_id = s.template_id
               ORDER BY s.created_at DESC"""
        )
        rows = cur.fetchall()

    return {
        "schedules": [
            {
                "schedule_id": r[0], "template_name": r[1], "report_type": r[2],
                "recipients": r[3], "cron": r[4], "format": r[5],
                "enabled": r[6],
                "last_run_at": r[7].isoformat() if r[7] else None,
                "next_run_at": r[8].isoformat() if r[8] else None,
            }
            for r in rows
        ]
    }


@router.post("/schedules", status_code=201)
async def create_schedule(
    body: CreateScheduleRequest,
    user: CurrentUser = Depends(require_role("manager")),
):
    """Create a report schedule (manager+ only)."""
    import json
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO fact_report_schedule (template_id, recipients, cron, format)
               VALUES (%s, %s, %s, %s) RETURNING schedule_id""",
            (body.template_id, json.dumps(body.recipients), body.cron, body.format),
        )
        schedule_id = cur.fetchone()[0]
        conn.commit()

    return {"schedule_id": schedule_id}


@router.delete("/schedules/{schedule_id}")
async def delete_schedule(
    schedule_id: int,
    user: CurrentUser = Depends(require_role("manager")),
):
    """Delete a report schedule."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM fact_report_schedule WHERE schedule_id = %s RETURNING schedule_id",
            (schedule_id,),
        )
        row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return {"deleted": True}


@router.get("/deliveries")
async def list_deliveries(
    limit: int = Query(50, ge=1, le=500),
    user: CurrentUser = Depends(get_current_user),
):
    """List report delivery history."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT d.delivery_id, t.name, d.status, d.file_path,
                      d.error, d.created_at, d.delivered_at
               FROM fact_report_delivery d
               JOIN fact_report_schedule s ON s.schedule_id = d.schedule_id
               JOIN dim_report_template t ON t.template_id = s.template_id
               ORDER BY d.created_at DESC LIMIT %s""",
            (limit,),
        )
        rows = cur.fetchall()

    return {
        "deliveries": [
            {
                "delivery_id": r[0], "template_name": r[1], "status": r[2],
                "file_path": r[3], "error": r[4],
                "created_at": r[5].isoformat() if r[5] else None,
                "delivered_at": r[6].isoformat() if r[6] else None,
            }
            for r in rows
        ]
    }
