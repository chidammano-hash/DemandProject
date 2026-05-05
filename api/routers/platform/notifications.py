"""Notification endpoints (Spec 08-04)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from api.auth import require_api_key
from api.core import get_conn
from common.auth import CurrentUser, get_current_user, require_role

router = APIRouter(prefix="/notifications", tags=["notifications"])


class TestNotificationRequest(BaseModel):
    channel: str = "email"
    recipient: str = ""
    message: str = "Test notification from Supply Chain Command Center"


@router.get("/history")
async def notification_history(
    event_type: str = Query("", description="Filter by event type"),
    status: str = Query("", description="Filter by status"),
    limit: int = Query(50, ge=1, le=500),
    user: CurrentUser = Depends(get_current_user),
):
    """View notification delivery history."""
    where = []
    params: list = []
    if event_type:
        where.append("event_type = %s")
        params.append(event_type)
    if status:
        where.append("status = %s")
        params.append(status)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT notification_id, event_type, severity, recipient,
                       subject, status, error, created_at, delivered_at
                FROM fact_notification_log
                {where_sql}
                ORDER BY created_at DESC LIMIT %s""",
            [*params, limit],
        )
        rows = cur.fetchall()

    return {
        "notifications": [
            {
                "notification_id": r[0], "event_type": r[1], "severity": r[2],
                "recipient": r[3], "subject": r[4], "status": r[5],
                "error": r[6],
                "created_at": r[7].isoformat() if r[7] else None,
                "delivered_at": r[8].isoformat() if r[8] else None,
            }
            for r in rows
        ]
    }


@router.get("/channels")
async def list_channels(user: CurrentUser = Depends(get_current_user)):
    """List configured notification channels."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT channel_id, channel_type, enabled, created_at FROM dim_notification_channel ORDER BY channel_id"
        )
        rows = cur.fetchall()

    return {
        "channels": [
            {"channel_id": r[0], "channel_type": r[1], "enabled": r[2],
             "created_at": r[3].isoformat() if r[3] else None}
            for r in rows
        ]
    }


@router.post("/test")
async def test_notification(
    body: TestNotificationRequest,
    admin: CurrentUser = Depends(require_role("manager")),
    api_key: str = Depends(require_api_key),
):
    """Send a test notification (manager+ only)."""
    from common.services.notification_engine import NotificationEngine
    engine = NotificationEngine()
    results = engine.send(
        event_type="test",
        severity="info",
        subject="Test Notification",
        body=body.message,
        recipient=body.recipient,
    )
    return {"results": results}
