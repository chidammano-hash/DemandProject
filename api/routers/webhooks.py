"""Webhook management endpoints (Spec 08-10)."""
from __future__ import annotations

import json
import secrets

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.auth import require_api_key
from api.core import get_conn
from common.auth import CurrentUser, get_current_user, require_role

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


class RegisterWebhookRequest(BaseModel):
    url: str
    event_types: list[str] = []


@router.post("/register", status_code=201)
async def register_webhook(
    body: RegisterWebhookRequest,
    user: CurrentUser = Depends(require_role("manager")),
    api_key: str = Depends(require_api_key),
):
    """Register a new webhook endpoint."""
    secret = secrets.token_hex(32)
    uid = user.user_id if user.user_id not in ("anonymous", "api-key-user") else None
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO dim_webhook_registration (url, secret, event_types, created_by)
               VALUES (%s, %s, %s, %s) RETURNING webhook_id""",
            (body.url, secret, json.dumps(body.event_types), uid),
        )
        webhook_id = cur.fetchone()[0]
        conn.commit()

    return {"webhook_id": webhook_id, "secret": secret}


@router.get("")
async def list_webhooks(user: CurrentUser = Depends(require_role("manager"))):
    """List registered webhooks."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT webhook_id, url, event_types, is_active, created_at
               FROM dim_webhook_registration ORDER BY created_at DESC"""
        )
        rows = cur.fetchall()

    return {
        "webhooks": [
            {"webhook_id": r[0], "url": r[1], "event_types": r[2],
             "is_active": r[3], "created_at": r[4].isoformat() if r[4] else None}
            for r in rows
        ]
    }


@router.post("/{webhook_id}/test")
async def test_webhook(
    webhook_id: int,
    user: CurrentUser = Depends(require_role("manager")),
    api_key: str = Depends(require_api_key),
):
    """Send a test event to a webhook."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT url, secret FROM dim_webhook_registration WHERE webhook_id = %s AND is_active = TRUE",
            (webhook_id,),
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Webhook not found or inactive")

    from common.webhook_dispatcher import dispatch_webhook
    result = dispatch_webhook(
        url=row[0], secret=row[1],
        event_type="test",
        payload={"message": "Test webhook from Supply Chain Command Center"},
    )
    return result


@router.delete("/{webhook_id}")
async def delete_webhook(
    webhook_id: int,
    user: CurrentUser = Depends(require_role("manager")),
    api_key: str = Depends(require_api_key),
):
    """Deactivate a webhook."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE dim_webhook_registration SET is_active = FALSE WHERE webhook_id = %s RETURNING webhook_id",
            (webhook_id,),
        )
        row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return {"webhook_id": webhook_id, "deactivated": True}


@router.get("/deliveries")
async def webhook_deliveries(
    webhook_id: int = Query(0, description="Filter by webhook ID"),
    limit: int = Query(50, ge=1, le=500),
    user: CurrentUser = Depends(require_role("manager")),
):
    """View webhook delivery history."""
    where = []
    params: list = []
    if webhook_id:
        where.append("d.webhook_id = %s")
        params.append(webhook_id)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT d.delivery_id, d.webhook_id, w.url, d.event_type,
                       d.status_code, d.attempt, d.status, d.created_at, d.delivered_at
                FROM fact_webhook_delivery d
                JOIN dim_webhook_registration w ON w.webhook_id = d.webhook_id
                {where_sql}
                ORDER BY d.created_at DESC LIMIT %s""",
            [*params, limit],
        )
        rows = cur.fetchall()

    return {
        "deliveries": [
            {
                "delivery_id": r[0], "webhook_id": r[1], "url": r[2],
                "event_type": r[3], "status_code": r[4], "attempt": r[5],
                "status": r[6],
                "created_at": r[7].isoformat() if r[7] else None,
                "delivered_at": r[8].isoformat() if r[8] else None,
            }
            for r in rows
        ]
    }
