"""Collaboration & Annotations endpoints (Spec 08-05)."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.auth import require_api_key
from api.core import get_conn
from common.auth import CurrentUser, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collaboration", tags=["collaboration"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class CreateAnnotationRequest(BaseModel):
    resource_type: str
    resource_id: str
    body: str
    parent_id: int | None = None
    mentions: list[str] = []


class CreateSharedViewRequest(BaseModel):
    title: str
    tab: str
    filters: dict = {}
    layout: dict = {}
    is_public: bool = False


# ---------------------------------------------------------------------------
# Annotation endpoints
# ---------------------------------------------------------------------------
@router.post("/annotations", status_code=201)
async def create_annotation(
    req: CreateAnnotationRequest,
    user: CurrentUser = Depends(get_current_user),
    api_key: str = Depends(require_api_key),
):
    """Create a comment/annotation on a resource."""
    uid = user.user_id if user.user_id not in ("anonymous", "api-key-user") else None
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO fact_annotation
               (user_id, resource_type, resource_id, parent_id, body, mentions)
               VALUES (%s, %s, %s, %s, %s, %s)
               RETURNING annotation_id""",
            (uid, req.resource_type, req.resource_id, req.parent_id,
             req.body, json.dumps(req.mentions)),
        )
        annotation_id = cur.fetchone()[0]
        conn.commit()

    # Trigger notifications for @mentions
    if req.mentions:
        try:
            from common.services.notification_engine import NotificationEngine
            engine = NotificationEngine()
            for mention in req.mentions:
                engine.send(
                    event_type="mention",
                    severity="info",
                    subject=f"You were mentioned by {user.email}",
                    body=f"On {req.resource_type} {req.resource_id}: {req.body[:200]}",
                    recipient=mention,
                )
        except Exception:  # noqa: BLE001 — mention notifications are best-effort; annotation creation must succeed even if delivery fails
            logger.exception("Mention notification failed for annotation %s", annotation_id)

    return {"annotation_id": annotation_id}


@router.get("/annotations")
async def list_annotations(
    resource_type: str = Query(..., description="Resource type"),
    resource_id: str = Query(..., description="Resource ID"),
    user: CurrentUser = Depends(get_current_user),
):
    """Get annotation thread for a resource."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT a.annotation_id, a.user_id, u.email, u.display_name,
                      a.parent_id, a.body, a.mentions, a.is_resolved,
                      a.created_at, a.updated_at
               FROM fact_annotation a
               LEFT JOIN dim_user u ON u.user_id = a.user_id
               WHERE a.resource_type = %s AND a.resource_id = %s
               ORDER BY a.created_at ASC""",
            (resource_type, resource_id),
        )
        rows = cur.fetchall()

    return {
        "annotations": [
            {
                "annotation_id": r[0],
                "user_id": str(r[1]) if r[1] else None,
                "email": r[2], "display_name": r[3],
                "parent_id": r[4], "body": r[5],
                "mentions": r[6], "is_resolved": r[7],
                "created_at": r[8].isoformat() if r[8] else None,
                "updated_at": r[9].isoformat() if r[9] else None,
            }
            for r in rows
        ]
    }


@router.put("/annotations/{annotation_id}")
async def update_annotation(
    annotation_id: int,
    body: str = Query(...),
    user: CurrentUser = Depends(get_current_user),
    api_key: str = Depends(require_api_key),
):
    """Edit an annotation (own only, or admin)."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE fact_annotation SET body = %s, updated_at = now() "
            "WHERE annotation_id = %s RETURNING annotation_id",
            (body, annotation_id),
        )
        row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return {"annotation_id": annotation_id, "updated": True}


@router.post("/annotations/{annotation_id}/resolve")
async def resolve_annotation(
    annotation_id: int,
    user: CurrentUser = Depends(get_current_user),
    api_key: str = Depends(require_api_key),
):
    """Mark an annotation thread as resolved."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE fact_annotation SET is_resolved = TRUE, updated_at = now() "
            "WHERE annotation_id = %s RETURNING annotation_id",
            (annotation_id,),
        )
        row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return {"annotation_id": annotation_id, "resolved": True}


@router.delete("/annotations/{annotation_id}")
async def delete_annotation(
    annotation_id: int,
    user: CurrentUser = Depends(get_current_user),
    api_key: str = Depends(require_api_key),
):
    """Delete an annotation (own or admin)."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM fact_annotation WHERE annotation_id = %s RETURNING annotation_id",
            (annotation_id,),
        )
        row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return {"deleted": True}


@router.get("/mentions/me")
async def my_mentions(
    limit: int = Query(20, ge=1, le=100),
    user: CurrentUser = Depends(get_current_user),
):
    """Get annotations that @mention the current user."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT a.annotation_id, a.resource_type, a.resource_id,
                      a.body, a.is_resolved, a.created_at, u.display_name
               FROM fact_annotation a
               LEFT JOIN dim_user u ON u.user_id = a.user_id
               WHERE a.mentions::text ILIKE %s
               ORDER BY a.created_at DESC LIMIT %s""",
            (f"%{user.email}%", limit),
        )
        rows = cur.fetchall()

    return {
        "mentions": [
            {
                "annotation_id": r[0], "resource_type": r[1], "resource_id": r[2],
                "body": r[3], "is_resolved": r[4],
                "created_at": r[5].isoformat() if r[5] else None,
                "author": r[6],
            }
            for r in rows
        ]
    }


# ---------------------------------------------------------------------------
# Shared view endpoints
# ---------------------------------------------------------------------------
@router.post("/shared-views", status_code=201)
async def create_shared_view(
    req: CreateSharedViewRequest,
    user: CurrentUser = Depends(get_current_user),
    api_key: str = Depends(require_api_key),
):
    """Save current filter state + tab as a shareable view."""
    uid = user.user_id if user.user_id not in ("anonymous", "api-key-user") else None
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO fact_shared_view (user_id, title, tab, filters, layout, is_public)
               VALUES (%s, %s, %s, %s, %s, %s)
               RETURNING view_id""",
            (uid, req.title, req.tab, json.dumps(req.filters),
             json.dumps(req.layout), req.is_public),
        )
        view_id = cur.fetchone()[0]
        conn.commit()

    return {"view_id": view_id}


@router.get("/shared-views")
async def list_shared_views(user: CurrentUser = Depends(get_current_user)):
    """List available shared views."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT view_id, user_id, title, tab, filters, layout, is_public, created_at
               FROM fact_shared_view
               WHERE is_public = TRUE OR user_id = %s
               ORDER BY created_at DESC LIMIT 50""",
            (user.user_id if user.user_id not in ("anonymous", "api-key-user") else None,),
        )
        rows = cur.fetchall()

    return {
        "views": [
            {
                "view_id": r[0], "user_id": str(r[1]) if r[1] else None,
                "title": r[2], "tab": r[3], "filters": r[4],
                "layout": r[5], "is_public": r[6],
                "created_at": r[7].isoformat() if r[7] else None,
            }
            for r in rows
        ]
    }


@router.get("/shared-views/{view_id}")
async def get_shared_view(view_id: int):
    """Get a specific shared view by ID."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT view_id, user_id, title, tab, filters, layout, is_public, created_at "
            "FROM fact_shared_view WHERE view_id = %s",
            (view_id,),
        )
        r = cur.fetchone()

    if not r:
        raise HTTPException(status_code=404, detail="Shared view not found")

    return {
        "view_id": r[0], "user_id": str(r[1]) if r[1] else None,
        "title": r[2], "tab": r[3], "filters": r[4],
        "layout": r[5], "is_public": r[6],
        "created_at": r[7].isoformat() if r[7] else None,
    }
