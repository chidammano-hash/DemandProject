"""User management endpoints (admin-only)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.auth import require_api_key
from api.core import get_conn
from common.auth import (
    CurrentUser,
    hash_password,
    log_audit,
    require_role,
)

router = APIRouter(prefix="/users", tags=["users"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class CreateUserRequest(BaseModel):
    email: str
    display_name: str = ""
    role: str = "viewer"
    password: str


class UpdateUserRequest(BaseModel):
    display_name: str | None = None
    role: str | None = None
    is_active: bool | None = None
    password: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("")
async def list_users(
    admin: CurrentUser = Depends(require_role("admin")),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List all users (admin only)."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM dim_user")
        total = cur.fetchone()[0]
        cur.execute(
            "SELECT user_id, email, display_name, role, is_active, created_at, last_login_at "
            "FROM dim_user ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (limit, offset),
        )
        rows = cur.fetchall()

    return {
        "total": total,
        "users": [
            {
                "user_id": str(r[0]),
                "email": r[1],
                "display_name": r[2],
                "role": r[3],
                "is_active": r[4],
                "created_at": r[5].isoformat() if r[5] else None,
                "last_login_at": r[6].isoformat() if r[6] else None,
            }
            for r in rows
        ],
    }


@router.post("", status_code=201)
async def create_user(
    body: CreateUserRequest,
    admin: CurrentUser = Depends(require_role("admin")),
    api_key: str = Depends(require_api_key),
):
    """Create a new user (admin only)."""
    valid_roles = {"viewer", "planner", "manager", "admin"}
    if body.role not in valid_roles:
        raise HTTPException(status_code=422, detail=f"Invalid role: {body.role}")
    if len(body.password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters")

    pw_hash = hash_password(body.password)
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO dim_user (email, display_name, role, password_hash) "
                "VALUES (%s, %s, %s, %s) RETURNING user_id",
                (body.email, body.display_name, body.role, pw_hash),
            )
            user_id = cur.fetchone()[0]
            conn.commit()
    except Exception as e:
        if "unique" in str(e).lower() or "duplicate" in str(e).lower():
            raise HTTPException(status_code=409, detail="Email already exists")
        raise

    log_audit(
        admin,
        action="create",
        resource_type="user",
        resource_id=str(user_id),
        new_value={"email": body.email, "role": body.role},
    )

    return {"user_id": str(user_id), "email": body.email, "role": body.role}


@router.put("/{user_id}")
async def update_user(
    user_id: str,
    body: UpdateUserRequest,
    admin: CurrentUser = Depends(require_role("admin")),
    api_key: str = Depends(require_api_key),
):
    """Update a user's profile (admin only)."""
    updates = []
    params = []

    if body.display_name is not None:
        updates.append("display_name = %s")
        params.append(body.display_name)
    if body.role is not None:
        valid_roles = {"viewer", "planner", "manager", "admin"}
        if body.role not in valid_roles:
            raise HTTPException(status_code=422, detail=f"Invalid role: {body.role}")
        updates.append("role = %s")
        params.append(body.role)
    if body.is_active is not None:
        updates.append("is_active = %s")
        params.append(body.is_active)
    if body.password is not None:
        if len(body.password) < 8:
            raise HTTPException(status_code=422, detail="Password must be at least 8 characters")
        updates.append("password_hash = %s")
        params.append(hash_password(body.password))

    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    params.append(user_id)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE dim_user SET {', '.join(updates)} WHERE user_id = %s RETURNING user_id",
            params,
        )
        row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="User not found")

    log_audit(
        admin,
        action="update",
        resource_type="user",
        resource_id=user_id,
        new_value=body.model_dump(exclude_none=True, exclude={"password"}),
    )

    return {"user_id": user_id, "updated": True}


@router.get("/audit-log")
async def get_audit_log(
    admin: CurrentUser = Depends(require_role("manager")),
    resource_type: str = Query("", description="Filter by resource type"),
    user_id: str = Query("", description="Filter by user_id"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """View audit log (manager+ only)."""
    where = []
    params = []
    if resource_type:
        where.append("a.resource_type = %s")
        params.append(resource_type)
    if user_id:
        where.append("a.user_id = %s::uuid")
        params.append(user_id)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM fact_audit_log a {where_sql}", params)
        total = cur.fetchone()[0]
        cur.execute(
            f"""SELECT a.audit_id, a.user_id, u.email, u.display_name,
                       a.action, a.resource_type, a.resource_id,
                       a.old_value, a.new_value, a.created_at
                FROM fact_audit_log a
                LEFT JOIN dim_user u ON u.user_id = a.user_id
                {where_sql}
                ORDER BY a.created_at DESC
                LIMIT %s OFFSET %s""",
            [*params, limit, offset],
        )
        rows = cur.fetchall()

    return {
        "total": total,
        "entries": [
            {
                "audit_id": r[0],
                "user_id": str(r[1]) if r[1] else None,
                "email": r[2],
                "display_name": r[3],
                "action": r[4],
                "resource_type": r[5],
                "resource_id": r[6],
                "old_value": r[7],
                "new_value": r[8],
                "created_at": r[9].isoformat() if r[9] else None,
            }
            for r in rows
        ],
    }
