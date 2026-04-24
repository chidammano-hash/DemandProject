"""Authentication endpoints: login, refresh, me, logout."""
from __future__ import annotations

import logging

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.core import get_conn

logger = logging.getLogger(__name__)
from common.auth import (
    CurrentUser,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    log_audit,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: dict


class RefreshRequest(BaseModel):
    refresh_token: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    """Authenticate with email + password, return JWT tokens."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT user_id, email, display_name, role, password_hash, is_active "
            "FROM dim_user WHERE email = %s",
            (body.email,),
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user_id, email, display_name, role, password_hash, is_active = row

    if not is_active:
        raise HTTPException(status_code=401, detail="Account disabled")

    if not verify_password(body.password, password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    uid = str(user_id)
    access = create_access_token(uid, email, role)
    refresh = create_refresh_token(uid)

    # Update last_login_at
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE dim_user SET last_login_at = now() WHERE user_id = %s",
                (uid,),
            )
            conn.commit()
    except psycopg.Error:
        logger.exception("Failed to update last_login_at for user %s", uid)

    log_audit(
        CurrentUser(user_id=uid, email=email, role=role),
        action="login",
        resource_type="session",
    )

    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        user={
            "user_id": uid,
            "email": email,
            "display_name": display_name,
            "role": role,
        },
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(body: RefreshRequest):
    """Exchange a valid refresh token for new access + refresh tokens."""
    payload = decode_token(body.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")

    uid = payload["sub"]
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT user_id, email, display_name, role, is_active "
            "FROM dim_user WHERE user_id = %s",
            (uid,),
        )
        row = cur.fetchone()

    if not row or not row[4]:
        raise HTTPException(status_code=401, detail="User not found or disabled")

    user_id, email, display_name, role, _ = row
    access = create_access_token(str(user_id), email, role)
    new_refresh = create_refresh_token(str(user_id))

    return TokenResponse(
        access_token=access,
        refresh_token=new_refresh,
        user={
            "user_id": str(user_id),
            "email": email,
            "display_name": display_name,
            "role": role,
        },
    )


@router.get("/me")
async def get_me(user: CurrentUser = Depends(get_current_user)):
    """Return the current authenticated user's profile."""
    if user.user_id in ("anonymous", "api-key-user"):
        return {
            "user_id": user.user_id,
            "email": user.email,
            "display_name": user.role.title(),
            "role": user.role,
        }

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT user_id, email, display_name, role, is_active, created_at, last_login_at "
            "FROM dim_user WHERE user_id = %s",
            (user.user_id,),
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "user_id": str(row[0]),
        "email": row[1],
        "display_name": row[2],
        "role": row[3],
        "is_active": row[4],
        "created_at": row[5].isoformat() if row[5] else None,
        "last_login_at": row[6].isoformat() if row[6] else None,
    }


@router.post("/logout")
async def logout(user: CurrentUser = Depends(get_current_user)):
    """Log out (client-side token disposal). Server records the event."""
    log_audit(user, action="logout", resource_type="session")
    return {"message": "Logged out"}
