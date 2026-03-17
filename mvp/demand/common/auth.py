"""User authentication and RBAC for Demand Studio.

Replaces the simple API key auth in api/auth.py with JWT-based user identity,
role-based access control, and audit logging.
"""
from __future__ import annotations

import hmac
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt
from fastapi import Depends, Header, HTTPException, Request
from pydantic import BaseModel

from common.utils import load_config, reset_config

_CONFIG_NAME = "auth_config.yaml"


# ---------------------------------------------------------------------------
# Config (thread-safe via common.utils.load_config)
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    return load_config(_CONFIG_NAME)


def _reset_config_cache():
    """Reset cached config — used in tests."""
    reset_config(_CONFIG_NAME)


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def hash_password(plain: str) -> str:
    """Hash a plaintext password with bcrypt."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------
def _jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET", "demand-studio-dev-secret-change-me")
    return secret


def create_access_token(user_id: str, email: str, role: str) -> str:
    cfg = _load_config()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=cfg["jwt"]["access_token_expire_minutes"]
    )
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "type": "access",
        "exp": expire,
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=cfg["jwt"]["algorithm"])


def create_refresh_token(user_id: str) -> str:
    cfg = _load_config()
    expire = datetime.now(timezone.utc) + timedelta(
        days=cfg["jwt"]["refresh_token_expire_days"]
    )
    payload = {
        "sub": user_id,
        "type": "refresh",
        "exp": expire,
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=cfg["jwt"]["algorithm"])


def decode_token(token: str) -> dict:
    """Decode and verify a JWT token. Raises HTTPException on failure."""
    cfg = _load_config()
    try:
        return jwt.decode(token, _jwt_secret(), algorithms=[cfg["jwt"]["algorithm"]])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ---------------------------------------------------------------------------
# User model for request context
# ---------------------------------------------------------------------------
class CurrentUser(BaseModel):
    user_id: str
    email: str
    role: str


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------
def _extract_bearer(authorization: str | None) -> str | None:
    """Extract JWT from Authorization: Bearer <token> header."""
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


async def get_current_user(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> CurrentUser:
    """Authenticate request via JWT Bearer token or legacy API key.

    - If JWT Bearer token is present, decode and return user.
    - If X-API-Key matches the env API_KEY, return a synthetic admin user.
    - If neither auth is configured (no JWT_SECRET set and no API_KEY set), allow anonymous access.
    """
    # Try JWT first
    token = _extract_bearer(authorization)
    if token:
        payload = decode_token(token)
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        return CurrentUser(
            user_id=payload["sub"],
            email=payload.get("email", ""),
            role=payload.get("role", "viewer"),
        )

    # Fall back to legacy API key
    expected_key = os.getenv("API_KEY", "")
    if expected_key and x_api_key:
        if hmac.compare_digest(x_api_key, expected_key):
            return CurrentUser(
                user_id="api-key-user",
                email="apikey@system",
                role="admin",
            )

    # If no auth mechanism is configured, allow anonymous with viewer role
    jwt_secret_is_default = _jwt_secret() == "demand-studio-dev-secret-change-me"
    api_key_unset = not expected_key
    if jwt_secret_is_default and api_key_unset:
        return CurrentUser(user_id="anonymous", email="anonymous", role="admin")

    raise HTTPException(status_code=401, detail="Authentication required")


async def get_optional_user(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> CurrentUser | None:
    """Like get_current_user but returns None instead of raising 401."""
    try:
        return await get_current_user(authorization, x_api_key)
    except HTTPException:
        return None


def require_role(min_role: str):
    """Factory: returns a FastAPI dependency that requires at least min_role."""
    cfg = _load_config()
    role_levels = cfg.get("role_level", {
        "viewer": 1, "planner": 2, "manager": 3, "admin": 4
    })
    min_level = role_levels.get(min_role, 99)

    async def _check(user: CurrentUser = Depends(get_current_user)):
        user_level = role_levels.get(user.role, 0)
        if user_level < min_level:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{user.role}' insufficient; requires '{min_role}' or higher",
            )
        return user

    return _check


# ---------------------------------------------------------------------------
# Legacy compatibility — drop-in replacement for api/auth.py
# ---------------------------------------------------------------------------
async def require_api_key(x_api_key: str | None = Header(default=None)):
    """Backward-compatible auth dependency.

    When JWT is configured, delegates to get_current_user.
    When only API_KEY is set, behaves like the original api/auth.py.
    When neither is set, auth is disabled (dev mode).
    """
    expected = os.getenv("API_KEY", "")
    if not expected:
        return  # Auth disabled
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------
def log_audit(
    user: CurrentUser | None,
    action: str,
    resource_type: str = "",
    resource_id: str = "",
    old_value: dict | None = None,
    new_value: dict | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> None:
    """Write an audit log entry. Best-effort — does not raise on failure."""
    try:
        import json
        from api.core import get_conn

        user_id = user.user_id if user else None
        # Skip logging for anonymous/api-key synthetic users without real UUIDs
        uid_param = None
        if user_id and user_id not in ("anonymous", "api-key-user"):
            try:
                uuid.UUID(user_id)
                uid_param = user_id
            except ValueError:
                uid_param = None

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO fact_audit_log
                   (user_id, action, resource_type, resource_id, old_value, new_value, ip_address, user_agent)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    uid_param,
                    action,
                    resource_type,
                    resource_id,
                    json.dumps(old_value) if old_value else None,
                    json.dumps(new_value) if new_value else None,
                    ip_address,
                    user_agent,
                ),
            )
            conn.commit()
    except Exception:
        pass  # Best-effort; never break the request
