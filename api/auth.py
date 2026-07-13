"""Authentication dependency for mutation endpoints.

Interactive browser users authenticate with a JWT and service automation may
continue to use ``X-API-Key``.  Keeping both paths here lets every existing
write guard remain consistent without ever exposing the service key to the UI.
"""
from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException

from common.auth import decode_token
from common.core.utils import load_config


async def require_api_key(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    """Authorize a planner JWT or the service API key.

    Auth remains disabled for local development when neither ``API_KEY`` nor a
    non-default ``JWT_SECRET`` is configured.
    """
    expected = os.getenv("API_KEY", "")
    if expected and x_api_key and hmac.compare_digest(x_api_key, expected):
        return

    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            payload = decode_token(token)
            if payload.get("type") != "access":
                raise HTTPException(status_code=401, detail="Invalid token type")
            role_levels = load_config("auth_config.yaml").get("role_level", {})
            if role_levels.get(payload.get("role"), 0) < role_levels.get("planner", 2):
                raise HTTPException(status_code=403, detail="Planner role or higher required")
            return

    jwt_is_configured = bool(os.getenv("JWT_SECRET"))
    if not expected and not jwt_is_configured:
        return
    raise HTTPException(status_code=401, detail="Authentication required")
