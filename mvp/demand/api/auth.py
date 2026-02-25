"""API key authentication dependency for mutation endpoints."""
from __future__ import annotations

import os

from fastapi import Header, HTTPException


async def require_api_key(x_api_key: str | None = Header(default=None)):
    """Validate X-API-Key header. Auth is disabled when API_KEY env var is unset."""
    expected = os.getenv("API_KEY", "")
    if not expected:
        return  # Auth disabled when no key configured
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
