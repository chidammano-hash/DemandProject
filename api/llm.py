"""OpenAI client management for the Supply Chain Command Center API.

Provides lazy-initialized OpenAI client shared by chat + intel routers.
"""
from __future__ import annotations

import os

from fastapi import HTTPException


# ---------------------------------------------------------------------------
# OpenAI client (shared by chat + intel routers)
# ---------------------------------------------------------------------------
_openai_client = None


def get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or len(api_key) < 20:
            raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client
