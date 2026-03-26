"""LLM client management for the Supply Chain Command Center API.

Provides lazy-initialized LLM clients with failover support:
- Primary: OpenAI (gpt-4o / gpt-4o-mini)
- Fallback: Anthropic (claude-sonnet-4-20250514)
"""
from __future__ import annotations

import logging
import os

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI client (primary — used by chat, intel, ai_planner, tuning_advisor)
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


# ---------------------------------------------------------------------------
# Anthropic client (fallback)
# ---------------------------------------------------------------------------
_anthropic_client = None


def get_anthropic():
    """Get Anthropic client for failover. Returns None if not configured."""
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key or len(api_key) < 20:
            return None
        try:
            from anthropic import Anthropic
            _anthropic_client = Anthropic(api_key=api_key)
        except ImportError:
            logger.debug("anthropic package not installed — failover unavailable")
            return None
    return _anthropic_client


# ---------------------------------------------------------------------------
# Unified completion with failover
# ---------------------------------------------------------------------------
def chat_completion(
    messages: list[dict],
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> str:
    """Send a chat completion request with automatic provider failover.

    Tries OpenAI first; on failure, falls back to Anthropic if configured.
    Returns the assistant's response text.
    """
    # Try OpenAI first
    try:
        client = get_openai()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except HTTPException:
        raise  # Don't catch our own 503
    except Exception as openai_err:
        logger.warning("OpenAI request failed: %s — attempting Anthropic failover", openai_err)

    # Fallback to Anthropic
    anthropic_client = get_anthropic()
    if anthropic_client is None:
        raise HTTPException(
            status_code=503,
            detail="OpenAI request failed and Anthropic failover not configured",
        )

    try:
        # Convert OpenAI messages to Anthropic format
        system_msg = ""
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            system=system_msg,
            messages=anthropic_messages,
        )
        return response.content[0].text
    except Exception as anthropic_err:
        logger.exception("Both OpenAI and Anthropic failed")
        raise HTTPException(
            status_code=503,
            detail=f"All LLM providers failed: OpenAI={openai_err}, Anthropic={anthropic_err}",
        ) from anthropic_err
