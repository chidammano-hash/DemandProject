"""Auth-mode resolution for the SKU Chatbot Agent SDK runtime.

The Claude Agent SDK delegates model auth to its bundled Claude Code runtime.
This module turns the ``auth.mode`` config switch into the environment the SDK
should run with, and fails loud — at request time, not mid-stream — when a
non-``auto`` mode is selected without its credentials.

  auto    -> inherit the surrounding Claude Code session (no API key needed)
  api_key -> require ANTHROPIC_API_KEY
  bedrock -> CLAUDE_CODE_USE_BEDROCK=1 (AWS creds resolved by the SDK runtime)
  vertex  -> CLAUDE_CODE_USE_VERTEX=1 (GCP creds resolved by the SDK runtime)
"""
from __future__ import annotations

import os

_VALID_MODES = ("auto", "api_key", "bedrock", "vertex")


class SkuChatAuthError(RuntimeError):
    """Raised when the configured auth mode lacks its required credentials."""


def auth_mode(config: dict) -> str:
    """Return the configured auth mode (lowercased; defaults to ``auto``)."""
    return str((config.get("auth") or {}).get("mode", "auto")).lower()


def resolve_auth_env(config: dict) -> dict[str, str]:
    """Return env overrides to hand the Agent SDK for the configured auth mode.

    Raises
    ------
    SkuChatAuthError
        If the mode is unknown, or ``api_key`` is selected without
        ``ANTHROPIC_API_KEY`` present.
    """
    mode = auth_mode(config)
    if mode == "auto":
        return {}
    if mode == "api_key":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise SkuChatAuthError(
                "auth.mode='api_key' requires ANTHROPIC_API_KEY in the environment"
            )
        return {"ANTHROPIC_API_KEY": key}
    if mode == "bedrock":
        return {"CLAUDE_CODE_USE_BEDROCK": "1"}
    if mode == "vertex":
        return {"CLAUDE_CODE_USE_VERTEX": "1"}
    raise SkuChatAuthError(
        f"unknown auth.mode {mode!r}; expected one of {_VALID_MODES}"
    )
