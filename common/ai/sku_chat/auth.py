"""Auth-mode resolution for the SKU Chatbot agent runtimes.

Claude and Codex both have a local-subscription path for development plus an
API-key path for standalone automation. This module turns the SKU-chat config
switches into the environment each runtime should receive, and fails loud at
request time when a selected mode lacks credentials.

  runtime.provider:
  claude  -> use claude-agent-sdk / Claude Code auth
  codex   -> use Codex CLI auth through ``codex exec``

  auth.mode for claude:
  auto    -> inherit the surrounding Claude Code session (no API key needed)
  api_key -> require ANTHROPIC_API_KEY
  bedrock -> CLAUDE_CODE_USE_BEDROCK=1 (AWS creds resolved by the SDK runtime)
  vertex  -> CLAUDE_CODE_USE_VERTEX=1 (GCP creds resolved by the SDK runtime)

  auth.mode for codex:
  auto    -> inherit saved Codex CLI auth (ChatGPT sign-in or configured auth)
  api_key -> require CODEX_API_KEY and inject it for the ``codex exec`` call
"""
from __future__ import annotations

import os

_VALID_PROVIDERS = ("claude", "codex")
_VALID_MODES = ("auto", "api_key", "bedrock", "vertex")
_VALID_CODEX_MODES = ("auto", "api_key")


class SkuChatAuthError(RuntimeError):
    """Raised when the configured auth mode lacks its required credentials."""


def runtime_provider(config: dict) -> str:
    """Return the selected agent runtime provider (defaults to ``claude``)."""
    provider = str((config.get("runtime") or {}).get("provider", "claude")).lower()
    if provider not in _VALID_PROVIDERS:
        raise SkuChatAuthError(
            f"unknown runtime.provider {provider!r}; expected one of {_VALID_PROVIDERS}"
        )
    return provider


def auth_mode(config: dict) -> str:
    """Return the configured auth mode (lowercased; defaults to ``auto``)."""
    return str((config.get("auth") or {}).get("mode", "auto")).lower()


def resolve_auth_env(config: dict, *, provider: str | None = None) -> dict[str, str]:
    """Return env overrides for the configured runtime and auth mode.

    Raises
    ------
    SkuChatAuthError
        If the runtime/mode is unknown, or ``api_key`` is selected without the
        corresponding provider key present.
    """
    provider = provider or runtime_provider(config)
    if provider == "codex":
        return _resolve_codex_auth_env(config)
    if provider != "claude":
        raise SkuChatAuthError(
            f"unknown runtime.provider {provider!r}; expected one of {_VALID_PROVIDERS}"
        )
    return _resolve_claude_auth_env(config)


def _resolve_claude_auth_env(config: dict) -> dict[str, str]:
    """Return env overrides for the Claude Agent SDK runtime."""
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


def _resolve_codex_auth_env(config: dict) -> dict[str, str]:
    """Return env overrides for the Codex CLI runtime."""
    mode = auth_mode(config)
    if mode == "auto":
        return {}
    if mode == "api_key":
        key = os.getenv("CODEX_API_KEY")
        if not key:
            raise SkuChatAuthError(
                "runtime.provider='codex' with auth.mode='api_key' requires "
                "CODEX_API_KEY in the environment"
            )
        return {"CODEX_API_KEY": key}
    raise SkuChatAuthError(
        f"runtime.provider='codex' supports auth.mode values {_VALID_CODEX_MODES}; "
        f"got {mode!r}"
    )
