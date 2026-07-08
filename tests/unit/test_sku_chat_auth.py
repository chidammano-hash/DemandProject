"""Unit tests for auth-mode resolution (common/ai/sku_chat/auth.py)."""
from __future__ import annotations

import pytest

from common.ai.sku_chat import auth


def test_auto_mode_returns_empty_env():
    assert auth.resolve_auth_env({"auth": {"mode": "auto"}}) == {}


def test_default_mode_is_auto():
    assert auth.auth_mode({}) == "auto"
    assert auth.resolve_auth_env({}) == {}


def test_default_runtime_provider_is_claude():
    assert auth.runtime_provider({}) == "claude"


def test_api_key_mode_requires_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(auth.SkuChatAuthError):
        auth.resolve_auth_env({"auth": {"mode": "api_key"}})


def test_api_key_mode_passes_key_through(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    env = auth.resolve_auth_env({"auth": {"mode": "api_key"}})
    assert env == {"ANTHROPIC_API_KEY": "sk-ant-test"}


def test_bedrock_mode():
    assert auth.resolve_auth_env({"auth": {"mode": "bedrock"}}) == {
        "CLAUDE_CODE_USE_BEDROCK": "1"
    }


def test_vertex_mode():
    assert auth.resolve_auth_env({"auth": {"mode": "vertex"}}) == {
        "CLAUDE_CODE_USE_VERTEX": "1"
    }


def test_unknown_mode_raises():
    with pytest.raises(auth.SkuChatAuthError):
        auth.resolve_auth_env({"auth": {"mode": "nonsense"}})


def test_codex_auto_mode_returns_empty_env():
    cfg = {"runtime": {"provider": "codex"}, "auth": {"mode": "auto"}}
    assert auth.runtime_provider(cfg) == "codex"
    assert auth.resolve_auth_env(cfg) == {}


def test_codex_api_key_mode_requires_codex_key(monkeypatch):
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    cfg = {"runtime": {"provider": "codex"}, "auth": {"mode": "api_key"}}
    with pytest.raises(auth.SkuChatAuthError, match="CODEX_API_KEY"):
        auth.resolve_auth_env(cfg)


def test_codex_api_key_mode_passes_key_through(monkeypatch):
    monkeypatch.setenv("CODEX_API_KEY", "sk-codex-test")
    cfg = {"runtime": {"provider": "codex"}, "auth": {"mode": "api_key"}}
    assert auth.resolve_auth_env(cfg) == {"CODEX_API_KEY": "sk-codex-test"}


def test_codex_rejects_claude_only_modes():
    cfg = {"runtime": {"provider": "codex"}, "auth": {"mode": "bedrock"}}
    with pytest.raises(auth.SkuChatAuthError, match=r"supports auth\.mode"):
        auth.resolve_auth_env(cfg)


def test_unknown_runtime_provider_raises():
    with pytest.raises(auth.SkuChatAuthError, match=r"runtime\.provider"):
        auth.runtime_provider({"runtime": {"provider": "nonsense"}})
