"""Unit tests for auth-mode resolution (common/ai/sku_chat/auth.py)."""
from __future__ import annotations

import pytest

from common.ai.sku_chat import auth


def test_auto_mode_returns_empty_env():
    assert auth.resolve_auth_env({"auth": {"mode": "auto"}}) == {}


def test_default_mode_is_auto():
    assert auth.auth_mode({}) == "auto"
    assert auth.resolve_auth_env({}) == {}


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
