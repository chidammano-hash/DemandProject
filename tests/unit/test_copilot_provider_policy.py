"""Provider-boundary tests for the grounded planning Copilot."""

from __future__ import annotations

import pytest

from common.ai.copilot.config import CopilotSettings, build_provider


class _Client:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def _settings(**provider_overrides: object) -> CopilotSettings:
    provider = {
        "mode": "local",
        "provider": "ollama",
        "model": "demand-copilot:latest",
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key_env": "OPENAI_API_KEY",
        "cloud_consent_env": "DEMAND_AI_ALLOW_CLOUD",
        "max_output_tokens": 1200,
        "max_input_chars": 16000,
    }
    provider.update(provider_overrides)
    return CopilotSettings.model_validate(
        {
            "provider": provider,
            "runtime": {
                "max_turns": 6,
                "max_prompt_chars": 4000,
                "max_tool_calls": 8,
                "session_ttl_minutes": 60,
                "content_retention_days": 30,
                "turn_timeout_seconds": 60,
                "max_session_turns": 30,
                "max_conversation_chars": 4000,
                "max_concurrent_turns": 2,
            },
        }
    )


def test_local_provider_rejects_non_loopback_endpoint() -> None:
    settings = _settings(base_url="https://example.com/v1")

    with pytest.raises(RuntimeError, match="loopback"):
        build_provider(settings, environment={}, client_factory=_Client)


def test_cloud_provider_requires_explicit_consent_and_key() -> None:
    settings = _settings(
        mode="cloud_explicit",
        provider="openai",
        base_url=None,
    )

    with pytest.raises(RuntimeError, match="explicitly enabled"):
        build_provider(settings, environment={}, client_factory=_Client)
    with pytest.raises(RuntimeError, match="API key"):
        build_provider(
            settings,
            environment={"DEMAND_AI_ALLOW_CLOUD": "true"},
            client_factory=_Client,
        )


def test_local_provider_never_uses_cloud_credentials() -> None:
    settings = _settings()

    provider = build_provider(
        settings,
        environment={"OPENAI_API_KEY": "must-not-be-used"},
        client_factory=_Client,
    )

    assert provider is not None
    assert provider.client.kwargs == {
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama-local",
    }
    assert provider.mode == "local"


def test_disabled_provider_constructs_no_client() -> None:
    settings = _settings(mode="disabled")

    assert build_provider(settings, environment={}, client_factory=_Client) is None
