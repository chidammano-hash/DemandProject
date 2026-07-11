"""Fail-closed provider and runtime configuration for the grounded Copilot."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderSettings(_StrictSettings):
    mode: Literal["disabled", "local", "cloud_explicit"]
    provider: Literal["ollama", "openai"]
    model: str = Field(min_length=1, max_length=200)
    base_url: str | None
    api_key_env: str = Field(min_length=1, max_length=100)
    cloud_consent_env: str = Field(min_length=1, max_length=100)
    max_output_tokens: int = Field(ge=1, le=8_000)
    max_input_chars: int = Field(ge=2_000, le=200_000)

    @model_validator(mode="after")
    def validate_mode(self) -> ProviderSettings:
        if self.mode == "local" and (self.provider != "ollama" or self.base_url is None):
            raise ValueError("local mode requires Ollama and a base URL")
        if self.mode == "cloud_explicit" and (
            self.provider != "openai" or self.base_url is not None
        ):
            raise ValueError("cloud mode requires OpenAI without a custom base URL")
        return self


class RuntimeSettings(_StrictSettings):
    max_turns: int = Field(ge=1, le=12)
    max_prompt_chars: int = Field(ge=1, le=20_000)
    max_tool_calls: int = Field(ge=1, le=20)
    session_ttl_minutes: int = Field(ge=15, le=240)
    content_retention_days: int = Field(ge=1, le=365)
    turn_timeout_seconds: int = Field(ge=5, le=300)
    max_session_turns: int = Field(ge=1, le=100)
    max_conversation_chars: int = Field(ge=0, le=50_000)
    max_concurrent_turns: int = Field(ge=1, le=32)


class CopilotSettings(_StrictSettings):
    provider: ProviderSettings
    runtime: RuntimeSettings


@dataclass(frozen=True, slots=True)
class ConfiguredProvider:
    """Constructed client plus explicit data-boundary metadata."""

    client: Any
    mode: Literal["local", "cloud_explicit"]
    provider: str
    model: str
    store: bool
    max_output_tokens: int
    max_input_chars: int


ClientFactory = Callable[..., Any]


def validate_local_base_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise RuntimeError("local Copilot base URL must use a loopback host")
    if parsed.username is not None or parsed.password is not None:
        raise RuntimeError("local Copilot base URL cannot contain credentials")
    return raw_url


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes"}


def build_provider(
    settings: CopilotSettings,
    *,
    environment: Mapping[str, str] | None = None,
    client_factory: ClientFactory,
) -> ConfiguredProvider | None:
    """Build one explicitly selected provider; never perform provider fallback."""
    provider = settings.provider
    env = os.environ if environment is None else environment
    if provider.mode == "disabled":
        return None
    if provider.mode == "local":
        if provider.base_url is None:
            raise RuntimeError("local Copilot base URL is required")
        client = client_factory(
            base_url=validate_local_base_url(provider.base_url),
            api_key="ollama-local",
        )
        return ConfiguredProvider(
            client=client,
            mode="local",
            provider=provider.provider,
            model=provider.model,
            store=False,
            max_output_tokens=provider.max_output_tokens,
            max_input_chars=provider.max_input_chars,
        )
    if not _truthy(env.get(provider.cloud_consent_env)):
        raise RuntimeError("cloud Copilot must be explicitly enabled")
    api_key = env.get(provider.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError("cloud Copilot API key is required")
    return ConfiguredProvider(
        client=client_factory(api_key=api_key),
        mode="cloud_explicit",
        provider=provider.provider,
        model=provider.model,
        store=False,
        max_output_tokens=provider.max_output_tokens,
        max_input_chars=provider.max_input_chars,
    )
