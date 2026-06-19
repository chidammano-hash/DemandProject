"""Unit tests for common.ai.llm_client — provider routing and chat call paths.

Spec: docs/specs/02-forecasting/27-ai-champion-forecast.md
Covers provider construction (env-var validation), OpenAI-shaped chat path
(ollama / openai_compat / openai), Anthropic chat path, and build_from_config.

All tests mock the underlying SDK; no real API calls are made.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from common.ai.llm_client import (
    ChatResponse,
    LLMClient,
    LLMClientError,
    LLMJSONParseError,
    build_from_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Env vars the client inspects — clear all of them before each provider test to
# avoid pollution from the host shell.
LLM_ENV_VARS = (
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)


def _clear_llm_env(monkeypatch):
    for key in LLM_ENV_VARS:
        monkeypatch.delenv(key, raising=False)


def _mk_openai_resp(content: str, prompt_tokens: int = 12, completion_tokens: int = 7):
    """Build a MagicMock that mimics openai.types.chat.ChatCompletion."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    resp.model_dump.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def _mk_anthropic_resp(text: str, input_tokens: int = 22, output_tokens: int = 9):
    """Build a MagicMock that mimics anthropic.types.Message."""
    block = SimpleNamespace(text=text)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    resp = MagicMock()
    resp.content = [block]
    resp.usage = usage
    resp.model_dump.return_value = {"content": [{"text": text}]}
    return resp


# ---------------------------------------------------------------------------
# Provider routing — LLMClient.__init__
# ---------------------------------------------------------------------------

class TestProviderRouting:
    def test_ollama_builds_openai_with_default_base_url(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        with patch("openai.OpenAI") as mock_openai_cls:
            LLMClient(provider="ollama", model="qwen2.5:32b", timeout=30)
            mock_openai_cls.assert_called_once_with(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                timeout=30,
            )

    def test_ollama_uses_explicit_base_url_when_provided(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        with patch("openai.OpenAI") as mock_openai_cls:
            LLMClient(provider="ollama", model="qwen2.5:32b",
                      base_url="http://other-host:11434/v1")
            kwargs = mock_openai_cls.call_args.kwargs
            assert kwargs["base_url"] == "http://other-host:11434/v1"
            assert kwargs["api_key"] == "ollama"

    def test_openai_compat_missing_base_url_raises(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        with pytest.raises(LLMClientError, match="LLM_BASE_URL"):
            LLMClient(provider="openai_compat", model="meta-llama/Llama-3.3-70B-Instruct-Turbo")

    def test_openai_compat_missing_api_key_raises(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("LLM_BASE_URL", "https://api.together.xyz/v1")
        with pytest.raises(LLMClientError, match="LLM_API_KEY"):
            LLMClient(provider="openai_compat", model="meta-llama/Llama-3.3-70B-Instruct-Turbo")

    def test_openai_compat_with_env_vars_constructs_ok(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("LLM_BASE_URL", "https://api.together.xyz/v1")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        with patch("openai.OpenAI") as mock_openai_cls:
            LLMClient(provider="openai_compat",
                      model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
            mock_openai_cls.assert_called_once_with(
                base_url="https://api.together.xyz/v1",
                api_key="sk-test",
                timeout=60,
            )

    def test_openai_missing_api_key_raises(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        with pytest.raises(LLMClientError, match="OPENAI_API_KEY"):
            LLMClient(provider="openai", model="gpt-4o")

    def test_openai_with_api_key_constructs_ok(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        with patch("openai.OpenAI") as mock_openai_cls:
            LLMClient(provider="openai", model="gpt-4o", timeout=45)
            mock_openai_cls.assert_called_once_with(api_key="sk-openai-test", timeout=45)

    def test_anthropic_missing_api_key_raises(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        with pytest.raises(LLMClientError, match="ANTHROPIC_API_KEY"):
            LLMClient(provider="anthropic", model="claude-opus-4-7")

    def test_anthropic_with_api_key_constructs_ok(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        with patch("anthropic.Anthropic") as mock_anthropic_cls:
            LLMClient(provider="anthropic", model="claude-opus-4-7", timeout=90)
            mock_anthropic_cls.assert_called_once_with(api_key="sk-ant-test", timeout=90)

    def test_unknown_provider_raises(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        with pytest.raises(LLMClientError, match="Unknown provider: bogus"):
            LLMClient(provider="bogus", model="x")


# ---------------------------------------------------------------------------
# OpenAI-shaped chat() path — covers ollama, openai_compat, openai
# ---------------------------------------------------------------------------

class TestOpenAILikeChat:
    def _build_with_mock(self, monkeypatch, *, provider="ollama", extra=None):
        _clear_llm_env(monkeypatch)
        mock_client = MagicMock()
        with patch("openai.OpenAI", return_value=mock_client):
            client = LLMClient(provider=provider, model="qwen2.5:32b", extra=extra)
        return client, mock_client

    def test_chat_returns_chat_response_with_parsed_json(self, monkeypatch):
        client, mock_sdk = self._build_with_mock(monkeypatch)
        mock_sdk.chat.completions.create.return_value = _mk_openai_resp(
            '{"recommendation_code": "KEEP", "confidence": 0.9}',
            prompt_tokens=15,
            completion_tokens=8,
        )
        resp = client.chat(
            [{"role": "user", "content": "hi"}],
            json_mode=True,
            temperature=0.0,
            max_tokens=128,
        )
        assert isinstance(resp, ChatResponse)
        assert resp.parsed == {"recommendation_code": "KEEP", "confidence": 0.9}
        assert resp.tokens_in == 15
        assert resp.tokens_out == 8
        assert resp.provider == "ollama"
        assert resp.model == "qwen2.5:32b"
        assert resp.elapsed_ms >= 0

    def test_chat_forwards_temperature_and_max_tokens(self, monkeypatch):
        client, mock_sdk = self._build_with_mock(monkeypatch)
        mock_sdk.chat.completions.create.return_value = _mk_openai_resp('{"a": 1}')
        client.chat(
            [{"role": "user", "content": "x"}],
            json_mode=True,
            temperature=0.7,
            max_tokens=512,
        )
        kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 512
        assert kwargs["model"] == "qwen2.5:32b"

    def test_chat_sets_response_format_only_in_json_mode(self, monkeypatch):
        client, mock_sdk = self._build_with_mock(monkeypatch)
        mock_sdk.chat.completions.create.return_value = _mk_openai_resp("plain text response")
        client.chat([{"role": "user", "content": "x"}], json_mode=False)
        kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        assert "response_format" not in kwargs

    def test_chat_sets_json_object_response_format_in_json_mode(self, monkeypatch):
        client, mock_sdk = self._build_with_mock(monkeypatch)
        mock_sdk.chat.completions.create.return_value = _mk_openai_resp('{"x": 1}')
        client.chat([{"role": "user", "content": "x"}], json_mode=True)
        kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_chat_json_mode_non_json_raises_llm_json_parse_error(self, monkeypatch):
        client, mock_sdk = self._build_with_mock(monkeypatch)
        mock_sdk.chat.completions.create.return_value = _mk_openai_resp(
            "Sorry I can't respond with JSON today"
        )
        with pytest.raises(LLMJSONParseError):
            client.chat([{"role": "user", "content": "x"}], json_mode=True)

    def test_chat_non_json_mode_returns_raw_text(self, monkeypatch):
        client, mock_sdk = self._build_with_mock(monkeypatch)
        mock_sdk.chat.completions.create.return_value = _mk_openai_resp("hello world")
        resp = client.chat([{"role": "user", "content": "x"}], json_mode=False)
        assert resp.text == "hello world"
        assert resp.parsed is None

    def test_ollama_extras_passed_via_extra_body(self, monkeypatch):
        client, mock_sdk = self._build_with_mock(
            monkeypatch,
            provider="ollama",
            extra={"keep_alive": "24h", "num_ctx": 8192},
        )
        mock_sdk.chat.completions.create.return_value = _mk_openai_resp('{"x": 1}')
        client.chat([{"role": "user", "content": "x"}], json_mode=True)
        kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"] == {
            "keep_alive": "24h",
            "options": {"num_ctx": 8192},
        }

    def test_ollama_keep_alive_only(self, monkeypatch):
        client, mock_sdk = self._build_with_mock(
            monkeypatch, provider="ollama", extra={"keep_alive": "1h"},
        )
        mock_sdk.chat.completions.create.return_value = _mk_openai_resp('{"x": 1}')
        client.chat([{"role": "user", "content": "x"}], json_mode=True)
        kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"] == {"keep_alive": "1h"}

    def test_ollama_num_ctx_only(self, monkeypatch):
        client, mock_sdk = self._build_with_mock(
            monkeypatch, provider="ollama", extra={"num_ctx": 4096},
        )
        mock_sdk.chat.completions.create.return_value = _mk_openai_resp('{"x": 1}')
        client.chat([{"role": "user", "content": "x"}], json_mode=True)
        kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"] == {"options": {"num_ctx": 4096}}

    def test_ollama_no_extras_omits_extra_body(self, monkeypatch):
        client, mock_sdk = self._build_with_mock(monkeypatch, provider="ollama")
        mock_sdk.chat.completions.create.return_value = _mk_openai_resp('{"x": 1}')
        client.chat([{"role": "user", "content": "x"}], json_mode=True)
        kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        assert "extra_body" not in kwargs

    def test_openai_provider_does_not_send_extras(self, monkeypatch):
        # openai_compat / openai paths should NOT inject ollama extras even if
        # `extra` is populated (the construct branches on provider == "ollama").
        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_client = MagicMock()
        with patch("openai.OpenAI", return_value=mock_client):
            client = LLMClient(
                provider="openai",
                model="gpt-4o",
                extra={"keep_alive": "24h", "num_ctx": 8192},
            )
        mock_client.chat.completions.create.return_value = _mk_openai_resp('{"x": 1}')
        client.chat([{"role": "user", "content": "x"}], json_mode=True)
        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "extra_body" not in kwargs

    def test_chat_handles_null_content_as_empty_string(self, monkeypatch):
        client, mock_sdk = self._build_with_mock(monkeypatch)
        # Some providers return content=None when they error mid-stream;
        # the client must coerce to "" rather than crash on `len()` etc.
        mock_sdk.chat.completions.create.return_value = _mk_openai_resp(None)
        with pytest.raises(LLMJSONParseError):
            client.chat([{"role": "user", "content": "x"}], json_mode=True)


# ---------------------------------------------------------------------------
# Anthropic chat() path
# ---------------------------------------------------------------------------

class TestAnthropicChat:
    def _build_with_mock(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        mock_client = MagicMock()
        with patch("anthropic.Anthropic", return_value=mock_client):
            client = LLMClient(provider="anthropic", model="claude-opus-4-7")
        return client, mock_client

    def test_system_message_split_out(self, monkeypatch):
        client, mock_sdk = self._build_with_mock(monkeypatch)
        mock_sdk.messages.create.return_value = _mk_anthropic_resp('{"x": 1}')
        client.chat(
            [
                {"role": "system", "content": "You are a planner. JSON only."},
                {"role": "user", "content": "decide"},
            ],
            json_mode=True,
        )
        kwargs = mock_sdk.messages.create.call_args.kwargs
        assert kwargs["system"] == "You are a planner. JSON only."
        # Only the user message is forwarded; system is hoisted.
        assert kwargs["messages"] == [{"role": "user", "content": "decide"}]

    def test_json_instruction_appended_when_system_lacks_json(self, monkeypatch):
        client, mock_sdk = self._build_with_mock(monkeypatch)
        mock_sdk.messages.create.return_value = _mk_anthropic_resp('{"x": 1}')
        client.chat(
            [
                {"role": "system", "content": "You are a planner."},
                {"role": "user", "content": "decide"},
            ],
            json_mode=True,
        )
        system_arg = mock_sdk.messages.create.call_args.kwargs["system"]
        assert "Respond ONLY with valid JSON" in system_arg
        assert system_arg.startswith("You are a planner.")

    def test_temperature_deprecated_retries_without_temperature(self, monkeypatch):
        """Newer Claude models reject `temperature`; the client retries without it."""
        import httpx
        import anthropic

        client, mock_sdk = self._build_with_mock(monkeypatch)
        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        err = anthropic.BadRequestError(
            message="`temperature` is deprecated for this model.",
            response=httpx.Response(400, request=req),
            body=None,
        )
        mock_sdk.messages.create.side_effect = [err, _mk_anthropic_resp('{"x": 1}')]

        resp = client.chat([{"role": "user", "content": "decide"}], json_mode=True)

        assert resp.parsed == {"x": 1}
        assert mock_sdk.messages.create.call_count == 2
        # First attempt included temperature; the retry dropped it.
        assert "temperature" in mock_sdk.messages.create.call_args_list[0].kwargs
        assert "temperature" not in mock_sdk.messages.create.call_args_list[1].kwargs

    def test_other_bad_request_becomes_llm_client_error(self, monkeypatch):
        """A non-temperature 400 surfaces as LLMClientError (degrades gracefully upstream)."""
        import httpx
        import anthropic

        client, mock_sdk = self._build_with_mock(monkeypatch)
        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        mock_sdk.messages.create.side_effect = anthropic.BadRequestError(
            message="max_tokens too large", response=httpx.Response(400, request=req), body=None,
        )
        with pytest.raises(LLMClientError):
            client.chat([{"role": "user", "content": "decide"}], json_mode=True)

    def test_json_instruction_not_duplicated_when_system_mentions_json(self, monkeypatch):
        client, mock_sdk = self._build_with_mock(monkeypatch)
        mock_sdk.messages.create.return_value = _mk_anthropic_resp('{"x": 1}')
        client.chat(
            [
                {"role": "system", "content": "You are a planner. Output JSON only."},
                {"role": "user", "content": "decide"},
            ],
            json_mode=True,
        )
        system_arg = mock_sdk.messages.create.call_args.kwargs["system"]
        assert system_arg == "You are a planner. Output JSON only."
        # No duplication of the fallback instruction
        assert system_arg.count("Respond ONLY") == 0

    def test_usage_tokens_mapped_to_chat_response(self, monkeypatch):
        client, mock_sdk = self._build_with_mock(monkeypatch)
        mock_sdk.messages.create.return_value = _mk_anthropic_resp(
            '{"x": 1}', input_tokens=123, output_tokens=45,
        )
        resp = client.chat(
            [{"role": "user", "content": "x"}],
            json_mode=True,
        )
        assert resp.tokens_in == 123
        assert resp.tokens_out == 45
        assert resp.provider == "anthropic"
        assert resp.model == "claude-opus-4-7"
        assert resp.parsed == {"x": 1}

    def test_temperature_and_max_tokens_forwarded(self, monkeypatch):
        client, mock_sdk = self._build_with_mock(monkeypatch)
        mock_sdk.messages.create.return_value = _mk_anthropic_resp('{"x": 1}')
        client.chat(
            [{"role": "user", "content": "x"}],
            json_mode=True,
            temperature=0.4,
            max_tokens=2048,
        )
        kwargs = mock_sdk.messages.create.call_args.kwargs
        assert kwargs["temperature"] == 0.4
        assert kwargs["max_tokens"] == 2048

    def test_non_json_mode_does_not_modify_system(self, monkeypatch):
        client, mock_sdk = self._build_with_mock(monkeypatch)
        mock_sdk.messages.create.return_value = _mk_anthropic_resp("free-form prose")
        client.chat(
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "go"},
            ],
            json_mode=False,
        )
        system_arg = mock_sdk.messages.create.call_args.kwargs["system"]
        assert system_arg == "be brief"


# ---------------------------------------------------------------------------
# build_from_config
# ---------------------------------------------------------------------------

class TestBuildFromConfig:
    def _ollama_cfg(self):
        return {
            "provider": "ollama",
            "models": {"ollama": "qwen2.5:32b", "anthropic": "claude-opus-4-7"},
            "endpoints": {"ollama": "http://localhost:11434/v1"},
            "cost_controls": {"per_call_timeout_seconds": 60},
            "ollama": {
                "keep_alive": "24h",
                "num_ctx": 8192,
                "temperature": 0.0,
                "request_timeout_seconds": 120,
            },
        }

    def test_ollama_wires_extras_and_base_url(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        with patch("openai.OpenAI") as mock_openai_cls:
            client = build_from_config(self._ollama_cfg())
            mock_openai_cls.assert_called_once()
            kwargs = mock_openai_cls.call_args.kwargs
            assert kwargs["base_url"] == "http://localhost:11434/v1"
            assert kwargs["api_key"] == "ollama"
        assert client.provider == "ollama"
        assert client.model == "qwen2.5:32b"
        assert client.extra == {"keep_alive": "24h", "num_ctx": 8192, "temperature": 0.0}

    def test_ollama_request_timeout_overrides_cost_control_timeout(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        cfg = self._ollama_cfg()
        cfg["cost_controls"]["per_call_timeout_seconds"] = 30  # would lose to 120
        with patch("openai.OpenAI") as mock_openai_cls:
            client = build_from_config(cfg)
            assert mock_openai_cls.call_args.kwargs["timeout"] == 120
        assert client.timeout == 120

    def test_ollama_without_request_timeout_uses_cost_control_timeout(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        cfg = self._ollama_cfg()
        cfg["cost_controls"]["per_call_timeout_seconds"] = 45
        del cfg["ollama"]["request_timeout_seconds"]
        with patch("openai.OpenAI") as mock_openai_cls:
            client = build_from_config(cfg)
            assert mock_openai_cls.call_args.kwargs["timeout"] == 45
        assert client.timeout == 45

    def test_override_provider_anthropic_ignores_provider_field(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        with patch("anthropic.Anthropic") as mock_anthropic_cls:
            client = build_from_config(self._ollama_cfg(), override_provider="anthropic")
            mock_anthropic_cls.assert_called_once()
        assert client.provider == "anthropic"
        assert client.model == "claude-opus-4-7"
        # Ollama extras must not bleed into non-ollama providers.
        assert client.extra == {}

    def test_anthropic_uses_per_call_timeout(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        cfg = self._ollama_cfg()
        cfg["provider"] = "anthropic"
        cfg["cost_controls"]["per_call_timeout_seconds"] = 75
        with patch("anthropic.Anthropic") as mock_anthropic_cls:
            build_from_config(cfg)
            assert mock_anthropic_cls.call_args.kwargs["timeout"] == 75

    def test_missing_cost_controls_falls_back_to_default_timeout(self, monkeypatch):
        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        cfg = {
            "provider": "openai",
            "models": {"openai": "gpt-4o"},
        }
        with patch("openai.OpenAI") as mock_openai_cls:
            build_from_config(cfg)
            assert mock_openai_cls.call_args.kwargs["timeout"] == 60
