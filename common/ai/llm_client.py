"""Provider-agnostic LLM client.

Shared by the AI Champion adjuster (common/ai/champion_adjuster.py) and the
agentic agents (ai_planner, tuning_advisor).

Supports four providers:
  - ollama         (default; local; $0; OpenAI-compatible API at :11434/v1)
  - openai_compat  (Together, Fireworks, DeepInfra, Groq — same API shape as Ollama)
  - openai         (GPT-4o etc.)
  - anthropic      (Claude family, e.g. claude-opus-4-7)

All providers expose the same interface: chat(messages, json_mode=True) -> ChatResponse.
The Ollama and openai_compat paths share the OpenAI Python SDK — they only differ
by base_url and api_key.

Strict JSON-mode is required for structured recommendation schemas. Each call
returns a ChatResponse including raw response text, parsed JSON dict, token
counts, and elapsed milliseconds.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChatResponse:
    """Uniform response shape across all providers."""
    text: str                          # raw text content
    parsed: dict[str, Any] | None      # parsed JSON (None if json_mode failed to parse)
    tokens_in: int
    tokens_out: int
    elapsed_ms: int
    provider: str
    model: str
    raw: dict[str, Any]                # full provider response for audit log


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LLMClientError(RuntimeError):
    """Raised when an LLM call fails after retries or hits a config error."""


class LLMJSONParseError(LLMClientError):
    """Raised when json_mode=True but the response is not valid JSON."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMClient:
    """Single-call, single-provider LLM wrapper.

    Construct one per run. Reuses the underlying provider SDK across calls so
    connection pooling and prompt caching (Anthropic) work.

    Parameters
    ----------
    provider : str  one of {"ollama","anthropic","openai","openai_compat"}
    model    : str  provider-specific model id (e.g. "qwen2.5:32b", "claude-opus-4-7")
    base_url : str | None  endpoint URL; only used for ollama/openai_compat
    timeout  : int  per-call timeout in seconds
    extra    : dict | None  provider-specific extras (e.g. ollama keep_alive, num_ctx)
    """

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        base_url: str | None = None,
        timeout: int = 60,
        extra: dict[str, Any] | None = None,
    ):
        self.provider = provider
        self.model = model
        self.timeout = timeout
        self.extra = extra or {}
        self._client = self._build_client(base_url)

    # ---- provider client construction --------------------------------------

    def _build_client(self, base_url: str | None):
        if self.provider == "ollama":
            from openai import OpenAI
            return OpenAI(
                base_url=base_url or "http://localhost:11434/v1",
                api_key="ollama",          # ignored by Ollama, required by SDK
                timeout=self.timeout,
            )
        if self.provider == "openai_compat":
            from openai import OpenAI
            url = base_url or os.environ.get("LLM_BASE_URL")
            api_key = os.environ.get("LLM_API_KEY")
            if not url:
                raise LLMClientError(
                    "openai_compat provider requires LLM_BASE_URL "
                    "(env var or base_url arg)"
                )
            if not api_key:
                raise LLMClientError("openai_compat provider requires LLM_API_KEY env var")
            return OpenAI(base_url=url, api_key=api_key, timeout=self.timeout)
        if self.provider == "openai":
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise LLMClientError("openai provider requires OPENAI_API_KEY env var")
            return OpenAI(api_key=api_key, timeout=self.timeout)
        if self.provider == "anthropic":
            from anthropic import Anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMClientError("anthropic provider requires ANTHROPIC_API_KEY env var")
            return Anthropic(api_key=api_key, timeout=self.timeout)
        raise LLMClientError(f"Unknown provider: {self.provider}")

    # ---- chat call ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """Run one chat completion. Returns ChatResponse uniform across providers."""
        t0 = time.perf_counter()

        if self.provider == "anthropic":
            response = self._chat_anthropic(messages, temperature, max_tokens, json_mode)
        else:
            response = self._chat_openai_like(messages, temperature, max_tokens, json_mode)

        response.elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if json_mode:
            try:
                response.parsed = json.loads(response.text)
            except json.JSONDecodeError as exc:
                log.warning("LLM returned non-JSON despite json_mode=True (provider=%s): %s",
                            self.provider, response.text[:200])
                raise LLMJSONParseError(f"Could not parse JSON from {self.provider}") from exc

        return response

    # ---- OpenAI-shaped path (covers ollama, openai_compat, openai) ---------

    def _chat_openai_like(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Ollama-specific extras pass through via extra_body (OpenAI SDK escape hatch)
        if self.provider == "ollama" and self.extra:
            extra_body = {}
            if "keep_alive" in self.extra:
                extra_body["keep_alive"] = self.extra["keep_alive"]
            if "num_ctx" in self.extra:
                extra_body.setdefault("options", {})["num_ctx"] = self.extra["num_ctx"]
            if extra_body:
                kwargs["extra_body"] = extra_body

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        tokens_in = getattr(usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0
        return ChatResponse(
            text=text,
            parsed=None,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            elapsed_ms=0,                  # filled by caller
            provider=self.provider,
            model=self.model,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {"text": text},
        )

    # ---- Anthropic path ----------------------------------------------------

    def _chat_anthropic(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> ChatResponse:
        # Anthropic separates system from messages; convert if present.
        system = ""
        msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                msgs.append(m)

        # Force JSON via prompt — Anthropic doesn't have a strict response_format,
        # but the model honors a clear "respond ONLY with JSON" instruction at temp=0.
        if json_mode and "JSON" not in system.upper():
            system = (system + "\n\nRespond ONLY with valid JSON, no prose.").strip()

        resp = self._client.messages.create(
            model=self.model,
            messages=msgs,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        return ChatResponse(
            text=text,
            parsed=None,
            tokens_in=resp.usage.input_tokens,
            tokens_out=resp.usage.output_tokens,
            elapsed_ms=0,
            provider=self.provider,
            model=self.model,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {"text": text},
        )


# ---------------------------------------------------------------------------
# Convenience: build from config dict (the ai_champion config shape)
# ---------------------------------------------------------------------------

def build_from_config(config: dict, *, override_provider: str | None = None) -> LLMClient:
    """Build an LLMClient from a provider config dict (e.g. ai_champion_config).

    Looks at config["provider"] (or override_provider), then config["models"][provider]
    and config["endpoints"][provider]. Wires in ollama-specific extras from
    config["ollama"] when applicable.
    """
    provider = override_provider or config.get("provider", "ollama")
    model = config["models"][provider]
    endpoints = config.get("endpoints", {})
    base_url = endpoints.get(provider)
    timeout = config.get("cost_controls", {}).get("per_call_timeout_seconds", 60)

    extra = {}
    if provider == "ollama":
        ollama_cfg = config.get("ollama", {})
        for key in ("keep_alive", "num_ctx", "temperature"):
            if key in ollama_cfg:
                extra[key] = ollama_cfg[key]
        if "request_timeout_seconds" in ollama_cfg:
            timeout = ollama_cfg["request_timeout_seconds"]

    return LLMClient(provider=provider, model=model, base_url=base_url,
                     timeout=timeout, extra=extra)


def tools_to_openai(tools: list[dict]) -> list[dict]:
    """Convert Anthropic-format tool definitions to OpenAI function-calling format.

    Shared by the agentic agents (ai_planner, tuning_advisor) which author tool
    schemas in Anthropic shape but also drive OpenAI-compatible providers.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]
