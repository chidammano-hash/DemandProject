"""Tests for the customer-analytics AI assistant runtime."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from common.ai.customer_analytics_assistant import answer_customer_question


@pytest.mark.asyncio
async def test_local_runtime_reuses_codex_login_and_grounded_context(monkeypatch):
    monkeypatch.delenv("CUSTOMER_ANALYTICS_AI_RUNTIME", raising=False)
    context = {
        "filters": {"state": "TX"},
        "kpis": [{"key": "fill_rate", "value": 91.2, "delta": -1.4}],
        "top_customers": [{"customer_name": "North Market", "demand_qty": 1200}],
        "service_risks": [{"customer_name": "South Market", "fill_rate": 72.0}],
    }
    captured: dict[str, str] = {}

    async def fake_codex(prompt: str, **kwargs: object) -> str:
        captured["prompt"] = prompt
        captured["model"] = str(kwargs["model_id"])
        return "Texas fill rate is 91.2%, down 1.4 points."

    with (
        patch(
            "common.ai.customer_analytics_assistant.get_sku_chat_config",
            return_value={
                "runtime": {"provider": "codex"},
                "auth": {"mode": "auto"},
                "codex_models": {
                    "fast": "gpt-fast",
                    "standard": "gpt-standard",
                    "deep": "gpt-deep",
                },
                "routing": {"default_tier": "standard"},
                "codex": {"binary": "codex", "sandbox": "read-only"},
                "guardrails": {"timeout_seconds": 30},
            },
        ),
        patch(
            "common.ai.customer_analytics_assistant._load_assistant_config",
            return_value={"runtime": {"provider": "codex"}, "max_history_messages": 6},
        ),
        patch(
            "common.ai.customer_analytics_assistant.resolve_auth_env",
            return_value={},
        ),
        patch(
            "common.ai.customer_analytics_assistant._run_codex_exec",
            side_effect=fake_codex,
        ),
    ):
        result = await answer_customer_question(
            "Why is service slipping in Texas?",
            context,
            history=[],
        )

    assert result.provider == "codex"
    assert result.model == "gpt-deep"
    assert result.answer.startswith("Texas fill rate")
    assert '"state": "TX"' in captured["prompt"]
    assert "91.2" in captured["prompt"]
    assert captured["model"] == "gpt-deep"


@pytest.mark.asyncio
async def test_production_runtime_uses_openai_api(monkeypatch):
    monkeypatch.setenv("CUSTOMER_ANALYTICS_AI_RUNTIME", "openai")

    class FakeResponse:
        text = "Concentration is stable."

    class FakeClient:
        provider = "openai"
        model = "gpt-5-mini"

        def chat(self, messages, **kwargs):
            assert "customer analytics" in messages[0]["content"].lower()
            assert kwargs["json_mode"] is False
            return FakeResponse()

    with (
        patch(
            "common.ai.customer_analytics_assistant._load_assistant_config",
            return_value={
                "runtime": {"provider": "codex"},
                "models": {"openai": "gpt-5-mini"},
                "cost_controls": {"per_call_timeout_seconds": 30},
            },
        ),
        patch(
            "common.ai.customer_analytics_assistant.build_from_config",
            return_value=FakeClient(),
        ),
    ):
        result = await answer_customer_question(
            "Summarize concentration risk",
            {"kpis": []},
            history=[],
        )

    assert result.provider == "openai"
    assert result.model == "gpt-5-mini"
    assert result.answer == "Concentration is stable."
