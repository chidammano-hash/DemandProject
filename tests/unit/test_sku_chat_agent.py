"""Unit tests for the SKU chat agent's SDK-message mapping and degradation path."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from common.ai.sku_chat.agent import (
    SkuChatAgent,
    SkuChatContext,
    sdk_message_to_events,
)
from common.ai.sku_chat.prompts import build_user_prompt
from common.ai.sku_chat.tools import AgentSdkUnavailableError


# Fakes named exactly as the Agent SDK classes — sdk_message_to_events dispatches
# on type(msg).__name__ (StreamEvent / AssistantMessage / ResultMessage).
class StreamEvent:
    def __init__(self, event):
        self.event = event


class AssistantMessage:
    def __init__(self, content):
        self.content = content


class ResultMessage:
    def __init__(self, result, cost, usage):
        self.result = result
        self.total_cost_usd = cost
        self.usage = usage


class ToolUseBlock:
    type = "tool_use"

    def __init__(self, name, tool_input):
        self.name = name
        self.input = tool_input


def test_stream_event_text_delta_maps_to_text_chunk():
    msg = StreamEvent(
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}
    )
    assert sdk_message_to_events(msg) == [{"type": "text", "chunk": "Hi"}]


def test_stream_event_non_text_delta_is_ignored():
    msg = StreamEvent({"type": "message_start"})
    assert sdk_message_to_events(msg) == []


def test_result_message_maps_to_result_event():
    msg = ResultMessage("Final answer", 0.012, {"output_tokens": 7})
    out = sdk_message_to_events(msg)
    assert out == [
        {
            "type": "result",
            "text": "Final answer",
            "cost_usd": 0.012,
            "usage": {"output_tokens": 7},
        }
    ]


def test_assistant_message_emits_tool_events_only():
    msg = AssistantMessage([ToolUseBlock("mcp__sku__get_sku_profile", {"item_id": "1"})])
    out = sdk_message_to_events(msg)
    assert out == [
        {
            "type": "tool",
            "name": "mcp__sku__get_sku_profile",
            "input": {"item_id": "1"},
        }
    ]


def test_unknown_message_maps_to_nothing():
    assert sdk_message_to_events(object()) == []


def test_build_user_prompt_includes_page_focus_and_sku():
    ctx = SkuChatContext("100320", "RETAIL", "DC1")
    prompt = build_user_prompt("Why did it miss?", ctx, page_focus="the Inventory Planning page.")
    assert "Page context: the Inventory Planning page." in prompt
    assert "item_id=100320" in prompt
    assert "User question: Why did it miss?" in prompt


def test_build_user_prompt_without_sku_notes_no_selection():
    ctx = SkuChatContext("", "", "")
    prompt = build_user_prompt("Hello", ctx)
    assert "No specific SKU" in prompt


@pytest.mark.asyncio
async def test_stream_turn_degrades_to_meta_then_error_when_sdk_unavailable():
    """When the Agent SDK can't load, a turn degrades to meta + error (no live call).

    Forced via patch so the test is hermetic whether or not the `agent` extra is
    installed — never makes a real model call.
    """
    agent = SkuChatAgent(pool=None, config={})
    ctx = SkuChatContext("100320", "RETAIL", "DC1")
    with patch(
        "common.ai.sku_chat.agent._import_query",
        side_effect=AgentSdkUnavailableError("claude-agent-sdk is not installed"),
    ):
        events = [ev async for ev in agent.stream_turn("Why did Q3 miss?", ctx)]

    assert events[0]["type"] == "meta"
    assert events[0]["tier"] == "deep"
    assert events[-1]["type"] == "error"
    assert "claude-agent-sdk" in events[-1]["message"]


@pytest.mark.asyncio
async def test_stream_turn_uses_codex_runtime_when_configured():
    """Codex mode emits meta/text/result without importing the Claude SDK."""

    def fake_context(pool, ctx, *, history_months, peer_limit):
        assert pool == "pool"
        assert ctx.item_id == "100320"
        assert history_months == 12
        assert peer_limit == 3
        return {"profile": {"found": True, "item_id": ctx.item_id}}

    async def fake_codex(prompt, *, model_id, cfg, env, timeout_s):
        assert model_id == "gpt-5.5"
        assert cfg["runtime"]["provider"] == "codex"
        assert env == {}
        assert "SKU context JSON" in prompt
        assert timeout_s == 15
        return "Codex answer from SKU context."

    agent = SkuChatAgent(
        pool="pool",
        config={
            "runtime": {"provider": "codex"},
            "auth": {"mode": "auto"},
            "codex_models": {
                "fast": "gpt-5.4-mini",
                "standard": "gpt-5.5",
                "deep": "gpt-5.5",
            },
            "context": {"history_lookback_months": 12, "cluster_peer_limit": 3},
            "guardrails": {"timeout_seconds": 15},
        },
    )
    ctx = SkuChatContext("100320", "RETAIL", "DC1")
    with patch("common.ai.sku_chat.agent._build_codex_context", fake_context), patch(
        "common.ai.sku_chat.agent._run_codex_exec", fake_codex
    ):
        events = [
            ev async for ev in agent.stream_turn(
                "Provide a detailed summary of demand and forecast trends", ctx
            )
        ]

    assert events == [
        {"type": "meta", "tier": "standard", "model": "gpt-5.5", "runtime": "codex"},
        {"type": "text", "chunk": "Codex answer from SKU context."},
        {
            "type": "result",
            "text": "Codex answer from SKU context.",
            "cost_usd": None,
            "usage": None,
        },
    ]
