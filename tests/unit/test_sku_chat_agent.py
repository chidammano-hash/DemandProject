"""Unit tests for the SKU chat agent's SDK-message mapping and degradation path."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from common.ai.sku_chat import agent as agent_module
from common.ai.sku_chat.agent import (
    CodexRuntimeError,
    SkuChatAgent,
    SkuChatContext,
    _resolve_codex_binary,
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


def test_effective_sku_context_infers_missing_fields_from_text():
    ctx = agent_module._effective_sku_context(
        SkuChatContext("", "", ""),
        "all",
        [
            {
                "role": "assistant",
                "content": "SKU selected: `item_id=11958`, `customer_group=ALL`, `loc=1401-BULK`.",
            }
        ],
        max_history=20,
    )

    assert ctx == SkuChatContext("11958", "ALL", "1401-BULK")


def test_effective_sku_context_keeps_explicit_fields_over_history():
    ctx = agent_module._effective_sku_context(
        SkuChatContext("100320", "", ""),
        "What about SKU 11958 @ 1401-BULK?",
        [],
        max_history=20,
    )

    assert ctx == SkuChatContext("100320", "", "1401-BULK")


def test_resolve_codex_binary_uses_path(monkeypatch):
    monkeypatch.setattr(agent_module.shutil, "which", lambda name: f"/bin/{name}")
    assert _resolve_codex_binary("codex") == "/bin/codex"


def test_resolve_codex_binary_uses_codex_cli_path(monkeypatch, tmp_path):
    binary = tmp_path / "codex"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)

    monkeypatch.setattr(agent_module.shutil, "which", lambda name: None)
    monkeypatch.setenv("CODEX_CLI_PATH", str(binary))
    monkeypatch.setattr(agent_module, "_CODEX_BINARY_FALLBACKS", ())

    assert _resolve_codex_binary("codex") == str(binary)


def test_resolve_codex_binary_uses_codex_app_fallback(monkeypatch, tmp_path):
    binary = tmp_path / "codex"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)

    monkeypatch.setattr(agent_module.shutil, "which", lambda name: None)
    monkeypatch.delenv("CODEX_CLI_PATH", raising=False)
    monkeypatch.setattr(agent_module, "_CODEX_BINARY_FALLBACKS", (str(binary),))

    assert _resolve_codex_binary("codex") == str(binary)


def test_resolve_codex_binary_raises_for_missing_custom_path(tmp_path):
    missing = tmp_path / "codex"
    with pytest.raises(CodexRuntimeError, match="not executable"):
        _resolve_codex_binary(str(missing))


def test_build_codex_context_keeps_partial_data_when_source_fails(monkeypatch):
    def fail_inventory(*args):
        raise RuntimeError("inventory MV unavailable")

    monkeypatch.setattr(
        agent_module.sku_data,
        "fetch_sku_profile",
        lambda *args: {"found": True, "item_id": args[1]},
    )
    monkeypatch.setattr(
        agent_module.sku_data,
        "fetch_sku_sales_history",
        lambda *args: {"months": 1, "history": []},
    )
    monkeypatch.setattr(
        agent_module.sku_data,
        "fetch_sku_forecast",
        lambda *args: {"horizon": 1, "forecast": []},
    )
    monkeypatch.setattr(agent_module.sku_data, "fetch_sku_inventory", fail_inventory)
    monkeypatch.setattr(
        agent_module.sku_data,
        "fetch_sku_accuracy",
        lambda *args: {"metrics": []},
    )
    monkeypatch.setattr(
        agent_module.sku_data,
        "fetch_sku_cluster_peers",
        lambda *args: {"peer_count": 0, "peers": []},
    )

    context = agent_module._build_codex_context(
        pool="pool",
        ctx=SkuChatContext("11958", "ALL", "1401-BULK"),
        history_months=24,
        peer_limit=10,
    )

    assert context["profile"]["found"] is True
    assert context["inventory"] == {
        "available": False,
        "error": "source unavailable",
    }
    assert context["context_errors"] == [{"section": "inventory", "status": "unavailable"}]


@pytest.mark.asyncio
async def test_run_codex_exec_uses_current_cli_flags(monkeypatch):
    calls = {}

    class Proc:
        returncode = 0

        async def communicate(self, input=None):
            calls["input"] = input
            return b"answer", b""

        def kill(self):
            calls["killed"] = True

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        calls["cmd"] = cmd
        calls["kwargs"] = kwargs
        return Proc()

    monkeypatch.setattr(agent_module, "_resolve_codex_binary", lambda binary: "/bin/codex")
    monkeypatch.setattr(agent_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    out = await agent_module._run_codex_exec(
        "hello",
        model_id="gpt-5.5",
        cfg={"codex": {"sandbox": "read-only", "approval_policy": "never"}},
        env={},
        timeout_s=5,
    )

    assert out == "answer"
    assert "--ask-for-approval" not in calls["cmd"]
    assert calls["cmd"][:6] == (
        "/bin/codex",
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "-c",
    )
    assert 'approval_policy="never"' in calls["cmd"]
    assert calls["cmd"][-3:] == ("--model", "gpt-5.5", "-")
    assert calls["input"] == b"hello"


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
    with (
        patch("common.ai.sku_chat.agent._build_codex_context", fake_context),
        patch("common.ai.sku_chat.agent._run_codex_exec", fake_codex),
    ):
        events = [
            ev
            async for ev in agent.stream_turn(
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


@pytest.mark.asyncio
async def test_stream_turn_uses_history_sku_context_for_codex_runtime():
    """Codex mode keeps selected SKU context across follow-up questions."""

    def fake_context(pool, ctx, *, history_months, peer_limit):
        assert pool == "pool"
        assert ctx == SkuChatContext("11958", "ALL", "1401-BULK")
        assert history_months == 24
        assert peer_limit == 10
        return {
            "sku": {
                "item_id": ctx.item_id,
                "customer_group": ctx.customer_group,
                "loc": ctx.loc,
            },
            "profile": {"found": True, "item_id": ctx.item_id},
        }

    async def fake_codex(prompt, *, model_id, cfg, env, timeout_s):
        assert model_id == "gpt-5.4-mini"
        assert "item_id=11958" in prompt
        assert '"loc": "1401-BULK"' in prompt
        return "Codex answer with loaded SKU facts."

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
        },
    )
    history = [
        {
            "role": "assistant",
            "content": "SKU selected: `item_id=11958`, `customer_group=ALL`, `loc=1401-BULK`.",
        }
    ]
    with (
        patch("common.ai.sku_chat.agent._build_codex_context", fake_context),
        patch("common.ai.sku_chat.agent._run_codex_exec", fake_codex),
    ):
        events = [
            ev
            async for ev in agent.stream_turn(
                "all",
                SkuChatContext("", "", ""),
                history=history,
            )
        ]

    assert events[-1]["text"] == "Codex answer with loaded SKU facts."
