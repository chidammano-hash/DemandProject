"""SkuChatAgent — the Claude Agent SDK runner for the SKU Chatbot.

Per turn the agent: routes to a model tier, builds the read-only in-process tool
server, runs the Agent SDK ``query`` loop under a wall-clock timeout, and maps
the SDK's streamed messages to SSE event dicts the router relays to the browser.

The SDK is imported lazily; if it is not installed the turn yields a single
``error`` event instead of raising, so the API stays importable without the
``agent`` extra.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from common.ai.sku_chat import auth, model_router, prompts, tools
from common.ai.sku_chat.tools import AgentSdkUnavailableError

log = logging.getLogger(__name__)


@dataclass
class SkuChatContext:
    """The SKU a conversation is scoped to."""

    item_id: str
    customer_group: str
    loc: str


def _import_query():
    try:
        from claude_agent_sdk import (  # type: ignore
            ClaudeAgentOptions,
            query,
        )
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise AgentSdkUnavailableError(
            "claude-agent-sdk is not installed. Run `uv sync --extra agent`."
        ) from exc
    return query, ClaudeAgentOptions


def sdk_message_to_events(msg: Any) -> list[dict[str, Any]]:
    """Map one Agent SDK stream message to SSE event dicts.

    Pure and duck-typed (dispatches on the class name) so it is unit-testable
    with lightweight fakes and never imports the SDK.

    - ``StreamEvent``     -> incremental ``text`` chunks (text_delta only)
    - ``AssistantMessage``-> ``tool`` events (tool_use blocks; text comes via
      StreamEvent so it is not duplicated here)
    - ``ResultMessage``   -> a terminal ``result`` event with text + usage/cost
    """
    name = type(msg).__name__

    if name == "StreamEvent":
        event = getattr(msg, "event", None) or {}
        if isinstance(event, dict) and event.get("type") == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                return [{"type": "text", "chunk": delta["text"]}]
        return []

    if name == "ResultMessage":
        return [
            {
                "type": "result",
                "text": getattr(msg, "result", None),
                "cost_usd": getattr(msg, "total_cost_usd", None),
                "usage": getattr(msg, "usage", None),
            }
        ]

    if name == "AssistantMessage":
        events: list[dict[str, Any]] = []
        for block in getattr(msg, "content", None) or []:
            is_tool = getattr(block, "type", None) == "tool_use" or (
                getattr(block, "name", None) and getattr(block, "input", None) is not None
            )
            if is_tool:
                events.append(
                    {
                        "type": "tool",
                        "name": getattr(block, "name", None),
                        "input": getattr(block, "input", None),
                    }
                )
        return events

    return []


class SkuChatAgent:
    """Runs one SKU-scoped chat turn against the Claude Agent SDK."""

    def __init__(self, pool: Any, config: dict):
        self.pool = pool
        self.config = config or {}

    async def stream_turn(
        self,
        question: str,
        ctx: SkuChatContext,
        *,
        history: list[dict[str, str]] | None = None,
        model_tier: str | None = None,
        model: str | None = None,
        session_id: str | None = None,
        page_focus: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield SSE event dicts for one turn (``meta``/``text``/``tool``/``result``/``error``)."""
        cfg = self.config
        guardrails = cfg.get("guardrails") or {}
        context_cfg = cfg.get("context") or {}
        timeout_s = float(guardrails.get("timeout_seconds", 60))
        max_turns = int(guardrails.get("max_turns", 12))
        max_history = int(guardrails.get("max_history_messages", 20))
        history_months = int(context_cfg.get("history_lookback_months", 24))
        peer_limit = int(context_cfg.get("cluster_peer_limit", 10))
        champion_adjust_enabled = bool((cfg.get("champion_adjust") or {}).get("enabled", True))

        tier, model_id = model_router.select_model(
            question, cfg, override_tier=model_tier, override_model=model
        )
        yield {"type": "meta", "tier": tier, "model": model_id}

        try:
            query_fn, options_cls = _import_query()
            env = auth.resolve_auth_env(cfg)
            tool_server = tools.build_sku_tool_server(
                self.pool,
                history_months=history_months,
                peer_limit=peer_limit,
                session_id=session_id,
                customer_group=ctx.customer_group,
                champion_adjust_enabled=champion_adjust_enabled,
            )
        except (AgentSdkUnavailableError, auth.SkuChatAuthError) as exc:
            yield {"type": "error", "message": str(exc)}
            return

        system_prompt = cfg.get("system_prompt") or prompts.DEFAULT_SYSTEM_PROMPT
        user_prompt = prompts.build_user_prompt(
            question, ctx, history=history, max_history=max_history, page_focus=page_focus
        )
        allowed = (cfg.get("tools") or {}).get("allowed") or tools.SKU_TOOL_NAMES

        options = options_cls(
            model=model_id,
            system_prompt=system_prompt,
            mcp_servers={"sku": tool_server},
            allowed_tools=allowed,
            permission_mode="bypassPermissions",
            setting_sources=[],
            include_partial_messages=True,
            max_turns=max_turns,
            env=env,
        )

        try:
            async with asyncio.timeout(timeout_s):
                async for msg in query_fn(prompt=user_prompt, options=options):
                    for event in sdk_message_to_events(msg):
                        yield event
        except TimeoutError:
            log.warning(
                "sku-chat turn timed out after %ss for %s@%s",
                timeout_s,
                ctx.item_id,
                ctx.loc,
            )
            yield {
                "type": "error",
                "message": f"Response exceeded the {int(timeout_s)}s limit; truncated.",
                "truncated": True,
            }
