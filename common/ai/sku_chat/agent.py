"""SkuChatAgent — selectable Claude/Codex runner for the SKU Chatbot.

Per turn the agent routes to a model tier, then runs one of two configured
runtimes:

* ``claude`` uses the Claude Agent SDK with the existing in-process MCP tools.
* ``codex`` invokes ``codex exec`` with a read-only SKU context snapshot.

Both runtimes are lazy. If their local CLI/SDK dependency is missing, the turn
yields an ``error`` event instead of breaking API startup.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from common.ai.sku_chat import auth, model_router, prompts, sku_data, tools
from common.ai.sku_chat.tools import AgentSdkUnavailableError
from common.core.paths import PROJECT_ROOT

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


class CodexRuntimeError(RuntimeError):
    """Raised when the Codex runtime cannot complete a turn."""


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


def _json_context(data: dict[str, Any]) -> str:
    """Compact, deterministic JSON block for prompt context."""
    return json.dumps(data, default=str, indent=2, sort_keys=True)


def _build_codex_context(
    pool: Any,
    ctx: SkuChatContext,
    *,
    history_months: int,
    peer_limit: int,
) -> dict[str, Any]:
    """Fetch the read-only SKU context Codex receives instead of live MCP tools."""
    return {
        "sku": {
            "item_id": ctx.item_id,
            "customer_group": ctx.customer_group,
            "loc": ctx.loc,
        },
        "profile": sku_data.fetch_sku_profile(
            pool, ctx.item_id, ctx.customer_group, ctx.loc
        ) if ctx.item_id and ctx.loc else None,
        "sales_history": sku_data.fetch_sku_sales_history(
            pool, ctx.item_id, ctx.loc, history_months
        ) if ctx.item_id and ctx.loc else None,
        "forecast": sku_data.fetch_sku_forecast(
            pool, ctx.item_id, ctx.loc
        ) if ctx.item_id and ctx.loc else None,
        "inventory": sku_data.fetch_sku_inventory(
            pool, ctx.item_id, ctx.loc, history_months
        ) if ctx.item_id and ctx.loc else None,
        "accuracy": sku_data.fetch_sku_accuracy(
            pool, ctx.item_id, ctx.customer_group, ctx.loc
        ) if ctx.item_id and ctx.loc else None,
        "cluster_peers": sku_data.fetch_sku_cluster_peers(
            pool, ctx.item_id, ctx.customer_group, ctx.loc, peer_limit
        ) if ctx.item_id and ctx.loc else None,
    }


def _build_codex_prompt(
    question: str,
    ctx: SkuChatContext,
    cfg: dict,
    *,
    history: list[dict[str, str]] | None,
    max_history: int,
    page_focus: str | None,
    context_data: dict[str, Any],
) -> str:
    """Compose the non-interactive Codex prompt from system rules + data."""
    system_prompt = cfg.get("system_prompt") or prompts.DEFAULT_SYSTEM_PROMPT
    user_prompt = prompts.build_user_prompt(
        question, ctx, history=history, max_history=max_history, page_focus=page_focus
    )
    return (
        f"{system_prompt}\n\n"
        "Runtime note: you are running through Codex CLI in non-interactive mode. "
        "Use only the JSON context below for business facts. Do not edit files, run "
        "commands, or claim to have called live tools. The champion-adjustment "
        "approval tool is only available in the Claude runtime; in Codex mode, "
        "explain recommendations but do not stage changes.\n\n"
        f"SKU context JSON:\n{_json_context(context_data)}\n\n"
        f"{user_prompt}"
    )


async def _run_codex_exec(
    prompt: str,
    *,
    model_id: str,
    cfg: dict,
    env: dict[str, str],
    timeout_s: float,
) -> str:
    """Run Codex CLI non-interactively and return the final assistant text."""
    codex_cfg = cfg.get("codex") or {}
    binary = str(codex_cfg.get("binary", "codex"))
    sandbox = str(codex_cfg.get("sandbox", "read-only"))
    approval = str(codex_cfg.get("approval_policy", "never"))
    cwd = str(PROJECT_ROOT)

    if os.sep not in binary and shutil.which(binary) is None:
        raise CodexRuntimeError(
            "codex CLI is not installed or not on PATH. Install/sign in to Codex, "
            "or switch config/ai/sku_chat_config.yaml runtime.provider back to 'claude'."
        )

    cmd = [
        binary,
        "exec",
        "--ephemeral",
        "--sandbox",
        sandbox,
        "--ask-for-approval",
        approval,
        "--model",
        model_id,
        "-",
    ]
    proc_env = {**os.environ, **env}
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            env=proc_env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise CodexRuntimeError(f"could not start codex exec: {exc}") from exc
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=prompt.encode("utf-8")),
            timeout=timeout_s,
        )
    except TimeoutError as exc:
        proc.kill()
        await proc.communicate()
        raise CodexRuntimeError(
            f"Codex exceeded the {int(timeout_s)}s limit; truncated."
        ) from exc

    out = stdout.decode("utf-8", errors="replace").strip()
    err = stderr.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        detail = err[-1000:] or out[-1000:] or "no output"
        raise CodexRuntimeError(f"codex exec failed ({proc.returncode}): {detail}")
    return out


class SkuChatAgent:
    """Runs one SKU-scoped chat turn against the configured agent runtime."""

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

        try:
            provider = auth.runtime_provider(cfg)
        except auth.SkuChatAuthError as exc:
            yield {"type": "error", "message": str(exc)}
            return
        tier, model_id = model_router.select_model(
            question,
            cfg,
            provider=provider,
            override_tier=model_tier,
            override_model=model,
        )
        yield {"type": "meta", "tier": tier, "model": model_id, "runtime": provider}

        if provider == "codex":
            async for event in self._stream_codex_turn(
                question,
                ctx,
                cfg=cfg,
                model_id=model_id,
                history=history,
                max_history=max_history,
                history_months=history_months,
                peer_limit=peer_limit,
                page_focus=page_focus,
                timeout_s=timeout_s,
            ):
                yield event
            return

        try:
            query_fn, options_cls = _import_query()
            env = auth.resolve_auth_env(cfg, provider=provider)
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

    async def _stream_codex_turn(
        self,
        question: str,
        ctx: SkuChatContext,
        *,
        cfg: dict,
        model_id: str,
        history: list[dict[str, str]] | None,
        max_history: int,
        history_months: int,
        peer_limit: int,
        page_focus: str | None,
        timeout_s: float,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield SSE events for a Codex CLI-backed turn."""
        try:
            env = auth.resolve_auth_env(cfg, provider="codex")
            try:
                context_data = await asyncio.to_thread(
                    _build_codex_context,
                    self.pool,
                    ctx,
                    history_months=history_months,
                    peer_limit=peer_limit,
                )
            except Exception as exc:  # noqa: BLE001, RUF100 — stream context-load failures
                log.exception("sku-chat Codex context load failed for %s@%s", ctx.item_id, ctx.loc)
                raise CodexRuntimeError(
                    "Could not load SKU context for the Codex runtime."
                ) from exc
            prompt = _build_codex_prompt(
                question,
                ctx,
                cfg,
                history=history,
                max_history=max_history,
                page_focus=page_focus,
                context_data=context_data,
            )
            answer = await _run_codex_exec(
                prompt,
                model_id=model_id,
                cfg=cfg,
                env=env,
                timeout_s=timeout_s,
            )
        except (auth.SkuChatAuthError, CodexRuntimeError) as exc:
            yield {"type": "error", "message": str(exc)}
            return

        yield {"type": "text", "chunk": answer}
        yield {"type": "result", "text": answer, "cost_usd": None, "usage": None}
