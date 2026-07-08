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
import re
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from common.ai.sku_chat import auth, model_router, prompts, sku_data, tools
from common.ai.sku_chat.tools import AgentSdkUnavailableError
from common.core.paths import PROJECT_ROOT

log = logging.getLogger(__name__)

_CODEX_BINARY_FALLBACKS = (
    "/Applications/Codex.app/Contents/Resources/codex",
    "~/Applications/Codex.app/Contents/Resources/codex",
)


@dataclass
class SkuChatContext:
    """The SKU a conversation is scoped to."""

    item_id: str
    customer_group: str
    loc: str


_SKU_FIELD_PATTERNS = {
    "item_id": re.compile(
        r"(?<![\w-])item[_\s-]?id\s*[:=]\s*`?([A-Za-z0-9_.-]+)",
        re.IGNORECASE,
    ),
    "customer_group": re.compile(
        r"(?<![\w-])customer[_\s-]?group\s*[:=]\s*`?([A-Za-z0-9_.-]+)",
        re.IGNORECASE,
    ),
    "loc": re.compile(
        r"(?<![\w-])loc(?:ation)?\s*[:=]\s*`?([A-Za-z0-9_.-]+)",
        re.IGNORECASE,
    ),
}
_SKU_AT_LOC_PATTERNS = (
    re.compile(
        r"(?<![\w.-])sku\s+([A-Za-z0-9][A-Za-z0-9_.-]*)\s*@\s*([A-Za-z0-9][A-Za-z0-9_.-]*)",
        re.IGNORECASE,
    ),
    re.compile(r"(?<![\w.-])([0-9][A-Za-z0-9_.-]*)\s*@\s*([A-Za-z0-9][A-Za-z0-9_.-]*)"),
)
_CODEX_ACTION_RE = re.compile(
    r"<sku_chat_action>\s*(\{.*?\})\s*</sku_chat_action>",
    re.DOTALL,
)


def _clean_sku_context_value(value: str) -> str:
    """Trim punctuation commonly attached to SKU values in chat text."""
    return value.strip().strip("`'\".,;:()[]{}")


def _overlay_sku_context(base: SkuChatContext, updates: SkuChatContext) -> SkuChatContext:
    """Return ``base`` with non-empty fields from ``updates`` applied."""
    return SkuChatContext(
        updates.item_id or base.item_id,
        updates.customer_group or base.customer_group,
        updates.loc or base.loc,
    )


def _sku_context_from_text(text: str) -> SkuChatContext:
    """Extract SKU context fields from a single chat message."""
    found: dict[str, str] = {"item_id": "", "customer_group": "", "loc": ""}
    for field, pattern in _SKU_FIELD_PATTERNS.items():
        for match in pattern.finditer(text):
            found[field] = _clean_sku_context_value(match.group(1))

    for pattern in _SKU_AT_LOC_PATTERNS:
        for match in pattern.finditer(text):
            found["item_id"] = _clean_sku_context_value(match.group(1))
            found["loc"] = _clean_sku_context_value(match.group(2))

    return SkuChatContext(found["item_id"], found["customer_group"], found["loc"])


def _effective_sku_context(
    ctx: SkuChatContext,
    question: str,
    history: list[dict[str, str]] | None,
    *,
    max_history: int,
) -> SkuChatContext:
    """Fill missing request context from recent transcript text.

    The UI can send follow-up turns like "all" after a selected-SKU message while
    the structured item fields are blank. The explicit request body still wins;
    transcript inference only fills gaps.
    """
    inferred = SkuChatContext("", "", "")
    for msg in (history or [])[-max_history:]:
        inferred = _overlay_sku_context(
            inferred,
            _sku_context_from_text(msg.get("content", "")),
        )
    inferred = _overlay_sku_context(inferred, _sku_context_from_text(question))
    return SkuChatContext(
        ctx.item_id or inferred.item_id,
        ctx.customer_group or inferred.customer_group,
        ctx.loc or inferred.loc,
    )


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


def _extract_codex_adjustment_action(answer: str) -> tuple[str, str | None]:
    """Strip a Codex action block and return its adjustment rationale, if any."""
    rationale: str | None = None
    for match in _CODEX_ACTION_RE.finditer(answer):
        try:
            action = json.loads(match.group(1))
        except json.JSONDecodeError:
            log.warning("sku-chat Codex returned malformed action JSON")
            continue
        adjustment = action.get("apply_champion_adjustment")
        if isinstance(adjustment, dict) and isinstance(adjustment.get("rationale"), str):
            rationale = adjustment["rationale"].strip() or None

    clean_answer = _CODEX_ACTION_RE.sub("", answer).strip()
    return clean_answer, rationale


def _is_executable(path: str) -> bool:
    """Return true when a candidate CLI path exists and can be executed."""
    expanded = os.path.expanduser(path)
    return os.path.isfile(expanded) and os.access(expanded, os.X_OK)


def _resolve_codex_binary(binary: str) -> str:
    """Resolve the Codex CLI binary from config, PATH, or Codex.app fallback."""
    expanded = os.path.expanduser(binary)
    if os.sep in binary:
        if _is_executable(expanded):
            return expanded
        raise CodexRuntimeError(f"configured codex.binary is not executable: {expanded}")

    found = shutil.which(binary)
    if found:
        return found

    if binary == "codex":
        candidates = [os.getenv("CODEX_CLI_PATH"), *_CODEX_BINARY_FALLBACKS]
        for candidate in candidates:
            if candidate and _is_executable(candidate):
                return os.path.expanduser(candidate)

    raise CodexRuntimeError(
        "codex CLI is not installed or not visible to the API process. "
        "Set config/ai/sku_chat_config.yaml codex.binary to the Codex CLI path, "
        "set CODEX_CLI_PATH, install codex on PATH, or switch runtime.provider "
        "back to 'claude'."
    )


def _fetch_codex_context_section(
    section: str,
    errors: list[dict[str, str]],
    fetcher: Any,
    *args: Any,
) -> dict[str, Any]:
    """Fetch one context section without aborting the whole Codex turn."""
    try:
        return fetcher(*args)
    except Exception:  # noqa: BLE001, RUF100 — unavailable sources should not abort chat
        log.exception("sku-chat Codex context section %s failed", section)
        errors.append({"section": section, "status": "unavailable"})
        return {"available": False, "error": "source unavailable"}


def _build_codex_context(
    pool: Any,
    ctx: SkuChatContext,
    *,
    history_months: int,
    peer_limit: int,
) -> dict[str, Any]:
    """Fetch the read-only SKU context Codex receives instead of live MCP tools."""
    errors: list[dict[str, str]] = []
    data: dict[str, Any] = {
        "sku": {
            "item_id": ctx.item_id,
            "customer_group": ctx.customer_group,
            "loc": ctx.loc,
        },
        "context_errors": errors,
    }
    if not (ctx.item_id and ctx.loc):
        data.update(
            {
                "profile": None,
                "sales_history": None,
                "forecast": None,
                "inventory": None,
                "accuracy": None,
                "cluster_peers": None,
            }
        )
        return data

    data["profile"] = _fetch_codex_context_section(
        "profile",
        errors,
        sku_data.fetch_sku_profile,
        pool,
        ctx.item_id,
        ctx.customer_group,
        ctx.loc,
    )
    data["sales_history"] = _fetch_codex_context_section(
        "sales_history",
        errors,
        sku_data.fetch_sku_sales_history,
        pool,
        ctx.item_id,
        ctx.loc,
        history_months,
    )
    data["forecast"] = _fetch_codex_context_section(
        "forecast",
        errors,
        sku_data.fetch_sku_forecast,
        pool,
        ctx.item_id,
        ctx.loc,
    )
    data["inventory"] = _fetch_codex_context_section(
        "inventory",
        errors,
        sku_data.fetch_sku_inventory,
        pool,
        ctx.item_id,
        ctx.loc,
        history_months,
    )
    data["accuracy"] = _fetch_codex_context_section(
        "accuracy",
        errors,
        sku_data.fetch_sku_accuracy,
        pool,
        ctx.item_id,
        ctx.customer_group,
        ctx.loc,
    )
    data["cluster_peers"] = _fetch_codex_context_section(
        "cluster_peers",
        errors,
        sku_data.fetch_sku_cluster_peers,
        pool,
        ctx.item_id,
        ctx.customer_group,
        ctx.loc,
        peer_limit,
    )
    return data


def _build_codex_prompt(
    question: str,
    ctx: SkuChatContext,
    cfg: dict,
    *,
    history: list[dict[str, str]] | None,
    max_history: int,
    page_focus: str | None,
    context_data: dict[str, Any],
    champion_adjust_enabled: bool,
) -> str:
    """Compose the non-interactive Codex prompt from system rules + data."""
    system_prompt = cfg.get("system_prompt") or prompts.DEFAULT_SYSTEM_PROMPT
    user_prompt = prompts.build_user_prompt(
        question, ctx, history=history, max_history=max_history, page_focus=page_focus
    )
    if champion_adjust_enabled:
        adjustment_note = (
            "If the planner asks you to stage or apply a champion forecast adjustment, "
            "first explain your evidence in normal text. If, and only if, an adjustment "
            "is warranted, append exactly one final action block using this shape and no "
            "extra text after it: "
            '<sku_chat_action>{"apply_champion_adjustment":{"rationale":"short evidence-based '
            'reason"}}</sku_chat_action>. The server will stage the proposal for planner '
            "approval; it will not apply the forecast until approval."
        )
    else:
        adjustment_note = (
            "Champion-adjustment staging is disabled for this chat. Explain recommendations, "
            "but do not request a staged adjustment."
        )
    return (
        f"{system_prompt}\n\n"
        "Runtime note: you are running through Codex CLI in non-interactive mode. "
        "Use only the JSON context below for business facts. Do not edit files, run "
        f"commands, or claim to have called live tools. {adjustment_note}\n\n"
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
    resolved_binary = _resolve_codex_binary(binary)

    cmd = [
        resolved_binary,
        "exec",
        "--ephemeral",
        "--sandbox",
        sandbox,
        "-c",
        f'approval_policy="{approval}"',
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
        raise CodexRuntimeError(f"Codex exceeded the {int(timeout_s)}s limit; truncated.") from exc

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
        effective_ctx = _effective_sku_context(
            ctx,
            question,
            history,
            max_history=max_history,
        )

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
                effective_ctx,
                cfg=cfg,
                model_id=model_id,
                history=history,
                max_history=max_history,
                history_months=history_months,
                peer_limit=peer_limit,
                session_id=session_id,
                champion_adjust_enabled=champion_adjust_enabled,
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
                customer_group=effective_ctx.customer_group,
                champion_adjust_enabled=champion_adjust_enabled,
            )
        except (AgentSdkUnavailableError, auth.SkuChatAuthError) as exc:
            yield {"type": "error", "message": str(exc)}
            return

        system_prompt = cfg.get("system_prompt") or prompts.DEFAULT_SYSTEM_PROMPT
        user_prompt = prompts.build_user_prompt(
            question, effective_ctx, history=history, max_history=max_history, page_focus=page_focus
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
                effective_ctx.item_id,
                effective_ctx.loc,
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
        session_id: str | None,
        champion_adjust_enabled: bool,
        page_focus: str | None,
        timeout_s: float,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield SSE events for a Codex CLI-backed turn."""
        staged_event: dict[str, Any] | None = None
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
                champion_adjust_enabled=champion_adjust_enabled,
            )
            answer = await _run_codex_exec(
                prompt,
                model_id=model_id,
                cfg=cfg,
                env=env,
                timeout_s=timeout_s,
            )
            answer, rationale = _extract_codex_adjustment_action(answer)
            if champion_adjust_enabled and rationale and ctx.item_id and ctx.loc:
                from common.ai.sku_chat import champion_adjust

                try:
                    staged = await asyncio.to_thread(
                        champion_adjust.stage_adjustment,
                        self.pool,
                        session_id=session_id,
                        item_id=ctx.item_id,
                        customer_group=ctx.customer_group,
                        loc=ctx.loc,
                        rationale=rationale,
                    )
                except champion_adjust.AdjustmentError as exc:
                    staged_event = {
                        "type": "tool",
                        "name": "mcp__sku__apply_champion_adjustment",
                        "input": {
                            "item_id": ctx.item_id,
                            "loc": ctx.loc,
                            "rationale": rationale,
                        },
                        "staged": False,
                        "error": str(exc),
                    }
                else:
                    staged_event = {
                        "type": "tool",
                        "name": "mcp__sku__apply_champion_adjustment",
                        "input": {
                            "item_id": ctx.item_id,
                            "loc": ctx.loc,
                            "rationale": rationale,
                        },
                        "staged": True,
                        "approval_id": staged["approval_id"],
                    }
        except (auth.SkuChatAuthError, CodexRuntimeError) as exc:
            yield {"type": "error", "message": str(exc)}
            return

        if staged_event is not None:
            yield staged_event
        yield {"type": "text", "chunk": answer}
        yield {"type": "result", "text": answer, "cost_usd": None, "usage": None}
