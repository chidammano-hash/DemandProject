"""Read-only SDK MCP tool server for the SKU Chatbot.

Builds the in-process Claude Agent SDK tool server that exposes the per-SKU read
functions in :mod:`sku_data` to Claude. Each tool handler offloads its blocking
psycopg call to a thread so it never blocks the event loop. The SDK is imported
lazily (``uv sync --extra agent``) so this module loads without it installed —
the hard dependency only bites when :func:`build_sku_tool_server` is called.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from common.ai.sku_chat import sku_data

# Fully-qualified names (``mcp__<server>__<tool>``) for the allow-list.
SKU_TOOL_NAMES = [
    "mcp__sku__search_skus",
    "mcp__sku__get_sku_profile",
    "mcp__sku__get_sku_sales_history",
    "mcp__sku__get_sku_forecast",
    "mcp__sku__get_sku_inventory",
    "mcp__sku__get_sku_accuracy",
    "mcp__sku__get_sku_cluster_peers",
    "mcp__sku__apply_champion_adjustment",  # registered only when champion_adjust is enabled
]


class AgentSdkUnavailableError(RuntimeError):
    """Raised when ``claude-agent-sdk`` is required but not installed."""


def _import_sdk():
    try:
        from claude_agent_sdk import (  # type: ignore
            create_sdk_mcp_server,
            tool,
        )
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise AgentSdkUnavailableError(
            "claude-agent-sdk is not installed. Run `uv sync --extra agent`."
        ) from exc
    return tool, create_sdk_mcp_server


def _ok(data: Any) -> dict[str, Any]:
    """Wrap a JSON-serialisable payload as an MCP text tool result."""
    return {"content": [{"type": "text", "text": json.dumps(data, default=str)}]}


def build_sku_tool_server(
    pool: Any,
    *,
    history_months: int = 24,
    peer_limit: int = 10,
    session_id: str | None = None,
    customer_group: str = "",
    champion_adjust_enabled: bool = False,
):
    """Create the in-process ``sku`` MCP server bound to this DB pool.

    When ``champion_adjust_enabled`` is set, a write-capable
    ``apply_champion_adjustment`` tool is registered that STAGES a proposal for
    planner approval (it does not write the forecast itself).

    Raises :class:`AgentSdkUnavailableError` if the Agent SDK is not installed.
    """
    tool, create_sdk_mcp_server = _import_sdk()

    @tool(
        "search_skus",
        "Find SKUs whose item_id matches a search string. Returns up to `limit` "
        "(item_id, customer_group, loc) keys.",
        {"query": str, "limit": int},
    )
    async def search_skus(args: dict[str, Any]):
        data = await asyncio.to_thread(
            sku_data.search_skus, pool, args["query"], args.get("limit", 10)
        )
        return _ok(data)

    @tool(
        "get_sku_profile",
        "Demand-behaviour profile for one SKU: mean/CV, intermittency, "
        "seasonality, ABC-XYZ class, ml_cluster, execution lag.",
        {"item_id": str, "customer_group": str, "loc": str},
    )
    async def get_sku_profile(args: dict[str, Any]):
        data = await asyncio.to_thread(
            sku_data.fetch_sku_profile,
            pool,
            args["item_id"],
            args.get("customer_group", ""),
            args["loc"],
        )
        return _ok(data)

    @tool(
        "get_sku_sales_history",
        "Monthly demand history for one SKU (item + location).",
        {"item_id": str, "loc": str, "months": int},
    )
    async def get_sku_sales_history(args: dict[str, Any]):
        data = await asyncio.to_thread(
            sku_data.fetch_sku_sales_history,
            pool,
            args["item_id"],
            args["loc"],
            args.get("months", history_months),
        )
        return _ok(data)

    @tool(
        "get_sku_forecast",
        "Forward production forecast (latest plan version) with confidence "
        "bands and the model that generated each month.",
        {"item_id": str, "loc": str},
    )
    async def get_sku_forecast(args: dict[str, Any]):
        data = await asyncio.to_thread(
            sku_data.fetch_sku_forecast, pool, args["item_id"], args["loc"]
        )
        return _ok(data)

    @tool(
        "get_sku_inventory",
        "Monthly inventory position: on-hand, on-order, sales, lead time.",
        {"item_id": str, "loc": str, "months": int},
    )
    async def get_sku_inventory(args: dict[str, Any]):
        data = await asyncio.to_thread(
            sku_data.fetch_sku_inventory,
            pool,
            args["item_id"],
            args["loc"],
            args.get("months", history_months),
        )
        return _ok(data)

    @tool(
        "get_sku_accuracy",
        "Per-model, per-lag forecast accuracy for one SKU: WAPE, bias, accuracy.",
        {"item_id": str, "customer_group": str, "loc": str},
    )
    async def get_sku_accuracy(args: dict[str, Any]):
        data = await asyncio.to_thread(
            sku_data.fetch_sku_accuracy,
            pool,
            args["item_id"],
            args.get("customer_group", ""),
            args["loc"],
        )
        return _ok(data)

    @tool(
        "get_sku_cluster_peers",
        "Other SKUs in the same ml_cluster as this SKU, for comparison.",
        {"item_id": str, "customer_group": str, "loc": str},
    )
    async def get_sku_cluster_peers(args: dict[str, Any]):
        data = await asyncio.to_thread(
            sku_data.fetch_sku_cluster_peers,
            pool,
            args["item_id"],
            args.get("customer_group", ""),
            args["loc"],
            peer_limit,
        )
        return _ok(data)

    tool_list = [
        search_skus,
        get_sku_profile,
        get_sku_sales_history,
        get_sku_forecast,
        get_sku_inventory,
        get_sku_accuracy,
        get_sku_cluster_peers,
    ]

    if champion_adjust_enabled:
        from common.ai.sku_chat import champion_adjust

        @tool(
            "apply_champion_adjustment",
            "Propose an adjustment to THIS SKU's champion forecast (e.g. scale the next few "
            "months up or down). This STAGES the proposal and requests the planner's approval "
            "in the chat — it is NOT applied until the planner approves. Pass a short rationale "
            "describing the change you recommend and why.",
            {"item_id": str, "loc": str, "rationale": str},
        )
        async def apply_champion_adjustment(args: dict[str, Any]):
            try:
                staged = await asyncio.to_thread(
                    champion_adjust.stage_adjustment,
                    pool,
                    session_id=session_id,
                    item_id=args["item_id"],
                    customer_group=customer_group,
                    loc=args["loc"],
                    rationale=args.get("rationale", ""),
                )
            except champion_adjust.AdjustmentError as exc:
                return _ok({"staged": False, "error": str(exc)})
            prev = staged["preview"]
            return _ok(
                {
                    "staged": True,
                    "approval_id": staged["approval_id"],
                    "recommendation_code": prev.get("recommendation_code"),
                    "pct_change": prev.get("rec_pct_change"),
                    "confidence": prev.get("confidence"),
                    "months": prev.get("months"),
                    "rationale": prev.get("rationale"),
                    "note": (
                        "Staged — the planner must approve before this is applied. Tell them "
                        "what you propose; an approval card is shown below the answer."
                    ),
                }
            )

        tool_list.append(apply_champion_adjustment)

    return create_sdk_mcp_server(name="sku", version="1.0.0", tools=tool_list)
