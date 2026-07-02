"""Prompt construction for the SKU Chatbot.

``DEFAULT_SYSTEM_PROMPT`` is the fallback persona when the config omits one.
``build_user_prompt`` injects the SKU context + recent history into the user
turn (kept out of the system prompt so the system prefix stays stable).
"""
from __future__ import annotations

from typing import Any

DEFAULT_SYSTEM_PROMPT = (
    "You are a supply-chain SKU specialist embedded in the Supply Chain Command "
    "Center. You answer a planner's questions about ONE SKU (item + customer "
    "group + location) using ONLY the read-only tools provided.\n\n"
    "Rules:\n"
    "- Every numeric claim (demand, forecast, accuracy, bias, inventory, lead "
    "time, DOS) MUST come from a tool result. Never invent numbers.\n"
    "- Call the smallest set of tools needed, then answer concisely.\n"
    "- When reasoning about 'why', ground each step in data you fetched.\n"
    "- If a question is out of scope, or asks to change/write data, say you can "
    "only read and discuss this SKU's analytics.\n"
    "- The SKU grain is (item_id, customer_group, loc); use the full grain when "
    "the customer group is known."
)


def build_user_prompt(
    question: str,
    ctx: Any,
    *,
    history: list[dict[str, str]] | None = None,
    max_history: int = 20,
    page_focus: str | None = None,
) -> str:
    """Compose the user turn: page context, SKU scope, transcript, then the question.

    ``ctx`` is duck-typed: any object exposing ``item_id``, ``customer_group``,
    and ``loc`` attributes. ``page_focus`` is a short description of the UI page
    the planner is on, so answers are tailored to that page.
    """
    lines: list[str] = []
    if page_focus:
        lines.append(f"Page context: {page_focus}")

    item_id = getattr(ctx, "item_id", "")
    if item_id:
        cg = getattr(ctx, "customer_group", "") or "ANY"
        lines.append(
            f"SKU under discussion: item_id={item_id}, "
            f"customer_group={cg}, loc={getattr(ctx, 'loc', '')}."
        )
    else:
        lines.append(
            "No specific SKU is selected — if a SKU-level answer is needed, ask the "
            "planner which SKU, or use search_skus to resolve one."
        )

    if history:
        recent = history[-max_history:]
        lines.append("\nConversation so far:")
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
    lines.append(f"\nUser question: {question}")
    return "\n".join(lines)
