"""Phase 3 persistence for the SKU Chatbot — best-effort session/message/call logging.

Every function is best-effort: on any psycopg error (including the tables not
existing because sql/196 hasn't been applied) it logs and no-ops, so the chat
keeps streaming. Tables are created by sql/196_create_sku_chat_log.sql.
"""
from __future__ import annotations

import logging
from typing import Any

import psycopg

from common.core.sql_helpers import row_to_dict_from_cursor

log = logging.getLogger(__name__)


def ensure_session(
    pool: Any,
    session_id: str,
    item_id: str,
    customer_group: str,
    loc: str,
    created_by: str | None = None,
) -> None:
    """Insert the session row (or bump last_active_at if it already exists)."""
    try:
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sku_chat_session "
                "(session_id, item_id, customer_group, loc, created_by) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (session_id) DO UPDATE SET last_active_at = now()",
                [session_id, item_id, customer_group or "", loc, created_by],
            )
    except psycopg.Error:
        log.exception("sku-chat: ensure_session failed for session %s", session_id)


def save_message(
    pool: Any,
    session_id: str,
    role: str,
    content: str,
    model: str | None = None,
    tier: str | None = None,
) -> int | None:
    """Persist one turn; return its message id (or None on failure)."""
    try:
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sku_chat_message (session_id, role, content, model, tier) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                [session_id, role, content, model, tier],
            )
            row = cur.fetchone()
            return int(row[0]) if row else None
    except psycopg.Error:
        log.exception("sku-chat: save_message failed for session %s", session_id)
        return None


def log_call(
    pool: Any,
    session_id: str,
    message_id: int | None,
    *,
    model: str | None,
    tier: str | None,
    usage: dict[str, Any] | None,
    cost_usd: float | None,
    tool_calls: int,
    latency_ms: int,
    truncated: bool,
) -> None:
    """Persist per-turn cost/usage/latency observability."""
    u = usage or {}
    try:
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sku_chat_call_log "
                "(session_id, message_id, model, tier, input_tokens, output_tokens, "
                "cache_read_tokens, total_cost_usd, tool_calls, latency_ms, truncated) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    session_id,
                    message_id,
                    model,
                    tier,
                    u.get("input_tokens"),
                    u.get("output_tokens"),
                    u.get("cache_read_input_tokens"),
                    cost_usd,
                    int(tool_calls),
                    latency_ms,
                    bool(truncated),
                ],
            )
    except psycopg.Error:
        log.exception("sku-chat: log_call failed for session %s", session_id)


def get_session(pool: Any, session_id: str) -> dict[str, Any] | None:
    """Return the session row plus its ordered messages, or None if absent/error."""
    try:
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT session_id, item_id, customer_group, loc, created_by, "
                "created_at, last_active_at FROM sku_chat_session WHERE session_id = %s",
                [session_id],
            )
            srow = cur.fetchone()
            if srow is None:
                return None
            session = row_to_dict_from_cursor(cur, srow)
            cur.execute(
                "SELECT id, role, content, model, tier, created_at "
                "FROM sku_chat_message WHERE session_id = %s ORDER BY id",
                [session_id],
            )
            session["messages"] = [
                row_to_dict_from_cursor(cur, r) for r in cur.fetchall()
            ]
            return session
    except psycopg.Error:
        log.exception("sku-chat: get_session failed for session %s", session_id)
        return None
