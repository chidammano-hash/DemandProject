"""Agentic champion-forecast adjustment for the SKU Chatbot — staging + approval.

The chatbot's ``apply_champion_adjustment`` tool calls :func:`stage_adjustment`,
which reuses the tested adjuster engine (``common/ai/champion_adjust_service``):
``adjust_dfu`` produces a guardrail-validated preview and we persist it as a
``pending`` row. The forecast is NOT written. Only when the planner approves
(``POST /sku-chat/adjustment/{id}``) does :func:`apply_adjustment` call
``save_adjustment`` — the same guarded write the AI Adjust panel uses — so the
agent can never mutate ``fact_ai_champion_forecast`` directly.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

import psycopg
from psycopg.types.json import Jsonb
from pydantic import ValidationError

from common.ai import champion_adjust_service as svc
from common.core.sql_helpers import row_to_dict_from_cursor

log = logging.getLogger(__name__)


class AdjustmentError(RuntimeError):
    """Raised when an adjustment cannot be staged or applied."""


def stage_adjustment(
    pool: Any,
    *,
    session_id: str | None,
    item_id: str,
    customer_group: str,
    loc: str,
    rationale: str,
    created_by: str | None = None,
) -> dict[str, Any]:
    """Build a guardrail-validated preview and persist it as a pending row.

    Returns ``{"approval_id", "preview"}``. Raises :class:`AdjustmentError`
    (e.g. no champion forecast for this SKU, or the adjuster's provider is
    misconfigured) so the tool can report a clean message to the planner.
    """
    try:
        preview = svc.adjust_dfu(item_id, loc, user_comment=rationale).to_dict()
    except (ValueError, psycopg.Error) as exc:  # NoChampionForecast/UnknownProvider are ValueError
        raise AdjustmentError(f"could not prepare adjustment ({type(exc).__name__})") from exc

    approval_id = str(uuid.uuid4())
    try:
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sku_chat_pending_adjustment "
                "(approval_id, session_id, item_id, customer_group, loc, preview, created_by) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [approval_id, session_id, item_id, customer_group or "", loc, Jsonb(preview), created_by],
            )
    except psycopg.Error as exc:
        raise AdjustmentError("could not stage adjustment") from exc
    return {"approval_id": approval_id, "preview": preview}


def list_pending(pool: Any, session_id: str) -> list[dict[str, Any]]:
    """Return the still-pending adjustments for a session (best-effort)."""
    try:
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT approval_id, item_id, customer_group, loc, preview, created_at "
                "FROM sku_chat_pending_adjustment "
                "WHERE session_id = %s AND status = 'pending' ORDER BY created_at",
                [session_id],
            )
            return [row_to_dict_from_cursor(cur, r) for r in cur.fetchall()]
    except psycopg.Error:
        log.exception("sku-chat: list_pending adjustments failed for session %s", session_id)
        return []


def _get(pool: Any, approval_id: str) -> dict[str, Any] | None:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT approval_id, session_id, item_id, customer_group, loc, preview, status "
            "FROM sku_chat_pending_adjustment WHERE approval_id = %s",
            [approval_id],
        )
        row = cur.fetchone()
        return row_to_dict_from_cursor(cur, row) if row is not None else None


def _mark(pool: Any, approval_id: str, status: str) -> None:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE sku_chat_pending_adjustment SET status = %s, decided_at = now() "
            "WHERE approval_id = %s",
            [status, approval_id],
        )


def reject_adjustment(pool: Any, approval_id: str) -> dict[str, Any]:
    """Mark a pending adjustment rejected (no forecast write)."""
    try:
        _mark(pool, approval_id, "rejected")
    except psycopg.Error as exc:
        raise AdjustmentError("could not reject adjustment") from exc
    return {"approval_id": approval_id, "status": "rejected"}


def apply_adjustment(
    pool: Any, approval_id: str, *, provider: str | None = None
) -> dict[str, Any]:
    """Approve + apply a staged adjustment via the guarded ``save_adjustment``.

    Quantities are re-derived server-side from the champion baseline and the
    guardrails re-applied inside ``save_adjustment`` — the staged numbers are
    never trusted blindly.
    """
    try:
        pend = _get(pool, approval_id)
    except psycopg.Error as exc:
        raise AdjustmentError("could not load adjustment") from exc
    if pend is None or pend.get("status") != "pending":
        raise AdjustmentError("unknown or already-decided adjustment")

    preview = pend["preview"]  # JSONB → dict
    recommendation = {
        "recommendation_code": preview.get("recommendation_code"),
        "pct_change": preview.get("rec_pct_change"),
        "proposed_qty": preview.get("proposed_qty"),
        "apply_horizon_months": preview.get("apply_horizon_months", 3),
        "confidence": preview.get("confidence", 0.0),
        "rationale": preview.get("rationale", ""),
        "evidence_keys": preview.get("evidence_keys") or [],
    }
    try:
        result = svc.save_adjustment(
            pend["item_id"], pend["loc"], provider=provider, recommendation=recommendation
        )
    except (ValueError, ValidationError, psycopg.Error) as exc:
        raise AdjustmentError(f"could not apply adjustment ({type(exc).__name__})") from exc

    _mark(pool, approval_id, "approved")
    return {"approval_id": approval_id, "status": "approved", "result": result}
