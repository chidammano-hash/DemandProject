"""Reversible action ledger with a 24h KPI quiet-period auto-rollback.

Gen-4 Roadmap AI-8 (Stream H Phase 2). When the closed-loop exception
orchestrator applies an action, it writes one row here describing the
applied change *plus* the inverse payload needed to undo it. A background
sweeper runs periodically and rolls back any action whose quiet-period
horizon has elapsed when a KPI regression is detected.

Companion table: ``sql/165_create_fact_reversible_action.sql``.

The KPI regression detector is intentionally a TODO — the first pass
returns ``[]`` so the sweeper is a no-op in production until the detector
is wired in. Tests cover the ``apply`` and sweep mechanics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Default quiet-period horizon. Kept as a named constant (not a magic
# number) so operators can raise/lower it without touching code.
_DEFAULT_QUIET_PERIOD_HOURS: int = 24

VALID_STATUSES: frozenset[str] = frozenset({
    "applied",       # action just committed; still inside the quiet window
    "rolled_back",   # sweeper reverted it due to KPI regression
    "expired",       # quiet window elapsed without regression; action is final
    "confirmed",     # a human explicitly locked it in before expiry
})


@dataclass
class ReversibleAction:
    """Input payload for a single ``apply`` call.

    Attributes:
        action_type: e.g. ``'expedite_transfer'``, ``'emergency_po'``.
        target_kind: e.g. ``'exception_id'``, ``'po_id'``.
        target_id: identifier of the target entity.
        rollback_payload: spec describing how to undo this action.
        applied_by: agent / user id (optional).
        ledger_id: optional cross-link to ``ai_decision_ledger.id``.
        quiet_period_hours: override for the default 24h window.
    """

    action_type: str
    target_kind: str
    target_id: str
    rollback_payload: dict[str, Any] = field(default_factory=dict)
    applied_by: str | None = None
    ledger_id: int | None = None
    quiet_period_hours: int = _DEFAULT_QUIET_PERIOD_HOURS


def _now_utc() -> datetime:
    """Indirection so tests can monkeypatch the clock if needed."""
    return datetime.now(timezone.utc)


def apply(cursor: Any, action: ReversibleAction) -> int:
    """Write one ``status='applied'`` row and return its id.

    Caller owns the transaction. ``expires_at`` is set to
    ``applied_at + quiet_period_hours``.
    """
    if not action.action_type or not action.target_kind or not action.target_id:
        raise ValueError("action_type, target_kind and target_id are required")
    if action.quiet_period_hours <= 0:
        raise ValueError("quiet_period_hours must be positive")

    applied_at = _now_utc()
    expires_at = applied_at + timedelta(hours=action.quiet_period_hours)
    payload_json = json.dumps(action.rollback_payload or {}, sort_keys=True)

    cursor.execute(
        """
        INSERT INTO fact_reversible_action
            (action_type, target_kind, target_id, applied_at, expires_at,
             rollback_payload, status, applied_by, ledger_id)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb, 'applied', %s, %s)
        RETURNING id
        """,
        (
            action.action_type,
            action.target_kind,
            action.target_id,
            applied_at,
            expires_at,
            payload_json,
            action.applied_by,
            action.ledger_id,
        ),
    )
    new_id = cursor.fetchone()[0]
    logger.info(
        "reversible_action apply id=%s type=%s target=%s/%s expires=%s",
        new_id, action.action_type, action.target_kind, action.target_id, expires_at,
    )
    return new_id


def detect_kpi_regressions(cursor: Any, action_ids: list[int]) -> set[int]:
    """Return the subset of *action_ids* whose applied action regressed a KPI.

    TODO(gen-4 AI-8): join against fact_kpi_snapshot / mv_control_tower_kpis
    deltas to flag regressions (e.g. fill-rate drop > threshold, stockout
    spike, cash delta negative vs baseline). For now returns an empty set,
    which makes :func:`rollback_pending` a no-op.
    """
    if not action_ids:
        return set()
    logger.debug("detect_kpi_regressions called for %d actions (stub)", len(action_ids))
    return set()


def _fetch_pending_actions(cursor: Any) -> list[tuple[int, str, str, str, dict[str, Any]]]:
    """Return ``(id, action_type, target_kind, target_id, rollback_payload)`` for actions whose quiet window has elapsed."""
    cursor.execute(
        """
        SELECT id, action_type, target_kind, target_id, rollback_payload
          FROM fact_reversible_action
         WHERE status = 'applied'
           AND expires_at <= NOW()
         ORDER BY expires_at
        """
    )
    rows = cursor.fetchall() or []
    result: list[tuple[int, str, str, str, dict[str, Any]]] = []
    for r in rows:
        payload = r[4]
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                payload = {}
        result.append((r[0], r[1], r[2], r[3], payload or {}))
    return result


def _mark_status(cursor: Any, action_id: int, status: str, reason: str | None = None) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status {status!r}")
    cursor.execute(
        """
        UPDATE fact_reversible_action
           SET status = %s,
               rolled_back_at = CASE WHEN %s = 'rolled_back' THEN NOW() ELSE rolled_back_at END,
               rollback_reason = COALESCE(%s, rollback_reason)
         WHERE id = %s
        """,
        (status, status, reason, action_id),
    )


def _reverse_action(cursor: Any, action_id: int, action_type: str, payload: dict[str, Any]) -> None:
    """Execute the inverse side-effect described by ``payload``.

    TODO(gen-4 AI-8): wire per-action reversal handlers (transfer cancel,
    PO cancel, allocation undo). The scaffold only logs and relies on the
    sweeper to flip status — real mutations land in a later pass.
    """
    logger.info(
        "reversible_action reverse id=%s type=%s payload_keys=%s (stub)",
        action_id, action_type, sorted(payload.keys()),
    )


def rollback_pending(cursor: Any) -> list[int]:
    """Sweeper: find expired 'applied' rows, rollback any with KPI regressions.

    Returns the list of action ids that were rolled back. Rows that expire
    without a regression are marked ``'expired'`` so the sweeper does not
    re-examine them on the next tick.
    """
    pending = _fetch_pending_actions(cursor)
    if not pending:
        return []

    ids = [row[0] for row in pending]
    regressions = detect_kpi_regressions(cursor, ids)

    rolled: list[int] = []
    for action_id, action_type, _target_kind, _target_id, payload in pending:
        if action_id in regressions:
            _reverse_action(cursor, action_id, action_type, payload)
            _mark_status(cursor, action_id, "rolled_back", reason="kpi_regression")
            rolled.append(action_id)
        else:
            _mark_status(cursor, action_id, "expired")
    logger.info(
        "rollback_pending swept=%d rolled_back=%d expired=%d",
        len(pending), len(rolled), len(pending) - len(rolled),
    )
    return rolled


__all__ = [
    "ReversibleAction",
    "VALID_STATUSES",
    "apply",
    "detect_kpi_regressions",
    "rollback_pending",
]
