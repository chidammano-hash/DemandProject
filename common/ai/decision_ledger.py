"""Append-only, hash-chained ledger for AI agent decisions.

Gen-4 roadmap AI-10 P0. Every action issued by an AI agent is recorded as
one immutable row. The DB trigger in `sql/137_create_ai_decision_ledger.sql`
enforces both append-only semantics and chain integrity; this module is the
canonical client for writers and verifiers.

Writers call :func:`append_decision`. Auditors call :func:`verify_chain` to
walk the full ledger and confirm every row's hash matches the canonical
computation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

GENESIS_HASH: str = "0" * 64

VALID_TIERS: frozenset[str] = frozenset({
    "advisory",            # agent proposes, human decides
    "suggestive",          # agent proposes with a default-yes in UI, human confirms
    "auto_within_policy",  # agent auto-applies if within policy guardrails
    "autonomous",          # agent auto-applies without confirmation (rare)
})


@dataclass(frozen=True)
class DecisionRecord:
    """Input payload for a single ledger append."""

    agent_id: str
    action_type: str
    autonomy_tier: str
    subject_kind: str | None = None
    subject_id: str | None = None
    payload: dict[str, Any] | None = None
    policy_id: str | None = None
    actor: str | None = None
    outcome: str | None = None


def compute_row_hash(record: DecisionRecord, prev_hash: str) -> str:
    """SHA-256 over the canonical tuple. Matches the DB trigger definition."""
    payload_json = json.dumps(record.payload or {}, sort_keys=True, separators=(",", ":"))
    canonical = "|".join([
        record.agent_id or "",
        record.action_type or "",
        record.autonomy_tier or "",
        record.subject_kind or "",
        record.subject_id or "",
        payload_json,
        record.policy_id or "",
        prev_hash,
    ])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _fetch_latest_hash(cursor: Any) -> str:
    cursor.execute("SELECT row_hash FROM ai_decision_ledger ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    return row[0] if row and row[0] else GENESIS_HASH


def append_decision(cursor: Any, record: DecisionRecord) -> tuple[int, str]:
    """Append one decision row. Returns (id, row_hash).

    Caller owns the transaction: the ledger trigger rejects bad chains by
    raising, so wrap this call in its own savepoint when writing multiple
    rows if you want partial-failure isolation.
    """
    if record.autonomy_tier not in VALID_TIERS:
        raise ValueError(
            f"Invalid autonomy_tier '{record.autonomy_tier}'. "
            f"Must be one of {sorted(VALID_TIERS)}."
        )
    if not record.agent_id or not record.action_type:
        raise ValueError("agent_id and action_type are required")

    prev_hash = _fetch_latest_hash(cursor)
    row_hash = compute_row_hash(record, prev_hash)
    payload_json = json.dumps(record.payload or {}, sort_keys=True, separators=(",", ":"))

    cursor.execute(
        """
        INSERT INTO ai_decision_ledger
            (agent_id, action_type, autonomy_tier, subject_kind, subject_id,
             payload, policy_id, prev_hash, row_hash, actor, outcome)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            record.agent_id,
            record.action_type,
            record.autonomy_tier,
            record.subject_kind,
            record.subject_id,
            payload_json,
            record.policy_id,
            prev_hash,
            row_hash,
            record.actor,
            record.outcome,
        ),
    )
    new_id = cursor.fetchone()[0]
    logger.info(
        "ai_decision_ledger append id=%s agent=%s action=%s tier=%s",
        new_id, record.agent_id, record.action_type, record.autonomy_tier,
    )
    return new_id, row_hash


def verify_chain(cursor: Any, *, limit: int | None = None) -> tuple[bool, list[dict[str, Any]]]:
    """Walk the ledger in ID order and verify every hash + linkage.

    Returns (ok, errors). When ok is True, errors is empty. Each error
    dict contains the offending id plus a reason string.
    """
    sql = (
        "SELECT id, agent_id, action_type, autonomy_tier, subject_kind, subject_id, "
        "       payload::text, policy_id, prev_hash, row_hash "
        "FROM ai_decision_ledger ORDER BY id"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    cursor.execute(sql)
    rows = cursor.fetchall()

    errors: list[dict[str, Any]] = []
    expected_prev = GENESIS_HASH
    for r in rows:
        (
            row_id, agent_id, action_type, tier,
            subject_kind, subject_id, payload_text, policy_id,
            prev_hash, row_hash,
        ) = r
        if prev_hash != expected_prev:
            errors.append({"id": row_id, "reason": f"prev_hash mismatch: expected {expected_prev}, got {prev_hash}"})
            # Continue walking so operators see every break, not just the first.
        record = DecisionRecord(
            agent_id=agent_id,
            action_type=action_type,
            autonomy_tier=tier,
            subject_kind=subject_kind,
            subject_id=subject_id,
            payload=json.loads(payload_text) if payload_text else {},
            policy_id=policy_id,
        )
        computed = compute_row_hash(record, prev_hash)
        if computed != row_hash:
            errors.append({"id": row_id, "reason": f"row_hash mismatch: expected {computed}, got {row_hash}"})
        expected_prev = row_hash

    return (len(errors) == 0, errors)


__all__ = [
    "GENESIS_HASH",
    "VALID_TIERS",
    "DecisionRecord",
    "append_decision",
    "compute_row_hash",
    "verify_chain",
]
