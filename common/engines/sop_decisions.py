"""Write-path helper for the fact_sop_decisions audit log (Gen-4 SC-3).

Every S&OP decision (promote / approve / publish / override / rollback) should
call ``log_sop_decision`` before or after its state mutation so we have a
queryable audit trail. See sql/147_create_fact_sop_decisions.sql.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


ALLOWED_DECISION_TYPES: frozenset[str] = frozenset({
    "promote", "approve", "publish", "override",
    "demote", "reject", "rollback", "advance",
})


def log_sop_decision(
    conn: Any,
    *,
    decision_type: str,
    decided_by: str,
    cycle_id: int | None = None,
    scenario_id: str | None = None,
    rationale: dict | None = None,
    prior_state: dict | None = None,
    new_state: dict | None = None,
    commit: bool = False,
) -> int | None:
    """Insert a row into fact_sop_decisions.

    Args:
        conn:           Live psycopg connection (caller owns transaction lifecycle
                        unless ``commit=True``).
        decision_type:  One of ALLOWED_DECISION_TYPES (raises ValueError otherwise).
        decided_by:     User id / email / service account.
        cycle_id:       Optional FK to fact_sop_cycles.cycle_id.
        scenario_id:    Optional scenario identifier.
        rationale:      Optional JSON-serializable dict with notes/gap_summary/etc.
        prior_state:    Optional before-snapshot dict.
        new_state:      Optional after-snapshot dict.
        commit:         When True, commits the insert; otherwise caller commits.

    Returns:
        The inserted row id, or None if the table isn't present (schema not applied yet).
    """
    if decision_type not in ALLOWED_DECISION_TYPES:
        raise ValueError(
            f"Unsupported decision_type {decision_type!r}; "
            f"expected one of {sorted(ALLOWED_DECISION_TYPES)}"
        )

    def _j(d: dict | None) -> str | None:
        return json.dumps(d) if d else None

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO fact_sop_decisions
                    (cycle_id, scenario_id, decision_type, decided_by,
                     rationale, prior_state, new_state)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
                RETURNING id
                """,
                (
                    cycle_id, scenario_id, decision_type, decided_by,
                    _j(rationale), _j(prior_state), _j(new_state),
                ),
            )
            row = cur.fetchone()
            new_id = int(row[0]) if row else None
        if commit:
            conn.commit()
        return new_id
    except Exception as exc:
        # Only catch broadly here because we want to NEVER break the upstream
        # decision flow if the audit log is unavailable. psycopg-specific
        # errors (table missing, constraint violation) all land here.
        logger.warning("log_sop_decision skipped: %s", exc)
        try:
            conn.rollback()
        except Exception:
            pass
        return None
