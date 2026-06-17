"""Record consensus forecast override approvals in the AI decision ledger."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import psycopg

from common.ai.decision_ledger import DecisionRecord, append_decision

logger = logging.getLogger(__name__)


def record_override_approval(
    cursor: Any,
    *,
    override_id: int,
    item_id: str,
    loc: str,
    override_month: date,
    override_type: str,
    actor: str,
    source: str,
) -> None:
    """Append a ledger row when a planner override becomes approved.

    Best-effort: logs and swallows ledger errors so override workflow is not blocked.
    """
    try:
        append_decision(
            cursor,
            DecisionRecord(
                agent_id="consensus_planner",
                action_type="forecast_override_approved",
                autonomy_tier="suggestive",
                subject_kind="dfu",
                subject_id=f"{item_id}-{loc}",
                payload={
                    "override_id": override_id,
                    "override_month": override_month.isoformat(),
                    "override_type": override_type,
                    "source": source,
                },
                policy_id="consensus_override",
                actor=actor,
                outcome="applied",
            ),
        )
    except (psycopg.Error, ValueError):
        logger.exception(
            "Failed to append override %s to decision ledger (source=%s)",
            override_id,
            source,
        )
