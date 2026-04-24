"""Shadow / A-B champion-challenger rollout (scaffold).

Gen-4 Stream G / AI-7.

The production scorer consults an active ``ShadowRollout`` to decide
whether to tee a prediction to the challenger model. A follow-up
analysis compares logged challenger predictions to champion accuracy
over the rollout window.

TODO(gen-4):
  - Wire ``scripts/forecast_generate.py`` to query ``fact_shadow_rollout``
    for active rows and tee predictions via :meth:`ShadowRollout.should_tee`.
  - Compute and persist ``observed_metrics`` in an end-of-window job.
  - Expose ``/forecast/shadow/{challenger_id}/metrics`` endpoint.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from common.ai.decision_ledger import DecisionRecord, append_decision

logger = logging.getLogger(__name__)

VALID_STATUS: frozenset[str] = frozenset({"proposed", "active", "completed", "aborted"})


@dataclass
class ShadowRollout:
    """One shadow / A-B rollout plan.

    Attributes:
        champion_id: currently promoted production model id.
        challenger_id: candidate model being shadowed.
        start_ts: UTC rollout start.
        traffic_pct: fraction (0..1) of traffic routed to challenger.
                     0.0 means shadow-only (log, do not serve).
        end_ts: UTC rollout end (None while active).
        status: one of VALID_STATUS.
        observed_metrics: filled on completion.
        notes: rationale / decision log.
        rollout_id: stable UUID set on insert.
    """

    champion_id: str
    challenger_id: str
    traffic_pct: float = 0.0
    start_ts: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_ts: datetime | None = None
    status: str = "proposed"
    observed_metrics: dict[str, Any] | None = None
    notes: str | None = None
    rollout_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.traffic_pct) <= 1.0):
            raise ValueError(
                f"traffic_pct must be in [0, 1]; got {self.traffic_pct}"
            )
        if self.status not in VALID_STATUS:
            raise ValueError(
                f"Invalid status {self.status!r}; expected one of {sorted(VALID_STATUS)}"
            )
        if self.champion_id == self.challenger_id:
            raise ValueError("champion_id and challenger_id must differ")

    # ------------------------------------------------------------------
    # Traffic routing
    # ------------------------------------------------------------------

    def should_tee(self, *, rng: np.random.Generator | None = None) -> bool:
        """Return True when this prediction should be routed to challenger.

        Shadow-only rollouts (``traffic_pct == 0.0``) always return False
        from a *routing* perspective — challenger predictions are still
        computed separately for logging.
        """
        if self.status != "active":
            return False
        if self.traffic_pct <= 0.0:
            return False
        rng = rng or np.random.default_rng()
        return bool(rng.uniform(0.0, 1.0) < self.traffic_pct)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def insert(self, cursor: Any) -> int:
        """Insert this rollout into ``fact_shadow_rollout`` and return id.

        Also appends an entry to the AI decision ledger so the rollout
        decision is auditable.
        """
        cursor.execute(
            """
            INSERT INTO fact_shadow_rollout
                (champion_id, challenger_id, traffic_pct, start_ts, end_ts,
                 status, observed_metrics, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
            RETURNING id
            """,
            (
                self.champion_id,
                self.challenger_id,
                float(self.traffic_pct),
                self.start_ts,
                self.end_ts,
                self.status,
                _json_or_none(self.observed_metrics),
                self.notes,
            ),
        )
        new_id = cursor.fetchone()[0]
        try:
            append_decision(
                cursor,
                DecisionRecord(
                    agent_id="model_registry",
                    action_type="shadow_rollout",
                    autonomy_tier="advisory",
                    subject_kind="model_id",
                    subject_id=self.challenger_id,
                    payload={
                        "champion_id": self.champion_id,
                        "challenger_id": self.challenger_id,
                        "traffic_pct": float(self.traffic_pct),
                        "status": self.status,
                    },
                    policy_id="shadow_rollout",
                    actor="api",
                    outcome="recorded",
                ),
            )
        except (ValueError, KeyError) as exc:
            # Ledger writes are best-effort for rollouts; log & move on.
            logger.warning("shadow_rollout ledger append failed: %s", exc)
        return new_id


def _json_or_none(value: dict[str, Any] | None) -> str | None:
    import json
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


__all__ = ["ShadowRollout", "VALID_STATUS"]
