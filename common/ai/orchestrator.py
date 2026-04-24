"""Closed-loop exception orchestrator.

Gen-4 Roadmap AI-8 (Stream H Phase 2/3). Wires up the detect → simulate →
rank → route pipeline for replenishment exceptions:

    1. ``detect(conn)`` reads open rows from ``fact_replenishment_exceptions``.
    2. ``simulate_options(exception)`` builds candidate actions (expedite
       transfer, emergency PO, reallocate) and asks the digital twin to
       project end-of-horizon stock for each.
    3. ``rank(options)`` sorts by expected stock-at-horizon minus cost.
    4. ``route(option)`` calls the policy engine; if permitted and tier
       ≥ ``auto_within_policy`` it applies via the reversible ledger and
       records the decision; otherwise it enqueues for human review.

The class is deliberately thin — production behavior (cost models, real
transfer/PO SQL, KPI scoring) lands in a later pass. This file holds the
plumbing so the policy engine, twin, reversible ledger, and decision
ledger are all glued together exactly once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from common.ai.decision_ledger import DecisionRecord, append_decision
from common.ai.policy_engine import ActionContext, PolicyDecision, evaluate
from common.ai.reversible import ReversibleAction, apply as apply_reversible
from common.twin.state import TwinState

logger = logging.getLogger(__name__)

# Tiers at or above this threshold trigger auto-apply. Anything below
# goes to the human-review queue. Kept as a named constant so future
# tweaks don't bury a magic string in branching logic.
_AUTO_APPLY_TIERS: frozenset[str] = frozenset({"auto_within_policy", "autonomous"})

# Default MC horizon when projecting each option. Matches the TwinState
# default behavior (horizon = max lead-time draw) but we pin it here so
# options are compared on equal footing.
_SIMULATION_HORIZON_DAYS: int = 30
_SIMULATION_N_ITER: int = 2_000

# Default candidate action types the orchestrator builds per exception.
# Each one maps to a policy_id in config/agent_autonomy.yaml.
_CANDIDATE_ACTIONS: tuple[tuple[str, str], ...] = (
    ("expedite_transfer", "supply.auto_transfer"),
    ("emergency_po", "supply.auto_transfer"),       # TODO: dedicated policy when minted
    ("reallocate", "supply.auto_transfer"),         # TODO: dedicated policy when minted
)


@dataclass
class Exception_:
    """One open replenishment exception row.

    Named ``Exception_`` with a trailing underscore to avoid shadowing the
    Python builtin. The public alias ``Exception`` is re-exported below so
    callers can still write ``ExceptionOrchestrator.Exception``.
    """

    exception_id: str
    item_id: str
    loc: str
    severity: str
    exception_type: str
    current_qty_on_hand: float | None = None
    recommended_order_qty: float | None = None


@dataclass
class Option:
    """One candidate action evaluated against the twin."""

    action_type: str
    policy_id: str
    exception: Exception_
    projected_stock_at_horizon: float   # expected end-of-horizon on-hand
    cost: float                         # dollar / effort cost proxy
    score: float                        # expected_stock - cost (rank key)
    payload: dict[str, Any] = field(default_factory=dict)


# Public alias so outside callers may ``from common.ai.orchestrator import Exception``.
Exception = Exception_


class ExceptionOrchestrator:
    """Detect → simulate → rank → route loop for replenishment exceptions.

    Construction takes no args; all DB work flows through method arguments
    so tests can inject mocks.
    """

    def __init__(
        self,
        *,
        agent_id: str = "exception_agent",
        horizon_days: int = _SIMULATION_HORIZON_DAYS,
        n_iter: int = _SIMULATION_N_ITER,
    ) -> None:
        self.agent_id = agent_id
        self.horizon_days = horizon_days
        self.n_iter = n_iter

    # ------------------------------------------------------------------
    # Detect
    # ------------------------------------------------------------------

    def detect(self, conn: Any) -> list[Exception_]:
        """Return every open exception ordered by severity then date."""
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT exception_id, item_id, loc, severity, exception_type,
                       current_qty_on_hand, recommended_order_qty
                  FROM fact_replenishment_exceptions
                 WHERE status = 'open'
                 ORDER BY
                    CASE severity
                        WHEN 'critical' THEN 1
                        WHEN 'high'     THEN 2
                        WHEN 'medium'   THEN 3
                        WHEN 'low'      THEN 4
                        ELSE 5
                    END,
                    exception_date
                """
            )
            rows = cur.fetchall() or []

        exceptions: list[Exception_] = []
        for row in rows:
            exceptions.append(
                Exception_(
                    exception_id=row[0],
                    item_id=row[1],
                    loc=row[2],
                    severity=row[3],
                    exception_type=row[4],
                    current_qty_on_hand=float(row[5]) if row[5] is not None else None,
                    recommended_order_qty=float(row[6]) if row[6] is not None else None,
                )
            )
        logger.info("orchestrator.detect found %d open exceptions", len(exceptions))
        return exceptions

    # ------------------------------------------------------------------
    # Simulate options
    # ------------------------------------------------------------------

    def simulate_options(
        self,
        exception: Exception_,
        twin_state: TwinState,
    ) -> list[Option]:
        """Build ≥3 candidate actions and score each via the twin.

        ``twin_state`` is passed in (not re-loaded from DB) so tests can
        inject a mock and so production callers can amortize the load
        cost across the full candidate set.
        """
        order_qty = float(exception.recommended_order_qty or 0.0)
        options: list[Option] = []

        for action_type, policy_id in _CANDIDATE_ACTIONS:
            extra_stock, cost = _default_effect(action_type, order_qty)
            stock_draws = twin_state.simulate(
                scenario={
                    "extra_stock": extra_stock,
                    "horizon_days": self.horizon_days,
                },
                n_iter=self.n_iter,
            )
            projected = float(np.mean(stock_draws))
            score = projected - cost
            options.append(
                Option(
                    action_type=action_type,
                    policy_id=policy_id,
                    exception=exception,
                    projected_stock_at_horizon=projected,
                    cost=cost,
                    score=score,
                    payload={
                        "extra_stock": extra_stock,
                        "order_qty": order_qty,
                        "item_id": exception.item_id,
                        "loc": exception.loc,
                    },
                )
            )
        return options

    # ------------------------------------------------------------------
    # Rank
    # ------------------------------------------------------------------

    @staticmethod
    def rank(options: list[Option]) -> list[Option]:
        """Sort options by score descending (``expected_stock - cost``)."""
        return sorted(options, key=lambda o: o.score, reverse=True)

    # ------------------------------------------------------------------
    # Route
    # ------------------------------------------------------------------

    def route(self, conn: Any, option: Option, *, requested_tier: str = "auto_within_policy") -> dict[str, Any]:
        """Evaluate policy, act or enqueue, and log to the decision ledger.

        Returns a summary dict so callers / tests can assert the result
        without re-reading the DB:

            {
                "exception_id": ...,
                "action_type": ...,
                "policy_decision": PolicyDecision,
                "applied": bool,
                "reversible_action_id": int | None,
                "ledger_id": int,
            }
        """
        ctx = ActionContext(
            policy_id=option.policy_id,
            requested_tier=requested_tier,
            blast_radius_skus=1,
            units=option.payload.get("order_qty"),
        )
        decision: PolicyDecision = evaluate(ctx)

        applied = (
            decision.permitted
            and decision.effective_tier in _AUTO_APPLY_TIERS
        )

        reversible_id: int | None = None
        outcome = "applied" if applied else "queued_for_review"
        ledger_payload: dict[str, Any] = {
            "action_type": option.action_type,
            "projected_stock_at_horizon": option.projected_stock_at_horizon,
            "cost": option.cost,
            "score": option.score,
            "payload": option.payload,
            "policy_reasons": decision.reasons,
        }

        with conn.cursor() as cur:
            if applied:
                reversible_id = apply_reversible(
                    cur,
                    ReversibleAction(
                        action_type=option.action_type,
                        target_kind="exception_id",
                        target_id=option.exception.exception_id,
                        rollback_payload={
                            "reverse_of": option.action_type,
                            "order_qty": option.payload.get("order_qty"),
                            "item_id": option.exception.item_id,
                            "loc": option.exception.loc,
                        },
                        applied_by=self.agent_id,
                    ),
                )
                ledger_payload["reversible_action_id"] = reversible_id

            ledger_id, _row_hash = append_decision(
                cur,
                DecisionRecord(
                    agent_id=self.agent_id,
                    action_type=option.action_type,
                    autonomy_tier=decision.effective_tier,
                    subject_kind="exception_id",
                    subject_id=option.exception.exception_id,
                    payload=ledger_payload,
                    policy_id=option.policy_id,
                    actor=self.agent_id,
                    outcome=outcome,
                ),
            )

        logger.info(
            "orchestrator.route exception=%s action=%s applied=%s ledger_id=%s",
            option.exception.exception_id, option.action_type, applied, ledger_id,
        )
        return {
            "exception_id": option.exception.exception_id,
            "action_type": option.action_type,
            "policy_decision": decision,
            "applied": applied,
            "reversible_action_id": reversible_id,
            "ledger_id": ledger_id,
        }


def _default_effect(action_type: str, order_qty: float) -> tuple[float, float]:
    """Return ``(extra_stock, cost)`` for a candidate action.

    Cost is a rough proxy pending the real cost model. Units align so the
    ``score = projected_stock - cost`` comparison is internally consistent
    within a single orchestrator run, which is all rank() needs.
    """
    if action_type == "expedite_transfer":
        return order_qty * 0.8, order_qty * 0.10   # fast, modest premium
    if action_type == "emergency_po":
        return order_qty, order_qty * 0.25         # full coverage, higher premium
    if action_type == "reallocate":
        return order_qty * 0.5, order_qty * 0.05   # partial, cheapest
    return 0.0, 0.0


__all__ = [
    "ExceptionOrchestrator",
    "Exception",
    "Option",
]
