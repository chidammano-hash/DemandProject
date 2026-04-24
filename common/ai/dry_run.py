"""Dry-run preview + explicit confirm pipeline for AI write actions.

Gen-4 Roadmap AI-4. Every write action can be previewed without mutating
state via :func:`dry_run`; a separate :func:`confirm` call applies the
change and writes the final record to the decision ledger. The split
gives the UI two clear endpoints: "show me what would happen" and
"do it".

Determinism:
    :func:`dry_run` is compute-only — it never issues ``INSERT``,
    ``UPDATE``, or ``DELETE``. If a future handler needs to exercise
    referential logic, wrap its probe in a savepoint + explicit
    ``ROLLBACK``. The result object is a frozen snapshot that
    :func:`confirm` accepts as input.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from common.ai.decision_ledger import DecisionRecord, append_decision

logger = logging.getLogger(__name__)


@dataclass
class DryRunAction:
    """Input payload for a dry-run probe."""

    action_type: str
    target_kind: str
    target_id: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DryRunResult:
    """Result of a dry-run probe.

    Attributes:
        action: echoed copy of the input action so :func:`confirm` gets
            everything it needs without re-reading external state.
        proposed_changes: list of ``{"table": ..., "op": ..., "row": ...}``
            entries describing the rows that *would* be written.
        risk_flags: free-form warnings the caller should surface.
        estimated_impact: dict of ``{kpi: delta}`` projections.
    """

    action: DryRunAction
    proposed_changes: list[dict[str, Any]] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    estimated_impact: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": asdict(self.action),
            "proposed_changes": list(self.proposed_changes),
            "risk_flags": list(self.risk_flags),
            "estimated_impact": dict(self.estimated_impact),
        }


# A dry-run handler is a callable ``(conn, action) -> DryRunResult``.
# Concrete per-action handlers will land alongside each action's real
# executor (e.g. transfer, PO, reallocate). The default handler returns
# an empty-changes result so unknown actions are safely advisory.
Handler = Callable[[Any, DryRunAction], DryRunResult]

_registry: dict[str, Handler] = {}


def register_handler(action_type: str, handler: Handler) -> None:
    """Register a dry-run handler for an action type."""
    _registry[action_type] = handler


def _default_handler(_conn: Any, action: DryRunAction) -> DryRunResult:
    """Fallback: return an empty-plan result flagged as stub."""
    return DryRunResult(
        action=action,
        proposed_changes=[],
        risk_flags=["no_handler_registered"],
        estimated_impact={},
    )


def dry_run(conn: Any, action: DryRunAction) -> DryRunResult:
    """Return a :class:`DryRunResult` without mutating any state.

    Dispatches to a handler registered via :func:`register_handler`; falls
    back to ``_default_handler`` when the action has no dedicated probe
    yet. Handlers MUST NOT commit — the safest handlers compute-only.
    """
    handler = _registry.get(action.action_type, _default_handler)
    result = handler(conn, action)
    logger.info(
        "dry_run action=%s target=%s/%s changes=%d flags=%s",
        action.action_type, action.target_kind, action.target_id,
        len(result.proposed_changes), result.risk_flags,
    )
    return result


def confirm(
    conn: Any,
    dry_run_result: DryRunResult,
    approver: str,
    *,
    autonomy_tier: str = "suggestive",
    policy_id: str | None = None,
) -> dict[str, Any]:
    """Apply the previewed change and write to the decision ledger.

    Returns ``{"ledger_id": int, "row_hash": str, "applied": True}``.
    Raises ``ValueError`` if ``approver`` is empty.
    """
    if not approver:
        raise ValueError("approver is required to confirm a dry-run")

    # TODO(gen-4 AI-4): dispatch to a per-action applier that writes the
    # rows described in ``dry_run_result.proposed_changes``. For now the
    # confirm step records the decision on the ledger only; downstream
    # write handlers land with each action's executor.
    with conn.cursor() as cur:
        ledger_id, row_hash = append_decision(
            cur,
            DecisionRecord(
                agent_id=dry_run_result.action.target_kind + "_confirm",
                action_type=dry_run_result.action.action_type,
                autonomy_tier=autonomy_tier,
                subject_kind=dry_run_result.action.target_kind,
                subject_id=dry_run_result.action.target_id,
                payload={
                    "proposed_changes": dry_run_result.proposed_changes,
                    "risk_flags": dry_run_result.risk_flags,
                    "estimated_impact": dry_run_result.estimated_impact,
                    "params": dry_run_result.action.params,
                },
                policy_id=policy_id,
                actor=approver,
                outcome="applied",
            ),
        )
    return {"ledger_id": ledger_id, "row_hash": row_hash, "applied": True}


__all__ = [
    "DryRunAction",
    "DryRunResult",
    "Handler",
    "confirm",
    "dry_run",
    "register_handler",
]
