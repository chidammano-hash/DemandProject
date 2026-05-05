"""Policy-as-code guardrails for AI agent write actions.

Gen-4 roadmap AI-1 P0. Every action proposed by an agent is evaluated
against a named policy from `config/ai/agent_autonomy.yaml`. The policy
declares:
  - `tier`: the maximum autonomy tier allowed (advisory, suggestive,
    auto_within_policy, autonomous)
  - `guardrails`: declarative constraints the engine checks before the
    write is permitted

This module is the *only* gate between agents and mutation. Every writer
must call :func:`evaluate` first; denied actions must not reach the DB.
Permitted actions are logged to `ai_decision_ledger` via
:mod:`common.ai.decision_ledger`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from common.core.utils import load_config
from common.ai.decision_ledger import VALID_TIERS

logger = logging.getLogger(__name__)

# Tier ordering from least to most autonomous. Comparisons use the index.
_TIER_ORDER: list[str] = [
    "advisory",
    "suggestive",
    "auto_within_policy",
    "autonomous",
]


@dataclass
class ActionContext:
    """Runtime inputs the engine needs to evaluate a proposed action."""

    policy_id: str
    requested_tier: str                 # the tier the agent wants to act at
    blast_radius_skus: int = 0          # number of SKUs this action will touch
    pct_change: float | None = None     # relative change magnitude, if applicable
    units: float | None = None          # absolute units involved
    dollars: float | None = None        # absolute dollars involved
    has_human_review: bool = False      # operator confirmation captured
    open_critical_exceptions: int = 0
    last_action_age_hours: float | None = None  # for cooldown checks
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyDecision:
    """Outcome of a policy evaluation."""

    permitted: bool
    effective_tier: str                 # the tier to record on the ledger row
    policy_id: str
    reasons: list[str]                  # one string per guardrail fired

    @property
    def blocked_reasons(self) -> list[str]:
        return self.reasons


def _tier_rank(tier: str) -> int:
    try:
        return _TIER_ORDER.index(tier)
    except ValueError:
        return -1


def _load_policies() -> dict[str, Any]:
    try:
        return load_config("agent_autonomy")
    except FileNotFoundError:
        logger.warning("agent_autonomy.yaml not found — defaulting all actions to advisory")
        return {"default": {"tier": "advisory", "max_blast_radius_skus": 0}, "policies": {}}


def _lookup_policy(policies_cfg: dict[str, Any], policy_id: str) -> dict[str, Any]:
    """Return the policy block for *policy_id*, or the default."""
    nested = (policies_cfg.get("policies") or {}).get(policy_id)
    if nested:
        return nested
    return policies_cfg.get("default") or {"tier": "advisory"}


def evaluate(ctx: ActionContext) -> PolicyDecision:
    """Evaluate *ctx* against its policy. Returns a :class:`PolicyDecision`.

    Never raises for policy violations — returns ``permitted=False`` with
    reasons. Raises :class:`ValueError` only for invalid inputs.
    """
    if ctx.requested_tier not in VALID_TIERS:
        raise ValueError(f"Invalid requested_tier '{ctx.requested_tier}'")

    cfg = _load_policies()
    policy = _lookup_policy(cfg, ctx.policy_id)

    max_tier = policy.get("tier", "advisory")
    guardrails = policy.get("guardrails") or {}
    reasons: list[str] = []

    # Tier ceiling — requested tier must not exceed max allowed
    if _tier_rank(ctx.requested_tier) > _tier_rank(max_tier):
        reasons.append(
            f"requested_tier={ctx.requested_tier} exceeds policy max={max_tier}"
        )

    # Human review requirement
    if guardrails.get("requires_human_review") and not ctx.has_human_review:
        reasons.append("policy requires human review; none captured")

    # Blast radius (SKU count)
    max_blast = guardrails.get("max_blast_radius_skus")
    if max_blast is not None and ctx.blast_radius_skus > int(max_blast):
        reasons.append(
            f"blast_radius_skus={ctx.blast_radius_skus} exceeds max={max_blast}"
        )

    # Relative change ceiling
    max_pct = guardrails.get("max_pct_change_per_sku")
    if max_pct is not None and ctx.pct_change is not None and abs(ctx.pct_change) > float(max_pct):
        reasons.append(
            f"pct_change={ctx.pct_change:.2f} exceeds max={max_pct}"
        )

    # Absolute unit / dollar ceilings
    max_units = guardrails.get("max_units_per_action")
    if max_units is not None and ctx.units is not None and ctx.units > float(max_units):
        reasons.append(f"units={ctx.units} exceeds max={max_units}")

    max_dollars = guardrails.get("max_dollar_per_action")
    if max_dollars is not None and ctx.dollars is not None and ctx.dollars > float(max_dollars):
        reasons.append(f"dollars={ctx.dollars} exceeds max={max_dollars}")

    # Cooldown between successive actions
    cooldown = guardrails.get("cooldown_hours")
    if (
        cooldown is not None
        and ctx.last_action_age_hours is not None
        and ctx.last_action_age_hours < float(cooldown)
    ):
        reasons.append(
            f"cooldown not elapsed: {ctx.last_action_age_hours:.1f}h < {cooldown}h"
        )

    # Open critical exceptions must be resolved first
    if guardrails.get("requires_open_exceptions_resolved") and ctx.open_critical_exceptions > 0:
        reasons.append(
            f"{ctx.open_critical_exceptions} open critical exceptions must resolve first"
        )

    permitted = not reasons
    return PolicyDecision(
        permitted=permitted,
        effective_tier=ctx.requested_tier if permitted else "advisory",
        policy_id=ctx.policy_id,
        reasons=reasons,
    )


__all__ = ["ActionContext", "PolicyDecision", "evaluate"]
