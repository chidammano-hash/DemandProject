"""Tests for the agent autonomy policy engine."""
from __future__ import annotations

import pytest

from common.ai.policy_engine import ActionContext, evaluate


def _ctx(**overrides) -> ActionContext:
    base = {
        "policy_id": "supply.adjust_safety_stock",
        "requested_tier": "suggestive",
        "blast_radius_skus": 10,
        "pct_change": 10.0,
    }
    base.update(overrides)
    return ActionContext(**base)


def test_permits_action_within_policy():
    decision = evaluate(_ctx())
    assert decision.permitted is True
    assert decision.effective_tier == "suggestive"
    assert decision.reasons == []


def test_blocks_when_requested_tier_exceeds_policy_max():
    # supply.adjust_safety_stock is capped at 'suggestive'
    decision = evaluate(_ctx(requested_tier="autonomous"))
    assert decision.permitted is False
    assert any("exceeds policy max" in r for r in decision.reasons)
    # Effective tier falls back to advisory when blocked
    assert decision.effective_tier == "advisory"


def test_blocks_when_blast_radius_exceeds_max():
    decision = evaluate(_ctx(blast_radius_skus=100_000))
    assert decision.permitted is False
    assert any("blast_radius_skus" in r for r in decision.reasons)


def test_blocks_when_pct_change_exceeds_max():
    # max_pct_change_per_sku = 25 in config
    decision = evaluate(_ctx(pct_change=40.0))
    assert decision.permitted is False
    assert any("pct_change" in r for r in decision.reasons)


def test_accepts_negative_pct_change_within_max():
    decision = evaluate(_ctx(pct_change=-20.0))
    assert decision.permitted is True


def test_blocks_when_human_review_required_and_missing():
    decision = evaluate(
        _ctx(
            policy_id="supply.auto_transfer",
            requested_tier="advisory",
            has_human_review=False,
            units=100,
            dollars=500,
        )
    )
    assert decision.permitted is False
    assert any("human review" in r for r in decision.reasons)


def test_permits_when_human_review_supplied():
    decision = evaluate(
        _ctx(
            policy_id="supply.auto_transfer",
            requested_tier="advisory",
            has_human_review=True,
            units=100,
            dollars=500,
        )
    )
    assert decision.permitted is True


def test_blocks_when_units_exceed_cap():
    decision = evaluate(
        _ctx(
            policy_id="supply.auto_transfer",
            requested_tier="advisory",
            has_human_review=True,
            units=100_000,
        )
    )
    assert decision.permitted is False
    assert any("units=" in r for r in decision.reasons)


def test_blocks_on_cooldown():
    decision = evaluate(
        _ctx(
            policy_id="demand.retrain_cold_start",
            requested_tier="auto_within_policy",
            last_action_age_hours=1.0,
        )
    )
    assert decision.permitted is False
    assert any("cooldown" in r for r in decision.reasons)


def test_blocks_when_open_exceptions_block_publish():
    decision = evaluate(
        _ctx(
            policy_id="sop.publish_scenario",
            requested_tier="advisory",
            has_human_review=True,
            open_critical_exceptions=3,
        )
    )
    assert decision.permitted is False
    assert any("open critical exceptions" in r for r in decision.reasons)


def test_default_policy_is_advisory_only():
    # Unknown policy falls back to default (tier=advisory, blast=0)
    decision = evaluate(_ctx(policy_id="nonexistent.action", requested_tier="advisory"))
    assert decision.permitted is False
    assert any("blast_radius_skus" in r for r in decision.reasons)


def test_rejects_invalid_tier_input():
    with pytest.raises(ValueError):
        evaluate(_ctx(requested_tier="god_mode"))
