"""Unit tests for common.ai.champion_adjuster — pure-function logic.

Spec: docs/specs/02-forecasting/27-ai-champion-forecast.md

Covers apply_guardrails and apply_recommendation. The LLM call path
(common.ai.llm_client) is exercised separately with live integration tests
when Ollama is available.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from common.ai.champion_adjuster import (
    Recommendation,
    apply_guardrails,
    apply_recommendation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASELINE = [
    ("2026-05", 100.0),
    ("2026-06", 110.0),
    ("2026-07", 120.0),
]


def make_rec(**overrides) -> Recommendation:
    base = {
        "recommendation_code": "KEEP",
        "confidence": 0.8,
        "rationale": "trend looks stable, no anomalies in recent six months",
        "evidence_keys": ["trend_stable"],
        "apply_horizon_months": 3,
    }
    base.update(overrides)
    return Recommendation(**base)


GUARDRAILS = {
    "max_abs_pct_change": 50,
    "min_confidence": 0.60,
    "require_evidence": True,
    "reject_on_horizon_overflow": True,
}


# ---------------------------------------------------------------------------
# apply_recommendation — pure forecast math
# ---------------------------------------------------------------------------

class TestApplyRecommendation:
    def test_keep_returns_baseline_unchanged(self):
        rec = make_rec(recommendation_code="KEEP")
        out = apply_recommendation(rec, BASELINE)
        assert out == BASELINE

    def test_override_to_baseline_returns_baseline(self):
        rec = make_rec(recommendation_code="OVERRIDE_TO_BASELINE")
        out = apply_recommendation(rec, BASELINE)
        assert out == BASELINE

    def test_scale_up_applies_pct_within_horizon(self):
        rec = make_rec(recommendation_code="SCALE_UP", pct_change=20.0,
                       apply_horizon_months=2)
        out = apply_recommendation(rec, BASELINE)
        # First 2 months get +20%, third unchanged
        assert out[0] == ("2026-05", 120.0)
        assert out[1] == ("2026-06", 132.0)
        assert out[2] == ("2026-07", 120.0)

    def test_scale_down_at_pct_floor_zeroes_qty(self):
        # Pydantic schema floors pct_change at -100 (full zero-out).
        rec = make_rec(recommendation_code="SCALE_DOWN", pct_change=-100.0,
                       confidence=0.9, evidence_keys=["dropoff"])
        out = apply_recommendation(rec, BASELINE)
        assert out[0][1] == 0.0
        assert out[1][1] == 0.0
        assert out[2][1] == 0.0

    def test_scale_up_full_horizon(self):
        rec = make_rec(recommendation_code="SCALE_UP", pct_change=10.0,
                       apply_horizon_months=3)
        out = apply_recommendation(rec, BASELINE)
        assert [round(q, 2) for _, q in out] == [110.0, 121.0, 132.0]

    def test_replace_uses_proposed_qty(self):
        rec = make_rec(
            recommendation_code="REPLACE",
            proposed_qty=[200.0, 200.0, 200.0],
            apply_horizon_months=3,
        )
        out = apply_recommendation(rec, BASELINE)
        assert [q for _, q in out] == [200.0, 200.0, 200.0]

    def test_replace_partial_horizon_keeps_baseline_for_remainder(self):
        rec = make_rec(
            recommendation_code="REPLACE",
            proposed_qty=[200.0],
            apply_horizon_months=1,
        )
        out = apply_recommendation(rec, BASELINE)
        assert out[0][1] == 200.0
        assert out[1] == BASELINE[1]
        assert out[2] == BASELINE[2]


# ---------------------------------------------------------------------------
# apply_guardrails — clip / downgrade logic
# ---------------------------------------------------------------------------

class TestApplyGuardrails:
    def test_low_confidence_downgraded_to_override_to_baseline(self):
        rec = make_rec(recommendation_code="SCALE_UP", pct_change=20.0,
                       confidence=0.3)
        out = apply_guardrails(rec, GUARDRAILS, horizon_months=3)
        assert out.recommendation_code == "OVERRIDE_TO_BASELINE"
        assert out.pct_change is None
        assert "downgraded" in out.rationale.lower()

    def test_missing_evidence_downgrades_non_keep(self):
        rec = make_rec(recommendation_code="SCALE_UP", pct_change=20.0,
                       evidence_keys=[])
        out = apply_guardrails(rec, GUARDRAILS, horizon_months=3)
        assert out.recommendation_code == "OVERRIDE_TO_BASELINE"

    def test_missing_evidence_does_not_downgrade_keep(self):
        rec = make_rec(recommendation_code="KEEP", evidence_keys=[])
        out = apply_guardrails(rec, GUARDRAILS, horizon_months=3)
        assert out.recommendation_code == "KEEP"

    def test_pct_change_above_max_clipped(self):
        rec = make_rec(recommendation_code="SCALE_UP", pct_change=80.0)
        out = apply_guardrails(rec, GUARDRAILS, horizon_months=3)
        assert out.pct_change == 50.0
        assert "clipped" in out.rationale.lower()

    def test_pct_change_below_min_clipped_negative(self):
        rec = make_rec(recommendation_code="SCALE_DOWN", pct_change=-80.0)
        out = apply_guardrails(rec, GUARDRAILS, horizon_months=3)
        assert out.pct_change == -50.0

    def test_pct_change_within_bounds_preserved(self):
        rec = make_rec(recommendation_code="SCALE_UP", pct_change=25.0)
        out = apply_guardrails(rec, GUARDRAILS, horizon_months=3)
        assert out.pct_change == 25.0
        assert "clipped" not in out.rationale.lower()

    def test_horizon_overflow_clipped(self):
        rec = make_rec(recommendation_code="SCALE_UP", pct_change=20.0,
                       apply_horizon_months=6)
        out = apply_guardrails(rec, GUARDRAILS, horizon_months=3)
        assert out.apply_horizon_months == 3


# ---------------------------------------------------------------------------
# Recommendation Pydantic validation
# ---------------------------------------------------------------------------

class TestRecommendationValidation:
    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            Recommendation(
                recommendation_code="KEEP",
                confidence=1.5,
                rationale="x" * 30,
                evidence_keys=[],
            )

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            Recommendation(
                recommendation_code="KEEP",
                confidence=0.8,
                rationale="x" * 30,
                evidence_keys=[],
                bogus_field="oops",
            )

    def test_short_rationale_rejected(self):
        with pytest.raises(ValidationError):
            Recommendation(
                recommendation_code="KEEP",
                confidence=0.8,
                rationale="too short",
                evidence_keys=[],
            )
