"""Unit tests for the AI Champion forward adjuster per-DFU logic.

Covers scripts.forecasting.generate_ai_champion_forecast._adjust_one — the pure
compute path (no DB): a valid recommendation scales the champion forecast over
the apply horizon, and any LLM failure falls back to KEEP (= champion).
"""
from __future__ import annotations

import datetime

import pytest

from common.ai.llm_client import LLMClientError
from scripts.forecasting.generate_ai_champion_forecast import _adjust_one

_GUARDRAILS = {"max_abs_pct_change": 50, "min_confidence": 0.60, "require_evidence": True}
_PLAN_MONTH = datetime.date(2026, 4, 1)
_FORWARD = [
    ("2026-04", 100.0, datetime.date(2026, 4, 1), 1),
    ("2026-05", 100.0, datetime.date(2026, 5, 1), 2),
    ("2026-06", 100.0, datetime.date(2026, 6, 1), 3),
    ("2026-07", 100.0, datetime.date(2026, 7, 1), 4),
]


class _StubResp:
    def __init__(self, parsed):
        self.parsed = parsed


class _StubClient:
    """Minimal LLMClient stand-in: chat() returns a fixed parsed JSON, or raises."""
    def __init__(self, parsed=None, raises=None):
        self._parsed = parsed
        self._raises = raises

    def chat(self, messages, *, json_mode=True, max_tokens=1024):
        if self._raises:
            raise self._raises
        return _StubResp(self._parsed)


def test_scale_up_applies_over_horizon():
    """A SCALE_UP +10% with apply_horizon=2 scales the first 2 months only."""
    rec = {
        "recommendation_code": "SCALE_UP", "pct_change": 10.0,
        "apply_horizon_months": 2, "confidence": 0.8,
        "rationale": "recent uptrend vs baseline", "evidence_keys": ["trend_break"],
    }
    client = _StubClient(parsed=rec)
    rows, changed = _adjust_one(
        client, ("100", "L1"), _FORWARD, actuals=[("2026-03", 90.0)],
        meta=("c1", "A"), customers=None, plan_month=_PLAN_MONTH,
        guardrails=_GUARDRAILS, horizon_months=3,
    )
    assert changed is True
    assert len(rows) == 4
    assert [r["ai_qty"] for r in rows] == pytest.approx([110.0, 110.0, 100.0, 100.0])
    assert [r["champion_qty"] for r in rows] == [100.0, 100.0, 100.0, 100.0]
    assert all(r["recommendation_code"] == "SCALE_UP" for r in rows)


def test_low_confidence_downgrades_to_keep():
    """Confidence below the guardrail floor → OVERRIDE_TO_BASELINE (= champion)."""
    rec = {
        "recommendation_code": "SCALE_DOWN", "pct_change": 30.0,
        "apply_horizon_months": 3, "confidence": 0.3,
        "rationale": "weak signal", "evidence_keys": ["maybe"],
    }
    rows, changed = _adjust_one(
        _StubClient(parsed=rec), ("100", "L1"), _FORWARD, actuals=[],
        meta=(None, None), customers=None, plan_month=_PLAN_MONTH,
        guardrails=_GUARDRAILS, horizon_months=3,
    )
    assert changed is False
    assert [r["ai_qty"] for r in rows] == [100.0, 100.0, 100.0, 100.0]
    assert all(r["recommendation_code"] == "OVERRIDE_TO_BASELINE" for r in rows)


def test_llm_failure_falls_back_to_champion():
    """Any LLM/parse error keeps the champion baseline (changed=False, KEEP)."""
    rows, changed = _adjust_one(
        _StubClient(raises=LLMClientError("boom")), ("100", "L1"), _FORWARD,
        actuals=[], meta=(None, None), customers=None, plan_month=_PLAN_MONTH,
        guardrails=_GUARDRAILS, horizon_months=3,
    )
    assert changed is False
    assert [r["ai_qty"] for r in rows] == [100.0, 100.0, 100.0, 100.0]
    assert all(r["recommendation_code"] == "KEEP" for r in rows)
    assert rows[0]["confidence"] is None
