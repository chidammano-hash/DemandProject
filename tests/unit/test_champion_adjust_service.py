"""Unit tests for the interactive AI Champion adjust service (pure helpers)."""
from datetime import date

import pytest

from common.ai.champion_adjust_service import (
    UnknownProvider,
    _months_from_rec,
    _resolve_provider_model,
)
from common.ai.champion_adjuster import DfuContext, Recommendation, build_user_prompt


def _forward():
    return [
        ("2026-05", 100.0, date(2026, 5, 1), 1),
        ("2026-06", 200.0, date(2026, 6, 1), 2),
        ("2026-07", 300.0, date(2026, 7, 1), 3),
        ("2026-08", 400.0, date(2026, 8, 1), 4),  # beyond horizon=3
    ]


def test_months_from_rec_scales_within_horizon_only():
    rec = Recommendation(recommendation_code="SCALE_UP", pct_change=10.0,
                         apply_horizon_months=3, confidence=0.9, rationale="trend up clearly")
    months = _months_from_rec(rec, _forward())
    # First 3 months scaled by +10%, 4th untouched.
    assert months[0].ai_qty == pytest.approx(110.0)
    assert months[2].ai_qty == pytest.approx(330.0)
    assert months[3].ai_qty == pytest.approx(400.0)
    # Derived per-month pct_change reflects the actual change.
    assert months[0].pct_change == pytest.approx(10.0)
    assert months[3].pct_change == pytest.approx(0.0)


def test_months_from_rec_keep_is_identity():
    rec = Recommendation(recommendation_code="KEEP", confidence=0.5, rationale="baseline is fine")
    months = _months_from_rec(rec, _forward())
    assert [m.ai_qty for m in months] == [100.0, 200.0, 300.0, 400.0]


def test_resolve_provider_model():
    cfg = {"provider": "ollama", "models": {"ollama": "llama3.1:8b", "google": "gemini-2.0-flash"}}
    assert _resolve_provider_model(cfg, None) == ("ollama", "llama3.1:8b")
    assert _resolve_provider_model(cfg, "google") == ("google", "gemini-2.0-flash")
    with pytest.raises(UnknownProvider):
        _resolve_provider_model(cfg, "mystery")


def test_prompt_includes_item_and_location_attributes():
    ctx = DfuContext(
        item_id="100", loc="L1", forecast_run_month=date(2026, 4, 1),
        actuals_last_24m=[("2026-03", 90.0)], baseline_forecast=[("2026-05", 100.0)],
        item_attrs={"brand": "Acme", "category": "Spirits"},
        location_attrs={"site": "Dallas DC", "state": "TX"},
    )
    prompt = build_user_prompt(ctx)
    assert "Item attributes: brand=Acme, category=Spirits" in prompt
    assert "Location attributes: site=Dallas DC, state=TX" in prompt
