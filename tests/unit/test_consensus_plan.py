"""
F2.3 — Unit tests for generate_consensus_plan.py

Tests apply_override() and resolve_conflicts() pure functions.
"""

import sys
import os
import importlib
from datetime import datetime, date

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------

import scripts.generate_consensus_plan as cp


# ---------------------------------------------------------------------------
# apply_override
# ---------------------------------------------------------------------------

class TestApplyOverride:
    def test_multiplier_promo(self):
        """PROMO with multiplier 1.25: consensus = base * 1.25"""
        qty, delta = cp.apply_override(
            statistical_qty=400.0,
            override_type="PROMO",
            override_qty=None,
            override_multiplier=1.25,
            override_additive_qty=0.0,
            is_hard_override=False,
        )
        assert qty == pytest.approx(500.0)
        assert delta == pytest.approx(100.0)

    def test_additive_lift(self):
        """Additive +50 on top of base."""
        qty, delta = cp.apply_override(
            statistical_qty=200.0,
            override_type="MANUAL",
            override_qty=None,
            override_multiplier=1.0,
            override_additive_qty=50.0,
            is_hard_override=False,
        )
        assert qty == pytest.approx(250.0)
        assert delta == pytest.approx(50.0)

    def test_multiplier_and_additive(self):
        """Multiplier + additive applied together."""
        qty, delta = cp.apply_override(
            statistical_qty=200.0,
            override_type="PROMO",
            override_qty=None,
            override_multiplier=1.1,
            override_additive_qty=30.0,
            is_hard_override=False,
        )
        # 200 * 1.1 + 30 = 250
        assert qty == pytest.approx(250.0)
        assert delta == pytest.approx(50.0)

    def test_hard_override_replaces_base(self):
        """Hard override: override_qty fully replaces statistical."""
        qty, delta = cp.apply_override(
            statistical_qty=400.0,
            override_type="CAPACITY_LOCK",
            override_qty=300.0,
            override_multiplier=None,
            override_additive_qty=0.0,
            is_hard_override=True,
        )
        assert qty == pytest.approx(300.0)
        assert delta == pytest.approx(-100.0)

    def test_hard_override_floors_at_zero(self):
        """Hard override of 0 should produce 0 consensus qty."""
        qty, delta = cp.apply_override(
            statistical_qty=200.0,
            override_type="PHASE_OUT",
            override_qty=0.0,
            override_multiplier=None,
            override_additive_qty=0.0,
            is_hard_override=True,
        )
        assert qty == pytest.approx(0.0)
        assert delta == pytest.approx(-200.0)

    def test_negative_result_clamped_to_zero(self):
        """Multiplier so small that result would be negative — clamp to 0."""
        qty, delta = cp.apply_override(
            statistical_qty=100.0,
            override_type="PHASE_OUT",
            override_qty=None,
            override_multiplier=0.0,
            override_additive_qty=-50.0,
            is_hard_override=False,
        )
        # 100 * 0 + (-50) = -50 → clamped to 0
        assert qty == pytest.approx(0.0)
        assert delta == pytest.approx(-100.0)

    def test_no_override_params_uses_base(self):
        """No multiplier, no additive, not hard: consensus == statistical."""
        qty, delta = cp.apply_override(
            statistical_qty=350.0,
            override_type="MANUAL",
            override_qty=None,
            override_multiplier=None,
            override_additive_qty=0.0,
            is_hard_override=False,
        )
        assert qty == pytest.approx(350.0)
        assert delta == pytest.approx(0.0)

    def test_phase_out_multiplier_reduction(self):
        """PHASE_OUT with multiplier 0.5 halves the forecast."""
        qty, delta = cp.apply_override(
            statistical_qty=500.0,
            override_type="PHASE_OUT",
            override_qty=None,
            override_multiplier=0.5,
            override_additive_qty=0.0,
            is_hard_override=False,
        )
        assert qty == pytest.approx(250.0)
        assert delta == pytest.approx(-250.0)


# ---------------------------------------------------------------------------
# resolve_conflicts
# ---------------------------------------------------------------------------

def _make_override(
    override_id: int,
    override_type: str,
    priority_rank: int = 5,
    created_at=None,
) -> dict:
    return {
        "override_id": override_id,
        "item_no": "100320",
        "loc": "1401-BULK",
        "override_month": date(2026, 5, 1),
        "override_type": override_type,
        "priority_rank": priority_rank,
        "created_at": created_at or datetime(2026, 3, 1, 8, 0, 0),
    }


class TestResolveConflicts:
    def test_single_override_returns_itself(self):
        o = _make_override(1, "PROMO")
        result = cp.resolve_conflicts([o])
        assert result["override_id"] == 1

    def test_capacity_lock_beats_promo(self):
        promo = _make_override(1, "PROMO", priority_rank=1)
        cap_lock = _make_override(2, "CAPACITY_LOCK", priority_rank=5)
        result = cp.resolve_conflicts([promo, cap_lock])
        assert result["override_id"] == 2  # CAPACITY_LOCK wins

    def test_promo_beats_manual_by_type_priority(self):
        manual = _make_override(1, "MANUAL", priority_rank=1)
        promo = _make_override(2, "PROMO", priority_rank=3)
        result = cp.resolve_conflicts([manual, promo])
        assert result["override_id"] == 2  # PROMO has lower _type_priority

    def test_same_type_lower_priority_rank_wins(self):
        o1 = _make_override(1, "PROMO", priority_rank=3)
        o2 = _make_override(2, "PROMO", priority_rank=1)
        result = cp.resolve_conflicts([o1, o2])
        assert result["override_id"] == 2  # lower rank = higher priority

    def test_same_type_same_rank_latest_wins(self):
        older = _make_override(1, "PROMO", priority_rank=1, created_at=datetime(2026, 2, 1))
        newer = _make_override(2, "PROMO", priority_rank=1, created_at=datetime(2026, 3, 1))
        result = cp.resolve_conflicts([older, newer])
        assert result["override_id"] == 2  # more recent wins

    def test_three_way_conflict_capacity_lock_wins(self):
        manual = _make_override(1, "MANUAL", priority_rank=1)
        promo = _make_override(2, "PROMO", priority_rank=1)
        cap = _make_override(3, "CAPACITY_LOCK", priority_rank=5)
        result = cp.resolve_conflicts([manual, promo, cap])
        assert result["override_id"] == 3

    def test_launch_equals_promo_priority_rank_breaks_tie(self):
        """LAUNCH and PROMO have same type priority; lower rank wins."""
        launch = _make_override(1, "LAUNCH", priority_rank=2)
        promo = _make_override(2, "PROMO", priority_rank=1)
        result = cp.resolve_conflicts([launch, promo])
        assert result["override_id"] == 2  # PROMO rank=1 < LAUNCH rank=2
