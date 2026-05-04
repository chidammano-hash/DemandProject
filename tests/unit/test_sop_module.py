"""Unit tests for scripts/run_sop_cycle.py — F4.2."""

import pytest
from datetime import date
from scripts.ops.run_sop_cycle import (
    next_stage,
    is_terminal_stage,
    compute_cycle_dates,
    STAGE_ORDER,
)


class TestNextStage:
    def test_demand_review_to_supply_review(self):
        assert next_stage("demand_review") == "supply_review"

    def test_supply_review_to_pre_sop(self):
        assert next_stage("supply_review") == "pre_sop"

    def test_pre_sop_to_executive_sop(self):
        assert next_stage("pre_sop") == "executive_sop"

    def test_executive_sop_to_approved(self):
        assert next_stage("executive_sop") == "approved"

    def test_approved_to_closed(self):
        assert next_stage("approved") == "closed"

    def test_closed_raises(self):
        with pytest.raises(ValueError):
            next_stage("closed")

    def test_full_stage_progression(self):
        stage = "demand_review"
        visited = [stage]
        while stage != "approved":
            stage = next_stage(stage)
            visited.append(stage)
        assert visited == STAGE_ORDER[:-1]  # all except 'closed'


class TestIsTerminalStage:
    def test_closed_is_terminal(self):
        assert is_terminal_stage("closed") is True

    def test_approved_is_not_terminal(self):
        assert is_terminal_stage("approved") is False

    def test_demand_review_is_not_terminal(self):
        assert is_terminal_stage("demand_review") is False


class TestComputeCycleDates:
    def test_dates_in_prior_month(self):
        cycle_month = date(2026, 5, 1)
        cfg = {
            "demand_review_day": 5,
            "supply_review_day": 10,
            "pre_sop_day": 15,
            "executive_sop_day": 20,
        }
        dates = compute_cycle_dates(cycle_month, cfg)
        # All dates should be in April 2026 (prior month)
        assert dates["demand_review_date"] == date(2026, 4, 5)
        assert dates["supply_review_date"] == date(2026, 4, 10)
        assert dates["pre_sop_date"] == date(2026, 4, 15)
        assert dates["executive_sop_date"] == date(2026, 4, 20)

    def test_default_cfg(self):
        cycle_month = date(2026, 6, 1)
        dates = compute_cycle_dates(cycle_month, {})
        # With empty config, days default to 5, 10, 15, 20
        assert dates["demand_review_date"].month == 5
        assert dates["executive_sop_date"].month == 5

    def test_stage_order_within_month(self):
        cycle_month = date(2026, 5, 1)
        cfg = {
            "demand_review_day": 5,
            "supply_review_day": 10,
            "pre_sop_day": 15,
            "executive_sop_day": 20,
        }
        dates = compute_cycle_dates(cycle_month, cfg)
        assert dates["demand_review_date"] < dates["supply_review_date"]
        assert dates["supply_review_date"] < dates["pre_sop_date"]
        assert dates["pre_sop_date"] < dates["executive_sop_date"]
