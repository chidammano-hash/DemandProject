"""Gen-4 Roadmap Stream F (Supply Chain closures) — unit tests.

Covers:
  - SC-2 periodic-review ROP protects (LT + R/2)
  - SC-2 supply-yield variability folded into SS combined
  - SC-4 max_service / equalize_dos rebalancing solvers
  - SC-8 root_cause_key + severity_band + SLA computation
  - SC-10 canonical accuracy metric
"""
from __future__ import annotations

import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# SC-2: Periodic-review ROP  (LT + R/2)
# ---------------------------------------------------------------------------

def test_periodic_rop_adds_half_review_period():
    from common.inventory.safety_stock import compute_reorder_point_periodic

    # D_avg=10, LT=14, R=7  →  ROP = 10 * (14 + 3.5) + SS = 175 + SS
    rop = compute_reorder_point_periodic(
        ss_combined=50.0,
        avg_daily_demand=10.0,
        lt_mean_days=14.0,
        review_cycle_days=7.0,
    )
    assert rop == pytest.approx(10.0 * 17.5 + 50.0)


def test_periodic_rop_zero_review_period_equals_continuous():
    """R=0 degenerates to LT-only protection (continuous ROP)."""
    from common.inventory.safety_stock import (
        compute_reorder_point,
        compute_reorder_point_periodic,
    )

    ss = 50.0
    d_avg = 10.0
    lt = 14.0
    assert compute_reorder_point_periodic(ss, d_avg, lt, 0.0) == pytest.approx(
        compute_reorder_point(ss, d_avg, lt)
    )


# ---------------------------------------------------------------------------
# SC-2: Supply-yield variability folds into SS combined
# ---------------------------------------------------------------------------

def test_ss_combined_with_yield_variance_adds_component():
    from common.inventory.safety_stock import compute_ss_combined

    # z=1.64, sigma_d=2, LT=14, d_avg=10, lt_std=2
    base = compute_ss_combined(
        z_score=1.64, sigma_demand=2.0, lt_mean_days=14.0,
        avg_daily_demand=10.0, lt_std_days=2.0, yield_std_days=0.0,
    )
    with_yield = compute_ss_combined(
        z_score=1.64, sigma_demand=2.0, lt_mean_days=14.0,
        avg_daily_demand=10.0, lt_std_days=2.0, yield_std_days=1.5,
    )
    # yield>0 MUST increase SS (variance additive under uncorrelated assumption)
    assert with_yield > base
    # Exact: sqrt(14*4 + 100*(4+2.25)) * 1.64
    expected = 1.64 * math.sqrt(14 * 4 + 100 * (4 + 2.25))
    assert with_yield == pytest.approx(expected)


def test_ss_components_returns_ss_yield_only():
    from common.inventory.safety_stock import compute_ss_components

    result = compute_ss_components(
        z=1.64,
        demand_mean_monthly=300.0,
        demand_std_monthly=30.0,
        lt_mean_days=14.0,
        lt_std_days=2.0,
        yield_std_days=1.0,
    )
    assert "ss_yield_only" in result
    assert result["ss_yield_only"] > 0
    # Zero yield → zero ss_yield_only
    r0 = compute_ss_components(
        z=1.64, demand_mean_monthly=300.0, demand_std_monthly=30.0,
        lt_mean_days=14.0, lt_std_days=2.0, yield_std_days=0.0,
    )
    assert r0["ss_yield_only"] == 0.0


# ---------------------------------------------------------------------------
# SC-4: max_service and equalize_dos solvers
# ---------------------------------------------------------------------------

def _mk_candidate(**kw):
    """Build a candidate dict with sensible defaults for solver tests."""
    base = {
        "item_id": "X1", "source_loc": "A", "dest_loc": "B", "lane_id": None,
        "recommended_qty": 10,
        "source_on_hand": 200, "source_dos": 20,
        "source_ss_target": 50, "source_excess_qty": 150,
        "dest_on_hand": 5,  "dest_dos": 1,
        "dest_ss_target": 50, "dest_shortage_qty": 45,
        "abc_class": "A", "cost_per_unit": 0.5, "fixed_cost": 0,
        "transfer_cost": 5, "net_benefit": 100, "roi": 20,
        "urgency": "critical", "transfer_lt_days": 3, "dest_reorder_point": 0,
    }
    base.update(kw)
    return base


def test_max_service_solver_prefers_largest_shortage():
    from scripts.inventory.compute_rebalancing import max_service_solver

    c1 = _mk_candidate(item_id="X1", dest_shortage_qty=10, dest_ss_target=100)
    c2 = _mk_candidate(item_id="X2", dest_shortage_qty=90, dest_ss_target=100,
                       source_excess_qty=90)
    result = max_service_solver([c1, c2], {})
    assert result[0]["item_id"] == "X2"


def test_equalize_dos_skips_when_all_within_tolerance():
    from scripts.inventory.compute_rebalancing import equalize_dos_solver

    # Both locs at DOS=10, mean=10, tolerance=0.10 → no move should happen
    state = {
        ("X1", "A"): {"item_id": "X1", "loc": "A", "dos": 10, "daily_sales": 1,
                     "on_hand": 100, "ss_target": 50},
        ("X1", "B"): {"item_id": "X1", "loc": "B", "dos": 10, "daily_sales": 1,
                     "on_hand": 100, "ss_target": 50},
    }
    c = _mk_candidate(source_dos=10, dest_dos=10)
    result = equalize_dos_solver([c], {}, state)
    assert result == []


def test_equalize_dos_selects_surplus_to_deficit():
    from scripts.inventory.compute_rebalancing import equalize_dos_solver

    state = {
        ("X1", "A"): {"item_id": "X1", "loc": "A", "dos": 30, "daily_sales": 10},
        ("X1", "B"): {"item_id": "X1", "loc": "B", "dos": 2,  "daily_sales": 10},
    }
    c = _mk_candidate(source_dos=30, dest_dos=2, recommended_qty=50)
    result = equalize_dos_solver([c], {}, state)
    assert len(result) == 1
    assert result[0]["recommended_qty"] == 50


# ---------------------------------------------------------------------------
# SC-8: root_cause_key + severity_band + sla_due_at
# ---------------------------------------------------------------------------

def test_derive_severity_band_critical_high_medium_low():
    from common.engines.exception_engine import derive_severity_band

    assert derive_severity_band(0.90) == "critical"
    assert derive_severity_band(0.60) == "high"
    assert derive_severity_band(0.30) == "medium"
    assert derive_severity_band(0.10) == "low"


def test_compute_sla_due_at_respects_band_hours():
    from common.engines.exception_engine import compute_sla_due_at

    t0 = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)
    critical_due = compute_sla_due_at(t0, "critical")
    high_due = compute_sla_due_at(t0, "high")
    # Critical SLA must be tighter than High
    assert critical_due < high_due
    # Default: critical=4h, high=24h
    assert (critical_due - t0).total_seconds() == 4 * 3600
    assert (high_due - t0).total_seconds() == 24 * 3600


def test_compute_root_cause_key_stable_and_grouping():
    from common.engines.exception_engine import compute_root_cause_key

    sd_over = {"bias_pct": 30.0}
    sd_under = {"bias_pct": -30.0}
    k_over = compute_root_cause_key("forecast_bias", sd_over)
    k_under = compute_root_cause_key("forecast_bias", sd_under)
    # Same detector, same direction bucket → same key
    k_over_again = compute_root_cause_key("forecast_bias", {"bias_pct": 55.0})
    assert k_over == k_over_again
    # Opposite direction → different key
    assert k_over != k_under


# ---------------------------------------------------------------------------
# SC-10: canonical accuracy metric
# ---------------------------------------------------------------------------

def test_compute_accuracy_perfect_forecast():
    from common.services.metrics import compute_accuracy

    a = [100.0, 200.0, 300.0]
    assert compute_accuracy(a, a) == pytest.approx(100.0)


def test_compute_accuracy_matches_wape_definition():
    from common.services.metrics import compute_accuracy

    # A=[10, 10], F=[8, 12] → |F-A|=[2,2]=4, |SUM A|=20, WAPE=0.2, acc=80
    assert compute_accuracy([10, 10], [8, 12]) == pytest.approx(80.0)


def test_compute_accuracy_clamps_at_zero():
    from common.services.metrics import compute_accuracy

    # extreme over-forecast: WAPE>1 → clamp to 0
    acc = compute_accuracy([1.0, 1.0], [100.0, 100.0])
    assert acc == 0.0


def test_compute_accuracy_zero_actuals_returns_none():
    from common.services.metrics import compute_accuracy

    assert compute_accuracy([0.0, 0.0], [5.0, 5.0]) is None


def test_compute_accuracy_length_mismatch_raises():
    from common.services.metrics import compute_accuracy

    with pytest.raises(ValueError):
        compute_accuracy([1.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# SC-3: SOP decision log helper
# ---------------------------------------------------------------------------

def test_log_sop_decision_rejects_bad_type():
    from common.engines.sop_decisions import log_sop_decision

    class _StubConn:
        def cursor(self):
            raise AssertionError("should not reach cursor for invalid type")
        def rollback(self):   # pragma: no cover
            pass

    with pytest.raises(ValueError):
        log_sop_decision(
            _StubConn(), decision_type="not_a_real_thing", decided_by="tester",
        )


def test_log_sop_decision_table_missing_returns_none():
    """When fact_sop_decisions isn't present, helper returns None (non-fatal)."""
    from common.engines.sop_decisions import log_sop_decision

    class _RaisingCursor:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def execute(self, *a, **kw):
            raise RuntimeError("relation fact_sop_decisions does not exist")
        def fetchone(self):   # pragma: no cover
            return None

    class _StubConn:
        def cursor(self):
            return _RaisingCursor()
        def rollback(self):
            pass
        def commit(self):
            pass

    result = log_sop_decision(
        _StubConn(), decision_type="promote", decided_by="tester",
        rationale={"note": "test"},
    )
    assert result is None
