"""Unit tests for IPfeature13 investment optimization pure functions."""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.compute_investment_plan import (
    compute_marginal_roi,
    estimate_current_csl,
    BASE_CSL,
)


# ---------------------------------------------------------------------------
# compute_marginal_roi
# ---------------------------------------------------------------------------

def test_marginal_roi_basic():
    roi = compute_marginal_roi(csl_increment=0.05, investment_increment=500.0)
    assert roi == pytest.approx(0.05 / 500.0)


def test_marginal_roi_zero_investment():
    # Should not divide by zero — investment floor = 1
    roi = compute_marginal_roi(csl_increment=0.10, investment_increment=0.0)
    assert roi == pytest.approx(0.10 / 1.0)


def test_marginal_roi_zero_csl_increment():
    assert compute_marginal_roi(0.0, 1000.0) == 0.0


def test_marginal_roi_high_roi():
    roi = compute_marginal_roi(csl_increment=0.20, investment_increment=100.0)
    assert roi == pytest.approx(0.002)


# ---------------------------------------------------------------------------
# estimate_current_csl
# ---------------------------------------------------------------------------

def test_current_csl_fully_funded():
    csl = estimate_current_csl(100.0, 100.0, 0.95)
    assert csl == pytest.approx(0.95)


def test_current_csl_overfunded():
    # current > recommended → at target
    csl = estimate_current_csl(150.0, 100.0, 0.95)
    assert csl == pytest.approx(0.95)


def test_current_csl_zero_ss():
    csl = estimate_current_csl(0.0, 100.0, 0.95)
    assert csl == pytest.approx(BASE_CSL)


def test_current_csl_half_funded():
    csl = estimate_current_csl(50.0, 100.0, 0.95)
    expected = BASE_CSL + (0.95 - BASE_CSL) * 0.5
    assert csl == pytest.approx(expected)


def test_current_csl_zero_recommended():
    # recommended=0 → already at target
    csl = estimate_current_csl(10.0, 0.0, 0.90)
    assert csl == pytest.approx(0.90)


def test_current_csl_interpolation_monotone():
    csl_0 = estimate_current_csl(0.0, 100.0, 0.95)
    csl_50 = estimate_current_csl(50.0, 100.0, 0.95)
    csl_100 = estimate_current_csl(100.0, 100.0, 0.95)
    assert csl_0 <= csl_50 <= csl_100


def test_marginal_roi_ranking_high_roi_first():
    """Items with higher ROI should rank first."""
    high_roi = compute_marginal_roi(0.10, 100.0)
    low_roi = compute_marginal_roi(0.10, 1000.0)
    assert high_roi > low_roi


def test_csl_increment_positive_for_underfunded():
    current_csl = estimate_current_csl(50.0, 100.0, 0.95)
    csl_increment = max(0.0, 0.95 - current_csl)
    assert csl_increment > 0
