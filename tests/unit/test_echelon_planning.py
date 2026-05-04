"""Unit tests for scripts/compute_echelon_targets.py — F3.5."""

import math
import pytest
from scripts.inventory.compute_echelon_targets import (
    compute_pooled_sigma,
    compute_echelon_ss,
    compute_echelon_rop,
    compute_downstream_coverage_days,
    compute_cascade_risk_score,
)


class TestComputePooledSigma:
    def test_single_store(self):
        result = compute_pooled_sigma([15.0])
        assert result == pytest.approx(15.0, abs=1e-4)

    def test_three_stores(self):
        # sqrt(225 + 144 + 324) = sqrt(693) ≈ 26.32
        result = compute_pooled_sigma([15.0, 12.0, 18.0])
        assert result == pytest.approx(math.sqrt(225 + 144 + 324), abs=1e-2)

    def test_pooled_less_than_sum(self):
        # Pooled variance < naive sum, capturing risk-pooling benefit
        stores = [15.0, 12.0, 18.0]
        pooled = compute_pooled_sigma(stores)
        naive_sum = sum(stores)
        assert pooled < naive_sum

    def test_empty_list_returns_zero(self):
        assert compute_pooled_sigma([]) == pytest.approx(0.0, abs=1e-4)

    def test_identical_stores(self):
        # N stores of σ → pooled = σ × sqrt(N)
        sigma = 15.0
        n = 4
        result = compute_pooled_sigma([sigma] * n)
        assert result == pytest.approx(sigma * math.sqrt(n), abs=1e-4)


class TestComputeEchelonSS:
    def test_demand_variability_only(self):
        # sigma_lt = 0 → SS = Z × σ_demand × sqrt(mean_LT)
        z = 1.645
        result = compute_echelon_ss(45, 26.3, 10, 0.0, z)
        expected = z * math.sqrt(10 * 26.3 ** 2)
        assert result == pytest.approx(expected, abs=1e-1)

    def test_combined_formula(self):
        # mean_demand=45, σ_demand=26.3, mean_LT=10, σ_LT=2, Z=1.645
        result = compute_echelon_ss(45, 26.3, 10, 2.0, 1.645)
        variance = 10 * (26.3 ** 2) + (45 ** 2) * (2 ** 2)
        expected = 1.645 * math.sqrt(variance)
        assert result == pytest.approx(expected, abs=1e-1)

    def test_floor_at_zero(self):
        result = compute_echelon_ss(0, 0, 10, 0, 1.645)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_higher_z_score_higher_ss(self):
        ss_95 = compute_echelon_ss(45, 26.3, 10, 2.0, 1.645)
        ss_99 = compute_echelon_ss(45, 26.3, 10, 2.0, 2.326)
        assert ss_99 > ss_95


class TestComputeEchelonROP:
    def test_basic_rop(self):
        # ROP = mean_demand × mean_LT + SS
        result = compute_echelon_rop(45.0, 10.0, 137.0)
        assert result == pytest.approx(587.0, abs=1e-2)

    def test_zero_demand(self):
        result = compute_echelon_rop(0.0, 10.0, 100.0)
        assert result == pytest.approx(100.0, abs=1e-4)


class TestComputeDownstreamCoverage:
    def test_normal_coverage(self):
        result = compute_downstream_coverage_days(500.0, 45.0)
        assert result == pytest.approx(500 / 45, abs=1e-2)

    def test_zero_demand(self):
        result = compute_downstream_coverage_days(500.0, 0.0)
        assert result == pytest.approx(0.0, abs=1e-4)


class TestComputeCascadeRiskScore:
    def test_no_risk_when_above_rop(self):
        score, severity = compute_cascade_risk_score(3, 600.0, 587.0)
        assert score == pytest.approx(0.0, abs=1e-4)
        assert severity == "ok"

    def test_critical_risk(self):
        # Large shortfall × many stores → critical (capped at 100)
        # shortfall_pct = (587-10)/587 ≈ 0.983, raw = 0.983 × 20 × 10 = 196.6 → capped at 100
        score, severity = compute_cascade_risk_score(20, 10.0, 587.0)
        assert severity == "critical"
        assert score == 100.0  # capped at 100

    def test_risk_increases_with_downstream_count(self):
        score_1, _ = compute_cascade_risk_score(1, 400.0, 587.0)
        score_5, _ = compute_cascade_risk_score(5, 400.0, 587.0)
        assert score_5 > score_1
