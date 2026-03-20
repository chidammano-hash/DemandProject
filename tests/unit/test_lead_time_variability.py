"""Tests for scripts/compute_lead_time_variability.py — IPfeature2."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.compute_lead_time_variability import (
    extract_lt_change_points,
    compute_lt_metrics,
    classify_lt_variability_class,
)


@pytest.fixture
def default_config():
    return {
        "history": {"history_months": 12},
        "change_point": {"min_observations": 3},
        "cv_thresholds": {"stable": 0.15, "moderate": 0.40},
        "batch": {"batch_size": 500},
    }


# ---------------------------------------------------------------------------
# extract_lt_change_points()
# ---------------------------------------------------------------------------

class TestExtractLtChangePoints:
    def test_empty_series_returns_empty(self):
        assert extract_lt_change_points([]) == []

    def test_single_element_returns_it(self):
        assert extract_lt_change_points([30.0]) == [30.0]

    def test_constant_series_returns_single_observation(self):
        """All identical values → only the first is kept (no change-points)."""
        result = extract_lt_change_points([30.0, 30.0, 30.0, 30.0, 30.0])
        assert result == [30.0]

    def test_first_value_always_included(self):
        result = extract_lt_change_points([7.0, 14.0, 14.0])
        assert result[0] == 7.0

    def test_detects_single_change(self):
        result = extract_lt_change_points([30.0, 30.0, 45.0, 45.0])
        assert result == [30.0, 45.0]

    def test_detects_multiple_changes(self):
        lt = [30.0, 30.0, 45.0, 45.0, 60.0, 60.0, 30.0]
        result = extract_lt_change_points(lt)
        assert result == [30.0, 45.0, 60.0, 30.0]

    def test_every_day_different(self):
        lt = [10.0, 20.0, 30.0, 40.0]
        result = extract_lt_change_points(lt)
        assert result == [10.0, 20.0, 30.0, 40.0]

    def test_returns_only_change_point_values(self):
        """Intermediate plateau is not repeated."""
        lt = [10.0] * 5 + [20.0] * 3 + [10.0] * 2
        result = extract_lt_change_points(lt)
        assert result == [10.0, 20.0, 10.0]


# ---------------------------------------------------------------------------
# compute_lt_metrics()
# ---------------------------------------------------------------------------

class TestComputeLtMetrics:
    def test_insufficient_observations_returns_none(self, default_config):
        """Fewer than min_observations (3) → None."""
        assert compute_lt_metrics([30.0, 45.0], default_config) is None

    def test_exactly_min_observations_ok(self, default_config):
        result = compute_lt_metrics([30.0, 45.0, 60.0], default_config)
        assert result is not None

    def test_mean_computed_correctly(self, default_config):
        result = compute_lt_metrics([10.0, 20.0, 30.0], default_config)
        assert result["lt_mean_days"] == pytest.approx(20.0)

    def test_std_computed_correctly(self, default_config):
        import numpy as np
        obs = [10.0, 20.0, 30.0]
        result = compute_lt_metrics(obs, default_config)
        expected_std = float(np.std(obs, ddof=1))
        assert result["lt_std_days"] == pytest.approx(expected_std)

    def test_cv_equals_std_over_mean(self, default_config):
        result = compute_lt_metrics([10.0, 20.0, 30.0], default_config)
        expected_cv = result["lt_std_days"] / result["lt_mean_days"]
        assert result["lt_cv"] == pytest.approx(expected_cv)

    def test_min_max_correct(self, default_config):
        result = compute_lt_metrics([15.0, 30.0, 45.0, 7.0], default_config)
        assert result["lt_min_days"] == pytest.approx(7.0)
        assert result["lt_max_days"] == pytest.approx(45.0)

    def test_percentiles_ordered(self, default_config):
        obs = list(range(5, 55, 5))  # 5,10,...,50 — 10 observations
        result = compute_lt_metrics(obs, default_config)
        assert result["lt_p25_days"] <= result["lt_p50_days"]
        assert result["lt_p50_days"] <= result["lt_p75_days"]
        assert result["lt_p75_days"] <= result["lt_p95_days"]

    def test_constant_observations_std_is_zero(self, default_config):
        result = compute_lt_metrics([30.0, 30.0, 30.0], default_config)
        assert result["lt_std_days"] == pytest.approx(0.0)
        assert result["lt_cv"] == pytest.approx(0.0)

    def test_observation_count_matches_input_length(self, default_config):
        obs = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = compute_lt_metrics(obs, default_config)
        assert result["observation_count"] == 5

    def test_output_keys_present(self, default_config):
        result = compute_lt_metrics([10.0, 20.0, 30.0], default_config)
        expected = {
            "lt_mean_days", "lt_std_days", "lt_cv",
            "lt_min_days", "lt_max_days",
            "lt_p25_days", "lt_p50_days", "lt_p75_days", "lt_p95_days",
            "observation_count",
        }
        assert expected.issubset(set(result.keys()))

    def test_zero_mean_cv_is_zero(self, default_config):
        """All-zero observations: mean=0 → CV should be 0 (not division error)."""
        result = compute_lt_metrics([0.0, 0.0, 0.0], default_config)
        assert result["lt_cv"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# classify_lt_variability_class()
# ---------------------------------------------------------------------------

class TestClassifyLtVariabilityClass:
    def test_none_cv_returns_none(self, default_config):
        assert classify_lt_variability_class(None, default_config) is None

    def test_stable_class(self, default_config):
        assert classify_lt_variability_class(0.0, default_config) == "stable"
        assert classify_lt_variability_class(0.10, default_config) == "stable"
        assert classify_lt_variability_class(0.14, default_config) == "stable"

    def test_stable_boundary(self, default_config):
        """CV exactly at stable threshold → moderate."""
        assert classify_lt_variability_class(0.15, default_config) == "moderate"

    def test_moderate_class(self, default_config):
        assert classify_lt_variability_class(0.20, default_config) == "moderate"
        assert classify_lt_variability_class(0.39, default_config) == "moderate"

    def test_moderate_boundary(self, default_config):
        """CV exactly at moderate threshold → volatile."""
        assert classify_lt_variability_class(0.40, default_config) == "volatile"

    def test_volatile_class(self, default_config):
        assert classify_lt_variability_class(0.50, default_config) == "volatile"
        assert classify_lt_variability_class(1.50, default_config) == "volatile"
