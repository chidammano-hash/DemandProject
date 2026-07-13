"""Tests for compute_min_cluster_rows() dynamic cluster sizing."""

from unittest.mock import patch

import pytest

from common.core.constants import MIN_CLUSTER_ROWS, compute_min_cluster_rows


class TestComputeMinClusterRows:
    """Tests for the dynamic minimum cluster rows computation."""

    def test_high_feature_count_above_floor(self):
        """66 features * 3 spf * 1.25 = 248, which exceeds floor of 50."""
        result = compute_min_cluster_rows(66, samples_per_feature=3)
        assert result == 247  # int(66 * 3 * 1.25) = int(247.5) = 247

    def test_low_feature_count_returns_floor(self):
        """10 features * 3 spf * 1.25 = 37.5 → 37, below floor → returns 50."""
        result = compute_min_cluster_rows(10, samples_per_feature=3)
        assert result == 50

    def test_zero_features_returns_floor(self):
        """0 features should return the floor value."""
        result = compute_min_cluster_rows(0, samples_per_feature=3)
        assert result == 50

    def test_custom_samples_per_feature(self):
        """Custom spf should scale proportionally."""
        result = compute_min_cluster_rows(20, samples_per_feature=5)
        # 20 * 5 * 1.25 = 125
        assert result == 125

    def test_custom_floor(self):
        """Custom floor should be respected when computed value is lower."""
        result = compute_min_cluster_rows(5, samples_per_feature=3, floor=100)
        # 5 * 3 * 1.25 = 18.75 → 18, below floor=100
        assert result == 100

    def test_result_at_least_floor(self):
        """Result should never be below the floor regardless of inputs."""
        for n in range(0, 15):
            result = compute_min_cluster_rows(n, samples_per_feature=3)
            assert result >= 50, f"n_features={n} gave result {result} below floor"

    def test_result_is_int(self):
        """Result should always be an integer."""
        result = compute_min_cluster_rows(66, samples_per_feature=3)
        assert isinstance(result, int)

    def test_reads_config_when_spf_is_none(self):
        """When samples_per_feature is None, should read from forecast_pipeline_config.yaml."""
        mock_cfg = {"clustering": {"cluster_sizing": {"samples_per_feature": 5}}}
        with patch("common.core.utils.load_forecast_pipeline_config", return_value=mock_cfg):
            result = compute_min_cluster_rows(20)
            # 20 * 5 * 1.25 = 125
            assert result == 125

    @pytest.mark.parametrize(
        "mock_cfg",
        [
            {},
            {"clustering": {}},
            {"clustering": {"samples_per_feature": 5}},
            {"clustering": {"cluster_sizing": {}}},
        ],
    )
    def test_missing_exact_config_path_fails_loud(self, mock_cfg):
        """Cluster sizing has one required YAML path and no Python fallback."""
        with patch("common.core.utils.load_forecast_pipeline_config", return_value=mock_cfg):
            with pytest.raises((KeyError, ValueError), match="samples_per_feature"):
                compute_min_cluster_rows(66)

    def test_static_constant_still_available(self):
        """MIN_CLUSTER_ROWS constant should still be 50 for backward compat."""
        assert MIN_CLUSTER_ROWS == 50

    def test_scales_above_static_constant(self):
        """With enough features, compute_min_cluster_rows > MIN_CLUSTER_ROWS."""
        result = compute_min_cluster_rows(66, samples_per_feature=3)
        assert result > MIN_CLUSTER_ROWS
