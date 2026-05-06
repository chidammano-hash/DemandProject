"""Tests for common.ml.clustering package (features, training, labeling, scenario)."""

import json
import re
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs


# ============================================================================
# 0. Package re-export smoke test
# ============================================================================

class TestPackageImports:
    """Verify the clustering __init__.py re-exports everything."""

    def test_all_public_symbols_importable(self):
        # Lightweight constants live at the package root.
        from common.ml.clustering import CORE_FEATURES, LOG_TRANSFORM_FEATURES

        # Heavy helpers (training/labeling/scenario) are imported from their
        # submodules directly so the package init stays free of matplotlib /
        # sklearn / scipy. See common/ml/clustering/__init__.py docstring.
        from common.ml.clustering.features import compute_time_series_features
        from common.ml.clustering.labeling import assign_cluster_labels
        from common.ml.clustering.scenario import (
            generate_scenario_id,
            get_scenario_result,
            promote_scenario,
        )
        from common.ml.clustering.training import find_optimal_k, merge_small_clusters

        # Smoke: make sure they are callable / iterable
        assert len(CORE_FEATURES) > 0
        assert len(LOG_TRANSFORM_FEATURES) > 0
        assert callable(compute_time_series_features)
        assert callable(find_optimal_k)
        assert callable(merge_small_clusters)
        assert callable(assign_cluster_labels)
        assert callable(generate_scenario_id)
        assert callable(promote_scenario)
        assert callable(get_scenario_result)


# ============================================================================
# 1. Features
# ============================================================================

def _make_monthly_df(values: list[float], start: str = "2022-01-01") -> pd.DataFrame:
    """Build a DataFrame with ``startdate`` (monthly) and ``qty`` columns."""
    dates = pd.date_range(start, periods=len(values), freq="MS")
    return pd.DataFrame({"startdate": dates, "qty": values})


class TestComputeTimeSeriesFeatures:
    """Tests for compute_time_series_features()."""

    def test_24_month_demand_all_dimensions(self):
        """24-month demand produces features from all 6 dimensions."""
        from common.ml.clustering.features import compute_time_series_features

        np.random.seed(42)
        # Moderate demand with a seasonal bump in months 6-8
        base = [100 + i * 2 for i in range(24)]
        base[5] += 50
        base[6] += 60
        base[7] += 40
        df = _make_monthly_df(base)

        feat = compute_time_series_features(df)

        # Volume
        assert "mean_demand" in feat.index
        assert "cv_demand" in feat.index
        assert "iqr_demand" in feat.index

        # Trend
        assert "trend_slope" in feat.index
        assert "trend_slope_norm" in feat.index
        assert "trend_r2" in feat.index

        # Seasonality
        assert "seasonality_strength" in feat.index
        assert "seasonal_amplitude" in feat.index
        assert "seasonal_r2" in feat.index
        assert "yoy_correlation" in feat.index

        # Periodicity
        assert "periodicity_strength" in feat.index

        # Intermittency
        assert "zero_demand_pct" in feat.index
        assert "adi" in feat.index

        # Lifecycle
        assert "months_available" in feat.index
        assert "recency_ratio" in feat.index
        assert "cagr" in feat.index

    def test_months_available_matches_input(self):
        from common.ml.clustering.features import compute_time_series_features

        df = _make_monthly_df([10.0] * 30)
        feat = compute_time_series_features(df)
        assert feat["months_available"] == 30

    def test_empty_dataframe_returns_empty_series(self):
        from common.ml.clustering.features import compute_time_series_features

        df = pd.DataFrame(columns=["startdate", "qty"])
        feat = compute_time_series_features(df)
        assert len(feat) == 0

    def test_flat_demand_zero_cv_zero_trend(self):
        """Flat demand should have CV=0, trend_slope=0, and zero_demand_pct=0."""
        from common.ml.clustering.features import compute_time_series_features

        df = _make_monthly_df([50.0] * 24)
        feat = compute_time_series_features(df)

        assert feat["cv_demand"] == pytest.approx(0.0, abs=1e-9)
        assert feat["trend_slope"] == pytest.approx(0.0, abs=1e-9)
        assert feat["zero_demand_pct"] == pytest.approx(0.0)
        assert feat["demand_stability"] == pytest.approx(1.0)

    def test_short_history_below_12_months(self):
        """< 12 months: seasonality features default to 0."""
        from common.ml.clustering.features import compute_time_series_features

        df = _make_monthly_df([10.0, 20.0, 30.0, 40.0])
        feat = compute_time_series_features(df)

        assert feat["seasonality_strength"] == 0.0
        assert feat["seasonal_r2"] == 0.0
        assert feat["yoy_correlation"] == 0.0
        assert feat["cagr"] == 0.0

    def test_strongly_trending_demand(self):
        """Linearly increasing demand should have positive trend_slope and trend_r2."""
        from common.ml.clustering.features import compute_time_series_features

        values = [float(10 + 5 * i) for i in range(24)]
        df = _make_monthly_df(values)
        feat = compute_time_series_features(df)

        assert feat["trend_slope"] > 0
        assert feat["trend_slope_norm"] > 0
        assert feat["trend_r2"] > 0.9  # nearly perfect linear
        assert feat["trend_direction"] == 1

    def test_intermittent_demand(self):
        """Many zeros should have high zero_demand_pct and ADI > 1."""
        from common.ml.clustering.features import compute_time_series_features

        values = [0, 0, 10, 0, 0, 0, 20, 0, 0, 0, 0, 15,
                  0, 0, 0, 5, 0, 0, 0, 0, 0, 10, 0, 0]
        df = _make_monthly_df(values)
        feat = compute_time_series_features(df)

        assert feat["zero_demand_pct"] > 0.5
        assert feat["adi"] > 1.0

    def test_backward_compat_aliases(self):
        """Verify backward-compat alias fields are present."""
        from common.ml.clustering.features import compute_time_series_features

        df = _make_monthly_df([100.0] * 24)
        feat = compute_time_series_features(df)

        assert "year_over_year_correlation" in feat.index
        assert feat["year_over_year_correlation"] == feat["yoy_correlation"]
        assert "growth_rate" in feat.index
        assert feat["growth_rate"] == feat["cagr"]
        assert "recent_vs_historical" in feat.index
        assert feat["recent_vs_historical"] == feat["recency_ratio"]
        assert "sparsity_score" in feat.index
        assert feat["sparsity_score"] == feat["zero_demand_pct"]


class TestSeasonalR2:
    """Tests for _seasonal_r2()."""

    def test_returns_zero_when_less_than_24_months(self):
        from common.ml.clustering.features import _seasonal_r2

        y = np.array([1.0, 2.0, 3.0])
        months = np.array([1.0, 2.0, 3.0])
        assert _seasonal_r2(y, months) == 0.0

    def test_returns_zero_for_flat_signal(self):
        from common.ml.clustering.features import _seasonal_r2

        y = np.ones(36)
        months = np.tile(np.arange(1, 13, dtype=float), 3)
        assert _seasonal_r2(y, months) == 0.0

    def test_positive_for_seasonal_signal(self):
        from common.ml.clustering.features import _seasonal_r2

        # 3 years of strong seasonality (sine wave)
        months = np.tile(np.arange(1, 13, dtype=float), 3)
        y = 100 + 50 * np.sin(2 * np.pi * months / 12)
        r2 = _seasonal_r2(y, months)
        assert r2 > 0.5


class TestPeriodicityStrength:
    """Tests for _periodicity_strength()."""

    def test_returns_zero_when_less_than_12(self):
        from common.ml.clustering.features import _periodicity_strength

        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert _periodicity_strength(y) == 0.0

    def test_returns_zero_for_flat_signal(self):
        from common.ml.clustering.features import _periodicity_strength

        y = np.ones(24)
        assert _periodicity_strength(y) == 0.0

    def test_strong_periodic_signal(self):
        from common.ml.clustering.features import _periodicity_strength

        t = np.arange(36)
        y = np.sin(2 * np.pi * t / 12)  # exact 12-month cycle
        strength = _periodicity_strength(y)
        assert strength > 0.5


class TestADI:
    """Tests for _adi()."""

    def test_all_zeros(self):
        from common.ml.clustering.features import _adi

        y = np.zeros(12)
        assert _adi(y) == 12.0  # len(demand_values)

    def test_single_nonzero(self):
        from common.ml.clustering.features import _adi

        y = np.array([0, 0, 5, 0, 0, 0])
        assert _adi(y) == 6.0  # len

    def test_two_nonzero(self):
        from common.ml.clustering.features import _adi

        y = np.array([0, 10, 0, 0, 20, 0])
        # nonzero at index 1, 4 -> gap = 3 -> mean = 3.0
        assert _adi(y) == pytest.approx(3.0)

    def test_multiple_nonzero(self):
        from common.ml.clustering.features import _adi

        y = np.array([5, 0, 10, 0, 0, 15, 20])
        # nonzero at 0, 2, 5, 6 -> gaps [2, 3, 1] -> mean 2.0
        assert _adi(y) == pytest.approx(2.0)

    def test_all_nonzero(self):
        from common.ml.clustering.features import _adi

        y = np.array([1, 2, 3, 4, 5])
        # gaps between consecutive: [1,1,1,1] -> mean = 1.0
        assert _adi(y) == pytest.approx(1.0)


# ============================================================================
# 2. Training
# ============================================================================

class TestCoreFeatures:
    """Tests for CORE_FEATURES and LOG_TRANSFORM_FEATURES."""

    def test_core_features_count(self):
        from common.ml.clustering.constants import CORE_FEATURES

        assert len(CORE_FEATURES) == 14

    def test_log_transform_features_all_in_expected_set(self):
        from common.ml.clustering.constants import LOG_TRANSFORM_FEATURES

        expected = {
            "mean_demand", "median_demand", "std_demand",
            "total_demand", "max_demand", "iqr_demand", "adi",
        }
        assert set(LOG_TRANSFORM_FEATURES) == expected


class TestFindOptimalK:
    """Tests for find_optimal_k()."""

    def test_finds_k_near_true_centers(self):
        from common.ml.clustering.training import find_optimal_k

        np.random.seed(0)
        X, _ = make_blobs(n_samples=300, centers=3, n_features=5, random_state=0)
        result = find_optimal_k(X, k_range=(2, 5), min_cluster_size_pct=5.0)

        assert "optimal_k" in result
        assert result["optimal_k"] in [2, 3, 4, 5]
        assert "k_values" in result
        assert result["k_values"] == [2, 3, 4, 5]
        assert len(result["silhouette_scores"]) == 4
        assert len(result["ch_scores"]) == 4
        assert len(result["combined_scores"]) == 4
        assert len(result["feasible_mask"]) == 4

    def test_returns_elbow_and_silhouette_variants(self):
        from common.ml.clustering.training import find_optimal_k

        X, _ = make_blobs(n_samples=200, centers=3, n_features=4, random_state=42)
        result = find_optimal_k(X, k_range=(2, 5))

        assert "optimal_k_elbow" in result
        assert "optimal_k_silhouette" in result
        assert result["optimal_k_elbow"] in [2, 3, 4, 5]
        assert result["optimal_k_silhouette"] in [2, 3, 4, 5]

    def test_all_infeasible_falls_back_to_best_silhouette(self):
        """When min_cluster_size_pct is impossibly high, fall back to silhouette."""
        from common.ml.clustering.training import find_optimal_k

        X, _ = make_blobs(n_samples=100, centers=2, n_features=3, random_state=0)
        # 60% threshold -- any 2+ clusters can't all have >= 60% each
        result = find_optimal_k(X, k_range=(2, 4), min_cluster_size_pct=60.0)

        # All should be infeasible
        assert not any(result["feasible_mask"])
        # Should still return a valid K
        assert result["optimal_k"] in [2, 3, 4]

    def test_inertias_decrease_with_k(self):
        from common.ml.clustering.training import find_optimal_k

        X, _ = make_blobs(n_samples=300, centers=4, n_features=5, random_state=1)
        result = find_optimal_k(X, k_range=(2, 6), min_cluster_size_pct=2.0)

        inertias = result["inertias"]
        # Inertia should generally decrease as K increases
        assert inertias[0] > inertias[-1]


class TestMergeSmallClusters:
    """Tests for merge_small_clusters()."""

    def test_no_merge_when_all_large(self):
        from common.ml.clustering.training import merge_small_clusters

        X = np.random.RandomState(0).randn(100, 3)
        labels = np.array([0] * 50 + [1] * 50)
        centroids = np.array([X[:50].mean(axis=0), X[50:].mean(axis=0)])

        new_labels, new_centroids = merge_small_clusters(X, labels, centroids, min_size=10)
        assert set(new_labels) == {0, 1}
        assert new_centroids.shape[0] == 2

    def test_merges_small_cluster_into_nearest(self):
        from common.ml.clustering.training import merge_small_clusters

        rng = np.random.RandomState(42)
        # Cluster 0: 50 points near [0, 0]
        c0 = rng.randn(50, 2) + np.array([0, 0])
        # Cluster 1: 50 points near [10, 10]
        c1 = rng.randn(50, 2) + np.array([10, 10])
        # Cluster 2: 3 points near [1, 1] (small, should merge into 0)
        c2 = rng.randn(3, 2) + np.array([1, 1])

        X = np.vstack([c0, c1, c2])
        labels = np.array([0] * 50 + [1] * 50 + [2] * 3)
        centroids = np.array([c0.mean(axis=0), c1.mean(axis=0), c2.mean(axis=0)])

        new_labels, new_centroids = merge_small_clusters(X, labels, centroids, min_size=5)

        # Cluster 2 should be gone -- merged into nearest (0)
        assert set(new_labels) == {0, 1}
        assert new_centroids.shape[0] == 2
        # The 3 points that were cluster 2 should now be cluster 0 (after re-numbering)
        # All labels should be either 0 or 1
        assert np.all((new_labels == 0) | (new_labels == 1))

    def test_contiguous_labels_after_merge(self):
        from common.ml.clustering.training import merge_small_clusters

        rng = np.random.RandomState(7)
        X = rng.randn(60, 2)
        # Cluster 0: 30, Cluster 1: 2 (tiny), Cluster 2: 28
        labels = np.array([0] * 30 + [1] * 2 + [2] * 28)
        centroids = np.array([
            X[:30].mean(axis=0),
            X[30:32].mean(axis=0),
            X[32:].mean(axis=0),
        ])

        new_labels, _ = merge_small_clusters(X, labels, centroids, min_size=5)

        unique_labels = sorted(set(new_labels))
        # Labels should be contiguous 0..K-1
        assert unique_labels == list(range(len(unique_labels)))


# ============================================================================
# 3. Labeling
# ============================================================================

class TestAssignClusterLabels:
    """Tests for assign_cluster_labels()."""

    @pytest.fixture()
    def volume_thresholds(self):
        return {
            "very_high": 5000.0,
            "high": 1000.0,
            "low": 100.0,
            "very_low": 20.0,
        }

    @pytest.fixture()
    def cv_thresholds(self):
        return {
            "very_steady": 0.2,
            "steady": 0.4,
            "volatile": 0.8,
            "very_volatile": 1.2,
        }

    @pytest.fixture()
    def labeling_config(self):
        return {
            "seasonality_threshold": 0.3,
            "seasonality_r2_threshold": 0.25,
            "periodicity_threshold": 0.25,
            "zero_demand_threshold": 0.15,
            "adi_threshold": 1.5,
            "trend_r2_threshold": 0.25,
            "cagr_growing": 5.0,
            "cagr_declining": -5.0,
            "recency_ratio_high": 1.2,
            "recency_ratio_low": 0.8,
        }

    def _make_centroids(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_basic_labeling_three_clusters(
        self, volume_thresholds, cv_thresholds, labeling_config
    ):
        from common.ml.clustering.labeling import assign_cluster_labels

        centroids = self._make_centroids([
            # High-volume seasonal
            {"cluster_id": 0, "mean_demand": 2000, "cv_demand": 0.5,
             "seasonal_amplitude": 0.6, "seasonal_r2": 0.4,
             "periodicity_strength": 0.1, "zero_demand_pct": 0.0,
             "adi": 1.0, "trend_r2": 0.1, "cagr": 2.0,
             "recency_ratio": 1.0},
            # Low-volume intermittent
            {"cluster_id": 1, "mean_demand": 50, "cv_demand": 1.5,
             "seasonal_amplitude": 0.1, "seasonal_r2": 0.05,
             "periodicity_strength": 0.05, "zero_demand_pct": 0.5,
             "adi": 3.0, "trend_r2": 0.05, "cagr": 0.0,
             "recency_ratio": 0.9},
            # Very-high volume growing
            {"cluster_id": 2, "mean_demand": 6000, "cv_demand": 0.3,
             "seasonal_amplitude": 0.1, "seasonal_r2": 0.05,
             "periodicity_strength": 0.1, "zero_demand_pct": 0.0,
             "adi": 1.0, "trend_r2": 0.5, "cagr": 10.0,
             "recency_ratio": 1.1},
        ])

        labels = assign_cluster_labels(
            centroids, volume_thresholds, cv_thresholds, labeling_config
        )

        assert len(labels) == 3
        # Cluster 0 should include "seasonal" and "high_volume"
        assert "high_volume" in labels[0]
        assert "seasonal" in labels[0]
        # Cluster 1 should be intermittent
        assert "intermittent" in labels[1]
        # Cluster 2 should include "growing"
        assert "growing" in labels[2]

    def test_all_labels_unique(
        self, volume_thresholds, cv_thresholds, labeling_config
    ):
        """Even with similar centroids, disambiguation should produce unique labels."""
        from common.ml.clustering.labeling import assign_cluster_labels

        # Two clusters that would both be "medium_volume_steady"
        centroids = self._make_centroids([
            {"cluster_id": 0, "mean_demand": 300, "cv_demand": 0.3,
             "seasonal_amplitude": 0.1, "seasonal_r2": 0.1,
             "periodicity_strength": 0.05, "zero_demand_pct": 0.0,
             "adi": 1.0, "trend_r2": 0.1, "cagr": 1.0,
             "recency_ratio": 1.0},
            {"cluster_id": 1, "mean_demand": 350, "cv_demand": 0.35,
             "seasonal_amplitude": 0.1, "seasonal_r2": 0.1,
             "periodicity_strength": 0.05, "zero_demand_pct": 0.0,
             "adi": 1.0, "trend_r2": 0.1, "cagr": 1.0,
             "recency_ratio": 1.0},
        ])

        labels = assign_cluster_labels(
            centroids, volume_thresholds, cv_thresholds, labeling_config
        )

        label_values = list(labels.values())
        assert len(label_values) == len(set(label_values)), "Labels should be unique"

    def test_volume_tiers(
        self, volume_thresholds, cv_thresholds, labeling_config
    ):
        """Check that volume tiers map correctly to labels."""
        from common.ml.clustering.labeling import assign_cluster_labels

        centroids = self._make_centroids([
            # very_high
            {"cluster_id": 0, "mean_demand": 6000, "cv_demand": 0.3,
             "seasonal_amplitude": 0.0, "seasonal_r2": 0.0,
             "periodicity_strength": 0.0, "zero_demand_pct": 0.0,
             "adi": 1.0, "trend_r2": 0.0, "cagr": 0.0,
             "recency_ratio": 1.0},
            # very_low
            {"cluster_id": 1, "mean_demand": 10, "cv_demand": 0.3,
             "seasonal_amplitude": 0.0, "seasonal_r2": 0.0,
             "periodicity_strength": 0.0, "zero_demand_pct": 0.0,
             "adi": 1.0, "trend_r2": 0.0, "cagr": 0.0,
             "recency_ratio": 1.0},
        ])

        labels = assign_cluster_labels(
            centroids, volume_thresholds, cv_thresholds, labeling_config
        )

        assert "very_high_volume" in labels[0]
        assert "very_low_volume" in labels[1]

    def test_missing_volume_threshold_raises(
        self, cv_thresholds, labeling_config
    ):
        from common.ml.clustering.labeling import assign_cluster_labels

        bad_thresholds = {"high": 1000}  # missing very_high, low, very_low
        centroids = self._make_centroids([
            {"cluster_id": 0, "mean_demand": 100, "cv_demand": 0.3,
             "seasonal_amplitude": 0.0, "seasonal_r2": 0.0,
             "periodicity_strength": 0.0, "zero_demand_pct": 0.0,
             "adi": 1.0, "trend_r2": 0.0, "cagr": 0.0,
             "recency_ratio": 1.0},
        ])

        with pytest.raises(ValueError, match="missing required keys"):
            assign_cluster_labels(
                centroids, bad_thresholds, cv_thresholds, labeling_config
            )

    def test_periodic_cluster(
        self, volume_thresholds, cv_thresholds, labeling_config
    ):
        from common.ml.clustering.labeling import assign_cluster_labels

        centroids = self._make_centroids([
            {"cluster_id": 0, "mean_demand": 500, "cv_demand": 0.5,
             "seasonal_amplitude": 0.1, "seasonal_r2": 0.1,
             "periodicity_strength": 0.5, "zero_demand_pct": 0.0,
             "adi": 1.0, "trend_r2": 0.1, "cagr": 0.0,
             "recency_ratio": 1.0},
        ])

        labels = assign_cluster_labels(
            centroids, volume_thresholds, cv_thresholds, labeling_config
        )

        assert "periodic" in labels[0]


# ============================================================================
# 4. Scenario
# ============================================================================

class TestGenerateScenarioId:
    """Tests for generate_scenario_id()."""

    def test_format_matches_pattern(self):
        from common.ml.clustering.scenario import generate_scenario_id

        sid = generate_scenario_id()
        assert re.match(r"^sc_\d{8}_\d{6}_[0-9a-f]{4}$", sid)

    def test_uniqueness(self):
        from common.ml.clustering.scenario import generate_scenario_id

        ids = {generate_scenario_id() for _ in range(50)}
        assert len(ids) == 50


class TestLoadConfigDefaults:
    """Tests for _load_config_defaults()."""

    def test_returns_promoted_experiment_config(self):
        from common.ml.clustering.scenario import _load_config_defaults

        mock_row = (
            # feature_params (dict)
            {"time_window_months": 48, "min_months_history": 18},
            # model_params (dict)
            {"k_range": [5, 12], "min_cluster_size_pct": 3.0, "use_pca": True, "pca_components": 8},
            # label_params (dict)
            {"volume_high": 0.80, "volume_low": 0.20,
             "cv_steady": 0.35, "cv_volatile": 0.9,
             "seasonality_threshold": 0.4, "zero_demand_threshold": 0.2},
        )

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = mock_row
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("common.ml.clustering.scenario.get_db_params", return_value={}), \
             patch("psycopg.connect", return_value=mock_conn):
            cfg = _load_config_defaults()

        assert cfg["time_window_months"] == 48
        assert cfg["min_months_history"] == 18
        assert cfg["k_range"] == [5, 12]
        assert cfg["min_cluster_size_pct"] == 3.0
        assert cfg["use_pca"] is True
        assert cfg["pca_components"] == 8
        assert cfg["labeling"]["volume_thresholds"]["high"] == 0.80
        assert cfg["labeling"]["cv_thresholds"]["volatile"] == 0.9
        assert cfg["labeling"]["seasonality_threshold"] == 0.4

    def test_returns_fallback_when_no_promoted_experiment(self):
        from common.ml.clustering.scenario import _load_config_defaults, _FALLBACK_DEFAULTS

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None  # no promoted row
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("common.ml.clustering.scenario.get_db_params", return_value={}), \
             patch("psycopg.connect", return_value=mock_conn):
            cfg = _load_config_defaults()

        assert cfg["time_window_months"] == _FALLBACK_DEFAULTS["time_window_months"]
        assert cfg["k_range"] == _FALLBACK_DEFAULTS["k_range"]
        assert cfg["min_cluster_size_pct"] == _FALLBACK_DEFAULTS["min_cluster_size_pct"]

    def test_returns_fallback_when_db_unavailable(self):
        from common.ml.clustering.scenario import _load_config_defaults, _FALLBACK_DEFAULTS

        with patch("common.ml.clustering.scenario.get_db_params", side_effect=Exception("no DB")):
            cfg = _load_config_defaults()

        assert cfg["time_window_months"] == _FALLBACK_DEFAULTS["time_window_months"]
        assert cfg["k_range"] == _FALLBACK_DEFAULTS["k_range"]

    def test_fallback_deep_copy_isolation(self):
        """Mutating the returned config must NOT affect _FALLBACK_DEFAULTS."""
        from common.ml.clustering.scenario import _load_config_defaults, _FALLBACK_DEFAULTS

        with patch("common.ml.clustering.scenario.get_db_params", side_effect=Exception("no DB")):
            cfg = _load_config_defaults()

        original_k = list(_FALLBACK_DEFAULTS["k_range"])
        cfg["k_range"] = [99, 99]
        assert _FALLBACK_DEFAULTS["k_range"] == original_k

    def test_json_string_params_parsed(self):
        """When DB returns JSON strings instead of dicts, they should be parsed."""
        from common.ml.clustering.scenario import _load_config_defaults

        mock_row = (
            json.dumps({"time_window_months": 30}),
            json.dumps({"k_range": [3, 8]}),
            json.dumps({"volume_high": 0.70}),
        )

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = mock_row
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("common.ml.clustering.scenario.get_db_params", return_value={}), \
             patch("psycopg.connect", return_value=mock_conn):
            cfg = _load_config_defaults()

        assert cfg["time_window_months"] == 30
        assert cfg["k_range"] == [3, 8]


class TestGetScenarioResult:
    """Tests for get_scenario_result()."""

    def test_returns_none_for_invalid_id(self):
        from common.ml.clustering.scenario import get_scenario_result

        result = get_scenario_result("bad_id")
        assert result is None

    def test_returns_none_when_no_result_file(self, tmp_path):
        from common.ml.clustering.scenario import get_scenario_result

        # Valid format but dir doesn't exist
        with patch("common.ml.clustering.scenario.SCENARIO_BASE", tmp_path):
            result = get_scenario_result("sc_20260101_120000_abcd")
        assert result is None

    def test_returns_data_when_result_exists(self, tmp_path):
        from common.ml.clustering.scenario import get_scenario_result

        scenario_id = "sc_20260101_120000_abcd"
        scenario_dir = tmp_path / scenario_id
        scenario_dir.mkdir()
        payload = {"result": {"optimal_k": 5, "silhouette_score": 0.42}}
        (scenario_dir / "scenario_result.json").write_text(json.dumps(payload))

        with patch("common.ml.clustering.scenario.SCENARIO_BASE", tmp_path):
            result = get_scenario_result(scenario_id)

        assert result is not None
        assert result["result"]["optimal_k"] == 5

    def test_path_traversal_rejected(self):
        from common.ml.clustering.scenario import get_scenario_result

        # This doesn't match the regex so it's rejected
        result = get_scenario_result("../../../etc/passwd")
        assert result is None


class TestSafeScenarioDir:
    """Tests for _safe_scenario_dir() path validation."""

    def test_valid_id_returns_path(self):
        from common.ml.clustering.scenario import _safe_scenario_dir

        path = _safe_scenario_dir("sc_20260101_120000_abcd")
        assert "sc_20260101_120000_abcd" in str(path)

    def test_invalid_format_raises(self):
        from common.ml.clustering.scenario import _safe_scenario_dir

        with pytest.raises(ValueError, match="Invalid scenario_id"):
            _safe_scenario_dir("not-valid")

    def test_empty_string_raises(self):
        from common.ml.clustering.scenario import _safe_scenario_dir

        with pytest.raises(ValueError, match="Invalid scenario_id"):
            _safe_scenario_dir("")
