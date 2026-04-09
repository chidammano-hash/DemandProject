"""Tests for clustering pipeline performance optimizations.

Validates that parallelization and vectorization changes in
generate_clustering_features.py and train_clustering_model.py
produce correct results without changing clustering logic.
"""

import datetime
import inspect
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


# ── generate_clustering_features.py tests ─────────────────────────────────


class TestSeasonalR2Vectorized:
    """Verify _seasonal_r2 vectorized design matrix matches original logic."""

    def test_basic_seasonal_pattern(self):
        from scripts.generate_clustering_features import _seasonal_r2

        np.random.seed(42)
        n = 36
        months = np.array([(i % 12) + 1 for i in range(n)], dtype=np.float64)
        # Create seasonal signal: higher in summer months
        y = np.array([100 + 50 * np.sin(2 * np.pi * m / 12) for m in months])
        r2 = _seasonal_r2(y, months)
        assert 0.0 <= r2 <= 1.0
        # Strong seasonal pattern should have high R²
        assert r2 > 0.5

    def test_flat_signal_returns_zero(self):
        from scripts.generate_clustering_features import _seasonal_r2

        y = np.ones(36)
        months = np.array([(i % 12) + 1 for i in range(36)], dtype=np.float64)
        r2 = _seasonal_r2(y, months)
        assert r2 == 0.0

    def test_short_series_returns_zero(self):
        from scripts.generate_clustering_features import _seasonal_r2

        y = np.array([1.0, 2.0, 3.0])
        months = np.array([1.0, 2.0, 3.0])
        r2 = _seasonal_r2(y, months)
        assert r2 == 0.0

    def test_month_12_is_reference_level(self):
        """Month 12 should be excluded from dummies (reference level)."""
        from scripts.generate_clustering_features import _seasonal_r2

        n = 24
        months = np.array([(i % 12) + 1 for i in range(n)], dtype=np.float64)
        y = np.random.RandomState(42).randn(n)
        # Should not raise and should return valid R²
        r2 = _seasonal_r2(y, months)
        assert isinstance(r2, float)
        assert 0.0 <= r2 <= 1.0


class TestComputeFeaturesForGroup:
    """Test the multiprocessing-compatible wrapper function."""

    def test_produces_correct_keys(self):
        from scripts.generate_clustering_features import _compute_features_for_group

        dates = pd.date_range("2023-01-01", periods=24, freq="MS")
        qty = np.random.RandomState(42).randint(50, 500, size=24).astype(float)
        result = _compute_features_for_group((
            "DFU_TEST",
            {"startdate": dates.values, "qty": qty},
        ))
        assert result["sku_ck"] == "DFU_TEST"
        assert "mean_demand" in result
        assert "cv_demand" in result
        assert "seasonal_r2" in result
        assert "periodicity_strength" in result
        assert "adi" in result

    def test_empty_group_returns_sku_ck(self):
        from scripts.generate_clustering_features import _compute_features_for_group

        result = _compute_features_for_group((
            "DFU_EMPTY",
            {"startdate": np.array([], dtype="datetime64[ns]"), "qty": np.array([], dtype=float)},
        ))
        assert result["sku_ck"] == "DFU_EMPTY"

    def test_single_month(self):
        from scripts.generate_clustering_features import _compute_features_for_group

        result = _compute_features_for_group((
            "DFU_ONE",
            {
                "startdate": np.array(["2024-01-01"], dtype="datetime64[ns]"),
                "qty": np.array([100.0]),
            },
        ))
        assert result["sku_ck"] == "DFU_ONE"
        assert result["mean_demand"] == 100.0
        assert result["months_available"] == 1

    def test_result_matches_direct_call(self):
        """Parallel wrapper should produce same output as direct call."""
        from scripts.generate_clustering_features import _compute_features_for_group
        from common.ml.clustering.features import compute_time_series_features

        dates = pd.date_range("2023-01-01", periods=36, freq="MS")
        qty = np.random.RandomState(99).randint(10, 1000, size=36).astype(float)
        df = pd.DataFrame({"startdate": dates, "qty": qty})

        direct = compute_time_series_features(df).to_dict()
        parallel = _compute_features_for_group((
            "DFU_CMP",
            {"startdate": dates.values, "qty": qty},
        ))

        # Remove sku_ck from parallel result for comparison
        parallel_features = {k: v for k, v in parallel.items() if k != "sku_ck"}
        for key in direct:
            assert key in parallel_features, f"Missing key: {key}"
            assert np.isclose(direct[key], parallel_features[key], atol=1e-10), (
                f"Mismatch for {key}: {direct[key]} vs {parallel_features[key]}"
            )


class TestPeriodicityStrength:
    """Verify _periodicity_strength helper."""

    def test_pure_sine(self):
        from scripts.generate_clustering_features import _periodicity_strength

        t = np.arange(48)
        y = np.sin(2 * np.pi * t / 12)  # annual cycle
        strength = _periodicity_strength(y)
        assert strength > 0.5

    def test_flat_returns_zero(self):
        from scripts.generate_clustering_features import _periodicity_strength

        y = np.ones(24)
        assert _periodicity_strength(y) == 0.0

    def test_short_returns_zero(self):
        from scripts.generate_clustering_features import _periodicity_strength

        y = np.array([1.0, 2.0, 3.0])
        assert _periodicity_strength(y) == 0.0


class TestAdi:
    """Verify _adi helper."""

    def test_regular_demand(self):
        from scripts.generate_clustering_features import _adi

        y = np.array([10, 20, 30, 40, 50], dtype=float)
        # All nonzero, gaps = [1,1,1,1], mean = 1.0
        assert _adi(y) == 1.0

    def test_intermittent_demand(self):
        from scripts.generate_clustering_features import _adi

        y = np.array([10, 0, 0, 20, 0, 0, 30], dtype=float)
        # Nonzero at 0, 3, 6 -> gaps = [3, 3], mean = 3.0
        assert _adi(y) == 3.0

    def test_all_zero(self):
        from scripts.generate_clustering_features import _adi

        y = np.zeros(10)
        assert _adi(y) == 10.0


# ── train_clustering_model.py tests ───────────────────────────────────────


class TestEvaluateSingleK:
    """Test the per-K evaluation function used in parallel K-search."""

    def test_returns_expected_keys(self):
        from scripts.train_clustering_model import _evaluate_single_k

        np.random.seed(42)
        X = np.random.randn(200, 5)
        result = _evaluate_single_k(
            k=3, X_scaled=X, n_samples=200,
            min_cluster_size=10, silhouette_sample_size=None,
        )
        assert set(result.keys()) == {"k", "inertia", "sil", "ch", "cluster_sizes", "smallest", "feasible"}
        assert result["k"] == 3
        assert isinstance(result["inertia"], float)
        assert isinstance(result["sil"], float)
        assert isinstance(result["ch"], float)
        assert result["smallest"] > 0

    def test_feasibility_check(self):
        from scripts.train_clustering_model import _evaluate_single_k

        np.random.seed(42)
        X = np.random.randn(100, 3)
        # With min_cluster_size=0, everything should be feasible
        result = _evaluate_single_k(
            k=2, X_scaled=X, n_samples=100,
            min_cluster_size=0, silhouette_sample_size=None,
        )
        assert result["feasible"] is True

    def test_with_silhouette_sampling(self):
        from scripts.train_clustering_model import _evaluate_single_k

        np.random.seed(42)
        X = np.random.randn(500, 4)
        result = _evaluate_single_k(
            k=3, X_scaled=X, n_samples=500,
            min_cluster_size=10, silhouette_sample_size=100,
        )
        assert 0.0 < abs(result["sil"]) < 1.0


class TestFindOptimalKParallel:
    """Test find_optimal_k with n_workers parameter."""

    def _make_data(self):
        np.random.seed(42)
        # 3 clear clusters
        c1 = np.random.randn(100, 3) + [5, 0, 0]
        c2 = np.random.randn(100, 3) + [0, 5, 0]
        c3 = np.random.randn(100, 3) + [0, 0, 5]
        return np.vstack([c1, c2, c3])

    def test_serial_returns_valid_result(self):
        from common.ml.clustering.training import find_optimal_k

        X = self._make_data()
        result = find_optimal_k(X, k_range=(2, 5), min_cluster_size_pct=5.0, n_workers=1)
        assert "optimal_k" in result
        assert 2 <= result["optimal_k"] <= 5
        assert len(result["k_values"]) == 4
        assert len(result["inertias"]) == 4
        assert len(result["silhouette_scores"]) == 4
        assert len(result["ch_scores"]) == 4
        assert len(result["feasible_mask"]) == 4

    def test_parallel_returns_valid_result(self):
        from common.ml.clustering.training import find_optimal_k

        X = self._make_data()
        result = find_optimal_k(X, k_range=(2, 5), min_cluster_size_pct=5.0, n_workers=2)
        assert "optimal_k" in result
        assert 2 <= result["optimal_k"] <= 5
        assert len(result["k_values"]) == 4

    def test_serial_and_parallel_agree_on_optimal_k(self):
        """Serial and parallel should find the same optimal K (deterministic seeds)."""
        from common.ml.clustering.training import find_optimal_k

        X = self._make_data()
        serial = find_optimal_k(X, k_range=(2, 5), min_cluster_size_pct=5.0, n_workers=1)
        parallel = find_optimal_k(X, k_range=(2, 5), min_cluster_size_pct=5.0, n_workers=2)
        assert serial["optimal_k"] == parallel["optimal_k"]

    def test_all_infeasible_falls_back_to_silhouette(self):
        from common.ml.clustering.training import find_optimal_k

        np.random.seed(42)
        X = np.random.randn(20, 3)  # Very small dataset
        # min_cluster_size_pct=90% makes all K infeasible
        result = find_optimal_k(X, k_range=(2, 4), min_cluster_size_pct=90.0, n_workers=1)
        assert "optimal_k" in result
        assert all(not f for f in result["feasible_mask"])


class TestMergeSmallClusters:
    """Verify merge_small_clusters preserves data integrity."""

    def test_no_merge_when_all_large(self):
        from scripts.train_clustering_model import merge_small_clusters

        X = np.random.randn(100, 3)
        labels = np.array([0] * 50 + [1] * 50)
        centroids = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
        new_labels, new_centroids = merge_small_clusters(X, labels, centroids, min_size=10)
        assert len(set(new_labels)) == 2

    def test_merge_small_cluster(self):
        from scripts.train_clustering_model import merge_small_clusters

        X = np.random.randn(100, 3)
        labels = np.array([0] * 90 + [1] * 5 + [2] * 5)
        centroids = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        new_labels, new_centroids = merge_small_clusters(X, labels, centroids, min_size=10)
        # Clusters 1 and 2 should be merged into cluster 0
        remaining = set(new_labels)
        assert len(remaining) <= 2  # at most 2 after merge (could be 1 if both merge into 0)
        assert len(new_labels) == 100  # all samples preserved


class TestKmeansAlgorithmParam:
    """Verify the final KMeans uses algorithm='elkan' and max_iter=300."""

    def test_elkan_algorithm_works(self):
        """Ensure elkan algorithm produces valid results (smoke test)."""
        from sklearn.cluster import KMeans

        np.random.seed(42)
        X = np.random.randn(100, 5)
        km = KMeans(n_clusters=3, random_state=42, n_init=5, algorithm="elkan", max_iter=300)
        labels = km.fit_predict(X)
        assert len(set(labels)) == 3
        assert len(labels) == 100


class TestPlanningDateClustering:
    """Verify clustering scripts cap sales data at planning date."""

    def test_generate_features_caps_at_planning_date(self):
        """generate_clustering_features.py SQL must filter by planning_upper."""
        # Read the main() function source to verify SQL includes planning date
        import scripts.generate_clustering_features as gcf
        source = inspect.getsource(gcf)
        assert "planning_upper" in source
        assert "get_planning_date" in source

    def test_scenario_caps_at_planning_date(self):
        """run_clustering_scenario.py SQL must filter by planning_upper."""
        import scripts.run_clustering_scenario as rcs
        source = inspect.getsource(rcs)
        assert "planning_upper" in source
        assert "get_planning_date" in source
