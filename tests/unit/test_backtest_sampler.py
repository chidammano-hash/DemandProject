"""Unit tests for common/ml/backtest_sampler.py — stratified sampling,
allocation methods, accuracy deviation estimation, and validation."""

import math
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_strata():
    """Build sample strata dict."""
    return {
        0: {
            "cluster_label": "0",
            "n_dfus": 20000,
            "mean_demand": 250.0,
            "cv": 0.8,
            "zero_pct": 0.1,
            "sku_cks": [f"DFU_{i}" for i in range(20000)],
        },
        1: {
            "cluster_label": "1",
            "n_dfus": 15000,
            "mean_demand": 50.0,
            "cv": 1.5,
            "zero_pct": 0.35,
            "sku_cks": [f"DFU_{i}" for i in range(20000, 35000)],
        },
        2: {
            "cluster_label": "2",
            "n_dfus": 15000,
            "mean_demand": 120.0,
            "cv": 0.9,
            "zero_pct": 0.2,
            "sku_cks": [f"DFU_{i}" for i in range(35000, 50000)],
        },
    }


def _mock_conn_with_strata():
    """Build a mock connection that returns cluster strata rows."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        ("0", 20000, 250.0, 0.8, 0.1, [f"DFU_{i}" for i in range(20000)]),
        ("1", 15000, 50.0, 1.5, 0.35, [f"DFU_{i}" for i in range(20000, 35000)]),
        ("2", 15000, 120.0, 0.9, 0.2, [f"DFU_{i}" for i in range(35000, 50000)]),
    ]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return mock_conn


# ---------------------------------------------------------------------------
# _allocate_proportional
# ---------------------------------------------------------------------------


class TestAllocateProportional:
    def test_proportional_distributes_by_cluster_size(self):
        from common.ml.backtest_sampler import _allocate_proportional
        strata = _mock_strata()
        allocation = _allocate_proportional(strata, target_n=5000, min_per_cluster=10)
        # Cluster 0 has 20000/50000 = 40% => ~2000
        assert allocation[0] == 2000
        # Cluster 1 has 15000/50000 = 30% => ~1500
        assert allocation[1] == 1500
        # Cluster 2 has 15000/50000 = 30% => ~1500
        assert allocation[2] == 1500
        # Total should be close to target
        assert sum(allocation.values()) == 5000

    def test_proportional_enforces_min_per_cluster(self):
        from common.ml.backtest_sampler import _allocate_proportional
        strata = {
            0: {"n_dfus": 10000, "sku_cks": []},
            1: {"n_dfus": 5, "sku_cks": []},  # Very small cluster
        }
        allocation = _allocate_proportional(strata, target_n=100, min_per_cluster=10)
        assert allocation[1] >= 10

    def test_proportional_empty_strata(self):
        from common.ml.backtest_sampler import _allocate_proportional
        assert _allocate_proportional({}, target_n=5000, min_per_cluster=10) == {}


# ---------------------------------------------------------------------------
# _allocate_equal
# ---------------------------------------------------------------------------


class TestAllocateEqual:
    def test_equal_distributes_evenly(self):
        from common.ml.backtest_sampler import _allocate_equal
        strata = _mock_strata()
        allocation = _allocate_equal(strata, target_n=3000, min_per_cluster=10)
        # 3 clusters, 3000 / 3 = 1000 each
        assert allocation[0] == 1000
        assert allocation[1] == 1000
        assert allocation[2] == 1000

    def test_equal_enforces_min_per_cluster(self):
        from common.ml.backtest_sampler import _allocate_equal
        strata = _mock_strata()
        allocation = _allocate_equal(strata, target_n=9, min_per_cluster=10)
        # target_n/3 = 3, but min_per_cluster = 10
        for cid in allocation:
            assert allocation[cid] >= 10

    def test_equal_empty_strata(self):
        from common.ml.backtest_sampler import _allocate_equal
        assert _allocate_equal({}, target_n=3000, min_per_cluster=10) == {}


# ---------------------------------------------------------------------------
# _allocate_sqrt
# ---------------------------------------------------------------------------


class TestAllocateSqrt:
    def test_sqrt_distributes_by_sqrt_size(self):
        from common.ml.backtest_sampler import _allocate_sqrt
        strata = _mock_strata()
        allocation = _allocate_sqrt(strata, target_n=5000, min_per_cluster=10)
        # Sqrt allocation weights: sqrt(20000) ≈ 141.4, sqrt(15000) ≈ 122.5
        # Total sqrt ≈ 386.4
        # Cluster 0: 5000 * (141.4/386.4) ≈ 1830
        # Clusters 1,2: 5000 * (122.5/386.4) ≈ 1585 each
        assert allocation[0] > allocation[1]  # Bigger cluster gets more
        assert allocation[1] == allocation[2]  # Same-size clusters get same
        assert all(v >= 10 for v in allocation.values())

    def test_sqrt_empty_strata(self):
        from common.ml.backtest_sampler import _allocate_sqrt
        assert _allocate_sqrt({}, target_n=5000, min_per_cluster=10) == {}


# ---------------------------------------------------------------------------
# stratified_sample
# ---------------------------------------------------------------------------


class TestStratifiedSample:
    def test_proportional_returns_correct_count(self):
        from common.ml.backtest_sampler import stratified_sample
        mock_conn = _mock_conn_with_strata()
        with patch("common.ml.backtest_sampler._load_sampling_config", return_value={
            "seed": 42, "min_per_cluster": 10,
        }):
            sampled = stratified_sample(mock_conn, target_n=5000, method="proportional", seed=42)
        # Should sample close to 5000 DFUs
        assert 4500 <= len(sampled) <= 5500
        # All should be strings
        assert all(isinstance(s, str) for s in sampled)

    def test_equal_returns_dfus(self):
        from common.ml.backtest_sampler import stratified_sample
        mock_conn = _mock_conn_with_strata()
        with patch("common.ml.backtest_sampler._load_sampling_config", return_value={
            "seed": 42, "min_per_cluster": 10,
        }):
            sampled = stratified_sample(mock_conn, target_n=3000, method="equal", seed=42)
        assert len(sampled) > 0

    def test_sqrt_returns_dfus(self):
        from common.ml.backtest_sampler import stratified_sample
        mock_conn = _mock_conn_with_strata()
        with patch("common.ml.backtest_sampler._load_sampling_config", return_value={
            "seed": 42, "min_per_cluster": 10,
        }):
            sampled = stratified_sample(mock_conn, target_n=3000, method="sqrt", seed=42)
        assert len(sampled) > 0

    def test_unknown_method_raises(self):
        from common.ml.backtest_sampler import stratified_sample
        mock_conn = MagicMock()
        with patch("common.ml.backtest_sampler._load_sampling_config", return_value={
            "seed": 42, "min_per_cluster": 10,
        }):
            with pytest.raises(ValueError, match="Unknown sampling method"):
                stratified_sample(mock_conn, target_n=5000, method="bogus")

    def test_empty_strata_returns_empty(self):
        from common.ml.backtest_sampler import stratified_sample
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        with patch("common.ml.backtest_sampler._load_sampling_config", return_value={
            "seed": 42, "min_per_cluster": 10,
        }):
            sampled = stratified_sample(mock_conn, target_n=5000, method="proportional")
        assert sampled == []

    def test_reproducibility_with_seed(self):
        from common.ml.backtest_sampler import stratified_sample
        mock_conn_1 = _mock_conn_with_strata()
        mock_conn_2 = _mock_conn_with_strata()
        cfg = {"seed": 42, "min_per_cluster": 10}
        with patch("common.ml.backtest_sampler._load_sampling_config", return_value=cfg):
            sample_1 = stratified_sample(mock_conn_1, target_n=1000, method="proportional", seed=42)
        with patch("common.ml.backtest_sampler._load_sampling_config", return_value=cfg):
            sample_2 = stratified_sample(mock_conn_2, target_n=1000, method="proportional", seed=42)
        assert sample_1 == sample_2


# ---------------------------------------------------------------------------
# estimate_accuracy_deviation
# ---------------------------------------------------------------------------


class TestEstimateAccuracyDeviation:
    def test_basic_deviation(self):
        from common.ml.backtest_sampler import estimate_accuracy_deviation
        dev = estimate_accuracy_deviation(sample_size=5000, total_size=50000, n_clusters=10)
        assert dev > 0
        # Should be a reasonable percentage: < 5pp for 10% sample
        assert dev < 5.0

    def test_larger_sample_smaller_deviation(self):
        from common.ml.backtest_sampler import estimate_accuracy_deviation
        dev_small = estimate_accuracy_deviation(1000, 50000, 10)
        dev_large = estimate_accuracy_deviation(10000, 50000, 10)
        assert dev_large < dev_small

    def test_full_sample_zero_deviation(self):
        from common.ml.backtest_sampler import estimate_accuracy_deviation
        dev = estimate_accuracy_deviation(50000, 50000, 10)
        assert dev == 0.0

    def test_zero_sample_zero_deviation(self):
        from common.ml.backtest_sampler import estimate_accuracy_deviation
        assert estimate_accuracy_deviation(0, 50000, 10) == 0.0

    def test_zero_population_zero_deviation(self):
        from common.ml.backtest_sampler import estimate_accuracy_deviation
        assert estimate_accuracy_deviation(5000, 0, 10) == 0.0

    def test_single_cluster_no_efficiency(self):
        from common.ml.backtest_sampler import estimate_accuracy_deviation
        dev_1 = estimate_accuracy_deviation(5000, 50000, 1)
        dev_10 = estimate_accuracy_deviation(5000, 50000, 10)
        # Single cluster has strat_efficiency=1.0 (less reduction)
        assert dev_1 > dev_10


# ---------------------------------------------------------------------------
# validate_sample_representativeness
# ---------------------------------------------------------------------------


class TestValidateSampleRepresentativeness:
    def test_representative_sample(self):
        from common.ml.backtest_sampler import validate_sample_representativeness
        full = {0: {"mean_demand": 100.0, "cv": 0.5}}
        sampled = {0: {"mean_demand": 101.0, "cv": 0.51}}  # Very close
        with patch("common.ml.backtest_sampler._load_sampling_config", return_value={
            "representativeness_threshold": 0.05,
        }):
            results = validate_sample_representativeness(sampled, full)
        assert results[0]["is_representative"] is True
        assert results[0]["ks_stat"] < 0.05

    def test_unrepresentative_sample(self):
        from common.ml.backtest_sampler import validate_sample_representativeness
        full = {0: {"mean_demand": 100.0, "cv": 0.5}}
        sampled = {0: {"mean_demand": 200.0, "cv": 1.0}}  # Very different
        with patch("common.ml.backtest_sampler._load_sampling_config", return_value={
            "representativeness_threshold": 0.05,
        }):
            results = validate_sample_representativeness(sampled, full)
        assert results[0]["is_representative"] is False
        assert results[0]["ks_stat"] >= 0.05

    def test_missing_cluster_in_sample(self):
        from common.ml.backtest_sampler import validate_sample_representativeness
        full = {0: {"mean_demand": 100.0, "cv": 0.5}}
        sampled = {}  # Cluster 0 not in sample
        with patch("common.ml.backtest_sampler._load_sampling_config", return_value={
            "representativeness_threshold": 0.05,
        }):
            results = validate_sample_representativeness(sampled, full)
        assert results[0]["is_representative"] is False
        assert results[0]["ks_stat"] == 1.0

    def test_zero_mean_demand_in_full(self):
        from common.ml.backtest_sampler import validate_sample_representativeness
        full = {0: {"mean_demand": 0.0, "cv": 0.0}}
        sampled = {0: {"mean_demand": 0.0, "cv": 0.0}}
        with patch("common.ml.backtest_sampler._load_sampling_config", return_value={
            "representativeness_threshold": 0.05,
        }):
            results = validate_sample_representativeness(sampled, full)
        assert results[0]["mean_ratio"] == 0.0

    def test_multiple_clusters(self):
        from common.ml.backtest_sampler import validate_sample_representativeness
        full = {
            0: {"mean_demand": 100.0, "cv": 0.5},
            1: {"mean_demand": 50.0, "cv": 1.2},
        }
        sampled = {
            0: {"mean_demand": 102.0, "cv": 0.51},
            1: {"mean_demand": 48.0, "cv": 1.15},
        }
        with patch("common.ml.backtest_sampler._load_sampling_config", return_value={
            "representativeness_threshold": 0.05,
        }):
            results = validate_sample_representativeness(sampled, full)
        assert 0 in results
        assert 1 in results
        for cid in results:
            assert "ks_stat" in results[cid]
            assert "mean_ratio" in results[cid]
            assert "cv_ratio" in results[cid]
            assert "is_representative" in results[cid]


# ---------------------------------------------------------------------------
# filter_backtest_data
# ---------------------------------------------------------------------------


class TestFilterBacktestData:
    def test_filters_to_sampled_skus(self):
        from common.ml.backtest_sampler import filter_backtest_data
        sales = pd.DataFrame({
            "sku_ck": ["A", "A", "B", "B", "C", "C"],
            "qty": [100, 200, 300, 400, 500, 600],
        })
        attrs = pd.DataFrame({
            "sku_ck": ["A", "B", "C"],
            "ml_cluster": [0, 1, 2],
        })
        f_sales, f_attrs = filter_backtest_data(sales, attrs, ["A", "C"])
        assert len(f_sales) == 4  # A (2 rows) + C (2 rows)
        assert len(f_attrs) == 2  # A + C
        assert set(f_sales["sku_ck"].unique()) == {"A", "C"}
        assert set(f_attrs["sku_ck"].unique()) == {"A", "C"}

    def test_empty_sample(self):
        from common.ml.backtest_sampler import filter_backtest_data
        sales = pd.DataFrame({"sku_ck": ["A", "B"], "qty": [100, 200]})
        attrs = pd.DataFrame({"sku_ck": ["A", "B"], "ml_cluster": [0, 1]})
        f_sales, f_attrs = filter_backtest_data(sales, attrs, [])
        assert len(f_sales) == 0
        assert len(f_attrs) == 0

    def test_all_sampled(self):
        from common.ml.backtest_sampler import filter_backtest_data
        sales = pd.DataFrame({"sku_ck": ["A", "B"], "qty": [100, 200]})
        attrs = pd.DataFrame({"sku_ck": ["A", "B"], "ml_cluster": [0, 1]})
        f_sales, f_attrs = filter_backtest_data(sales, attrs, ["A", "B"])
        assert len(f_sales) == 2
        assert len(f_attrs) == 2

    def test_nonexistent_skus_ignored(self):
        from common.ml.backtest_sampler import filter_backtest_data
        sales = pd.DataFrame({"sku_ck": ["A", "B"], "qty": [100, 200]})
        attrs = pd.DataFrame({"sku_ck": ["A", "B"], "ml_cluster": [0, 1]})
        f_sales, f_attrs = filter_backtest_data(sales, attrs, ["A", "Z"])
        assert len(f_sales) == 1  # Only A
        assert len(f_attrs) == 1


# ---------------------------------------------------------------------------
# compute_cluster_strata
# ---------------------------------------------------------------------------


class TestComputeClusterStrata:
    def test_returns_strata_dict(self):
        from common.ml.backtest_sampler import compute_cluster_strata
        mock_conn = _mock_conn_with_strata()
        strata = compute_cluster_strata(mock_conn)
        assert len(strata) == 3
        assert strata[0]["n_dfus"] == 20000
        assert strata[0]["cluster_label"] == "0"
        assert len(strata[0]["sku_cks"]) == 20000
        assert strata[1]["n_dfus"] == 15000
        assert strata[2]["n_dfus"] == 15000

    def test_empty_result(self):
        from common.ml.backtest_sampler import compute_cluster_strata
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        strata = compute_cluster_strata(mock_conn)
        assert strata == {}

    def test_null_handling(self):
        from common.ml.backtest_sampler import compute_cluster_strata
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("0", 100, None, None, None, ["DFU_1"]),
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        strata = compute_cluster_strata(mock_conn)
        assert strata[0]["mean_demand"] == 0.0
        assert strata[0]["cv"] == 0.0
        assert strata[0]["zero_pct"] == 0.0
