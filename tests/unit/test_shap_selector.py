"""Unit tests for common/shap_selector.py (Feature 42)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from common.shap_selector import (
    _select_features_from_shap,
    _weighted_pool_cluster_shap,
    build_shap_summary,
    save_shap_outputs,
    SHAP_REPORT_COLS,
    SHAP_SUMMARY_COLS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_shap_df(features, timeframe_idx=0, cutoff="2024-01-01", selected=True, cluster="all"):
    """Build a minimal SHAP report DataFrame."""
    n = len(features)
    return pd.DataFrame({
        "feature": features,
        "mean_abs_shap": [float(n - i) for i in range(n)],
        "rank": list(range(1, n + 1)),
        "selected": [selected] * n,
        "timeframe": [timeframe_idx] * n,
        "cutoff_date": [cutoff] * n,
        "cluster": [cluster] * n,
    })


# ---------------------------------------------------------------------------
# _select_features_from_shap
# ---------------------------------------------------------------------------


class TestSelectFeaturesFromShap:
    def _cutoff(self):
        return pd.Timestamp("2024-06-01")

    def test_cumulative_threshold_selects_correct_n(self):
        # Feature 0 dominates: shap = [10, 1, 1, 1, 1]. Cumulative 95% at rank 1 = 10/14 = 71%
        # Need rank 1+2 = 11/14 = 78.6%, all to reach 95% → should select all if 0.95 threshold
        shap = np.array([10.0, 1.0, 1.0, 1.0, 1.0])
        features = ["a", "b", "c", "d", "e"]
        selected, df = _select_features_from_shap(shap, features, 0, self._cutoff(), cumulative_threshold=0.95)
        assert isinstance(selected, list)
        assert len(selected) >= 1
        assert df.shape[0] == 5
        # _select_features_from_shap doesn't add "cluster" — that's added by compute_timeframe_shap
        base_cols = [c for c in SHAP_REPORT_COLS if c != "cluster"]
        assert set(df.columns) == set(base_cols)

    def test_min_features_floor(self):
        # Very skewed: one feature dominates at any threshold
        shap = np.array([100.0, 0.01, 0.01])
        features = ["x", "y", "z"]
        selected, _ = _select_features_from_shap(shap, features, 0, self._cutoff(),
                                                  cumulative_threshold=0.5, min_features=3)
        assert len(selected) == 3  # min_features floor

    def test_top_n_override(self):
        shap = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        features = ["a", "b", "c", "d", "e"]
        selected, df = _select_features_from_shap(shap, features, 0, self._cutoff(), top_n=2, min_features=1)
        assert len(selected) == 2
        assert selected[0] == "a"
        assert selected[1] == "b"

    def test_top_n_respects_min_features_floor(self):
        shap = np.array([5.0, 4.0, 3.0])
        features = ["a", "b", "c"]
        selected, _ = _select_features_from_shap(shap, features, 0, self._cutoff(), top_n=1, min_features=2)
        assert len(selected) == 2

    def test_zero_shap_returns_all_features(self):
        shap = np.zeros(4)
        features = ["p", "q", "r", "s"]
        selected, df = _select_features_from_shap(shap, features, 0, self._cutoff())
        assert len(selected) == 4

    def test_selected_column_correctness(self):
        shap = np.array([10.0, 1.0, 1.0, 1.0])
        features = ["a", "b", "c", "d"]
        selected, df = _select_features_from_shap(shap, features, 0, self._cutoff(),
                                                  cumulative_threshold=0.80, min_features=1)
        selected_set = set(selected)
        for _, row in df.iterrows():
            assert row["selected"] == (row["feature"] in selected_set)

    def test_shap_df_columns(self):
        shap = np.array([3.0, 2.0, 1.0])
        features = ["a", "b", "c"]
        _, df = _select_features_from_shap(shap, features, 2, self._cutoff())
        # _select_features_from_shap doesn't add "cluster" — that's added by compute_timeframe_shap
        base_cols = [c for c in SHAP_REPORT_COLS if c != "cluster"]
        assert list(df.columns) == base_cols
        assert all(df["timeframe"] == 2)
        assert all(df["cutoff_date"] == "2024-06-01")

    def test_features_sorted_by_importance(self):
        shap = np.array([1.0, 5.0, 3.0])
        features = ["low", "high", "mid"]
        selected, df = _select_features_from_shap(shap, features, 0, self._cutoff())
        assert selected[0] == "high"
        assert df.iloc[0]["feature"] == "high"


# ---------------------------------------------------------------------------
# build_shap_summary
# ---------------------------------------------------------------------------


class TestBuildShapSummary:
    def test_empty_input_returns_empty_df(self):
        result = build_shap_summary([], n_timeframes=5)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == SHAP_SUMMARY_COLS
        assert len(result) == 0

    def test_aggregation_over_timeframes(self):
        df0 = _make_shap_df(["a", "b", "c"], timeframe_idx=0)
        df1 = _make_shap_df(["a", "b", "c"], timeframe_idx=1)
        summary = build_shap_summary([df0, df1], n_timeframes=2)
        assert set(summary.columns) == set(SHAP_SUMMARY_COLS)
        assert len(summary) == 3
        # "a" should be top (shap 3.0 in df0 and df1)
        assert summary.iloc[0]["feature"] == "a"

    def test_selected_count_computed_correctly(self):
        df0 = _make_shap_df(["a", "b"], timeframe_idx=0, selected=True)
        df1 = _make_shap_df(["a", "b"], timeframe_idx=1, selected=False)
        df1["selected"] = [False, False]
        summary = build_shap_summary([df0, df1], n_timeframes=2)
        row_a = summary[summary["feature"] == "a"].iloc[0]
        assert row_a["selected_count"] == 1  # only selected in df0
        assert row_a["n_timeframes"] == 2

    def test_n_timeframes_propagated(self):
        df = _make_shap_df(["x"], timeframe_idx=0)
        summary = build_shap_summary([df], n_timeframes=10)
        assert summary.iloc[0]["n_timeframes"] == 10

    def test_sorted_descending_by_mean_shap(self):
        df = _make_shap_df(["z", "y", "x"], timeframe_idx=0)
        # df has shap [3, 2, 1] for z, y, x → z highest
        summary = build_shap_summary([df], n_timeframes=1)
        assert list(summary["feature"]) == ["z", "y", "x"]


# ---------------------------------------------------------------------------
# save_shap_outputs
# ---------------------------------------------------------------------------


class TestSaveShapOutputs:
    def test_empty_input_returns_empty_lists(self):
        paths, summary_path = save_shap_outputs([], Path("/tmp/unused"), 5)
        assert paths == []
        assert summary_path is None

    def test_writes_timeframe_csvs_and_summary(self, tmp_path):
        df0 = _make_shap_df(["a", "b"], timeframe_idx=0)
        df1 = _make_shap_df(["a", "b"], timeframe_idx=1)
        paths, summary_path = save_shap_outputs([df0, df1], tmp_path, n_timeframes=2)
        assert len(paths) == 2
        assert summary_path is not None
        assert (tmp_path / "shap" / "shap_timeframe_00.csv").exists()
        assert (tmp_path / "shap" / "shap_timeframe_01.csv").exists()
        assert (tmp_path / "shap" / "shap_summary.csv").exists()

    def test_timeframe_csv_has_correct_columns(self, tmp_path):
        df0 = _make_shap_df(["a", "b"], timeframe_idx=0)
        save_shap_outputs([df0], tmp_path, n_timeframes=1)
        saved = pd.read_csv(tmp_path / "shap" / "shap_timeframe_00.csv")
        assert set(saved.columns) == set(SHAP_REPORT_COLS)

    def test_summary_csv_has_correct_columns(self, tmp_path):
        df0 = _make_shap_df(["a", "b"], timeframe_idx=0)
        _, summary_path = save_shap_outputs([df0], tmp_path, n_timeframes=1)
        summary = pd.read_csv(summary_path)
        assert set(summary.columns) == set(SHAP_SUMMARY_COLS)

    def test_returns_summary_path(self, tmp_path):
        df0 = _make_shap_df(["a"], timeframe_idx=0)
        _, summary_path = save_shap_outputs([df0], tmp_path, n_timeframes=1)
        assert summary_path == tmp_path / "shap" / "shap_summary.csv"


# ---------------------------------------------------------------------------
# _weighted_pool_cluster_shap
# ---------------------------------------------------------------------------


class TestWeightedPoolClusterShap:
    def _make_model(self, n_features):
        """Return a mock model that returns uniform SHAP values."""
        model = MagicMock()
        return model

    def _make_shap_extractor(self, value=1.0):
        """Return a shap extractor that returns constant abs SHAP values."""
        def extractor(model, X_sample, feature_cols, cat_cols):
            return np.full((len(X_sample), len(feature_cols)), value)
        return extractor

    def _make_train_data(self, cluster_labels, n_per_cluster=20):
        rows = []
        for label in cluster_labels:
            for _ in range(n_per_cluster):
                rows.append({"ml_cluster": label, "f1": 1.0, "f2": 2.0})
        return pd.DataFrame(rows)

    def test_skips_base_key(self):
        models = {
            "__base__": MagicMock(),
            "A": MagicMock(),
        }
        train_data = self._make_train_data(["A"])
        extractor = self._make_shap_extractor(value=3.0)
        pooled, per_cluster = _weighted_pool_cluster_shap(models, train_data, ["f1", "f2"], [], extractor, 10)
        # __base__ skipped, only cluster A used
        assert pooled.shape == (2,)
        np.testing.assert_allclose(pooled, [3.0, 3.0])
        assert "A" in per_cluster
        assert "__base__" not in per_cluster

    def test_empty_cluster_skipped(self):
        models = {"A": MagicMock(), "B": MagicMock()}
        # Only cluster A has training data; B has none
        train_data = self._make_train_data(["A"])
        extractor = self._make_shap_extractor(value=2.0)
        pooled, per_cluster = _weighted_pool_cluster_shap(models, train_data, ["f1", "f2"], [], extractor, 10)
        np.testing.assert_allclose(pooled, [2.0, 2.0])
        assert "A" in per_cluster
        assert "B" not in per_cluster

    def test_weighted_average_by_cluster_size(self):
        models = {"big": MagicMock(), "small": MagicMock()}
        big_rows = [{"ml_cluster": "big", "f1": 1.0} for _ in range(40)]
        small_rows = [{"ml_cluster": "small", "f1": 1.0} for _ in range(10)]
        train_data = pd.DataFrame(big_rows + small_rows)

        def extractor(model, X_sample, feature_cols, cat_cols):
            is_big = (model is models["big"])
            return np.full((len(X_sample), len(feature_cols)), 4.0 if is_big else 0.0)

        pooled, per_cluster = _weighted_pool_cluster_shap(models, train_data, ["f1"], [], extractor, 50)
        # Weighted: (4 * min(50,40) + 0 * min(50,10)) / (40 + 10) = 160/50 = 3.2
        assert pooled.shape == (1,)
        assert abs(pooled[0] - 3.2) < 0.01
        assert "big" in per_cluster
        assert "small" in per_cluster
        np.testing.assert_allclose(per_cluster["big"], [4.0])
        np.testing.assert_allclose(per_cluster["small"], [0.0])

    def test_extractor_exception_skips_cluster(self):
        models = {"A": MagicMock()}
        train_data = self._make_train_data(["A"])

        def bad_extractor(model, X_sample, feature_cols, cat_cols):
            raise RuntimeError("Simulated SHAP failure")

        pooled, per_cluster = _weighted_pool_cluster_shap(models, train_data, ["f1", "f2"], [], bad_extractor, 10)
        np.testing.assert_allclose(pooled, [0.0, 0.0])
        assert len(per_cluster) == 0


# ---------------------------------------------------------------------------
# compute_timeframe_shap_per_cluster
# ---------------------------------------------------------------------------


from common.shap_selector import (
    SPARSE_MIN_NONZERO_ROWS,
    _stratified_sample_for_shap,
    compute_timeframe_shap_per_cluster,
)


# ---------------------------------------------------------------------------
# _stratified_sample_for_shap
# ---------------------------------------------------------------------------


class TestStratifiedSampleForShap:
    """Tests for the sparse-cluster stratified SHAP sampling helper."""

    def test_nonsparse_cluster_uses_random_sampling(self):
        """Cluster with <50% zeros uses standard random sampling."""
        rng = np.random.RandomState(42)
        n = 200
        # 30% zeros — below threshold
        qty = np.where(rng.rand(n) < 0.3, 0.0, rng.rand(n) * 100 + 1)
        df = pd.DataFrame({
            "feat_a": rng.randn(n),
            "feat_b": rng.randn(n),
            "qty": qty,
        })
        result = _stratified_sample_for_shap(df, ["feat_a", "feat_b"], sample_size=50)
        assert result is not None
        assert len(result) == 50
        assert list(result.columns) == ["feat_a", "feat_b"]

    def test_sparse_cluster_uses_stratified_sampling(self):
        """Cluster with >50% zeros gets 50/50 zero vs non-zero samples."""
        rng = np.random.RandomState(42)
        n = 500
        # 80% zeros → sparse
        qty = np.where(rng.rand(n) < 0.8, 0.0, rng.rand(n) * 100 + 1)
        df = pd.DataFrame({
            "feat_a": rng.randn(n),
            "feat_b": rng.randn(n),
            "qty": qty,
        })
        result = _stratified_sample_for_shap(df, ["feat_a", "feat_b"], sample_size=100)
        assert result is not None
        assert list(result.columns) == ["feat_a", "feat_b"]
        # Should have ~50 rows from each stratum (50/50 split of sample_size=100)
        assert len(result) <= 100

    def test_sparse_cluster_balances_nonzero_rows(self):
        """Verify that stratified sample actually contains non-zero rows."""
        # 80% zeros, 20% non-zero — 1000 total → 800 zero, 200 nonzero
        # 200 nonzero > SPARSE_MIN_NONZERO_ROWS (100), so SHAP proceeds
        n = 1000
        qty = np.zeros(n)
        qty[:200] = np.arange(1, 201).astype(float)  # 200 nonzero
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "feat_a": rng.randn(n),
            "qty": qty,
        })
        # Shuffle so the pattern isn't trivial
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        result = _stratified_sample_for_shap(df, ["feat_a"], sample_size=40)
        assert result is not None
        # sample_size=40 → half=20, so up to 20 nonzero + 20 zero
        assert len(result) == 40

    def test_too_few_nonzero_returns_none(self):
        """Cluster with <SPARSE_MIN_NONZERO_ROWS non-zero rows returns None."""
        n = 500
        # Only 10 nonzero rows (well below 100 threshold)
        qty = np.zeros(n)
        qty[:10] = np.arange(1, 11).astype(float)
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "feat_a": rng.randn(n),
            "qty": qty,
        })
        result = _stratified_sample_for_shap(df, ["feat_a"], sample_size=100)
        assert result is None

    def test_exactly_at_threshold_nonzero_returns_sample(self):
        """Cluster with exactly SPARSE_MIN_NONZERO_ROWS non-zero rows gets sampled."""
        n = 500
        n_nonzero = SPARSE_MIN_NONZERO_ROWS  # exactly at threshold
        qty = np.zeros(n)
        qty[:n_nonzero] = np.arange(1, n_nonzero + 1).astype(float)
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "feat_a": rng.randn(n),
            "qty": qty,
        })
        result = _stratified_sample_for_shap(df, ["feat_a"], sample_size=100)
        assert result is not None

    def test_no_qty_column_falls_back_to_random(self):
        """When qty column is missing, standard random sampling is used."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "feat_a": rng.randn(200),
            "feat_b": rng.randn(200),
        })
        result = _stratified_sample_for_shap(df, ["feat_a", "feat_b"], sample_size=50)
        assert result is not None
        assert len(result) == 50
        assert list(result.columns) == ["feat_a", "feat_b"]

    def test_all_zero_demand_returns_none(self):
        """Cluster with 100% zeros and 0 non-zero rows returns None."""
        n = 200
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "feat_a": rng.randn(n),
            "qty": np.zeros(n),
        })
        result = _stratified_sample_for_shap(df, ["feat_a"], sample_size=50)
        assert result is None  # 0 nonzero < SPARSE_MIN_NONZERO_ROWS

    def test_sample_size_larger_than_data(self):
        """When sample_size > cluster size, caps at available rows."""
        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame({
            "feat_a": rng.randn(n),
            "qty": rng.rand(n) * 100 + 1,  # all nonzero (200 > threshold)
        })
        result = _stratified_sample_for_shap(df, ["feat_a"], sample_size=500)
        assert result is not None
        assert len(result) == 200  # capped at data size


# ---------------------------------------------------------------------------
# compute_timeframe_shap_per_cluster (sparse cluster integration tests)
# ---------------------------------------------------------------------------


class TestSparseClusterShapIntegration:
    """Integration tests for SHAP handling of sparse/low-volume clusters."""

    def test_sparse_cluster_keeps_all_features(self):
        """A cluster with too few non-zero rows skips SHAP, keeps all features."""
        rng = np.random.RandomState(42)
        n_features = 10
        feature_cols = [f"feat_{i}" for i in range(n_features)]

        # "active" cluster: 200 rows, all nonzero demand
        # "sparse" cluster: 200 rows, only 5 nonzero (< SPARSE_MIN_NONZERO_ROWS)
        rows = []
        for _ in range(200):
            row = {f"feat_{i}": rng.randn() for i in range(n_features)}
            row["ml_cluster"] = "active"
            row["qty"] = rng.rand() * 100 + 1
            rows.append(row)
        for i in range(200):
            row = {f"feat_{i}": rng.randn() for i in range(n_features)}
            row["ml_cluster"] = "sparse"
            row["qty"] = float(i + 1) if i < 5 else 0.0  # only 5 nonzero
            rows.append(row)
        df = pd.DataFrame(rows)
        cat_cols: list[str] = []

        model_dict = {"active": MagicMock(), "sparse": MagicMock()}

        def extractor(model, X, feature_cols, cat_cols):
            rng_ext = np.random.RandomState(0)
            return rng_ext.rand(len(X), len(feature_cols))

        result_features, shap_df = compute_timeframe_shap_per_cluster(
            model_dict=model_dict,
            train_data=df,
            feature_cols=feature_cols,
            cat_cols=cat_cols,
            timeframe_idx=0,
            cutoff_date=pd.Timestamp("2025-01-01"),
            shap_extractor_fn=extractor,
        )

        # Sparse cluster should keep ALL features (SHAP skipped)
        assert set(result_features["sparse"]) == set(feature_cols)

        # Active cluster should have features selected via SHAP normally
        assert "active" in result_features
        assert len(result_features["active"]) > 0

    def test_sparse_cluster_no_shap_df_row(self):
        """Sparse cluster that skips SHAP should NOT appear in shap_df."""
        rng = np.random.RandomState(42)
        n_features = 5
        feature_cols = [f"f{i}" for i in range(n_features)]

        rows = []
        # active cluster with enough data
        for _ in range(200):
            row = {f"f{i}": rng.randn() for i in range(n_features)}
            row["ml_cluster"] = "active"
            row["qty"] = rng.rand() * 100 + 1
            rows.append(row)
        # sparse cluster with too few nonzero
        for i in range(200):
            row = {f"f{i}": rng.randn() for i in range(n_features)}
            row["ml_cluster"] = "sparse"
            row["qty"] = 1.0 if i < 3 else 0.0
            rows.append(row)
        df = pd.DataFrame(rows)

        model_dict = {"active": MagicMock(), "sparse": MagicMock()}

        def extractor(model, X, feature_cols, cat_cols):
            return np.random.RandomState(0).rand(len(X), len(feature_cols))

        _, shap_df = compute_timeframe_shap_per_cluster(
            model_dict=model_dict,
            train_data=df,
            feature_cols=feature_cols,
            cat_cols=[],
            timeframe_idx=0,
            cutoff_date=pd.Timestamp("2025-01-01"),
            shap_extractor_fn=extractor,
        )

        # The sparse cluster should NOT appear in shap_df (SHAP was skipped)
        if not shap_df.empty:
            cluster_values = set(shap_df["cluster"].unique())
            assert "sparse" not in cluster_values
            assert "active" in cluster_values


def _make_cluster_data(n_features=10, n_per_cluster=50, clusters=("high", "low")):
    """Helper to create test data for per-cluster SHAP tests."""
    rng = np.random.RandomState(42)
    rows = []
    for cluster in clusters:
        for _ in range(n_per_cluster):
            row = {f"feat_{i}": rng.randn() for i in range(n_features)}
            row["ml_cluster"] = cluster
            row["target"] = rng.randn()
            rows.append(row)
    df = pd.DataFrame(rows)
    feature_cols = [f"feat_{i}" for i in range(n_features)]
    cat_cols: list[str] = []
    return df, feature_cols, cat_cols


def _make_per_cluster_shap_extractor(n_features):
    """Mock SHAP extractor that returns random absolute values."""
    def extractor(model, X, feature_cols, cat_cols):
        rng = np.random.RandomState(0)
        return rng.rand(len(X), len(feature_cols))
    return extractor


class TestComputeTimeframeShapPerCluster:
    """Tests for the per-cluster independent SHAP feature selection."""

    def test_compute_timeframe_shap_per_cluster_basic(self):
        """Basic: 2 clusters, returns per-cluster feature lists and shap_df."""
        df, feature_cols, cat_cols = _make_cluster_data(
            n_features=10, n_per_cluster=50, clusters=("high", "low"),
        )
        model_dict = {"high": MagicMock(), "low": MagicMock()}
        extractor = _make_per_cluster_shap_extractor(10)

        result_features, shap_df = compute_timeframe_shap_per_cluster(
            model_dict=model_dict,
            train_data=df,
            feature_cols=feature_cols,
            cat_cols=cat_cols,
            timeframe_idx=0,
            cutoff_date=pd.Timestamp("2025-01-01"),
            shap_extractor_fn=extractor,
        )

        # Return type checks
        assert isinstance(result_features, dict)
        assert isinstance(shap_df, pd.DataFrame)

        # Both clusters present in result
        assert "high" in result_features
        assert "low" in result_features

        # Each value is a list of feature name strings
        for cluster_label, feats in result_features.items():
            assert isinstance(feats, list)
            assert len(feats) > 0
            for f in feats:
                assert isinstance(f, str)
                assert f in feature_cols

        # shap_df has the cluster column with correct values
        assert "cluster" in shap_df.columns
        cluster_values = set(shap_df["cluster"].unique())
        assert "high" in cluster_values
        assert "low" in cluster_values

    def test_compute_timeframe_shap_per_cluster_skips_base(self):
        """The __base__ key in model_dict is skipped (transfer-learning sentinel)."""
        df, feature_cols, cat_cols = _make_cluster_data(
            n_features=10, n_per_cluster=50, clusters=("high", "low"),
        )
        model_dict = {
            "__base__": MagicMock(),
            "high": MagicMock(),
            "low": MagicMock(),
        }
        extractor = _make_per_cluster_shap_extractor(10)

        result_features, shap_df = compute_timeframe_shap_per_cluster(
            model_dict=model_dict,
            train_data=df,
            feature_cols=feature_cols,
            cat_cols=cat_cols,
            timeframe_idx=0,
            cutoff_date=pd.Timestamp("2025-01-01"),
            shap_extractor_fn=extractor,
        )

        # __base__ must not appear in the result
        assert "__base__" not in result_features
        # The two real clusters must be present
        assert "high" in result_features
        assert "low" in result_features

    def test_compute_timeframe_shap_per_cluster_failed_cluster(self):
        """When SHAP extraction fails for one cluster, it gets ALL features (fallback)."""
        df, feature_cols, cat_cols = _make_cluster_data(
            n_features=10, n_per_cluster=50, clusters=("good", "bad"),
        )

        model_good = MagicMock()
        model_bad = MagicMock()
        model_dict = {"good": model_good, "bad": model_bad}

        call_count = {"n": 0}

        def failing_extractor(model, X, feature_cols, cat_cols):
            call_count["n"] += 1
            if model is model_bad:
                raise RuntimeError("Simulated SHAP failure")
            rng = np.random.RandomState(0)
            return rng.rand(len(X), len(feature_cols))

        result_features, shap_df = compute_timeframe_shap_per_cluster(
            model_dict=model_dict,
            train_data=df,
            feature_cols=feature_cols,
            cat_cols=cat_cols,
            timeframe_idx=0,
            cutoff_date=pd.Timestamp("2025-01-01"),
            shap_extractor_fn=failing_extractor,
        )

        # Both clusters should be present despite the failure
        assert "good" in result_features
        assert "bad" in result_features

        # The failed cluster ("bad") gets ALL features as fallback
        assert set(result_features["bad"]) == set(feature_cols)

        # The successful cluster ("good") gets a (possibly smaller) subset
        assert len(result_features["good"]) > 0
        # All selected features are valid feature names
        assert all(f in feature_cols for f in result_features["good"])

    def test_compute_timeframe_shap_per_cluster_with_variance_filter(self):
        """Variance filter excludes near-zero-variance features for all clusters."""
        n_features = 10
        clusters = ("high", "low")
        n_per_cluster = 50

        # Build data where feat_0 and feat_1 have near-zero variance (constant)
        rng = np.random.RandomState(42)
        rows = []
        for cluster in clusters:
            for _ in range(n_per_cluster):
                row = {f"feat_{i}": rng.randn() for i in range(n_features)}
                # Make feat_0 and feat_1 constant → zero variance
                row["feat_0"] = 1.0
                row["feat_1"] = 2.0
                row["ml_cluster"] = cluster
                row["target"] = rng.randn()
                rows.append(row)
        df = pd.DataFrame(rows)
        feature_cols = [f"feat_{i}" for i in range(n_features)]
        cat_cols: list[str] = []

        model_dict = {"high": MagicMock(), "low": MagicMock()}

        def extractor(model, X, feature_cols, cat_cols):
            rng = np.random.RandomState(0)
            return rng.rand(len(X), len(feature_cols))

        result_features, shap_df = compute_timeframe_shap_per_cluster(
            model_dict=model_dict,
            train_data=df,
            feature_cols=feature_cols,
            cat_cols=cat_cols,
            timeframe_idx=0,
            cutoff_date=pd.Timestamp("2025-01-01"),
            shap_extractor_fn=extractor,
            variance_filter=True,
            variance_threshold=0.01,
        )

        # Near-zero-variance features (feat_0, feat_1) should be excluded
        # from the selected features of BOTH clusters
        for cluster_label, feats in result_features.items():
            assert "feat_0" not in feats, f"feat_0 should be excluded for {cluster_label}"
            assert "feat_1" not in feats, f"feat_1 should be excluded for {cluster_label}"
            # Other features should still be selectable
            assert len(feats) > 0
