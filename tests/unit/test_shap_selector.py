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
