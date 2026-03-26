"""Tests for small-cluster naive fallback in the backtest pipeline.

Verifies that clusters with fewer than MIN_CLUSTER_ROWS use a seasonal
naive baseline (per-month historical mean) instead of zeroing predictions.
"""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from common.constants import MIN_CLUSTER_ROWS


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_train_df(
    cluster_label: str,
    n_rows: int,
    *,
    base_qty: float = 100.0,
    seasonal: bool = False,
) -> pd.DataFrame:
    """Build a training DataFrame for a single cluster.

    When seasonal=True, qty varies by calendar month (month * 10).
    """
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="MS")
    qty_values = []
    for d in dates:
        if seasonal:
            qty_values.append(base_qty + d.month * 10)
        else:
            qty_values.append(base_qty)
    return pd.DataFrame({
        "sku_ck": [f"SKU_{i:03d}" for i in range(n_rows)],
        "item_id": [f"ITEM_{i:03d}" for i in range(n_rows)],
        "customer_group": ["CG1"] * n_rows,
        "loc": ["L1"] * n_rows,
        "startdate": dates,
        "qty": qty_values,
        "ml_cluster": [cluster_label] * n_rows,
        "month": [d.month for d in dates],
        "region": ["R1"] * n_rows,
        "brand": ["BR1"] * n_rows,
        "abc_vol": ["A"] * n_rows,
    })


def _make_pred_df(
    cluster_label: str,
    n_rows: int,
    start_date: str = "2025-01-01",
) -> pd.DataFrame:
    """Build a prediction DataFrame for a single cluster."""
    dates = pd.date_range(start_date, periods=n_rows, freq="MS")
    return pd.DataFrame({
        "sku_ck": [f"PRED_{i:03d}" for i in range(n_rows)],
        "item_id": [f"PITEM_{i:03d}" for i in range(n_rows)],
        "customer_group": ["CG1"] * n_rows,
        "loc": ["L1"] * n_rows,
        "startdate": dates,
        "ml_cluster": [cluster_label] * n_rows,
        "month": [d.month for d in dates],
        "region": ["R1"] * n_rows,
        "brand": ["BR1"] * n_rows,
        "abc_vol": ["A"] * n_rows,
    })


# ── _compute_naive_fallback tests ────────────────────────────────────────────


class TestComputeNaiveFallback:
    """Tests for the _compute_naive_fallback helper function."""

    def test_fallback_uses_per_month_historical_mean(self):
        """Predictions should use the mean demand for the matching calendar month."""
        from scripts.run_backtest import _compute_naive_fallback

        # Training data with seasonal pattern: Jan=110, Feb=120, Mar=130, ...
        train = _make_train_df("small_cluster", 12, seasonal=True)
        # Predict for Jan, Feb, Mar
        pred = _make_pred_df("small_cluster", 3, start_date="2025-01-01")

        result = _compute_naive_fallback(train, pred)

        assert list(result.columns) == [
            "sku_ck", "item_id", "customer_group", "loc", "startdate", "basefcst_pref",
        ]
        # Jan mean = 100 + 1*10 = 110, Feb = 120, Mar = 130
        assert result["basefcst_pref"].iloc[0] == pytest.approx(110.0)
        assert result["basefcst_pref"].iloc[1] == pytest.approx(120.0)
        assert result["basefcst_pref"].iloc[2] == pytest.approx(130.0)

    def test_fallback_uses_overall_mean_for_missing_months(self):
        """If prediction month has no training history, use the overall mean."""
        from scripts.run_backtest import _compute_naive_fallback

        # Training data only for Jan, Feb, Mar (months 1-3)
        train = _make_train_df("small_cluster", 3, seasonal=True)
        # Predict for July (month 7) — not in training data
        pred = _make_pred_df("small_cluster", 1, start_date="2025-07-01")

        result = _compute_naive_fallback(train, pred)

        # Overall mean of training: (110 + 120 + 130) / 3 = 120
        assert result["basefcst_pref"].iloc[0] == pytest.approx(120.0)

    def test_fallback_zero_demand_history_returns_zero(self):
        """If all training demand is zero, fallback should legitimately be zero."""
        from scripts.run_backtest import _compute_naive_fallback

        train = _make_train_df("small_cluster", 12, base_qty=0.0)
        pred = _make_pred_df("small_cluster", 3)

        result = _compute_naive_fallback(train, pred)

        assert (result["basefcst_pref"] == 0.0).all()

    def test_fallback_empty_train_returns_zero(self):
        """If training data is empty, fallback should be zero."""
        from scripts.run_backtest import _compute_naive_fallback

        train = _make_train_df("small_cluster", 0)
        pred = _make_pred_df("small_cluster", 3)

        result = _compute_naive_fallback(train, pred)

        assert (result["basefcst_pref"] == 0.0).all()

    def test_fallback_negative_clipped_to_zero(self):
        """Negative historical means should be clipped to zero."""
        from scripts.run_backtest import _compute_naive_fallback

        train = _make_train_df("small_cluster", 3, base_qty=-50.0)
        pred = _make_pred_df("small_cluster", 3)

        result = _compute_naive_fallback(train, pred)

        assert (result["basefcst_pref"] >= 0.0).all()

    def test_fallback_has_correct_columns(self):
        """Result DataFrame must have exactly the expected columns."""
        from scripts.run_backtest import _compute_naive_fallback

        train = _make_train_df("small_cluster", 10, seasonal=True)
        pred = _make_pred_df("small_cluster", 5)

        result = _compute_naive_fallback(train, pred)

        expected_cols = ["sku_ck", "item_id", "customer_group", "loc", "startdate", "basefcst_pref"]
        assert list(result.columns) == expected_cols
        assert len(result) == 5


# ── _train_single_cluster fallback marker tests ─────────────────────────────


class TestTrainSingleClusterFallback:
    """Tests that _train_single_cluster returns the fallback marker for small clusters."""

    def _make_registry(self) -> dict:
        """Build a minimal mock registry for testing."""
        return {
            "needs_cat_dtype_cast": False,
            "constant_target_guard": True,
            "iter_param": "n_estimators",
            "fit_extras_per_cluster": lambda p, i: {},
        }

    def test_small_cluster_returns_fallback_marker(self):
        """Cluster with fewer than MIN_CLUSTER_ROWS should return 'fallback_needed'."""
        from scripts.run_backtest import _train_single_cluster

        small_n = MIN_CLUSTER_ROWS - 1  # 49 rows
        train = _make_train_df("tiny", small_n)
        pred = _make_pred_df("tiny", 3)

        cl, result, model, meta = _train_single_cluster(
            "tiny", 1, 1,
            train, pred,
            ["month"], [], {},
            model_name="lgbm",
            registry=self._make_registry(),
            model_class=MagicMock,
            lib_module=MagicMock(),
        )

        assert cl == "tiny"
        assert result is not None
        assert model is None
        assert meta == "fallback_needed"

    def test_cluster_at_min_rows_is_not_fallback(self):
        """Cluster with exactly MIN_CLUSTER_ROWS should train normally (not fallback)."""
        from scripts.run_backtest import _train_single_cluster

        n = MIN_CLUSTER_ROWS  # 50 rows
        train = _make_train_df("normal", n, base_qty=100.0, seasonal=True)
        pred = _make_pred_df("normal", 3)

        # 20% of 50 = 10 val rows, 3 pred rows => predict called twice
        n_val = max(1, int(n * 0.20))  # 10
        mock_model_instance = MagicMock()
        mock_model_instance.predict.side_effect = [
            np.array([100.0] * 3),       # X_pred (3 rows)
            np.array([100.0] * n_val),    # X_val (10 rows)
        ]
        mock_model_class = MagicMock(return_value=mock_model_instance)

        registry = self._make_registry()

        with patch("scripts.run_backtest.fit_model"):
            with patch("scripts.run_backtest.get_best_iteration", return_value=100):
                with patch("scripts.run_backtest.compute_cluster_demand_stats", return_value={
                    "mean_demand": 100.0, "cv_demand": 0.1,
                    "zero_demand_pct": 0.0, "seasonal_amplitude": 0.0,
                }):
                    with patch("scripts.run_backtest.resolve_cluster_params", return_value=({}, "default")):
                        cl, result, model, meta = _train_single_cluster(
                            "normal", 1, 1,
                            train, pred,
                            ["month"], [], {"n_estimators": 100},
                            model_name="lgbm",
                            registry=registry,
                            model_class=mock_model_class,
                            lib_module=MagicMock(),
                        )

        assert cl == "normal"
        assert result is not None
        assert model is not None
        assert isinstance(meta, dict)
        assert meta != "fallback_needed"

    def test_empty_pred_returns_none(self):
        """Cluster with no prediction rows should return None result."""
        from scripts.run_backtest import _train_single_cluster

        train = _make_train_df("empty_pred", 10)
        pred = _make_pred_df("empty_pred", 0)

        cl, result, model, meta = _train_single_cluster(
            "empty_pred", 1, 1,
            train, pred,
            ["month"], [], {},
            model_name="lgbm",
            registry=self._make_registry(),
            model_class=MagicMock,
            lib_module=MagicMock(),
        )

        assert cl == "empty_pred"
        assert result is None
        assert model is None
        assert meta is None


# ── Integration: train_and_predict_per_cluster with fallback ─────────────────


class TestPerClusterWithFallback:
    """Integration tests that train_and_predict_per_cluster applies naive fallback."""

    def _make_mixed_data(self):
        """Create train/pred DataFrames with one large and one small cluster."""
        # Large cluster: 60 rows (above MIN_CLUSTER_ROWS)
        large_train = _make_train_df("big_cluster", 60, base_qty=200.0, seasonal=True)
        large_pred = _make_pred_df("big_cluster", 3)

        # Small cluster: 10 rows (below MIN_CLUSTER_ROWS)
        small_train = _make_train_df("tiny_cluster", 10, base_qty=50.0, seasonal=True)
        small_pred = _make_pred_df("tiny_cluster", 3, start_date="2025-01-01")

        train_df = pd.concat([large_train, small_train], ignore_index=True)
        pred_df = pd.concat([large_pred, small_pred], ignore_index=True)
        return train_df, pred_df

    def test_small_cluster_uses_naive_fallback_not_zero(self):
        """Small clusters should get naive fallback, not basefcst_pref=0."""
        from scripts.run_backtest import train_and_predict_per_cluster

        train_df, pred_df = self._make_mixed_data()

        # Mock model that returns correctly sized arrays for any input
        mock_model_instance = MagicMock()
        mock_model_instance.predict.side_effect = lambda x: np.full(len(x), 200.0)
        mock_model_class = MagicMock(return_value=mock_model_instance)

        registry = {
            "needs_cat_dtype_cast": False,
            "constant_target_guard": True,
            "iter_param": "n_estimators",
            "fit_extras_per_cluster": lambda p, i: {},
        }

        with patch("scripts.run_backtest.fit_model"):
            with patch("scripts.run_backtest.get_best_iteration", return_value=100):
                with patch("scripts.run_backtest.compute_cluster_demand_stats", return_value={
                    "mean_demand": 200.0, "cv_demand": 0.1,
                    "zero_demand_pct": 0.0, "seasonal_amplitude": 0.0,
                }):
                    with patch("scripts.run_backtest.resolve_cluster_params", return_value=({}, "default")):
                        result, models, meta = train_and_predict_per_cluster(
                            train_df, pred_df,
                            ["month"], [], {"n_estimators": 100},
                            model_name="lgbm",
                            registry=registry,
                            model_class=mock_model_class,
                            lib_module=MagicMock(),
                        )

        # Should have predictions for both clusters
        assert len(result) == 6  # 3 from big + 3 from tiny

        # Small cluster predictions should NOT be zero
        tiny_preds = result[result["sku_ck"].str.startswith("PRED_")].copy()
        # Get the tiny_cluster's predictions by merging with pred_df
        tiny_pred_skus = pred_df[pred_df["ml_cluster"] == "tiny_cluster"]["sku_ck"].tolist()
        tiny_results = result[result["sku_ck"].isin(tiny_pred_skus)]

        # The small cluster had seasonal demand (50 + month*10)
        # Jan=60, Feb=70, ... so predictions for Jan/Feb/Mar should be ~60/70/80
        assert (tiny_results["basefcst_pref"] > 0).all(), (
            "Small cluster predictions should not be zero — naive fallback expected"
        )

    def test_large_cluster_unaffected_by_fallback(self):
        """Clusters at or above MIN_CLUSTER_ROWS should train normally."""
        from scripts.run_backtest import train_and_predict_per_cluster

        train_df, pred_df = self._make_mixed_data()

        mock_model_instance = MagicMock()
        mock_model_instance.predict.side_effect = lambda x: np.full(len(x), 200.0)
        mock_model_class = MagicMock(return_value=mock_model_instance)

        registry = {
            "needs_cat_dtype_cast": False,
            "constant_target_guard": True,
            "iter_param": "n_estimators",
            "fit_extras_per_cluster": lambda p, i: {},
        }

        with patch("scripts.run_backtest.fit_model"):
            with patch("scripts.run_backtest.get_best_iteration", return_value=100):
                with patch("scripts.run_backtest.compute_cluster_demand_stats", return_value={
                    "mean_demand": 200.0, "cv_demand": 0.1,
                    "zero_demand_pct": 0.0, "seasonal_amplitude": 0.0,
                }):
                    with patch("scripts.run_backtest.resolve_cluster_params", return_value=({}, "default")):
                        result, models, meta = train_and_predict_per_cluster(
                            train_df, pred_df,
                            ["month"], [], {"n_estimators": 100},
                            model_name="lgbm",
                            registry=registry,
                            model_class=mock_model_class,
                            lib_module=MagicMock(),
                        )

        # Large cluster should have a model trained
        assert "big_cluster" in models
        # Small cluster should NOT have a model
        assert "tiny_cluster" not in models

    def test_all_clusters_above_min_no_fallback(self):
        """When all clusters are large enough, no fallback is triggered."""
        from scripts.run_backtest import train_and_predict_per_cluster

        train_a = _make_train_df("cluster_a", 60, base_qty=100.0, seasonal=True)
        train_b = _make_train_df("cluster_b", 55, base_qty=200.0, seasonal=True)
        pred_a = _make_pred_df("cluster_a", 2)
        pred_b = _make_pred_df("cluster_b", 2)

        train_df = pd.concat([train_a, train_b], ignore_index=True)
        pred_df = pd.concat([pred_a, pred_b], ignore_index=True)

        mock_model_instance = MagicMock()
        mock_model_instance.predict.side_effect = lambda x: np.full(len(x), 150.0)
        mock_model_class = MagicMock(return_value=mock_model_instance)

        registry = {
            "needs_cat_dtype_cast": False,
            "constant_target_guard": True,
            "iter_param": "n_estimators",
            "fit_extras_per_cluster": lambda p, i: {},
        }

        with patch("scripts.run_backtest.fit_model"):
            with patch("scripts.run_backtest.get_best_iteration", return_value=100):
                with patch("scripts.run_backtest.compute_cluster_demand_stats", return_value={
                    "mean_demand": 150.0, "cv_demand": 0.1,
                    "zero_demand_pct": 0.0, "seasonal_amplitude": 0.0,
                }):
                    with patch("scripts.run_backtest.resolve_cluster_params", return_value=({}, "default")):
                        result, models, meta = train_and_predict_per_cluster(
                            train_df, pred_df,
                            ["month"], [], {"n_estimators": 100},
                            model_name="lgbm",
                            registry=registry,
                            model_class=mock_model_class,
                            lib_module=MagicMock(),
                        )

        # Both clusters trained
        assert "cluster_a" in models
        assert "cluster_b" in models
        assert len(result) == 4
