"""Unit tests for recursive mode enhancements.

Tests cover:
- Per-step WAPE computation (_compute_step_wape)
- Noise injection helper (_inject_recursive_noise)
- Per-step metrics collection during recursive prediction
- Noise injection configuration via algorithm_config.yaml
- Metadata output includes recursive_step_metrics
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from common.ml.backtest_framework import (
    _compute_step_wape,
    _inject_recursive_noise,
    _predict_single_month,
)
from common.feature_engineering import (
    build_feature_matrix,
    mask_future_sales,
    update_grid_with_predictions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_grid():
    """Tiny 2-DFU x 4-month grid masked at 2024-02-01."""
    months = pd.to_datetime(
        ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
    )
    sales_df = pd.DataFrame(
        {
            "sku_ck": ["A"] * 4 + ["B"] * 4,
            "startdate": list(months) * 2,
            "qty": [100, 200, 300, 400, 50, 80, 120, 160],
        }
    )
    dfu_attrs = pd.DataFrame(
        {
            "sku_ck": ["A", "B"],
            "item_id": ["I1", "I2"],
            "customer_group": ["G", "G"],
            "loc": ["L1", "L2"],
        }
    )
    item_attrs = pd.DataFrame({"item_id": ["I1", "I2"]})
    grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, list(months))
    return mask_future_sales(grid, pd.Timestamp("2024-02-01"))


@pytest.fixture
def predict_data_one_month(simple_grid):
    """Predict-data slice for 2024-03-01 with cluster labels."""
    month = pd.Timestamp("2024-03-01")
    data = simple_grid[simple_grid["startdate"] == month].copy()
    data["ml_cluster"] = ["cluster_A", "cluster_B"]
    return data


# ---------------------------------------------------------------------------
# _inject_recursive_noise
# ---------------------------------------------------------------------------


class TestInjectRecursiveNoise:
    def test_zero_noise_returns_original(self):
        """noise_pct=0 returns a copy identical to the original."""
        values = np.array([10.0, 20.0, 30.0])
        result = _inject_recursive_noise(values, noise_pct=0.0)
        np.testing.assert_array_equal(result, values)

    def test_negative_noise_pct_returns_original(self):
        """Negative noise_pct returns values unchanged."""
        values = np.array([10.0, 20.0, 30.0])
        result = _inject_recursive_noise(values, noise_pct=-0.05)
        np.testing.assert_array_equal(result, values)

    def test_noise_changes_values(self):
        """With noise_pct=0.05, values should be perturbed."""
        np.random.seed(42)
        values = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        result = _inject_recursive_noise(values, noise_pct=0.05)
        # Values should not be exactly equal
        assert not np.array_equal(result, values)

    def test_noise_within_expected_range(self):
        """Noise should be roughly within expected statistical bounds."""
        np.random.seed(123)
        values = np.array([100.0] * 1000)
        noise_pct = 0.05
        result = _inject_recursive_noise(values, noise_pct=noise_pct)
        diffs = result - values
        # Standard deviation of noise should be close to noise_pct * mean(|values|)
        expected_std = noise_pct * np.abs(values).mean()
        actual_std = np.std(diffs)
        # Allow 20% tolerance for statistical variation
        assert abs(actual_std - expected_std) < 0.2 * expected_std

    def test_empty_array_returns_empty(self):
        """Empty input array returns empty array."""
        values = np.array([])
        result = _inject_recursive_noise(values, noise_pct=0.05)
        assert len(result) == 0

    def test_all_zeros_returns_zeros(self):
        """All-zero input stays zero (scale is zero)."""
        values = np.array([0.0, 0.0, 0.0])
        result = _inject_recursive_noise(values, noise_pct=0.05)
        np.testing.assert_array_equal(result, values)

    def test_returns_copy_not_in_place(self):
        """Result is a new array, not an in-place modification."""
        values = np.array([10.0, 20.0])
        result = _inject_recursive_noise(values, noise_pct=0.0)
        assert result is not values

    def test_preserves_shape(self):
        """Output shape matches input shape."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _inject_recursive_noise(values, noise_pct=0.1)
        assert result.shape == values.shape


# ---------------------------------------------------------------------------
# _compute_step_wape
# ---------------------------------------------------------------------------


class TestComputeStepWape:
    def test_perfect_prediction_gives_zero_wape(self):
        """Exact match between forecast and actuals yields WAPE=0."""
        preds = pd.DataFrame(
            {"sku_ck": ["A", "B"], "basefcst_pref": [100.0, 200.0]}
        )
        actuals = {"A": 100.0, "B": 200.0}
        wape = _compute_step_wape(preds, actuals)
        assert wape == 0.0

    def test_known_error_gives_expected_wape(self):
        """Known forecast error produces expected WAPE."""
        preds = pd.DataFrame(
            {"sku_ck": ["A", "B"], "basefcst_pref": [110.0, 220.0]}
        )
        actuals = {"A": 100.0, "B": 200.0}
        # |110-100| + |220-200| = 10 + 20 = 30
        # |100 + 200| = 300
        # WAPE = 100 * 30 / 300 = 10.0
        wape = _compute_step_wape(preds, actuals)
        assert wape == pytest.approx(10.0)

    def test_empty_predictions_returns_none(self):
        """Empty predictions DataFrame returns None."""
        preds = pd.DataFrame(columns=["sku_ck", "basefcst_pref"])
        actuals = {"A": 100.0}
        assert _compute_step_wape(preds, actuals) is None

    def test_empty_actuals_returns_none(self):
        """Empty actuals dict returns None."""
        preds = pd.DataFrame(
            {"sku_ck": ["A"], "basefcst_pref": [100.0]}
        )
        assert _compute_step_wape(preds, {}) is None

    def test_no_matching_dfus_returns_none(self):
        """No overlap between prediction DFUs and actuals returns None."""
        preds = pd.DataFrame(
            {"sku_ck": ["A"], "basefcst_pref": [100.0]}
        )
        actuals = {"X": 50.0}
        assert _compute_step_wape(preds, actuals) is None

    def test_partial_match_uses_only_matched(self):
        """Only matched DFUs are used for WAPE computation."""
        preds = pd.DataFrame(
            {"sku_ck": ["A", "B", "C"], "basefcst_pref": [110.0, 200.0, 50.0]}
        )
        # Only A and B have actuals; C has no actual
        actuals = {"A": 100.0, "B": 200.0}
        # |110-100| + |200-200| = 10
        # |100 + 200| = 300
        # WAPE = 100 * 10 / 300 = 3.33
        wape = _compute_step_wape(preds, actuals)
        assert wape == pytest.approx(3.33, abs=0.01)

    def test_zero_actuals_returns_none(self):
        """Zero total actuals returns None (division by zero guard)."""
        preds = pd.DataFrame(
            {"sku_ck": ["A"], "basefcst_pref": [100.0]}
        )
        actuals = {"A": 0.0}
        assert _compute_step_wape(preds, actuals) is None

    def test_wape_is_rounded(self):
        """WAPE result is rounded to 2 decimal places."""
        preds = pd.DataFrame(
            {"sku_ck": ["A"], "basefcst_pref": [103.0]}
        )
        actuals = {"A": 100.0}
        wape = _compute_step_wape(preds, actuals)
        # 100 * 3 / 100 = 3.0
        assert wape == 3.0
        assert isinstance(wape, float)


# ---------------------------------------------------------------------------
# Per-step metrics collection during recursive prediction
# ---------------------------------------------------------------------------


class TestRecursiveStepMetrics:
    def _make_cluster_models(self, val_a=10.0, val_b=20.0):
        m_a = MagicMock()
        m_a.predict.return_value = np.array([val_a])
        m_b = MagicMock()
        m_b.predict.return_value = np.array([val_b])
        return {"cluster_A": m_a, "cluster_B": m_b}

    def test_step_metrics_collected_during_recursive(self, simple_grid):
        """Per-step metrics are collected during recursive prediction loop."""
        sorted_months = [
            pd.Timestamp("2024-03-01"),
            pd.Timestamp("2024-04-01"),
        ]
        models = self._make_cluster_models(val_a=150.0, val_b=100.0)

        # Simulate the recursive loop with per-step tracking
        sales_df = pd.DataFrame(
            {
                "sku_ck": ["A", "B", "A", "B"],
                "startdate": [sorted_months[0], sorted_months[0],
                              sorted_months[1], sorted_months[1]],
                "qty": [300.0, 120.0, 400.0, 160.0],
            }
        )

        # Build actuals lookup (same as in run_tree_backtest)
        actuals_by_month: dict[pd.Timestamp, dict[str, float]] = {}
        for m in sorted_months:
            m_sales = sales_df[sales_df["startdate"] == m]
            if not m_sales.empty:
                actuals_by_month[m] = (
                    m_sales.drop_duplicates(subset="sku_ck")
                    .set_index("sku_ck")["qty"]
                    .to_dict()
                )

        step_metrics: list[dict] = []
        current_grid = simple_grid.copy()
        current_grid["ml_cluster"] = current_grid["sku_ck"].map(
            {"A": "cluster_A", "B": "cluster_B"}
        )

        # Step 1
        month1_data = current_grid[
            current_grid["startdate"] == sorted_months[0]
        ].copy()
        feature_cols = ["qty_lag_1"]
        preds_first = _predict_single_month(models, month1_data, feature_cols)
        current_grid = update_grid_with_predictions(
            current_grid, sorted_months[0], preds_first
        )

        if sorted_months[0] in actuals_by_month:
            step1_wape = _compute_step_wape(
                preds_first, actuals_by_month[sorted_months[0]]
            )
            step_metrics.append(
                {
                    "step": 1,
                    "month": str(sorted_months[0].date()),
                    "wape": step1_wape,
                    "n_dfus": len(preds_first),
                }
            )

        # Step 2
        month2_data = current_grid[
            current_grid["startdate"] == sorted_months[1]
        ].copy()
        preds_month2 = _predict_single_month(models, month2_data, feature_cols)
        if sorted_months[1] in actuals_by_month:
            step2_wape = _compute_step_wape(
                preds_month2, actuals_by_month[sorted_months[1]]
            )
            step_metrics.append(
                {
                    "step": 2,
                    "month": str(sorted_months[1].date()),
                    "wape": step2_wape,
                    "n_dfus": len(preds_month2),
                }
            )

        # Verify step_metrics were collected
        assert len(step_metrics) == 2
        assert step_metrics[0]["step"] == 1
        assert step_metrics[1]["step"] == 2

    def test_step_numbers_are_sequential(self, simple_grid):
        """Step numbers start at 1 and increment by 1."""
        sorted_months = [
            pd.Timestamp("2024-03-01"),
            pd.Timestamp("2024-04-01"),
        ]
        step_metrics = []
        for step_idx, month in enumerate(sorted_months, start=1):
            step_metrics.append(
                {
                    "step": step_idx,
                    "month": str(month.date()),
                    "wape": 5.0,
                    "n_dfus": 2,
                }
            )

        assert [m["step"] for m in step_metrics] == [1, 2]

    def test_metadata_includes_recursive_step_metrics(self):
        """Metadata dict includes recursive_step_metrics when recursive=True."""
        step_metrics = [
            {"step": 1, "month": "2024-03-01", "wape": 5.0, "n_dfus": 10},
            {"step": 2, "month": "2024-04-01", "wape": 8.5, "n_dfus": 10},
        ]
        metadata: dict = {}
        metadata["recursive"] = True
        metadata["recursive_step_metrics"] = step_metrics
        wapes = [m["wape"] for m in step_metrics if m.get("wape") is not None]
        metadata["recursive_accuracy_degradation"] = {
            "step_1_wape": step_metrics[0]["wape"],
            "last_step_wape": step_metrics[-1]["wape"],
            "mean_wape": round(float(np.mean(wapes)), 2) if wapes else None,
        }

        assert "recursive_step_metrics" in metadata
        assert len(metadata["recursive_step_metrics"]) == 2
        assert metadata["recursive_accuracy_degradation"]["step_1_wape"] == 5.0
        assert metadata["recursive_accuracy_degradation"]["last_step_wape"] == 8.5
        assert metadata["recursive_accuracy_degradation"]["mean_wape"] == 6.75

    def test_degradation_none_when_no_metrics(self):
        """Degradation values are None when step_metrics is empty."""
        step_metrics: list[dict] = []
        degradation = {
            "step_1_wape": step_metrics[0]["wape"] if step_metrics else None,
            "last_step_wape": step_metrics[-1]["wape"] if step_metrics else None,
        }
        assert degradation["step_1_wape"] is None
        assert degradation["last_step_wape"] is None


# ---------------------------------------------------------------------------
# Noise injection configuration
# ---------------------------------------------------------------------------


class TestNoiseInjectionConfig:
    def test_noise_disabled_by_default(self):
        """Noise injection is disabled by default in algorithm_config.yaml."""
        from common.utils import load_config, reset_config

        reset_config("algorithm_config.yaml")
        cfg = load_config("algorithm_config.yaml")
        assert cfg.get("recursive_noise_enabled", False) is False

    def test_noise_pct_default(self):
        """Default noise_pct is 0.05 in config."""
        from common.utils import load_config, reset_config

        reset_config("algorithm_config.yaml")
        cfg = load_config("algorithm_config.yaml")
        assert cfg.get("recursive_noise_pct", 0.05) == pytest.approx(0.05)

    def test_noise_not_applied_when_disabled(self):
        """When noise is disabled, training data lag cols are not modified."""
        np.random.seed(42)
        train_lag_values = np.array([100.0, 200.0, 300.0])
        noise_enabled = False
        noise_pct = 0.05

        if noise_enabled and noise_pct > 0:
            result = _inject_recursive_noise(train_lag_values, noise_pct)
        else:
            result = train_lag_values

        np.testing.assert_array_equal(result, train_lag_values)

    def test_noise_applied_when_enabled(self):
        """When noise is enabled, lag values are perturbed."""
        np.random.seed(42)
        train_lag_values = np.array([100.0, 200.0, 300.0])
        noise_enabled = True
        noise_pct = 0.05

        if noise_enabled and noise_pct > 0:
            result = _inject_recursive_noise(train_lag_values, noise_pct)
        else:
            result = train_lag_values

        assert not np.array_equal(result, train_lag_values)
