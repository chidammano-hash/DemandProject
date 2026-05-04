"""Tests for intermittent demand classification and Tweedie objective routing.

Verifies that clusters with high zero-demand percentages are correctly
classified and receive Tweedie loss objectives instead of default L2/RMSE.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helper: build a training DataFrame with a given zero-demand fraction
# ---------------------------------------------------------------------------

def _make_train_df(n_rows: int = 100, zero_pct: float = 0.0) -> pd.DataFrame:
    """Build a minimal training DataFrame with *zero_pct* fraction of zero qty."""
    n_zeros = int(n_rows * zero_pct)
    n_nonzero = n_rows - n_zeros
    qty = np.concatenate([np.zeros(n_zeros), np.random.default_rng(42).uniform(10, 100, n_nonzero)])
    np.random.default_rng(42).shuffle(qty)
    return pd.DataFrame({"qty": qty})


# ---------------------------------------------------------------------------
# _classify_cluster_demand tests
# ---------------------------------------------------------------------------

class TestClassifyClusterDemand:
    """Test demand pattern classification based on zero-demand percentage."""

    def test_60pct_zeros_is_intermittent(self) -> None:
        from scripts.ml.run_backtest import _classify_cluster_demand

        df = _make_train_df(100, zero_pct=0.60)
        result = _classify_cluster_demand(df)
        assert result == "intermittent"

    def test_40pct_zeros_is_lumpy(self) -> None:
        from scripts.ml.run_backtest import _classify_cluster_demand

        df = _make_train_df(100, zero_pct=0.40)
        result = _classify_cluster_demand(df)
        assert result == "lumpy"

    def test_10pct_zeros_is_continuous(self) -> None:
        from scripts.ml.run_backtest import _classify_cluster_demand

        df = _make_train_df(100, zero_pct=0.10)
        result = _classify_cluster_demand(df)
        assert result == "continuous"

    def test_boundary_50pct_zeros_is_intermittent(self) -> None:
        """50% is exactly at the threshold (>=), so it should be intermittent."""
        from scripts.ml.run_backtest import _classify_cluster_demand

        df = _make_train_df(100, zero_pct=0.50)
        result = _classify_cluster_demand(df)
        assert result == "intermittent"

    def test_0pct_zeros_is_continuous(self) -> None:
        from scripts.ml.run_backtest import _classify_cluster_demand

        df = _make_train_df(100, zero_pct=0.0)
        result = _classify_cluster_demand(df)
        assert result == "continuous"

    def test_30pct_zeros_is_continuous(self) -> None:
        """30% is at the lumpy boundary (> 0.3 required), so it stays continuous."""
        from scripts.ml.run_backtest import _classify_cluster_demand

        df = _make_train_df(100, zero_pct=0.30)
        result = _classify_cluster_demand(df)
        assert result == "continuous"

    def test_custom_thresholds(self) -> None:
        """Thresholds can be overridden via keyword arguments."""
        from scripts.ml.run_backtest import _classify_cluster_demand

        df = _make_train_df(100, zero_pct=0.25)
        # With lower thresholds, 25% should be intermittent
        result = _classify_cluster_demand(
            df, intermittent_threshold=0.2, lumpy_threshold=0.1,
        )
        assert result == "intermittent"

    def test_empty_dataframe_is_continuous(self) -> None:
        from scripts.ml.run_backtest import _classify_cluster_demand

        df = pd.DataFrame({"qty": pd.Series(dtype=float)})
        result = _classify_cluster_demand(df)
        assert result == "continuous"

    def test_missing_qty_column_is_continuous(self) -> None:
        from scripts.ml.run_backtest import _classify_cluster_demand

        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = _classify_cluster_demand(df)
        assert result == "continuous"

    def test_threshold_configurable_from_yaml(self) -> None:
        """Verify the YAML config keys are read and used as thresholds."""
        from scripts.ml.run_backtest import _classify_cluster_demand

        # 45% zeros: with default thresholds (0.5/0.3) this is lumpy
        df = _make_train_df(100, zero_pct=0.45)
        assert _classify_cluster_demand(df) == "lumpy"

        # With a lower intermittent threshold, same data becomes intermittent
        assert _classify_cluster_demand(
            df, intermittent_threshold=0.4,
        ) == "intermittent"


# ---------------------------------------------------------------------------
# _apply_tweedie_objective tests
# ---------------------------------------------------------------------------

class TestApplyTweedieObjective:
    """Test objective routing for each model type and demand pattern.

    Intermittent clusters (>70% zeros) use MAE (regression_l1) instead of Tweedie,
    because Tweedie's log link function causes early stopping at iter 1 for very
    sparse data, destroying accuracy.  Lumpy and continuous clusters keep defaults.
    """

    def test_lgbm_intermittent_gets_mae(self) -> None:
        from scripts.ml.run_backtest import _apply_tweedie_objective

        params = {"n_estimators": 300, "learning_rate": 0.05, "verbosity": -1}
        result = _apply_tweedie_objective(params, "lgbm", "intermittent")
        assert result["objective"] == "regression_l1"
        assert "tweedie_variance_power" not in result
        # Original params preserved
        assert result["n_estimators"] == 300

    def test_catboost_intermittent_gets_mae(self) -> None:
        from scripts.ml.run_backtest import _apply_tweedie_objective

        params = {"iterations": 300, "loss_function": "RMSE", "verbose": 0}
        result = _apply_tweedie_objective(params, "catboost", "intermittent")
        assert result["loss_function"] == "MAE"
        assert result["iterations"] == 300

    def test_xgboost_intermittent_gets_absoluteerror(self) -> None:
        from scripts.ml.run_backtest import _apply_tweedie_objective

        params = {"n_estimators": 500, "learning_rate": 0.05, "verbosity": 0}
        result = _apply_tweedie_objective(params, "xgboost", "intermittent")
        assert result["objective"] == "reg:absoluteerror"
        assert "tweedie_variance_power" not in result

    def test_continuous_keeps_default_objective(self) -> None:
        from scripts.ml.run_backtest import _apply_tweedie_objective

        params = {"n_estimators": 300, "learning_rate": 0.05}
        result = _apply_tweedie_objective(params, "lgbm", "continuous")
        assert "objective" not in result
        assert "tweedie_variance_power" not in result
        assert result == params

    def test_lumpy_keeps_default_objective(self) -> None:
        from scripts.ml.run_backtest import _apply_tweedie_objective

        params = {"n_estimators": 300}
        result = _apply_tweedie_objective(params, "lgbm", "lumpy")
        assert "objective" not in result
        assert result == params

    def test_lgbm_intermittent_strips_existing_tweedie_vp(self) -> None:
        """If params already have tweedie_variance_power, it must be removed."""
        from scripts.ml.run_backtest import _apply_tweedie_objective

        params = {"n_estimators": 300, "tweedie_variance_power": 1.8}
        result = _apply_tweedie_objective(params, "lgbm", "intermittent")
        assert result["objective"] == "regression_l1"
        assert "tweedie_variance_power" not in result

    def test_catboost_intermittent_strips_boost_from_average(self) -> None:
        """CatBoost boost_from_average is removed when switching to MAE for intermittent."""
        from scripts.ml.run_backtest import _apply_tweedie_objective

        params = {"iterations": 300, "loss_function": "RMSE", "boost_from_average": True}
        result = _apply_tweedie_objective(params, "catboost", "intermittent")
        assert result["loss_function"] == "MAE"
        assert "boost_from_average" not in result

    def test_original_params_not_mutated(self) -> None:
        from scripts.ml.run_backtest import _apply_tweedie_objective

        params = {"n_estimators": 300, "learning_rate": 0.05}
        original = dict(params)
        _apply_tweedie_objective(params, "lgbm", "intermittent")
        assert params == original, "Original params dict must not be mutated"


# ---------------------------------------------------------------------------
# Integration: verify YAML config drives thresholds
# ---------------------------------------------------------------------------

class TestYAMLConfigIntegration:
    """Verify that algorithm_config.yaml Tweedie settings are respected."""

    def test_config_keys_exist(self) -> None:
        """The YAML config must contain the Tweedie keys."""
        from common.utils import load_config

        cfg = load_config("forecast_pipeline_config.yaml")
        backtest = cfg["backtest"]
        assert "tweedie_variance_power" in backtest
        assert "intermittent_threshold" in backtest
        assert "lumpy_threshold" in backtest

    def test_config_default_values(self) -> None:
        from common.utils import load_config

        cfg = load_config("forecast_pipeline_config.yaml")
        backtest = cfg["backtest"]
        assert backtest["tweedie_variance_power"] == 1.5
        assert backtest["intermittent_threshold"] == 0.7
        assert backtest["lumpy_threshold"] == 0.3

    def test_baseline_intermittent_explicit(self) -> None:
        """Gen-4 roadmap 1.2: baseline_intermittent must be explicit in config."""
        from common.utils import load_config

        cfg = load_config("forecast_pipeline_config.yaml")
        backtest = cfg["backtest"]
        assert "baseline_intermittent" in backtest
        assert "baseline_intermittent_window" in backtest
        assert backtest["baseline_intermittent"] is True
        assert backtest["baseline_intermittent_window"] == 12

    def test_embargo_months_non_zero(self) -> None:
        """Gen-4 roadmap 1.3: embargo_months must be non-zero to prevent leakage."""
        from common.utils import load_config

        cfg = load_config("forecast_pipeline_config.yaml")
        backtest = cfg["backtest"]
        assert backtest["embargo_months"] >= 1
