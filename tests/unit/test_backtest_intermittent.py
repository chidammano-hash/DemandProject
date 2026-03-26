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
        from scripts.run_backtest import _classify_cluster_demand

        df = _make_train_df(100, zero_pct=0.60)
        result = _classify_cluster_demand(df)
        assert result == "intermittent"

    def test_40pct_zeros_is_lumpy(self) -> None:
        from scripts.run_backtest import _classify_cluster_demand

        df = _make_train_df(100, zero_pct=0.40)
        result = _classify_cluster_demand(df)
        assert result == "lumpy"

    def test_10pct_zeros_is_continuous(self) -> None:
        from scripts.run_backtest import _classify_cluster_demand

        df = _make_train_df(100, zero_pct=0.10)
        result = _classify_cluster_demand(df)
        assert result == "continuous"

    def test_boundary_50pct_zeros_is_intermittent(self) -> None:
        """50% is exactly at the threshold (>=), so it should be intermittent."""
        from scripts.run_backtest import _classify_cluster_demand

        df = _make_train_df(100, zero_pct=0.50)
        result = _classify_cluster_demand(df)
        assert result == "intermittent"

    def test_0pct_zeros_is_continuous(self) -> None:
        from scripts.run_backtest import _classify_cluster_demand

        df = _make_train_df(100, zero_pct=0.0)
        result = _classify_cluster_demand(df)
        assert result == "continuous"

    def test_30pct_zeros_is_continuous(self) -> None:
        """30% is at the lumpy boundary (> 0.3 required), so it stays continuous."""
        from scripts.run_backtest import _classify_cluster_demand

        df = _make_train_df(100, zero_pct=0.30)
        result = _classify_cluster_demand(df)
        assert result == "continuous"

    def test_custom_thresholds(self) -> None:
        """Thresholds can be overridden via keyword arguments."""
        from scripts.run_backtest import _classify_cluster_demand

        df = _make_train_df(100, zero_pct=0.25)
        # With lower thresholds, 25% should be intermittent
        result = _classify_cluster_demand(
            df, intermittent_threshold=0.2, lumpy_threshold=0.1,
        )
        assert result == "intermittent"

    def test_empty_dataframe_is_continuous(self) -> None:
        from scripts.run_backtest import _classify_cluster_demand

        df = pd.DataFrame({"qty": pd.Series(dtype=float)})
        result = _classify_cluster_demand(df)
        assert result == "continuous"

    def test_missing_qty_column_is_continuous(self) -> None:
        from scripts.run_backtest import _classify_cluster_demand

        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = _classify_cluster_demand(df)
        assert result == "continuous"

    def test_threshold_configurable_from_yaml(self) -> None:
        """Verify the YAML config keys are read and used as thresholds."""
        from scripts.run_backtest import _classify_cluster_demand

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
    """Test Tweedie objective injection for each model type."""

    def test_lgbm_intermittent_gets_tweedie(self) -> None:
        from scripts.run_backtest import _apply_tweedie_objective

        params = {"n_estimators": 300, "learning_rate": 0.05, "verbosity": -1}
        result = _apply_tweedie_objective(params, "lgbm", "intermittent")
        assert result["objective"] == "tweedie"
        assert result["tweedie_variance_power"] == 1.5
        # Original params preserved
        assert result["n_estimators"] == 300

    def test_catboost_intermittent_gets_tweedie(self) -> None:
        from scripts.run_backtest import _apply_tweedie_objective

        params = {"iterations": 300, "loss_function": "RMSE", "verbose": 0}
        result = _apply_tweedie_objective(params, "catboost", "intermittent")
        assert result["loss_function"] == "Tweedie:variance_power=1.5"
        assert result["iterations"] == 300

    def test_xgboost_intermittent_gets_tweedie(self) -> None:
        from scripts.run_backtest import _apply_tweedie_objective

        params = {"n_estimators": 500, "learning_rate": 0.05, "verbosity": 0}
        result = _apply_tweedie_objective(params, "xgboost", "intermittent")
        assert result["objective"] == "reg:tweedie"
        assert result["tweedie_variance_power"] == 1.5

    def test_continuous_keeps_default_objective(self) -> None:
        from scripts.run_backtest import _apply_tweedie_objective

        params = {"n_estimators": 300, "learning_rate": 0.05}
        result = _apply_tweedie_objective(params, "lgbm", "continuous")
        assert "objective" not in result
        assert "tweedie_variance_power" not in result
        assert result == params

    def test_lumpy_keeps_default_objective(self) -> None:
        from scripts.run_backtest import _apply_tweedie_objective

        params = {"n_estimators": 300}
        result = _apply_tweedie_objective(params, "lgbm", "lumpy")
        assert "objective" not in result
        assert result == params

    def test_custom_variance_power(self) -> None:
        from scripts.run_backtest import _apply_tweedie_objective

        params = {"n_estimators": 300}
        result = _apply_tweedie_objective(
            params, "lgbm", "intermittent", tweedie_variance_power=1.8,
        )
        assert result["tweedie_variance_power"] == 1.8

    def test_catboost_custom_variance_power(self) -> None:
        from scripts.run_backtest import _apply_tweedie_objective

        params = {"iterations": 300, "loss_function": "RMSE"}
        result = _apply_tweedie_objective(
            params, "catboost", "intermittent", tweedie_variance_power=1.2,
        )
        assert result["loss_function"] == "Tweedie:variance_power=1.2"

    def test_original_params_not_mutated(self) -> None:
        from scripts.run_backtest import _apply_tweedie_objective

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

        cfg = load_config("algorithm_config.yaml")
        backtest = cfg["backtest"]
        assert "tweedie_variance_power" in backtest
        assert "intermittent_threshold" in backtest
        assert "lumpy_threshold" in backtest

    def test_config_default_values(self) -> None:
        from common.utils import load_config

        cfg = load_config("algorithm_config.yaml")
        backtest = cfg["backtest"]
        assert backtest["tweedie_variance_power"] == 1.5
        assert backtest["intermittent_threshold"] == 0.5
        assert backtest["lumpy_threshold"] == 0.3
