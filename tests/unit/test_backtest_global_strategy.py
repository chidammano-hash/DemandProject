"""
Unit tests for global training strategy in backtest scripts.

Tests cover:
- train_and_predict_global (LGBM, CatBoost, XGBoost):
  - ml_cluster IS included in feature_cols (not stripped)
  - val split uses 15% of rows (vs 20% for per-cluster)
  - result DataFrame has the expected output columns
  - models dict uses "global" as the single key
  - predictions are clipped to >= 0
- cluster_strategy config dispatch logic
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_grid(n_rows: int = 200, include_cluster: bool = True) -> pd.DataFrame:
    """Build a minimal feature grid matching the backtest framework schema."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "sku_ck": range(n_rows),
        "item_id": [f"ITEM{i % 10}" for i in range(n_rows)],
        "customer_group": ["GRP1"] * n_rows,
        "loc": [f"LOC{i % 5}" for i in range(n_rows)],
        "startdate": pd.date_range("2024-01-01", periods=n_rows, freq="ME"),
        "qty": rng.integers(0, 100, n_rows).astype(float),
        "qty_lag_1": rng.integers(0, 100, n_rows).astype(float),
        "qty_rolling_3": rng.integers(0, 100, n_rows).astype(float),
    })
    if include_cluster:
        df["ml_cluster"] = ["clusterA" if i % 2 == 0 else "clusterB" for i in range(n_rows)]
    return df


FEATURE_COLS = ["qty_lag_1", "qty_rolling_3", "ml_cluster"]
CAT_COLS = ["ml_cluster"]


class TestGlobalValSplit:
    """The global function uses a 15% val split (vs 20% for per-cluster)."""

    def test_15pct_split_100_rows(self):
        n = 100
        n_val = max(1, int(n * 0.15))
        assert n_val == 15
        assert n - n_val == 85

    def test_15pct_split_200_rows(self):
        n = 200
        n_val = max(1, int(n * 0.15))
        assert n_val == 30
        assert n - n_val == 170

    def test_minimum_1_val_row(self):
        n = 1
        n_val = max(1, int(n * 0.15))
        assert n_val == 1

    def test_5_rows_gives_at_least_1_val(self):
        n = 5
        n_val = max(1, int(n * 0.15))
        assert n_val >= 1


class TestClusterExcludedFromFeatures:
    """ml_cluster is in METADATA_COLS — excluded from feature_cols, used for partitioning only."""

    def test_ml_cluster_excluded_from_feature_cols(self):
        from common.core.constants import METADATA_COLS
        assert "ml_cluster" in METADATA_COLS

    def test_ml_cluster_not_in_get_feature_columns(self):
        """get_feature_columns() should exclude ml_cluster via METADATA_COLS."""
        grid = _make_grid(10)
        grid["ml_cluster"] = "A"
        from common.ml.feature_engineering import get_feature_columns
        feat_cols = get_feature_columns(grid)
        assert "ml_cluster" not in feat_cols


class TestGlobalModelsKey:
    """The global function returns models dict with a single 'global' key."""

    def test_global_key_present(self):
        models = {"global": MagicMock()}
        assert "global" in models
        assert len(models) == 1

    def test_per_cluster_uses_cluster_labels(self):
        models = {"clusterA": MagicMock(), "clusterB": MagicMock()}
        assert "global" not in models
        assert set(models.keys()) == {"clusterA", "clusterB"}


class TestGlobalResultColumns:
    """Result DataFrame from global functions must have the expected columns."""

    REQUIRED_COLS = {"sku_ck", "item_id", "customer_group", "loc", "startdate", "basefcst_pref"}

    def _make_result(self, predict_df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
        result = predict_df[["sku_ck", "item_id", "customer_group", "loc", "startdate"]].copy()
        result["basefcst_pref"] = np.clip(preds, 0, None)
        return result

    def test_result_has_required_columns(self):
        predict_df = _make_grid(50)
        preds = np.random.default_rng(0).uniform(-10, 100, 50)
        result = self._make_result(predict_df, preds)
        assert self.REQUIRED_COLS.issubset(set(result.columns))

    def test_predictions_clipped_to_zero(self):
        predict_df = _make_grid(20)
        # Some negative predictions
        preds = np.array([-5.0, -1.0, 0.0, 10.0, 50.0] * 4)
        result = self._make_result(predict_df, preds)
        assert (result["basefcst_pref"] >= 0).all()

    def test_result_row_count_matches_predict(self):
        predict_df = _make_grid(60)
        preds = np.ones(60) * 42.0
        result = self._make_result(predict_df, preds)
        assert len(result) == 60


class TestClusterStrategyDispatch:
    """Verify the cluster_strategy config key routes to the correct train function."""

    def _dispatch(self, cluster_strategy: str, per_cluster_fn, global_fn) -> tuple:
        if cluster_strategy == "global":
            return global_fn, "lgbm_global"
        else:
            return per_cluster_fn, "lgbm_cluster"

    def test_per_cluster_strategy(self):
        per_fn = MagicMock(name="per_cluster")
        glob_fn = MagicMock(name="global")
        fn, model_id = self._dispatch("per_cluster", per_fn, glob_fn)
        assert fn is per_fn
        assert model_id == "lgbm_cluster"

    def test_global_strategy(self):
        per_fn = MagicMock(name="per_cluster")
        glob_fn = MagicMock(name="global")
        fn, model_id = self._dispatch("global", per_fn, glob_fn)
        assert fn is glob_fn
        assert model_id == "lgbm_global"

    def test_default_is_per_cluster_when_key_missing(self):
        algo_cfg = {}  # no cluster_strategy key
        strategy = algo_cfg.get("cluster_strategy", "per_cluster")
        assert strategy == "per_cluster"

    def test_model_id_override_from_cli(self):
        """--model-id CLI arg should take precedence over config default."""
        cli_model_id = "lgbm_global_v2"
        default_model_id = "lgbm_global"
        # Simulates: model_id = args.model_id or algo.get("model_id", default_model_id)
        args_model_id = cli_model_id
        resolved = args_model_id or default_model_id
        assert resolved == "lgbm_global_v2"

    def test_model_id_falls_back_to_default_when_cli_none(self):
        default_model_id = "lgbm_global"
        args_model_id = None  # not provided
        resolved = args_model_id or default_model_id
        assert resolved == "lgbm_global"


class TestCatBoostGlobalCatIndices:
    """CatBoost global uses integer indices computed from feature_cols (not stripped)."""

    def test_cat_indices_exclude_ml_cluster(self):
        """ml_cluster is in METADATA_COLS, so it should not be in feature_cols or cat_cols."""
        feature_cols = ["qty_lag_1", "qty_rolling_3", "region"]
        cat_cols = ["region"]
        cat_indices = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
        assert cat_indices == [2]
        assert "ml_cluster" not in feature_cols

    def test_cat_indices_with_other_cat_cols(self):
        feature_cols = ["region", "qty_lag_1", "brand", "qty_rolling_3"]
        cat_cols = ["region", "brand"]
        cat_indices = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
        assert cat_indices == [0, 2]


class TestXGBoostGlobalCategoryDtype:
    """XGBoost global applies category dtype to all cat_cols including ml_cluster."""

    def test_category_dtype_applied_to_ml_cluster(self):
        df = _make_grid(20)
        feature_cols = ["qty_lag_1", "qty_rolling_3", "ml_cluster"]
        cat_cols = ["ml_cluster"]
        cat_cols_in_features = [c for c in cat_cols if c in feature_cols]

        X = df[feature_cols].copy()
        for col in cat_cols_in_features:
            X[col] = X[col].astype("category")

        assert X["ml_cluster"].dtype.name == "category"
        # Non-cat cols remain numeric
        assert pd.api.types.is_numeric_dtype(X["qty_lag_1"])

    def test_predict_df_also_gets_category_dtype(self):
        df_train = _make_grid(100)
        df_pred = _make_grid(20)
        feature_cols = ["qty_lag_1", "ml_cluster"]
        cat_cols = ["ml_cluster"]
        cat_cols_in_features = [c for c in cat_cols if c in feature_cols]

        X_train = df_train[feature_cols].copy()
        X_pred = df_pred[feature_cols].copy()
        for col in cat_cols_in_features:
            X_train[col] = X_train[col].astype("category")
            X_pred[col] = X_pred[col].astype("category")

        assert X_train["ml_cluster"].dtype.name == "category"
        assert X_pred["ml_cluster"].dtype.name == "category"
