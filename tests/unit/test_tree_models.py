"""Unit tests for scripts.algorithm_testing.tree_models.run_tree_models.

Strategy
--------
- Never train a real ML model.  importlib.import_module, fit_model, and
  get_best_iteration are all patched so tests run in milliseconds.
- The mock model's .predict() returns a fixed constant array so we can assert
  on output values without floating-point surprises.

Fixture design
--------------
_MIN_GROUP_ROWS = 50 in the source.  Each DFU contributes 12 training rows
(n_train_months=12).  To exceed the threshold, a partition group needs
at least 5 DFUs (5 × 12 = 60 > 50).  Helper _make_grid() defaults to
N_DFUS_PER_GROUP DFUs per group for tests that need predictions.

Patched symbols (all via scripts.algorithm_testing.tree_models.*)
----------------------------------------------------------
  fit_model              – no-op (avoids real training)
  get_best_iteration     – returns fixed int
  importlib.import_module – returns fake lib_module with fake model class
  get_feature_columns    – returns controlled list
  load_config            – returns minimal algo config dict

Critical ordering rule
----------------------
``patch("scripts.algorithm_testing.tree_models.importlib.import_module", ...)`` must
ALWAYS be the LAST entry in every ``with (P1, P2, ..., importlib_patch):``
block.  Patching importlib.import_module first causes Python's patch()
machinery (which internally calls importlib.import_module to resolve the
target module name) to receive our fake lib_module instead of the real
module.  This silently undoes any patches that were set up before the
importlib patch was entered.

Note on ImportError test
------------------------
We monkeypatch the ``importlib`` attribute on the module object directly
(rather than using patch()) because patch() itself uses importlib internally
and would receive the ImportError side-effect during setup.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
import scripts.algorithm_testing.tree_models as _tree_models_mod
from scripts.algorithm_testing.tree_models import run_tree_models


# ---------------------------------------------------------------------------
# Constants / tuning knobs
# ---------------------------------------------------------------------------

# Number of DFUs per archetype/cluster group.  Must be large enough so that
# n_train_months * N_DFUS_PER_GROUP >= _MIN_GROUP_ROWS (50).
N_DFUS_PER_GROUP = 5
N_TRAIN_MONTHS = 12  # 5 × 12 = 60 ≥ 50

_FEATURE_COLS = ["qty_lag_1", "rolling_mean_3m", "ml_cluster"]
_PREDICT_COLS = ["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
_FIXED_PRED = 42.0
_MODEL_ID = "lgbm_cluster"

# Minimal algo_config consumed by run_tree_models
_ALGO_CONFIG: dict[str, Any] = {
    "algorithms": {
        "lgbm": {
            "enabled": True,
            "model_id": _MODEL_ID,
            "n_estimators": 100,
        },
    },
}


# ---------------------------------------------------------------------------
# Grid / fixture builders
# ---------------------------------------------------------------------------

def _make_group_skus(group_label: str, n: int = N_DFUS_PER_GROUP) -> list[str]:
    return [f"SKU_{group_label}_{i:03d}" for i in range(n)]


def _make_grid(
    sku_to_cluster: dict[str, str],
    n_train_months: int = N_TRAIN_MONTHS,
    predict_month: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build a minimal feature grid.

    sku_to_cluster maps each SKU to its ml_cluster label.
    Each SKU gets ``n_train_months`` training rows followed by one predict row.
    """
    if predict_month is None:
        predict_month = pd.Timestamp("2025-01-01")

    rows = []
    rng = np.random.default_rng(0)
    for sku, cluster in sku_to_cluster.items():
        train_dates = pd.date_range("2024-01-01", periods=n_train_months, freq="MS")
        for d in train_dates:
            rows.append({
                "sku_ck": sku,
                "startdate": d,
                "qty": float(rng.integers(50, 200)),
                "ml_cluster": cluster,
                "qty_lag_1": float(rng.integers(40, 190)),
                "rolling_mean_3m": float(rng.integers(40, 190)),
            })
        # Predict row
        rows.append({
            "sku_ck": sku,
            "startdate": predict_month,
            "qty": np.nan,
            "ml_cluster": cluster,
            "qty_lag_1": float(rng.integers(40, 190)),
            "rolling_mean_3m": float(rng.integers(40, 190)),
        })
    return pd.DataFrame(rows)


def _make_classification_df(sku_to_archetype: dict[str, str]) -> pd.DataFrame:
    """Build a minimal classification DataFrame."""
    return pd.DataFrame({
        "sku_ck": list(sku_to_archetype.keys()),
        "archetype": list(sku_to_archetype.values()),
    })


def _fake_lib_module(fixed_pred: float = _FIXED_PRED) -> MagicMock:
    """Return a mock 'lightgbm' module with a fake LGBMRegressor class.

    model.predict returns a per-row array via side_effect to work correctly
    with variable-size prediction batches.
    """
    model_instance = MagicMock()
    model_instance.predict.side_effect = lambda X: np.full(len(X), fixed_pred)
    model_instance.best_iteration_ = 50

    model_class = MagicMock(return_value=model_instance)

    lib_module = MagicMock()
    lib_module.LGBMRegressor = model_class
    lib_module.early_stopping.return_value = "es"
    lib_module.log_evaluation.return_value = "log"
    return lib_module


# ---------------------------------------------------------------------------
# Test 1 — archetype partitioning when classification_df is provided
# ---------------------------------------------------------------------------


class TestArchetypePartitionUsed:
    """When classification_df is given, the function iterates by archetype."""

    def test_archetype_partition_used_when_classification_df_provided(self):
        """Output must include DFUs from both archetypes."""
        predict_month = pd.Timestamp("2025-01-01")

        smooth_skus = _make_group_skus("smooth")
        erratic_skus = _make_group_skus("erratic")
        all_skus = smooth_skus + erratic_skus

        sku_to_cluster = {s: "C1" for s in smooth_skus} | {s: "C2" for s in erratic_skus}
        sku_to_archetype = {s: "smooth" for s in smooth_skus} | {s: "erratic" for s in erratic_skus}

        grid = _make_grid(sku_to_cluster, predict_month=predict_month)
        classification_df = _make_classification_df(sku_to_archetype)

        lib_module = _fake_lib_module()

        # importlib patch LAST to preserve fit_model/get_feature_columns patches
        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.load_config", return_value=_ALGO_CONFIG),
            patch("scripts.algorithm_testing.tree_models.importlib.import_module", return_value=lib_module),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[predict_month],
                enabled_models={"lgbm": {"enabled": True}},
                classification_df=classification_df,
            )

        assert isinstance(result, pd.DataFrame)
        assert set(_PREDICT_COLS).issubset(result.columns), (
            f"Missing columns: {set(_PREDICT_COLS) - set(result.columns)}"
        )
        assert set(result["sku_ck"].tolist()) == set(all_skus), (
            f"Expected all {len(all_skus)} SKUs; got {sorted(result['sku_ck'].tolist())}"
        )

    def test_archetype_partition_produces_one_row_per_sku_predict_month(self):
        """Each SKU × predict_month yields exactly one prediction row."""
        predict_month = pd.Timestamp("2025-01-01")

        skus = _make_group_skus("group_A")  # all in same archetype, one group
        sku_to_cluster = {s: "C1" for s in skus}
        sku_to_archetype = {s: "smooth" for s in skus}

        grid = _make_grid(sku_to_cluster, predict_month=predict_month)
        classification_df = _make_classification_df(sku_to_archetype)

        lib_module = _fake_lib_module()

        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.load_config", return_value=_ALGO_CONFIG),
            patch("scripts.algorithm_testing.tree_models.importlib.import_module", return_value=lib_module),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[predict_month],
                enabled_models={"lgbm": {}},
                classification_df=classification_df,
            )

        assert len(result) == len(skus), (
            f"Expected {len(skus)} rows (one per SKU), got {len(result)}"
        )

    def test_archetype_column_absent_from_classification_df_falls_back_to_ml_cluster(self):
        """If classification_df lacks 'archetype' column, ml_cluster is used."""
        predict_month = pd.Timestamp("2025-01-01")

        skus = _make_group_skus("G")
        sku_to_cluster = {s: "C1" for s in skus}
        grid = _make_grid(sku_to_cluster, predict_month=predict_month)

        # classification_df has wrong column name — no 'archetype'
        bad_classification_df = pd.DataFrame({
            "sku_ck": skus,
            "demand_type": ["stable"] * len(skus),
        })

        lib_module = _fake_lib_module()

        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.load_config", return_value=_ALGO_CONFIG),
            patch("scripts.algorithm_testing.tree_models.importlib.import_module", return_value=lib_module),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[predict_month],
                enabled_models={"lgbm": {}},
                classification_df=bad_classification_df,
            )

        # Falls back to ml_cluster — all SKUs from the single cluster should appear
        assert isinstance(result, pd.DataFrame)
        assert set(result["sku_ck"].tolist()) == set(skus)


# ---------------------------------------------------------------------------
# Test 2 — ml_cluster fallback when classification_df is None
# ---------------------------------------------------------------------------


class TestMlClusterFallback:
    """When classification_df=None, the function partitions by ml_cluster."""

    def test_ml_cluster_fallback_when_no_classification_df(self):
        """Output must cover all DFUs when classification_df is omitted."""
        predict_month = pd.Timestamp("2025-01-01")

        group_a = _make_group_skus("A")
        group_b = _make_group_skus("B")
        all_skus = group_a + group_b

        sku_to_cluster = {s: "C1" for s in group_a} | {s: "C2" for s in group_b}
        grid = _make_grid(sku_to_cluster, predict_month=predict_month)

        lib_module = _fake_lib_module()

        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.load_config", return_value=_ALGO_CONFIG),
            patch("scripts.algorithm_testing.tree_models.importlib.import_module", return_value=lib_module),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[predict_month],
                enabled_models={"lgbm": {}},
                classification_df=None,
            )

        assert isinstance(result, pd.DataFrame)
        assert set(_PREDICT_COLS).issubset(result.columns)
        assert set(result["sku_ck"].tolist()) == set(all_skus)

    def test_ml_cluster_fallback_algorithm_id_set_correctly(self):
        """algorithm_id in output must match model_id from algo_config."""
        predict_month = pd.Timestamp("2025-01-01")

        skus = _make_group_skus("G")
        sku_to_cluster = {s: "C1" for s in skus}
        grid = _make_grid(sku_to_cluster, predict_month=predict_month)

        lib_module = _fake_lib_module()

        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.load_config", return_value=_ALGO_CONFIG),
            patch("scripts.algorithm_testing.tree_models.importlib.import_module", return_value=lib_module),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[predict_month],
                enabled_models={"lgbm": {}},
                classification_df=None,
            )

        assert (result["algorithm_id"] == _MODEL_ID).all(), (
            f"Expected algorithm_id={_MODEL_ID!r} for all rows; got {result['algorithm_id'].unique()}"
        )

    def test_predictions_are_non_negative(self):
        """Predictions must be clipped to >= 0 (np.maximum guard in source)."""
        predict_month = pd.Timestamp("2025-01-01")

        skus = _make_group_skus("G")
        sku_to_cluster = {s: "C1" for s in skus}
        grid = _make_grid(sku_to_cluster, predict_month=predict_month)

        lib_module = _fake_lib_module()
        # Make predict return negative values — the function should clip them
        lib_module.LGBMRegressor.return_value.predict.side_effect = (
            lambda X: np.full(len(X), -99.0)
        )

        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.load_config", return_value=_ALGO_CONFIG),
            patch("scripts.algorithm_testing.tree_models.importlib.import_module", return_value=lib_module),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[predict_month],
                enabled_models={"lgbm": {}},
                classification_df=None,
            )

        assert (result["basefcst_pref"] >= 0).all(), (
            "basefcst_pref must never be negative after np.maximum clip"
        )


# ---------------------------------------------------------------------------
# Test 3 — empty predict_months returns empty DataFrame with correct columns
# ---------------------------------------------------------------------------


class TestEmptyPredictMonths:
    def test_empty_predict_months_returns_empty(self):
        """When no rows match predict_months, an empty DataFrame is returned."""
        skus = _make_group_skus("G")
        sku_to_cluster = {s: "C1" for s in skus}
        grid = _make_grid(sku_to_cluster)

        lib_module = _fake_lib_module()

        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.load_config", return_value=_ALGO_CONFIG),
            patch("scripts.algorithm_testing.tree_models.importlib.import_module", return_value=lib_module),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[pd.Timestamp("2099-01-01")],  # no matching rows in grid
                enabled_models={"lgbm": {}},
                classification_df=None,
            )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == _PREDICT_COLS, (
            f"Expected columns {_PREDICT_COLS}, got {list(result.columns)}"
        )

    def test_empty_predict_months_list_returns_empty(self):
        """predict_months=[] (empty list) must return empty DataFrame."""
        skus = _make_group_skus("G")
        sku_to_cluster = {s: "C1" for s in skus}
        grid = _make_grid(sku_to_cluster)

        lib_module = _fake_lib_module()

        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.load_config", return_value=_ALGO_CONFIG),
            patch("scripts.algorithm_testing.tree_models.importlib.import_module", return_value=lib_module),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[],
                enabled_models={"lgbm": {}},
                classification_df=None,
            )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == _PREDICT_COLS


# ---------------------------------------------------------------------------
# Test 4 — unknown model name is skipped gracefully
# ---------------------------------------------------------------------------


class TestUnknownModelSkipped:
    def test_unknown_model_skipped(self):
        """An unknown model name must be skipped without raising an exception."""
        predict_month = pd.Timestamp("2025-01-01")
        skus = _make_group_skus("G")
        sku_to_cluster = {s: "C1" for s in skus}
        grid = _make_grid(sku_to_cluster, predict_month=predict_month)

        lib_module = _fake_lib_module()

        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.load_config", return_value=_ALGO_CONFIG),
            patch("scripts.algorithm_testing.tree_models.importlib.import_module", return_value=lib_module),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[predict_month],
                enabled_models={"prophet_unknown": {}},   # not in _MODEL_LIB
                classification_df=None,
            )

        # No predictions produced — returns empty sentinel frame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == _PREDICT_COLS

    def test_mixed_known_unknown_models_only_known_produces_output(self):
        """Unknown model is silently skipped; known model still produces output."""
        predict_month = pd.Timestamp("2025-01-01")
        skus = _make_group_skus("G")
        sku_to_cluster = {s: "C1" for s in skus}
        grid = _make_grid(sku_to_cluster, predict_month=predict_month)

        lib_module = _fake_lib_module()

        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.load_config", return_value=_ALGO_CONFIG),
            patch("scripts.algorithm_testing.tree_models.importlib.import_module", return_value=lib_module),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[predict_month],
                enabled_models={"lgbm": {}, "made_up_model": {}},
                classification_df=None,
            )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert (result["algorithm_id"] == _MODEL_ID).all()

    def test_only_unknown_models_returns_correct_empty_columns(self):
        """When ALL models are unknown, result must still have the standard columns."""
        predict_month = pd.Timestamp("2025-01-01")
        skus = _make_group_skus("G")
        sku_to_cluster = {s: "C1" for s in skus}
        grid = _make_grid(sku_to_cluster, predict_month=predict_month)

        lib_module = _fake_lib_module()

        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.load_config", return_value=_ALGO_CONFIG),
            patch("scripts.algorithm_testing.tree_models.importlib.import_module", return_value=lib_module),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[predict_month],
                enabled_models={"no_such_1": {}, "no_such_2": {}},
                classification_df=None,
            )

        assert list(result.columns) == _PREDICT_COLS


# ---------------------------------------------------------------------------
# Test 5 — archetype treated as categorical feature
# ---------------------------------------------------------------------------


class TestArchetypeAsCategoricalFeature:
    """When classification_df is provided, 'archetype' should appear in cat_cols
    passed to fit_model, and DFU assignments in the output must match the input.
    """

    def test_archetype_added_to_features(self):
        """'archetype' must appear in cat_cols passed to fit_model.

        The importlib patch must be placed LAST in the with-block so that
        the fit_model side_effect is still active when run_tree_models executes.
        """
        predict_month = pd.Timestamp("2025-01-01")

        skus = _make_group_skus("smooth")
        sku_to_cluster = {s: "C1" for s in skus}
        sku_to_archetype = {s: "smooth" for s in skus}

        grid = _make_grid(sku_to_cluster, predict_month=predict_month)
        classification_df = _make_classification_df(sku_to_archetype)

        lib_module = _fake_lib_module()

        # Capture the cat_cols passed into fit_model so we can inspect them
        captured_cat_cols: list[list[str]] = []

        def capture_fit(
            model, model_name, X_tr, y_tr, X_val, y_val,
            cat_cols, feature_cols, lib_module_, max_iterations,
        ):
            captured_cat_cols.append(list(cat_cols))

        feature_cols_with_archetype = _FEATURE_COLS + ["archetype"]

        # importlib patch LAST — preserves the fit_model side_effect
        with (
            patch("scripts.algorithm_testing.tree_models.fit_model", side_effect=capture_fit),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch(
                "scripts.algorithm_testing.tree_models.get_feature_columns",
                return_value=feature_cols_with_archetype,
            ),
            patch("scripts.algorithm_testing.tree_models.load_config", return_value=_ALGO_CONFIG),
            patch("scripts.algorithm_testing.tree_models.importlib.import_module", return_value=lib_module),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[predict_month],
                enabled_models={"lgbm": {}},
                classification_df=classification_df,
            )

        assert isinstance(result, pd.DataFrame)
        assert set(result["sku_ck"].tolist()) == set(skus)

        # 'archetype' must appear in the cat_cols passed to fit_model
        assert len(captured_cat_cols) > 0, "fit_model was never called"
        for call_cat_cols in captured_cat_cols:
            assert "archetype" in call_cat_cols, (
                f"'archetype' missing from cat_cols passed to fit_model: {call_cat_cols}"
            )

    def test_archetype_values_match_classification_df(self):
        """Predictions are produced for every SKU in the classification_df."""
        predict_month = pd.Timestamp("2025-01-01")

        smooth_skus = _make_group_skus("smooth")
        lumpy_skus = _make_group_skus("lumpy")
        all_skus = smooth_skus + lumpy_skus

        sku_to_cluster = {s: "C1" for s in smooth_skus} | {s: "C2" for s in lumpy_skus}
        sku_to_archetype = {s: "smooth" for s in smooth_skus} | {s: "lumpy" for s in lumpy_skus}

        grid = _make_grid(sku_to_cluster, predict_month=predict_month)
        classification_df = _make_classification_df(sku_to_archetype)

        lib_module = _fake_lib_module()

        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.load_config", return_value=_ALGO_CONFIG),
            patch("scripts.algorithm_testing.tree_models.importlib.import_module", return_value=lib_module),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[predict_month],
                enabled_models={"lgbm": {}},
                classification_df=classification_df,
            )

        assert set(result["sku_ck"].tolist()) == set(all_skus)
        assert (result["basefcst_pref"] == _FIXED_PRED).all()

    def test_unclassified_dfu_uses_unclassified_archetype(self):
        """A DFU not present in classification_df gets archetype='unclassified'.

        We verify this by ensuring that SKU gets predicted (i.e. falls into
        the 'unclassified' partition group rather than being dropped).
        """
        predict_month = pd.Timestamp("2025-01-01")

        # Two groups: classified (smooth) and unclassified
        classified_skus = _make_group_skus("smooth")
        unclassified_skus = _make_group_skus("unk")
        all_skus = classified_skus + unclassified_skus

        sku_to_cluster = (
            {s: "C1" for s in classified_skus}
            | {s: "C2" for s in unclassified_skus}
        )
        # Only classified_skus appear in classification_df; unclassified_skus get fillna
        classification_df = _make_classification_df({s: "smooth" for s in classified_skus})

        grid = _make_grid(sku_to_cluster, predict_month=predict_month)

        lib_module = _fake_lib_module()

        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.load_config", return_value=_ALGO_CONFIG),
            patch("scripts.algorithm_testing.tree_models.importlib.import_module", return_value=lib_module),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[predict_month],
                enabled_models={"lgbm": {}},
                classification_df=classification_df,
            )

        # Both groups should appear in output
        assert set(result["sku_ck"].tolist()) == set(all_skus), (
            f"Expected all {len(all_skus)} SKUs; missing: "
            f"{set(all_skus) - set(result['sku_ck'].tolist())}"
        )


# ---------------------------------------------------------------------------
# Test 6 — small training group skipped (< _MIN_GROUP_ROWS)
# ---------------------------------------------------------------------------


class TestSmallGroupSkipped:
    """A partition group with fewer than _MIN_GROUP_ROWS training rows must be
    skipped rather than crashing or producing garbage predictions.
    """

    def test_group_below_min_rows_produces_no_output_for_that_group(self):
        """SKUs in a tiny cluster (< 50 train rows) are skipped;
        SKUs in a large cluster produce predictions.
        """
        predict_month = pd.Timestamp("2025-01-01")

        # Large cluster: 6 DFUs × 12 = 72 train rows ≥ 50
        large_skus = _make_group_skus("L", n=6)
        # Tiny cluster: 1 DFU × 3 = 3 train rows < 50
        tiny_skus = ["SKU_S001"]

        large_grid = _make_grid(
            {s: "C_large" for s in large_skus},
            n_train_months=12,
            predict_month=predict_month,
        )
        tiny_grid = _make_grid(
            {s: "C_tiny" for s in tiny_skus},
            n_train_months=3,   # 1 × 3 = 3 < 50
            predict_month=predict_month,
        )
        grid = pd.concat([large_grid, tiny_grid], ignore_index=True)

        lib_module = _fake_lib_module()

        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.load_config", return_value=_ALGO_CONFIG),
            patch("scripts.algorithm_testing.tree_models.importlib.import_module", return_value=lib_module),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[predict_month],
                enabled_models={"lgbm": {}},
                classification_df=None,
            )

        result_skus = set(result["sku_ck"].tolist())

        # Large cluster DFUs must appear in the output
        for sku in large_skus:
            assert sku in result_skus, f"{sku} from large cluster should be in output"

        # Tiny cluster DFU must NOT appear (skipped due to < 50 train rows)
        assert "SKU_S001" not in result_skus, (
            "SKU_S001 from tiny cluster should have been skipped"
        )


# ---------------------------------------------------------------------------
# Test 7 — constant-target guard (avoids model crash)
# ---------------------------------------------------------------------------


class TestConstantTargetGuard:
    """When all training targets in a group are identical, the function must
    return the constant value directly, bypassing the model entirely.
    """

    def test_constant_target_returns_constant_prediction(self):
        """All-constant qty=100 targets produce predictions of 100.0."""
        predict_month = pd.Timestamp("2025-01-01")

        skus = _make_group_skus("G", n=6)  # 6 × 10 = 60 ≥ 50
        rows = []
        for sku in skus:
            for d in pd.date_range("2024-01-01", periods=10, freq="MS"):
                rows.append({
                    "sku_ck": sku, "startdate": d, "qty": 100.0, "ml_cluster": "C1",
                    "qty_lag_1": 100.0, "rolling_mean_3m": 100.0,
                })
            rows.append({
                "sku_ck": sku, "startdate": predict_month, "qty": np.nan,
                "ml_cluster": "C1", "qty_lag_1": 100.0, "rolling_mean_3m": 100.0,
            })
        grid = pd.DataFrame(rows)

        lib_module = _fake_lib_module(fixed_pred=999.0)
        # predict should NOT be called for constant-target path

        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.load_config", return_value=_ALGO_CONFIG),
            patch("scripts.algorithm_testing.tree_models.importlib.import_module", return_value=lib_module),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[predict_month],
                enabled_models={"lgbm": {}},
                classification_df=None,
            )

        assert len(result) == len(skus)
        assert (result["basefcst_pref"] == 100.0).all(), (
            f"Expected constant predictions of 100.0; got {result['basefcst_pref'].tolist()}"
        )

    def test_constant_target_zero_clipped_to_zero(self):
        """Constant target of 0 produces predictions of 0.0 (non-negative)."""
        predict_month = pd.Timestamp("2025-01-01")

        skus = _make_group_skus("Z", n=6)
        rows = []
        for sku in skus:
            for d in pd.date_range("2024-01-01", periods=10, freq="MS"):
                rows.append({
                    "sku_ck": sku, "startdate": d, "qty": 0.0, "ml_cluster": "C1",
                    "qty_lag_1": 0.0, "rolling_mean_3m": 0.0,
                })
            rows.append({
                "sku_ck": sku, "startdate": predict_month, "qty": np.nan,
                "ml_cluster": "C1", "qty_lag_1": 0.0, "rolling_mean_3m": 0.0,
            })
        grid = pd.DataFrame(rows)

        lib_module = _fake_lib_module()

        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.load_config", return_value=_ALGO_CONFIG),
            patch("scripts.algorithm_testing.tree_models.importlib.import_module", return_value=lib_module),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[predict_month],
                enabled_models={"lgbm": {}},
                classification_df=None,
            )

        assert len(result) == len(skus)
        assert (result["basefcst_pref"] == 0.0).all()


# ---------------------------------------------------------------------------
# Test 8 — library import error skips model gracefully
# ---------------------------------------------------------------------------


class TestLibraryImportError:
    """When the ML library is not installed, the model must be skipped cleanly.

    We monkeypatch importlib directly on the module under test (rather than
    using patch() with a side_effect) because patch() itself uses
    importlib.import_module to resolve symbols, and patching it globally
    breaks patch() infrastructure.
    """

    def test_import_error_skips_model_returns_empty(self, monkeypatch):
        """ImportError from importlib.import_module causes the model to be skipped."""
        predict_month = pd.Timestamp("2025-01-01")
        skus = _make_group_skus("G")
        sku_to_cluster = {s: "C1" for s in skus}
        grid = _make_grid(sku_to_cluster, predict_month=predict_month)

        # Fake importlib that raises for any module import
        fake_importlib = SimpleNamespace(
            import_module=MagicMock(side_effect=ImportError("lightgbm not installed"))
        )
        monkeypatch.setattr(_tree_models_mod, "importlib", fake_importlib)

        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.load_config", return_value=_ALGO_CONFIG),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[predict_month],
                enabled_models={"lgbm": {}},
                classification_df=None,
            )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == _PREDICT_COLS

    def test_import_error_is_non_fatal_with_second_valid_model(self, monkeypatch):
        """If lgbm import fails but catboost is also enabled and available,
        catboost still produces output (graceful degradation).
        """
        predict_month = pd.Timestamp("2025-01-01")
        skus = _make_group_skus("G")
        sku_to_cluster = {s: "C1" for s in skus}
        grid = _make_grid(sku_to_cluster, predict_month=predict_month)

        cat_lib = MagicMock()
        cat_model = MagicMock()
        cat_model.predict.side_effect = lambda X: np.full(len(X), _FIXED_PRED)
        cat_model.best_iteration_ = 50
        cat_lib.CatBoostRegressor = MagicMock(return_value=cat_model)

        def selective_import(name):
            if name == "lightgbm":
                raise ImportError("lightgbm not installed")
            return cat_lib

        fake_importlib = SimpleNamespace(import_module=selective_import)
        monkeypatch.setattr(_tree_models_mod, "importlib", fake_importlib)

        catboost_config: dict[str, Any] = {
            "algorithms": {
                "lgbm": {"enabled": False, "model_id": "lgbm_cluster", "n_estimators": 100},
                "catboost": {"enabled": True, "model_id": "catboost_cluster", "iterations": 100},
            },
        }

        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.load_config", return_value=catboost_config),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[predict_month],
                enabled_models={"lgbm": {}, "catboost": {}},
                classification_df=None,
            )

        # catboost should still produce output
        assert len(result) > 0, "catboost should have produced predictions despite lgbm ImportError"
        assert (result["algorithm_id"] == "catboost_cluster").all()


# ---------------------------------------------------------------------------
# Test 9 — algo_config passed explicitly overrides load_config
# ---------------------------------------------------------------------------


class TestAlgoConfigOverride:
    """Passing algo_config explicitly must suppress the load_config() call."""

    def test_explicit_algo_config_used_without_calling_load_config(self):
        """load_config must NOT be called when algo_config is provided."""
        predict_month = pd.Timestamp("2025-01-01")
        skus = _make_group_skus("G")
        sku_to_cluster = {s: "C1" for s in skus}
        grid = _make_grid(sku_to_cluster, predict_month=predict_month)

        lib_module = _fake_lib_module()
        mock_load_config = MagicMock()

        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.load_config", mock_load_config),
            patch("scripts.algorithm_testing.tree_models.importlib.import_module", return_value=lib_module),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[predict_month],
                enabled_models={"lgbm": {}},
                classification_df=None,
                algo_config=_ALGO_CONFIG,   # explicit override
            )

        mock_load_config.assert_not_called()
        assert len(result) > 0

    def test_none_algo_config_triggers_load_config(self, monkeypatch):
        """When algo_config=None, load_forecast_pipeline_config must be called.

        We clear the _config_store cache first so that load_config is not
        short-circuited by a cached value from a previous test run.
        """
        predict_month = pd.Timestamp("2025-01-01")
        skus = _make_group_skus("G")
        sku_to_cluster = {s: "C1" for s in skus}
        grid = _make_grid(sku_to_cluster, predict_month=predict_month)

        lib_module = _fake_lib_module()
        # Pipeline config format: params nested under algorithms.<model_id>.params
        _PIPELINE_CONFIG = {
            "algorithms": {
                "lgbm_cluster": {
                    "type": "tree",
                    "enabled": True,
                    "cluster_strategy": "per_cluster",
                    "params": {"n_estimators": 100},
                },
            },
        }
        mock_load_pipeline = MagicMock(return_value=_PIPELINE_CONFIG)

        # Patch load_forecast_pipeline_config in the module under test
        monkeypatch.setattr(_tree_models_mod, "load_forecast_pipeline_config", mock_load_pipeline)

        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.importlib.import_module", return_value=lib_module),
        ):
            run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[predict_month],
                enabled_models={"lgbm": {}},
                classification_df=None,
                algo_config=None,
            )

        mock_load_pipeline.assert_called_once()


# ---------------------------------------------------------------------------
# Test 10 — output schema is always consistent
# ---------------------------------------------------------------------------


class TestOutputSchema:
    """run_tree_models must always return the four standard columns, regardless
    of the path taken (normal, empty, unknown model, etc.).
    """

    @pytest.mark.parametrize("use_classification_df", [False, True])
    def test_output_columns_always_present(self, use_classification_df):
        predict_month = pd.Timestamp("2025-01-01")
        skus = _make_group_skus("G")
        sku_to_cluster = {s: "C1" for s in skus}
        grid = _make_grid(sku_to_cluster, predict_month=predict_month)

        classification_df = None
        if use_classification_df:
            classification_df = _make_classification_df({s: "smooth" for s in skus})

        lib_module = _fake_lib_module()

        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.load_config", return_value=_ALGO_CONFIG),
            patch("scripts.algorithm_testing.tree_models.importlib.import_module", return_value=lib_module),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[predict_month],
                enabled_models={"lgbm": {}},
                classification_df=classification_df,
            )

        for col in _PREDICT_COLS:
            assert col in result.columns, f"Missing column '{col}' in output"

    def test_output_columns_in_empty_result(self):
        """Empty result (no predict rows) must still carry the four columns."""
        skus = _make_group_skus("G")
        sku_to_cluster = {s: "C1" for s in skus}
        grid = _make_grid(sku_to_cluster)

        lib_module = _fake_lib_module()

        with (
            patch("scripts.algorithm_testing.tree_models.fit_model"),
            patch("scripts.algorithm_testing.tree_models.get_best_iteration", return_value=50),
            patch("scripts.algorithm_testing.tree_models.get_feature_columns", return_value=_FEATURE_COLS),
            patch("scripts.algorithm_testing.tree_models.load_config", return_value=_ALGO_CONFIG),
            patch("scripts.algorithm_testing.tree_models.importlib.import_module", return_value=lib_module),
        ):
            result = run_tree_models(
                grid=grid,
                train_end=pd.Timestamp("2024-12-01"),
                predict_months=[pd.Timestamp("2099-01-01")],
                enabled_models={"lgbm": {}},
                classification_df=None,
            )

        assert list(result.columns) == _PREDICT_COLS
