"""Fail-closed coverage tests for LightGBM cluster backtests.

Every row emitted by the LightGBM backtest must come from a fitted LightGBM
model. Undersized or unmatched clusters are rejected instead of being
relabelled seasonal-naive or zero forecasts.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from common.core.constants import MIN_CLUSTER_ROWS


def _make_train_df(
    cluster_label: str,
    n_rows: int,
    *,
    base_qty: float = 100.0,
) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="MS")
    return pd.DataFrame(
        {
            "sku_ck": [f"SKU_{cluster_label}_{i:03d}" for i in range(n_rows)],
            "item_id": [f"ITEM_{cluster_label}_{i:03d}" for i in range(n_rows)],
            "customer_group": ["CG1"] * n_rows,
            "loc": ["L1"] * n_rows,
            "startdate": dates,
            "qty": [base_qty + date.month for date in dates],
            "ml_cluster": [cluster_label] * n_rows,
            "month": [date.month for date in dates],
        }
    )


def _make_pred_df(
    cluster_label: str,
    n_rows: int,
    *,
    start_date: str = "2025-01-01",
) -> pd.DataFrame:
    dates = pd.date_range(start_date, periods=n_rows, freq="MS")
    return pd.DataFrame(
        {
            "sku_ck": [f"PRED_{cluster_label}_{i:03d}" for i in range(n_rows)],
            "item_id": [f"PITEM_{cluster_label}_{i:03d}" for i in range(n_rows)],
            "customer_group": ["CG1"] * n_rows,
            "loc": ["L1"] * n_rows,
            "startdate": dates,
            "ml_cluster": [cluster_label] * n_rows,
            "month": [date.month for date in dates],
        }
    )


def _registry() -> dict:
    return {
        "needs_cat_dtype_cast": False,
        "constant_target_guard": True,
        "iter_param": "n_estimators",
        "fit_extras_per_cluster": lambda params, iter_param: {},
    }


def _single_cluster_kwargs() -> dict:
    return {
        "model_name": "lgbm",
        "model_class": MagicMock,
        "lib_module": MagicMock(),
        "needs_cat_dtype_cast": False,
        "constant_target_guard": True,
        "iter_param": "n_estimators",
        "fit_extras": {},
    }


def test_single_cluster_below_fit_floor_is_rejected() -> None:
    """An undersized cluster cannot emit a heuristic under the LightGBM ID."""
    from scripts.ml.run_backtest import _train_single_cluster

    train = _make_train_df("tiny", MIN_CLUSTER_ROWS - 1)
    pred = _make_pred_df("tiny", 3)

    with pytest.raises(ValueError, match=r"too few training rows.*tiny"):
        _train_single_cluster(
            "tiny",
            1,
            1,
            train,
            pred,
            ["month"],
            [],
            {},
            **_single_cluster_kwargs(),
        )


def test_prediction_cluster_without_training_cluster_is_rejected() -> None:
    """A promoted label absent from training cannot be silently dropped."""
    from scripts.ml.run_backtest import train_and_predict_per_cluster

    train = _make_train_df("known", MIN_CLUSTER_ROWS + 10)
    pred = pd.concat(
        [_make_pred_df("known", 2), _make_pred_df("not_trained", 2)],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match=r"no matching training cluster.*not_trained"):
        train_and_predict_per_cluster(
            train,
            pred,
            ["month"],
            [],
            {"n_estimators": 100},
            model_name="lgbm",
            registry=_registry(),
            model_class=MagicMock,
            lib_module=MagicMock(),
        )


def test_null_prediction_cluster_is_rejected() -> None:
    """Rows without a promoted label cannot be emitted as zero forecasts."""
    from scripts.ml.run_backtest import train_and_predict_per_cluster

    train = _make_train_df("known", MIN_CLUSTER_ROWS + 10)
    pred = _make_pred_df("known", 2)
    pred.loc[pred.index[0], "ml_cluster"] = None

    with pytest.raises(ValueError, match="null cluster assignment"):
        train_and_predict_per_cluster(
            train,
            pred,
            ["month"],
            [],
            {"n_estimators": 100},
            model_name="lgbm",
            registry=_registry(),
            model_class=MagicMock,
            lib_module=MagicMock(),
        )


def test_cluster_at_fit_floor_uses_registry_model_for_all_rows() -> None:
    """A valid cluster still trains and predicts through the retained model."""
    from scripts.ml.run_backtest import _train_single_cluster

    train = _make_train_df("valid", MIN_CLUSTER_ROWS)
    pred = _make_pred_df("valid", 3)
    n_val = max(1, int(MIN_CLUSTER_ROWS * 0.20))
    eval_model = MagicMock()
    eval_model.predict.return_value = np.full(n_val, 100.0)
    final_model = MagicMock()
    final_model.predict.return_value = np.full(len(pred), 100.0)

    demand_stats = {
        "mean_demand": 100.0,
        "cv_demand": 0.1,
        "zero_demand_pct": 0.0,
        "seasonal_amplitude": 0.0,
    }
    with (
        patch(
            "scripts.ml.run_backtest.build_tree_model",
            side_effect=[eval_model, final_model],
        ) as build,
        patch("scripts.ml.run_backtest.fit_model"),
        patch("scripts.ml.run_backtest.fit_final_model") as final_fit,
        patch("scripts.ml.run_backtest.get_best_iteration", return_value=100),
        patch(
            "scripts.ml.run_backtest.compute_cluster_demand_stats",
            return_value=demand_stats,
        ),
        patch(
            "scripts.ml.run_backtest.resolve_cluster_params",
            return_value=({"n_estimators": 100}, "default"),
        ),
    ):
        cluster, result, model, meta = _train_single_cluster(
            "valid",
            1,
            1,
            train,
            pred,
            ["month"],
            [],
            {"n_estimators": 100},
            **_single_cluster_kwargs(),
        )

    assert cluster == "valid"
    assert len(result) == len(pred)
    assert model is final_model
    assert meta["train_rows"] == MIN_CLUSTER_ROWS
    assert build.call_args_list[0].args == ("lgbm", {"n_estimators": 100})
    assert build.call_args_list[1].args == ("lgbm", {"n_estimators": 100})
    final_fit.assert_called_once()


def test_empty_prediction_partition_is_skipped() -> None:
    """A partition with no requested rows remains an explicit no-op."""
    from scripts.ml.run_backtest import _train_single_cluster

    cluster, result, model, meta = _train_single_cluster(
        "unused",
        1,
        1,
        _make_train_df("unused", 1),
        _make_pred_df("unused", 0),
        ["month"],
        [],
        {},
        **_single_cluster_kwargs(),
    )

    assert cluster == "unused"
    assert result is None
    assert model is None
    assert meta is None
