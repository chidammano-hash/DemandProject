"""Production NHITS/NBEATS must infer from an immutable global fit."""

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from common.core.constants import FORECAST_QTY_COL
from common.ml.neural_forecast import FittedNeuralModel
from common.ml.production_non_tree import run_canonical_non_tree_forecast


def _sales() -> pd.DataFrame:
    months = pd.date_range("2025-07-01", periods=12, freq="MS")
    frame = pd.concat(
        [
            pd.DataFrame(
                {"sku_ck": sku_ck, "startdate": months, "qty": quantity}
            )
            for sku_ck, quantity in (("sku-1", 10.0), ("sku-2", 20.0))
        ],
        ignore_index=True,
    )
    frame.attrs["history_end"] = pd.Timestamp("2026-06-01")
    return frame


def _attrs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sku_ck": "sku-1",
                "item_id": "item-1",
                "customer_group": "group-1",
                "loc": "loc-1",
            },
            {
                "sku_ck": "sku-2",
                "item_id": "item-2",
                "customer_group": "group-2",
                "loc": "loc-2",
            },
        ]
    )


def _target() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sku_ck": "sku-1",
                "item_id": "item-1",
                "customer_group": "group-1",
                "loc": "loc-1",
                "cluster_id": "stable",
            }
        ]
    )


def _fitted(model_id: str = "nhits") -> FittedNeuralModel:
    return FittedNeuralModel(
        neural_forecast=MagicMock(h=6),
        model_id=model_id,
        fitted_horizon=6,
        min_history=12,
        training_dfu_count=50_000,
    )


def _params() -> dict[str, object]:
    return {
        "h": 6,
        "input_size": 24,
        "max_steps": 500,
        "batch_size": 32,
        "learning_rate": 0.001,
        "scaler_type": "standard",
        "early_stop_patience_steps": -1,
        "min_history": 12,
        "random_seed": 42,
        "start_padding_enabled": True,
        "val_size": 0,
        "deterministic": True,
    }


def test_production_neural_uses_the_fitted_model_and_only_target_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _predict(fitted, sales_df, predict_months):
        captured["fitted"] = fitted
        captured["sales"] = sales_df.copy()
        predictions = pd.DataFrame(
            {
                "sku_ck": "sku-1",
                "startdate": predict_months,
                FORECAST_QTY_COL: 7.0,
                "algorithm_id": "nhits",
            }
        )
        predictions.attrs["fitted_horizon"] = 6
        predictions.attrs["prediction_horizon"] = len(predict_months)
        return predictions

    monkeypatch.setattr(
        "common.ml.production_non_tree.predict_neural_model",
        _predict,
        raising=False,
    )
    months = list(pd.date_range("2026-07-01", periods=8, freq="MS"))
    fitted = _fitted()

    rows = run_canonical_non_tree_forecast(
        model_id="nhits",
        sales_df=_sales(),
        dfu_attrs=_attrs(),
        item_attrs=pd.DataFrame(),
        target_dfus=_target(),
        predict_months=months,
        params=_params(),
        forecast_month_generated=date(2026, 7, 1),
        run_id="immutable-neural-run",
        sigma_lookup={},
        ci_cfg=None,
        fitted_neural_model=fitted,
    )

    assert captured["fitted"] is fitted
    assert set(captured["sales"]["sku_ck"]) == {"sku-1"}
    assert [row["is_recursive"] for row in rows] == [False] * 6 + [True] * 2
    assert [row["lag_source"] for row in rows] == ["actual"] * 6 + ["predicted"] * 2


def test_production_neural_requires_a_matching_fitted_artifact() -> None:
    common = {
        "model_id": "nhits",
        "sales_df": _sales(),
        "dfu_attrs": _attrs(),
        "item_attrs": pd.DataFrame(),
        "target_dfus": _target(),
        "predict_months": [pd.Timestamp("2026-07-01")],
        "params": _params(),
        "forecast_month_generated": date(2026, 7, 1),
        "run_id": "missing-neural-artifact",
        "sigma_lookup": {},
        "ci_cfg": None,
    }

    with pytest.raises(RuntimeError, match="fitted nhits production artifact"):
        run_canonical_non_tree_forecast(**common, fitted_neural_model=None)
    with pytest.raises(RuntimeError, match="artifact model nbeats does not match nhits"):
        run_canonical_non_tree_forecast(
            **common,
            fitted_neural_model=_fitted("nbeats"),
        )
