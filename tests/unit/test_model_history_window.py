"""Backtest and production adapters must use the same bounded history window."""

import pandas as pd
import pytest

from common.core.constants import FORECAST_QTY_COL


def test_chronos_backtest_applies_the_production_history_window() -> None:
    import inspect

    from common.ml.foundation_backtest import run_foundation_backtest

    source = inspect.getsource(run_foundation_backtest)
    assert 'cfg["production_forecast"]["lookback_months"]' in source
    assert "select_bounded_history(" in source


def test_bounded_history_keeps_exact_calendar_window_per_cutoff() -> None:
    from common.ml.monthly_history import select_bounded_history

    sales = pd.DataFrame(
        {
            "sku_ck": "sku-1",
            "startdate": pd.date_range("2021-01-01", periods=60, freq="MS"),
            "qty": range(60),
        }
    )

    bounded = select_bounded_history(
        sales,
        history_end=pd.Timestamp("2025-12-01"),
        lookback_months=36,
    )

    assert len(bounded) == 36
    assert bounded["startdate"].min() == pd.Timestamp("2023-01-01")
    assert bounded["startdate"].max() == pd.Timestamp("2025-12-01")
    assert bounded.attrs["history_end"] == pd.Timestamp("2025-12-01")


@pytest.mark.parametrize("lookback_months", [0, -1])
def test_bounded_history_rejects_invalid_window(lookback_months: int) -> None:
    from common.ml.monthly_history import select_bounded_history

    with pytest.raises(ValueError, match="lookback_months"):
        select_bounded_history(
            pd.DataFrame(
                {
                    "sku_ck": ["sku-1"],
                    "startdate": [pd.Timestamp("2025-12-01")],
                    "qty": [1.0],
                }
            ),
            history_end=pd.Timestamp("2025-12-01"),
            lookback_months=lookback_months,
        )


def test_chronos_covariates_are_built_from_the_same_bounded_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from common.ml.foundation_backtest import PerTimeframeContext
    from scripts.ml import run_backtest_chronos2_enriched as adapter

    train_sales = pd.DataFrame(
        {
            "sku_ck": ["sku-1"] * 36,
            "startdate": pd.date_range("2023-01-01", periods=36, freq="MS"),
            "qty": range(36),
        }
    )
    train_sales.attrs["history_end"] = pd.Timestamp("2025-12-01")
    dfu_attrs = pd.DataFrame(
        {
            "sku_ck": ["sku-1", "outside-window"],
            "item_id": ["item-1", "item-2"],
            "customer_group": ["customer-1", "customer-2"],
            "loc": ["loc-1", "loc-2"],
        }
    )
    captured: dict[str, object] = {}

    def fake_build_feature_matrix(
        sales: pd.DataFrame,
        attrs: pd.DataFrame,
        _item_attrs: pd.DataFrame,
        months: list[pd.Timestamp],
        *,
        cat_dtype: str,
    ) -> pd.DataFrame:
        captured["feature_sales"] = sales.copy()
        captured["feature_skus"] = attrs["sku_ck"].tolist()
        captured["feature_months"] = months
        captured["cat_dtype"] = cat_dtype
        return sales[["sku_ck", "startdate"]].assign(qty_lag_1=0.0)

    def fake_run_chronos(
        sales: pd.DataFrame,
        predict_months: list[pd.Timestamp],
        _params: dict,
        *,
        feature_grid: pd.DataFrame,
    ) -> pd.DataFrame:
        captured["model_sales"] = sales.copy()
        captured["feature_grid"] = feature_grid.copy()
        return pd.DataFrame(
            {
                "sku_ck": ["sku-1"],
                "startdate": [predict_months[0]],
                FORECAST_QTY_COL: [1.0],
                "algorithm_id": ["chronos2_enriched"],
            }
        )

    monkeypatch.setattr(adapter, "build_feature_matrix", fake_build_feature_matrix)
    monkeypatch.setattr(adapter, "run_chronos2_enriched", fake_run_chronos)

    result = adapter._per_timeframe_hook(
        PerTimeframeContext(
            tf={"train_end": pd.Timestamp("2025-12-01")},
            ti=0,
            n_total=1,
            train_sales=train_sales,
            predict_months=[pd.Timestamp("2026-01-01")],
            model_params={"prediction_length": 1},
            label="tf-1",
        ),
        (dfu_attrs, pd.DataFrame()),
    )

    assert not result.empty
    assert captured["feature_skus"] == ["sku-1"]
    assert captured["feature_months"] == list(
        pd.date_range("2023-01-01", periods=36, freq="MS")
    )
    assert captured["cat_dtype"] == "str"
    assert len(captured["feature_sales"]) == 36
    assert len(captured["model_sales"]) == 36
