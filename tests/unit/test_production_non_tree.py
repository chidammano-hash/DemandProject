"""Production inference tests for the four canonical non-tree models."""

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from common.core.constants import FORECAST_QTY_COL
from common.ml.neural_forecast import FittedNeuralModel
from common.ml.production_non_tree import (
    complete_monthly_history,
    run_canonical_non_tree_forecast,
)


def _sales() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sku_ck": ["sku-1", "sku-1", "sku-2"],
            "startdate": pd.to_datetime(["2026-03-01", "2026-05-01", "2026-05-01"]),
            "qty": [10.0, 30.0, 5.0],
        }
    )
    frame.attrs["history_end"] = pd.Timestamp("2026-06-01")
    return frame


def _attrs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_ck": ["sku-1", "sku-2"],
            "item_id": ["item-1", "item-2"],
            "customer_group": ["group-1", "group-2"],
            "loc": ["loc-1", "loc-2"],
            "ml_cluster": ["smooth", "lumpy"],
            "brand": ["a", "b"],
            "region": ["r", "r"],
            "abc_vol": ["A", "B"],
            "execution_lag": [0, 1],
            "total_lt": [1, 2],
        }
    )


def _targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_ck": ["sku-1"],
            "item_id": ["item-1"],
            "customer_group": ["group-1"],
            "loc": ["loc-1"],
            "cluster_id": ["smooth"],
        }
    )


def _predictions(model_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_ck": ["sku-1", "sku-1"],
            "startdate": pd.to_datetime(["2026-07-01", "2026-08-01"]),
            FORECAST_QTY_COL: [40.0, 44.0],
            "algorithm_id": [model_id, model_id],
        }
    )


def _neural_params(*, min_history: int = 3) -> dict[str, object]:
    return {
        "h": 6,
        "input_size": 24,
        "max_steps": 10,
        "batch_size": 2,
        "learning_rate": 0.001,
        "scaler_type": "standard",
        "early_stop_patience_steps": -1,
        "min_history": min_history,
        "random_seed": 42,
        "start_padding_enabled": True,
        "val_size": 0,
        "deterministic": True,
    }


def _fitted_neural(
    model_id: str = "nhits",
    *,
    min_history: int = 3,
) -> FittedNeuralModel:
    return FittedNeuralModel(
        neural_forecast=MagicMock(h=6),
        model_id=model_id,
        fitted_horizon=6,
        min_history=min_history,
        training_dfu_count=2,
    )


def test_complete_history_fills_internal_and_trailing_closed_months():
    result = complete_monthly_history(_sales())

    sku_1 = result[result["sku_ck"] == "sku-1"].set_index("startdate")["qty"]
    assert list(sku_1.index) == list(pd.date_range("2026-03-01", "2026-06-01", freq="MS"))
    assert sku_1.loc[pd.Timestamp("2026-04-01")] == 0.0
    assert sku_1.loc[pd.Timestamp("2026-06-01")] == 0.0


def test_complete_history_rejects_sales_after_configured_closed_month():
    sales = _sales()
    future = pd.DataFrame(
        {
            "sku_ck": ["sku-1"],
            "startdate": pd.to_datetime(["2026-07-01"]),
            "qty": [99.0],
        }
    )
    sales = pd.concat([sales, future], ignore_index=True)
    sales.attrs["history_end"] = pd.Timestamp("2026-06-01")

    with pytest.raises(ValueError, match="after the configured closed month"):
        complete_monthly_history(sales)


def test_complete_history_rejects_target_with_no_declared_first_sale():
    sales = _sales()
    sales["first_sale_month"] = pd.NaT

    with pytest.raises(ValueError, match="no observed first sale"):
        complete_monthly_history(sales)


def test_neural_production_path_uses_real_adapter_and_preserves_lineage(monkeypatch):
    calls = []

    def fake_run(fitted, sales_df, predict_months):
        calls.append((fitted, sales_df, predict_months))
        return _predictions("nhits")

    monkeypatch.setattr("common.ml.production_non_tree.predict_neural_model", fake_run)
    months = [pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-01")]

    rows = run_canonical_non_tree_forecast(
        model_id="nhits",
        sales_df=_sales(),
        dfu_attrs=_attrs(),
        item_attrs=pd.DataFrame(),
        target_dfus=_targets(),
        predict_months=months,
        params=_neural_params(),
        forecast_month_generated=date(2026, 7, 1),
        run_id="run-1",
        sigma_lookup={},
        ci_cfg=None,
        fitted_neural_model=_fitted_neural(),
    )

    assert len(calls) == 1
    assert len(rows) == 2
    assert {row["model_id"] for row in rows} == {"nhits"}
    assert {row["customer_group"] for row in rows} == {"group-1"}
    assert [row["horizon_months"] for row in rows] == [1, 2]
    assert all(row["is_recursive"] is False for row in rows)
    assert all(row["lag_source"] == "actual" for row in rows)


def test_adapter_coverage_gap_fails_closed(monkeypatch):
    incomplete = _predictions("nbeats").iloc[:1]
    monkeypatch.setattr(
        "common.ml.production_non_tree.predict_neural_model",
        lambda *_args, **_kwargs: incomplete,
    )

    with pytest.raises(RuntimeError, match="missing 1 required DFU-month"):
        run_canonical_non_tree_forecast(
            model_id="nbeats",
            sales_df=_sales(),
            dfu_attrs=_attrs(),
            item_attrs=pd.DataFrame(),
            target_dfus=_targets(),
            predict_months=[pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-01")],
            params=_neural_params(),
            forecast_month_generated=date(2026, 7, 1),
            run_id="run-1",
            sigma_lookup={},
            ci_cfg=None,
            fitted_neural_model=_fitted_neural("nbeats"),
        )


def test_adapter_output_requires_explicit_model_identity(monkeypatch):
    predictions = _predictions("nhits").drop(columns="algorithm_id")
    monkeypatch.setattr(
        "common.ml.production_non_tree.predict_neural_model",
        lambda *_args, **_kwargs: predictions,
    )

    with pytest.raises(RuntimeError, match=r"missing required columns.*algorithm_id"):
        run_canonical_non_tree_forecast(
            model_id="nhits",
            sales_df=_sales(),
            dfu_attrs=_attrs(),
            item_attrs=pd.DataFrame(),
            target_dfus=_targets(),
            predict_months=[pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-01")],
            params=_neural_params(),
            forecast_month_generated=date(2026, 7, 1),
            run_id="run-1",
            sigma_lookup={},
            ci_cfg=None,
            fitted_neural_model=_fitted_neural(),
        )


def test_target_sku_identity_must_match_dfu_attributes():
    targets = _targets()
    targets.loc[0, "customer_group"] = "wrong-group"

    with pytest.raises(ValueError, match="does not match DFU attributes"):
        run_canonical_non_tree_forecast(
            model_id="nhits",
            sales_df=_sales(),
            dfu_attrs=_attrs(),
            item_attrs=pd.DataFrame(),
            target_dfus=targets,
            predict_months=[pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-01")],
            params=_neural_params(),
            forecast_month_generated=date(2026, 7, 1),
            run_id="run-1",
            sigma_lookup={},
            ci_cfg=None,
        )


def test_unrelated_invalid_attribute_row_does_not_block_target(monkeypatch):
    attrs = pd.concat(
        [
            _attrs(),
            pd.DataFrame(
                {
                    "sku_ck": ["unrelated"],
                    "item_id": ["other"],
                    "customer_group": [None],
                    "loc": ["other-loc"],
                }
            ),
        ],
        ignore_index=True,
    )
    monkeypatch.setattr(
        "common.ml.production_non_tree.predict_neural_model",
        lambda *_args, **_kwargs: _predictions("nhits"),
    )

    rows = run_canonical_non_tree_forecast(
        model_id="nhits",
        sales_df=_sales(),
        dfu_attrs=attrs,
        item_attrs=pd.DataFrame(),
        target_dfus=_targets(),
        predict_months=[pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-01")],
        params=_neural_params(),
        forecast_month_generated=date(2026, 7, 1),
        run_id="run-1",
        sigma_lookup={},
        ci_cfg=None,
        fitted_neural_model=_fitted_neural(),
    )

    assert len(rows) == 2


def test_target_with_no_sales_fails_before_expensive_model_run(monkeypatch):
    targets = _targets()
    targets.loc[0, "sku_ck"] = "sku-missing"
    attrs = _attrs()
    attrs.loc[0, "sku_ck"] = "sku-missing"
    called = False

    def fake_run(*_args, **_kwargs):
        nonlocal called
        called = True
        return _predictions("nhits")

    monkeypatch.setattr("common.ml.production_non_tree.predict_neural_model", fake_run)

    with pytest.raises(ValueError, match="no sales history"):
        run_canonical_non_tree_forecast(
            model_id="nhits",
            sales_df=_sales(),
            dfu_attrs=attrs,
            item_attrs=pd.DataFrame(),
            target_dfus=targets,
            predict_months=[pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-01")],
            params=_neural_params(),
            forecast_month_generated=date(2026, 7, 1),
            run_id="run-1",
            sigma_lookup={},
            ci_cfg=None,
            fitted_neural_model=_fitted_neural(),
        )

    assert called is False


def test_history_must_end_immediately_before_record_month():
    sales = _sales()
    sales.attrs["history_end"] = pd.Timestamp("2026-05-01")
    sales = sales[sales["startdate"] <= pd.Timestamp("2026-05-01")].copy()
    sales.attrs["history_end"] = pd.Timestamp("2026-05-01")
    with pytest.raises(ValueError, match="must end in 2026-06"):
        run_canonical_non_tree_forecast(
            model_id="nhits",
            sales_df=sales,
            dfu_attrs=_attrs(),
            item_attrs=pd.DataFrame(),
            target_dfus=_targets(),
            predict_months=[pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-01")],
            params=_neural_params(),
            forecast_month_generated=date(2026, 7, 1),
            run_id="run-1",
            sigma_lookup={},
            ci_cfg=None,
        )


def test_mstl_uses_configured_model_params_without_unconfigured_worker_key(monkeypatch):
    captured: dict[str, object] = {}

    def fake_mstl(sales_df, predict_months, **kwargs):
        captured["sales_skus"] = set(sales_df["sku_ck"])
        captured["months"] = predict_months
        captured["kwargs"] = kwargs
        return _predictions("mstl")

    monkeypatch.setattr("common.ml.production_non_tree.run_mstl", fake_mstl)

    rows = run_canonical_non_tree_forecast(
        model_id="mstl",
        sales_df=_sales(),
        dfu_attrs=_attrs(),
        item_attrs=pd.DataFrame(),
        target_dfus=_targets(),
        predict_months=[pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-01")],
        params={"season_length": 12, "min_history": 3},
        forecast_month_generated=date(2026, 7, 1),
        run_id="run-1",
        sigma_lookup={},
        ci_cfg=None,
    )

    assert len(rows) == 2
    assert captured["sales_skus"] == {"sku-1"}
    assert captured["kwargs"] == {
        "season_length": 12,
        "min_history": 3,
        "n_workers": 1,
    }


def test_chronos_normalizes_sku_identity_and_builds_target_only_features(monkeypatch):
    sales = _sales()
    sales["sku_ck"] = sales["sku_ck"].map({"sku-1": 101, "sku-2": 202})
    attrs = _attrs()
    attrs["sku_ck"] = [101, 202]
    targets = _targets()
    targets["sku_ck"] = [101]
    captured: dict[str, object] = {}

    def fake_features(sales_df, dfu_attrs, item_attrs, all_months, cat_dtype):
        captured["feature_sales"] = sales_df.copy()
        captured["feature_attrs"] = dfu_attrs.copy()
        captured["item_attrs"] = item_attrs.copy()
        captured["all_months"] = all_months
        captured["cat_dtype"] = cat_dtype
        return pd.DataFrame(
            {
                "sku_ck": ["101"],
                "startdate": [pd.Timestamp("2026-06-01")],
            }
        )

    def fake_chronos(sales_df, predict_months, params, feature_grid):
        captured["chronos_sales"] = sales_df.copy()
        captured["params"] = params
        captured["feature_grid"] = feature_grid
        return _predictions("chronos2_enriched").assign(sku_ck="101")

    monkeypatch.setattr("common.ml.production_non_tree.build_feature_matrix", fake_features)
    monkeypatch.setattr("common.ml.production_non_tree.run_chronos2_enriched", fake_chronos)

    rows = run_canonical_non_tree_forecast(
        model_id="chronos2_enriched",
        sales_df=sales,
        dfu_attrs=attrs,
        item_attrs=pd.DataFrame(
            {
                "item_id": ["item-1", "item-2"],
                "case_weight": [1.0, 2.0],
            }
        ),
        target_dfus=targets,
        predict_months=[pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-01")],
        params={"device": "cpu", "batch_size": 2, "prediction_length": 6},
        forecast_month_generated=date(2026, 7, 1),
        run_id="run-1",
        sigma_lookup={},
        ci_cfg=None,
    )

    assert len(rows) == 2
    assert set(captured["feature_sales"]["sku_ck"]) == {"101"}
    assert set(captured["feature_attrs"]["sku_ck"]) == {"101"}
    assert set(captured["item_attrs"]["item_id"]) == {"item-1"}
    assert captured["cat_dtype"] == "str"
    assert captured["params"] == {
        "device": "cpu",
        "batch_size": 2,
        "prediction_length": 6,
    }


def test_non_tree_rows_compute_configured_confidence_intervals(monkeypatch):
    monkeypatch.setattr(
        "common.ml.production_non_tree.predict_neural_model",
        lambda *_args, **_kwargs: _predictions("nhits"),
    )

    rows = run_canonical_non_tree_forecast(
        model_id="nhits",
        sales_df=_sales(),
        dfu_attrs=_attrs(),
        item_attrs=pd.DataFrame(),
        target_dfus=_targets(),
        predict_months=[pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-01")],
        params=_neural_params(),
        forecast_month_generated=date(2026, 7, 1),
        run_id="run-1",
        sigma_lookup={("item-1", "loc-1"): 10.0},
        ci_cfg={"z_lower": 1.0, "z_upper": 1.0, "horizon_scaling": "none"},
        fitted_neural_model=_fitted_neural(),
    )

    assert rows[0]["forecast_qty_lower"] == 30.0
    assert rows[0]["forecast_qty_upper"] == 50.0
    assert rows[1]["forecast_qty_lower"] == 34.0
    assert rows[1]["forecast_qty_upper"] == 54.0


def test_unknown_non_tree_model_is_rejected():
    with pytest.raises(ValueError, match="Unsupported canonical non-tree model"):
        run_canonical_non_tree_forecast(
            model_id="retired_model",
            sales_df=_sales(),
            dfu_attrs=_attrs(),
            item_attrs=pd.DataFrame(),
            target_dfus=_targets(),
            predict_months=[pd.Timestamp("2026-07-01")],
            params={},
            forecast_month_generated=date(2026, 7, 1),
            run_id="run-1",
            sigma_lookup={},
            ci_cfg=None,
        )
