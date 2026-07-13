"""Production recursive smoothing must match the evaluated LightGBM contract."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from scripts.forecasting.generate_production_forecasts import (
    build_inference_grid,
    generate_forecasts_batch,
)
from tests.unit.test_production_forecast import (
    _make_dfu_attrs,
    _make_sales,
)


def test_batch_recursive_lag_smoothing_starts_after_third_prediction() -> None:
    sales = _make_sales(n_months=24, start="2024-01-01")
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=4)
    grid.loc[3, "qty_lag_1"] = 40.0
    model = MagicMock()
    model.booster_ = None
    model.predict.side_effect = [
        np.array([10.0]),
        np.array([20.0]),
        np.array([30.0]),
        np.array([40.0]),
    ]
    artifact = {
        "model": model,
        "feature_cols": ["qty_lag_1", "qty_lag_2"],
        "recursive_training": {
            "enabled": True,
            "noise_enabled": True,
            "noise_pct": 0.05,
            "lag_smooth": 0.25,
        },
    }

    generate_forecasts_batch(
        artifact=artifact,
        dfu_list=[
            (
                {
                    "item_id": "ITEM001",
                    "customer_group": "GROUP1",
                    "loc": "LOC1",
                    "cluster_id": 2,
                },
                grid,
            )
        ],
        horizon=4,
        forecast_month_generated="2026-01",
        run_id="smoothing-contract",
        model_id="lgbm_cluster",
    )

    second_step = model.predict.call_args_list[1].args[0].iloc[0]
    third_step = model.predict.call_args_list[2].args[0].iloc[0]
    fourth_step = model.predict.call_args_list[3].args[0].iloc[0]
    assert second_step["qty_lag_1"] == 10.0
    assert third_step["qty_lag_1"] == 20.0
    assert fourth_step["qty_lag_1"] == 32.5  # 75% step-3 prediction + 25% prior lag


def test_batch_rejects_invalid_recursive_lag_smoothing_contract() -> None:
    sales = _make_sales(n_months=24, start="2024-01-01")
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=1)
    artifact = {
        "model": MagicMock(),
        "feature_cols": ["qty_lag_1"],
        "recursive_training": {"lag_smooth": 1.5},
    }

    try:
        generate_forecasts_batch(
            artifact=artifact,
            dfu_list=[
                (
                    {
                        "item_id": "ITEM001",
                        "customer_group": "GROUP1",
                        "loc": "LOC1",
                        "cluster_id": 2,
                    },
                    grid,
                )
            ],
            horizon=1,
            forecast_month_generated="2026-01",
            run_id="invalid-smoothing-contract",
            model_id="lgbm_cluster",
        )
    except ValueError as exc:
        assert "lag_smooth" in str(exc)
    else:  # pragma: no cover - production must fail closed
        raise AssertionError("invalid recursive lag smoothing was accepted")


def test_checksummed_tree_artifact_requires_recursive_contract() -> None:
    sales = _make_sales(n_months=24, start="2024-01-01")
    attrs = _make_dfu_attrs()
    grid = build_inference_grid("ITEM001", "LOC1", 2, sales, attrs, horizon=1)
    artifact = {
        "model": MagicMock(),
        "feature_cols": ["qty_lag_1"],
        "config_checksum": "a" * 64,
    }

    try:
        generate_forecasts_batch(
            artifact=artifact,
            dfu_list=[
                (
                    {
                        "item_id": "ITEM001",
                        "customer_group": "GROUP1",
                        "loc": "LOC1",
                        "cluster_id": 2,
                    },
                    grid,
                )
            ],
            horizon=1,
            forecast_month_generated="2026-01",
            run_id="missing-smoothing-contract",
            model_id="lgbm_cluster",
        )
    except ValueError as exc:
        assert "recursive_training" in str(exc)
    else:  # pragma: no cover - production must fail closed
        raise AssertionError("checksummed artifact omitted its recursive contract")
