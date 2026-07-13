"""API tests for the production-bound per-DFU LightGBM SHAP endpoint."""

from __future__ import annotations

import datetime
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport

from tests.api.conftest import make_pool

_HISTORY_END = datetime.date(2024, 5, 1)
_HISTORY_START = datetime.date(2023, 1, 1)
_FEATURE_COLS = [
    "qty_lag_1",
    "qty_lag_2",
    "rolling_mean_3m",
    "rolling_std_3m",
    "month",
    "quarter",
]
_ARTIFACT = {
    "model": MagicMock(),
    "feature_cols": _FEATURE_COLS,
    "categorical_encoders": {},
}

# sku_ck, ml_cluster, execution_lag, total_lt, brand, region, abc_vol,
# customer_group, bpc, item_proof, case_weight, item_location_dfu_count
_DFU_ROW = (
    "sku-1",
    "0",
    0,
    14,
    "brand_a",
    "NE",
    "A",
    "grp1",
    12.0,
    40.0,
    15.0,
    1,
)

_SALES_ROWS = [(datetime.date(2023, month, 1), float(100 + month)) for month in range(1, 13)] + [
    (datetime.date(2024, month, 1), float(112 + month)) for month in range(1, 6)
]

_MOCK_SHAP = np.asarray(
    [
        [10.0, -5.0, 3.0, 1.0, -0.5, 0.2],
        [8.0, -4.0, 2.5, 0.8, -0.4, 0.1],
        [9.0, -4.5, 2.8, 0.9, -0.45, 0.15],
        [11.0, -5.5, 3.2, 1.1, -0.55, 0.25],
        [10.5, -5.2, 3.1, 1.05, -0.52, 0.22],
    ]
)


def _loaded_set(
    artifact: dict | None = None,
    *,
    cluster_label: str = "0",
    cluster_strategy: str = "per_cluster",
) -> SimpleNamespace:
    return SimpleNamespace(
        artifacts={cluster_label: artifact or _ARTIFACT},
        ref=SimpleNamespace(
            artifact_set_id="artifact-set-1",
            metadata={
                "cluster_strategy": cluster_strategy,
                "model_config": {"feature_history": {"lookback_months": 17}},
            },
        ),
    )


def _client_patches(
    pool,
    *,
    loaded_set: SimpleNamespace | None = None,
    sales_rows: list[tuple[datetime.date, float]] | None = None,
    future_rows: list[tuple[datetime.date, float, str]] | None = None,
    shap_values: np.ndarray | None = None,
):
    history = sales_rows if sales_rows is not None else _SALES_ROWS
    future = future_rows if future_rows is not None else []
    values = shap_values if shap_values is not None else _MOCK_SHAP
    return (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "api.routers.forecasting.shap._load_active_lgbm_artifact_set",
            return_value=loaded_set or _loaded_set(),
        ),
        patch(
            "api.routers.forecasting.shap._load_shap_sales_history",
            return_value=(history, _HISTORY_START, _HISTORY_END),
        ),
        patch(
            "api.routers.forecasting.shap._load_future_forecast_rows",
            return_value=future,
        ),
        patch(
            "api.routers.forecasting.shap._compute_shap_full",
            return_value=(values, np.full(len(values), 120.0)),
        ),
    )


async def _get(params: dict[str, object]) -> httpx.Response:
    from api.main import app

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get("/forecast/shap/lgbm_cluster/dfu", params=params)


@pytest.mark.asyncio
async def test_dfu_shap_uses_active_production_artifact() -> None:
    pool, _conn, cursor = make_pool()
    cursor.fetchall.return_value = [_DFU_ROW]

    with ExitStack() as stack:
        for context in _client_patches(pool):
            stack.enter_context(context)
        response = await _get(
            {
                "item_id": "100320",
                "loc": "1401-BULK",
                "customer_group": "grp1",
                "top_n": 6,
            }
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["item_id"] == "100320"
    assert payload["customer_group"] == "grp1"
    assert payload["loc"] == "1401-BULK"
    assert payload["model_id"] == "lgbm_cluster"
    assert payload["cluster_id"] == "0"
    assert payload["artifact_set_id"] == "artifact-set-1"
    assert payload["history_end"] == "2024-05-01"
    assert payload["top_n"] == 6
    assert len(payload["points"]) == 5


@pytest.mark.asyncio
async def test_dfu_shap_rejects_ambiguous_customer_group() -> None:
    pool, _conn, cursor = make_pool()
    first_group = (*_DFU_ROW[:-1], 2)
    second_group = (*_DFU_ROW[:7], "grp2", *_DFU_ROW[8:-1], 2)
    cursor.fetchall.return_value = [first_group, second_group]

    with patch("api.core._get_pool", return_value=pool):
        response = await _get({"item_id": "100320", "loc": "1401-BULK"})

    assert response.status_code == 422
    assert "customer_group" in response.json()["detail"]


@pytest.mark.asyncio
async def test_dfu_shap_filters_exact_customer_group() -> None:
    pool, _conn, cursor = make_pool()
    cursor.fetchall.return_value = [_DFU_ROW]

    with ExitStack() as stack:
        for context in _client_patches(pool):
            stack.enter_context(context)
        response = await _get(
            {
                "item_id": "100320",
                "loc": "1401-BULK",
                "customer_group": "grp1",
            }
        )

    assert response.status_code == 200, response.text
    _query, params = cursor.execute.call_args_list[0].args
    assert "customer_group" in str(_query)
    assert params == ("100320", "1401-BULK", "grp1", "grp1")


@pytest.mark.asyncio
async def test_dfu_shap_returns_404_when_dfu_is_missing() -> None:
    pool, _conn, cursor = make_pool()
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        response = await _get(
            {
                "item_id": "UNKNOWN",
                "loc": "NOWHERE",
                "customer_group": "grp1",
            }
        )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_dfu_shap_reports_missing_active_artifact() -> None:
    pool, _conn, cursor = make_pool()
    cursor.fetchall.return_value = [_DFU_ROW]

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "api.routers.forecasting.shap._load_active_lgbm_artifact_set",
            side_effect=FileNotFoundError("active.json"),
        ),
    ):
        response = await _get(
            {
                "item_id": "100320",
                "loc": "1401-BULK",
                "customer_group": "grp1",
            }
        )

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "active LightGBM production artifact" in detail
    assert "train-production MODEL=lgbm_cluster" in detail


@pytest.mark.asyncio
async def test_dfu_shap_reports_stale_active_artifact() -> None:
    pool, _conn, cursor = make_pool()
    cursor.fetchall.return_value = [_DFU_ROW]

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "api.routers.forecasting.shap._load_active_lgbm_artifact_set",
            side_effect=RuntimeError("lineage mismatch"),
        ),
    ):
        response = await _get(
            {
                "item_id": "100320",
                "loc": "1401-BULK",
                "customer_group": "grp1",
            }
        )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "stale or invalid" in detail
    assert "lineage mismatch" not in detail


@pytest.mark.asyncio
async def test_dfu_shap_uses_persisted_categorical_encoder_codes() -> None:
    pool, _conn, cursor = make_pool()
    cursor.fetchall.return_value = [_DFU_ROW]
    artifact = {
        "model": MagicMock(),
        "feature_cols": ["qty_lag_1", "region", "brand", "abc_vol"],
        # Deliberately not alphabetical: a live-universe rebuild would assign
        # different values and explain a different decision path.
        "categorical_encoders": {
            "region": {"SW": 0, "NE": 1},
            "brand": {"brand_z": 0, "brand_a": 1},
            "abc_vol": {"C": 0, "A": 1},
        },
    }
    captured: dict[str, pd.DataFrame] = {}

    def _capture(_model, matrix, _model_id, _feature_cols):
        captured["matrix"] = matrix.copy()
        return _MOCK_SHAP[:, :4], np.full(len(_MOCK_SHAP), 120.0)

    with ExitStack() as stack:
        for context in _client_patches(
            pool,
            loaded_set=_loaded_set(artifact),
            shap_values=_MOCK_SHAP[:, :4],
        ):
            stack.enter_context(context)
        stack.enter_context(
            patch("api.routers.forecasting.shap._compute_shap_full", side_effect=_capture)
        )
        response = await _get(
            {
                "item_id": "100320",
                "loc": "1401-BULK",
                "customer_group": "grp1",
            }
        )

    assert response.status_code == 200, response.text
    matrix = captured["matrix"]
    assert matrix[["region", "brand", "abc_vol"]].drop_duplicates().to_dict("records") == [
        {"region": 1, "brand": 1, "abc_vol": 1}
    ]


@pytest.mark.asyncio
async def test_dfu_shap_rejects_category_absent_from_persisted_encoder() -> None:
    pool, _conn, cursor = make_pool()
    cursor.fetchall.return_value = [_DFU_ROW]
    artifact = {
        "model": MagicMock(),
        "feature_cols": ["qty_lag_1", "brand"],
        "categorical_encoders": {"brand": {"brand_z": 0}},
    }

    with ExitStack() as stack:
        for context in _client_patches(pool, loaded_set=_loaded_set(artifact)):
            stack.enter_context(context)
        response = await _get(
            {
                "item_id": "100320",
                "loc": "1401-BULK",
                "customer_group": "grp1",
            }
        )

    assert response.status_code == 409
    assert "categorical encoder" in response.json()["detail"]


@pytest.mark.asyncio
async def test_dfu_shap_loads_artifact_history_window_not_display_window() -> None:
    pool, _conn, cursor = make_pool()
    cursor.fetchall.return_value = [_DFU_ROW]

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "api.routers.forecasting.shap._load_active_lgbm_artifact_set",
            return_value=_loaded_set(),
        ),
        patch(
            "api.routers.forecasting.shap._load_shap_sales_history",
            return_value=(_SALES_ROWS, _HISTORY_START, _HISTORY_END),
        ) as history_loader,
        patch(
            "api.routers.forecasting.shap._load_future_forecast_rows",
            return_value=[],
        ),
        patch(
            "api.routers.forecasting.shap._compute_shap_full",
            return_value=(_MOCK_SHAP, np.full(len(_MOCK_SHAP), 120.0)),
        ),
    ):
        response = await _get(
            {
                "item_id": "100320",
                "loc": "1401-BULK",
                "customer_group": "grp1",
                "lookback_months": 12,
            }
        )

    assert response.status_code == 200, response.text
    assert history_loader.call_args.kwargs["lookback_months"] == 17


@pytest.mark.asyncio
async def test_dfu_shap_rejects_dimension_row_without_sales_history() -> None:
    pool, _conn, cursor = make_pool()
    cursor.fetchall.return_value = [_DFU_ROW]

    with ExitStack() as stack:
        for context in _client_patches(pool, sales_rows=[]):
            stack.enter_context(context)
        response = await _get(
            {
                "item_id": "100320",
                "loc": "1401-BULK",
                "customer_group": "grp1",
            }
        )

    assert response.status_code == 422
    assert "no canonical sales history" in response.json()["detail"]


@pytest.mark.asyncio
async def test_dfu_shap_top_n_validation_runs_before_database_access() -> None:
    pool, _conn, _cursor = make_pool()
    with patch("api.core._get_pool", return_value=pool):
        response = await _get(
            {
                "item_id": "100320",
                "loc": "1401-BULK",
                "customer_group": "grp1",
                "top_n": 999,
            }
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_dfu_shap_includes_future_plan_rows() -> None:
    pool, _conn, cursor = make_pool()
    cursor.fetchall.return_value = [_DFU_ROW]
    future_rows = [
        (datetime.date(2024, 6, 1), 150.0, "lgbm_cluster"),
        (datetime.date(2024, 7, 1), 155.0, "lgbm_cluster"),
    ]
    shap_values = np.vstack([_MOCK_SHAP, _MOCK_SHAP[:2]])

    with ExitStack() as stack:
        for context in _client_patches(
            pool,
            future_rows=future_rows,
            shap_values=shap_values,
        ):
            stack.enter_context(context)
        response = await _get(
            {
                "item_id": "100320",
                "loc": "1401-BULK",
                "customer_group": "grp1",
            }
        )

    assert response.status_code == 200, response.text
    future_points = [point for point in response.json()["points"] if point["is_future"]]
    assert len(future_points) == 2
    assert response.json()["future_lag_model_id"] == "lgbm_cluster"


@pytest.mark.asyncio
async def test_dfu_shap_does_not_use_aggregate_future_for_one_of_many_groups() -> None:
    pool, _conn, cursor = make_pool()
    cursor.fetchall.return_value = [(*_DFU_ROW[:-1], 2)]

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "api.routers.forecasting.shap._load_active_lgbm_artifact_set",
            return_value=_loaded_set(),
        ),
        patch(
            "api.routers.forecasting.shap._load_shap_sales_history",
            return_value=(_SALES_ROWS, _HISTORY_START, _HISTORY_END),
        ),
        patch(
            "api.routers.forecasting.shap._load_future_forecast_rows",
        ) as future_loader,
        patch(
            "api.routers.forecasting.shap._compute_shap_full",
            return_value=(_MOCK_SHAP, np.full(len(_MOCK_SHAP), 120.0)),
        ),
    ):
        response = await _get(
            {
                "item_id": "100320",
                "loc": "1401-BULK",
                "customer_group": "grp1",
            }
        )

    assert response.status_code == 200, response.text
    future_loader.assert_not_called()
    assert response.json()["future_lag_model_id"] is None
    assert not any(point["is_future"] for point in response.json()["points"])
