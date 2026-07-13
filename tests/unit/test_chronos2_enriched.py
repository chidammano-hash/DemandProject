"""Regression tests for the canonical Chronos 2 Enriched adapter."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pandas as pd
import pytest
import torch

from common.core.constants import FORECAST_QTY_COL
from common.ml import chronos2_enriched
from scripts.ml.run_backtest_chronos2_enriched import spec as backtest_spec


class _FakeChronosPipeline:
    instance: _FakeChronosPipeline
    load_calls: list[tuple[tuple[Any, ...], dict[str, Any]]]

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.failure_on_call: int | None = None
        self.non_finite = False

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> _FakeChronosPipeline:
        cls.load_calls.append((args, kwargs))
        return cls.instance

    def predict(
        self,
        inputs: list[dict[str, Any]],
        *,
        prediction_length: int,
        batch_size: int,
    ) -> list[torch.Tensor]:
        self.calls.append(
            {
                "inputs": inputs,
                "prediction_length": prediction_length,
                "batch_size": batch_size,
            }
        )
        if self.failure_on_call == len(self.calls):
            raise ValueError("model inference failed")
        value = float("nan") if self.non_finite else 7.0
        return [
            torch.full((1, 3, prediction_length), value, dtype=torch.float32)
            for _entry in inputs
        ]


@pytest.fixture
def fake_chronos(monkeypatch: pytest.MonkeyPatch) -> _FakeChronosPipeline:
    pipeline = _FakeChronosPipeline()
    _FakeChronosPipeline.instance = pipeline
    _FakeChronosPipeline.load_calls = []
    package = ModuleType("chronos")
    package.Chronos2Pipeline = _FakeChronosPipeline
    monkeypatch.setitem(sys.modules, "chronos", package)
    chronos2_enriched._chronos_pipeline_cache.clear()
    return pipeline


def _params(
    *,
    prediction_length: int = 1,
    batch_size: int = 2,
    min_history: int = 3,
) -> dict[str, Any]:
    return {
        "device": "cpu",
        "batch_size": batch_size,
        "prediction_length": prediction_length,
        "min_history": min_history,
        "model_name": "amazon/chronos-2",
        "model_revision": "pinned-test-revision",
    }


def _sales(*, two_dfus: bool = False) -> pd.DataFrame:
    rows = [
        {"sku_ck": "sku-1", "startdate": "2026-01-01", "qty": 10.0},
        {"sku_ck": "sku-1", "startdate": "2026-03-01", "qty": 30.0},
    ]
    if two_dfus:
        rows.extend(
            [
                {"sku_ck": "sku-2", "startdate": "2026-01-01", "qty": 11.0},
                {"sku_ck": "sku-2", "startdate": "2026-02-01", "qty": 12.0},
                {"sku_ck": "sku-2", "startdate": "2026-03-01", "qty": 13.0},
            ]
        )
    return pd.DataFrame(rows)


def test_backtest_does_not_load_unused_customer_feature_table() -> None:
    """The canonical adapter's 30 covariates do not include customer features."""
    assert backtest_spec.include_customer_features is False


def test_sparse_history_is_monthly_complete_and_covariates_align_by_date(
    fake_chronos: _FakeChronosPipeline,
) -> None:
    feature_grid = pd.DataFrame(
        {
            "sku_ck": ["sku-1", "sku-1", "sku-1"],
            # Deliberately unsorted so positional/tail alignment is detectable.
            "startdate": ["2026-03-01", "2026-01-01", "2026-02-01"],
            "qty_lag_1": [303.0, 101.0, 202.0],
        }
    )

    result = chronos2_enriched.run_chronos2_enriched(
        _sales(),
        [pd.Timestamp("2026-04-01")],
        _params(),
        feature_grid=feature_grid,
    )

    model_input = fake_chronos.calls[0]["inputs"][0]
    assert model_input["target"].tolist() == [10.0, 0.0, 30.0]
    assert model_input["past_covariates"]["qty_lag_1"].tolist() == [101.0, 202.0, 303.0]
    assert result[["sku_ck", "startdate", FORECAST_QTY_COL]].to_dict("records") == [
        {
            "sku_ck": "sku-1",
            "startdate": pd.Timestamp("2026-04-01"),
            FORECAST_QTY_COL: 7.0,
        }
    ]


def test_configured_min_history_controls_chronos_population(
    fake_chronos: _FakeChronosPipeline,
) -> None:
    result = chronos2_enriched.run_chronos2_enriched(
        _sales(),
        [pd.Timestamp("2026-04-01")],
        _params(min_history=4),
    )

    assert result.empty
    assert not fake_chronos.calls


def test_chronos_min_history_is_required(
    fake_chronos: _FakeChronosPipeline,
) -> None:
    params = _params()
    del params["min_history"]

    with pytest.raises((KeyError, ValueError), match="min_history"):
        chronos2_enriched.run_chronos2_enriched(
            _sales(),
            [pd.Timestamp("2026-04-01")],
            params,
        )

    assert not fake_chronos.calls


def test_feature_grid_must_cover_every_normalized_history_month(
    fake_chronos: _FakeChronosPipeline,
) -> None:
    incomplete_grid = pd.DataFrame(
        {
            "sku_ck": ["sku-1", "sku-1"],
            "startdate": ["2026-01-01", "2026-03-01"],
            "qty_lag_1": [1.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="missing 1 required DFU-month"):
        chronos2_enriched.run_chronos2_enriched(
            _sales(),
            [pd.Timestamp("2026-04-01")],
            _params(),
            feature_grid=incomplete_grid,
        )

    assert not fake_chronos.calls


def test_feature_grid_rejects_non_finite_covariates(
    fake_chronos: _FakeChronosPipeline,
) -> None:
    invalid_grid = pd.DataFrame(
        {
            "sku_ck": ["sku-1", "sku-1", "sku-1"],
            "startdate": pd.date_range("2026-01-01", periods=3, freq="MS"),
            "qty_lag_1": [0.0, float("inf"), 10.0],
        }
    )

    with pytest.raises(ValueError, match="non-finite covariate"):
        chronos2_enriched.run_chronos2_enriched(
            _sales(),
            [pd.Timestamp("2026-04-01")],
            _params(),
            feature_grid=invalid_grid,
        )

    assert not fake_chronos.calls


def test_publish_horizon_extends_the_evaluated_direct_horizon_without_reloading(
    fake_chronos: _FakeChronosPipeline,
) -> None:
    result = chronos2_enriched.run_chronos2_enriched(
        _sales(),
        list(pd.date_range("2026-04-01", periods=8, freq="MS")),
        _params(prediction_length=6),
    )

    assert fake_chronos.calls[0]["prediction_length"] == 8
    assert len(result) == 8
    assert result.attrs["direct_horizon"] == 6
    assert result.attrs["prediction_horizon"] == 8


def test_model_load_is_pinned_to_the_configured_revision(
    fake_chronos: _FakeChronosPipeline,
) -> None:
    chronos2_enriched.run_chronos2_enriched(
        _sales(),
        [pd.Timestamp("2026-04-01")],
        _params(),
    )

    args, kwargs = _FakeChronosPipeline.load_calls[0]
    assert args[0] == "amazon/chronos-2"
    assert kwargs["revision"] == "pinned-test-revision"


def test_model_dependency_is_required_for_active_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chronos2_enriched, "_check_chronos2", lambda: False)

    with pytest.raises(RuntimeError, match="chronos-forecasting"):
        chronos2_enriched.run_chronos2_enriched(
            _sales(),
            [pd.Timestamp("2026-04-01")],
            _params(),
        )


def test_any_failed_chunk_aborts_the_whole_forecast(
    fake_chronos: _FakeChronosPipeline,
) -> None:
    fake_chronos.failure_on_call = 2

    with pytest.raises(RuntimeError, match="chunk 2/2 failed"):
        chronos2_enriched.run_chronos2_enriched(
            _sales(two_dfus=True),
            [pd.Timestamp("2026-04-01")],
            _params(batch_size=1),
        )


def test_non_finite_model_output_aborts_the_whole_forecast(
    fake_chronos: _FakeChronosPipeline,
) -> None:
    fake_chronos.non_finite = True

    with pytest.raises(RuntimeError, match="non-finite"):
        chronos2_enriched.run_chronos2_enriched(
            _sales(),
            [pd.Timestamp("2026-04-01")],
            _params(),
        )
