"""Focused guards for the canonical neural and foundation adapters."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from common.core.constants import FORECAST_QTY_COL
from common.core.paths import PROJECT_ROOT


def _neural_params(**overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "h": 6,
        "input_size": 24,
        "max_steps": 10,
        "batch_size": 4,
        "learning_rate": 0.001,
        "scaler_type": "standard",
        "early_stop_patience_steps": -1,
        "min_history": 2,
        "random_seed": 73,
        "start_padding_enabled": False,
        "val_size": 2,
        "deterministic": True,
    }
    params.update(overrides)
    return params


def _neural_sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_ck": ["sku-1"] * 3 + ["short"],
            "startdate": [
                *pd.date_range("2026-04-01", periods=3, freq="MS"),
                pd.Timestamp("2026-06-01"),
            ],
            "qty": [1.0, 2.0, 3.0, 9.0],
        }
    )


def _install_fake_neuralforecast_models(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    models = ModuleType("neuralforecast.models")
    models.NBEATS = Mock(name="NBEATS")
    models.NHITS = Mock(name="NHITS")
    monkeypatch.setitem(sys.modules, "neuralforecast.models", models)
    return models


def test_neural_adapter_supports_only_nhits_and_nbeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from common.ml import neural_forecast

    models = _install_fake_neuralforecast_models(monkeypatch)
    params = _neural_params()

    assert neural_forecast.SUPPORTED_NEURAL_MODELS == frozenset({"nhits", "nbeats"})
    neural_forecast._build_model("nhits", params)
    neural_forecast._build_model("nbeats", params)
    assert models.NHITS.call_args.kwargs["random_seed"] == 73
    assert models.NBEATS.call_args.kwargs["random_seed"] == 73
    assert models.NHITS.call_args.kwargs["start_padding_enabled"] is False
    assert models.NHITS.call_args.kwargs["deterministic"] is True
    assert models.NHITS.call_args.kwargs["enable_progress_bar"] is False
    assert models.NBEATS.call_args.kwargs["enable_progress_bar"] is False

    for retired_model in (
        "tft",
        "deepar",
        "tide",
        "tcn",
        "patchtst",
        "itransformer",
    ):
        with pytest.raises(ValueError, match="Unsupported neural model"):
            neural_forecast._build_model(retired_model, params)


def test_chronos_adapter_has_one_direct_entry_point() -> None:
    from common.ml import chronos2_enriched

    assert callable(chronos2_enriched.run_chronos2_enriched)
    assert not hasattr(chronos2_enriched, "run_foundation_models")

    result = chronos2_enriched.run_chronos2_enriched(
        pd.DataFrame(),
        [pd.Timestamp("2026-08-01")],
        {"device": "cpu", "batch_size": 2, "prediction_length": 1},
    )
    assert result.empty
    assert list(result.columns) == [
        "sku_ck",
        "startdate",
        FORECAST_QTY_COL,
        "algorithm_id",
    ]


def test_gpu_required_fails_instead_of_silently_using_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from common.ml import chronos2_enriched, neural_forecast

    torch = ModuleType("torch")
    torch.backends = Mock()
    torch.backends.mps.is_available.return_value = False
    torch.cuda = Mock()
    torch.cuda.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("DEMAND_CHRONOS_GPU", "on")
    monkeypatch.setenv("DEMAND_NEURAL_GPU", "on")

    with pytest.raises(RuntimeError, match="GPU acceleration was required"):
        chronos2_enriched._resolve_device("auto")
    with pytest.raises(RuntimeError, match="GPU acceleration was required"):
        neural_forecast._detect_device()


def test_gpu_required_selects_apple_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from common.ml import chronos2_enriched, neural_forecast

    torch = ModuleType("torch")
    torch.backends = Mock()
    torch.backends.mps.is_available.return_value = True
    torch.cuda = Mock()
    torch.cuda.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("DEMAND_CHRONOS_GPU", "on")
    monkeypatch.setenv("DEMAND_NEURAL_GPU", "on")

    assert chronos2_enriched._resolve_device("auto") == "mps"
    assert neural_forecast._detect_device() == "mps"


def test_forecast_model_families_can_use_independent_accelerator_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from common.ml import chronos2_enriched, neural_forecast

    torch = ModuleType("torch")
    torch.backends = Mock()
    torch.backends.mps.is_available.return_value = True
    torch.cuda = Mock()
    torch.cuda.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("DEMAND_CHRONOS_GPU", "on")
    monkeypatch.setenv("DEMAND_NEURAL_GPU", "off")

    assert chronos2_enriched._resolve_device("auto") == "mps"
    assert neural_forecast._detect_device() == "cpu"


def test_chronos_adapter_rejects_non_finite_predictions_instead_of_zero_fill() -> None:
    from common.ml.chronos2_enriched import _validate_forecast_values

    with pytest.raises(RuntimeError, match="non-finite"):
        _validate_forecast_values(
            np.array([10.0, float("nan")]),
            sku_ck="sku-1",
        )


def test_neural_fit_preserves_configured_h_and_predict_extends_same_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from common.ml import neural_forecast

    captured: dict[str, object] = {}

    class FakeNeuralForecast:
        def __init__(self, *, models, freq):
            captured["models"] = models
            captured["freq"] = freq

        def fit(self, *, df, val_size):
            captured["fit_rows"] = len(df)
            captured["val_size"] = val_size

        def predict(self, *, df, h):
            captured["predict_rows"] = len(df)
            captured["predict_h"] = h
            return pd.DataFrame(
                {
                    "unique_id": ["sku-1"] * 8,
                    "ds": pd.date_range("2026-07-01", periods=8, freq="MS"),
                    "NHITS": range(8),
                }
            )

    package = ModuleType("neuralforecast")
    package.NeuralForecast = FakeNeuralForecast
    monkeypatch.setitem(sys.modules, "neuralforecast", package)
    monkeypatch.setattr(neural_forecast, "_check_neuralforecast", lambda: True)
    monkeypatch.setattr(neural_forecast, "_detect_device", lambda: "cpu")

    def fake_build(model_id, params, accelerator):
        captured["model_id"] = model_id
        captured["params"] = params
        captured["accelerator"] = accelerator
        return object()

    monkeypatch.setattr(neural_forecast, "_build_model", fake_build)
    fitted = neural_forecast.fit_neural_model(
        _neural_sales(),
        model_id="nhits",
        params=_neural_params(),
        accelerator="cpu",
    )
    monkeypatch.setattr(
        neural_forecast,
        "_build_model",
        lambda *_args, **_kwargs: pytest.fail("prediction must not rebuild the model"),
    )
    result = neural_forecast.predict_neural_model(
        fitted,
        _neural_sales(),
        list(pd.date_range("2026-07-01", periods=8, freq="MS")),
    )

    assert captured["params"]["h"] == 6
    assert captured["fit_rows"] == 3
    assert captured["predict_rows"] == 3
    assert captured["predict_h"] == 8
    assert captured["val_size"] == 2
    assert fitted.model_id == "nhits"
    assert fitted.fitted_horizon == 6
    assert fitted.min_history == 2
    assert fitted.training_dfu_count == 1
    assert fitted.training_row_count == 3
    assert len(fitted.training_cohort_checksum) == 64
    assert len(fitted.training_data_checksum) == 64
    assert fitted.training_contract_version == neural_forecast.NEURAL_TRAINING_CONTRACT_VERSION
    assert set(fitted.runtime_contract) == {"neuralforecast", "numpy", "pandas", "python"}
    assert len(result) == 8
    assert result.attrs["fitted_horizon"] == 6


def test_neural_fit_densifies_internal_calendar_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from common.ml import neural_forecast

    captured: dict[str, pd.DataFrame] = {}

    class FakeNeuralForecast:
        def __init__(self, *, models, freq):
            pass

        def fit(self, *, df, val_size):
            captured["fit"] = df.copy()

    package = ModuleType("neuralforecast")
    package.NeuralForecast = FakeNeuralForecast
    monkeypatch.setitem(sys.modules, "neuralforecast", package)
    monkeypatch.setattr(neural_forecast, "_check_neuralforecast", lambda: True)
    monkeypatch.setattr(neural_forecast, "_build_model", lambda *_args, **_kwargs: object())

    sparse = pd.DataFrame(
        {
            "sku_ck": ["sku-1", "sku-1"],
            "startdate": pd.to_datetime(["2026-01-01", "2026-03-01"]),
            "qty": [5.0, 7.0],
        }
    )
    sparse.attrs["history_end"] = pd.Timestamp("2026-03-01")

    neural_forecast.fit_neural_model(
        sparse,
        model_id="nhits",
        params=_neural_params(min_history=3),
        accelerator="cpu",
    )

    fitted = captured["fit"].set_index("ds")
    assert list(fitted.index) == list(pd.date_range("2026-01-01", periods=3, freq="MS"))
    assert fitted.loc[pd.Timestamp("2026-02-01"), "y"] == 0.0


def test_neural_training_lineage_covers_exact_normalized_cohort_and_values() -> None:
    from common.ml.neural_forecast import derive_neural_training_lineage

    sales = _neural_sales()
    runtime = {
        "python": "3.12.0",
        "numpy": "2.2.0",
        "pandas": "2.2.0",
        "neuralforecast": "3.0.0",
    }
    first = derive_neural_training_lineage(
        sales,
        min_history=2,
        runtime_contract=runtime,
    )
    reordered = derive_neural_training_lineage(
        sales.sample(frac=1, random_state=9),
        min_history=2,
        runtime_contract=dict(reversed(list(runtime.items()))),
    )
    changed_value = derive_neural_training_lineage(
        sales.assign(qty=lambda frame: frame["qty"] + 1),
        min_history=2,
        runtime_contract=runtime,
    )
    changed_cohort = derive_neural_training_lineage(
        pd.concat(
            [sales, sales.query("sku_ck == 'sku-1'").assign(sku_ck="sku-extra")],
            ignore_index=True,
        ),
        min_history=2,
        runtime_contract=runtime,
    )

    assert reordered == first
    assert first.training_dfu_count == 1
    assert first.training_row_count == 3
    assert len(first.training_cohort_checksum) == 64
    assert len(first.training_data_checksum) == 64
    assert changed_value.training_cohort_checksum == first.training_cohort_checksum
    assert changed_value.training_data_checksum != first.training_data_checksum
    assert changed_cohort.training_cohort_checksum != first.training_cohort_checksum
    assert changed_cohort.training_data_checksum != first.training_data_checksum


def test_neural_adapter_does_not_turn_non_finite_predictions_into_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from common.ml import neural_forecast

    class FakeNeuralForecast:
        def __init__(self, *, models, freq):
            pass

        def fit(self, *, df, val_size):
            pass

        def predict(self, *, df, h):
            return pd.DataFrame(
                {
                    "unique_id": ["sku-1"],
                    "ds": [pd.Timestamp("2026-07-01")],
                    "NHITS": [float("nan")],
                }
            )

    package = ModuleType("neuralforecast")
    package.NeuralForecast = FakeNeuralForecast
    monkeypatch.setitem(sys.modules, "neuralforecast", package)
    monkeypatch.setattr(neural_forecast, "_check_neuralforecast", lambda: True)
    monkeypatch.setattr(neural_forecast, "_detect_device", lambda: "cpu")
    monkeypatch.setattr(neural_forecast, "_build_model", lambda *_args, **_kwargs: object())

    with pytest.raises(RuntimeError, match="non-finite"):
        neural_forecast.run_neural_models(
            _neural_sales().query("sku_ck == 'sku-1'"),
            [pd.Timestamp("2026-07-01")],
            {"nhits": _neural_params(h=1)},
        )


def test_neural_missing_dependency_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    from common.ml import neural_forecast

    monkeypatch.setattr(neural_forecast, "_check_neuralforecast", lambda: False)

    with pytest.raises(RuntimeError, match="neuralforecast is required"):
        neural_forecast.fit_neural_model(
            _neural_sales(),
            model_id="nhits",
            params=_neural_params(),
        )


def test_neural_fit_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    from common.ml import neural_forecast

    class FakeNeuralForecast:
        def __init__(self, *, models, freq):
            pass

        def fit(self, *, df, val_size):
            raise RuntimeError("fit exploded")

    package = ModuleType("neuralforecast")
    package.NeuralForecast = FakeNeuralForecast
    monkeypatch.setitem(sys.modules, "neuralforecast", package)
    monkeypatch.setattr(neural_forecast, "_check_neuralforecast", lambda: True)
    monkeypatch.setattr(neural_forecast, "_build_model", lambda *_args, **_kwargs: object())

    with pytest.raises(RuntimeError, match="fit exploded"):
        neural_forecast.fit_neural_model(
            _neural_sales(),
            model_id="nhits",
            params=_neural_params(),
            accelerator="cpu",
        )


@pytest.mark.parametrize(
    ("forecast", "error"),
    [
        (
            pd.DataFrame({"ds": [pd.Timestamp("2026-07-01")], "NHITS": [1.0]}),
            "missing required columns",
        ),
        (
            pd.DataFrame(
                {
                    "unique_id": ["sku-1"],
                    "ds": [pd.Timestamp("2026-07-01")],
                }
            ),
            "forecast column",
        ),
        (
            pd.DataFrame(
                {
                    "unique_id": ["sku-1"],
                    "ds": [pd.Timestamp("2026-07-01")],
                    "NBEATS": [1.0],
                }
            ),
            "model identity",
        ),
        (
            pd.DataFrame(
                {
                    "unique_id": ["wrong"],
                    "ds": [pd.Timestamp("2026-07-01")],
                    "NHITS": [1.0],
                }
            ),
            "series IDs",
        ),
        (
            pd.DataFrame(
                {
                    "unique_id": ["sku-1", "sku-1"],
                    "ds": [pd.Timestamp("2026-07-01")] * 2,
                    "NHITS": [1.0, 2.0],
                }
            ),
            "duplicate",
        ),
        (
            pd.DataFrame(
                {
                    "unique_id": ["sku-1"],
                    "ds": [pd.Timestamp("2026-08-01")],
                    "NHITS": [1.0],
                }
            ),
            "forecast months",
        ),
    ],
)
def test_neural_prediction_rejects_malformed_or_incomplete_output(
    forecast: pd.DataFrame,
    error: str,
) -> None:
    from common.ml import neural_forecast

    class FakeNeuralForecast:
        def predict(self, *, df, h):
            return forecast

    fitted = neural_forecast.FittedNeuralModel(
        neural_forecast=FakeNeuralForecast(),
        model_id="nhits",
        fitted_horizon=1,
        min_history=2,
        training_dfu_count=1,
    )

    with pytest.raises(RuntimeError, match=error):
        neural_forecast.predict_neural_model(
            fitted,
            _neural_sales(),
            [pd.Timestamp("2026-07-01")],
        )


def test_neural_predict_failure_propagates() -> None:
    from common.ml import neural_forecast

    class FakeNeuralForecast:
        def predict(self, *, df, h):
            raise ValueError("predict exploded")

    fitted = neural_forecast.FittedNeuralModel(
        neural_forecast=FakeNeuralForecast(),
        model_id="nhits",
        fitted_horizon=1,
        min_history=2,
        training_dfu_count=1,
    )

    with pytest.raises(ValueError, match="predict exploded"):
        neural_forecast.predict_neural_model(
            fitted,
            _neural_sales(),
            [pd.Timestamp("2026-07-01")],
        )


def test_neural_prediction_rejects_fitted_horizon_metadata_mismatch() -> None:
    from common.ml import neural_forecast

    class FakeNeuralForecast:
        h = 2

    fitted = neural_forecast.FittedNeuralModel(
        neural_forecast=FakeNeuralForecast(),
        model_id="nhits",
        fitted_horizon=1,
        min_history=2,
        training_dfu_count=1,
    )

    with pytest.raises(RuntimeError, match="fitted horizon metadata"):
        neural_forecast.predict_neural_model(
            fitted,
            _neural_sales(),
            [pd.Timestamp("2026-07-01")],
        )


def test_neural_prediction_requires_complete_dfu_month_coverage() -> None:
    from common.ml import neural_forecast

    live_sales = pd.concat(
        [
            _neural_sales().query("sku_ck == 'sku-1'"),
            _neural_sales()
            .query("sku_ck == 'sku-1'")
            .assign(sku_ck="sku-2", qty=lambda frame: frame["qty"] + 10),
        ],
        ignore_index=True,
    )

    class FakeNeuralForecast:
        def predict(self, *, df, h):
            return pd.DataFrame(
                {
                    "unique_id": ["sku-1", "sku-2", "sku-1"],
                    "ds": pd.to_datetime(["2026-07-01", "2026-07-01", "2026-08-01"]),
                    "NHITS": [1.0, 2.0, 3.0],
                }
            )

    fitted = neural_forecast.FittedNeuralModel(
        neural_forecast=FakeNeuralForecast(),
        model_id="nhits",
        fitted_horizon=2,
        min_history=2,
        training_dfu_count=2,
    )

    with pytest.raises(RuntimeError, match="incomplete forecast coverage"):
        neural_forecast.predict_neural_model(
            fitted,
            live_sales,
            list(pd.date_range("2026-07-01", periods=2, freq="MS")),
        )


def test_neural_multi_model_run_is_all_or_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    from common.ml import neural_forecast

    def fake_fit(sales_df, model_id, params, accelerator=None):
        return neural_forecast.FittedNeuralModel(
            neural_forecast=object(),
            model_id=model_id,
            fitted_horizon=int(params["h"]),
            min_history=int(params["min_history"]),
            training_dfu_count=1,
        )

    def fake_predict(fitted, sales_df, predict_months):
        if fitted.model_id == "nbeats":
            raise RuntimeError("second model failed")
        return pd.DataFrame(
            {
                "sku_ck": ["sku-1"],
                "startdate": [pd.Timestamp("2026-07-01")],
                FORECAST_QTY_COL: [1.0],
                "algorithm_id": ["nhits"],
            }
        )

    monkeypatch.setattr(neural_forecast, "fit_neural_model", fake_fit)
    monkeypatch.setattr(neural_forecast, "predict_neural_model", fake_predict)

    with pytest.raises(RuntimeError, match="second model failed"):
        neural_forecast.run_neural_models(
            _neural_sales(),
            [pd.Timestamp("2026-07-01")],
            {"nhits": _neural_params(h=1), "nbeats": _neural_params(h=1)},
        )


def test_complete_monthly_history_respects_declared_introduction_month() -> None:
    from common.ml.production_non_tree import complete_monthly_history

    calendar = pd.date_range("2026-01-01", periods=6, freq="MS")
    sales = pd.DataFrame(
        {
            "sku_ck": ["sku-1"] * 6,
            "startdate": calendar,
            "qty": [0.0, 0.0, 10.0, 0.0, 20.0, 0.0],
            "first_sale_month": pd.Timestamp("2026-03-01"),
        }
    )
    sales.attrs["history_end"] = calendar[-1]

    complete = complete_monthly_history(sales)

    assert complete["startdate"].min() == pd.Timestamp("2026-03-01")
    assert complete["qty"].tolist() == [10.0, 0.0, 20.0, 0.0]


def test_expert_panel_namespace_is_removed() -> None:
    package_path = PROJECT_ROOT / "common" / "ml" / "expert_panel"

    assert not any(package_path.glob("*.py"))
