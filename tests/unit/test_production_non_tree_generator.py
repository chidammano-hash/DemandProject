"""Generator wiring tests for canonical production non-tree adapters."""

from datetime import date
from unittest.mock import MagicMock

import pandas as pd

from common.ml.neural_forecast import FittedNeuralModel, NeuralCohortIdentity
from scripts.forecasting import generate_production_forecasts as generator


def test_tree_artifact_loader_never_loads_non_tree_models(monkeypatch):
    calls: list[str] = []

    def fake_load(model_id, config):
        calls.append(model_id)
        return {"cluster": {"model": object(), "feature_cols": []}}

    monkeypatch.setattr(generator, "load_active_models", fake_load)

    loaded = generator._load_tree_models(
        {"lgbm_cluster", "mstl", "nhits", "nbeats", "chronos2_enriched"},
        {"mstl", "nhits", "nbeats", "chronos2_enriched"},
        {},
    )

    assert calls == ["lgbm_cluster"]
    assert set(loaded) == {"lgbm_cluster"}


def test_generator_passes_exact_yaml_params_to_canonical_adapter(monkeypatch):
    captured: dict[str, object] = {}

    def fake_adapter(**kwargs):
        captured.update(kwargs)
        return [{"model_id": kwargs["model_id"]}]

    monkeypatch.setattr(generator, "run_canonical_non_tree_forecast", fake_adapter)
    config = {
        "algorithms": {
            "mstl": {
                "params": {"season_length": 12, "min_history": 25},
            }
        }
    }
    months = [pd.Timestamp("2026-07-01")]
    sales = pd.DataFrame()
    attrs = pd.DataFrame()
    items = pd.DataFrame()
    targets = pd.DataFrame()

    result = generator._generate_canonical_non_tree_rows(
        config=config,
        model_id="mstl",
        sales_df=sales,
        dfu_attrs=attrs,
        item_attrs=items,
        target_dfus=targets,
        predict_months=months,
        forecast_month_generated=date(2026, 7, 1),
        run_id="run-1",
    )

    assert result == [{"model_id": "mstl"}]
    assert captured["params"] == {"season_length": 12, "min_history": 25}
    assert captured["target_dfus"] is targets
    assert captured["predict_months"] == months
    assert captured["sigma_lookup"] == {}
    assert captured["ci_cfg"] is None


def test_generator_passes_loaded_neural_artifact_to_adapter(monkeypatch):
    captured: dict[str, object] = {}

    def fake_adapter(**kwargs):
        captured.update(kwargs)
        return [{"model_id": kwargs["model_id"]}]

    fitted = FittedNeuralModel(
        neural_forecast=MagicMock(),
        model_id="nhits",
        fitted_horizon=6,
        min_history=12,
        training_dfu_count=100,
    )
    monkeypatch.setattr(generator, "run_canonical_non_tree_forecast", fake_adapter)

    generator._generate_canonical_non_tree_rows(
        config={"algorithms": {"nhits": {"params": {"h": 6, "min_history": 12}}}},
        model_id="nhits",
        sales_df=pd.DataFrame(),
        dfu_attrs=pd.DataFrame(),
        item_attrs=pd.DataFrame(),
        target_dfus=pd.DataFrame(),
        predict_months=[pd.Timestamp("2026-07-01")],
        forecast_month_generated=date(2026, 7, 1),
        run_id="run-neural",
        fitted_neural_model=fitted,
    )

    assert captured["fitted_neural_model"] is fitted


def test_neural_loader_loads_each_needed_model_once_with_exact_lineage(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_load(**kwargs):
        calls.append(kwargs)
        fitted = FittedNeuralModel(
            neural_forecast=MagicMock(),
            model_id=str(kwargs["model_id"]),
            fitted_horizon=6,
            min_history=12,
            training_dfu_count=100,
        )
        return MagicMock(fitted_model=fitted)

    monkeypatch.setattr(generator, "load_active_neural_artifact", fake_load)
    config = {
        "model_registry": {"base_path": "data/models"},
        "algorithms": {
            "nhits": {"params": {"h": 6, "min_history": 12}},
            "nbeats": {"params": {"h": 6, "min_history": 12}},
        },
    }

    loaded = generator._load_neural_models(
        {"lgbm_cluster", "nhits", "nbeats"},
        config,
        source_sales_batch_id=91,
        data_checksum="a" * 64,
        history_end=date(2026, 6, 1),
        expected_cohorts={
            "nbeats": NeuralCohortIdentity(checksum="b" * 64, dfu_count=100),
            "nhits": NeuralCohortIdentity(checksum="b" * 64, dfu_count=100),
        },
    )

    assert set(loaded) == {"nhits", "nbeats"}
    assert [call["model_id"] for call in calls] == ["nbeats", "nhits"]
    assert all(call["source_sales_batch_id"] == 91 for call in calls)
    assert all(call["data_checksum"] == "a" * 64 for call in calls)
    assert all(call["history_end"] == date(2026, 6, 1) for call in calls)
    assert all(call["expected_training_cohort_checksum"] == "b" * 64 for call in calls)
