"""Configuration contract shared by evaluated and production forecast adapters."""

import pytest

from common.core.utils import get_algorithm_params, load_forecast_pipeline_config
from scripts.ml.run_backtest_chronos2_enriched import _extract_params


def test_neural_models_define_every_reproducibility_parameter() -> None:
    config = load_forecast_pipeline_config()
    required = {
        "h",
        "input_size",
        "max_steps",
        "batch_size",
        "learning_rate",
        "scaler_type",
        "early_stop_patience_steps",
        "min_history",
        "random_seed",
        "start_padding_enabled",
        "val_size",
        "deterministic",
    }

    for model_id in ("nbeats", "nhits"):
        params = config["algorithms"][model_id]["params"]
        assert set(params) == required
        assert params["h"] == config["backtest"]["forecast_horizon"]
        assert params["random_seed"] == config["tuning"]["random_seed"]


def test_chronos_model_revision_is_pinned_and_forwarded_to_backtest() -> None:
    params = load_forecast_pipeline_config()["algorithms"]["chronos2_enriched"][
        "params"
    ]

    assert params["model_name"] == "amazon/chronos-2"
    assert len(params["model_revision"]) == 40
    assert _extract_params(params)["model_revision"] == params["model_revision"]
    assert params["min_history"] > 0
    assert _extract_params(params)["min_history"] == params["min_history"]


def test_every_canonical_model_exposes_explicit_parameter_mapping() -> None:
    for model_id in (
        "lgbm_cluster",
        "nhits",
        "nbeats",
        "mstl",
        "chronos2_enriched",
    ):
        assert get_algorithm_params(model_id)


def test_algorithm_params_reject_missing_model_or_legacy_flat_config(monkeypatch) -> None:
    monkeypatch.setattr(
        "common.core.utils.load_forecast_pipeline_config",
        lambda: {
            "algorithms": {
                "legacy_flat": {"learning_rate": 0.1},
            }
        },
    )

    with pytest.raises(ValueError, match="missing explicit params"):
        get_algorithm_params("legacy_flat")
    with pytest.raises(ValueError, match="not configured"):
        get_algorithm_params("missing")
