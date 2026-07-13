"""Production forecast generation model-roster boundary tests."""

import inspect

import pytest

from scripts.forecasting import generate_production_forecasts as production


def test_production_model_guard_accepts_canonical_forecast_model(monkeypatch):
    monkeypatch.setattr(
        production,
        "get_forecastable_model_ids",
        lambda: ["lgbm_cluster", "mstl"],
    )

    production._validate_forecastable_model_ids({"mstl"}, source="--model-id")


def test_production_model_guard_rejects_retired_forecast_model(monkeypatch):
    monkeypatch.setattr(
        production,
        "get_forecastable_model_ids",
        lambda: ["lgbm_cluster", "mstl"],
    )

    with pytest.raises(ValueError, match="catboost_cluster"):
        production._validate_forecastable_model_ids(
            {"catboost_cluster"},
            source="--model-id",
        )


def test_production_generator_has_no_retired_enriched_model_hook():
    """Canonical generation must not retain unreachable variant loaders."""
    source = inspect.getsource(production)

    assert not hasattr(production, "_load_customer_features_for_inference")
    assert "cust_enriched" not in source
    assert "hierarchical\" in m" not in source
