"""Guards for the intentionally small forecasting product roster."""

from common.core.utils import (
    get_competing_model_ids,
    get_forecastable_model_ids,
    load_forecast_pipeline_config,
)


EXPECTED_MODELS = {
    "lgbm_cluster",
    "nhits",
    "nbeats",
    "mstl",
    "chronos2_enriched",
}


def test_algorithm_roster_contains_exactly_five_supported_models():
    cfg = load_forecast_pipeline_config()

    assert set(cfg["algorithms"]) == EXPECTED_MODELS
    assert set(get_competing_model_ids()) == EXPECTED_MODELS
    assert set(get_forecastable_model_ids()) == EXPECTED_MODELS


def test_fallbacks_and_confidence_intervals_stay_inside_lite_roster():
    cfg = load_forecast_pipeline_config()

    assert cfg["champion"]["fallback_model_id"] in EXPECTED_MODELS
    assert cfg["production_forecast"]["cold_start_model_id"] in EXPECTED_MODELS
    assert cfg["production_forecast"]["fallback_model_id"] in EXPECTED_MODELS
    assert set(cfg["production_forecast"]["confidence_interval"]["source_model_ids"]) <= EXPECTED_MODELS
