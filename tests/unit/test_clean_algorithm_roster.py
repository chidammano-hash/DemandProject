from scripts.ml.clean_algorithm_roster import DERIVED_IDS, REFERENCE_IDS, retained_ids


def test_cleanup_retains_active_models_and_derived_ensemble():
    assert retained_ids("backtest_lag_archive") == frozenset({
        "lgbm_cluster", "chronos2_enriched", "mstl", "nbeats", "nhits",
        "champion", "ensemble",
    })
    assert DERIVED_IDS == frozenset({"champion", "ensemble"})


def test_external_fact_preserves_benchmark_series():
    retained = retained_ids("fact_external_forecast_monthly")
    assert REFERENCE_IDS <= retained
    assert "catboost_cluster" not in retained


def test_ai_adjustment_fact_preserves_its_derived_identity():
    assert "ai_champion" in retained_ids("fact_ai_champion_forecast")
