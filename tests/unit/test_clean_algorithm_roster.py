from scripts.ml.clean_algorithm_roster import REFERENCE_IDS, retained_ids


def test_cleanup_retains_only_active_models_and_champion():
    assert retained_ids("backtest_lag_archive") == frozenset({
        "lgbm_cluster", "chronos2_enriched", "mstl", "nbeats", "nhits", "champion",
    })


def test_external_fact_preserves_benchmark_series():
    retained = retained_ids("fact_external_forecast_monthly")
    assert REFERENCE_IDS <= retained
    assert "catboost_cluster" not in retained
