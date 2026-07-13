"""Schema guard for the canonical champion-experiment model roster."""

from common.core.paths import PROJECT_ROOT

DDL = (PROJECT_ROOT / "sql" / "205_enforce_champion_model_roster.sql").read_text()
INITIAL_DDL = (PROJECT_ROOT / "sql" / "102_champion_experiments.sql").read_text()


def test_champion_experiment_default_contains_only_canonical_models() -> None:
    for model_id in (
        "lgbm_cluster",
        "nhits",
        "nbeats",
        "mstl",
        "chronos2_enriched",
    ):
        assert model_id in DDL

    for retired_model_id in ("catboost_cluster", "xgboost_cluster", "prophet", "arima"):
        assert retired_model_id not in DDL
        assert retired_model_id not in INITIAL_DDL


def test_new_experiment_rows_are_constrained_without_rewriting_history() -> None:
    assert "ALTER COLUMN models SET DEFAULT" in DDL
    assert "models <@" in DDL
    assert "jsonb_array_length(models) =" in DDL
    assert "models ? 'lgbm_cluster'" in DDL
    assert "NOT VALID" in DDL
