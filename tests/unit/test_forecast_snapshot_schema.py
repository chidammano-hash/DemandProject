"""Structural checks for the bounded forecast snapshot migration."""
from pathlib import Path

MIGRATION = Path("sql/202_create_forecast_snapshot.sql")


def test_snapshot_migration_enforces_frozen_roster_and_six_lags():
    ddl = MIGRATION.read_text()

    assert "CREATE TABLE IF NOT EXISTS forecast_snapshot_roster" in ddl
    assert "contender_rank BETWEEN 1 AND 3" in ddl
    assert "generation_run_id IS NOT NULL" in ddl
    assert "REFERENCES backtest_run(id)" in ddl
    assert "CREATE TABLE IF NOT EXISTS fact_forecast_snapshot" in ddl
    assert "FOREIGN KEY (record_month, model_id)" in ddl
    assert "REFERENCES forecast_snapshot_roster (record_month, model_id)" in ddl
    assert "chk_fact_forecast_snapshot_lag CHECK (lag BETWEEN 0 AND 5)" in ddl
    assert "CREATE MATERIALIZED VIEW IF NOT EXISTS agg_accuracy_snapshot" in ddl
    assert "CURRENT_TIMESTAMP AS last_refresh_at" in ddl
