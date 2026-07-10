"""Structural guards for run-scoped, auditable forecast promotion DDL."""

from pathlib import Path

MIGRATION = Path("sql/203_create_forecast_generation_run.sql")


def _ddl() -> str:
    return MIGRATION.read_text()


def test_generation_manifest_allows_only_ready_release_candidates_to_be_eligible():
    ddl = _ddl()
    compact = " ".join(ddl.split())

    assert "CREATE TABLE IF NOT EXISTS forecast_generation_run" in ddl
    assert "PRIMARY KEY (run_id)" in ddl
    assert "generation_purpose IN (" in ddl
    assert "'release_candidate'" in ddl
    assert "'snapshot_contender'" in ddl
    assert "'legacy_invalid'" in ddl
    assert "run_status IN ('generating', 'ready', 'invalid', 'promoted', 'archived')" in ddl
    assert "NOT promotion_eligible" in ddl
    assert "generation_purpose = 'release_candidate'" in ddl
    assert "run_status = 'ready'" in ddl
    for column in (
        "requested_model_id VARCHAR(100)",
        "forecast_month_generated DATE",
        "horizon_months SMALLINT",
        "dfu_count INTEGER",
        "champion_experiment_id INTEGER",
        "cluster_experiment_id INTEGER",
        "source_sales_batch_id BIGINT",
        "routing_artifact_checksum TEXT",
        "champion_results_checksum TEXT",
        "artifact_checksum TEXT",
    ):
        assert column in compact
    assert "REFERENCES champion_experiment (experiment_id)" in compact
    assert "REFERENCES cluster_experiment (experiment_id)" in compact
    assert "REFERENCES audit_load_batch (batch_id)" in compact
    assert "requested_model_id <> 'champion'" in ddl
    assert "ADD COLUMN IF NOT EXISTS champion_experiment_id INTEGER" in ddl
    assert "results_artifact_checksum" in ddl
    assert "results_forecast_checksum" in ddl
    assert "fk_external_forecast_champion_experiment" in ddl


def test_legacy_runs_are_classified_without_becoming_promotable():
    ddl = _ddl()

    roster_insert = ddl.index("FROM forecast_snapshot_roster")
    legacy_insert = ddl.index("FROM fact_production_forecast_staging")
    staging_backfill = ddl.index("UPDATE fact_production_forecast_staging AS staging")

    assert roster_insert < legacy_insert < staging_backfill
    assert "'snapshot_contender'" in ddl[roster_insert - 700 : roster_insert]
    assert "'legacy_invalid'" in ddl[legacy_insert - 900 : legacy_insert]
    assert "FALSE" in ddl[legacy_insert - 900 : legacy_insert]
    assert "legacy staging run predates the release-candidate manifest" in ddl


def test_staging_identity_is_run_and_purpose_scoped_with_manifest_fk():
    ddl = _ddl()
    compact = " ".join(ddl.split())

    assert "ADD COLUMN IF NOT EXISTS generation_purpose TEXT" in ddl
    assert "ADD COLUMN IF NOT EXISTS candidate_model_id VARCHAR(100)" in ddl
    assert "ALTER COLUMN generation_purpose SET NOT NULL" in ddl
    assert "ALTER COLUMN candidate_model_id SET NOT NULL" in ddl
    assert "DROP INDEX IF EXISTS uq_staging_model_dfu_month" in ddl
    assert "CREATE UNIQUE INDEX IF NOT EXISTS uq_staging_run_candidate_dfu_month" in ddl
    assert (
        "(run_id, generation_purpose, candidate_model_id, item_id, loc, forecast_month)" in compact
    )
    assert "FOREIGN KEY (run_id, generation_purpose, forecast_month_generated)" in ddl
    assert (
        "REFERENCES forecast_generation_run "
        "(run_id, generation_purpose, forecast_month_generated)" in compact
    )
    assert "GENERATED ALWAYS AS" in ddl
    assert "FOREIGN KEY (generation_run_id, generation_purpose, record_month)" in ddl
    assert "chk_staging_forecast_months" in ddl
    assert "chk_staging_forecast_quantities" in ddl
    assert "forecast_qty >= 0" in ddl
    assert "forecast_qty_lower <= forecast_qty" in ddl
    assert "forecast_qty <= forecast_qty_upper" in ddl
    assert "horizon_months > 0" in ddl


def test_champion_candidate_can_preserve_multiple_source_models():
    ddl = _ddl()

    assert "CHECK (model_id = candidate_model_id)" not in ddl
    assert (
        "(run_id, generation_purpose, candidate_model_id, item_id, loc, forecast_month)"
        in " ".join(ddl.split())
    )


def test_promotion_and_production_rows_gain_exact_lineage_evidence():
    ddl = _ddl()

    for column in (
        "source_run_id UUID",
        "production_run_id UUID",
        "gate_report JSONB",
        "candidate_checksum TEXT",
        "production_checksum TEXT",
        "archive_checksum TEXT",
        "archived_at TIMESTAMPTZ",
        "replaces_promotion_id INTEGER",
    ):
        assert column in ddl

    assert "ALTER TABLE fact_production_forecast" in ddl
    assert "ADD COLUMN IF NOT EXISTS promotion_log_id INTEGER" in ddl
    assert "ADD COLUMN IF NOT EXISTS lineage_status TEXT" in ddl
    assert "lineage_status IN ('legacy_unverified', 'verified')" in ddl
    assert "lineage_status = 'legacy_unverified'" in ddl
    assert "ADD COLUMN IF NOT EXISTS is_recursive BOOLEAN" in ddl
    assert "ADD COLUMN IF NOT EXISTS lag_source VARCHAR(20)" in ddl
    assert "ADD COLUMN IF NOT EXISTS source_promotion_id INTEGER" in ddl
    assert "fk_forecast_snapshot_source_promotion" in ddl
    assert "source_run_id IS NOT NULL" in ddl
    assert "promotion_log_id IS NOT NULL" in ddl
    assert "FOREIGN KEY (promotion_log_id, source_run_id, run_id)" in ddl
    assert "REFERENCES model_promotion_log (id, source_run_id, production_run_id)" in ddl
    assert "DEFERRABLE INITIALLY DEFERRED" in ddl
    assert "candidate_checksum IS NULL" in ddl
    assert "candidate_checksum = production_checksum" in ddl
    assert "archive_checksum IS NULL AND archived_at IS NULL" in ddl
    assert "replaces_promotion_id <> id" in ddl


def test_active_legacy_release_is_linked_only_when_run_and_count_are_unambiguous():
    ddl = _ddl()

    backfill = ddl.index("WITH production_run_summary AS")
    production_lineage = ddl.index("UPDATE fact_production_forecast AS forecast")

    assert backfill < production_lineage
    assert "COUNT(DISTINCT run_id) = 1" in ddl
    assert "promotion.total_rows = summary.row_count" in ddl
    assert "SET production_run_id = summary.production_run_id" in ddl
    assert "SET promotion_log_id = promotion.id" in ddl
    assert "lineage_status = 'legacy_unverified'" in ddl


def test_duplicate_active_promotions_are_repaired_before_unique_index():
    ddl = _ddl()

    repair = ddl.index("ROW_NUMBER() OVER")
    drop_old_index = ddl.index("DROP INDEX IF EXISTS idx_promotion_log_active")
    unique_index = ddl.index("CREATE UNIQUE INDEX IF NOT EXISTS uq_model_promotion_log_one_active")

    assert repair < drop_old_index < unique_index
    assert "ORDER BY promoted_at DESC, id DESC" in ddl
    assert "WHERE is_active" in ddl
    assert "SET is_active = FALSE" in ddl
    assert "ON model_promotion_log ((is_active))" in ddl
    assert "WHERE is_active" in ddl[unique_index : unique_index + 250]
