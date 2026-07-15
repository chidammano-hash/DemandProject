from common.core.paths import SQL_DIR


def test_shadow_candidate_migration_is_non_promotable() -> None:
    ddl = (SQL_DIR / "217_add_customer_bottom_up_shadow_staging.sql").read_text()
    compact = " ".join(ddl.split()).lower()

    assert "'shadow_candidate'" in compact
    assert "chk_forecast_generation_run_purpose" in compact
    assert "chk_forecast_generation_run_promoted_purpose" in compact
    assert "run_status <> 'promoted' or generation_purpose = 'release_candidate'" in compact
    assert "chk_forecast_generation_run_shadow_ready_evidence" in compact
    assert "requested_model_id = 'customer_bottom_up'" in compact
    assert "metadata ? 'customer_bottom_up_staging'" in compact
    assert "trg_customer_bottom_up_shadow_staging_insert_guard" in compact
    assert "trg_customer_bottom_up_shadow_staging_update_guard" in compact
    assert "trg_customer_bottom_up_shadow_staging_delete_guard" in compact
    assert "trg_customer_bottom_up_shadow_manifest_guard" in compact
    assert "ready customer bottom-up shadow staging is immutable" in compact
    assert "referencing new table as inserted_rows" in compact
    assert "referencing old table as old_rows new table as new_rows" in compact
    assert "referencing old table as deleted_rows" in compact
    assert compact.count("for each statement") == 3
