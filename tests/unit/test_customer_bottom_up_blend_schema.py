"""Structural guards for customer bottom-up blend evidence DDL."""

from pathlib import Path

MIGRATION = Path("sql/216_create_customer_bottom_up_blend.sql")


def _ddl() -> str:
    return MIGRATION.read_text()


def _compact(value: str) -> str:
    return " ".join(value.split())


def _function_body(ddl: str, function_name: str) -> str:
    start = ddl.index(f"CREATE OR REPLACE FUNCTION {function_name}")
    end = ddl.index("$$ LANGUAGE plpgsql;", start)
    return ddl[start:end]


def test_backtest_component_enforces_exact_one_month_horizon() -> None:
    ddl = _compact(_ddl())

    assert "forecast_month = (forecast_origin + INTERVAL '1 month')::date" in ddl
    assert "forecast_origin <= forecast_month" not in ddl


def test_backtest_component_formula_accepts_four_decimal_evidence() -> None:
    ddl = _compact(_ddl())

    assert "blended_qty NUMERIC(18,4) NOT NULL" in ddl
    assert "ABS( blended_qty - ( effective_customer_weight * normalized_customer_qty" in ddl
    assert ") <= 0.0001" in ddl


def test_backtest_run_freezes_exact_source_population_size() -> None:
    ddl = _compact(_ddl())

    assert "source_series_count INTEGER NOT NULL" in ddl
    assert "source_series_checksum TEXT NOT NULL" in ddl
    assert "AND source_series_count >= 0" in ddl
    assert "source_series_checksum ~ '^[0-9a-f]{64}$'" in ddl


def test_forward_components_reference_one_exact_backtest_lineage() -> None:
    ddl = _compact(_ddl())

    assert (
        "CONSTRAINT uq_customer_backtest_lineage UNIQUE ( run_id, customer_run_id, "
        "source_promotion_id, source_production_run_id )"
    ) in ddl


def test_accuracy_gate_cannot_configure_positive_wape_degradation() -> None:
    ddl = _compact(_ddl())

    assert "AND max_wape_degradation_pct = 0" in ddl
    assert "max_wape_degradation_pct >= 0" not in ddl
    assert (
        "CONSTRAINT fk_customer_bottom_up_blend_backtest_lineage FOREIGN KEY ( "
        "backtest_run_id, customer_run_id, source_promotion_id, "
        "source_production_run_id ) REFERENCES customer_forecast_backtest_run ( "
        "run_id, customer_run_id, source_promotion_id, source_production_run_id )"
    ) in ddl


def test_customer_forecast_rows_can_only_change_while_parent_is_generating() -> None:
    ddl = _ddl()
    compact = _compact(ddl)
    body = _function_body(ddl, "fact_customer_forecast_guard_completed_run")
    parent_body = _function_body(
        ddl,
        "customer_forecast_run_guard_completed_terminal",
    )

    assert "parent_status <> 'generating'" in body
    assert "FOR UPDATE OF run" in body
    assert "FROM inserted_rows" in body
    assert "FROM old_rows" in body
    assert "FROM new_rows" in body
    assert "FROM deleted_rows" in body
    assert "AFTER INSERT ON fact_customer_forecast" in compact
    assert "AFTER UPDATE ON fact_customer_forecast" in compact
    assert "AFTER DELETE ON fact_customer_forecast" in compact
    assert "REFERENCING NEW TABLE AS inserted_rows" in compact
    assert "REFERENCING OLD TABLE AS old_rows NEW TABLE AS new_rows" in compact
    assert "REFERENCING OLD TABLE AS deleted_rows" in compact
    assert "OLD.run_status = 'completed'" in parent_body
    assert "OLD.run_status <> 'generating'" in parent_body
    assert "EXISTS ( SELECT 1 FROM fact_customer_forecast" in _compact(parent_body)
    assert "TO_JSONB(NEW) - 'job_id'" in parent_body
    assert "BEFORE UPDATE OR DELETE ON customer_forecast_run" in compact


def test_customer_forecast_new_writes_are_croston_only_and_source_bound() -> None:
    ddl = _compact(_ddl())

    assert "ADD COLUMN IF NOT EXISTS source_customer_demand_batch_id BIGINT" in ddl
    assert "REFERENCES audit_load_batch (batch_id) NOT VALID" in ddl
    assert "idx_customer_forecast_source_demand_batch" in ddl
    assert "idx_audit_load_batch_customer_demand_completed" in ddl
    assert "CHECK (model_id = 'croston') NOT VALID" in ddl
    assert "CHECK (route_model_id = 'croston') NOT VALID" in ddl


def test_customer_job_reconciliation_has_a_run_identity_index() -> None:
    ddl = _compact(_ddl())

    assert "idx_job_history_customer_forecast_run" in ddl
    assert "ON job_history (job_type, ((params ->> 'run_id')), submitted_at DESC)" in ddl
    assert "'generate_customer_forecast_backtest'" in ddl
    assert "'generate_customer_forecast_blend'" in ddl


def test_customer_demand_profile_refresh_is_bound_to_one_exact_load_batch() -> None:
    ddl = _compact(_ddl())

    assert "CREATE TABLE IF NOT EXISTS customer_demand_profile_refresh_state" in ddl
    assert "singleton_id SMALLINT PRIMARY KEY DEFAULT 1" in ddl
    assert "CHECK (singleton_id = 1)" in ddl
    assert "source_batch_id BIGINT NOT NULL" in ddl
    assert "REFERENCES audit_load_batch (batch_id) ON DELETE RESTRICT" in ddl


def test_completed_backtest_manifest_is_immutable() -> None:
    ddl = _ddl()
    compact = _compact(ddl)
    body = _function_body(
        ddl,
        "customer_forecast_backtest_run_guard_completed_terminal",
    )

    assert "OLD.run_status = 'completed'" in body
    assert "TO_JSONB(NEW) - 'job_id'" in body
    assert "completed customer_forecast_backtest_run is immutable" in body
    assert "BEFORE UPDATE OR DELETE ON customer_forecast_backtest_run" in compact


def test_backtest_evidence_inserts_require_a_generating_parent() -> None:
    ddl = _ddl()
    compact = _compact(ddl)

    for table in (
        "customer_bottom_up_backtest_component",
        "customer_bottom_up_backtest_accuracy",
    ):
        insert_body = _function_body(ddl, f"{table}_guard_insert")
        mutation_body = _function_body(ddl, f"{table}_block_mutation")
        assert "FROM customer_forecast_backtest_run AS run" in insert_body
        assert "FROM inserted_rows" in insert_body
        assert "inserted.backtest_run_id = run.run_id" in insert_body
        assert "parent_status <> 'generating'" in insert_body
        assert "FOR UPDATE OF run" in insert_body
        assert f"{table} is append-only" in mutation_body
        assert (
            f"AFTER INSERT ON {table} REFERENCING NEW TABLE AS inserted_rows "
            f"FOR EACH STATEMENT EXECUTE FUNCTION {table}_guard_insert()"
        ) in compact
        assert f"BEFORE UPDATE OR DELETE ON {table}" in compact


def test_blend_component_inserts_require_a_generating_manifest() -> None:
    ddl = _ddl()
    compact = _compact(ddl)
    insert_body = _function_body(ddl, "customer_bottom_up_blend_component_guard_insert")
    mutation_body = _function_body(
        ddl,
        "customer_bottom_up_blend_component_block_mutation",
    )

    assert "FROM forecast_generation_run AS run" in insert_body
    assert "FROM inserted_rows" in insert_body
    assert "inserted.run_id = run.run_id" in insert_body
    assert "parent_status <> 'generating'" in insert_body
    assert "FOR UPDATE OF run" in insert_body
    assert "customer_bottom_up_blend_component is append-only" in mutation_body
    assert (
        "AFTER INSERT ON customer_bottom_up_blend_component "
        "REFERENCING NEW TABLE AS inserted_rows FOR EACH STATEMENT "
        "EXECUTE FUNCTION customer_bottom_up_blend_component_guard_insert()"
    ) in compact
    assert "BEFORE UPDATE OR DELETE ON customer_bottom_up_blend_component" in compact


def test_only_one_customer_blend_manifest_can_be_generating() -> None:
    ddl = _compact(_ddl())

    assert "CREATE UNIQUE INDEX IF NOT EXISTS uq_customer_bottom_up_blend_active" in ddl
    assert "ON forecast_generation_run ((1))" in ddl
    assert (
        "WHERE run_status = 'generating' AND metadata ? 'customer_bottom_up_blend'" in ddl
    )
