"""Structural guards for the customer rule-router v2 migration."""

from pathlib import Path

MIGRATION = Path("sql/219_enable_customer_rule_router_v2.sql")


def _compact(value: str) -> str:
    return " ".join(value.split())


def test_router_v2_profile_preserves_v1_fields_and_adds_fixed_window_metrics() -> None:
    ddl = _compact(MIGRATION.read_text())

    for column in (
        "first_month",
        "last_month",
        "source_latest_month",
        "last_sales_month",
        "first_demand_month",
        "demand_months_last_12",
        "last_demand_month",
        "demand_months_last_18",
        "demand_sum_last_18",
        "demand_sumsq_last_18",
        "demand_months_recent_6",
        "demand_months_previous_6",
        "demand_sum_recent_6",
        "demand_sum_previous_6",
        "seasonal_repeat_validated",
    ):
        assert column in ddl

    assert ddl.count("source_month.latest_month - INTERVAL '17 months'") == 3
    assert ddl.count("source_month.latest_month - INTERVAL '11 months'") == 3
    assert ddl.count("source_month.latest_month - INTERVAL '6 months'") == 2
    assert ddl.count("source_month.latest_month - INTERVAL '5 months'") == 2
    assert (
        "MAX(demand.startdate) FILTER (WHERE demand.demand_qty > 0) AS last_demand_month"
    ) in ddl
    assert "demand.demand_qty * demand.demand_qty" in ddl
    assert ddl.count("COALESCE(") == 4
    assert "FALSE::boolean AS seasonal_repeat_validated" in ddl


def test_router_v2_profile_uses_the_atomic_locked_side_build() -> None:
    ddl = _compact(MIGRATION.read_text())

    assert "pg_advisory_xact_lock" in ddl
    assert "customer_demand_load_and_profile_refresh" in ddl
    assert "CREATE MATERIALIZED VIEW mv_customer_demand_series_profile_router_v2" in ddl
    assert "DROP MATERIALIZED VIEW IF EXISTS mv_customer_demand_series_profile;" in ddl
    assert (
        "ALTER MATERIALIZED VIEW mv_customer_demand_series_profile_router_v2 "
        "RENAME TO mv_customer_demand_series_profile"
    ) in ddl


def test_router_v2_constraints_allow_exactly_the_eight_customer_routes() -> None:
    ddl = MIGRATION.read_text()
    routes = {
        "moving_average_3",
        "trailing_average_6",
        "seasonal_repeat_12",
        "tsb",
        "adida",
        "croston",
        "ses",
        "holt_damped",
    }

    for route in routes:
        assert ddl.count(f"'{route}'") == 2
    assert "CHECK (model_id = 'customer_rule_router_v2') NOT VALID" in ddl
    assert "customer_model_id = 'customer_rule_router_v2'" in ddl


def test_router_v2_retires_only_active_pre_v2_customer_manifests() -> None:
    ddl = _compact(MIGRATION.read_text())

    assert "UPDATE customer_forecast_batch AS batch" in ddl
    assert "batch.batch_status IN ('pending', 'running')" in ddl
    assert "UPDATE customer_forecast_run SET run_status = 'failed'" in ddl
    assert "WHERE run_status IN ('queued', 'generating')" in ddl
    assert "model_id <> 'customer_rule_router_v2'" in ddl
    assert "UPDATE customer_forecast_backtest_run SET run_status = 'failed'" in ddl
    assert "customer_model_id <> 'customer_rule_router_v2'" in ddl
    assert "UPDATE forecast_generation_run AS generation" in ddl
    assert "generation.run_status IN ('generating', 'ready')" in ddl
    assert "generation.metadata ? 'customer_bottom_up_blend'" in ddl
    assert "SET run_status = 'invalid', promotion_eligible = FALSE" in ddl
