from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from common.ml.customer_forecast_rules import (
    CROSTON_ROUTE_ID,
    CUSTOMER_RULE_ROUTER_MODEL_ID,
    MOVING_AVERAGE_ROUTE_ID,
    SEASONAL_REPEAT_ROUTE_ID,
    CustomerForecastRuleParameters,
)
from common.services.customer_forecast import (
    _resolve_run_window,
    build_customer_forecast_rows,
    build_customer_forecast_window,
    get_customer_forecast_settings,
    load_customer_forecast_readiness,
    prepare_customer_history,
)

_RULE_PARAMS = CustomerForecastRuleParameters(
    recent_demand_lookback_months=6,
    moving_average_window_months=3,
    repeat_history_lookback_months=12,
    repeat_history_min_demand_months=9,
)
_CROSTON_PARAMS = {
    "alpha": 0.1,
    "variant": "sba",
    "recursive": True,
    "recursive_damping": 0.5,
}


def test_customer_forecast_window_uses_closed_history_and_current_month() -> None:
    window = build_customer_forecast_window(
        date(2026, 7, 13),
        history_months=18,
        horizon_months=18,
    )

    assert window.history_start == date(2025, 1, 1)
    assert window.history_end == date(2026, 6, 30)
    assert window.forecast_start == date(2026, 7, 1)
    assert window.forecast_end == date(2027, 12, 31)
    assert len(window.forecast_months) == 18


def test_prepare_customer_history_routes_active_series_by_demand_pattern() -> None:
    frame = pd.DataFrame(
        [
            {
                "item_id": "ITEM-1",
                "location_id": "LOC-1",
                "customer_no": "CUST-1",
                "startdate": "2026-03-01",
                "demand_qty": 8.0,
                "sales_qty": 8.0,
                "series_first_month": "2024-10-01",
                "series_first_demand_month": "2026-03-01",
            },
            {
                "item_id": "ITEM-1",
                "location_id": "LOC-1",
                "customer_no": "CUST-1",
                "startdate": "2026-03-01",
                "demand_qty": 4.0,
                "sales_qty": 4.0,
                "series_first_month": "2024-10-01",
                "series_first_demand_month": "2026-03-01",
            },
            {
                "item_id": "ITEM-2",
                "location_id": "LOC-1",
                "customer_no": "CUST-2",
                "startdate": "2026-05-01",
                "demand_qty": 3.0,
                "sales_qty": 3.0,
                "series_first_month": "2025-05-01",
                "series_first_demand_month": "2026-05-01",
            },
            {
                "item_id": "ITEM-3",
                "location_id": "LOC-2",
                "customer_no": "CUST-3",
                "startdate": "2026-03-01",
                "demand_qty": 0.0,
                "sales_qty": 0.0,
                "series_first_month": "2024-01-01",
                "series_first_demand_month": None,
            },
        ]
    )
    window = build_customer_forecast_window(date(2026, 7, 13), 18, 18)

    prepared = prepare_customer_history(
        frame,
        window,
        recent_sales_lookback_months=6,
        rule_params=_RULE_PARAMS,
    )

    assert prepared.eligible_series_count == 2
    assert len(prepared.model_input) == 36
    assert prepared.model_input["qty"].sum() == pytest.approx(15.0)
    assert set(prepared.route_by_sku.values()) == {MOVING_AVERAGE_ROUTE_ID}
    assert all(str(value).startswith("customer_series_") for value in prepared.identity_by_sku)
    assert (
        prepared.model_input.loc[
            prepared.model_input["startdate"] == pd.Timestamp("2025-02-01"), "qty"
        ]
        .eq(0.0)
        .all()
    )
    assert prepared.skipped_series == [
        {
            "item_id": "ITEM-3",
            "location_id": "LOC-2",
            "customer_no": "CUST-3",
            "reason": "no_sales_last_6_months",
        }
    ]


def test_build_customer_forecast_rows_covers_all_three_rule_routes() -> None:
    dense_months = pd.date_range("2025-07-01", periods=9, freq="MS")
    frame = pd.DataFrame(
        [
            {
                "item_id": "ITEM-1",
                "location_id": "LOC-1",
                "customer_no": "RECENT",
                "startdate": "2026-05-01",
                "demand_qty": 6.0,
                "sales_qty": 6.0,
                "series_first_month": "2026-05-01",
                "series_first_demand_month": "2026-05-01",
            },
            *[
                {
                    "item_id": "ITEM-2",
                    "location_id": "LOC-1",
                    "customer_no": "DENSE",
                    "startdate": month,
                    "demand_qty": float(offset + 1),
                    "sales_qty": float(offset + 1),
                    "series_first_month": "2025-07-01",
                    "series_first_demand_month": "2025-07-01",
                }
                for offset, month in enumerate(dense_months)
            ],
            {
                "item_id": "ITEM-3",
                "location_id": "LOC-1",
                "customer_no": "SPARSE",
                "startdate": "2025-01-01",
                "demand_qty": 5.0,
                "sales_qty": 0.0,
                "series_first_month": "2025-01-01",
                "series_first_demand_month": "2025-01-01",
            },
            {
                "item_id": "ITEM-3",
                "location_id": "LOC-1",
                "customer_no": "SPARSE",
                "startdate": "2026-05-01",
                "demand_qty": 3.0,
                "sales_qty": 3.0,
                "series_first_month": "2025-01-01",
                "series_first_demand_month": "2025-01-01",
            },
        ]
    )
    window = build_customer_forecast_window(date(2026, 7, 13), 18, 18)
    prepared = prepare_customer_history(
        frame,
        window,
        recent_sales_lookback_months=6,
        rule_params=_RULE_PARAMS,
    )

    rows = build_customer_forecast_rows(
        prepared,
        window,
        rule_params=_RULE_PARAMS,
        croston_params=_CROSTON_PARAMS,
    )

    assert len(rows) == 54
    assert set(rows["model_id"]) == {
        MOVING_AVERAGE_ROUTE_ID,
        SEASONAL_REPEAT_ROUTE_ID,
        CROSTON_ROUTE_ID,
    }
    assert rows.groupby(["item_id", "location_id", "customer_no"]).size().tolist() == [18] * 3
    assert rows.groupby("model_id")["forecast_qty"].nunique().gt(1).all()


def test_series_without_recent_six_month_sales_is_ignored() -> None:
    frame = pd.DataFrame(
        [
            {
                "item_id": "ITEM-1",
                "location_id": "LOC-1",
                "customer_no": "CUST-1",
                "startdate": "2025-02-01",
                "demand_qty": 9.0,
                "sales_qty": 9.0,
                "series_first_month": "2024-01-01",
                "series_first_demand_month": "2025-02-01",
            }
        ]
    )
    window = build_customer_forecast_window(date(2026, 7, 13), 18, 18)

    prepared = prepare_customer_history(
        frame,
        window,
        recent_sales_lookback_months=6,
        rule_params=_RULE_PARAMS,
    )

    assert prepared.eligible_series_count == 0
    assert len(prepared.skipped_series) == 1
    assert prepared.skipped_series[0]["reason"] == "no_sales_last_6_months"


def test_prepare_customer_history_rejects_negative_demand() -> None:
    frame = pd.DataFrame(
        [
            {
                "item_id": "ITEM-1",
                "location_id": "LOC-1",
                "customer_no": "CUST-1",
                "startdate": "2025-01-01",
                "demand_qty": -1.0,
                "sales_qty": 0.0,
                "series_first_month": "2024-01-01",
                "series_first_demand_month": "2025-01-01",
            }
        ]
    )
    window = build_customer_forecast_window(date(2026, 7, 13), 18, 18)

    with pytest.raises(ValueError, match="negative demand"):
        prepare_customer_history(
            frame,
            window,
            recent_sales_lookback_months=6,
            rule_params=_RULE_PARAMS,
        )


@pytest.mark.parametrize(
    "first_demand_values",
    [
        pytest.param([None, "2026-05-01"], id="mixed-null-and-date"),
        pytest.param([None, None], id="missing-for-positive-demand"),
        pytest.param(["2026-05-15", "2026-05-15"], id="not-month-start"),
    ],
)
def test_prepare_customer_history_rejects_invalid_first_demand_metadata(
    first_demand_values: list[str | None],
) -> None:
    frame = pd.DataFrame(
        [
            {
                "item_id": "ITEM-1",
                "location_id": "LOC-1",
                "customer_no": "CUST-1",
                "startdate": month,
                "demand_qty": 5.0,
                "sales_qty": 5.0,
                "series_first_month": "2025-01-01",
                "series_first_demand_month": first_demand,
            }
            for month, first_demand in zip(
                ("2026-05-01", "2026-06-01"),
                first_demand_values,
                strict=True,
            )
        ]
    )
    window = build_customer_forecast_window(date(2026, 7, 13), 18, 18)

    with pytest.raises(ValueError, match="first demand month"):
        prepare_customer_history(
            frame,
            window,
            recent_sales_lookback_months=6,
            rule_params=_RULE_PARAMS,
        )


def test_customer_forecast_migration_defines_run_and_fact_tables() -> None:
    ddl = Path("sql/210_create_customer_forecast.sql").read_text()

    assert "CREATE TABLE IF NOT EXISTS customer_forecast_run" in ddl
    assert "CREATE TABLE IF NOT EXISTS fact_customer_forecast" in ddl
    assert "UNIQUE (run_id, item_id, location_id, customer_no, forecast_month)" in ddl
    assert "forecast_qty >= 0" in ddl
    assert "uq_customer_forecast_one_active" in ddl
    assert "skip_reason_counts" in ddl


def test_customer_forecast_readiness_uses_precomputed_series_activity() -> None:
    window = build_customer_forecast_window(date(2026, 7, 13), 18, 18)
    cursor = MagicMock()
    cursor.fetchone.return_value = (
        date(2026, 6, 1),
        12,
        10,
        2,
        3,
        4,
        3,
        0,
        0,
        0,
        91,
        91,
        0,
    )
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor

    readiness = load_customer_forecast_readiness(
        conn,
        window,
        recent_sales_lookback_months=6,
        rule_params=_RULE_PARAMS,
    )

    sql = cursor.execute.call_args.args[0]
    assert "mv_customer_demand_series_profile" in sql
    assert "MAX(source_latest_month)" in sql
    assert "last_sales_month" in sql
    assert "audit_load_batch" in sql
    assert "domain = 'customer_demand'" in sql
    assert "customer_demand_profile_refresh_state" in sql
    assert "status = 'running'" in sql
    assert "first_month <=" not in sql
    assert "fact_customer_demand_monthly" not in sql
    assert readiness["eligible_series"] == 10
    assert readiness["moving_average_series"] == 3
    assert readiness["seasonal_repeat_series"] == 4
    assert readiness["croston_series"] == 3
    assert "fallback_series" not in readiness
    assert readiness["dormant_series"] == 2
    assert readiness["forecastable_series"] == 10
    assert readiness["skipped_series"] == 2
    assert readiness["source_customer_demand_batch_id"] == 91


def test_customer_forecast_readiness_fails_closed_without_load_lineage() -> None:
    window = build_customer_forecast_window(date(2026, 7, 13), 18, 18)
    cursor = MagicMock()
    cursor.fetchone.return_value = (
        date(2026, 6, 1),
        12,
        10,
        2,
        3,
        4,
        3,
        0,
        0,
        0,
        None,
        None,
        0,
    )
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor

    readiness = load_customer_forecast_readiness(
        conn,
        window,
        recent_sales_lookback_months=6,
        rule_params=_RULE_PARAMS,
    )

    assert readiness["ready"] is False
    assert readiness["source_customer_demand_batch_id"] is None
    assert "completed customer-demand load" in readiness["blockers"][0]


@pytest.mark.parametrize(
    ("profile_batch_id", "active_load_count", "blocker"),
    [
        pytest.param(90, 0, "profile", id="stale-profile"),
        pytest.param(91, 1, "active", id="active-load"),
    ],
)
def test_customer_forecast_readiness_requires_current_inactive_profile_lineage(
    profile_batch_id: int,
    active_load_count: int,
    blocker: str,
) -> None:
    window = build_customer_forecast_window(date(2026, 7, 13), 18, 18)
    cursor = MagicMock()
    cursor.fetchone.return_value = (
        date(2026, 6, 1),
        12,
        10,
        2,
        3,
        4,
        3,
        0,
        0,
        0,
        91,
        profile_batch_id,
        active_load_count,
    )
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor

    readiness = load_customer_forecast_readiness(
        conn,
        window,
        recent_sales_lookback_months=6,
        rule_params=_RULE_PARAMS,
    )

    assert readiness["ready"] is False
    assert blocker in readiness["blockers"][0].lower()
    assert readiness["profile_customer_demand_batch_id"] == profile_batch_id
    assert readiness["active_customer_demand_loads"] == active_load_count


def test_customer_forecast_settings_define_the_complete_rule_router() -> None:
    config = {
        "customer_forecast": {
            "enabled": True,
            "model_id": "customer_rule_router",
            "rule_params": {
                "recent_demand_lookback_months": 6,
                "moving_average_window_months": 3,
                "repeat_history_lookback_months": 12,
                "repeat_history_min_demand_months": 9,
            },
            "croston_params": {
                "alpha": 0.1,
                "variant": "sba",
                "recursive": True,
                "recursive_damping": 0.5,
            },
            "history_months": 18,
            "horizon_months": 18,
            "recent_sales_lookback_months": 6,
            "batch_size": 10_000,
            "cpu_workers": 6,
            "max_batch_attempts": 3,
            "progress_interval_seconds": 5,
        }
    }

    with patch(
        "common.services.customer_forecast.load_forecast_pipeline_config",
        return_value=config,
    ):
        settings = get_customer_forecast_settings()

    assert settings["model_id"] == CUSTOMER_RULE_ROUTER_MODEL_ID
    assert settings["rule_params"] == _RULE_PARAMS.as_dict()
    assert settings["croston_params"] == {
        "alpha": 0.1,
        "variant": "sba",
        "recursive": True,
        "recursive_damping": 0.5,
    }
    assert settings["route_model_ids"] == [
        MOVING_AVERAGE_ROUTE_ID,
        SEASONAL_REPEAT_ROUTE_ID,
        CROSTON_ROUTE_ID,
    ]
    assert "chronos_workers" not in settings


def test_customer_forecast_profile_migration_defines_refreshable_series_grain() -> None:
    ddl = Path("sql/211_create_customer_demand_series_profile.sql").read_text()

    assert "CREATE MATERIALIZED VIEW mv_customer_demand_series_profile" in ddl
    assert "MIN(startdate) AS first_month" in ddl
    assert "MAX(startdate) AS last_month" in ddl
    assert "UNIQUE INDEX" in ddl
    assert "(item_id, location_id, customer_no)" in ddl


def test_customer_forecast_route_migration_records_model_composition() -> None:
    ddl = Path("sql/212_add_customer_forecast_model_routes.sql").read_text()

    assert "ADD COLUMN IF NOT EXISTS model_route_counts JSONB" in ddl
    assert "jsonb_build_object(model_id, eligible_series)" in ddl
    assert "chk_customer_forecast_model_route_counts" in ddl


def test_customer_forecast_batch_migration_defines_resumable_work_ledger() -> None:
    ddl = Path("sql/213_add_customer_forecast_batches.sql").read_text()

    assert "CREATE TABLE IF NOT EXISTS customer_forecast_batch" in ddl
    assert "CREATE TABLE IF NOT EXISTS customer_forecast_batch_series" in ddl
    assert "completed_series" in ddl
    assert "completed_batches" in ddl
    assert "batch_id" in ddl
    assert "uq_customer_forecast_batch_route" in ddl
    assert "idx_customer_forecast_batch_claim" in ddl


def test_customer_activity_profile_migration_precomputes_last_sale() -> None:
    ddl = Path("sql/214_add_customer_series_activity.sql").read_text()

    assert "MAX(startdate) FILTER (WHERE sales_qty > 0) AS last_sales_month" in ddl
    assert "idx_mv_customer_demand_series_profile_last_sales" in ddl


def test_customer_batch_integrity_migration_couples_run_and_batch() -> None:
    ddl = Path("sql/215_enforce_customer_batch_lineage.sql").read_text()

    assert "UNIQUE (run_id, batch_id)" in ddl
    assert ddl.count("FOREIGN KEY (run_id, batch_id)") == 2


def test_customer_rule_router_migration_enables_all_three_series_routes() -> None:
    ddl = Path("sql/218_enable_customer_rule_router.sql").read_text()
    compact = " ".join(ddl.split())

    assert "first_demand_month" in ddl
    assert "demand_months_last_12" in ddl
    assert "source_latest_month" in ddl
    assert "CHECK (model_id = 'customer_rule_router') NOT VALID" in ddl
    assert "'moving_average_3'" in ddl
    assert "'seasonal_repeat_12'" in ddl
    assert "customer_model_id = 'customer_rule_router'" in ddl
    assert "UPDATE customer_forecast_backtest_run SET run_status = 'failed'" in compact
    assert "WHERE run_status IN ('queued', 'generating')" in compact
    assert "customer_model_id <> 'customer_rule_router'" in compact
    assert "UPDATE forecast_generation_run AS generation" in compact
    assert "generation.metadata ? 'customer_bottom_up_blend'" in compact
    assert "SET run_status = 'invalid', promotion_eligible = FALSE" in compact


def test_run_window_is_restored_from_the_manifest() -> None:
    settings = {
        "enabled": True,
        "model_id": CUSTOMER_RULE_ROUTER_MODEL_ID,
        "rule_params": _RULE_PARAMS.as_dict(),
        "croston_params": _CROSTON_PARAMS,
        "history_months": 18,
        "horizon_months": 18,
    }
    cursor = MagicMock()
    cursor.fetchone.return_value = (
        "queued",
        date(2026, 7, 1),
        date(2025, 1, 1),
        date(2026, 6, 30),
        date(2026, 7, 1),
        date(2027, 12, 31),
        18,
        18,
        CUSTOMER_RULE_ROUTER_MODEL_ID,
        "a" * 64,
        91,
        91,
        91,
        0,
    )
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor

    with patch(
        "common.services.customer_forecast.customer_forecast_config_checksum",
        return_value="a" * 64,
    ):
        window, checksum = _resolve_run_window(conn, "run-1", settings)

    assert window.history_start == date(2025, 1, 1)
    assert window.forecast_end == date(2027, 12, 31)
    assert checksum == "a" * 64


def test_run_window_rejects_newer_customer_demand_batch() -> None:
    settings = {
        "enabled": True,
        "model_id": CUSTOMER_RULE_ROUTER_MODEL_ID,
        "rule_params": _RULE_PARAMS.as_dict(),
        "croston_params": _CROSTON_PARAMS,
        "history_months": 18,
        "horizon_months": 18,
    }
    cursor = MagicMock()
    cursor.fetchone.return_value = (
        "generating",
        date(2026, 7, 1),
        date(2025, 1, 1),
        date(2026, 6, 30),
        date(2026, 7, 1),
        date(2027, 12, 31),
        18,
        18,
        CUSTOMER_RULE_ROUTER_MODEL_ID,
        "a" * 64,
        91,
        92,
        92,
        0,
    )
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor

    with (
        patch(
            "common.services.customer_forecast.customer_forecast_config_checksum",
            return_value="a" * 64,
        ),
        pytest.raises(RuntimeError, match="Customer demand changed"),
    ):
        _resolve_run_window(conn, "run-1", settings)


def test_customer_forecast_script_persists_only_public_failure_summary() -> None:
    source = Path("scripts/forecasting/generate_customer_forecasts.py").read_text()

    assert '"customer forecast generation failed"' in source
    assert "str(exc)" not in source
