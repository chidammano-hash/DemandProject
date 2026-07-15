from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from common.services.customer_forecast import (
    _resolve_run_window,
    build_croston_forecast_rows,
    build_customer_forecast_window,
    get_customer_forecast_settings,
    load_customer_forecast_readiness,
    prepare_customer_history,
)


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


def test_prepare_customer_history_routes_every_active_series_to_croston() -> None:
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
            },
            {
                "item_id": "ITEM-1",
                "location_id": "LOC-1",
                "customer_no": "CUST-1",
                "startdate": "2026-03-01",
                "demand_qty": 4.0,
                "sales_qty": 4.0,
                "series_first_month": "2024-10-01",
            },
            {
                "item_id": "ITEM-2",
                "location_id": "LOC-1",
                "customer_no": "CUST-2",
                "startdate": "2026-05-01",
                "demand_qty": 3.0,
                "sales_qty": 3.0,
                "series_first_month": "2025-05-01",
            },
            {
                "item_id": "ITEM-3",
                "location_id": "LOC-2",
                "customer_no": "CUST-3",
                "startdate": "2026-03-01",
                "demand_qty": 0.0,
                "sales_qty": 0.0,
                "series_first_month": "2024-01-01",
            },
        ]
    )
    window = build_customer_forecast_window(date(2026, 7, 13), 18, 18)

    prepared = prepare_customer_history(frame, window, recent_sales_lookback_months=6)

    assert prepared.eligible_series_count == 2
    assert len(prepared.model_input) == 36
    assert prepared.model_input["qty"].sum() == pytest.approx(15.0)
    assert all(
        str(value).startswith("croston_customer_series_") for value in prepared.identity_by_sku
    )
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


def test_build_croston_forecast_rows_covers_every_active_series() -> None:
    frame = pd.DataFrame(
        [
            {
                "item_id": "ITEM-1",
                "location_id": "LOC-1",
                "customer_no": "CUST-1",
                "startdate": "2026-05-01",
                "demand_qty": 6.0,
                "sales_qty": 6.0,
                "series_first_month": "2026-05-01",
            },
            {
                "item_id": "ITEM-2",
                "location_id": "LOC-1",
                "customer_no": "CUST-2",
                "startdate": "2025-01-01",
                "demand_qty": 0.0,
                "sales_qty": 0.0,
                "series_first_month": "2024-01-01",
            },
        ]
    )
    window = build_customer_forecast_window(date(2026, 7, 13), 18, 18)
    prepared = prepare_customer_history(frame, window, recent_sales_lookback_months=6)

    rows = build_croston_forecast_rows(
        prepared,
        window,
        {
            "alpha": 0.1,
            "variant": "sba",
            "recursive": True,
            "recursive_damping": 0.5,
        },
    )

    assert len(rows) == 18
    assert set(rows["model_id"]) == {"croston"}
    assert rows.groupby(["item_id", "location_id", "customer_no"]).size().tolist() == [18]
    assert rows["forecast_qty"].nunique() > 1


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
            }
        ]
    )
    window = build_customer_forecast_window(date(2026, 7, 13), 18, 18)

    prepared = prepare_customer_history(frame, window, recent_sales_lookback_months=6)

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
            }
        ]
    )
    window = build_customer_forecast_window(date(2026, 7, 13), 18, 18)

    with pytest.raises(ValueError, match="negative demand"):
        prepare_customer_history(frame, window, recent_sales_lookback_months=6)


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
    cursor.fetchone.return_value = (date(2026, 6, 1), 12, 10, 2, 0, 0, 0, 91, 91, 0)
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor

    readiness = load_customer_forecast_readiness(
        conn,
        window,
        recent_sales_lookback_months=6,
    )

    sql = cursor.execute.call_args.args[0]
    assert "mv_customer_demand_series_profile" in sql
    assert "last_sales_month" in sql
    assert "audit_load_batch" in sql
    assert "domain = 'customer_demand'" in sql
    assert "customer_demand_profile_refresh_state" in sql
    assert "status = 'running'" in sql
    assert "first_month <=" not in sql
    assert "fact_customer_demand_monthly" not in sql
    assert readiness["eligible_series"] == 10
    assert readiness["croston_series"] == 10
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
    )

    assert readiness["ready"] is False
    assert blocker in readiness["blockers"][0].lower()
    assert readiness["profile_customer_demand_batch_id"] == profile_batch_id
    assert readiness["active_customer_demand_loads"] == active_load_count


def test_customer_forecast_settings_are_croston_only() -> None:
    config = {
        "customer_forecast": {
            "enabled": True,
            "model_id": "croston",
            "model_params": {
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

    assert settings["model_id"] == "croston"
    assert settings["model_params"] == {
        "alpha": 0.1,
        "variant": "sba",
        "recursive": True,
        "recursive_damping": 0.5,
    }
    assert "fallback_model_id" not in settings
    assert "fallback_params" not in settings
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


def test_run_window_is_restored_from_the_manifest() -> None:
    settings = {
        "enabled": True,
        "model_id": "croston",
        "model_params": {
            "alpha": 0.1,
            "variant": "sba",
            "recursive": True,
            "recursive_damping": 0.5,
        },
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
        "croston",
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
        "model_id": "croston",
        "model_params": {
            "alpha": 0.1,
            "variant": "sba",
            "recursive": True,
            "recursive_damping": 0.5,
        },
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
        "croston",
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
