from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from common.core.constants import FORECAST_QTY_COL
from common.services.customer_forecast import (
    _resolve_run_window,
    build_croston_forecast_rows,
    build_customer_forecast_rows,
    build_customer_forecast_window,
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


def test_prepare_customer_history_routes_ineligible_series_to_croston() -> None:
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

    assert prepared.eligible_series_count == 1
    assert prepared.fallback_series_count == 1
    assert prepared.forecastable_series_count == 2
    assert len(prepared.model_input) == 18
    assert len(prepared.fallback_model_input) == 18
    assert prepared.model_input["qty"].sum() == pytest.approx(12.0)
    assert (
        prepared.model_input.loc[
            prepared.model_input["startdate"] == pd.Timestamp("2025-02-01"), "qty"
        ].item()
        == 0.0
    )
    assert set(prepared.fallback_reason_by_sku.values()) == {"insufficient_history"}
    assert prepared.skipped_series == [
        {
            "item_id": "ITEM-3",
            "location_id": "LOC-2",
            "customer_no": "CUST-3",
            "reason": "no_sales_last_6_months",
        }
    ]


def test_build_croston_forecast_rows_covers_every_fallback_series() -> None:
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
        {"alpha": 0.1, "variant": "sba"},
    )

    assert len(rows) == 18
    assert set(rows["model_id"]) == {"croston"}
    assert rows.groupby(["item_id", "location_id", "customer_no"]).size().tolist() == [18]


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
    assert prepared.fallback_series_count == 0
    assert prepared.forecastable_series_count == 0
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


def test_build_customer_forecast_rows_requires_complete_horizon() -> None:
    frame = pd.DataFrame(
        [
            {
                "item_id": "ITEM-1",
                "location_id": "LOC-1",
                "customer_no": "CUST-1",
                "startdate": "2026-03-01",
                "demand_qty": 2.0,
                "sales_qty": 2.0,
                "series_first_month": "2024-01-01",
            }
        ]
    )
    window = build_customer_forecast_window(date(2026, 7, 13), 18, 18)
    prepared = prepare_customer_history(frame, window, recent_sales_lookback_months=6)
    sku_ck = prepared.model_input["sku_ck"].iat[0]
    predictions = pd.DataFrame(
        {
            "sku_ck": [sku_ck] * 18,
            "startdate": pd.date_range("2026-07-01", periods=18, freq="MS"),
            FORECAST_QTY_COL: [5.0] * 18,
            "algorithm_id": ["chronos2_enriched"] * 18,
        }
    )

    rows = build_customer_forecast_rows(
        prepared,
        predictions,
        window,
        model_id="chronos2_enriched",
    )

    assert len(rows) == 18
    assert rows["forecast_qty"].sum() == pytest.approx(90.0)
    assert set(rows["model_id"]) == {"chronos2_enriched"}
    assert rows["forecast_month"].min() == pd.Timestamp("2026-07-01")
    assert rows["forecast_month"].max() == pd.Timestamp("2027-12-01")
    assert rows[["item_id", "location_id", "customer_no"]].drop_duplicates().to_dict("records") == [
        {"item_id": "ITEM-1", "location_id": "LOC-1", "customer_no": "CUST-1"}
    ]

    with pytest.raises(RuntimeError, match="complete 18-month forecast"):
        build_customer_forecast_rows(
            prepared,
            predictions.iloc[:-1],
            window,
            model_id="chronos2_enriched",
        )

    predictions.loc[0, FORECAST_QTY_COL] = -4.0
    clipped = build_customer_forecast_rows(
        prepared,
        predictions,
        window,
        model_id="chronos2_enriched",
    )
    assert clipped["forecast_qty"].min() == 0.0


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
    cursor.fetchone.return_value = (date(2026, 6, 1), 12, 8, 2, 0, 0, 0)
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
    assert "fact_customer_demand_monthly" not in sql
    assert readiness["eligible_series"] == 8
    assert readiness["fallback_series"] == 2
    assert readiness["dormant_series"] == 2
    assert readiness["forecastable_series"] == 10
    assert readiness["skipped_series"] == 2


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
        "model_id": "chronos2_enriched",
        "fallback_model_id": "croston",
        "fallback_params": {"alpha": 0.1, "variant": "sba"},
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
        "chronos2_enriched",
        "a" * 64,
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
