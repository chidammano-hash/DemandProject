from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd

from common.services.customer_forecast import build_customer_forecast_window
from common.services.customer_forecast_batches import (
    CustomerForecastBatch,
    _build_batch_rows,
    claim_customer_forecast_batch,
    load_customer_forecast_progress,
)


def _mock_connection(cursor: MagicMock) -> MagicMock:
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    conn.transaction.return_value.__enter__.return_value = None
    return conn


def test_batch_claim_uses_skip_locked_and_returns_durable_identity() -> None:
    cursor = MagicMock()
    cursor.fetchone.return_value = (42, "croston", 7, 10_000, 2)
    conn = _mock_connection(cursor)

    batch = claim_customer_forecast_batch(
        conn,
        "run-1",
        ["croston"],
        max_attempts=3,
    )

    sql = cursor.execute.call_args.args[0]
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert "attempt_count < %s" in sql
    assert batch is not None
    assert batch.batch_id == 42
    assert batch.route_model_id == "croston"
    assert batch.series_count == 10_000


def test_progress_reports_exact_customer_skus_and_throughput_eta() -> None:
    cursor = MagicMock()
    cursor.fetchone.return_value = (
        "generating",
        100_000,
        25_000,
        10,
        3,
        450_000,
        datetime.now(UTC) - timedelta(minutes=20),
        None,
        {"chronos2_enriched": 60_000, "croston": 40_000},
    )
    conn = _mock_connection(cursor)

    progress = load_customer_forecast_progress(conn, "run-1")

    assert progress["completed_series"] == 25_000
    assert progress["total_series"] == 100_000
    assert progress["completed_batches"] == 3
    assert progress["progress_pct"] == 32
    assert 3500 <= progress["eta_seconds"] <= 3700


def test_croston_batch_builds_all_horizon_rows_from_one_history_frame() -> None:
    window = build_customer_forecast_window(date(2026, 7, 13), 18, 18)
    batch = CustomerForecastBatch(42, "croston", 0, 1, 1)
    history = pd.DataFrame(
        [
            {
                "item_id": "ITEM-1",
                "location_id": "LOC-1",
                "customer_no": "CUST-1",
                "startdate": date(2026, 5, 1),
                "demand_qty": 4.0,
                "sales_qty": 4.0,
                "series_first_month": date(2026, 5, 1),
            }
        ]
    )
    settings = {
        "model_id": "chronos2_enriched",
        "fallback_model_id": "croston",
        "recent_sales_lookback_months": 6,
        "fallback_params": {"alpha": 0.1, "variant": "sba"},
    }

    rows, source = _build_batch_rows(
        batch,
        history,
        window,
        settings,
        MagicMock(),
    )

    assert len(rows) == 18
    assert len(source) == 18
    assert set(rows["model_id"]) == {"croston"}
