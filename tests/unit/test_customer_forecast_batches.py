from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from common.services.customer_forecast import build_customer_forecast_window
from common.services.customer_forecast_batches import (
    CustomerForecastBatch,
    _build_batch_rows,
    _persist_completed_batch,
    claim_customer_forecast_batch,
    load_customer_forecast_progress,
    run_customer_forecast_worker,
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
        {"croston": 100_000},
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
        "model_id": "croston",
        "recent_sales_lookback_months": 6,
        "model_params": {"alpha": 0.1, "variant": "sba"},
    }

    rows, source = _build_batch_rows(
        batch,
        history,
        window,
        settings,
    )

    assert len(rows) == 18
    assert len(source) == 18
    assert set(rows["model_id"]) == {"croston"}


def test_worker_commits_read_transactions_before_claiming_and_persisting() -> None:
    """Long-lived workers must not retain claims or serialize batch completion."""
    conn = MagicMock()
    events: list[str] = []
    conn.commit.side_effect = lambda: events.append("commit")
    window = build_customer_forecast_window(date(2026, 7, 13), 18, 18)
    batch = CustomerForecastBatch(42, "croston", 0, 1, 1)
    history = pd.DataFrame([{"demand_qty": 1.0}])
    rows = pd.DataFrame([{"forecast_qty": 1.0}])
    source = pd.DataFrame([{"qty": 1.0}])

    with (
        patch(
            "common.services.customer_forecast_batches.get_customer_forecast_settings",
            return_value={"max_batch_attempts": 3},
        ),
        patch(
            "common.services.customer_forecast_batches._resolve_run_window",
            return_value=(window, "config-checksum"),
        ),
        patch(
            "common.services.customer_forecast_batches.claim_customer_forecast_batch",
            side_effect=lambda *_args, **_kwargs: (
                events.append("claim") or (batch if events.count("claim") == 1 else None)
            ),
        ),
        patch(
            "common.services.customer_forecast_batches.load_customer_forecast_batch_history",
            side_effect=lambda *_args: events.append("load") or history,
        ),
        patch(
            "common.services.customer_forecast_batches._build_batch_rows",
            side_effect=lambda *_args: events.append("build") or (rows, source),
        ),
        patch(
            "common.services.customer_forecast_batches._frame_checksum",
            return_value="source-checksum",
        ),
        patch(
            "common.services.customer_forecast_batches._persist_completed_batch",
            side_effect=lambda *_args: events.append("persist"),
        ),
    ):
        completed = run_customer_forecast_worker(conn, "run-1", ["croston"])

    assert completed == 1
    assert events == ["commit", "claim", "load", "commit", "build", "persist", "claim"]


def test_batch_persistence_rejects_customer_demand_lineage_drift_before_copy() -> None:
    cursor = MagicMock()
    cursor.fetchone.return_value = (91, 92, 92, 0)
    conn = _mock_connection(cursor)
    rows = pd.DataFrame(
        [
            {
                "item_id": "ITEM-1",
                "location_id": "LOC-1",
                "customer_no": "CUST-1",
                "forecast_month": pd.Timestamp("2026-07-01"),
                "forecast_qty": 1.0,
                "lower_bound": None,
                "upper_bound": None,
                "model_id": "croston",
            }
        ]
    )

    with pytest.raises(RuntimeError, match="Customer demand changed"):
        _persist_completed_batch(
            conn,
            "00000000-0000-0000-0000-000000000001",
            CustomerForecastBatch(42, "croston", 0, 1, 1),
            rows,
            "a" * 64,
            date(2026, 6, 30),
        )

    cursor.copy.assert_not_called()
    lineage_sql = cursor.execute.call_args_list[0].args[0]
    assert "source_customer_demand_batch_id" in lineage_sql
    assert "audit_load_batch" in lineage_sql
    assert "customer_demand_profile_refresh_state" in lineage_sql
    assert "status = 'running'" in lineage_sql


def test_batch_persistence_rejects_an_active_customer_demand_load() -> None:
    cursor = MagicMock()
    cursor.fetchone.return_value = (91, 91, 91, 1)
    conn = _mock_connection(cursor)

    with pytest.raises(RuntimeError, match="load is active"):
        _persist_completed_batch(
            conn,
            "00000000-0000-0000-0000-000000000001",
            CustomerForecastBatch(42, "croston", 0, 1, 1),
            pd.DataFrame(),
            "a" * 64,
            date(2026, 6, 30),
        )

    cursor.copy.assert_not_called()
