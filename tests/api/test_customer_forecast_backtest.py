from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from common.services.customer_forecast_backtest_population import (
    CustomerBacktestSourcePopulation,
)
from tests.api.conftest import make_pool

BACKTEST_RUN_ID = UUID("00000000-0000-0000-0000-000000000501")
CUSTOMER_RUN_ID = UUID("00000000-0000-0000-0000-000000000502")


@pytest.mark.asyncio
async def test_generate_customer_backtest_queues_config_driven_job() -> None:
    pool, _conn, cursor = make_pool(
        fetchone_returns=[
            (str(CUSTOMER_RUN_ID),),
            (
                42,
                "00000000-0000-0000-0000-000000000503",
                "00000000-0000-0000-0000-000000000504",
            ),
        ]
    )
    manager = MagicMock()
    manager.submit_job.return_value = "job_customer_backtest_1"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("common.services.job_registry.JobManager", return_value=manager),
        patch(
            "api.routers.forecasting.customer_forecast_blend."
            "compute_customer_backtest_source_population",
            return_value=CustomerBacktestSourcePopulation(2_646_964, 265, "d" * 64),
        ),
    ):
        from api.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/customer-forecast/backtest/generate")

    assert response.status_code == 202
    payload = response.json()
    assert payload["job_id"] == "job_customer_backtest_1"
    assert payload["status"] == "queued"
    queued_run_id = UUID(payload["run_id"])
    manager.submit_job.assert_called_once()
    assert manager.submit_job.call_args.args[0] == "generate_customer_forecast_backtest"
    assert manager.submit_job.call_args.args[1] == {
        "run_id": str(queued_run_id),
        "customer_run_id": str(CUSTOMER_RUN_ID),
    }
    insert_sql = next(
        call.args[0]
        for call in cursor.execute.call_args_list
        if "INSERT INTO customer_forecast_backtest_run" in call.args[0]
    )
    assert "'queued'" in insert_sql
    assert "source_series_count" in insert_sql
    assert "source_series_checksum" in insert_sql
    sql_calls = [call.args[0] for call in cursor.execute.call_args_list]
    assert any(
        "customer_forecast_run" in sql
        and "completed" in sql
        and "model_id" in sql
        and "config_checksum" in sql
        and "planning_month" in sql
        and "source_customer_demand_batch_id" in sql
        and "audit_load_batch" in sql
        and "customer_demand_profile_refresh_state" in sql
        and "status = 'running'" in sql
        for sql in sql_calls
    )
    assert any("model_promotion_log" in sql and "is_active" in sql for sql in sql_calls)
    source_call = next(
        call
        for call in cursor.execute.call_args_list
        if "model_promotion_log" in call.args[0]
    )
    assert "generation.forecast_month_generated = %s" in source_call.args[0]
    assert source_call.args[1] == (date(2026, 7, 1),)


@pytest.mark.asyncio
async def test_latest_customer_backtest_returns_one_common_cohort_comparison() -> None:
    completed_at = datetime(2026, 7, 14, 17, 0, tzinfo=UTC)
    run_row = (
        str(BACKTEST_RUN_ID),
        "job_customer_backtest_1",
        "completed",
        str(CUSTOMER_RUN_ID),
        date(2026, 7, 1),
        6,
        1200,
        7200,
        Decimal("150000.0000"),
        "a" * 64,
        completed_at,
        True,
        "passed",
        Decimal("-1.0000"),
        6,
        1000,
        Decimal("0.0000"),
        None,
    )
    accuracy_rows = [
        (
            "champion",
            7200,
            Decimal("150000.0000"),
            Decimal("30000.0000"),
            Decimal("4.1667"),
            Decimal("20.0000"),
            Decimal("2.0000"),
            Decimal("80.0000"),
        ),
        (
            "customer_bottom_up",
            7200,
            Decimal("150000.0000"),
            Decimal("33000.0000"),
            Decimal("4.5833"),
            Decimal("22.0000"),
            Decimal("-3.0000"),
            Decimal("78.0000"),
        ),
        (
            "customer_bottom_up_blend",
            7200,
            Decimal("150000.0000"),
            Decimal("28500.0000"),
            Decimal("3.9583"),
            Decimal("19.0000"),
            Decimal("-0.5000"),
            Decimal("81.0000"),
        ),
    ]
    pool, _conn, cursor = make_pool(
        fetchone_return=run_row,
        fetchall_return=accuracy_rows,
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/customer-forecast/backtest/latest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == str(BACKTEST_RUN_ID)
    assert payload["status"] == "completed"
    assert payload["customer_run_id"] == str(CUSTOMER_RUN_ID)
    assert payload["planning_month"] == "2026-07-01"
    assert payload["common_months"] == 6
    assert payload["common_dfus"] == 1200
    assert payload["common_rows"] == 7200
    assert payload["gate_passed"] is True
    assert payload["gate_reason"] == "passed"
    assert payload["blend_wape_degradation_pct"] == -1.0
    assert payload["min_common_months"] == 6
    assert payload["min_common_dfus"] == 1000
    assert payload["max_wape_degradation_pct"] == 0.0
    assert payload["error_summary"] is None
    assert [metric["model_id"] for metric in payload["metrics"]] == [
        "champion",
        "customer_bottom_up",
        "customer_bottom_up_blend",
    ]
    assert payload["metrics"][0]["wape_pct"] == 20.0
    assert payload["metrics"][1]["wape_pct"] == 22.0
    assert payload["metrics"][2]["wape_pct"] == 19.0
    sql_calls = [call.args[0] for call in cursor.execute.call_args_list]
    assert any("customer_forecast_backtest_run" in sql for sql in sql_calls)
    assert any("customer_bottom_up_backtest_accuracy" in sql for sql in sql_calls)


@pytest.mark.asyncio
async def test_latest_customer_backtest_exposes_terminal_error_summary() -> None:
    failed_at = datetime(2026, 7, 14, 17, 0, tzinfo=UTC)
    run_row = (
        str(BACKTEST_RUN_ID),
        "job_customer_backtest_1",
        "failed",
        str(CUSTOMER_RUN_ID),
        date(2026, 7, 1),
        None,
        None,
        None,
        None,
        None,
        failed_at,
        None,
        None,
        None,
        None,
        None,
        None,
        "Customer backtest worker failed.",
    )
    pool, _conn, _cursor = make_pool(fetchone_return=run_row, fetchall_return=[])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/customer-forecast/backtest/latest")

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["error_summary"] == "Customer backtest worker failed."
