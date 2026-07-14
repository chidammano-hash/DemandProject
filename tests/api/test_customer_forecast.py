from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import MagicMock, patch

import pytest

from common.services.cache import InMemoryBackend
from tests.api.conftest import make_pool


@pytest.mark.asyncio
async def test_readiness_resolves_july_history_and_forecast_windows() -> None:
    pool, _conn, cursor = make_pool(
        fetchone_return=(date(2026, 6, 1), 12, 10, 2, 0, 0, 0, 91, 91, 0)
    )

    cache = InMemoryBackend()
    with (
        patch("api.core._get_pool", return_value=pool),
        patch("common.services.cache.get_cache", return_value=cache),
        patch(
            "api.routers.forecasting.customer_forecast.get_planning_date",
            return_value=date(2026, 7, 13),
        ),
    ):
        from httpx import ASGITransport, AsyncClient

        from api.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/customer-forecast/readiness")
            repeated_response = await client.get("/customer-forecast/readiness")

    assert response.status_code == 200
    assert repeated_response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert payload["history_start"] == "2025-01-01"
    assert payload["history_end"] == "2026-06-30"
    assert payload["forecast_start"] == "2026-07-01"
    assert payload["forecast_end"] == "2027-12-31"
    assert payload["eligible_series"] == 10
    assert payload["croston_series"] == 10
    assert "fallback_series" not in payload
    assert payload["dormant_series"] == 2
    assert payload["forecastable_series"] == 10
    assert payload["skipped_series"] == 2
    assert payload["source_customer_demand_batch_id"] == 91
    readiness_calls = [
        call
        for call in cursor.execute.call_args_list
        if "mv_customer_demand_series_profile" in call.args[0]
    ]
    assert len(readiness_calls) == 1
    readiness_sql = readiness_calls[0].args[0]
    assert "mv_customer_demand_series_profile" in readiness_sql
    assert "last_sales_month" in readiness_sql
    assert "audit_load_batch" in readiness_sql
    assert "domain = 'customer_demand'" in readiness_sql
    assert "customer_demand_profile_refresh_state" in readiness_sql
    assert "status = 'running'" in readiness_sql
    assert "fact_customer_demand_monthly" not in readiness_sql


@pytest.mark.asyncio
async def test_generate_creates_run_and_submits_durable_job() -> None:
    pool, _conn, cursor = make_pool(
        fetchone_return=(date(2026, 6, 1), 12, 10, 2, 0, 0, 0, 91, 91, 0)
    )
    manager = MagicMock()
    manager.submit_job.return_value = "job_customer_1"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "api.routers.forecasting.customer_forecast.get_planning_date",
            return_value=date(2026, 7, 13),
        ),
        patch(
            "common.services.job_registry.JobManager",
            return_value=manager,
        ),
    ):
        from httpx import ASGITransport, AsyncClient

        from api.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/customer-forecast/generate")

    assert response.status_code == 202
    payload = response.json()
    assert payload["job_id"] == "job_customer_1"
    assert payload["status"] == "queued"
    assert payload["run_id"]
    manager.submit_job.assert_called_once()
    assert manager.submit_job.call_args.args[0] == "generate_customer_forecast"
    assert manager.submit_job.call_args.args[1]["run_id"] == payload["run_id"]
    insert_call = next(
        call
        for call in cursor.execute.call_args_list
        if "INSERT INTO customer_forecast_run" in call.args[0]
    )
    assert "source_customer_demand_batch_id" in insert_call.args[0]
    assert "started_at" in insert_call.args[0]
    assert 91 in insert_call.args[1]
    assert not any(
        "SET job_id = %s WHERE run_id" in call.args[0]
        for call in cursor.execute.call_args_list
    )


@pytest.mark.asyncio
async def test_generate_blocks_when_latest_closed_month_is_missing() -> None:
    pool, _conn, _cursor = make_pool(
        fetchone_return=(date(2026, 5, 1), 12, 10, 2, 0, 0, 0, 91, 91, 0)
    )

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "api.routers.forecasting.customer_forecast.get_planning_date",
            return_value=date(2026, 7, 13),
        ),
    ):
        from httpx import ASGITransport, AsyncClient

        from api.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/customer-forecast/generate")

    assert response.status_code == 409
    assert "June 2026" in response.json()["detail"]


@pytest.mark.asyncio
async def test_generate_requires_completed_customer_demand_batch_lineage() -> None:
    pool, _conn, _cursor = make_pool(
        fetchone_return=(date(2026, 6, 1), 12, 10, 2, 0, 0, 0, None, None, 0)
    )

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "api.routers.forecasting.customer_forecast.get_planning_date",
            return_value=date(2026, 7, 13),
        ),
    ):
        from httpx import ASGITransport, AsyncClient

        from api.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/customer-forecast/generate")

    assert response.status_code == 409
    assert "completed customer-demand load" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_customer_forecast_run_serializes_lineage() -> None:
    created_at = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
    run_row = (
        "0f2f73e8-9d8c-4f46-8410-2fce54ac68ad",
        "job_customer_1",
        "completed",
        date(2026, 7, 1),
        date(2025, 1, 1),
        date(2026, 6, 30),
        date(2026, 7, 1),
        date(2027, 12, 31),
        10,
        180,
        2,
        "croston",
        created_at,
        created_at,
        created_at,
        None,
        {},
        {"croston": 10},
        12,
        12,
        2,
        2,
    )
    pool, _conn, _cursor = make_pool(fetchone_return=run_row)

    with patch("api.core._get_pool", return_value=pool):
        from httpx import ASGITransport, AsyncClient

        from api.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/customer-forecast/runs/0f2f73e8-9d8c-4f46-8410-2fce54ac68ad"
            )

    assert response.status_code == 200
    assert response.json()["row_count"] == 180
    assert response.json()["model_id"] == "croston"
    assert response.json()["skip_reason_counts"] == {}
    assert response.json()["model_route_counts"] == {"croston": 10}
    select_sql = next(
        call.args[0]
        for call in _cursor.execute.call_args_list
        if "FROM customer_forecast_run run" in call.args[0]
    )
    assert "FROM job_history job" in select_sql
    assert "job.params ->> 'run_id' = run.run_id::text" in select_sql
    assert "job.submitted_at >= COALESCE(run.started_at, run.created_at)" in select_sql
    assert "job.created_at" not in select_sql


@pytest.mark.asyncio
async def test_export_rejects_an_incomplete_run() -> None:
    created_at = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
    run_row = (
        "0f2f73e8-9d8c-4f46-8410-2fce54ac68ad",
        "job_customer_1",
        "failed",
        date(2026, 7, 1),
        date(2025, 1, 1),
        date(2026, 6, 30),
        date(2026, 7, 1),
        date(2027, 12, 31),
        0,
        0,
        0,
        "croston",
        created_at,
        created_at,
        created_at,
        "model failed",
        {},
        12,
        5,
        2,
        1,
        {},
    )
    pool, _conn, _cursor = make_pool(fetchone_return=run_row)

    with patch("api.core._get_pool", return_value=pool):
        from httpx import ASGITransport, AsyncClient

        from api.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/customer-forecast/export",
                params={"run_id": "0f2f73e8-9d8c-4f46-8410-2fce54ac68ad"},
            )

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_retry_resumes_existing_customer_forecast_batches() -> None:
    pool, _conn, _cursor = make_pool(
        fetchone_return=("failed", date(2026, 7, 1), "a" * 64, 12, 91, 91, 91, 0)
    )
    manager = MagicMock()
    manager.submit_job.return_value = "job_customer_resume"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "api.routers.forecasting.customer_forecast.customer_forecast_config_checksum",
            return_value="a" * 64,
        ),
        patch("common.services.job_registry.JobManager", return_value=manager),
    ):
        from httpx import ASGITransport, AsyncClient

        from api.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/customer-forecast/runs/0f2f73e8-9d8c-4f46-8410-2fce54ac68ad/retry"
            )

    assert response.status_code == 202
    assert response.json()["job_id"] == "job_customer_resume"
    manager.submit_job.assert_called_once()
    assert manager.submit_job.call_args.args[1]["run_id"] == (
        "0f2f73e8-9d8c-4f46-8410-2fce54ac68ad"
    )
    executed_sql = [call.args[0] for call in _cursor.execute.call_args_list]
    assert any("SET attempt_count = 0" in sql for sql in executed_sql)


@pytest.mark.asyncio
async def test_retry_rejects_a_run_bound_to_an_older_customer_demand_load() -> None:
    pool, _conn, _cursor = make_pool(
        fetchone_return=("failed", date(2026, 7, 1), "a" * 64, 12, 91, 92, 92, 0)
    )

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "api.routers.forecasting.customer_forecast.customer_forecast_config_checksum",
            return_value="a" * 64,
        ),
        patch("common.services.job_registry.JobManager"),
    ):
        from httpx import ASGITransport, AsyncClient

        from api.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/customer-forecast/runs/0f2f73e8-9d8c-4f46-8410-2fce54ac68ad/retry"
            )

    assert response.status_code == 409
    assert "Customer demand changed" in response.json()["detail"]


def test_submission_reconciliation_repairs_jobs_and_retires_orphans() -> None:
    from api.routers.forecasting.customer_forecast import (
        _SUBMISSION_RECONCILIATION_GRACE_SECONDS,
        _reconcile_customer_forecast_submissions,
    )

    cursor = MagicMock()

    _reconcile_customer_forecast_submissions(cursor)

    repair_sql = cursor.execute.call_args_list[0].args[0]
    orphan_sql, orphan_params = cursor.execute.call_args_list[1].args
    assert "WITH latest_job AS" in repair_sql
    assert "job.params ->> 'run_id' = run.run_id::text" in repair_sql
    assert "job.submitted_at >= COALESCE(run.started_at, run.created_at)" in repair_sql
    assert "job.created_at" not in repair_sql
    assert "job.submitted_at >= COALESCE(run.started_at, run.created_at)" in orphan_sql
    assert "job.created_at" not in orphan_sql
    assert "job.error" not in repair_sql
    assert "'managed job failed'" in repair_sql
    assert "job submission was not persisted" in orphan_sql
    assert "COALESCE(run.started_at, run.created_at)" in orphan_sql
    assert orphan_params == (_SUBMISSION_RECONCILIATION_GRACE_SECONDS,)
