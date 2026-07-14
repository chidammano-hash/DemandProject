from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import MagicMock, patch

import pytest

from common.services.cache import InMemoryBackend
from tests.api.conftest import make_pool


@pytest.mark.asyncio
async def test_readiness_resolves_july_history_and_forecast_windows() -> None:
    pool, _conn, cursor = make_pool(fetchone_return=(date(2026, 6, 1), 12, 10, 0, 0, 0))

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

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert payload["history_start"] == "2025-01-01"
    assert payload["history_end"] == "2026-06-30"
    assert payload["forecast_start"] == "2026-07-01"
    assert payload["forecast_end"] == "2027-12-31"
    assert payload["eligible_series"] == 10
    assert payload["fallback_series"] == 2
    assert payload["forecastable_series"] == 12
    assert payload["skipped_series"] == 0
    assert "fact_customer_demand_monthly" in cursor.execute.call_args.args[0]


@pytest.mark.asyncio
async def test_generate_creates_run_and_submits_durable_job() -> None:
    pool, _conn, _cursor = make_pool(fetchone_return=(date(2026, 6, 1), 12, 10, 0, 0, 0))
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


@pytest.mark.asyncio
async def test_generate_blocks_when_latest_closed_month_is_missing() -> None:
    pool, _conn, _cursor = make_pool(fetchone_return=(date(2026, 5, 1), 12, 10, 0, 0, 0))

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
        "chronos2_enriched",
        created_at,
        created_at,
        created_at,
        None,
        {},
        {"chronos2_enriched": 10, "croston": 2},
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
    assert response.json()["model_id"] == "chronos2_enriched"
    assert response.json()["skip_reason_counts"] == {}
    assert response.json()["model_route_counts"] == {
        "chronos2_enriched": 10,
        "croston": 2,
    }


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
        "chronos2_enriched",
        created_at,
        created_at,
        created_at,
        "model failed",
        {},
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
