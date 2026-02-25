"""Tests for benchmark endpoint (Postgres vs Trino)."""

import pytest
from unittest.mock import patch, MagicMock
import httpx
from httpx import ASGITransport



@pytest.mark.asyncio
async def test_benchmark_invalid_domain(mock_pool):
    """GET /bench/compare with invalid domain should return 404."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/bench/compare?domain=nonexistent")
            assert response.status_code == 404


@pytest.mark.asyncio
async def test_benchmark_date_range_validation(mock_pool):
    """GET /bench/compare with inverted date range should return 422."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/bench/compare?domain=sales&start_date=2025-06-01&end_date=2024-01-01"
            )
            assert response.status_code == 422


@pytest.mark.asyncio
async def test_benchmark_success(mock_pool):
    """GET /bench/compare with mocked timing functions should return 200."""
    pool, _, _ = mock_pool
    mock_pg_runs = [0.01, 0.015, 0.012]
    mock_trino_runs = [0.05, 0.06, 0.055]

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("api.routers.benchmark.timed_postgres_query", return_value=mock_pg_runs),
        patch("api.routers.benchmark.timed_trino_query", return_value=mock_trino_runs),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/bench/compare?domain=sales&runs=3&warmup=0")
            assert response.status_code == 200
            data = response.json()
            assert data["domain"] == "sales"
            assert "results" in data
            assert len(data["results"]) >= 2  # at least count + page
            for result in data["results"]:
                assert "query" in result
                assert "postgres" in result
                assert "trino" in result
                assert "faster_backend" in result


@pytest.mark.asyncio
async def test_benchmark_trino_failure(mock_pool):
    """GET /bench/compare with Trino failure should return 503."""
    pool, _, _ = mock_pool
    mock_pg_runs = [0.01, 0.015]

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("api.routers.benchmark.timed_postgres_query", return_value=mock_pg_runs),
        patch("api.routers.benchmark.timed_trino_query", side_effect=RuntimeError("Trino not available")),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/bench/compare?domain=sales&runs=2&warmup=0")
            assert response.status_code == 503
