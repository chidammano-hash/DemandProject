"""Tests for SKU features endpoints."""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool


@pytest.mark.asyncio
async def test_summary_returns_aggregate_stats():
    pool, _, cursor = _make_pool()
    ts = datetime.datetime(2026, 4, 1, 12, 0, 0, tzinfo=datetime.UTC)

    # Four sequential fetchall/fetchone calls in summary:
    # 1. fetchone: total + latest_ts
    # 2. fetchall: seasonality_distribution
    # 3. fetchall: variability_distribution
    # 4. fetchall: trend_distribution
    # 5. fetchone: averages
    cursor.fetchone.side_effect = [
        (150, ts),  # total + max ts
        (0.45, 0.32, 0.61, 0.12, 1.5, 0.08, 500.0, 0.55),  # averages
    ]
    cursor.fetchall.side_effect = [
        [("seasonal", 80), ("flat", 50), ("trending", 20)],
        [("smooth", 60), ("erratic", 50), ("lumpy", 40)],
        [(-1, 30), (0, 90), (1, 30)],
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sku-features/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_skus"] == 150
    assert data["last_computed"] is not None
    assert "seasonal" in data["distributions"]["seasonality_profile"]
    assert len(data["distributions"]["variability_class"]) == 3
    assert len(data["distributions"]["trend_direction"]) == 3
    assert "cv_demand" in data["averages"]
    assert "Cache-Control" in resp.headers


@pytest.mark.asyncio
async def test_summary_empty_table():
    pool, _, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        (0, None),  # no features computed
        (None, None, None, None, None, None, None, None),
    ]
    cursor.fetchall.side_effect = [[], [], []]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sku-features/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_skus"] == 0
    assert data["last_computed"] is None


@pytest.mark.asyncio
async def test_list_returns_paginated_rows():
    pool, _, cursor = _make_pool()
    ts = datetime.datetime(2026, 4, 1, tzinfo=datetime.UTC)

    # Build a mock row with the correct number of columns:
    # sku_ck, item_id, loc, ml_cluster, seasonality_profile, variability_class,
    # trend_direction, features_computed_ts, + 36 feature columns = 44 total
    feature_vals = [0.5] * 36  # 36 feature columns
    mock_row = ("SKU001|LOC1", "SKU001", "LOC1", "C1", "seasonal", "smooth", 1, ts, *feature_vals)

    cursor.fetchone.side_effect = [(1,)]  # count
    cursor.fetchall.side_effect = [[mock_row]]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sku-features/list?limit=10&offset=0")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["limit"] == 10
    assert data["offset"] == 0
    assert len(data["rows"]) == 1
    assert data["rows"][0]["item_id"] == "SKU001"
    assert data["rows"][0]["seasonality_profile"] == "seasonal"


@pytest.mark.asyncio
async def test_list_with_filters():
    pool, _, cursor = _make_pool()

    cursor.fetchone.side_effect = [(0,)]
    cursor.fetchall.side_effect = [[]]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sku-features/list",
                params={
                    "seasonality_profile": "seasonal",
                    "variability_class": "smooth",
                    "trend_direction": 1,
                    "search": "ABC",
                    "sort_by": "cv_demand",
                    "sort_dir": "desc",
                },
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["rows"] == []


@pytest.mark.asyncio
async def test_list_invalid_sort_falls_back():
    """Invalid sort_by should fall back to item_id without error."""
    pool, _, cursor = _make_pool()
    cursor.fetchone.side_effect = [(0,)]
    cursor.fetchall.side_effect = [[]]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sku-features/list?sort_by=invalid_col")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_distributions_returns_histograms():
    pool, _, cursor = _make_pool()

    # For each of the 6 histogram features, we need:
    # 1. fetchone: min/max
    # 2. fetchall: bucket rows
    # So 6 fetchone calls + 6 fetchall calls
    cursor.fetchone.side_effect = [
        (0.1, 2.5),   # demand_cv min/max
        (0.0, 1.0),   # seasonal_amplitude min/max
        (0.0, 1.0),   # trend_r2 min/max
        (0.0, 1.0),   # intermittency_ratio min/max
        (1.0, 5.0),   # adi min/max
        (-0.5, 0.5),  # cagr min/max
    ]
    cursor.fetchall.side_effect = [
        [(1, 10), (2, 20), (3, 15)],  # demand_cv buckets
        [(1, 30), (2, 25)],            # seasonal_amplitude
        [(1, 40)],                      # trend_r2
        [(1, 50), (5, 10)],            # intermittency_ratio
        [(1, 35), (10, 5)],            # adi
        [(1, 20), (2, 15), (3, 10)],   # cagr
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sku-features/distributions?bins=20")

    assert resp.status_code == 200
    data = resp.json()
    assert data["bins"] == 20
    assert "features" in data
    assert "cv_demand" in data["features"]
    assert len(data["features"]["cv_demand"]) == 3
    # Each bucket has bin_start, bin_end, count
    bucket = data["features"]["cv_demand"][0]
    assert "bin_start" in bucket
    assert "bin_end" in bucket
    assert "count" in bucket


@pytest.mark.asyncio
async def test_distributions_empty_feature():
    """When a feature has no data, return an empty array."""
    pool, _, cursor = _make_pool()

    # All features return NULL min/max
    cursor.fetchone.side_effect = [
        (None, None),  # demand_cv
        (None, None),  # seasonal_amplitude
        (None, None),  # trend_r2
        (None, None),  # intermittency_ratio
        (None, None),  # adi
        (None, None),  # cagr
    ]
    # No fetchall calls should occur when min/max are None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sku-features/distributions")

    assert resp.status_code == 200
    data = resp.json()
    for feat in data["features"]:
        assert data["features"][feat] == []


@pytest.mark.asyncio
async def test_distributions_identical_values():
    """When all values for a feature are identical, return single bin."""
    pool, _, cursor = _make_pool()

    # First feature: identical values, rest: no data
    cursor.fetchone.side_effect = [
        (0.5, 0.5),   # demand_cv: same min and max
        (42,),         # count for identical-value case
        (None, None),  # seasonal_amplitude
        (None, None),  # trend_r2
        (None, None),  # intermittency_ratio
        (None, None),  # adi
        (None, None),  # cagr
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sku-features/distributions")

    assert resp.status_code == 200
    data = resp.json()
    cv_dist = data["features"]["cv_demand"]
    assert len(cv_dist) == 1
    assert cv_dist[0]["count"] == 42
    assert cv_dist[0]["bin_start"] == 0.5
    assert cv_dist[0]["bin_end"] == 0.5


@pytest.mark.asyncio
async def test_compute_submits_job():
    """POST /sku-features/compute should submit a background job."""
    pool, _, _cursor = _make_pool()

    mock_mgr = MagicMock()
    mock_mgr.submit_job.return_value = "job-abc-123"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "common.services.job_registry.JobManager",
            return_value=mock_mgr,
        ),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/sku-features/compute",
                json={"time_window_months": 24},
            )

    assert resp.status_code == 202
    data = resp.json()
    assert data["job_id"] == "job-abc-123"
    assert data["status"] == "queued"
    mock_mgr.submit_job.assert_called_once_with(
        "compute_sku_features",
        {"time_window_months": 24},
        label="Compute SKU Features",
        triggered_by="api",
    )


@pytest.mark.asyncio
async def test_compute_default_params():
    """POST /sku-features/compute with empty body uses default 36 months."""
    pool, _, _cursor = _make_pool()

    mock_mgr = MagicMock()
    mock_mgr.submit_job.return_value = "job-def-456"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "common.services.job_registry.JobManager",
            return_value=mock_mgr,
        ),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/sku-features/compute")

    assert resp.status_code == 202
    mock_mgr.submit_job.assert_called_once()
    call_params = mock_mgr.submit_job.call_args[0][1]
    assert call_params["time_window_months"] == 36
