"""API tests for /inv-planning/health/* endpoints.

IPfeature6 — Inventory Health Score Dashboard.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# GET /inv-planning/health/summary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_summary_200():
    """GET /inv-planning/health/summary returns 200 with expected keys."""
    pool, conn, cursor = _make_pool()
    # First fetchone = summary row, fetchall = histogram rows
    cursor.fetchone.return_value = (500, 72.5, 200, 150, 100, 50, 18.0, 20.0, 22.0, 19.0)
    cursor.description = [
        ("total_skus",), ("avg_health_score",),
        ("healthy_count",), ("monitor_count",), ("at_risk_count",), ("critical_count",),
        ("avg_score_ss",), ("avg_score_dos",), ("avg_score_stockout",), ("avg_score_forecast",),
    ]
    cursor.fetchall.return_value = [
        ("60-79", 150),
        ("80-100", 200),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/health/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert "total_skus" in data
    assert "by_tier" in data
    assert "avg_health_score" in data
    assert "component_avgs" in data
    assert "score_histogram" in data


@pytest.mark.asyncio
async def test_health_summary_by_tier_keys():
    """by_tier has healthy, monitor, at_risk, critical keys."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (100, 65.0, 40, 30, 20, 10, 17.0, 18.0, 22.0, 16.0)
    cursor.description = [
        ("total_skus",), ("avg_health_score",),
        ("healthy_count",), ("monitor_count",), ("at_risk_count",), ("critical_count",),
        ("avg_score_ss",), ("avg_score_dos",), ("avg_score_stockout",), ("avg_score_forecast",),
    ]
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/health/summary")

    data = resp.json()
    for tier in ("healthy", "monitor", "at_risk", "critical"):
        assert tier in data["by_tier"]


@pytest.mark.asyncio
async def test_health_summary_component_avgs_keys():
    """component_avgs has expected score keys."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (50, 55.0, 10, 20, 15, 5, 12.0, 15.0, 20.0, 15.0)
    cursor.description = [
        ("total_skus",), ("avg_health_score",),
        ("healthy_count",), ("monitor_count",), ("at_risk_count",), ("critical_count",),
        ("avg_score_ss",), ("avg_score_dos",), ("avg_score_stockout",), ("avg_score_forecast",),
    ]
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/health/summary")

    data = resp.json()
    ca = data["component_avgs"]
    for key in ("ss_coverage", "dos_target", "stockout_risk", "forecast_accuracy"):
        assert key in ca


@pytest.mark.asyncio
async def test_health_summary_with_filters():
    """Filter query params accepted without error."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (10, 70.0, 5, 3, 2, 0, 18.0, 20.0, 22.0, 18.0)
    cursor.description = [
        ("total_skus",), ("avg_health_score",),
        ("healthy_count",), ("monitor_count",), ("at_risk_count",), ("critical_count",),
        ("avg_score_ss",), ("avg_score_dos",), ("avg_score_stockout",), ("avg_score_forecast",),
    ]
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/health/summary?abc_vol=A&variability_class=low")

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /inv-planning/health/detail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_detail_200():
    """GET /inv-planning/health/detail returns 200 with total + rows."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (25,)
    cursor.fetchall.return_value = [
        ("ITEM001", "LOC1", "A", "low", "cluster_1",
         82, "healthy",
         25, 25, 25, 20,
         1.8, 22.5, 15.0, 30.0, False,
         0.12, 0),
    ]
    cursor.description = [
        ("item_id",), ("loc",), ("abc_vol",), ("variability_class",), ("cluster_assignment",),
        ("health_score",), ("health_tier",),
        ("score_ss_coverage",), ("score_dos_target",), ("score_stockout_risk",), ("score_forecast_accuracy",),
        ("ss_coverage",), ("current_dos",), ("target_dos_min",), ("target_dos_max",), ("is_below_ss",),
        ("recent_wape",), ("stockout_count_3m",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/health/detail")

    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "rows" in data
    assert isinstance(data["rows"], list)


@pytest.mark.asyncio
async def test_health_detail_row_keys():
    """Detail rows contain expected keys."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [
        ("ITEM001", "LOC1", "A", "low", "cluster_1",
         75, "monitor",
         18, 25, 20, 20,
         1.2, 20.0, 15.0, 30.0, False,
         0.22, 0),
    ]
    cursor.description = [
        ("item_id",), ("loc",), ("abc_vol",), ("variability_class",), ("cluster_assignment",),
        ("health_score",), ("health_tier",),
        ("score_ss_coverage",), ("score_dos_target",), ("score_stockout_risk",), ("score_forecast_accuracy",),
        ("ss_coverage",), ("current_dos",), ("target_dos_min",), ("target_dos_max",), ("is_below_ss",),
        ("recent_wape",), ("stockout_count_3m",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/health/detail")

    row = resp.json()["rows"][0]
    for key in ("item_id", "loc", "health_score", "health_tier",
                "score_ss_coverage", "score_dos_target", "score_stockout_risk", "score_forecast_accuracy"):
        assert key in row


@pytest.mark.asyncio
async def test_health_detail_filter_by_tier():
    """health_tier filter query param accepted without error."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (5,)
    cursor.fetchall.return_value = []
    cursor.description = [
        ("item_id",), ("loc",), ("abc_vol",), ("variability_class",), ("cluster_assignment",),
        ("health_score",), ("health_tier",),
        ("score_ss_coverage",), ("score_dos_target",), ("score_stockout_risk",), ("score_forecast_accuracy",),
        ("ss_coverage",), ("current_dos",), ("target_dos_min",), ("target_dos_max",), ("is_below_ss",),
        ("recent_wape",), ("stockout_count_3m",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/health/detail?health_tier=critical")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_detail_empty_view():
    """Empty materialized view returns empty list, not 500."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    cursor.description = [
        ("item_id",), ("loc",), ("abc_vol",), ("variability_class",), ("cluster_assignment",),
        ("health_score",), ("health_tier",),
        ("score_ss_coverage",), ("score_dos_target",), ("score_stockout_risk",), ("score_forecast_accuracy",),
        ("ss_coverage",), ("current_dos",), ("target_dos_min",), ("target_dos_max",), ("is_below_ss",),
        ("recent_wape",), ("stockout_count_3m",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/health/detail")

    assert resp.status_code == 200
    assert resp.json()["rows"] == []
    assert resp.json()["total"] == 0


# ---------------------------------------------------------------------------
# GET /inv-planning/health/heatmap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_heatmap_200():
    """GET /inv-planning/health/heatmap returns 200 with cells."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("A", "low",    82.5, 50, 2),
        ("A", "medium", 65.0, 30, 5),
        ("B", "low",    74.0, 40, 3),
        ("C", "high",   45.0, 20, 8),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/health/heatmap")

    assert resp.status_code == 200
    data = resp.json()
    assert "x_labels" in data
    assert "y_labels" in data
    assert "cells" in data
    assert isinstance(data["cells"], list)


@pytest.mark.asyncio
async def test_health_heatmap_cell_keys():
    """Heatmap cells have expected keys."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [("A", "low", 80.0, 10, 1)]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/health/heatmap")

    data = resp.json()
    assert len(data["cells"]) == 1
    cell = data["cells"][0]
    for key in ("x", "y", "avg_health_score", "count", "critical_count"):
        assert key in cell


@pytest.mark.asyncio
async def test_health_heatmap_custom_group_params():
    """Custom group_x / group_y query params accepted."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/health/heatmap?group_x=abc_vol&group_y=cluster_assignment"
            )

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_heatmap_invalid_group_falls_back_to_default():
    """Invalid group param uses fallback, does not raise 422."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/health/heatmap?group_x=invalid_col&group_y=also_invalid"
            )

    assert resp.status_code == 200
