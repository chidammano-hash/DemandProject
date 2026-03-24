"""API tests for F3.1 Forecast Bias Correction endpoints."""
import pytest
import datetime
from unittest.mock import patch
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool

pytest_plugins = ["anyio"]

_CORRECTION_ROW = (
    "ITEM001",
    "LOC001",
    datetime.date(2025, 3, 1),
    "cluster",
    "cluster_A",
    -0.12,
    0.88,
    False,
    -12.0,
    False,
    True,
    6,
    datetime.datetime(2025, 3, 15, 8, 0, 0),
)

_FLAGGED_ROW = (
    "ITEM002",
    "LOC002",
    datetime.date(2025, 3, 1),
    "item",
    -0.25,
    0.72,
    0.75,
    True,
    4,
)

_HISTORY_ROW = (
    "cluster_A",
    datetime.date(2025, 2, 1),
    -0.10,
    0.90,
    150,
    0.18,
    0.15,
    True,
)


@pytest.mark.asyncio
async def test_get_bias_corrections_200():
    pool, conn, cursor = _make_pool(fetchone_return=(5,))
    cursor.fetchall.return_value = [_CORRECTION_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/bias-corrections")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 5
    assert "corrections" in data
    assert len(data["corrections"]) == 1
    item = data["corrections"][0]
    assert item["item_id"] == "ITEM001"
    assert item["loc"] == "LOC001"
    assert item["segment_type"] == "cluster"
    assert item["plan_month"] == "2025-03-01"
    assert item["computed_at"] is not None


@pytest.mark.asyncio
async def test_get_bias_corrections_empty():
    pool, conn, cursor = _make_pool(fetchone_return=(0,))
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/bias-corrections")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["corrections"] == []


@pytest.mark.asyncio
async def test_get_bias_corrections_with_filters():
    pool, conn, cursor = _make_pool(fetchone_return=(1,))
    cursor.fetchall.return_value = [_CORRECTION_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/bias-corrections",
                params={"item_id": "ITEM001", "loc": "LOC001", "segment_type": "cluster"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1


@pytest.mark.asyncio
async def test_get_bias_corrections_summary_200():
    pool, conn, cursor = _make_pool(
        fetchone_return=(100, 80, 5, 3, -0.08, 0.92, datetime.datetime(2025, 3, 15, 8, 0, 0)),
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/bias-corrections/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_corrections"] == 100
    assert data["dfu_count"] == 80
    assert data["flagged_count"] == 5
    assert data["clipped_count"] == 3
    assert data["avg_rolling_bias"] == pytest.approx(-0.08)
    assert data["avg_correction_factor"] == pytest.approx(0.92)
    assert data["last_computed_at"] is not None


@pytest.mark.asyncio
async def test_get_bias_corrections_summary_empty():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None  # router checks `if not row`

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/bias-corrections/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_corrections"] == 0
    assert data["dfu_count"] == 0


@pytest.mark.asyncio
async def test_get_bias_corrections_summary_with_plan_month():
    pool, conn, cursor = _make_pool(
        fetchone_return=(50, 40, 2, 1, -0.05, 0.95, datetime.datetime(2025, 3, 15, 8, 0, 0)),
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/bias-corrections/summary", params={"plan_month": "2025-03-01"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["plan_month"] == "2025-03-01"


@pytest.mark.asyncio
async def test_get_flagged_bias_corrections_200():
    pool, conn, cursor = _make_pool(fetchone_return=(3,))
    cursor.fetchall.return_value = [_FLAGGED_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/bias-corrections/flagged")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert "flagged" in data
    assert len(data["flagged"]) == 1
    item = data["flagged"][0]
    assert item["item_id"] == "ITEM002"
    assert item["correction_was_clipped"] is True


@pytest.mark.asyncio
async def test_get_flagged_bias_corrections_empty():
    pool, conn, cursor = _make_pool(fetchone_return=(0,))
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/bias-corrections/flagged")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["flagged"] == []


@pytest.mark.asyncio
async def test_get_bias_correction_history_200():
    pool, conn, cursor = _make_pool(fetchall_return=[_HISTORY_ROW])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/bias-corrections/history")

    assert resp.status_code == 200
    data = resp.json()
    assert "history" in data
    assert len(data["history"]) == 1
    h = data["history"][0]
    assert h["segment_value"] == "cluster_A"
    assert h["computation_month"] == "2025-02-01"
    assert h["correction_improved_accuracy"] is True


@pytest.mark.asyncio
async def test_get_bias_correction_history_empty():
    pool, conn, cursor = _make_pool(fetchall_return=[])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/bias-corrections/history", params={"segment_type": "item"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["history"] == []
    assert data["segment_type"] == "item"


@pytest.mark.asyncio
async def test_get_bias_correction_history_with_segment_value():
    pool, conn, cursor = _make_pool(fetchall_return=[_HISTORY_ROW])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/bias-corrections/history",
                params={"segment_type": "cluster", "segment_value": "cluster_A", "months": 3},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["segment_value"] == "cluster_A"
    assert len(data["history"]) == 1
