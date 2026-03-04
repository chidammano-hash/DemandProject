"""API tests for /inv-planning/eoq/* endpoints — IPfeature4."""

import pytest
from unittest.mock import patch, MagicMock
import httpx
from httpx import ASGITransport


def _make_pool(fetchall_return=None, fetchone_return=None):
    cursor = MagicMock()
    cursor.fetchall.return_value = fetchall_return or []
    cursor.fetchone.return_value = fetchone_return or (0,)
    cursor.description = []

    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.connection.return_value = conn
    return pool, conn, cursor


# ---------------------------------------------------------------------------
# /inv-planning/eoq/summary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_eoq_summary_200():
    """GET /inv-planning/eoq/summary returns 200."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (150, 219.0, 1500.0, 5.5, 75000.0)
    cursor.fetchall.return_value = [
        ("A", 30, 310.0, 500.0, 25000.0, 3.9),
        ("B", 80, 220.0, 750.0, 35000.0, 5.5),
        ("C", 40, 80.0, 250.0, 15000.0, 8.0),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/eoq/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert "total_dfus" in data
    assert "avg_effective_eoq" in data
    assert "total_cycle_stock" in data
    assert "avg_order_frequency" in data
    assert "total_annual_cost" in data
    assert "by_abc" in data


@pytest.mark.asyncio
async def test_eoq_summary_empty_db():
    """Empty DB returns zeros/None, not 500."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0, None, None, None, None)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/eoq/summary")

    assert resp.status_code == 200
    assert resp.json()["total_dfus"] == 0
    assert resp.json()["by_abc"] == []


@pytest.mark.asyncio
async def test_eoq_summary_abc_filter():
    """abc_vol query param accepted without error."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (30, 310.0, 500.0, 3.9, 25000.0)
    cursor.fetchall.return_value = [("A", 30, 310.0, 500.0, 25000.0, 3.9)]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/eoq/summary?abc_vol=A")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_eoq_summary_by_abc_structure():
    """by_abc entries have all required keys."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (10, 200.0, 1000.0, 6.0, 5000.0)
    cursor.fetchall.return_value = [("A", 10, 200.0, 1000.0, 5000.0, 6.0)]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/eoq/summary")

    data = resp.json()
    assert len(data["by_abc"]) == 1
    entry = data["by_abc"][0]
    assert "abc_vol" in entry
    assert "count" in entry
    assert "avg_eoq" in entry
    assert "total_cycle_stock" in entry
    assert "total_annual_cost" in entry


# ---------------------------------------------------------------------------
# /inv-planning/eoq/detail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_eoq_detail_200():
    """GET /inv-planning/eoq/detail returns 200 with rows and total."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (5,)
    cursor.fetchall.return_value = [
        ("ITEM001", "LOC1", "A",
         100.0, 1200.0,
         50.0, 0.25, 10.0, 1.0,
         219.09, 219.09, 109.54, 5.48,
         273.86, 273.86, 547.72,
         None),
    ]
    cursor.description = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/eoq/detail")

    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "rows" in data
    assert isinstance(data["rows"], list)


@pytest.mark.asyncio
async def test_eoq_detail_pagination():
    """Pagination params accepted."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (100,)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/eoq/detail?limit=10&offset=20")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_eoq_detail_invalid_sort_falls_back():
    """Invalid sort_by falls back to total_annual_cost without 422."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/eoq/detail?sort_by=injection_attempt")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_eoq_detail_filter_by_item():
    """item filter accepted."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (2,)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/eoq/detail?item=ITEM001")

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /inv-planning/eoq/sensitivity
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_eoq_sensitivity_200():
    """GET /inv-planning/eoq/sensitivity returns 200 with curve."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (100.0,)

    import yaml
    import os

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "eoq_config.yaml"
    )
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.inv_planning.open", create=True,
               side_effect=lambda p, **kw: open(config_path)), \
         patch("yaml.safe_load", return_value=cfg):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/eoq/sensitivity")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_eoq_sensitivity_response_structure():
    """Sensitivity endpoint returns item_no, loc, avg_demand_monthly, curve."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (100.0,)

    import yaml
    import os

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "eoq_config.yaml"
    )
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    with patch("api.core._get_pool", return_value=pool), \
         patch("yaml.safe_load", return_value=cfg):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/eoq/sensitivity")

    assert resp.status_code == 200
    data = resp.json()
    assert "avg_demand_monthly" in data
    assert "curve" in data
    assert isinstance(data["curve"], list)
