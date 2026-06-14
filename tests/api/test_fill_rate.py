"""API tests for IPfeature8 Fill Rate endpoints."""
import pytest
import httpx
from unittest.mock import MagicMock, patch
from tests.api.conftest import make_pool as _make_pool

pytest_plugins = ["anyio"]


@pytest.mark.asyncio
async def test_fill_rate_summary_200():
    pool, conn, cursor = _make_pool(
        fetchone_return=(0.95, 10000.0, 9500.0, 500.0, 3),
    )
    cursor.description = [
        ("portfolio_fill_rate",), ("total_ordered",), ("total_shipped",),
        ("total_shortage_qty",), ("partial_fulfillment_events",),
    ]
    cursor.fetchall.side_effect = [[], [], []]  # abc, worst, trend queries

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fill-rate/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "portfolio_fill_rate" in data
    assert "by_abc" in data
    assert "worst_items" in data
    assert "trend" in data


@pytest.mark.asyncio
async def test_fill_rate_trend_200():
    pool, conn, cursor = _make_pool(fetchall_return=[
        ("2025-01-01", 0.93, 5000.0, 4650.0, 350.0),
    ])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fill-rate/trend")
    assert resp.status_code == 200
    data = resp.json()
    assert "months" in data


@pytest.mark.asyncio
async def test_fill_rate_detail_200():
    pool, conn, cursor = _make_pool(
        fetchone_return=(25,),
        fetchall_return=[
            ("ITEM1", "LOC1", "2025-01-01", 1000.0, 920.0, 0.92, 80.0, False, "A", None, "East"),
        ],
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fill-rate/detail")
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "rows" in data


@pytest.mark.asyncio
async def test_fill_rate_detail_filter_abc():
    pool, conn, cursor = _make_pool(
        fetchone_return=(5,),
        fetchall_return=[("ITEM1", "LOC1", "2025-01-01", 1000.0, 950.0, 0.95, 50.0, True, "A", None, None)],
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fill-rate/detail?abc_vol=A")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_fill_rate_detail_pagination():
    pool, conn, cursor = _make_pool(
        fetchone_return=(100,),
        fetchall_return=[("ITEM1", "LOC1", "2025-01-01", 100.0, 90.0, 0.9, 10.0, False, "B", None, None)],
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fill-rate/detail?limit=10&offset=20")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Gap Analysis endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fill_rate_gap_analysis_200():
    """Gap analysis endpoint returns decomposition structure."""
    # 12 columns: total_skus, avg_fill_rate, shortage_sku_count, total_shortage_qty,
    #   total_ordered, ss_shortfall_count, ss_shortfall_qty, demand_spike_count,
    #   demand_spike_qty, lt_delay_count, lt_delay_qty, _dummy
    pool, conn, cursor = _make_pool(
        fetchone_return=(
            100,        # total_skus
            0.9300,     # avg_fill_rate
            30,         # shortage_sku_count
            5000.0,     # total_shortage_qty
            50000.0,    # total_ordered
            15,         # ss_shortfall_count
            2500.0,     # ss_shortfall_qty
            8,          # demand_spike_count
            1200.0,     # demand_spike_qty
            5,          # lt_delay_count
            800.0,      # lt_delay_qty
            "ITEM1",    # _dummy
        ),
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fill-rate/gap-analysis")
    assert resp.status_code == 200
    data = resp.json()
    assert "target_fill_rate" in data
    assert "actual_fill_rate" in data
    assert "gap_pct" in data
    assert "decomposition" in data
    assert len(data["decomposition"]) == 4
    causes = [d["cause"] for d in data["decomposition"]]
    assert "Safety Stock Shortfall" in causes
    assert "Demand Spike (>20% above forecast)" in causes
    assert "Lead Time Delay" in causes
    assert "Other / Data Gap" in causes
    # Each decomposition item has the expected keys
    for item in data["decomposition"]:
        assert "impact_pct" in item
        assert "sku_count" in item
        assert "shortage_qty" in item


@pytest.mark.asyncio
async def test_fill_rate_gap_analysis_empty():
    """Gap analysis returns empty decomposition when no data."""
    pool, conn, cursor = _make_pool(
        fetchone_return=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None),
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fill-rate/gap-analysis")
    assert resp.status_code == 200
    data = resp.json()
    assert data["decomposition"] == []
    assert data["actual_fill_rate"] is None


@pytest.mark.asyncio
async def test_fill_rate_gap_analysis_with_month_filter():
    """Gap analysis accepts month and abc_vol query params."""
    pool, conn, cursor = _make_pool(
        fetchone_return=(
            50, 0.9500, 10, 1000.0, 20000.0,
            5, 500.0, 3, 300.0, 2, 100.0, "ITEM1",
        ),
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fill-rate/gap-analysis?month=2026-03&abc_vol=A")
    assert resp.status_code == 200
    data = resp.json()
    assert data["month"] == "2026-03"
    assert data["target_fill_rate"] == 0.98  # A-class target


# ---------------------------------------------------------------------------
# Service Level Waterfall Bridge (Issue #16)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_level_waterfall_bridge_200():
    """Bridge endpoint returns target, actual, steps, and by_class."""
    # fetchall for abc_rows: (abc_vol, sku_count, avg_fill_rate, class_ordered)
    abc_rows = [
        ("A", 50, 0.985, 30000.0),
        ("B", 80, 0.932, 15000.0),
        ("C", 120, 0.878, 5000.0),
    ]
    # fetchone for total_row: (portfolio_fill_rate, total_ordered)
    total_row = (0.953, 50000.0)

    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = abc_rows
    cursor.fetchone.return_value = total_row

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/service-level/waterfall")
    assert resp.status_code == 200
    data = resp.json()
    assert "target" in data
    assert "actual" in data
    assert "steps" in data
    assert "by_class" in data
    assert data["actual"] == 0.953

    # Steps: first is Target (total), last is Actual (total)
    steps = data["steps"]
    assert len(steps) >= 2
    assert steps[0]["type"] == "total"
    assert steps[0]["label"] == "Target"
    assert steps[-1]["type"] == "total"
    assert steps[-1]["label"] == "Actual"

    # Delta steps should be positive or negative
    for step in steps[1:-1]:
        assert step["type"] in ("positive", "negative")

    # by_class should have 3 entries
    assert len(data["by_class"]) == 3
    for cls in data["by_class"]:
        assert "abc_class" in cls
        assert "target_sl" in cls
        assert "gap" in cls
        assert "weighted_gap" in cls


@pytest.mark.asyncio
async def test_service_level_waterfall_bridge_empty():
    """Bridge endpoint returns empty when no data."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = (None, None)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/service-level/waterfall")
    assert resp.status_code == 200
    data = resp.json()
    assert data["target"] is None
    assert data["actual"] is None
    assert data["steps"] == []


@pytest.mark.asyncio
async def test_service_level_waterfall_bridge_with_month():
    """Bridge endpoint accepts month query param."""
    abc_rows = [
        ("A", 40, 0.990, 25000.0),
        ("B", 60, 0.940, 10000.0),
    ]
    total_row = (0.975, 35000.0)

    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = abc_rows
    cursor.fetchone.return_value = total_row

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/service-level/waterfall?month=2026-03")
    assert resp.status_code == 200
    data = resp.json()
    assert data["month"] == "2026-03"
    assert data["actual"] == 0.975


@pytest.mark.asyncio
async def test_fill_rate_trend_handles_unpopulated_mv():
    """F1.3: /fill-rate/trend degrades to empty + warning when
    mv_fill_rate_monthly has not been refreshed, instead of 500-ing."""
    import psycopg
    pool, conn, cursor = _make_pool()
    cursor.execute.side_effect = psycopg.errors.ObjectNotInPrerequisiteState(
        'materialized view "mv_fill_rate_monthly" has not been populated'
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fill-rate/trend")
    assert resp.status_code == 200
    data = resp.json()
    assert data["months"] == []
    assert "warning" in data
    assert "refresh" in data["warning"].lower()
