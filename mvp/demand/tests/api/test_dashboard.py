"""Tests for dashboard endpoints — /dashboard/kpis, /dashboard/alerts,
/dashboard/top-movers, /dashboard/heatmap."""

import pytest
from unittest.mock import patch, MagicMock
import httpx
from httpx import ASGITransport


@pytest.fixture
def mock_pool():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    mock_cursor.description = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.connection.return_value = mock_conn

    return pool, mock_conn, mock_cursor


# ===========================================================================
# /dashboard/kpis
# ===========================================================================

@pytest.mark.asyncio
async def test_dashboard_kpis_returns_structure(mock_pool):
    """GET /dashboard/kpis returns KPI structure with accuracy, wape, bias, totals, deltas."""
    pool, _, cursor = mock_pool
    # First call: current window KPIs (accuracy, wape, bias, total_forecast, total_actual)
    # Second call: prior window KPIs (accuracy, wape, bias)
    cursor.fetchone.side_effect = [
        (85.5, 14.5, 3.2, 50000.0, 48500.0),   # current window
        (82.0, 18.0, 5.1, None, None),            # prior window (only first 3 used)
    ]
    with patch("api.main._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/kpis")
            assert resp.status_code == 200
            data = resp.json()
            # Top-level KPI fields
            assert "accuracy_pct" in data
            assert "wape_pct" in data
            assert "bias_pct" in data
            assert "total_forecast" in data
            assert "total_actual" in data
            assert "window_months" in data
            assert "deltas" in data
            # Values from mock
            assert data["accuracy_pct"] == 85.5
            assert data["wape_pct"] == 14.5
            assert data["bias_pct"] == 3.2
            assert data["total_forecast"] == 50000.0
            assert data["total_actual"] == 48500.0
            assert data["window_months"] == 3  # default
            # Deltas: current - prior
            deltas = data["deltas"]
            assert "accuracy_pct" in deltas
            assert "wape_pct" in deltas
            assert "bias_pct" in deltas
            assert deltas["accuracy_pct"] == 3.5   # 85.5 - 82.0
            assert deltas["wape_pct"] == -3.5       # 14.5 - 18.0
            assert deltas["bias_pct"] == -1.9        # 3.2 - 5.1


@pytest.mark.asyncio
async def test_dashboard_kpis_null_for_empty_data(mock_pool):
    """GET /dashboard/kpis returns null for all KPIs when DB returns no data."""
    pool, _, cursor = mock_pool
    # Both calls return row with all NULLs
    cursor.fetchone.side_effect = [
        (None, None, None, None, None),  # current window
        (None, None, None, None, None),  # prior window
    ]
    with patch("api.main._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/kpis")
            assert resp.status_code == 200
            data = resp.json()
            assert data["accuracy_pct"] is None
            assert data["wape_pct"] is None
            assert data["bias_pct"] is None
            assert data["total_forecast"] is None
            assert data["total_actual"] is None
            # Deltas should also be null
            assert data["deltas"]["accuracy_pct"] is None
            assert data["deltas"]["wape_pct"] is None
            assert data["deltas"]["bias_pct"] is None


@pytest.mark.asyncio
async def test_dashboard_kpis_respects_window_parameter(mock_pool):
    """GET /dashboard/kpis?window=6 passes through the window value."""
    pool, _, cursor = mock_pool
    cursor.fetchone.side_effect = [
        (90.0, 10.0, 1.0, 100000.0, 99000.0),
        (88.0, 12.0, 2.0, None, None),
    ]
    with patch("api.main._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/kpis?window=6")
            assert resp.status_code == 200
            data = resp.json()
            assert data["window_months"] == 6


@pytest.mark.asyncio
async def test_dashboard_kpis_with_filter_params(mock_pool):
    """GET /dashboard/kpis with brand and market filters returns 200."""
    pool, _, cursor = mock_pool
    cursor.fetchone.side_effect = [
        (70.0, 30.0, -5.0, 20000.0, 21000.0),
        (None, None, None, None, None),
    ]
    with patch("api.main._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/kpis?brand=ACME&market=CA")
            assert resp.status_code == 200
            data = resp.json()
            assert data["accuracy_pct"] == 70.0


# ===========================================================================
# /dashboard/alerts
# ===========================================================================

@pytest.mark.asyncio
async def test_dashboard_alerts_returns_sorted_list(mock_pool):
    """GET /dashboard/alerts returns alerts list sorted by severity."""
    pool, _, cursor = mock_pool
    # The endpoint makes up to 3 separate DB calls (low accuracy, bias drift, demand spike).
    # Each gets its own with-block. We simulate:
    #   1st query (low accuracy): 25 low-accuracy DFUs  -> severity "high"
    #   2nd query (bias drift): 2 categories with bias  -> severity "medium"
    #   3rd query (demand spike): fetchone returns (5,)  -> severity "low"
    # Because each query opens a new connection via get_conn(), we use side_effect
    # on pool.connection to return fresh mocked connections for each call.

    def make_conn(fetchall_val, fetchone_val=None):
        conn = MagicMock()
        cur = MagicMock()
        cur.fetchall.return_value = fetchall_val
        cur.fetchone.return_value = fetchone_val
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        return conn

    # 25 rows for low-accuracy DFU count
    low_acc_rows = [(i,) for i in range(25)]
    conn1 = make_conn(fetchall_val=low_acc_rows)
    # 2 rows for bias drift categories
    conn2 = make_conn(fetchall_val=[("CatA", 25.0), ("CatB", -30.0)])
    # demand spike count
    conn3 = make_conn(fetchall_val=[], fetchone_val=(5,))

    pool.connection.side_effect = [conn1, conn2, conn3]

    with patch("api.main._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/alerts")
            assert resp.status_code == 200
            data = resp.json()
            assert "alerts" in data
            alerts = data["alerts"]
            assert len(alerts) == 3
            # Sorted by severity: high < medium < low
            assert alerts[0]["severity"] == "high"
            assert alerts[0]["type"] == "low_accuracy"
            assert alerts[1]["severity"] == "medium"
            assert alerts[1]["type"] == "bias_drift"
            assert alerts[2]["severity"] == "low"
            assert alerts[2]["type"] == "demand_spike"
            # Each alert has required fields
            for alert in alerts:
                assert "id" in alert
                assert "type" in alert
                assert "severity" in alert
                assert "title" in alert
                assert "detail" in alert
                assert "count" in alert


@pytest.mark.asyncio
async def test_dashboard_alerts_empty_when_no_thresholds_breached(mock_pool):
    """GET /dashboard/alerts returns empty list when no thresholds breached."""
    pool, _, cursor = mock_pool

    def make_conn(fetchall_val, fetchone_val=None):
        conn = MagicMock()
        cur = MagicMock()
        cur.fetchall.return_value = fetchall_val
        cur.fetchone.return_value = fetchone_val
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        return conn

    # No low-accuracy DFUs, no bias drift, no demand spikes
    conn1 = make_conn(fetchall_val=[])
    conn2 = make_conn(fetchall_val=[])
    conn3 = make_conn(fetchall_val=[], fetchone_val=(0,))

    pool.connection.side_effect = [conn1, conn2, conn3]

    with patch("api.main._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/alerts")
            assert resp.status_code == 200
            data = resp.json()
            assert data["alerts"] == []


@pytest.mark.asyncio
async def test_dashboard_alerts_respects_limit(mock_pool):
    """GET /dashboard/alerts?limit=1 caps the number of returned alerts."""
    pool, _, cursor = mock_pool

    def make_conn(fetchall_val, fetchone_val=None):
        conn = MagicMock()
        cur = MagicMock()
        cur.fetchall.return_value = fetchall_val
        cur.fetchone.return_value = fetchone_val
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        return conn

    # All three alerts fire
    conn1 = make_conn(fetchall_val=[(i,) for i in range(25)])
    conn2 = make_conn(fetchall_val=[("CatA", 25.0)])
    conn3 = make_conn(fetchall_val=[], fetchone_val=(15,))

    pool.connection.side_effect = [conn1, conn2, conn3]

    with patch("api.main._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/alerts?limit=1")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["alerts"]) == 1


# ===========================================================================
# /dashboard/top-movers
# ===========================================================================

@pytest.mark.asyncio
async def test_dashboard_top_movers_returns_movers_with_direction(mock_pool):
    """GET /dashboard/top-movers returns movers with direction field."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("ITEM-001", "Widget Alpha", 150.0, 100.0),   # up: +50
        ("ITEM-002", "Gadget Beta",   60.0, 120.0),   # down: -60
    ]
    with patch("api.main._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/top-movers")
            assert resp.status_code == 200
            data = resp.json()
            assert "movers" in data
            movers = data["movers"]
            assert len(movers) == 2
            # Each mover has required fields
            for m in movers:
                assert "item_description" in m
                assert "delta" in m
                assert "pct_change" in m
                assert "direction" in m
                assert m["direction"] in ("up", "down")
            # First mover: curr 150, prev 100 -> delta +50, direction up
            assert movers[0]["direction"] == "up"
            assert movers[0]["delta"] == 50.0
            assert movers[0]["pct_change"] == 50.0
            # Second mover: curr 60, prev 120 -> delta -60, direction down
            assert movers[1]["direction"] == "down"
            assert movers[1]["delta"] == -60.0
            assert movers[1]["pct_change"] == -50.0


@pytest.mark.asyncio
async def test_dashboard_top_movers_respects_direction_filter(mock_pool):
    """GET /dashboard/top-movers?direction=up only returns upward movers."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("ITEM-001", "Widget Alpha", 200.0, 100.0),   # up: +100
        ("ITEM-002", "Gadget Beta",   30.0, 120.0),   # down: -90
        ("ITEM-003", "Doohickey",    180.0, 150.0),    # up: +30
    ]
    with patch("api.main._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/top-movers?direction=up")
            assert resp.status_code == 200
            data = resp.json()
            movers = data["movers"]
            # Only upward movers returned
            for m in movers:
                assert m["direction"] == "up"


@pytest.mark.asyncio
async def test_dashboard_top_movers_respects_direction_down(mock_pool):
    """GET /dashboard/top-movers?direction=down only returns downward movers."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("ITEM-001", "Widget Alpha", 200.0, 100.0),   # up
        ("ITEM-002", "Gadget Beta",   30.0, 120.0),   # down
    ]
    with patch("api.main._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/top-movers?direction=down")
            assert resp.status_code == 200
            data = resp.json()
            movers = data["movers"]
            for m in movers:
                assert m["direction"] == "down"


@pytest.mark.asyncio
async def test_dashboard_top_movers_empty(mock_pool):
    """GET /dashboard/top-movers returns empty movers list when no data."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.main._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/top-movers")
            assert resp.status_code == 200
            data = resp.json()
            assert data["movers"] == []


@pytest.mark.asyncio
async def test_dashboard_top_movers_respects_limit(mock_pool):
    """GET /dashboard/top-movers?limit=1 caps the result list."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("ITEM-001", "Widget Alpha", 200.0, 100.0),
        ("ITEM-002", "Gadget Beta",  150.0, 100.0),
        ("ITEM-003", "Doohickey",    130.0, 100.0),
    ]
    with patch("api.main._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/top-movers?limit=1")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["movers"]) <= 1


# ===========================================================================
# /dashboard/heatmap
# ===========================================================================

@pytest.mark.asyncio
async def test_dashboard_heatmap_returns_rows_with_period_labels(mock_pool):
    """GET /dashboard/heatmap returns rows with period labels and accuracy values."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("Electronics", "Jan 25", 85.3),
        ("Electronics", "Feb 25", 88.1),
        ("Furniture",   "Jan 25", 72.0),
        ("Furniture",   "Feb 25", 75.5),
    ]
    with patch("api.main._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/heatmap")
            assert resp.status_code == 200
            data = resp.json()
            assert "rows" in data
            assert "period_labels" in data
            assert "metric" in data
            assert data["metric"] == "accuracy_pct"
            # Period labels from mock data
            assert data["period_labels"] == ["Jan 25", "Feb 25"]
            # Two group rows
            assert len(data["rows"]) == 2
            for row in data["rows"]:
                assert "label" in row
                assert "values" in row
                assert isinstance(row["values"], list)
                assert len(row["values"]) == 2


@pytest.mark.asyncio
async def test_dashboard_heatmap_empty_data(mock_pool):
    """GET /dashboard/heatmap returns empty rows when no data."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.main._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/heatmap")
            assert resp.status_code == 200
            data = resp.json()
            assert data["rows"] == []
            assert data["period_labels"] == []


@pytest.mark.asyncio
async def test_dashboard_heatmap_respects_grain_parameter(mock_pool):
    """GET /dashboard/heatmap?grain=brand groups by brand instead of category."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("BrandX", "Mar 25", 90.0),
    ]
    with patch("api.main._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/heatmap?grain=brand")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["rows"]) == 1
            assert data["rows"][0]["label"] == "BrandX"


@pytest.mark.asyncio
async def test_dashboard_heatmap_null_accuracy_replaced_with_zero(mock_pool):
    """Null accuracy values in heatmap should be replaced with 0."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("Electronics", "Jan 25", None),
    ]
    with patch("api.main._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/heatmap")
            assert resp.status_code == 200
            data = resp.json()
            assert data["rows"][0]["values"] == [0]


@pytest.mark.asyncio
async def test_dashboard_heatmap_respects_periods_parameter(mock_pool):
    """GET /dashboard/heatmap?periods=6 returns 200 with custom window."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.main._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/heatmap?periods=6")
            assert resp.status_code == 200
            data = resp.json()
            assert data["rows"] == []
