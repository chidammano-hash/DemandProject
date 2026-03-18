"""Tests for dashboard endpoints — /dashboard/kpis, /dashboard/alerts,
/dashboard/top-movers, /dashboard/heatmap, /dashboard/trend, /dashboard/planning-date."""

import pytest
from datetime import date
from unittest.mock import patch, MagicMock
import httpx
from httpx import ASGITransport

from common.planning_date import _reset_cache


# ===========================================================================
# /dashboard/planning-date
# ===========================================================================

@pytest.mark.asyncio
async def test_planning_date_frozen(mock_pool):
    """GET /dashboard/planning-date returns frozen planning date info."""
    _reset_cache()
    pool, _, _cursor = mock_pool
    frozen = date(2026, 2, 24)
    with patch("api.core._get_pool", return_value=pool), \
         patch("common.planning_date._resolve_date", return_value=frozen):
        _reset_cache()
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/planning-date")
    _reset_cache()
    assert resp.status_code == 200
    data = resp.json()
    assert data["planning_date"] == "2026-02-24"
    assert data["system_date"] == date.today().isoformat()
    assert data["is_frozen"] is True
    assert data["days_behind"] >= 0


@pytest.mark.asyncio
async def test_planning_date_live(mock_pool):
    """GET /dashboard/planning-date reports not frozen when using system date."""
    _reset_cache()
    pool, _, _cursor = mock_pool
    today = date.today()
    with patch("api.core._get_pool", return_value=pool), \
         patch("common.planning_date._resolve_date", return_value=today):
        _reset_cache()
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/planning-date")
    _reset_cache()
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_frozen"] is False
    assert data["days_behind"] == 0


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
    with patch("api.core._get_pool", return_value=pool):
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
    with patch("api.core._get_pool", return_value=pool):
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
    with patch("api.core._get_pool", return_value=pool):
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
    with patch("api.core._get_pool", return_value=pool):
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

    with patch("api.core._get_pool", return_value=pool):
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

    with patch("api.core._get_pool", return_value=pool):
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

    with patch("api.core._get_pool", return_value=pool):
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
    with patch("api.core._get_pool", return_value=pool):
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
    with patch("api.core._get_pool", return_value=pool):
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
    with patch("api.core._get_pool", return_value=pool):
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
    with patch("api.core._get_pool", return_value=pool):
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
    with patch("api.core._get_pool", return_value=pool):
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
        ("Electronics", "Jan 25", 85.3, 15),
        ("Electronics", "Feb 25", 88.1, 12),
        ("Furniture",   "Jan 25", 72.0, 8),
        ("Furniture",   "Feb 25", 75.5, 10),
    ]
    with patch("api.core._get_pool", return_value=pool):
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
                assert "counts" in row
                assert isinstance(row["values"], list)
                assert isinstance(row["counts"], list)
                assert len(row["values"]) == 2
                assert len(row["counts"]) == 2
            # Verify counts from mock
            assert data["rows"][0]["counts"] == [15, 12]
            assert data["rows"][1]["counts"] == [8, 10]


@pytest.mark.asyncio
async def test_dashboard_heatmap_empty_data(mock_pool):
    """GET /dashboard/heatmap returns empty rows when no data."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
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
        ("BrandX", "Mar 25", 90.0, 20),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/heatmap?grain=brand")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["rows"]) == 1
            assert data["rows"][0]["label"] == "BrandX"
            assert data["rows"][0]["counts"] == [20]


@pytest.mark.asyncio
async def test_dashboard_heatmap_null_accuracy_replaced_with_zero(mock_pool):
    """Null accuracy values in heatmap should be replaced with 0."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("Electronics", "Jan 25", None, 5),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/heatmap")
            assert resp.status_code == 200
            data = resp.json()
            assert data["rows"][0]["values"] == [0]
            assert data["rows"][0]["counts"] == [5]


@pytest.mark.asyncio
async def test_dashboard_heatmap_respects_periods_parameter(mock_pool):
    """GET /dashboard/heatmap?periods=6 returns 200 with custom window."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/heatmap?periods=6")
            assert resp.status_code == 200
            data = resp.json()
            assert data["rows"] == []


@pytest.mark.asyncio
async def test_dashboard_heatmap_applies_brand_category_market_filters(mock_pool):
    """GET /dashboard/heatmap with brand/category/market filters includes them in SQL."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("BrandX", "Jan 25", 92.0, 7),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/dashboard/heatmap?grain=brand&brand=BrandX&category=Electronics&market=CA"
            )
    assert resp.status_code == 200
    # Verify that the SQL included filter params — brand, category, and market
    sql_arg = cursor.execute.call_args[0][0]
    assert "i.brand_name = ANY" in sql_arg
    assert "i.class = ANY" in sql_arg
    assert "lo.state_id = ANY" in sql_arg
    assert "dim_location" in sql_arg
    # Verify filter values were passed in params
    params = cursor.execute.call_args[0][1]
    assert ["BrandX"] in params
    assert ["Electronics"] in params
    assert ["CA"] in params


@pytest.mark.asyncio
async def test_dashboard_heatmap_uses_model_id_not_lag(mock_pool):
    """GET /dashboard/heatmap filters by model_id='external', not lag=0."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/heatmap")
    assert resp.status_code == 200
    sql_arg = cursor.execute.call_args[0][0]
    assert "model_id = 'external'" in sql_arg
    assert "lag = 0" not in sql_arg


@pytest.mark.asyncio
async def test_dashboard_kpis_uses_model_id_not_lag(mock_pool):
    """GET /dashboard/kpis filters by model_id='external', not lag=0."""
    pool, _, cursor = mock_pool
    cursor.fetchone.side_effect = [
        (85.0, 15.0, 2.0, 50000.0, 48000.0),
        (80.0, 20.0, 4.0, None, None),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/kpis")
    assert resp.status_code == 200
    sql_current = cursor.execute.call_args_list[0][0][0]
    assert "model_id = 'external'" in sql_current
    assert "lag = 0" not in sql_current


@pytest.mark.asyncio
async def test_dashboard_trend_uses_model_id_not_lag(mock_pool):
    """GET /dashboard/trend filters by model_id param (default 'champion'), not lag=0."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/trend")
    assert resp.status_code == 200
    sql_arg = cursor.execute.call_args[0][0]
    assert "model_id = %s" in sql_arg
    assert "lag = 0" not in sql_arg
    # Default model param is 'champion'
    params = cursor.execute.call_args[0][1]
    assert params[0] == "champion"


@pytest.mark.asyncio
async def test_dashboard_trend_uses_planning_date(mock_pool):
    """GET /dashboard/trend uses planning date, not CURRENT_DATE."""
    _reset_cache()
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    frozen = date(2026, 2, 24)
    with patch("api.core._get_pool", return_value=pool), \
         patch("common.planning_date._resolve_date", return_value=frozen):
        _reset_cache()
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/trend")
    _reset_cache()
    assert resp.status_code == 200
    sql_arg = cursor.execute.call_args[0][0]
    assert "2026-02-24" in sql_arg
    assert "CURRENT_DATE" not in sql_arg


@pytest.mark.asyncio
async def test_dashboard_kpis_uses_planning_date(mock_pool):
    """GET /dashboard/kpis uses planning date, not CURRENT_DATE."""
    _reset_cache()
    pool, _, cursor = mock_pool
    cursor.fetchone.side_effect = [
        (85.0, 15.0, 2.0, 50000.0, 48000.0),
        (80.0, 20.0, 4.0, None, None),
    ]
    frozen = date(2026, 2, 24)
    with patch("api.core._get_pool", return_value=pool), \
         patch("common.planning_date._resolve_date", return_value=frozen):
        _reset_cache()
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/kpis")
    _reset_cache()
    assert resp.status_code == 200
    sql_arg = cursor.execute.call_args_list[0][0][0]
    assert "2026-02-24" in sql_arg
    assert "CURRENT_DATE" not in sql_arg


# ===========================================================================
# /dashboard/trend
# ===========================================================================

@pytest.mark.asyncio
async def test_dashboard_trend_returns_monthly_data(mock_pool):
    """GET /dashboard/trend returns monthly forecast vs actual totals."""
    pool, _, cursor = mock_pool
    from decimal import Decimal
    cursor.fetchall.return_value = [
        ("2025-10", Decimal("120000.5"), Decimal("115000.3")),
        ("2025-11", Decimal("125000.0"), Decimal("120000.0")),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/trend")
    assert resp.status_code == 200
    data = resp.json()
    assert "months" in data
    months = data["months"]
    assert len(months) == 2
    assert months[0]["month"] == "2025-10"
    assert months[0]["forecast"] == 120000.0  # rounded to 0 decimals
    assert months[0]["actual"] == 115000.0
    assert months[1]["month"] == "2025-11"


@pytest.mark.asyncio
async def test_dashboard_trend_empty_data(mock_pool):
    """GET /dashboard/trend returns empty months when no data."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/trend")
    assert resp.status_code == 200
    data = resp.json()
    assert data["months"] == []


@pytest.mark.asyncio
async def test_dashboard_trend_respects_window_parameter(mock_pool):
    """GET /dashboard/trend?window=6 returns 200."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/trend?window=6")
    assert resp.status_code == 200
    assert resp.json()["months"] == []


@pytest.mark.asyncio
async def test_dashboard_trend_with_filters(mock_pool):
    """GET /dashboard/trend with brand filter returns 200."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("2025-12", 50000.0, 48000.0),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/trend?brand=ACME&window=12")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["months"]) == 1
    assert data["months"][0]["forecast"] == 50000.0


@pytest.mark.asyncio
async def test_dashboard_trend_null_values(mock_pool):
    """GET /dashboard/trend handles null forecast/actual gracefully."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("2025-10", None, None),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/trend")
    assert resp.status_code == 200
    data = resp.json()
    assert data["months"][0]["forecast"] == 0
    assert data["months"][0]["actual"] == 0


# ===========================================================================
# /dashboard/customer-map
# ===========================================================================

@pytest.mark.asyncio
async def test_customer_map_by_state(mock_pool):
    """GET /dashboard/customer-map returns state-level locations with coordinates."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("CA", 500),
        ("TX", 350),
        ("NY", 200),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/customer-map?group_by=state")
    assert resp.status_code == 200
    data = resp.json()
    assert data["group_by"] == "state"
    assert data["total"] == 1050
    assert len(data["locations"]) == 3
    ca = data["locations"][0]
    assert ca["label"] == "CA"
    assert ca["customer_count"] == 500
    assert "lat" in ca
    assert "lon" in ca


def _mock_pgeocode(lats, lons):
    """Create a mock pgeocode.Nominatim that returns given lat/lon arrays."""
    import pandas as pd
    import numpy as np
    mock_nomi = MagicMock()
    df = pd.DataFrame({"latitude": lats, "longitude": lons})
    mock_nomi.query_postal_code.return_value = df
    return mock_nomi


@pytest.mark.asyncio
async def test_customer_map_by_zip(mock_pool):
    """GET /dashboard/customer-map?group_by=zip returns zip-level locations with pgeocode coords."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("90210", "CA", 50),
        ("10001", "NY", 30),
    ]
    mock_nomi = _mock_pgeocode([34.0901, 40.7484], [-118.4065, -73.9967])
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.dashboard._get_nomi", return_value=mock_nomi):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/customer-map?group_by=zip")
    assert resp.status_code == 200
    data = resp.json()
    assert data["group_by"] == "zip"
    assert data["total"] == 80
    bh = data["locations"][0]
    assert bh["label"] == "90210"
    assert bh["state"] == "CA"
    assert abs(bh["lat"] - 34.09) < 0.01
    assert abs(bh["lon"] - (-118.41)) < 0.01


@pytest.mark.asyncio
async def test_customer_map_by_city(mock_pool):
    """GET /dashboard/customer-map?group_by=city returns city groupings with pgeocode coords."""
    pool, _, cursor = mock_pool
    # City query returns 4 columns: city, state, count, common_zip
    cursor.fetchall.return_value = [
        ("Los Angeles", "CA", 120, "90001"),
        ("Houston", "TX", 80, "77001"),
    ]
    mock_nomi = _mock_pgeocode([33.9425, 29.7633], [-118.2551, -95.3633])
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.dashboard._get_nomi", return_value=mock_nomi):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/customer-map?group_by=city")
    assert resp.status_code == 200
    data = resp.json()
    assert data["group_by"] == "city"
    assert len(data["locations"]) == 2
    la = data["locations"][0]
    assert la["label"] == "Los Angeles"
    assert la["state"] == "CA"
    assert abs(la["lat"] - 33.94) < 0.1
    assert abs(la["lon"] - (-118.26)) < 0.1


@pytest.mark.asyncio
async def test_customer_map_empty(mock_pool):
    """GET /dashboard/customer-map returns empty list when no customers."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/customer-map")
    assert resp.status_code == 200
    data = resp.json()
    assert data["locations"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_customer_map_unknown_state(mock_pool):
    """States not in centroid lookup get no lat/lon."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("XX", 10),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/customer-map?group_by=state")
    assert resp.status_code == 200
    loc = resp.json()["locations"][0]
    assert loc["label"] == "XX"
    assert "lat" not in loc


@pytest.mark.asyncio
async def test_customer_map_zip_fallback_to_state_centroid(mock_pool):
    """Zip codes not found by pgeocode fall back to state centroid."""
    import numpy as np
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("00000", "CA", 5),
    ]
    mock_nomi = _mock_pgeocode([np.nan], [np.nan])
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.dashboard._get_nomi", return_value=mock_nomi):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dashboard/customer-map?group_by=zip")
    assert resp.status_code == 200
    loc = resp.json()["locations"][0]
    # Falls back to CA state centroid
    assert abs(loc["lat"] - 36.12) < 0.1
    assert "lon" in loc
