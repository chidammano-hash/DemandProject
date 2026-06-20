"""Tests for accuracy router — item/location filter support."""

import pytest
from unittest.mock import patch, MagicMock

import httpx
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool


def _make_slice_row(bucket="cluster_a", model_id="external", dfu_count=10,
                    sum_fcst=1000.0, sum_actual=900.0, sum_abs=100.0,
                    total_buckets=1):
    # 7th column = total distinct bucket count (COUNT(*) OVER ()), carried out of
    # the top-buckets CTE so the handler can report the `truncated` flag.
    return (bucket, model_id, dfu_count, sum_fcst, sum_actual, sum_abs, total_buckets)


def _make_lag_row(model_id="external", lag=0, dfu_count=10,
                  sum_fcst=1000.0, sum_actual=900.0, sum_abs=100.0):
    return (model_id, lag, dfu_count, sum_fcst, sum_actual, sum_abs)


# ---------------------------------------------------------------------------
# Slice endpoint — item/location filters
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_slice_item_filter_triggers_raw_path():
    """When item= is set, the endpoint should use the raw fact table path (not agg view)."""
    pool, conn, cursor = _make_pool(fetchall_return=[
        _make_slice_row("cluster_a", "external"),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/accuracy/slice", params={
                "group_by": "cluster_assignment",
                "lag": 0,
                "item": "100320",
            })
    assert resp.status_code == 200
    data = resp.json()
    # Should use raw fact table path because item filter is set
    assert data["source"] == "fact_external_forecast_monthly"
    assert len(data["rows"]) == 1


@pytest.mark.asyncio
async def test_slice_location_filter_triggers_raw_path():
    """When location= is set, the endpoint should use the raw fact table path."""
    pool, conn, cursor = _make_pool(fetchall_return=[
        _make_slice_row("cluster_b", "lgbm_cluster"),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/accuracy/slice", params={
                "group_by": "cluster_assignment",
                "lag": 0,
                "location": "1401-BULK",
            })
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "fact_external_forecast_monthly"


@pytest.mark.asyncio
async def test_slice_item_and_location_combined():
    """Both item and location can be set together."""
    pool, conn, cursor = _make_pool(fetchall_return=[
        _make_slice_row("cluster_c", "external"),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/accuracy/slice", params={
                "group_by": "cluster_assignment",
                "lag": 0,
                "item": "100320,100321",
                "location": "1401-BULK,1402-BULK",
            })
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "fact_external_forecast_monthly"

    # Verify the SQL contained IN clauses for item and location
    sql_call = cursor.execute.call_args_list[0]
    sql_text = sql_call[0][0]
    sql_params = sql_call[0][1]
    assert "f.item_id IN" in sql_text
    assert "f.loc IN" in sql_text
    assert "100320" in sql_params
    assert "100321" in sql_params
    assert "1401-BULK" in sql_params
    assert "1402-BULK" in sql_params


@pytest.mark.asyncio
async def test_slice_no_item_location_uses_agg_view():
    """Without item/location/brand/category/market filters, use pre-aggregated view."""
    pool, conn, cursor = _make_pool(fetchall_return=[
        ("cluster_a", "external", 50, 5000.0, 4500.0, 500.0, 1),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/accuracy/slice", params={
                "group_by": "cluster_assignment",
                "lag": 0,
            })
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "agg_accuracy_by_dim"


# ---------------------------------------------------------------------------
# Slice endpoint — bucket cap (limit / truncated)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_slice_accepts_limit_and_pushes_into_sql():
    """A `limit` param is accepted and threaded into the bucket-cap SQL as the LIMIT arg."""
    pool, _conn, cursor = _make_pool(fetchall_return=[
        ("cluster_a", "external", 50, 5000.0, 4500.0, 500.0, 1),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/accuracy/slice", params={
                "group_by": "cluster_assignment",
                "lag": 0,
                "limit": 25,
            })
    assert resp.status_code == 200
    data = resp.json()
    assert data["limit"] == 25
    # The LIMIT is parameterised (%s) and `limit` is the trailing bound param.
    sql_call = cursor.execute.call_args_list[0]
    sql_text, sql_params = sql_call[0][0], sql_call[0][1]
    assert "LIMIT %s" in sql_text
    assert sql_params[-1] == 25


@pytest.mark.asyncio
async def test_slice_reports_truncated_when_cap_hit():
    """`truncated` is True when total distinct buckets exceeds the requested limit."""
    # total_buckets (7th col) = 40 while limit = 5 → handler must flag truncation.
    pool, _conn, _cursor = _make_pool(fetchall_return=[
        ("cluster_a", "external", 50, 5000.0, 4500.0, 500.0, 40),
        ("cluster_b", "external", 30, 3000.0, 2800.0, 200.0, 40),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/accuracy/slice", params={
                "group_by": "cluster_assignment",
                "lag": 0,
                "limit": 5,
            })
    assert resp.status_code == 200
    data = resp.json()
    assert data["limit"] == 5
    assert data["truncated"] is True
    assert len(data["rows"]) == 2  # the kept buckets


@pytest.mark.asyncio
async def test_slice_not_truncated_when_under_cap():
    """`truncated` is False when total distinct buckets is within the limit."""
    pool, _conn, _cursor = _make_pool(fetchall_return=[
        ("cluster_a", "external", 50, 5000.0, 4500.0, 500.0, 2),
        ("cluster_b", "external", 30, 3000.0, 2800.0, 200.0, 2),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/accuracy/slice", params={
                "group_by": "cluster_assignment",
                "lag": 0,
                "limit": 1000,
            })
    assert resp.status_code == 200
    data = resp.json()
    assert data["truncated"] is False


@pytest.mark.asyncio
async def test_slice_rejects_out_of_range_limit():
    """`limit` is bounded 1..5000; out-of-range values are 422-rejected by FastAPI."""
    pool, _conn, _cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            too_big = await client.get("/forecast/accuracy/slice", params={"limit": 99999})
            too_small = await client.get("/forecast/accuracy/slice", params={"limit": 0})
    assert too_big.status_code == 422
    assert too_small.status_code == 422


# ---------------------------------------------------------------------------
# Lag-curve endpoint — item/location filters
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lag_curve_item_filter_triggers_raw_path():
    """When item= is set on lag-curve, should use raw backtest_lag_archive path."""
    pool, conn, cursor = _make_pool(fetchall_return=[
        _make_lag_row("external", 0),
        _make_lag_row("external", 1),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/accuracy/lag-curve", params={
                "item": "100320",
            })
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "backtest_lag_archive"
    assert len(data["by_lag"]) == 2


@pytest.mark.asyncio
async def test_lag_curve_location_filter_triggers_raw_path():
    """When location= is set on lag-curve, should use raw backtest_lag_archive path."""
    pool, conn, cursor = _make_pool(fetchall_return=[
        _make_lag_row("lgbm_cluster", 0),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/accuracy/lag-curve", params={
                "location": "1401-BULK",
            })
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "backtest_lag_archive"


@pytest.mark.asyncio
async def test_lag_curve_item_location_sql_contains_filters():
    """Verify the SQL includes IN clauses for item and location."""
    pool, conn, cursor = _make_pool(fetchall_return=[
        _make_lag_row("external", 0),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/accuracy/lag-curve", params={
                "item": "100320",
                "location": "1401-BULK",
            })
    assert resp.status_code == 200

    sql_call = cursor.execute.call_args_list[0]
    sql_text = sql_call[0][0]
    sql_params = sql_call[0][1]
    assert "a.item_id IN" in sql_text
    assert "a.loc IN" in sql_text
    assert "100320" in sql_params
    assert "1401-BULK" in sql_params


@pytest.mark.asyncio
async def test_lag_curve_no_filters_uses_agg_view():
    """Without item/location/brand/category/market, use pre-aggregated lag archive view."""
    pool, conn, cursor = _make_pool(fetchall_return=[
        ("external", 0, 100, 10000.0, 9000.0, 1000.0),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/accuracy/lag-curve")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "agg_accuracy_lag_archive"


# ---------------------------------------------------------------------------
# _add_item_location_filters unit tests
# ---------------------------------------------------------------------------

def test_add_item_location_filters_single_item():
    from api.routers.forecasting.accuracy import _add_item_location_filters
    where, params = [], []
    _add_item_location_filters(where, params, item_id_col="f.item_id", loc_col="f.loc",
                               item="100320", location=None)
    assert len(where) == 1
    assert "f.item_id IN" in where[0]
    assert params == ["100320"]


def test_add_item_location_filters_multi_items():
    from api.routers.forecasting.accuracy import _add_item_location_filters
    where, params = [], []
    _add_item_location_filters(where, params, item_id_col="f.item_id", loc_col="f.loc",
                               item="100320,100321", location="LOC1,LOC2")
    assert len(where) == 2
    assert params == ["100320", "100321", "LOC1", "LOC2"]


def test_add_item_location_filters_empty_strings():
    from api.routers.forecasting.accuracy import _add_item_location_filters
    where, params = [], []
    _add_item_location_filters(where, params, item_id_col="f.item_id", loc_col="f.loc",
                               item="", location="")
    assert len(where) == 0
    assert params == []


def test_add_item_location_filters_none():
    from api.routers.forecasting.accuracy import _add_item_location_filters
    where, params = [], []
    _add_item_location_filters(where, params, item_id_col="f.item_id", loc_col="f.loc",
                               item=None, location=None)
    assert len(where) == 0
    assert params == []


@pytest.mark.asyncio
async def test_lag_leaderboard_returns_ranked_models():
    pool, conn, cursor = _make_pool(fetchall_return=[
        (0, "lgbm_cluster", 100, 1000.0, 900.0, 100.0),
        (0, "external", 80, 1000.0, 900.0, 200.0),
        (1, "lgbm_cluster", 90, 800.0, 700.0, 120.0),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/accuracy/lag-leaderboard", params={"limit": 5})

    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "agg_accuracy_lag_archive"
    lag0 = next(lg for lg in data["lags"] if lg["lag"] == 0)
    assert lag0["rankings"][0]["model_id"] == "lgbm_cluster"
    assert lag0["rankings"][0]["rank"] == 1
    assert lag0["rankings"][0]["accuracy_pct"] is not None
