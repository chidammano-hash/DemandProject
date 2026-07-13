"""Tests for forecast accuracy endpoints."""

import pytest
from unittest.mock import patch, MagicMock
import httpx
from httpx import ASGITransport


@pytest.fixture
def mock_pool():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = (0,)
    mock_cursor.description = [("model_id",)]
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.connection.return_value = mock_conn

    return pool, mock_conn, mock_cursor


@pytest.mark.asyncio
async def test_forecast_models_endpoint(mock_pool):
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("external",), ("lgbm_cluster",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/domains/forecast/models")
            assert response.status_code == 200
            data = response.json()
            assert "models" in data


@pytest.mark.asyncio
async def test_accuracy_slice_endpoint(mock_pool):
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/accuracy/slice?group_by=cluster_assignment")
            assert response.status_code == 200
            data = response.json()
            assert "group_by" in data
            assert "rows" in data


@pytest.mark.asyncio
async def test_accuracy_slice_with_brand_filter(mock_pool):
    """brand param triggers raw-table path when no common_dfus."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/forecast/accuracy/slice?group_by=cluster_assignment&brand=BrandA"
            )
            assert response.status_code == 200
            data = response.json()
            assert "rows" in data
            assert data.get("source") == "fact_external_forecast_monthly"


@pytest.mark.asyncio
async def test_accuracy_slice_with_category_and_market_filter(mock_pool):
    """category and market params trigger raw-table path."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/forecast/accuracy/slice?group_by=model_id&category=CAT1,CAT2&market=NY"
            )
            assert response.status_code == 200
            data = response.json()
            assert "rows" in data


@pytest.mark.asyncio
async def test_lag_curve_endpoint(mock_pool):
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/accuracy/lag-curve")
            assert response.status_code == 200
            data = response.json()
            assert "by_lag" in data


@pytest.mark.asyncio
async def test_lag_curve_with_brand_filter(mock_pool):
    """brand param triggers raw backtest_lag_archive path."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/accuracy/lag-curve?brand=BrandA,BrandB")
            assert response.status_code == 200
            data = response.json()
            assert "by_lag" in data
            assert data.get("source") == "backtest_lag_archive"


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accuracy_slice_invalid_group_by(mock_pool):
    """Invalid group_by parameter should return 422."""
    pool, _, cursor = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/accuracy/slice?group_by=invalid_dim")
            assert response.status_code == 422
            data = response.json()
            assert "Invalid group_by" in data["detail"]


@pytest.mark.asyncio
async def test_accuracy_slice_too_many_models(mock_pool):
    """More than 20 models should return 422."""
    pool, _, cursor = mock_pool
    models = ",".join([f"model_{i}" for i in range(21)])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/forecast/accuracy/slice?models={models}")
            assert response.status_code == 422
            assert "max 20" in response.json()["detail"]


@pytest.mark.asyncio
async def test_lag_curve_too_many_models(mock_pool):
    """More than 20 models in lag-curve should return 422."""
    pool, _, cursor = mock_pool
    models = ",".join([f"model_{i}" for i in range(21)])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/forecast/accuracy/lag-curve?models={models}")
            assert response.status_code == 422


# ---------------------------------------------------------------------------
# Common DFUs path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accuracy_slice_common_dfus_path(mock_pool):
    """common_dfus=true with 2+ models triggers CTE intersection path."""
    pool, _, cursor = mock_pool
    # Three DB calls: main query, count query, per-model DFU counts
    cursor.fetchall.side_effect = [
        # main query: (bucket, model_id, dfu_count, sum_forecast, sum_actual,
        #              sum_abs_error, total_buckets)
        [("ClusterA", "lgbm_cluster", 10, 1000.0, 900.0, 100.0, 1),
         ("ClusterA", "mstl", 10, 1050.0, 900.0, 150.0, 1)],
        # per-model DFU counts
        [("lgbm_cluster", 15), ("mstl", 12)],
    ]
    cursor.fetchone.return_value = (8,)  # common DFU count
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/forecast/accuracy/slice"
                "?group_by=cluster_assignment"
                "&models=lgbm_cluster,mstl"
                "&common_dfus=true"
            )
    assert response.status_code == 200
    data = response.json()
    assert data["common_dfus"] is True
    assert data["common_dfu_count"] == 8
    assert "dfu_counts" in data
    assert len(data["rows"]) == 1
    assert "ClusterA" == data["rows"][0]["bucket"]
    assert "lgbm_cluster" in data["rows"][0]["by_model"]
    assert "mstl" in data["rows"][0]["by_model"]


@pytest.mark.asyncio
async def test_accuracy_slice_common_dfus_needs_two_models(mock_pool):
    """common_dfus with only 1 model should use standard path, not CTE."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/forecast/accuracy/slice"
                "?group_by=cluster_assignment"
                "&models=lgbm"
                "&common_dfus=true"
            )
    assert response.status_code == 200
    data = response.json()
    # With only 1 model, common_dfus is not activated — uses standard path
    assert data.get("source") == "agg_accuracy_by_dim"


# ---------------------------------------------------------------------------
# Slice with data pivot verification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accuracy_slice_returns_pivoted_data(mock_pool):
    """Verify the standard path correctly pivots by_model within each bucket."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        # (bucket, model_id, n_rows, sum_forecast, sum_actual, sum_abs_error, total_buckets)
        ("ClusterX", "external", 100, 5000.0, 4800.0, 200.0, 2),
        ("ClusterX", "lgbm_cluster", 100, 5100.0, 4800.0, 300.0, 2),
        ("ClusterY", "external", 50, 2000.0, 2100.0, 100.0, 2),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/forecast/accuracy/slice?group_by=cluster_assignment&models=external,lgbm"
            )
    assert response.status_code == 200
    data = response.json()
    rows = data["rows"]
    assert len(rows) == 2
    # Rows should be sorted by bucket
    assert rows[0]["bucket"] == "ClusterX"
    assert rows[1]["bucket"] == "ClusterY"
    # ClusterX has 2 models
    assert "external" in rows[0]["by_model"]
    assert "lgbm_cluster" in rows[0]["by_model"]
    # ClusterY has 1 model
    assert "external" in rows[1]["by_model"]


# ---------------------------------------------------------------------------
# Lag curve with data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lag_curve_returns_by_lag_structure(mock_pool):
    """Verify lag-curve response structures by_lag correctly."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        # (model_id, lag, n_rows, sum_forecast, sum_actual, sum_abs_error)
        ("external", 0, 100, 5000.0, 4800.0, 200.0),
        ("external", 1, 90, 4500.0, 4300.0, 200.0),
        ("lgbm_cluster", 0, 100, 5100.0, 4800.0, 300.0),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/forecast/accuracy/lag-curve?models=external,lgbm"
            )
    assert response.status_code == 200
    data = response.json()
    by_lag = data["by_lag"]
    assert len(by_lag) == 2  # lag 0 and lag 1
    assert by_lag[0]["lag"] == 0
    assert "external" in by_lag[0]["by_model"]
    assert "lgbm_cluster" in by_lag[0]["by_model"]
    assert by_lag[1]["lag"] == 1
    assert "external" in by_lag[1]["by_model"]


@pytest.mark.asyncio
async def test_lag_curve_common_dfus(mock_pool):
    """common_dfus in lag-curve triggers archive CTE path."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("lgbm_cluster", 0, 10, 1000.0, 900.0, 100.0),
        ("mstl", 0, 10, 1050.0, 900.0, 150.0),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/forecast/accuracy/lag-curve"
                "?models=lgbm_cluster,mstl"
                "&common_dfus=true"
            )
    assert response.status_code == 200
    data = response.json()
    assert data["common_dfus"] is True
    assert data["source"] == "backtest_lag_archive"


# ---------------------------------------------------------------------------
# Filter combination tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accuracy_slice_all_dim_filters(mock_pool):
    """All 5 dimension filters should be accepted."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/forecast/accuracy/slice"
                "?group_by=model_id"
                "&cluster_assignment=high_volume"
                "&supplier_desc=ACME"
                "&abc_vol=A"
                "&region=East"
                "&seasonality_profile=seasonal"
                "&lag=0"
                "&month_from=2024-01-01"
                "&month_to=2025-12-01"
            )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_accuracy_slice_with_execution_lag(mock_pool):
    """lag=-1 (default) triggers the execution lag filter."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/forecast/accuracy/slice?group_by=model_id&lag=-1"
            )
    assert response.status_code == 200
    data = response.json()
    assert data["lag_filter"] == -1


@pytest.mark.asyncio
async def test_accuracy_slice_include_dfu_count(mock_pool):
    """include_dfu_count=true triggers additional DFU coverage query."""
    pool, _, cursor = mock_pool
    cursor.fetchall.side_effect = [
        # main query: (bucket, model_id, n_rows, sum_forecast, sum_actual,
        #              sum_abs_error, total_buckets)
        [("ClusterA", "external", 100, 5000.0, 4800.0, 200.0, 1)],
        # DFU count query
        [("ClusterA", "external", 42)],
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/forecast/accuracy/slice"
                "?group_by=cluster_assignment"
                "&include_dfu_count=true"
            )
    assert response.status_code == 200
    data = response.json()
    assert len(data["rows"]) == 1
    # DFU count should be passed through to compute_kpis
    model_kpis = data["rows"][0]["by_model"]["external"]
    assert "dfu_count" in model_kpis
