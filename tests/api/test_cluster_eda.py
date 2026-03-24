"""Tests for cluster EDA endpoints — profile, error concentration,
demand distribution, residual analysis, and seasonality heatmap."""

import pytest
from unittest.mock import patch, MagicMock

import httpx
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# GET /cluster-eda/profile
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cluster_profile_returns_200():
    """GET /cluster-eda/profile returns cluster profile list."""
    pool, conn, cursor = _make_pool()
    # Two calls via _safe_query: profile_rows, accuracy_rows
    cursor.fetchall.side_effect = [
        [
            (0, 1000, 250.5, 0.85, 0.12, 230.0, 195.0),
            (1, 800, 50.3, 1.45, 0.35, 45.0, 65.0),
        ],
        [
            (0, 72.5, 0.275),
            (1, 55.3, 0.447),
        ],
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-eda/profile")
    assert resp.status_code == 200
    data = resp.json()
    assert "clusters" in data
    assert len(data["clusters"]) == 2
    cl0 = data["clusters"][0]
    assert cl0["ml_cluster"] == 0
    assert cl0["n_dfus"] == 1000
    assert cl0["mean_demand"] == 250.5
    assert cl0["cv_demand"] == 0.85
    assert cl0["zero_pct"] == 0.12
    assert cl0["accuracy_pct"] == 72.5
    assert cl0["wape"] == 0.275
    cl1 = data["clusters"][1]
    assert cl1["ml_cluster"] == 1
    assert cl1["accuracy_pct"] == 55.3


@pytest.mark.asyncio
async def test_cluster_profile_empty():
    """GET /cluster-eda/profile returns warning when no clusters exist."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.side_effect = [[], []]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-eda/profile")
    assert resp.status_code == 200
    data = resp.json()
    assert data["clusters"] == []
    assert data["warning"] == "No data available"


@pytest.mark.asyncio
async def test_cluster_profile_accuracy_missing_for_cluster():
    """Profile returns None for accuracy when accuracy data missing for a cluster."""
    pool, conn, cursor = _make_pool()
    # Profile has 2 clusters but accuracy only has 1
    cursor.fetchall.side_effect = [
        [
            (0, 500, 100.0, 0.5, 0.1, 90.0, 45.0),
            (1, 300, 30.0, 1.2, 0.4, 28.0, 33.0),
        ],
        [
            (0, 70.0, 0.30),
        ],
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-eda/profile")
    assert resp.status_code == 200
    data = resp.json()
    assert data["clusters"][0]["accuracy_pct"] == 70.0
    assert data["clusters"][1]["accuracy_pct"] is None


# ---------------------------------------------------------------------------
# GET /cluster-eda/error-concentration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_concentration_returns_200():
    """GET /cluster-eda/error-concentration returns all four sections."""
    pool, conn, cursor = _make_pool()
    # Four _safe_query calls: top_rows, month_rows, cluster_rows, abc_rows
    cursor.fetchall.side_effect = [
        # top_error_dfus: (top_10pct_share, top_20pct_share)
        [(0.35, 0.55)],
        # error_by_month: (month_num, wape, bias)
        [(1, 0.28, -0.02), (2, 0.31, 0.01)],
        # error_by_cluster: (cluster, wape, bias, share_of_total_error)
        [(0, 0.25, -0.01, 0.45), (1, 0.35, 0.03, 0.55)],
        # error_by_abc: (abc_class, wape, bias)
        [("A", 0.20, -0.005), ("B", 0.35, 0.02), ("C", 0.50, 0.05)],
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-eda/error-concentration")
    assert resp.status_code == 200
    data = resp.json()
    assert data["top_error_dfus"]["top_10pct_share"] == 0.35
    assert data["top_error_dfus"]["top_20pct_share"] == 0.55
    assert len(data["error_by_month"]) == 2
    assert data["error_by_month"][0]["month"] == 1
    assert data["error_by_month"][0]["wape"] == 0.28
    assert len(data["error_by_cluster"]) == 2
    assert data["error_by_cluster"][0]["share_of_total_error"] == 0.45
    assert len(data["error_by_abc"]) == 3
    assert data["error_by_abc"][0]["abc_class"] == "A"


@pytest.mark.asyncio
async def test_error_concentration_empty():
    """GET /cluster-eda/error-concentration handles no data gracefully."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.side_effect = [[], [], [], []]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-eda/error-concentration")
    assert resp.status_code == 200
    data = resp.json()
    assert data["top_error_dfus"]["top_10pct_share"] is None
    assert data["error_by_month"] == []
    assert data["error_by_cluster"] == []
    assert data["error_by_abc"] == []
    assert data["warning"] == "No data available"


@pytest.mark.asyncio
async def test_error_concentration_null_top_shares():
    """Error concentration handles NULL top_10pct_share gracefully."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.side_effect = [
        [(None, None)],  # top_rows with NULLs
        [(3, 0.40, 0.01)],
        [],
        [],
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-eda/error-concentration")
    assert resp.status_code == 200
    data = resp.json()
    assert data["top_error_dfus"]["top_10pct_share"] is None
    assert data["warning"] is None  # month_rows is not empty


# ---------------------------------------------------------------------------
# GET /cluster-eda/demand-distribution/{cluster_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_demand_distribution_returns_200():
    """GET /cluster-eda/demand-distribution/0 returns histogram and percentiles."""
    pool, conn, cursor = _make_pool()
    # _safe_fetchone (count), _safe_query (histogram), _safe_fetchone (percentiles), _safe_query (top_dfus)
    cursor.fetchone.side_effect = [
        (500,),           # count
        (10.0, 50.0, 120.0, 300.0, 750.0, 2500.0),  # percentiles
    ]
    cursor.fetchall.side_effect = [
        # histogram: (bucket, count)
        [("0", 50), ("1-10", 100), ("11-50", 150), ("51-100", 80), ("101-500", 70), ("501-1000", 30), ("1000+", 20)],
        # top_dfus: (sku_ck, mean_demand, cv)
        [("DFU_001", 950.5, 0.25), ("DFU_002", 800.3, 0.40)],
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-eda/demand-distribution/0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["cluster_id"] == 0
    assert data["n_dfus"] == 500
    assert len(data["histogram"]) == 7
    assert data["histogram"][0]["bucket"] == "0"
    assert data["histogram"][0]["count"] == 50
    assert data["percentiles"]["p10"] == 10.0
    assert data["percentiles"]["p50"] == 120.0
    assert data["percentiles"]["p99"] == 2500.0
    assert len(data["top_dfus"]) == 2
    assert data["top_dfus"][0]["sku_ck"] == "DFU_001"
    assert data["warning"] is None


@pytest.mark.asyncio
async def test_demand_distribution_empty_cluster():
    """GET /cluster-eda/demand-distribution/99 handles empty cluster."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        (0,),     # count = 0
        (None, None, None, None, None, None),  # no percentiles
    ]
    cursor.fetchall.side_effect = [
        [],  # no histogram
        [],  # no top_dfus
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-eda/demand-distribution/99")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_dfus"] == 0
    assert data["histogram"] == []
    assert data["percentiles"]["p10"] is None
    assert data["percentiles"]["p50"] is None
    assert data["top_dfus"] == []
    assert data["warning"] == "No data available"


@pytest.mark.asyncio
async def test_demand_distribution_null_count():
    """Demand distribution handles NULL count from DB."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        (None,),  # count is NULL
        (None, None, None, None, None, None),
    ]
    cursor.fetchall.side_effect = [[], []]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-eda/demand-distribution/5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_dfus"] == 0


# ---------------------------------------------------------------------------
# GET /cluster-eda/residual-analysis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_residual_analysis_returns_200():
    """GET /cluster-eda/residual-analysis returns all residual sections."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        (-5.2, 120.5, 0.35, 3.1),  # stats: mean, std, skew, kurtosis
    ]
    cursor.fetchall.side_effect = [
        # horizon: (lag, mean_error, rmse)
        [(0, -5.2, 120.5), (1, -8.1, 135.0)],
        # worst DFUs: (sku_ck, mean_abs_error, bias, cluster)
        [("DFU_BAD_1", 500.0, 0.35, 0), ("DFU_BAD_2", 450.0, -0.20, 1)],
        # bias by cluster: (cluster, bias)
        [(0, 0.02), (1, -0.05)],
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-eda/residual-analysis")
    assert resp.status_code == 200
    data = resp.json()
    assert data["residual_stats"]["mean"] == -5.2
    assert data["residual_stats"]["std"] == 120.5
    assert data["residual_stats"]["skew"] == 0.35
    assert data["residual_stats"]["kurtosis"] == 3.1
    assert len(data["residual_by_horizon"]) == 2
    assert data["residual_by_horizon"][0]["lag"] == 0
    assert len(data["worst_dfus"]) == 2
    assert data["worst_dfus"][0]["sku_ck"] == "DFU_BAD_1"
    assert len(data["bias_by_cluster"]) == 2
    assert data["bias_by_cluster"][0]["direction"] == "over"
    assert data["bias_by_cluster"][1]["direction"] == "under"
    assert data["warning"] is None


@pytest.mark.asyncio
async def test_residual_analysis_with_cluster_filter():
    """GET /cluster-eda/residual-analysis?cluster_id=0 filters by cluster."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [(-2.0, 80.0, 0.1, 2.5)]
    cursor.fetchall.side_effect = [
        [(0, -2.0, 80.0)],
        [("DFU_X", 200.0, 0.05, 0)],
        [(0, 0.01)],
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/cluster-eda/residual-analysis",
                params={"cluster_id": 0},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["residual_stats"]["mean"] == -2.0


@pytest.mark.asyncio
async def test_residual_analysis_empty():
    """Residual analysis handles no data gracefully."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [(None, None, None, None)]
    cursor.fetchall.side_effect = [[], [], []]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-eda/residual-analysis")
    assert resp.status_code == 200
    data = resp.json()
    assert data["residual_stats"]["mean"] is None
    assert data["residual_stats"]["std"] is None
    assert data["residual_by_horizon"] == []
    assert data["worst_dfus"] == []
    assert data["bias_by_cluster"] == []
    assert data["warning"] == "No data available"


@pytest.mark.asyncio
async def test_residual_analysis_custom_model_id():
    """GET /cluster-eda/residual-analysis?model_id=catboost_cluster passes model_id."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [(1.0, 50.0, -0.1, 2.8)]
    cursor.fetchall.side_effect = [
        [(0, 1.0, 50.0)],
        [("DFU_Y", 100.0, 0.01, 0)],
        [(0, 0.005)],
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/cluster-eda/residual-analysis",
                params={"model_id": "catboost_cluster"},
            )
    assert resp.status_code == 200
    # Verify model_id was passed as first param
    sql_call = cursor.execute.call_args_list[0]
    assert sql_call[0][1][0] == "catboost_cluster"


# ---------------------------------------------------------------------------
# GET /cluster-eda/seasonality-heatmap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seasonality_heatmap_returns_200():
    """GET /cluster-eda/seasonality-heatmap returns cluster x month WAPE matrix."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        (0, 1, 0.25), (0, 2, 0.30), (0, 3, 0.28),
        (1, 1, 0.40), (1, 2, 0.35), (1, 3, 0.38),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-eda/seasonality-heatmap")
    assert resp.status_code == 200
    data = resp.json()
    assert data["clusters"] == [0, 1]
    assert data["months"] == list(range(1, 13))
    assert len(data["values"]) == 2
    # Cluster 0 has data for months 1-3, rest are None
    assert data["values"][0][0] == 0.25  # month 1
    assert data["values"][0][1] == 0.30  # month 2
    assert data["values"][0][3] is None  # month 4 (no data)


@pytest.mark.asyncio
async def test_seasonality_heatmap_empty():
    """Seasonality heatmap returns empty structure when no data."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-eda/seasonality-heatmap")
    assert resp.status_code == 200
    data = resp.json()
    assert data["clusters"] == []
    assert data["months"] == list(range(1, 13))
    assert data["values"] == []
    assert data["warning"] == "No data available"


@pytest.mark.asyncio
async def test_seasonality_heatmap_single_cluster():
    """Heatmap with a single cluster still returns correct structure."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        (0, 6, 0.45), (0, 12, 0.60),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-eda/seasonality-heatmap")
    assert resp.status_code == 200
    data = resp.json()
    assert data["clusters"] == [0]
    assert len(data["values"]) == 1
    assert data["values"][0][5] == 0.45   # month 6
    assert data["values"][0][11] == 0.60  # month 12
    assert data["values"][0][0] is None   # month 1 has no data
