"""Tests for accuracy budget endpoints — decomposition, abc-breakdown,
model-comparison, monthly-trend, and forecast-value."""

import pytest
from unittest.mock import patch, MagicMock

import httpx
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool


def _assert_placeholder_param_parity(cursor):
    """Every parameterized query must bind exactly as many params as it has %s
    placeholders. The oracle / model-comparison IN-lists are built dynamically
    from the competing roster, so a count mismatch would raise at execute() time
    in real psycopg — but the fetch-mocked tests accept any args, so assert it
    explicitly here."""
    for call in cursor.execute.call_args_list:
        sql = call.args[0]
        params = call.args[1] if len(call.args) > 1 else None
        n_ph = sql.count("%s")
        if params is None:
            assert n_ph == 0, f"{n_ph} %s placeholders but no params: {sql[:90]!r}"
        else:
            assert n_ph == len(params), (
                f"placeholder/param mismatch: {n_ph} %s vs {len(params)} params: {sql[:90]!r}"
            )


@pytest.mark.asyncio
async def test_oracle_queries_placeholder_param_parity():
    """Guard the dynamic competing-roster IN-lists in the oracle/model-comparison
    queries against placeholder/param count drift (would 500 at runtime)."""
    # decomposition endpoint (7 queries, one with the dynamic oracle IN-list)
    pool, _conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        (500000.0, 2800000.0, 3000000.0, 50000),
        (400000.0, 2900000.0, 3000000.0, 2700000, 135000),
        (800000.0, 2500000.0, 3000000.0),
        (900000.0, 1000000.0),
    ]
    cursor.fetchall.side_effect = [
        [("A", 200000.0, 1500000.0, 1600000.0, 10000)],
        [("0", 100000.0, 700000.0, 750000.0, 12000, 0.05)],
        [(1, 40000.0, 250000.0)],
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/accuracy-budget/decomposition")
    assert resp.status_code == 200
    _assert_placeholder_param_parity(cursor)

    # model-comparison endpoint (per-model IN-list + oracle IN-list)
    pool, _conn, cursor = _make_pool()
    cursor.fetchall.side_effect = [[("lgbm_cluster", 100000.0, 700000.0, 750000.0)]]
    cursor.fetchone.side_effect = [(400000.0, 3000000.0)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/accuracy-budget/model-comparison")
    assert resp.status_code == 200
    _assert_placeholder_param_parity(cursor)


# ---------------------------------------------------------------------------
# GET /accuracy-budget/decomposition
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decomposition_returns_200():
    """GET /accuracy-budget/decomposition returns full accuracy budget."""
    pool, conn, cursor = _make_pool()
    # 7 sequential queries (a-g):
    # a) overall, b) by abc, c) by cluster, d) oracle, e) naive, f) monthly, g) A-class Q4
    cursor.fetchone.side_effect = [
        # a) overall: (sum_abs, sum_fcst, sum_actual, n_dfus)
        (500000.0, 2800000.0, 3000000.0, 50000),
        # d) oracle: (sum_abs, sum_fcst, sum_actual, n_rows, switched_rows)
        (400000.0, 2900000.0, 3000000.0, 2700000, 135000),
        # e) naive: (sum_abs, sum_prior_actual, sum_actual)
        (800000.0, 2500000.0, 3000000.0),
        # g) A-class Q4: (sum_fcst, sum_actual)
        (900000.0, 1000000.0),
    ]
    cursor.fetchall.side_effect = [
        # b) by abc: (abc_class, sum_abs, sum_fcst, sum_actual, n_dfus)
        [
            ("A", 200000.0, 1500000.0, 1600000.0, 10000),
            ("B", 150000.0, 800000.0, 850000.0, 15000),
            ("C", 150000.0, 500000.0, 550000.0, 25000),
        ],
        # c) by cluster: (cluster, sum_abs, sum_fcst, sum_actual, n_dfus, intermittency_ratio)
        [
            ("0", 100000.0, 700000.0, 750000.0, 12000, 0.05),
            ("1", 200000.0, 1000000.0, 1100000.0, 18000, 0.65),
            ("(unassigned)", 200000.0, 1100000.0, 1150000.0, 20000, 0.10),
        ],
        # f) monthly: (cal_month, sum_abs, sum_actual)
        [
            (1, 40000.0, 250000.0),
            (6, 60000.0, 200000.0),  # high WAPE month
            (12, 70000.0, 200000.0),  # high WAPE month
        ],
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/accuracy-budget/decomposition")
    assert resp.status_code == 200
    data = resp.json()
    assert data["current_accuracy"] is not None
    assert data["current_wape"] is not None
    assert data["current_bias"] is not None
    assert data["n_dfus"] == 50000
    assert data["model_id"] == "lgbm_cluster"
    assert data["oracle_ceiling"] is not None
    assert data["naive_baseline"] is not None
    assert data["forecast_value_added"] is not None
    assert data["addressable_gap"] is not None
    assert len(data["abc_breakdown"]) == 3
    assert data["abc_breakdown"][0]["abc_class"] == "A"
    assert len(data["cluster_breakdown"]) == 3
    assert len(data["components"]) > 0
    assert data["irreducible_noise"] is not None


@pytest.mark.asyncio
async def test_decomposition_custom_model():
    """GET /accuracy-budget/decomposition?model_id=catboost_cluster uses correct model."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        (100.0, 500.0, 600.0, 10),
        (80.0, 550.0, 600.0, 1000, 50),
        (200.0, 400.0, 600.0),
        (100.0, 200.0),
    ]
    cursor.fetchall.side_effect = [
        [("A", 50.0, 250.0, 300.0, 5)],
        [("0", 50.0, 250.0, 300.0, 5, 0.1)],
        [(1, 10.0, 60.0)],
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/accuracy-budget/decomposition",
                params={"model_id": "catboost_cluster"},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == "catboost_cluster"


@pytest.mark.asyncio
async def test_decomposition_zero_actual():
    """Decomposition handles zero actual sums without division errors."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        (0.0, 0.0, 0.0, 0),    # overall (all zeros)
        (0.0, 0.0, 0.0, 0, 0), # oracle
        (0.0, 0.0, 0.0),       # naive
        (0.0, 0.0),             # A Q4
    ]
    cursor.fetchall.side_effect = [[], [], []]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/accuracy-budget/decomposition")
    assert resp.status_code == 200
    data = resp.json()
    assert data["current_accuracy"] is None
    assert data["current_wape"] is None


# ---------------------------------------------------------------------------
# GET /accuracy-budget/abc-breakdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_abc_breakdown_returns_200():
    """GET /accuracy-budget/abc-breakdown returns classes with volume/error share."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("A", 200000.0, 1500000.0, 1600000.0, 10000),
        ("B", 150000.0, 800000.0, 850000.0, 15000),
        ("C", 150000.0, 500000.0, 550000.0, 25000),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/accuracy-budget/abc-breakdown")
    assert resp.status_code == 200
    data = resp.json()
    assert "classes" in data
    assert len(data["classes"]) == 3
    cls_a = data["classes"][0]
    assert cls_a["abc"] == "A"
    assert cls_a["accuracy"] is not None
    assert cls_a["wape"] is not None
    assert cls_a["bias"] is not None
    assert cls_a["n_dfus"] == 10000
    assert cls_a["volume_share"] is not None
    assert cls_a["error_share"] is not None


@pytest.mark.asyncio
async def test_abc_breakdown_empty():
    """GET /accuracy-budget/abc-breakdown handles empty data."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/accuracy-budget/abc-breakdown")
    assert resp.status_code == 200
    data = resp.json()
    assert data["classes"] == []


# ---------------------------------------------------------------------------
# GET /accuracy-budget/model-comparison
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_comparison_returns_200():
    """GET /accuracy-budget/model-comparison returns per-model accuracy."""
    pool, conn, cursor = _make_pool()
    # Two queries: per-model, oracle
    cursor.fetchall.return_value = [
        ("catboost_cluster", 300000.0, 1800000.0, 2000000.0),
        ("lgbm_cluster", 250000.0, 1850000.0, 2000000.0),
        ("xgboost_cluster", 280000.0, 1820000.0, 2000000.0),
    ]
    cursor.fetchone.return_value = (200000.0, 2000000.0)  # oracle
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/accuracy-budget/model-comparison")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert len(data["models"]) == 3
    assert data["models"][0]["model_id"] == "catboost_cluster"
    assert data["models"][0]["accuracy"] is not None
    assert data["models"][0]["wape"] is not None
    assert data["models"][0]["bias"] is not None
    assert "oracle_ceiling" in data
    assert data["oracle_ceiling"]["accuracy"] is not None


@pytest.mark.asyncio
async def test_model_comparison_empty():
    """GET /accuracy-budget/model-comparison handles empty results."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = (0.0, 0.0)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/accuracy-budget/model-comparison")
    assert resp.status_code == 200
    data = resp.json()
    assert data["models"] == []


# ---------------------------------------------------------------------------
# GET /accuracy-budget/monthly-trend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_monthly_trend_returns_200():
    """GET /accuracy-budget/monthly-trend returns monthly accuracy with flags."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        (1, 40000.0, 200000.0, 250000.0, 5000),
        (6, 80000.0, 180000.0, 200000.0, 5000),  # high WAPE
        (12, 30000.0, 220000.0, 230000.0, 5000),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/accuracy-budget/monthly-trend")
    assert resp.status_code == 200
    data = resp.json()
    assert "months" in data
    assert len(data["months"]) == 3
    assert data["months"][0]["month"] == 1
    assert data["months"][0]["accuracy"] is not None
    assert "worst_month" in data
    assert "best_month" in data
    assert data["worst_month"] is not None
    assert data["best_month"] is not None


@pytest.mark.asyncio
async def test_monthly_trend_empty():
    """GET /accuracy-budget/monthly-trend handles empty data."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/accuracy-budget/monthly-trend")
    assert resp.status_code == 200
    data = resp.json()
    assert data["months"] == []
    assert data["worst_month"] is None
    assert data["best_month"] is None


@pytest.mark.asyncio
async def test_monthly_trend_seasonal_flag():
    """Monthly trend flags months with WAPE >20% above average as seasonal_boundary."""
    pool, conn, cursor = _make_pool()
    # Month 6: WAPE = 80000/200000 = 40%
    # Month 1: WAPE = 10000/250000 = 4%
    # Average WAPE = 90000/450000 = 20%
    # 40% > 20% * 1.2 = 24% => flagged
    cursor.fetchall.return_value = [
        (1, 10000.0, 240000.0, 250000.0, 5000),
        (6, 80000.0, 120000.0, 200000.0, 5000),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/accuracy-budget/monthly-trend")
    assert resp.status_code == 200
    data = resp.json()
    # Month 6 should have the seasonal_boundary flag
    month_6 = [m for m in data["months"] if m["month"] == 6][0]
    assert month_6.get("flag") == "seasonal_boundary"


# ---------------------------------------------------------------------------
# GET /accuracy-budget/forecast-value
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forecast_value_returns_200():
    """GET /accuracy-budget/forecast-value returns baselines and value_added."""
    pool, conn, cursor = _make_pool()
    # 4 sequential queries: ml, seasonal_naive, rolling_3m, flat_last_month
    cursor.fetchone.side_effect = [
        (500000.0, 3000000.0),   # ml: (sum_abs, sum_actual)
        (800000.0, 3000000.0),   # seasonal naive
        (700000.0, 3000000.0),   # rolling 3m
        (900000.0, 3000000.0),   # flat last month
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/accuracy-budget/forecast-value")
    assert resp.status_code == 200
    data = resp.json()
    assert "baselines" in data
    assert len(data["baselines"]) == 3
    assert data["baselines"][0]["name"] == "seasonal_naive"
    assert data["baselines"][0]["accuracy"] is not None
    assert data["baselines"][1]["name"] == "rolling_3m_avg"
    assert data["baselines"][2]["name"] == "flat_last_month"
    assert "ml_model" in data
    assert data["ml_model"]["name"] == "lgbm_cluster"
    assert data["ml_model"]["accuracy"] is not None
    assert "value_added" in data
    assert "vs_seasonal_naive" in data["value_added"]
    assert "vs_rolling_3m" in data["value_added"]
    assert "vs_flat" in data["value_added"]


@pytest.mark.asyncio
async def test_forecast_value_ml_better_than_all_baselines():
    """ML model accuracy > all baselines => positive value_added."""
    pool, conn, cursor = _make_pool()
    # ML: accuracy = 100 - 100*(200000/1000000) = 80%
    # SN: accuracy = 100 - 100*(400000/1000000) = 60%
    # R3: accuracy = 100 - 100*(350000/1000000) = 65%
    # Flat: accuracy = 100 - 100*(500000/1000000) = 50%
    cursor.fetchone.side_effect = [
        (200000.0, 1000000.0),   # ml
        (400000.0, 1000000.0),   # seasonal naive
        (350000.0, 1000000.0),   # rolling 3m
        (500000.0, 1000000.0),   # flat
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/accuracy-budget/forecast-value")
    assert resp.status_code == 200
    data = resp.json()
    # All value_added should be positive
    assert data["value_added"]["vs_seasonal_naive"] > 0
    assert data["value_added"]["vs_rolling_3m"] > 0
    assert data["value_added"]["vs_flat"] > 0


@pytest.mark.asyncio
async def test_forecast_value_zero_actual():
    """Forecast value handles zero actual sums without division errors."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        (0.0, 0.0),  # ml
        (0.0, 0.0),  # seasonal naive
        (0.0, 0.0),  # rolling 3m
        (0.0, 0.0),  # flat
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/accuracy-budget/forecast-value")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ml_model"]["accuracy"] is None
    for baseline in data["baselines"]:
        assert baseline["accuracy"] is None


# ---------------------------------------------------------------------------
# Helpers unit tests
# ---------------------------------------------------------------------------


def test_wape_helper():
    from api.routers.forecasting.accuracy_budget import _wape
    assert _wape(200.0, 1000.0) == 20.0
    assert _wape(0.0, 1000.0) == 0.0
    assert _wape(100.0, 0.0) is None


def test_accuracy_helper():
    from api.routers.forecasting.accuracy_budget import _accuracy
    assert _accuracy(200.0, 1000.0) == 80.0
    assert _accuracy(0.0, 1000.0) == 100.0
    assert _accuracy(100.0, 0.0) is None


def test_bias_helper():
    from api.routers.forecasting.accuracy_budget import _bias
    result = _bias(1100.0, 1000.0)
    assert abs(result - 0.1) < 0.001
    assert _bias(500.0, 0.0) is None


def test_round_or_none_helper():
    from api.routers.forecasting.accuracy_budget import _round_or_none
    assert _round_or_none(3.14159, 2) == 3.14
    assert _round_or_none(None) is None
    assert _round_or_none(0.0, 4) == 0.0
