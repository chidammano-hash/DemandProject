"""API tests for /forecast/production/* endpoints — F1.1."""

import datetime
import pytest
from unittest.mock import patch, MagicMock
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# /forecast/production/versions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_versions_200():
    """GET /forecast/production/versions returns list of versions."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("2026-03", 1000, 12000, datetime.datetime(2026, 3, 1, 6, 0, 0)),
        ("2026-02", 980,  11760, datetime.datetime(2026, 2, 1, 6, 0, 0)),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/production/versions")

    assert resp.status_code == 200
    data = resp.json()
    assert "versions" in data
    assert len(data["versions"]) == 2
    v = data["versions"][0]
    assert "plan_version" in v
    assert "dfu_count" in v
    assert "total_rows" in v
    assert "generated_at" in v


@pytest.mark.asyncio
async def test_versions_empty():
    """Empty table returns empty versions list."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/production/versions")

    assert resp.status_code == 200
    assert resp.json()["versions"] == []


# ---------------------------------------------------------------------------
# /forecast/production/summary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summary_no_data():
    """When table is empty, summary returns zeros without 500."""
    pool, conn, cursor = _make_pool()
    # fetchone returns None → no plan_version → early return
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/production/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_dfu_count"] == 0
    assert data["total_forecast_qty"] == 0.0
    assert data["by_abc_class"] == []


@pytest.mark.asyncio
async def test_summary_with_version_param():
    """When plan_version is provided, skips resolution and returns summary."""
    pool, conn, cursor = _make_pool()
    # plan_version provided → no version-resolution fetchone.
    # Call order: fetchone (summary), fetchall (abc_rows), fetchone (ci_row).
    cursor.fetchone.side_effect = [
        (1000, 55000.0, datetime.datetime(2026, 3, 1, 6, 0, 0)),  # summary row
        (800, 1000, 35.0),                                          # ci_row
    ]
    cursor.fetchall.return_value = [
        ("A", 200, 20000.0),
        ("B", 500, 25000.0),
        ("C", 300, 10000.0),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/production/summary?plan_version=2026-03&horizon_months=3")

    assert resp.status_code == 200
    data = resp.json()
    assert data["plan_version"] == "2026-03"
    assert data["total_dfu_count"] == 1000
    assert len(data["by_abc_class"]) == 3
    assert "ci_coverage_pct" in data
    assert "avg_ci_width" in data


# ---------------------------------------------------------------------------
# /forecast/production (DFU series)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dfu_404_no_forecast():
    """DFU with no forecast rows returns 404."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None  # no plan_version found

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/production?item_id=ITEM001&loc=LOC1")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_dfu_200_with_version():
    """DFU forecast with plan_version returns forecast series."""
    pool, conn, cursor = _make_pool()
    # plan_version provided → skip fetchone for version resolution,
    # fetchone called once for promoted run lookup (return None = no promoted run)
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = [
        (
            datetime.date(2026, 4, 1),   # forecast_month
            150.0,                         # forecast_qty
            120.0,                         # forecast_qty_lower
            180.0,                         # forecast_qty_upper
            "lgbm_cluster",               # model_id
            2,                             # cluster_id
            1,                             # horizon_months
            True,                          # is_recursive
            "actual",                      # lag_source
            datetime.datetime(2026, 3, 1), # generated_at
        ),
        (
            datetime.date(2026, 5, 1),
            140.0, 110.0, 170.0,
            "lgbm_cluster", 2, 2, True, "predicted",
            datetime.datetime(2026, 3, 1),
        ),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/production?item_id=ITEM001&loc=LOC1&plan_version=2026-03"
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["item_id"] == "ITEM001"
    assert data["loc"] == "LOC1"
    assert data["plan_version"] == "2026-03"
    assert data["model_id"] == "lgbm_cluster"
    assert len(data["forecasts"]) == 2
    f = data["forecasts"][0]
    assert "forecast_month" in f
    assert "forecast_qty" in f
    assert "lag_source" in f
    assert f["lag_source"] == "actual"


@pytest.mark.asyncio
async def test_dfu_404_version_exists_but_no_rows():
    """Plan version exists but no rows for this DFU → 404."""
    pool, conn, cursor = _make_pool()
    # fetchall returns empty (no rows for this item/loc in the version)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/production?item_id=UNKNOWN&loc=LOC1&plan_version=2026-03"
            )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_dfu_horizon_capped():
    """horizon param is capped at 18 without error."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = [
        (
            datetime.date(2026, 4, 1), 100.0, None, None,
            "lgbm_cluster", 1, 1, False, "actual",
            datetime.datetime(2026, 3, 1),
        )
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/production?item_id=ITEM001&loc=LOC1&plan_version=2026-02&horizon=99"
            )

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_dfu_18_month_horizon():
    """18-month planning horizon returns all 18 forecast rows."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = [
        (
            datetime.date(2026, 2 + i if i < 11 else i - 10, 1),
            float(100 + i), None, None,
            "lgbm_cluster", 2, i + 1, i > 0, "actual" if i == 0 else "predicted",
            datetime.datetime(2026, 2, 24),
        )
        for i in range(18)
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/production?item_id=ITEM001&loc=LOC1&plan_version=2026-02&horizon=18"
            )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["forecasts"]) == 18
    assert data["forecasts"][0]["lag_source"] == "actual"
    assert data["forecasts"][1]["lag_source"] == "predicted"


@pytest.mark.asyncio
async def test_dfu_includes_promoted_run():
    """DFU forecast includes promoted tuning run metadata when present."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        (
            datetime.date(2026, 4, 1), 150.0, 120.0, 180.0,
            "lgbm_cluster", 2, 1, True, "actual",
            datetime.datetime(2026, 3, 1),
        ),
    ]
    # fetchone returns promoted run row
    cursor.fetchone.return_value = (
        17,                                     # run_id
        "enhanced_reg_v3",                       # run_label
        71.79,                                   # accuracy_pct
        28.21,                                   # wape
        datetime.datetime(2026, 3, 20, 14, 0),  # promoted_at
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/production?item_id=ITEM001&loc=LOC1&plan_version=2026-03"
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["promoted_run"] is not None
    assert data["promoted_run"]["run_id"] == 17
    assert data["promoted_run"]["run_label"] == "enhanced_reg_v3"
    assert data["promoted_run"]["accuracy_pct"] == 71.79
    assert data["promoted_run"]["wape"] == 28.21


@pytest.mark.asyncio
async def test_dfu_promoted_run_null_when_no_promoted():
    """promoted_run is null when no tuning run is promoted."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        (
            datetime.date(2026, 4, 1), 150.0, None, None,
            "lgbm_cluster", 1, 1, False, "actual",
            datetime.datetime(2026, 3, 1),
        ),
    ]
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/production?item_id=ITEM001&loc=LOC1&plan_version=2026-03"
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["promoted_run"] is None


# ---------------------------------------------------------------------------
# /forecast/production/staging — latest immutable release candidates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_staging_overlay_uses_latest_candidate_run_and_preserves_source_model():
    pool, _, cursor = _make_pool()
    run_id = "00000000-0000-0000-0000-000000000111"
    cursor.fetchall.return_value = [
        (
            "champion",
            "mstl",
            datetime.date(2026, 7, 1),
            100.0,
            90.0,
            110.0,
            1,
            "stable",
            "actual",
            datetime.datetime(2026, 7, 10, 12, 0, 0),
            run_id,
        )
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/forecast/production/staging?item_id=ITEM001&loc=1401-BULK"
            )

    assert response.status_code == 200
    row = response.json()["models"]["champion"][0]
    assert row["source_model_id"] == "mstl"
    assert row["source_run_id"] == run_id
    sql = cursor.execute.call_args.args[0]
    assert "forecast_generation_run" in sql
    assert "generation.run_rank = 1" in sql
    assert "staging.run_id = generation.run_id" in sql
    assert "metadata ->> %s = %s" in sql


# ---------------------------------------------------------------------------
# /forecast/candidate — per-model backtest (past, out-of-sample) predictions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_candidate_groups_by_model():
    """GET /forecast/candidate groups rows by model_id with forecast + actual."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        # model_id, forecast_month, forecast_qty, lower, upper,
        # actual_qty, accuracy_pct, wape, bias, horizon_months, cluster_id
        ("lgbm_cluster", datetime.date(2025, 1, 1), 100.0, 90.0, 110.0, 105.0, 95.2, 4.8, -0.05, 1, "c1"),
        ("lgbm_cluster", datetime.date(2025, 2, 1), 120.0, 108.0, 132.0, 118.0, 96.0, 4.0, 0.02, 1, "c1"),
        ("mstl", datetime.date(2025, 1, 1), 80.0, None, None, 105.0, 76.0, 24.0, -0.24, 1, None),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/candidate?item_id=100320&loc=1401-BULK")

    assert resp.status_code == 200
    data = resp.json()
    assert data["item_id"] == "100320"
    assert data["loc"] == "1401-BULK"
    assert set(data["models"].keys()) == {"lgbm_cluster", "mstl"}
    lgbm = data["models"]["lgbm_cluster"]
    assert len(lgbm) == 2
    assert lgbm[0]["forecast_qty"] == 100.0
    assert lgbm[0]["actual_qty"] == 105.0
    assert lgbm[0]["accuracy_pct"] == 95.2
    # Null CI bounds survive as None (not 0).
    assert data["models"]["mstl"][0]["forecast_qty_lower"] is None


@pytest.mark.asyncio
async def test_candidate_table_missing_returns_empty():
    """A missing fact_candidate_forecast table degrades to empty models (clean install)."""
    import psycopg
    pool, conn, cursor = _make_pool()
    cursor.execute.side_effect = psycopg.errors.UndefinedTable("relation does not exist")

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/candidate?item_id=X&loc=Y")

    assert resp.status_code == 200
    assert resp.json()["models"] == {}
