"""API tests for the Gen-4 G forecast explain endpoint."""
from __future__ import annotations

import pytest
from unittest.mock import patch
from httpx import ASGITransport, AsyncClient

from tests.api.conftest import make_pool as _make_pool


@pytest.mark.asyncio
async def test_explain_forecast_no_shap_table_still_200():
    # fetchone side-effects:
    # 1) shap table existence (False)
    # 2) forecast row
    # 3) ledger latest hash
    # 4) ledger insert returning id
    pool, conn, cur = _make_pool()
    cur.fetchone.side_effect = [
        (False,),
        ("2026-05-01", 123.45, "lgbm_cluster"),
        ("0" * 64,),
        (1,),
    ]
    # fetchall should not be called for SHAP since table missing; set empty
    cur.fetchall.return_value = []

    # explain.py calls _shap_table_exists -> fetchone(False) FIRST,
    # but in explain.py actual order is _fetch_forecast, then _fetch_top_features
    # which calls _shap_table_exists. Re-order side_effect to match.
    cur.fetchone.side_effect = [
        ("2026-05-01", 123.45, "lgbm_cluster"),  # forecast row
        (False,),                                  # shap table exists?
        ("0" * 64,),                               # ledger latest hash
        (1,),                                      # ledger insert id
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/forecast/explain/ITEM1/LOC1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["item_id"] == "ITEM1"
    assert data["loc"] == "LOC1"
    assert data["forecast"]["forecast_qty"] == 123.45
    assert data["top_features"] == []
    assert "scenarios" in data["counterfactual"]


@pytest.mark.asyncio
async def test_explain_forecast_with_shap_rows():
    pool, conn, cur = _make_pool()
    cur.fetchone.side_effect = [
        ("2026-05-01", 100.0, "lgbm_cluster"),  # forecast row
        (True,),                                 # shap table exists
        ("0" * 64,),                             # ledger hash
        (1,),                                    # ledger id
    ]
    cur.fetchall.return_value = [
        ("feature_a", 5.0, 2.0),
        ("feature_b", -3.0, 1.5),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/forecast/explain/ITEM2/LOC2")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["top_features"]) == 2
    assert data["top_features"][0]["name"] == "feature_a"
    # Counterfactual produced at least one scenario
    assert len(data["counterfactual"]["scenarios"]) >= 1


@pytest.mark.asyncio
async def test_explain_forecast_404_when_no_forecast():
    pool, conn, cur = _make_pool()
    cur.fetchone.return_value = None  # no forecast row

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/forecast/explain/UNKNOWN/UNK")
    assert resp.status_code == 404
