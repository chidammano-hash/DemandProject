"""API test for $-denominated Control Tower KPIs (Gen-4 Roadmap 1.7)."""
from __future__ import annotations

import httpx
import pytest
from httpx import ASGITransport
from unittest.mock import patch

from tests.api.conftest import make_pool as _make_pool


@pytest.mark.asyncio
async def test_control_tower_kpis_financial_200():
    pool, conn, cursor = _make_pool(
        fetchone_return=(
            2_500_000.0,  # inventory_value
            75_000.0,     # below_ss_value_gap
            120_000.0,    # excess_value
            180_000.0,    # open_exception_value
            95_000.0,     # critical_exception_value
            12_500.0,     # loss_of_sales_7d_value
            7_200.0,      # monthly_holding_cost
            45_000.0,     # shortage_value_3m
        )
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/control-tower/kpis-financial")

    assert resp.status_code == 200
    data = resp.json()
    assert data["currency"] == "USD"
    assert data["inventory_value"] == 2_500_000.0
    assert data["below_ss_value_gap"] == 75_000.0
    assert data["excess_value"] == 120_000.0
    assert data["open_exception_value"] == 180_000.0
    assert data["critical_exception_value"] == 95_000.0
    assert data["shortage_value_3m"] == 45_000.0


@pytest.mark.asyncio
async def test_control_tower_kpis_financial_empty():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None  # simulate no rows

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/control-tower/kpis-financial")

    assert resp.status_code == 200
    data = resp.json()
    assert data["inventory_value"] == 0.0
    assert data["open_exception_value"] == 0.0
