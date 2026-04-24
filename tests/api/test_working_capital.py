"""API tests for the Gen-4 SC-10 working_capital router."""
from unittest.mock import patch

import pytest

from tests.api.conftest import make_pool as _make_pool

pytest_plugins = ["anyio"]


@pytest.mark.asyncio
async def test_working_capital_dio_and_turns():
    """DIO and turns compute when avg_inventory_value and period_cogs present.

    avg_inv=1000, period_cogs=36500, 365 days -> cogs/day=100 -> DIO=10 -> turns=36.5
    """
    pool, conn, cursor = _make_pool(
        fetchone_return=(1000.0, 36500.0, 12, None),
    )

    with patch("api.core._get_pool", return_value=pool):
        from httpx import ASGITransport, AsyncClient

        from api.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/analytics/working-capital"
                "?period_from=2025-01-01&period_to=2025-12-31"
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["dio_days"] == pytest.approx(10.0, abs=0.1)
    assert data["inventory_turns"] == pytest.approx(36.5, abs=0.1)
    # DSO not provided → cash_to_cash should be None
    assert data["cash_to_cash_days"] is None
    assert data["period_from"] == "2025-01-01"
    assert "notes" in data


@pytest.mark.asyncio
async def test_working_capital_c2c_with_dso_override():
    """Supplying ?dso_days= allows C2C = DIO + DSO - DPO to compute."""
    # avg_inv=1000, cogs=36500/yr -> DIO=10; dpo=5 (mocked), dso=45 -> c2c=50
    pool, conn, cursor = _make_pool(
        fetchone_return=(1000.0, 36500.0, 12, 5.0),
    )

    with patch("api.core._get_pool", return_value=pool):
        from httpx import ASGITransport, AsyncClient

        from api.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/analytics/working-capital"
                "?period_from=2025-01-01&period_to=2025-12-31&dso_days=45"
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["dpo_days"] == pytest.approx(5.0)
    assert data["dso_days"] == pytest.approx(45.0)
    assert data["cash_to_cash_days"] == pytest.approx(10.0 + 45.0 - 5.0, abs=0.1)


@pytest.mark.asyncio
async def test_rolling_13_week_returns_weeks_envelope():
    from datetime import date
    pool, conn, cursor = _make_pool(fetchall_return=[
        (date(2026, 1, 5), 2026, 2, 1000, 950, 950),
        (date(2026, 1, 12), 2026, 3, 1200, 1100, 1100),
    ])

    with patch("api.core._get_pool", return_value=pool):
        from httpx import ASGITransport, AsyncClient

        from api.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/analytics/rolling-13-week")
    assert resp.status_code == 200
    data = resp.json()
    assert "weeks" in data
    assert len(data["weeks"]) == 2
    assert data["weeks"][0]["iso_week"] == 2
    assert data["weeks"][1]["qty_shipped"] == 1100
