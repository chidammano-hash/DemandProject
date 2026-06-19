"""Tests for POST /customer-analytics/recalculate.

Mirrors the SKU Features compute-job endpoint: the route submits a background
``refresh_customer_analytics`` job and returns 202 + job_id.
"""
from unittest.mock import MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport

from tests.api.conftest import make_pool


@pytest.mark.asyncio
async def test_recalculate_submits_job():
    """POST /customer-analytics/recalculate should submit a background job."""
    pool, _, _cursor = make_pool()

    mock_mgr = MagicMock()
    mock_mgr.submit_job.return_value = "job-ca-123"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("common.services.job_registry.JobManager", return_value=mock_mgr),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/customer-analytics/recalculate")

    assert resp.status_code == 202
    data = resp.json()
    assert data["job_id"] == "job-ca-123"
    assert data["status"] == "queued"
    mock_mgr.submit_job.assert_called_once_with(
        "refresh_customer_analytics",
        {},
        label="Recalculate Customer Analytics",
        triggered_by="api",
    )
