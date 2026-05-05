"""Tests for ``/integration/*`` endpoints (api/routers/platform/integration.py).

Uses the project-standard inline ``httpx.AsyncClient`` + ``ASGITransport``
pattern, with the runner dependency overridden via ``app.dependency_overrides``.
"""
from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport

from api.main import app
from api.routers.platform.integration import KNOWN_DOMAINS, _get_runner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_runner():
    """In-memory IntegrationRunner stand-in with MagicMock methods."""
    runner = MagicMock()
    runner.submit.return_value = "fake-job-id-123"
    runner.get.return_value = None
    runner.list.return_value = []
    runner.health.return_value = {"pool": "ok", "table": "ok"}
    return runner


@pytest.fixture
def override_runner(fake_runner):
    """Install fake runner via app.dependency_overrides; clean up after."""
    app.dependency_overrides[_get_runner] = lambda: fake_runner
    try:
        yield fake_runner
    finally:
        app.dependency_overrides.pop(_get_runner, None)


@pytest.fixture
async def client(mock_pool):
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


# ---------------------------------------------------------------------------
# /integration/health
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_health_endpoint_returns_200(client, override_runner):
    resp = await client.get("/integration/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["pool"] == "ok"
    assert body["table"] == "ok"


@pytest.mark.asyncio
async def test_health_passes_through_runner_payload(client, override_runner):
    override_runner.health.return_value = {"pool": "degraded", "table": "missing", "extra": 1}
    resp = await client.get("/integration/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["pool"] == "degraded"
    assert body["table"] == "missing"
    assert body["extra"] == 1  # extra='allow' on HealthStatus


# ---------------------------------------------------------------------------
# /integration/domains
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_domains_endpoint_lists_11_known_domains(client, override_runner):
    resp = await client.get("/integration/domains")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == len(KNOWN_DOMAINS) == 11
    names = [d["name"] for d in items]
    assert names == KNOWN_DOMAINS


@pytest.mark.asyncio
async def test_domains_marks_inventory_as_partitioned(client, override_runner):
    resp = await client.get("/integration/domains")
    items = {d["name"]: d for d in resp.json()["items"]}
    inv = items["inventory"]
    assert inv["partitioned"] is True
    assert inv["partition_format"] == "YYYY_MM"
    assert inv["partition_field"] == "snapshot_date"


@pytest.mark.asyncio
async def test_domains_marks_item_as_unpartitioned(client, override_runner):
    resp = await client.get("/integration/domains")
    items = {d["name"]: d for d in resp.json()["items"]}
    item = items["item"]
    assert item["partitioned"] is False
    assert item["partition_format"] is None
    assert item["partition_field"] is None


# ---------------------------------------------------------------------------
# POST /integration/jobs
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_submit_job_requires_api_key(client, override_runner, monkeypatch):
    """When API_KEY is set in env, omitting X-API-Key header returns 401."""
    monkeypatch.setenv("API_KEY", "secret-key")
    resp = await client.post(
        "/integration/jobs",
        json={"domain": "sales", "mode": "onetime"},
    )
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_submit_job_success(client, override_runner):
    resp = await client.post(
        "/integration/jobs",
        json={"domain": "sales", "mode": "onetime"},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["job_id"] == "fake-job-id-123"
    assert body["status"] == "queued"
    override_runner.submit.assert_called_once()


@pytest.mark.asyncio
async def test_submit_job_unknown_domain(client, override_runner):
    resp = await client.post(
        "/integration/jobs",
        json={"domain": "totally_made_up", "mode": "onetime"},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert "Unknown domain" in body["detail"]


@pytest.mark.asyncio
async def test_submit_job_invalid_mode(client, override_runner):
    resp = await client.post(
        "/integration/jobs",
        json={"domain": "sales", "mode": "garbage"},
    )
    # Pydantic Literal validation -> 422
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_submit_job_file_mode_partitioned_requires_slice(client, override_runner):
    resp = await client.post(
        "/integration/jobs",
        json={"domain": "customer_demand", "mode": "file"},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert "slice required" in body["detail"]


@pytest.mark.asyncio
async def test_submit_job_file_mode_partitioned_with_slice_ok(client, override_runner):
    resp = await client.post(
        "/integration/jobs",
        json={"domain": "customer_demand", "mode": "file", "slice": "2026-04"},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"


@pytest.mark.asyncio
async def test_submit_job_file_mode_partitioned_with_file_ok(client, override_runner):
    """Files inside the project data root are accepted."""
    from pathlib import Path
    valid = str(Path(__file__).resolve().parents[2] / "data" / "customer_demand_clean.csv")
    resp = await client.post(
        "/integration/jobs",
        json={"domain": "customer_demand", "mode": "file", "file": valid},
    )
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_submit_job_file_path_outside_data_root_rejected(client, override_runner):
    """Path-traversal / outside-root paths are rejected with 422."""
    resp = await client.post(
        "/integration/jobs",
        json={"domain": "customer_demand", "mode": "file", "file": "/tmp/x.csv"},
    )
    assert resp.status_code == 422
    assert "data" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_submit_job_file_mode_unpartitioned_no_slice_required(client, override_runner):
    """Non-partitioned domains accept file mode with no slice."""
    resp = await client.post(
        "/integration/jobs",
        json={"domain": "item", "mode": "file"},
    )
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_submit_job_passes_request_to_runner(client, override_runner):
    await client.post(
        "/integration/jobs",
        json={"domain": "forecast", "mode": "delta", "slice": "2026-03"},
    )
    override_runner.submit.assert_called_once()
    kwargs = override_runner.submit.call_args.kwargs
    assert kwargs["domain"] == "forecast"
    assert kwargs["mode"] == "delta"
    assert kwargs["slice"] == "2026-03"
    assert kwargs["triggered_by"] == "api"


# ---------------------------------------------------------------------------
# GET /integration/jobs
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_list_jobs_empty(client, override_runner):
    resp = await client.get("/integration/jobs")
    assert resp.status_code == 200
    assert resp.json() == {"items": []}


@pytest.mark.asyncio
async def test_list_jobs_returns_runner_items(client, override_runner):
    override_runner.list.return_value = [
        {
            "id": "job-1", "domain": "sales", "mode": "onetime",
            "slice": None, "file_path": None, "status": "success",
            "rows_loaded": 100, "error_message": None,
            "started_at": None, "completed_at": None,
            "duration_ms": 1234, "triggered_by": "api",
        },
    ]
    resp = await client.get("/integration/jobs")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["id"] == "job-1"


@pytest.mark.asyncio
async def test_list_jobs_with_domain_filter(client, override_runner):
    await client.get("/integration/jobs?domain=sales&limit=20")
    override_runner.list.assert_called_once_with(domain="sales", limit=20)


@pytest.mark.asyncio
async def test_list_jobs_unknown_domain_filter_422(client, override_runner):
    resp = await client.get("/integration/jobs?domain=totally_made_up")
    assert resp.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_limit", [0, -1, 201, 300])
async def test_list_jobs_limit_validation(client, override_runner, bad_limit):
    resp = await client.get(f"/integration/jobs?limit={bad_limit}")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_list_jobs_default_limit(client, override_runner):
    await client.get("/integration/jobs")
    override_runner.list.assert_called_once_with(domain=None, limit=50)


# ---------------------------------------------------------------------------
# GET /integration/jobs/{job_id}
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_job_404_when_missing(client, override_runner):
    override_runner.get.return_value = None
    resp = await client.get("/integration/jobs/nope-nope-nope")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_get_job_returns_job_when_exists(client, override_runner):
    override_runner.get.return_value = {
        "id": "real-id", "domain": "sales", "mode": "onetime",
        "slice": None, "file_path": None, "status": "running",
        "rows_loaded": 0, "error_message": None,
        "started_at": "2026-04-01T00:00:00", "completed_at": None,
        "duration_ms": None, "triggered_by": "api",
    }
    resp = await client.get("/integration/jobs/real-id")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "real-id"
    assert body["status"] == "running"
    override_runner.get.assert_called_once_with("real-id")
