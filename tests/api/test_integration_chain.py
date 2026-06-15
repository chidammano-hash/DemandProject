"""US17d — /integration/chains on the unified JobManager backend.

Chains submit as JobManager pipelines of load_domain steps; reads aggregate the
job_history pipeline rows (+ legacy integration_chain fallback) into the same
chain response shape. Inline httpx.AsyncClient + ASGITransport; the chain runner
is driven for real against a mocked pool, with JobManager patched for submission.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport

from api.main import app
from api.routers.platform.integration_chain import _get_chain_runner


@pytest.fixture
async def client(mock_pool):
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest.fixture
def real_runner(mock_pool):
    """Install a real ChainJobRunner bound to the mock pool via dependency override."""
    from common.services.integration_chain_jobs import ChainJobRunner

    pool, _, cursor = mock_pool
    app.dependency_overrides[_get_chain_runner] = lambda: ChainJobRunner(pool)
    try:
        yield cursor
    finally:
        app.dependency_overrides.pop(_get_chain_runner, None)


_PIPELINE_DESC = [
    ("pipeline_id",), ("pipeline_step",), ("job_id",), ("status",), ("params",),
    ("result",), ("error",), ("started_at",), ("completed_at",), ("triggered_by",),
]


# ---------------------------------------------------------------------------
# POST /integration/chains
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_submit_chain_builds_pipeline(client, real_runner):
    fake_mgr = MagicMock()
    fake_mgr.submit_pipeline.return_value = "pipe_abc123"
    with patch("common.services.job_registry.JobManager", return_value=fake_mgr):
        resp = await client.post(
            "/integration/chains",
            json={"jobs": [
                {"domain": "item", "mode": "onetime"},
                {"domain": "sales", "mode": "delta", "slice": "2026-04"},
            ]},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["chain_id"] == "pipe_abc123"
    assert body["status"] == "queued"
    assert [j["step"] for j in body["jobs"]] == [1, 2]
    # built a pipeline of ordered load_domain steps
    steps = fake_mgr.submit_pipeline.call_args.args[0]
    assert [s["job_type"] for s in steps] == ["load_domain", "load_domain"]
    assert steps[0]["params"]["domain"] == "item"
    assert steps[1]["params"]["slice"] == "2026-04"


@pytest.mark.asyncio
async def test_submit_chain_unknown_domain_422(client, real_runner):
    resp = await client.post(
        "/integration/chains",
        json={"jobs": [{"domain": "totally_made_up", "mode": "delta"}]},
    )
    assert resp.status_code == 422
    assert "Unknown domain" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_chain_detail_aggregates_step_jobs(real_runner, client):
    cursor = real_runner
    plan = [{"step": 1, "domain": "item", "mode": "onetime", "slice": None},
            {"step": 2, "domain": "sales", "mode": "delta", "slice": None}]
    cursor.description = _PIPELINE_DESC
    cursor.fetchall.return_value = [
        ("pipe_1", 1, "job-a", "completed",
         {"domain": "item", "mode": "onetime", "__pipeline_plan": plan},
         {"rows_loaded": 10}, None, "2026-06-01T10:00:00", "2026-06-01T10:01:00", "ui"),
        ("pipe_1", 2, "job-b", "running",
         {"domain": "sales", "mode": "delta"}, None, None,
         "2026-06-01T10:01:00", None, "ui"),
    ]
    resp = await client.get("/integration/chains/pipe_1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "pipe_1"
    assert body["total_steps"] == 2
    assert body["status"] == "running"
    jobs = sorted(body["jobs"], key=lambda j: j["step"])
    assert jobs[0]["status"] == "success"   # completed -> success
    assert jobs[0]["rows_loaded"] == 10
    assert jobs[1]["status"] == "running"


@pytest.mark.asyncio
async def test_chain_detail_404_when_missing(real_runner, client):
    cursor = real_runner
    cursor.description = _PIPELINE_DESC
    # No pipeline rows; legacy fallback also returns no chain row.
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = None
    resp = await client.get("/integration/chains/nope")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_chain_list_shape_unchanged(real_runner, client):
    cursor = real_runner
    plan = [{"step": 1, "domain": "item", "mode": "onetime", "slice": None}]
    cursor.description = [*_PIPELINE_DESC, ("submitted_at",)]
    # 1st fetchall: pipeline rows; 2nd: legacy integration_chain rows (empty)
    cursor.fetchall.side_effect = [
        [("pipe_1", 1, "job-a", "completed",
          {"domain": "item", "mode": "onetime", "__pipeline_plan": plan},
          {"rows_loaded": 5}, None, "2026-06-01T10:00:00", "2026-06-01T10:01:00",
          "ui", "2026-06-01T10:00:00")],
        [],
    ]
    resp = await client.get("/integration/chains")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    item = items[0]
    # exact ChainSummary field set
    for key in ("id", "status", "total_steps", "completed_steps", "failed_step",
                "started_at", "completed_at", "duration_ms", "triggered_by"):
        assert key in item
    assert item["id"] == "pipe_1"
    assert item["status"] == "success"
    assert item["total_steps"] == 1


# ---------------------------------------------------------------------------
# GET /integration/scan — unchanged by US17d
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scan_endpoint_unchanged(client):
    payload = {
        "scanned_at": "2026-06-01T00:00:00",
        "changes": [{"domain": "sales", "kind": "modified"}],
        "proposed_chain": [{"step": 1, "domain": "sales", "mode": "delta"}],
    }
    with patch(
        "api.routers.platform.integration_chain.scan_input_dir",
        return_value=payload,
    ):
        resp = await client.get("/integration/scan")
    assert resp.status_code == 200
    body = resp.json()
    assert body["scanned_at"] == "2026-06-01T00:00:00"
    assert body["proposed_chain"][0]["domain"] == "sales"
