"""Tests for /admin/* endpoints (Gen-4 Stream J).

Covers:
- POST /admin/llm/reset       — closes + clears singleton clients
- POST /admin/tuning/invalidate-stale
                              — no-op path when ``stale`` column absent
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_admin_reset_llm_no_existing_client(async_client):
    """POST /admin/llm/reset returns ok and does nothing when no client exists."""
    import api.llm as llm_mod

    # Ensure both singletons are None so nothing to close.
    llm_mod._openai_client = None
    llm_mod._anthropic_client = None

    resp = await async_client.post("/admin/llm/reset")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["openai_reset"] is False
    assert body["anthropic_reset"] is False


@pytest.mark.asyncio
async def test_admin_reset_llm_closes_clients(async_client):
    """POST /admin/llm/reset closes existing clients and clears singletons."""
    import api.llm as llm_mod

    openai_stub = MagicMock()
    anthropic_stub = MagicMock()
    llm_mod._openai_client = openai_stub
    llm_mod._anthropic_client = anthropic_stub

    try:
        resp = await async_client.post("/admin/llm/reset")
        assert resp.status_code == 200
        body = resp.json()
        assert body["openai_reset"] is True
        assert body["anthropic_reset"] is True
        # Both clients should have been close()d
        openai_stub.close.assert_called_once()
        anthropic_stub.close.assert_called_once()
        # Singletons cleared
        assert llm_mod._openai_client is None
        assert llm_mod._anthropic_client is None
    finally:
        llm_mod._openai_client = None
        llm_mod._anthropic_client = None


@pytest.mark.asyncio
async def test_admin_reset_llm_tolerates_close_errors(async_client):
    """close() raising OSError is logged but does not fail the request."""
    import api.llm as llm_mod

    broken = MagicMock()
    broken.close.side_effect = OSError("pipe closed")
    llm_mod._openai_client = broken
    llm_mod._anthropic_client = None

    try:
        resp = await async_client.post("/admin/llm/reset")
        assert resp.status_code == 200
        body = resp.json()
        assert body["openai_reset"] is True
        assert llm_mod._openai_client is None
    finally:
        llm_mod._openai_client = None
        llm_mod._anthropic_client = None


@pytest.mark.asyncio
async def test_admin_invalidate_stale_noop_when_table_absent(mock_pool, async_client):
    """When cluster_tuning_profile_state is absent, endpoint is a no-op."""
    _, _, cursor = mock_pool
    # First fetchone() = information_schema lookup, returns None (table missing)
    cursor.fetchone.return_value = None

    resp = await async_client.post("/admin/tuning/invalidate-stale")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "noop"
    assert body["invalidated"] == 0
    assert "cluster_tuning_profile_state" in body["reason"]


@pytest.mark.asyncio
async def test_admin_invalidate_stale_clears_flags(mock_pool, async_client):
    """Default mode clears stale flags on the state table."""
    _, _, cursor = mock_pool
    cursor.fetchone.return_value = (1,)  # information_schema -> table exists
    cursor.fetchall.return_value = [("L2_1",), ("L2_3",), ("L2_7",)]
    cursor.rowcount = 3

    resp = await async_client.post("/admin/tuning/invalidate-stale")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["invalidated"] == 3
    assert body["stale_clusters"] == ["L2_1", "L2_3", "L2_7"]
    update_sql = " ".join(
        str(c.args[0]) for c in cursor.execute.call_args_list if "UPDATE" in str(c.args[0])
    )
    assert "cluster_tuning_profile_state" in update_sql


@pytest.mark.asyncio
async def test_admin_invalidate_stale_nothing_stale(mock_pool, async_client):
    """No stale rows -> ok with zero invalidated, no UPDATE issued."""
    _, _, cursor = mock_pool
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = []

    resp = await async_client.post("/admin/tuning/invalidate-stale")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["invalidated"] == 0
    assert body["stale_clusters"] == []


@pytest.mark.asyncio
async def test_admin_invalidate_stale_retune_submits_job(mock_pool, async_client):
    """retune=true submits the tune_stale_clusters job and keeps flags set."""
    _, _, cursor = mock_pool
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [("L2_1",), ("L2_3",)]

    with patch("common.services.job_registry.JobManager") as manager_cls:
        manager_cls.return_value.submit_job.return_value = "job_retune_1"
        resp = await async_client.post("/admin/tuning/invalidate-stale?retune=true")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "retune_submitted"
    assert body["job_id"] == "job_retune_1"
    assert body["stale_clusters"] == ["L2_1", "L2_3"]
    submit_args = manager_cls.return_value.submit_job.call_args
    assert submit_args.args[0] == "tune_stale_clusters"
    assert submit_args.args[1] == {"model": "lgbm"}
    # Flags must NOT be cleared here — the tuning script clears them on success.
    update_calls = [
        c for c in cursor.execute.call_args_list if "UPDATE" in str(c.args[0])
    ]
    assert not update_calls


@pytest.mark.asyncio
async def test_admin_invalidate_stale_db_error_degrades(mock_pool, async_client):
    """DB errors degrade to a no-op response (status 200, status=noop)."""
    import psycopg

    _, _, cursor = mock_pool
    cursor.execute.side_effect = psycopg.Error("boom")

    resp = await async_client.post("/admin/tuning/invalidate-stale")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "noop"
    assert "db_error" in body["reason"]
    assert body["invalidated"] == 0
