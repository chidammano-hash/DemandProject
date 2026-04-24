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
async def test_admin_invalidate_stale_noop_when_column_absent(mock_pool, async_client):
    """When the cluster_tuning_profile.stale column is absent, endpoint is a no-op."""
    _, _, cursor = mock_pool
    # First fetchone() = information_schema lookup, returns None (column missing)
    cursor.fetchone.return_value = None

    resp = await async_client.post("/admin/tuning/invalidate-stale")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "noop"
    assert body["invalidated"] == 0
    assert "stale column" in body["reason"]


@pytest.mark.asyncio
async def test_admin_invalidate_stale_ok_when_column_present(mock_pool, async_client):
    """When stale column exists, endpoint counts + clears stale rows."""
    _, _, cursor = mock_pool
    # 1st fetchone: information_schema -> (1,) (column exists)
    # 2nd fetchone: count(*) -> (3,)
    cursor.fetchone.side_effect = [(1,), (3,)]

    resp = await async_client.post("/admin/tuning/invalidate-stale")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["invalidated"] == 3


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
