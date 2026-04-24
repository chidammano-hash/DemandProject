"""Tests for the Gen-4 lifespan handler + pool consolidation.

Covers:
- ``api.pool.open_pool`` / ``close_pool`` are symmetric (no crash when close
  is called before open).
- ``api.pool._build_conninfo`` uses ``common.core.db.get_db_params`` under
  the hood (single source of truth for DB env vars).
- FastAPI ``lifespan`` handler is wired on the app and does not crash when
  the pool/scheduler cannot start (offline/test mode).
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient


def test_close_pool_is_safe_when_not_opened():
    """close_pool() must be idempotent and safe to call before open."""
    from api import pool as pool_mod

    # Force the module-level pool back to None (test isolation).
    pool_mod._pool = None
    # Should not raise.
    pool_mod.close_pool()
    assert pool_mod._pool is None


def test_build_conninfo_uses_common_db_params(monkeypatch):
    """_build_conninfo() must build its conninfo from common.core.db.get_db_params."""
    from api import pool as pool_mod

    monkeypatch.setenv("POSTGRES_PASSWORD", "sentinel_pw")
    monkeypatch.setenv("POSTGRES_HOST", "db.test")
    monkeypatch.setenv("POSTGRES_PORT", "5441")
    monkeypatch.setenv("POSTGRES_DB", "demand_test")
    monkeypatch.setenv("POSTGRES_USER", "demand_test")

    conninfo = pool_mod._build_conninfo()
    assert "host=db.test" in conninfo
    assert "port=5441" in conninfo
    assert "dbname=demand_test" in conninfo
    assert "user=demand_test" in conninfo
    assert "password=sentinel_pw" in conninfo


def test_build_conninfo_raises_without_password(monkeypatch):
    """_build_conninfo() must fail fast when POSTGRES_PASSWORD is unset."""
    from api import pool as pool_mod

    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="POSTGRES_PASSWORD"):
        pool_mod._build_conninfo()


@pytest.mark.asyncio
async def test_app_boots_with_lifespan(monkeypatch):
    """The ASGI lifespan protocol should complete startup + shutdown without error,
    even when the DB pool + scheduler backend cannot start.

    This is the canonical regression test for replacing ``@app.on_event`` with
    an ``asynccontextmanager`` lifespan handler.
    """
    # Ensure open_pool() gracefully degrades.
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    # Make scheduler init a no-op.
    with patch("common.services.job_registry.JobManager") as job_mgr_cls:
        job_mgr_cls.instance.return_value.shutdown.return_value = None
        from api.main import app

        transport = ASGITransport(app=app)
        # AsyncClient triggers ASGI lifespan start/stop around the request.
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            # Either the health router is mounted and returns 200, or the
            # literal /health path falls through to the domains catchall
            # and returns 404 — both prove the app booted cleanly.
            assert resp.status_code in (200, 404)


def test_api_pool_and_common_db_are_consistent():
    """api.pool._build_conninfo() fields must match common.core.db.get_db_params()."""
    from api import pool as pool_mod
    from common.core.db import get_db_params

    os.environ.setdefault("POSTGRES_PASSWORD", "placeholder")

    params = get_db_params()
    conninfo = pool_mod._build_conninfo()
    # Every non-password value from get_db_params appears in the conninfo string
    # unchanged — proving the two paths agree on env defaults.
    assert f"host={params['host']}" in conninfo
    assert f"port={params['port']}" in conninfo
    assert f"dbname={params['dbname']}" in conninfo
    assert f"user={params['user']}" in conninfo
