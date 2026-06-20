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


@pytest.mark.parametrize(
    ("factory_name", "pool_symbol", "expected_max", "env_override"),
    [
        # Sync primary pool — default 12, env-override POOL_MAX_SIZE.
        ("_create_pool", "ConnectionPool", 12, ("POOL_MAX_SIZE", "37")),
        # Async primary pool — independent default 20 (NOT inherited from
        # POOL_MAX_SIZE), env-override ASYNC_POOL_MAX_SIZE.
        ("_create_async_pool", "AsyncConnectionPool", 20, ("ASYNC_POOL_MAX_SIZE", "41")),
    ],
)
def test_primary_pool_independent_max_size_defaults(
    monkeypatch, factory_name, pool_symbol, expected_max, env_override
):
    """Each primary pool must default to its OWN independent max_size and honour
    its dedicated env override. The async pool default (20) must NOT track the
    sync POOL_MAX_SIZE default (12) — they are sized independently (P0-1 fix).
    """
    from api import pool as pool_mod

    monkeypatch.setenv("POSTGRES_PASSWORD", "sentinel_pw")
    # Clear all sizing knobs so the code-level defaults are exercised.
    for var in ("POOL_MAX_SIZE", "ASYNC_POOL_MAX_SIZE", "READ_POOL_MAX_SIZE", "POOL_MIN_SIZE"):
        monkeypatch.delenv(var, raising=False)

    captured: dict[str, int] = {}

    def _capture(*_args, **kwargs):
        captured["max_size"] = kwargs["max_size"]
        return object()  # don't build a real pool

    monkeypatch.setattr(pool_mod, pool_symbol, _capture)

    # Default path: the pool's own code-level default.
    getattr(pool_mod, factory_name)()
    assert captured["max_size"] == expected_max

    # Override path: the pool's dedicated env var wins.
    env_var, env_val = env_override
    monkeypatch.setenv(env_var, env_val)
    getattr(pool_mod, factory_name)()
    assert captured["max_size"] == int(env_val)


def test_async_pool_default_does_not_inherit_sync_pool_size(monkeypatch):
    """Setting POOL_MAX_SIZE must NOT change the async primary pool size — the
    two pools are independent. Regression guard for the old coupled default
    (async fell back to POOL_MAX_SIZE)."""
    from api import pool as pool_mod

    monkeypatch.setenv("POSTGRES_PASSWORD", "sentinel_pw")
    monkeypatch.delenv("ASYNC_POOL_MAX_SIZE", raising=False)
    monkeypatch.delenv("POOL_MIN_SIZE", raising=False)
    # Crank the SYNC knob; the async default must stay at 20.
    monkeypatch.setenv("POOL_MAX_SIZE", "99")

    captured: dict[str, int] = {}

    def _capture(*_args, **kwargs):
        captured["max_size"] = kwargs["max_size"]
        return object()

    monkeypatch.setattr(pool_mod, "AsyncConnectionPool", _capture)
    pool_mod._create_async_pool()
    assert captured["max_size"] == 20


def test_read_pool_default_max_size_is_twelve(monkeypatch):
    """The read-replica pool must default to max_size=12 (independent ceiling
    against the replica's own max_connections)."""
    from api import pool as pool_mod

    monkeypatch.setenv("POSTGRES_PASSWORD", "sentinel_pw")
    monkeypatch.setenv("READ_REPLICA_URL", "postgresql://demand:pw@replica:5432/demand_mvp")
    for var in ("READ_POOL_MAX_SIZE", "ASYNC_POOL_MAX_SIZE", "POOL_MAX_SIZE", "POOL_MIN_SIZE"):
        monkeypatch.delenv(var, raising=False)

    captured: dict[str, int] = {}

    def _capture(*_args, **kwargs):
        captured["max_size"] = kwargs["max_size"]
        return object()

    monkeypatch.setattr(pool_mod, "ConnectionPool", _capture)
    pool_mod._create_read_pool()
    assert captured["max_size"] == 12


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
