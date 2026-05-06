"""Connection pool management for the Supply Chain Command Center API.

Provides lazy-initialized PostgreSQL connection pool via psycopg3 + psycopg_pool.

This module is the single authoritative source for pool configuration used by
the FastAPI app. It delegates the per-field environment-variable defaults to
``common.core.db.get_db_params`` so there is one source of truth for DB creds
across the codebase.

Both a sync (:class:`ConnectionPool`) and async (:class:`AsyncConnectionPool`)
pool are supported. The async pool is used by routers converted to ``async
def`` handlers (Item 19 pilot — customer_analytics + GET endpoints in
inv_planning_insights). They share the same conninfo + sizing knobs but are
independent pool instances.
"""
from __future__ import annotations

import logging
import os

from psycopg import AsyncConnection, Connection
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from common.core.db import get_db_params, get_read_replica_params

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection pools
# ---------------------------------------------------------------------------
# Primary (read/write) pool — always created.
_pool: ConnectionPool | None = None
_async_pool: AsyncConnectionPool | None = None
# Read-replica pool — lazily created only when ``READ_REPLICA_URL`` is set.
# Stays ``None`` for the entire process lifetime when the env var is unset
# (the common single-node case). Callers MUST go through
# :func:`api.core.get_read_only_conn` so the fall-back-to-primary logic
# stays in one place.
_read_pool: ConnectionPool | None = None


def _build_conninfo() -> str:
    """Build a psycopg-style conninfo string from ``get_db_params()``.

    Raises RuntimeError if ``POSTGRES_PASSWORD`` is unset — the pool
    cannot connect to a live DB without it and we prefer failing fast
    on process start over opaque connection errors at request time.
    """
    if not os.environ.get("POSTGRES_PASSWORD"):
        raise RuntimeError("Required environment variable 'POSTGRES_PASSWORD' is not set.")
    params = get_db_params()
    return (
        f"host={params['host']} "
        f"port={params['port']} "
        f"dbname={params['dbname']} "
        f"user={params['user']} "
        f"password={params['password']}"
    )


def _get_pool() -> ConnectionPool:
    """Return the lazily-created process-wide connection pool."""
    global _pool
    if _pool is None:
        _pool = _create_pool()
    return _pool


def _configure_connection(conn: Connection) -> None:
    """Run once per new pool connection. Sets a session-level statement_timeout
    so a runaway query can't pin a pool slot forever — it's killed at 30s and
    the slot returns to the pool. Override with PG_STATEMENT_TIMEOUT_MS.
    """
    timeout_ms = int(os.getenv("PG_STATEMENT_TIMEOUT_MS", "30000"))
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {timeout_ms}")
        conn.commit()
    except Exception as exc:  # noqa: BLE001 — never block pool init on this
        logger.warning("Failed to set statement_timeout on new connection: %s", exc)


def _create_pool() -> ConnectionPool:
    """Create a new ConnectionPool using the shared conninfo + resilience settings."""
    # Production resilience settings:
    # - timeout=10: fail fast (10s) if all connections are busy rather than blocking indefinitely
    # - max_lifetime=3600: recycle connections every hour to avoid stale/leaked server-side state
    # - reconnect_timeout=5: retry failed backend connections every 5s to recover from transient DB restarts
    # - configure: applies SET statement_timeout once per new backend connection
    # max_size sized for the Customer Analytics tab: 13 concurrent endpoints
    # × ~3 simultaneous planners + headroom. Below that, the 14th request
    # waits up to `timeout` seconds before erroring. Note that with N gunicorn
    # workers, total backend connections = N × max_size — keep that under
    # Postgres `max_connections` (default 100).
    return ConnectionPool(
        _build_conninfo(),
        min_size=int(os.getenv("POOL_MIN_SIZE", "5")),
        max_size=int(os.getenv("POOL_MAX_SIZE", "50")),
        open=True,
        timeout=10,
        max_lifetime=3600,
        reconnect_timeout=5,
        configure=_configure_connection,
    )


def open_pool() -> ConnectionPool:
    """Open (or return) the pool — called from the FastAPI lifespan handler on startup."""
    return _get_pool()


def close_pool() -> None:
    """Close the pool if it was opened. Safe to call when the pool is None."""
    global _pool
    if _pool is not None:
        try:
            _pool.close()
        except Exception:  # noqa: BLE001 — shutdown cleanup: never re-raise during teardown
            pass
        _pool = None


# ---------------------------------------------------------------------------
# Read-replica pool (Item 24 — opt-in routing for analytics endpoints)
#
# When ``READ_REPLICA_URL`` is set, analytics endpoints that opted in via
# :func:`api.core.get_read_only_conn` route to the replica; everything else
# (writes + non-opted-in reads) continues to use the primary pool. When the
# env var is unset, ``_read_replica_configured()`` returns ``False`` and
# every caller falls back to the primary pool — i.e. the configured-off
# behaviour is bit-for-bit identical to having no replica code at all.
# ---------------------------------------------------------------------------

def _read_replica_configured() -> bool:
    """Return True iff ``READ_REPLICA_URL`` is set and parses cleanly."""
    return get_read_replica_params() is not None


def _build_read_replica_conninfo() -> str | None:
    """Build a psycopg conninfo string for the read replica, or None if not configured."""
    params = get_read_replica_params()
    if params is None:
        return None
    return (
        f"host={params['host']} "
        f"port={params['port']} "
        f"dbname={params['dbname']} "
        f"user={params['user']} "
        f"password={params['password']}"
    )


def _create_read_pool() -> ConnectionPool:
    """Create a ConnectionPool for the read replica.

    Sized independently of the primary pool via ``READ_POOL_MIN_SIZE`` /
    ``READ_POOL_MAX_SIZE``; falls back to the primary pool sizing knobs
    so a default deployment that just sets ``READ_REPLICA_URL`` gets
    sensible defaults without extra tuning.
    """
    conninfo = _build_read_replica_conninfo()
    if conninfo is None:
        # Defensive: the public callers gate on _read_replica_configured()
        # before we get here. Keep the safety check so a future caller can't
        # silently build a None-conninfo pool.
        raise RuntimeError("READ_REPLICA_URL is not configured; cannot create read pool.")
    return ConnectionPool(
        conninfo,
        min_size=int(os.getenv("READ_POOL_MIN_SIZE", os.getenv("POOL_MIN_SIZE", "5"))),
        max_size=int(os.getenv("READ_POOL_MAX_SIZE", os.getenv("POOL_MAX_SIZE", "50"))),
        open=True,
        timeout=10,
        max_lifetime=3600,
        reconnect_timeout=5,
        configure=_configure_connection,
    )


def _get_read_pool() -> ConnectionPool:
    """Return the lazily-created read-replica pool. Caller must verify
    :func:`_read_replica_configured` first; otherwise this raises."""
    global _read_pool
    if _read_pool is None:
        _read_pool = _create_read_pool()
    return _read_pool


def close_read_pool() -> None:
    """Close the read-replica pool if it was opened. Safe to call when None."""
    global _read_pool
    if _read_pool is not None:
        try:
            _read_pool.close()
        except Exception:  # noqa: BLE001 — shutdown cleanup
            pass
        _read_pool = None


# Async sibling of the read pool — sized like the primary async pool. Lazily
# created on first use; stays None when ``READ_REPLICA_URL`` is unset.
_async_read_pool: AsyncConnectionPool | None = None


def _create_async_read_pool() -> AsyncConnectionPool:
    """Async sibling of :func:`_create_read_pool`."""
    conninfo = _build_read_replica_conninfo()
    if conninfo is None:
        raise RuntimeError("READ_REPLICA_URL is not configured; cannot create async read pool.")
    return AsyncConnectionPool(
        conninfo,
        min_size=int(os.getenv("READ_POOL_MIN_SIZE", os.getenv("ASYNC_POOL_MIN_SIZE", os.getenv("POOL_MIN_SIZE", "5")))),
        max_size=int(os.getenv("READ_POOL_MAX_SIZE", os.getenv("ASYNC_POOL_MAX_SIZE", os.getenv("POOL_MAX_SIZE", "50")))),
        open=False,
        timeout=10,
        max_lifetime=3600,
        reconnect_timeout=5,
        configure=_configure_async_connection,
    )


def _get_async_read_pool() -> AsyncConnectionPool:
    """Return the lazily-created async read-replica pool. Caller MUST verify
    :func:`_read_replica_configured` first; otherwise this raises.

    Like :func:`_get_async_pool`, this constructs but does NOT open the pool.
    The lifespan handler calls :func:`open_async_read_pool` to establish backend
    connections."""
    global _async_read_pool
    if _async_read_pool is None:
        _async_read_pool = _create_async_read_pool()
    return _async_read_pool


async def open_async_read_pool() -> AsyncConnectionPool | None:
    """Open the async read pool when configured; otherwise return None.

    Idempotent — safe to call multiple times. Errors during open are logged
    but never propagated, so a misconfigured replica cannot prevent app start
    (analytics endpoints fall through to the primary pool instead)."""
    if not _read_replica_configured():
        return None
    pool = _get_async_read_pool()
    try:
        await pool.open()
    except Exception as exc:  # noqa: BLE001 — never block app start on this
        logger.warning("Failed to open async read-replica pool: %s", exc)
    return pool


async def close_async_read_pool() -> None:
    """Close the async read pool on shutdown. Safe to call when None."""
    global _async_read_pool
    if _async_read_pool is not None:
        try:
            await _async_read_pool.close()
        except Exception:  # noqa: BLE001 — shutdown cleanup
            pass
        _async_read_pool = None


# ---------------------------------------------------------------------------
# Async connection pool (Item 19 pilot)
#
# Sized identically to the sync pool. Async handlers do NOT consume an anyio
# threadpool token, so the threadpool ceiling no longer caps Customer
# Analytics fan-out. Total backend connections = N gunicorn workers ×
# (sync max_size + async max_size); keep the sum under Postgres
# ``max_connections`` (default 100).
# ---------------------------------------------------------------------------

async def _configure_async_connection(conn: AsyncConnection) -> None:
    """Async sibling of ``_configure_connection`` — sets statement_timeout
    once per new pooled backend so a runaway query can't pin a slot."""
    timeout_ms = int(os.getenv("PG_STATEMENT_TIMEOUT_MS", "30000"))
    try:
        async with conn.cursor() as cur:
            await cur.execute(f"SET statement_timeout = {timeout_ms}")
        await conn.commit()
    except Exception as exc:  # noqa: BLE001 — never block pool init on this
        logger.warning("Failed to set statement_timeout on new async connection: %s", exc)


def _create_async_pool() -> AsyncConnectionPool:
    """Create a new AsyncConnectionPool with ``open=False`` so the lifespan
    handler can ``await pool.open()`` explicitly. Constructing with
    ``open=True`` from a synchronous context is deprecated in psycopg-pool
    3.2+."""
    return AsyncConnectionPool(
        _build_conninfo(),
        min_size=int(os.getenv("ASYNC_POOL_MIN_SIZE", os.getenv("POOL_MIN_SIZE", "5"))),
        max_size=int(os.getenv("ASYNC_POOL_MAX_SIZE", os.getenv("POOL_MAX_SIZE", "50"))),
        open=False,
        timeout=10,
        max_lifetime=3600,
        reconnect_timeout=5,
        configure=_configure_async_connection,
    )


def _get_async_pool() -> AsyncConnectionPool:
    """Return the lazily-created process-wide async connection pool.

    Note: lazily created but NOT opened here — call :func:`open_async_pool`
    from the lifespan handler to actually establish backend connections.
    """
    global _async_pool
    if _async_pool is None:
        _async_pool = _create_async_pool()
    return _async_pool


async def open_async_pool() -> AsyncConnectionPool:
    """Open (or return) the async pool. Called from the FastAPI lifespan
    handler on startup. Idempotent — safe to call multiple times."""
    pool = _get_async_pool()
    try:
        await pool.open()
    except Exception as exc:  # noqa: BLE001 — best effort
        logger.warning("Failed to open async DB pool: %s", exc)
    return pool


async def close_async_pool() -> None:
    """Close the async pool on shutdown. Safe to call when None."""
    global _async_pool
    if _async_pool is not None:
        try:
            await _async_pool.close()
        except Exception:  # noqa: BLE001 — shutdown cleanup
            pass
        _async_pool = None
