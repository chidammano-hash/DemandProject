"""Connection pool management for the Supply Chain Command Center API.

Provides lazy-initialized PostgreSQL connection pool via psycopg3 + psycopg_pool.

This module is the single authoritative source for pool configuration used by
the FastAPI app. It delegates the per-field environment-variable defaults to
``common.core.db.get_db_params`` so there is one source of truth for DB creds
across the codebase.
"""
from __future__ import annotations

import logging
import os

from psycopg import Connection
from psycopg_pool import ConnectionPool

from common.core.db import get_db_params

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------
_pool: ConnectionPool | None = None


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
