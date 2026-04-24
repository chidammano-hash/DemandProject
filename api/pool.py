"""Connection pool management for the Supply Chain Command Center API.

Provides lazy-initialized PostgreSQL connection pool via psycopg3 + psycopg_pool.

This module is the single authoritative source for pool configuration used by
the FastAPI app. It delegates the per-field environment-variable defaults to
``common.core.db.get_db_params`` so there is one source of truth for DB creds
across the codebase.
"""
from __future__ import annotations

import os

from psycopg_pool import ConnectionPool

from common.core.db import get_db_params


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


def _create_pool() -> ConnectionPool:
    """Create a new ConnectionPool using the shared conninfo + resilience settings."""
    # Production resilience settings:
    # - timeout=10: fail fast (10s) if all connections are busy rather than blocking indefinitely
    # - max_lifetime=3600: recycle connections every hour to avoid stale/leaked server-side state
    # - reconnect_timeout=5: retry failed backend connections every 5s to recover from transient DB restarts
    return ConnectionPool(
        _build_conninfo(),
        min_size=2,
        max_size=int(os.getenv("POOL_MAX_SIZE", "20")),
        open=True,
        timeout=10,
        max_lifetime=3600,
        reconnect_timeout=5,
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
