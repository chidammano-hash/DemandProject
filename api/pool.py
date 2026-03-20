"""Connection pool management for the Supply Chain Command Center API.

Provides lazy-initialized PostgreSQL connection pool via psycopg3 + psycopg_pool.
"""
from __future__ import annotations

import os

from psycopg_pool import ConnectionPool


# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------
_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        password = os.environ.get("POSTGRES_PASSWORD")
        if not password:
            raise RuntimeError("Required environment variable 'POSTGRES_PASSWORD' is not set.")
        conninfo = (
            f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
            f"port={os.getenv('POSTGRES_PORT', '5440')} "
            f"dbname={os.getenv('POSTGRES_DB', 'demand_mvp')} "
            f"user={os.getenv('POSTGRES_USER', 'demand')} "
            f"password={password}"
        )
        # Production resilience settings:
        # - timeout=10: fail fast (10s) if all connections are busy rather than blocking indefinitely
        # - max_lifetime=3600: recycle connections every hour to avoid stale/leaked server-side state
        # - reconnect_timeout=5: retry failed backend connections every 5s to recover from transient DB restarts
        _pool = ConnectionPool(
            conninfo,
            min_size=2,
            max_size=10,
            open=True,
            timeout=10,
            max_lifetime=3600,
            reconnect_timeout=5,
        )
    return _pool


def get_conn():
    return _get_pool().connection()
