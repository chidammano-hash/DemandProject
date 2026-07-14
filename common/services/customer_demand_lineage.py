"""Cross-process locking for exact customer-demand snapshot consumers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")

CUSTOMER_DEMAND_LOAD_LOCK_KEY = "customer_demand_load_and_profile_refresh"


@contextmanager
def customer_demand_snapshot_lock(conn: Any) -> Iterator[None]:
    """Hold the shared session lock across a caller-owned snapshot transaction."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_lock_shared(hashtext(%s))",
            (CUSTOMER_DEMAND_LOAD_LOCK_KEY,),
        )
    conn.commit()
    try:
        yield
    finally:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_unlock_shared(hashtext(%s))",
                (CUSTOMER_DEMAND_LOAD_LOCK_KEY,),
            )
        conn.commit()


def customer_demand_snapshot_locked(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Hold a shared session lock around a committed, consistent consumer run.

    The session lock is acquired and committed before the wrapped function
    opens its repeatable-read snapshot. If a loader currently owns the
    conflicting exclusive lock, the subsequent snapshot therefore starts only
    after that loader has published its refreshed batch marker.
    """

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        conn: Any = args[0] if args else kwargs.get("conn")
        if conn is None:
            raise TypeError("Customer-demand snapshot consumer requires a connection")
        with customer_demand_snapshot_lock(conn):
            result = func(*args, **kwargs)
            conn.commit()
            return result

    return wrapper
