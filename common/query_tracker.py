"""Query performance tracking middleware (Spec 08-03).

Logs slow API responses to fact_query_performance for monitoring.
"""
from __future__ import annotations

import threading
import time
from typing import Any


class QueryTracker:
    """Tracks API endpoint performance."""

    def __init__(self, slow_threshold_ms: float = 500.0):
        self.slow_threshold_ms = slow_threshold_ms
        self._slow_queries: list[dict] = []

    def record(
        self,
        endpoint: str,
        method: str,
        duration_ms: float,
        cache_hit: bool = False,
        user_id: str | None = None,
        params: dict | None = None,
    ) -> None:
        """Record a query performance entry. Writes to DB for slow queries."""
        if duration_ms >= self.slow_threshold_ms:
            entry = {
                "endpoint": endpoint,
                "method": method,
                "duration_ms": round(duration_ms, 2),
                "cache_hit": cache_hit,
                "user_id": user_id,
                "params": params,
            }
            self._slow_queries.append(entry)
            self._write_to_db(entry)

    def _write_to_db(self, entry: dict) -> None:
        """Persist slow query record. Best-effort."""
        try:
            import json
            from api.core import get_conn
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO fact_query_performance
                       (endpoint, method, duration_ms, cache_hit, user_id, params)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (
                        entry["endpoint"],
                        entry["method"],
                        entry["duration_ms"],
                        entry["cache_hit"],
                        entry.get("user_id"),
                        json.dumps(entry.get("params")) if entry.get("params") else None,
                    ),
                )
                conn.commit()
        except Exception:
            pass

    def get_recent_slow(self, limit: int = 50) -> list[dict]:
        """Return recent slow queries from memory."""
        return list(reversed(self._slow_queries[-limit:]))


# Singleton (thread-safe via double-checked locking)
_tracker: QueryTracker | None = None
_tracker_lock = threading.Lock()


def get_tracker() -> QueryTracker:
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = QueryTracker()
    return _tracker
