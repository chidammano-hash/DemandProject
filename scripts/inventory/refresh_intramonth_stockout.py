"""
IPfeature14 — Intra-Month Stockout Detection refresh script.

Refreshes mv_intramonth_stockout materialized view.

Usage:
    uv run python scripts/refresh_intramonth_stockout.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import psycopg
from common.db import get_db_params  # noqa: E402


def _safe_refresh(conn: psycopg.Connection, view: str) -> None:
    """Refresh a materialized view; fall back to non-concurrent on first population."""
    conn.autocommit = True
    try:
        conn.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}")
    except psycopg.errors.FeatureNotSupported:
        conn.execute(f"REFRESH MATERIALIZED VIEW {view}")


def run() -> None:
    print("Refreshing mv_intramonth_stockout...")
    with psycopg.connect(**get_db_params()) as conn:
        _safe_refresh(conn, "mv_intramonth_stockout")
    print("Done — mv_intramonth_stockout refreshed.")


if __name__ == "__main__":
    run()
