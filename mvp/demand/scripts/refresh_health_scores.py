"""IPfeature6: Refresh mv_inventory_health_score materialized view.

Executes REFRESH MATERIALIZED VIEW CONCURRENTLY mv_inventory_health_score
and logs timing and row count.

Usage:
    uv run python scripts/refresh_health_scores.py
"""
from __future__ import annotations

import os
import time

import psycopg
from dotenv import load_dotenv


def refresh(conn_params: dict) -> dict:
    """Refresh the health score view. Returns timing and row count."""
    start = time.perf_counter()

    with psycopg.connect(**conn_params) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            row = cur.execute(
                "SELECT relispopulated FROM pg_class WHERE relname = 'mv_inventory_health_score'"
            ).fetchone()
            concurrently = "CONCURRENTLY " if row and row[0] else ""
            cur.execute(f"REFRESH MATERIALIZED VIEW {concurrently}mv_inventory_health_score")
            cur.execute("SELECT COUNT(*) FROM mv_inventory_health_score")
            row_count = cur.fetchone()[0] or 0

    elapsed = round(time.perf_counter() - start, 2)
    return {"rows": int(row_count), "elapsed_s": elapsed}


def main() -> None:
    load_dotenv()

    conn_params = {
        "host":     os.getenv("POSTGRES_HOST", "localhost"),
        "port":     int(os.getenv("POSTGRES_PORT", "5440")),
        "dbname":   os.getenv("POSTGRES_DB", "demand_mvp"),
        "user":     os.getenv("POSTGRES_USER", "demand"),
        "password": os.getenv("POSTGRES_PASSWORD", "demand"),
    }

    print("[health] Refreshing mv_inventory_health_score…")
    result = refresh(conn_params)
    print(f"[health] Done — {result['rows']:,} rows in {result['elapsed_s']}s")


if __name__ == "__main__":
    main()
