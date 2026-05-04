"""IPfeature6: Refresh mv_inventory_health_score materialized view.

Executes REFRESH MATERIALIZED VIEW CONCURRENTLY mv_inventory_health_score
and logs timing and row count.

Usage:
    uv run python scripts/refresh_health_scores.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.db import get_db_params


_DEPENDENCIES = ("agg_inventory_monthly", "mv_inventory_forecast_monthly")


def _refresh_mv(cur: psycopg.Cursor, name: str) -> None:
    row = cur.execute(
        "SELECT relispopulated FROM pg_class WHERE relname = %s", (name,)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"materialized view {name!r} does not exist — apply DDL first")
    concurrently = "CONCURRENTLY " if row[0] else ""
    cur.execute(f"REFRESH MATERIALIZED VIEW {concurrently}{name}")


def refresh(conn_params: dict) -> dict:
    """Refresh the health score view. Returns timing and row count."""
    start = time.perf_counter()

    with psycopg.connect(**conn_params) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for dep in _DEPENDENCIES:
                row = cur.execute(
                    "SELECT relispopulated FROM pg_class WHERE relname = %s", (dep,)
                ).fetchone()
                if row is not None and not row[0]:
                    print(f"[health] Dependency {dep} not populated — refreshing first…")
                    _refresh_mv(cur, dep)

            _refresh_mv(cur, "mv_inventory_health_score")
            cur.execute("SELECT COUNT(*) FROM mv_inventory_health_score")
            row_count = cur.fetchone()[0] or 0

    elapsed = round(time.perf_counter() - start, 2)
    return {"rows": int(row_count), "elapsed_s": elapsed}


def main() -> None:
    load_dotenv(ROOT / ".env")

    print("[health] Refreshing mv_inventory_health_score…")
    result = refresh(get_db_params())
    print(f"[health] Done — {result['rows']:,} rows in {result['elapsed_s']}s")


if __name__ == "__main__":
    main()
