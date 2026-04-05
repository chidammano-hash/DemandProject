"""Drop unused indexes from PostgreSQL to reclaim disk space.

Identifies indexes with zero scans (never used by the query planner) and drops
them, excluding primary keys, unique constraints (needed for ON CONFLICT upserts),
and MLflow internal indexes.

Usage:
    python scripts/db/drop_unused_indexes.py              # Dry run (default)
    python scripts/db/drop_unused_indexes.py --execute     # Actually drop
    python scripts/db/drop_unused_indexes.py --min-size 1  # Only indexes >= 1 MB
"""

import argparse
import logging
import sys
from typing import Any

import psycopg

from common.core.db import get_db_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SKIP_PREFIXES = ("mlflow_",)

QUERY_UNUSED_INDEXES = """
SELECT
    s.schemaname,
    s.indexrelname AS index_name,
    s.relname     AS table_name,
    pg_relation_size(s.indexrelid) AS size_bytes,
    pg_size_pretty(pg_relation_size(s.indexrelid)) AS size_pretty,
    i.indisunique AS is_unique,
    i.indisprimary AS is_primary
FROM pg_stat_user_indexes s
JOIN pg_index i ON s.indexrelid = i.indexrelid
WHERE s.idx_scan = 0
  AND s.schemaname = 'public'
ORDER BY pg_relation_size(s.indexrelid) DESC;
"""


def _should_skip(row: dict[str, Any]) -> str | None:
    """Return skip reason or None if safe to drop."""
    if row["is_primary"]:
        return "primary key"
    if row["is_unique"]:
        return "unique constraint (needed for ON CONFLICT upserts)"
    for prefix in SKIP_PREFIXES:
        if row["index_name"].startswith(prefix):
            return f"matches skip prefix '{prefix}'"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop unused PostgreSQL indexes")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually drop indexes (default is dry-run)",
    )
    parser.add_argument(
        "--min-size",
        type=float,
        default=0,
        help="Minimum index size in MB to consider dropping (default: 0)",
    )
    args = parser.parse_args()

    db_params = get_db_params()

    with psycopg.connect(**db_params, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(QUERY_UNUSED_INDEXES)
            columns = [desc[0] for desc in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]

        if not rows:
            logger.info("No unused indexes found.")
            return

        total_reclaimable = 0
        drop_count = 0
        skip_count = 0
        skipped_unique_size = 0
        dropped_names: list[str] = []

        logger.info("=" * 70)
        logger.info("UNUSED INDEX ANALYSIS")
        logger.info("=" * 70)

        for row in rows:
            size_mb = row["size_bytes"] / (1024 * 1024)
            skip_reason = _should_skip(row)

            if skip_reason:
                skip_count += 1
                if row["is_unique"]:
                    skipped_unique_size += row["size_bytes"]
                if size_mb >= 100:
                    logger.info(
                        "  SKIP %-60s %8s  (%s)",
                        row["index_name"],
                        row["size_pretty"],
                        skip_reason,
                    )
                continue

            if size_mb < args.min_size:
                skip_count += 1
                continue

            total_reclaimable += row["size_bytes"]
            drop_count += 1

            if args.execute:
                drop_sql = (
                    f'DROP INDEX CONCURRENTLY IF EXISTS "{row["index_name"]}";'
                )
                logger.info(
                    "  DROP %-60s %8s",
                    row["index_name"],
                    row["size_pretty"],
                )
                try:
                    with conn.cursor() as cur:
                        cur.execute(drop_sql)
                    dropped_names.append(row["index_name"])
                except psycopg.Error as e:
                    logger.error(
                        "  FAIL %-60s %s", row["index_name"], e
                    )
            else:
                logger.info(
                    "  WOULD DROP %-55s %8s  (table: %s)",
                    row["index_name"],
                    row["size_pretty"],
                    row["table_name"],
                )

        logger.info("=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)
        pretty_total = f"{total_reclaimable / (1024**3):.1f} GB"
        pretty_unique = f"{skipped_unique_size / (1024**3):.1f} GB"
        logger.info("  Indexes to drop:    %d", drop_count)
        logger.info("  Indexes skipped:    %d (pkeys + unique constraints + mlflow)", skip_count)
        logger.info("  Space reclaimable:  %s", pretty_total)
        logger.info("  Unique idx kept:    %s (data integrity)", pretty_unique)

        if args.execute:
            logger.info("  Actually dropped:   %d indexes", len(dropped_names))
        else:
            logger.info("")
            logger.info("  DRY RUN — re-run with --execute to drop indexes")

        # Final DB size check
        with psycopg.connect(**db_params) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_size_pretty(pg_database_size(current_database()))"
                )
                db_size = cur.fetchone()[0]
                logger.info("  Current DB size:    %s", db_size)


if __name__ == "__main__":
    main()
