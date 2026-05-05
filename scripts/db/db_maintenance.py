"""PostgreSQL database maintenance script.

Runs ANALYZE, checks cache hit ratio, reports bloat, and manages retention.

Usage:
    python scripts/db/db_maintenance.py analyze         # Update planner statistics
    python scripts/db/db_maintenance.py health           # Full health report
    python scripts/db/db_maintenance.py retention         # Apply retention policies (dry-run)
    python scripts/db/db_maintenance.py retention --execute  # Actually drop old partitions
"""

import argparse
import logging
import sys
from datetime import date, timedelta

import psycopg
import yaml

from common.core.db import get_db_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_retention_config() -> dict:
    """Load retention policies from config."""
    try:
        with open("config/platform/db_maintenance_config.yaml") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("config/platform/db_maintenance_config.yaml not found, using defaults")
        return {
            "retention_policies": {
                "fact_customer_demand_monthly": {"months": 36, "type": "partition"},
                "fact_inventory_snapshot": {"months": 24, "type": "partition"},
                "job_history": {"months": 3, "type": "delete", "date_column": "submitted_at"},
                "ai_call_log": {"months": 2, "type": "delete", "date_column": "created_at"},
                "fact_audit_log": {"months": 12, "type": "delete", "date_column": "created_at"},
                "fact_query_performance": {"months": 1, "type": "delete", "date_column": "created_at"},
            }
        }


def run_analyze(conn: psycopg.Connection) -> None:
    """Run ANALYZE on all user tables to update planner statistics."""
    logger.info("Running ANALYZE on all tables...")

    with conn.cursor() as cur:
        cur.execute(
            "SELECT schemaname, tablename FROM pg_tables "
            "WHERE schemaname = 'public' ORDER BY tablename"
        )
        tables = cur.fetchall()

    analyzed = 0
    for schema, table in tables:
        try:
            conn.execute(f'ANALYZE "{schema}"."{table}"')
            analyzed += 1
        except psycopg.Error as e:
            logger.warning("  ANALYZE %s.%s failed: %s", schema, table, e)

    logger.info("  Analyzed %d / %d tables", analyzed, len(tables))


def run_health_report(conn: psycopg.Connection) -> None:
    """Print a comprehensive database health report."""
    logger.info("=" * 70)
    logger.info("DATABASE HEALTH REPORT")
    logger.info("=" * 70)

    with conn.cursor() as cur:
        # Database size
        cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
        db_size = cur.fetchone()[0]
        logger.info("  Database size:      %s", db_size)

        # Cache hit ratio
        cur.execute("""
            SELECT
                CASE WHEN blks_read + blks_hit > 0
                     THEN round(100.0 * blks_hit / (blks_read + blks_hit), 2)
                     ELSE 0 END AS cache_hit_pct,
                blks_read, blks_hit
            FROM pg_stat_database
            WHERE datname = current_database()
        """)
        row = cur.fetchone()
        cache_pct, blks_read, blks_hit = row
        status = "OK" if cache_pct >= 90 else "WARNING" if cache_pct >= 70 else "CRITICAL"
        logger.info("  Cache hit ratio:    %.1f%% [%s]", cache_pct, status)

        # Connection usage
        cur.execute("""
            SELECT count(*) as active,
                   (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') as max_conn
            FROM pg_stat_activity
            WHERE datname = current_database()
        """)
        active, max_conn = cur.fetchone()
        logger.info("  Connections:        %d / %d", active, max_conn)

        # Temp file usage
        cur.execute("""
            SELECT temp_files, pg_size_pretty(temp_bytes) as temp_size
            FROM pg_stat_database WHERE datname = current_database()
        """)
        temp_files, temp_size = cur.fetchone()
        logger.info("  Temp files/size:    %d files / %s", temp_files, temp_size)

        # Table bloat (dead tuples)
        cur.execute("""
            SELECT count(*) as bloated_tables,
                   coalesce(sum(n_dead_tup), 0) as total_dead
            FROM pg_stat_user_tables
            WHERE n_dead_tup > 1000
        """)
        bloated, total_dead = cur.fetchone()
        logger.info("  Tables with bloat:  %d (total dead tuples: %d)", bloated, total_dead)

        # Unused indexes
        cur.execute("""
            SELECT count(*) as unused_count,
                   pg_size_pretty(coalesce(sum(pg_relation_size(indexrelid)), 0)) as unused_size
            FROM pg_stat_user_indexes
            WHERE idx_scan = 0
              AND indexrelname NOT LIKE '%%_pkey'
              AND schemaname = 'public'
        """)
        unused_count, unused_size = cur.fetchone()
        logger.info("  Unused indexes:     %d (%s)", unused_count, unused_size)

        # Top 10 tables by size
        logger.info("")
        logger.info("  TOP 10 TABLES BY SIZE:")
        cur.execute("""
            SELECT relname,
                   pg_size_pretty(pg_total_relation_size(relid)) as total,
                   n_live_tup as rows
            FROM pg_stat_user_tables
            ORDER BY pg_total_relation_size(relid) DESC
            LIMIT 10
        """)
        for tbl, size, rows in cur.fetchall():
            logger.info("    %-45s %10s  %12d rows", tbl, size, rows)

        # Sequential scan hotspots
        logger.info("")
        logger.info("  SEQ SCAN HOTSPOTS (tables scanned >10x without index):")
        cur.execute("""
            SELECT relname, seq_scan, idx_scan, seq_tup_read,
                   pg_size_pretty(pg_relation_size(relid)) as size
            FROM pg_stat_user_tables
            WHERE seq_scan > 10 AND (idx_scan = 0 OR seq_scan > idx_scan * 2)
              AND n_live_tup > 10000
            ORDER BY seq_tup_read DESC
            LIMIT 10
        """)
        rows = cur.fetchall()
        if rows:
            for tbl, seq, idx, tup, size in rows:
                logger.info(
                    "    %-40s seq=%d idx=%d tup_read=%d (%s)",
                    tbl, seq, idx, tup, size,
                )
        else:
            logger.info("    (none)")

        # Long-running queries
        logger.info("")
        cur.execute("""
            SELECT pid, now() - query_start AS duration, left(query, 80) as query_snippet
            FROM pg_stat_activity
            WHERE (now() - query_start) > interval '5 seconds'
              AND state != 'idle'
            ORDER BY duration DESC
            LIMIT 5
        """)
        long_queries = cur.fetchall()
        if long_queries:
            logger.info("  LONG-RUNNING QUERIES:")
            for pid, dur, snippet in long_queries:
                logger.info("    PID %d (%s): %s", pid, dur, snippet)
        else:
            logger.info("  Long-running queries: none")

    logger.info("=" * 70)


def run_retention(conn: psycopg.Connection, *, execute: bool = False) -> None:
    """Apply data retention policies."""
    config = load_retention_config()
    policies = config.get("retention_policies", {})

    logger.info("=" * 70)
    logger.info("RETENTION POLICY APPLICATION%s", "" if execute else " (DRY RUN)")
    logger.info("=" * 70)

    for table, policy in policies.items():
        months = policy["months"]
        cutoff = date.today().replace(day=1) - timedelta(days=months * 30)
        policy_type = policy["type"]

        logger.info("  Table: %-40s  retention: %d months  cutoff: %s", table, months, cutoff)

        if policy_type == "partition":
            _apply_partition_retention(conn, table, cutoff, execute=execute)
        elif policy_type == "delete":
            date_col = policy["date_column"]
            _apply_delete_retention(conn, table, date_col, cutoff, execute=execute)

    if not execute:
        logger.info("")
        logger.info("  DRY RUN -- re-run with --execute to apply retention")


def _apply_partition_retention(
    conn: psycopg.Connection, parent: str, cutoff: date, *, execute: bool
) -> None:
    """Drop partitions older than cutoff date."""
    import re

    with conn.cursor() as cur:
        cur.execute("""
            SELECT child.relname
            FROM pg_inherits
            JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
            JOIN pg_class child ON pg_inherits.inhrelid = child.oid
            WHERE parent.relname = %s
            ORDER BY child.relname
        """, (parent,))
        partitions = [row[0] for row in cur.fetchall()]

    for part_name in partitions:
        if part_name.endswith("_default"):
            continue
        match = re.search(r"(\d{4})_(\d{2})$", part_name)
        if not match:
            continue
        part_date = date(int(match.group(1)), int(match.group(2)), 1)
        if part_date < cutoff:
            if execute:
                logger.info("    DROP TABLE %s (partition date: %s)", part_name, part_date)
                conn.execute(f"DROP TABLE IF EXISTS {part_name}")
            else:
                logger.info("    WOULD DROP %s (partition date: %s)", part_name, part_date)


def _apply_delete_retention(
    conn: psycopg.Connection,
    table: str,
    date_col: str,
    cutoff: date,
    *,
    execute: bool,
) -> None:
    """Delete rows older than cutoff date."""
    with conn.cursor() as cur:
        cur.execute(
            f'SELECT count(*) FROM "{table}" WHERE "{date_col}" < %s',
            (cutoff,),
        )
        count = cur.fetchone()[0]

    if count == 0:
        logger.info("    No rows to delete (0 rows before %s)", cutoff)
        return

    if execute:
        logger.info("    DELETE %d rows from %s where %s < %s", count, table, date_col, cutoff)
        conn.execute(
            f'DELETE FROM "{table}" WHERE "{date_col}" < %s', (cutoff,)
        )
        conn.execute(f'VACUUM ANALYZE "{table}"')
    else:
        logger.info("    WOULD DELETE %d rows from %s where %s < %s", count, table, date_col, cutoff)


def main() -> None:
    parser = argparse.ArgumentParser(description="PostgreSQL database maintenance")
    parser.add_argument(
        "action",
        choices=["analyze", "health", "retention"],
        help="Maintenance action to perform",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually apply changes (retention only)",
    )
    args = parser.parse_args()

    db_params = get_db_params()

    with psycopg.connect(**db_params, autocommit=True) as conn:
        if args.action == "analyze":
            run_analyze(conn)
        elif args.action == "health":
            run_health_report(conn)
        elif args.action == "retention":
            run_retention(conn, execute=args.execute)


if __name__ == "__main__":
    main()
