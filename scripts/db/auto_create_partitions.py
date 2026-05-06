"""Idempotently create the next N monthly partitions for partitioned fact tables.

Background
----------
Several fact tables in this project are RANGE-partitioned by a monthly date
column with concrete partitions provisioned up front in their migrations:

* ``fact_inventory_snapshot``        (sql/088, partition key ``snapshot_date``)
* ``fact_customer_demand_monthly``   (sql/110, partition key ``startdate``)
* ``fact_external_signal``           (sql/141, partition key ``event_ts``)

When the calendar advances past the last hardcoded partition, new rows fall
into the table's DEFAULT partition. That defeats partition pruning and slowly
turns the default partition into a giant grab-bag.

This script keeps a rolling window of ``--horizon-months`` (default 12)
monthly partitions ahead of today for every registered partitioned table.
It uses ``CREATE TABLE IF NOT EXISTS ... PARTITION OF`` so it is fully
idempotent — re-running it is always safe.

Run it monthly via cron, OR manually before any large backfill that may
write rows into a future month.

Usage
-----
    python scripts/db/auto_create_partitions.py                 # rolling 12 months
    python scripts/db/auto_create_partitions.py --horizon 6     # rolling 6 months
    python scripts/db/auto_create_partitions.py --dry-run       # print DDL only
    python scripts/db/auto_create_partitions.py --table fact_inventory_snapshot
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date

import psycopg

from common.core.db import get_db_params
from common.core.planning_date import get_planning_date

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry of monthly-partitioned fact tables.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PartitionedTable:
    """Definition of a monthly RANGE-partitioned table."""

    name: str
    """Parent table name (e.g. fact_inventory_snapshot)."""

    partition_prefix: str
    """Prefix used for child partition names. Children are named
    ``<prefix>_YYYY_MM``. Often equal to the parent name."""

    column_type: str
    """SQL type used in the FROM/TO bounds. ``date`` for date columns,
    ``timestamptz`` for timestamp columns."""


# Tables to manage. Add new partitioned tables here.
PARTITIONED_TABLES: tuple[PartitionedTable, ...] = (
    PartitionedTable(
        name="fact_inventory_snapshot",
        partition_prefix="fact_inventory_snapshot",
        column_type="date",
    ),
    PartitionedTable(
        name="fact_customer_demand_monthly",
        partition_prefix="fact_customer_demand_monthly",
        column_type="date",
    ),
    PartitionedTable(
        name="fact_external_signal",
        partition_prefix="fact_external_signal",
        column_type="timestamptz",
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_months(d: date, n: int) -> date:
    """Return the first day of the month ``n`` months after ``d``."""
    year = d.year + (d.month - 1 + n) // 12
    month = (d.month - 1 + n) % 12 + 1
    return date(year, month, 1)


def _format_bound(d: date, column_type: str) -> str:
    """Render a partition bound literal for the given column type."""
    iso = d.isoformat()
    if column_type == "timestamptz":
        return f"TIMESTAMPTZ '{iso} 00:00:00+00'"
    return f"DATE '{iso}'"


def build_partition_ddl(
    table: PartitionedTable,
    month_start: date,
) -> tuple[str, str]:
    """Return (partition_name, DDL) for a single month."""
    next_month = _add_months(month_start, 1)
    partition_name = f"{table.partition_prefix}_{month_start.year:04d}_{month_start.month:02d}"
    from_bound = _format_bound(month_start, table.column_type)
    to_bound = _format_bound(next_month, table.column_type)
    ddl = (
        f"CREATE TABLE IF NOT EXISTS {partition_name} "
        f"PARTITION OF {table.name} "
        f"FOR VALUES FROM ({from_bound}) TO ({to_bound});"
    )
    return partition_name, ddl


def _table_exists(conn: psycopg.Connection, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_class WHERE relname = %s AND relkind IN ('r', 'p')",
            (table,),
        )
        return cur.fetchone() is not None


def ensure_partitions(
    conn: psycopg.Connection,
    table: PartitionedTable,
    horizon_months: int,
    today: date | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Ensure ``horizon_months`` partitions exist starting at the current month.

    Returns the list of DDL statements emitted (whether or not the partition
    already existed; CREATE TABLE IF NOT EXISTS makes the call cheap).
    """
    if not _table_exists(conn, table.name):
        logger.warning("Skipping %s: parent table not present in database", table.name)
        return []

    today = today or get_planning_date()
    start = date(today.year, today.month, 1)

    statements: list[str] = []
    for offset in range(horizon_months):
        month_start = _add_months(start, offset)
        _, ddl = build_partition_ddl(table, month_start)
        statements.append(ddl)
        if dry_run:
            logger.info("[dry-run] %s", ddl)
        else:
            with conn.cursor() as cur:
                cur.execute(ddl)
            logger.info("Ensured partition %s_%04d_%02d",
                        table.partition_prefix, month_start.year, month_start.month)
    return statements


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Number of months to provision ahead from the current month (default: 12).",
    )
    parser.add_argument(
        "--table",
        action="append",
        default=None,
        help="Restrict to a specific parent table. May be passed multiple times. "
             "Default: every table in the registry.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print DDL without executing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse_args(argv)

    if args.horizon < 1:
        logger.error("--horizon must be at least 1")
        return 2

    selected: tuple[PartitionedTable, ...]
    if args.table:
        chosen = {t for t in args.table}
        selected = tuple(t for t in PARTITIONED_TABLES if t.name in chosen)
        unknown = chosen - {t.name for t in selected}
        if unknown:
            logger.error("Unknown table(s): %s. Known: %s",
                         ", ".join(sorted(unknown)),
                         ", ".join(t.name for t in PARTITIONED_TABLES))
            return 2
    else:
        selected = PARTITIONED_TABLES

    db_params = get_db_params()
    with psycopg.connect(**db_params) as conn:
        conn.autocommit = False
        for table in selected:
            try:
                ensure_partitions(
                    conn,
                    table,
                    horizon_months=args.horizon,
                    dry_run=args.dry_run,
                )
            except psycopg.Error:
                logger.exception("Failed to provision partitions for %s", table.name)
                conn.rollback()
                return 1
        if args.dry_run:
            conn.rollback()
        else:
            conn.commit()
    logger.info(
        "Done: ensured %d months ahead for %d table(s)%s",
        args.horizon,
        len(selected),
        " (dry-run, no changes committed)" if args.dry_run else "",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
