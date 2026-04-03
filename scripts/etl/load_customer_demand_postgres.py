#!/usr/bin/env python
"""Load normalized customer demand CSV into partitioned PostgreSQL table.

Target table: fact_customer_demand_monthly (monthly range partitioned by startdate)

Performance optimizations:
  - Single COPY of entire CSV into UNLOGGED staging table
  - Dedup rare duplicates (~0.3%) in staging via DELETE
  - Parallel per-month INSERT — one thread per partition, each with own connection
  - Indexes and constraints dropped during load, rebuilt after

Modes:
    Default:  Ensure partitions exist, UPSERT (ON CONFLICT demand_ck DO UPDATE)
    --month:  Drop+recreate that month's partition, load only matching rows
    --replace: Drop all partitions, recreate per month, parallel INSERT
    --dry-run: Parse CSV and report stats without touching the database

Usage:
    uv run python scripts/etl/load_customer_demand_postgres.py --replace
    uv run python scripts/etl/load_customer_demand_postgres.py --month 2026-01
    uv run python scripts/etl/load_customer_demand_postgres.py --dry-run
"""

import argparse
import csv
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.db import get_db_params
from common.services.perf_profiler import profiled_section

logger = logging.getLogger(__name__)

_TABLE = "fact_customer_demand_monthly"
_STG = "_stg_customer_demand_bulk"

_PG_WORK_MEM = "256MB"
_PG_MAINTENANCE_WORK_MEM = "512MB"
_MAX_PARALLEL_WORKERS = 6


# ---------------------------------------------------------------------------
# Partition helpers
# ---------------------------------------------------------------------------

def _partition_name(month_start: date) -> str:
    return f"{_TABLE}_{month_start:%Y_%m}"


def _month_range(month_start: date) -> tuple[str, str]:
    year, month = month_start.year, month_start.month
    end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return month_start.isoformat(), end.isoformat()


def _ensure_partition_exists(cur: psycopg.Cursor, month_start: date) -> str:
    part_name = _partition_name(month_start)
    start_str, end_str = _month_range(month_start)
    cur.execute("""
        SELECT 1 FROM pg_class
        WHERE relname = %s AND relnamespace = 'public'::regnamespace
    """, (part_name,))
    if not cur.fetchone():
        cur.execute(
            f'CREATE TABLE "{part_name}" PARTITION OF "{_TABLE}" '
            f"FOR VALUES FROM ('{start_str}') TO ('{end_str}')"
        )
    return part_name


def _drop_partition(cur: psycopg.Cursor, month_start: date) -> None:
    cur.execute(f'DROP TABLE IF EXISTS "{_partition_name(month_start)}"')


def _get_existing_partitions(cur: psycopg.Cursor) -> list[str]:
    cur.execute("""
        SELECT inhrelid::regclass::text
        FROM pg_inherits WHERE inhparent = %s::regclass ORDER BY 1
    """, (f"public.{_TABLE}",))
    return [r[0] for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def _stage_csv(cur: psycopg.Cursor, csv_path: Path) -> int:
    """COPY entire CSV into staging. No dedup, no index — keep it fast.

    Duplicates (~0.3%) are handled via ON CONFLICT during per-partition INSERT.
    """
    cur.execute(f"DROP TABLE IF EXISTS {_STG}")
    cur.execute(f"""
        CREATE UNLOGGED TABLE {_STG} (
            item_id TEXT, customer_no TEXT, site TEXT, location_id TEXT,
            startdate TEXT, demand_qty TEXT, sales_qty TEXT, oos_qty TEXT
        )
    """)

    t0 = time.time()
    with csv_path.open("rb") as f:
        with cur.copy(f"COPY {_STG} FROM STDIN WITH (FORMAT CSV, HEADER TRUE)") as copy:
            while chunk := f.read(8 * 1024 * 1024):
                copy.write(chunk)

    cur.execute(f"SELECT COUNT(*) FROM {_STG}")
    row_count = cur.fetchone()[0]
    logger.info("  Staged %s rows via COPY (%.1fs)", f"{row_count:,}", time.time() - t0)

    return row_count


def _discover_months(cur: psycopg.Cursor) -> dict[date, int]:
    cur.execute(f"""
        SELECT date_trunc('month', startdate::date)::date, COUNT(*)
        FROM {_STG} WHERE startdate IS NOT NULL AND startdate != ''
        GROUP BY 1 ORDER BY 1
    """)
    return {row[0]: row[1] for row in cur.fetchall()}


def _drop_staging(cur: psycopg.Cursor) -> None:
    cur.execute(f"DROP TABLE IF EXISTS {_STG}")


# ---------------------------------------------------------------------------
# INSERT SQL — per-month, directly into partition (no routing overhead)
# ---------------------------------------------------------------------------

def _build_partition_insert_sql(part_name: str) -> str:
    """INSERT directly into partition, pre-aggregating duplicates via GROUP BY.

    GROUP BY on ~7M rows per partition is fast (vs 297M for the whole table).
    No ON CONFLICT needed — GROUP BY guarantees unique keys.
    """
    return f"""
        INSERT INTO "{part_name}" (
            demand_ck, item_id, customer_no, site, location_id, startdate,
            demand_qty, sales_qty, oos_qty
        )
        SELECT
            s.item_id || '_' || s.customer_no || '_' || s.location_id || '_' || s.startdate::date,
            s.item_id, s.customer_no, MAX(s.site), s.location_id, s.startdate::date,
            SUM(s.demand_qty::numeric(18,4)),
            SUM(s.sales_qty::numeric(18,4)),
            SUM(s.oos_qty::numeric(18,4))
        FROM {_STG} s
        WHERE s.startdate::date >= %s AND s.startdate::date < %s
        GROUP BY s.item_id, s.customer_no, s.location_id, s.startdate::date
    """


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

_INDEXES = [
    ("idx_cust_demand_item",      "item_id"),
    ("idx_cust_demand_customer",  "customer_no"),
    ("idx_cust_demand_location",  "location_id"),
    ("idx_cust_demand_startdate", "startdate"),
    ("idx_cust_demand_item_loc",  "item_id, location_id"),
    ("idx_cust_demand_item_cust", "item_id, customer_no"),
    ("idx_cust_demand_site_cust", "site, customer_no"),
]


def _drop_indexes(cur: psycopg.Cursor) -> None:
    for name, _ in _INDEXES:
        cur.execute(f"DROP INDEX IF EXISTS {name}")
    logger.info("Dropped %d indexes", len(_INDEXES))


def _drop_constraints(cur: psycopg.Cursor) -> list[str]:
    cur.execute(f"""
        SELECT conname FROM pg_constraint
        WHERE conrelid = '"{_TABLE}"'::regclass AND contype = 'u'
    """)
    constraints = [r[0] for r in cur.fetchall()]
    for con in constraints:
        cur.execute(f'ALTER TABLE "{_TABLE}" DROP CONSTRAINT IF EXISTS "{con}"')
    if constraints:
        logger.info("Dropped %d unique constraints", len(constraints))
    return constraints


def _rebuild_indexes(cur: psycopg.Cursor) -> None:
    t0 = time.time()
    cur.execute(f"""
        ALTER TABLE "{_TABLE}"
        ADD CONSTRAINT uq_cust_demand_ck UNIQUE (demand_ck, startdate)
    """)
    for name, cols in _INDEXES:
        cur.execute(f'CREATE INDEX IF NOT EXISTS {name} ON "{_TABLE}" ({cols})')
    logger.info("Rebuilt constraint + %d indexes (%.1fs)", len(_INDEXES), time.time() - t0)


# ---------------------------------------------------------------------------
# Parallel per-month INSERT worker
# ---------------------------------------------------------------------------

def _insert_month_worker(
    db_params: dict, month_start: date, part_name: str,
) -> tuple[str, int, float]:
    """Worker: insert one month into its partition using a dedicated connection."""
    start_str, end_str = _month_range(month_start)
    sql = _build_partition_insert_sql(part_name)

    t0 = time.time()
    with psycopg.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SET work_mem = '{_PG_WORK_MEM}'")
            cur.execute("SET synchronous_commit = 'off'")
            cur.execute(sql, (start_str, end_str))
            loaded = cur.rowcount
        conn.commit()

    elapsed = time.time() - t0
    return month_start.isoformat(), loaded, elapsed


# ---------------------------------------------------------------------------
# Load modes
# ---------------------------------------------------------------------------

def _load_replace(db_params: dict, cur: psycopg.Cursor, months: dict[date, int]) -> int:
    """--replace: drop partitions, recreate, parallel INSERT per month.

    Keeps the unique constraint (needed for ON CONFLICT to handle duplicates).
    Drops only the secondary indexes for speed — rebuilt after.
    """
    # Drop existing
    existing = _get_existing_partitions(cur)
    for part in existing:
        cur.execute(f'DROP TABLE IF EXISTS "{part.replace("public.", "")}"')
    logger.info("Dropped %d existing partitions", len(existing))

    # Create all partitions (unique constraint auto-propagates from parent)
    part_names: dict[date, str] = {}
    for m in sorted(months):
        part_names[m] = _ensure_partition_exists(cur, m)

    # Drop all indexes + unique constraint (no ON CONFLICT needed — GROUP BY dedupes)
    _drop_indexes(cur)
    _drop_constraints(cur)

    # Commit so parallel workers can see partitions + staging table
    cur.connection.commit()

    # Parallel INSERT — one thread per month, ON CONFLICT handles duplicates
    n_workers = min(len(months), _MAX_PARALLEL_WORKERS)
    logger.info("  Parallel INSERT: %d months, %d workers", len(months), n_workers)

    total = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_insert_month_worker, db_params, m, part_names[m]): m
            for m in sorted(months)
        }
        for fut in as_completed(futures):
            iso, loaded, elapsed = fut.result()
            total += loaded
            logger.info("    %s: %s rows (%.1fs)", iso, f"{loaded:,}", elapsed)

    logger.info("  Total: %s rows across %d partitions (%.1fs)",
                f"{total:,}", len(months), time.time() - t0)

    # Rebuild unique constraint + secondary indexes
    _rebuild_indexes(cur)

    return total


def _load_single_month(db_params: dict, cur: psycopg.Cursor, month_filter: date) -> int:
    """--month: drop partition, recreate, INSERT."""
    _drop_partition(cur, month_filter)
    part_name = _ensure_partition_exists(cur, month_filter)
    cur.connection.commit()

    _, loaded, elapsed = _insert_month_worker(db_params, month_filter, part_name)
    logger.info("  %s: %s rows (%.1fs)", month_filter.isoformat(), f"{loaded:,}", elapsed)
    return loaded


def _load_default(db_params: dict, cur: psycopg.Cursor, months: dict[date, int]) -> int:
    """Default: ensure partitions, UPSERT."""
    for m in sorted(months):
        _ensure_partition_exists(cur, m)
    cur.connection.commit()

    # UPSERT via parent table (handles conflict resolution)
    upsert_sql = f"""
        INSERT INTO "{_TABLE}" (
            demand_ck, item_id, customer_no, site, location_id, startdate,
            demand_qty, sales_qty, oos_qty
        )
        SELECT
            s.item_id || '_' || s.customer_no || '_' || s.location_id || '_' || s.startdate::date,
            s.item_id, s.customer_no, s.site, s.location_id, s.startdate::date,
            s.demand_qty::numeric(18,4), s.sales_qty::numeric(18,4), s.oos_qty::numeric(18,4)
        FROM {_STG} s
        WHERE s.startdate IS NOT NULL AND s.startdate != ''
        ON CONFLICT (demand_ck, startdate) DO UPDATE SET
            demand_qty  = EXCLUDED.demand_qty,
            sales_qty   = EXCLUDED.sales_qty,
            oos_qty     = EXCLUDED.oos_qty,
            modified_ts = NOW()
    """
    t0 = time.time()
    cur.execute(upsert_sql)
    loaded = cur.rowcount
    logger.info("  Upserted %s rows (%.1fs)", f"{loaded:,}", time.time() - t0)
    return loaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Load normalized customer demand CSV into PostgreSQL",
    )
    parser.add_argument("--file", default=str(ROOT / "data" / "customer_demand_clean.csv"))
    parser.add_argument("--month", type=str, default=None, help="YYYY-MM")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.file).resolve()
    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        sys.exit(1)

    if args.dry_run:
        logger.info("DRY RUN: scanning %s ...", csv_path.name)
        months: dict[date, int] = defaultdict(int)
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                s = row.get("startdate", "").strip()
                if s:
                    d = date.fromisoformat(s)
                    months[date(d.year, d.month, 1)] += 1
        total = sum(months.values())
        logger.info("Would load %s rows across %d months", f"{total:,}", len(months))
        return

    t_start = time.time()
    db_params = get_db_params()

    with psycopg.connect(**db_params, autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SET work_mem = '{_PG_WORK_MEM}'")
            cur.execute(f"SET maintenance_work_mem = '{_PG_MAINTENANCE_WORK_MEM}'")

            # Step 1: Stage + dedup
            logger.info("Step 1: Staging %s ...", csv_path.name)
            with profiled_section("stage_csv"):
                staged = _stage_csv(cur, csv_path)
            conn.commit()  # commit so staging is visible to parallel workers

            # Step 2: Discover months
            months = _discover_months(cur)
            if not months:
                logger.error("No valid rows in staging")
                _drop_staging(cur)
                conn.commit()
                sys.exit(1)

            logger.info("Found %s unique rows across %d months",
                        f"{staged:,}", len(months))

            # Parse --month
            month_filter: date | None = None
            if args.month:
                parts = args.month.split("-")
                month_filter = date(int(parts[0]), int(parts[1]), 1)
                if month_filter not in months:
                    logger.warning("Month %s not in data", month_filter.isoformat())
                    _drop_staging(cur)
                    conn.commit()
                    sys.exit(1)

            # Step 3: Load
            logger.info("Step 2: Loading into partitions ...")
            with profiled_section("load_partitions"):
                if month_filter:
                    loaded = _load_single_month(db_params, cur, month_filter)
                elif args.replace:
                    loaded = _load_replace(db_params, cur, months)
                else:
                    loaded = _load_default(db_params, cur, months)

            # Step 4: Cleanup
            _drop_staging(cur)
            conn.commit()

    elapsed = time.time() - t_start
    rate = loaded / max(elapsed, 0.001)
    logger.info("Done: %s rows in %.1fs (%s rows/s)",
                f"{loaded:,}", elapsed, f"{rate:,.0f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
