"""
Load backtest predictions into Postgres.

Loads execution-lag predictions into fact_external_forecast_monthly (main table)
and all-lag predictions into backtest_lag_archive (archive table).

Uses COPY + temp table pattern (same as load_dataset_postgres.py).
Supports --replace to delete existing rows for a model_id before inserting.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def get_db_conn() -> dict[str, Any]:
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5440")),
        "dbname": os.getenv("POSTGRES_DB", "demand_mvp"),
        "user": os.getenv("POSTGRES_USER", "demand"),
        "password": os.getenv("POSTGRES_PASSWORD", "demand"),
    }


LOAD_COLS = [
    "forecast_ck", "dmdunit", "dmdgroup", "loc",
    "fcstdate", "startdate", "lag", "execution_lag",
    "basefcst_pref", "tothist_dmd", "model_id",
]

BATCH_SIZE = 2_000_000

# Indexes to drop/recreate for fast bulk load (excludes PK)
_SECONDARY_INDEXES = [
    "idx_fact_external_forecast_monthly_item",
    "idx_fact_external_forecast_monthly_loc",
    "idx_fact_external_forecast_monthly_fcstdate",
    "idx_fact_external_forecast_monthly_startdate",
    "idx_fact_external_forecast_monthly_lag",
    "idx_fact_external_forecast_monthly_model_id",
]
_INDEX_DDL = [
    "CREATE INDEX {name} ON fact_external_forecast_monthly (dmdunit)",
    "CREATE INDEX {name} ON fact_external_forecast_monthly (loc)",
    "CREATE INDEX {name} ON fact_external_forecast_monthly (fcstdate)",
    "CREATE INDEX {name} ON fact_external_forecast_monthly (startdate)",
    "CREATE INDEX {name} ON fact_external_forecast_monthly (lag)",
    "CREATE INDEX {name} ON fact_external_forecast_monthly (model_id)",
]
_CHECK_CONSTRAINTS = [
    "chk_fact_external_forecast_monthly_lag_0_4",
    "chk_fact_external_forecast_monthly_fcst_month_start",
    "chk_fact_external_forecast_monthly_start_month_start",
    "chk_fact_external_forecast_monthly_lag_matches_dates",
]
_UNIQUE_CONSTRAINT = "uq_forecast_ck_model"
_TABLE = "fact_external_forecast_monthly"

# ── Archive table constants ────────────────────────────────────────────────
_ARCHIVE_TABLE = "backtest_lag_archive"
ARCHIVE_COLS = [
    "forecast_ck", "dmdunit", "dmdgroup", "loc",
    "fcstdate", "startdate", "lag", "execution_lag",
    "basefcst_pref", "tothist_dmd", "model_id", "timeframe",
]
_ARCHIVE_SECONDARY_INDEXES = [
    "idx_backtest_lag_archive_model_id",
    "idx_backtest_lag_archive_dmdunit",
    "idx_backtest_lag_archive_startdate",
    "idx_backtest_lag_archive_lag",
]
_ARCHIVE_INDEX_DDL = [
    "CREATE INDEX {name} ON backtest_lag_archive (model_id)",
    "CREATE INDEX {name} ON backtest_lag_archive (dmdunit)",
    "CREATE INDEX {name} ON backtest_lag_archive (startdate)",
    "CREATE INDEX {name} ON backtest_lag_archive (lag)",
]
_ARCHIVE_CHECK_CONSTRAINTS = [
    "chk_backtest_lag_archive_lag_0_4",
    "chk_backtest_lag_archive_fcst_month_start",
    "chk_backtest_lag_archive_start_month_start",
    "chk_backtest_lag_archive_lag_matches_dates",
]
_ARCHIVE_UNIQUE_CONSTRAINT = "uq_backtest_lag_archive_ck"


def _drop_indexes_and_constraints(cur) -> None:
    """Drop secondary indexes and CHECK constraints for fast bulk insert."""
    for idx in _SECONDARY_INDEXES:
        cur.execute(f"DROP INDEX IF EXISTS {idx}")
    cur.execute(f"ALTER TABLE {_TABLE} DROP CONSTRAINT IF EXISTS {_UNIQUE_CONSTRAINT}")
    for ck in _CHECK_CONSTRAINTS:
        cur.execute(f"ALTER TABLE {_TABLE} DROP CONSTRAINT IF EXISTS {ck}")


def _recreate_indexes_and_constraints(cur) -> None:
    """Recreate indexes and constraints after bulk insert."""
    t0 = time.time()
    # Unique constraint first (needed for ON CONFLICT in future upserts)
    print("  Recreating UNIQUE constraint...")
    cur.execute(f"ALTER TABLE {_TABLE} ADD CONSTRAINT {_UNIQUE_CONSTRAINT} UNIQUE (forecast_ck, model_id)")
    print(f"    UNIQUE constraint created ({time.time() - t0:.1f}s)")

    print("  Recreating secondary indexes...")
    for name, ddl in zip(_SECONDARY_INDEXES, _INDEX_DDL):
        t1 = time.time()
        cur.execute(ddl.format(name=name))
        print(f"    {name} ({time.time() - t1:.1f}s)")

    print("  Recreating CHECK constraints...")
    cur.execute(f"""ALTER TABLE {_TABLE}
        ADD CONSTRAINT chk_fact_external_forecast_monthly_lag_0_4
            CHECK (lag BETWEEN 0 AND 4),
        ADD CONSTRAINT chk_fact_external_forecast_monthly_fcst_month_start
            CHECK (fcstdate = date_trunc('month', fcstdate)::date),
        ADD CONSTRAINT chk_fact_external_forecast_monthly_start_month_start
            CHECK (startdate = date_trunc('month', startdate)::date),
        ADD CONSTRAINT chk_fact_external_forecast_monthly_lag_matches_dates
            CHECK (((EXTRACT(YEAR FROM startdate)::int - EXTRACT(YEAR FROM fcstdate)::int) * 12
                  + (EXTRACT(MONTH FROM startdate)::int - EXTRACT(MONTH FROM fcstdate)::int)) = lag)
    """)
    print(f"  All indexes/constraints rebuilt in {time.time() - t0:.1f}s")


def _drop_archive_indexes_and_constraints(cur) -> None:
    """Drop archive table indexes and constraints for fast bulk insert."""
    for idx in _ARCHIVE_SECONDARY_INDEXES:
        cur.execute(f"DROP INDEX IF EXISTS {idx}")
    cur.execute(f"ALTER TABLE {_ARCHIVE_TABLE} DROP CONSTRAINT IF EXISTS {_ARCHIVE_UNIQUE_CONSTRAINT}")
    for ck in _ARCHIVE_CHECK_CONSTRAINTS:
        cur.execute(f"ALTER TABLE {_ARCHIVE_TABLE} DROP CONSTRAINT IF EXISTS {ck}")


def _recreate_archive_indexes_and_constraints(cur) -> None:
    """Recreate archive table indexes and constraints after bulk insert."""
    t0 = time.time()
    print("  Recreating archive UNIQUE constraint...")
    cur.execute(f"ALTER TABLE {_ARCHIVE_TABLE} ADD CONSTRAINT {_ARCHIVE_UNIQUE_CONSTRAINT} "
                f"UNIQUE (forecast_ck, model_id, lag)")
    print(f"    UNIQUE constraint created ({time.time() - t0:.1f}s)")

    print("  Recreating archive secondary indexes...")
    for name, ddl in zip(_ARCHIVE_SECONDARY_INDEXES, _ARCHIVE_INDEX_DDL):
        t1 = time.time()
        cur.execute(ddl.format(name=name))
        print(f"    {name} ({time.time() - t1:.1f}s)")

    print("  Recreating archive CHECK constraints...")
    cur.execute(f"""ALTER TABLE {_ARCHIVE_TABLE}
        ADD CONSTRAINT chk_backtest_lag_archive_lag_0_4
            CHECK (lag BETWEEN 0 AND 4),
        ADD CONSTRAINT chk_backtest_lag_archive_fcst_month_start
            CHECK (fcstdate = date_trunc('month', fcstdate)::date),
        ADD CONSTRAINT chk_backtest_lag_archive_start_month_start
            CHECK (startdate = date_trunc('month', startdate)::date),
        ADD CONSTRAINT chk_backtest_lag_archive_lag_matches_dates
            CHECK (((EXTRACT(YEAR FROM startdate)::int - EXTRACT(YEAR FROM fcstdate)::int) * 12
                  + (EXTRACT(MONTH FROM startdate)::int - EXTRACT(MONTH FROM fcstdate)::int)) = lag)
    """)
    print(f"  All archive indexes/constraints rebuilt in {time.time() - t0:.1f}s")


def _load_archive(db: dict, archive_path: Path, model_ids: list[str], replace: bool, model_id_filter: str | None) -> None:
    """Load all-lags CSV into backtest_lag_archive table."""
    if not archive_path.exists():
        print(f"\nArchive file not found: {archive_path} — skipping archive load")
        return

    print(f"\n{'='*60}")
    print(f"Loading archive from {archive_path}")
    archive_col_list = ", ".join(ARCHIVE_COLS)

    with psycopg.connect(**db) as conn:
        with conn.cursor() as cur:
            cur.execute("SET synchronous_commit = off")
            cur.execute("SET work_mem = '256MB'")
            cur.execute("SET maintenance_work_mem = '512MB'")

            # Staging table
            cur.execute(f"""
                CREATE TEMP TABLE _stg_archive (
                    _row_id SERIAL,
                    {', '.join(f'{c} TEXT' for c in ARCHIVE_COLS)}
                ) ON COMMIT DROP
            """)

            # Stream CSV
            copy_sql = f"COPY _stg_archive ({archive_col_list}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)"
            print("  Streaming archive CSV to staging table...")
            t0 = time.time()
            bytes_read = 0
            file_size = archive_path.stat().st_size
            last_pct = -1
            with cur.copy(copy_sql) as copy, archive_path.open("r", encoding="utf-8") as f:
                while chunk := f.read(1024 * 1024):
                    copy.write(chunk)
                    bytes_read += len(chunk)
                    pct = int(bytes_read * 100 / file_size)
                    if pct >= last_pct + 10:
                        last_pct = pct
                        print(f"    COPY progress: {pct}% ({bytes_read / (1024*1024):.0f} MB / {file_size / (1024*1024):.0f} MB)")
            print(f"  Staged {bytes_read / (1024*1024):.0f} MB in {time.time() - t0:.1f}s")

            if model_id_filter:
                cur.execute("DELETE FROM _stg_archive WHERE model_id != %s", (model_id_filter,))

            cur.execute("SELECT COUNT(*) FROM _stg_archive")
            staged_count = cur.fetchone()[0]
            print(f"  Staged archive rows: {staged_count:,}")

            if staged_count == 0:
                print("  No archive rows to load.")
                conn.rollback()
                return

            # Delete existing if --replace
            if replace:
                t0 = time.time()
                for mid in model_ids:
                    cur.execute(f"DELETE FROM {_ARCHIVE_TABLE} WHERE model_id = %s", (mid,))
                    deleted = cur.rowcount
                    print(f"  Deleted {deleted:,} existing archive rows for model_id='{mid}' ({time.time() - t0:.1f}s)")

            bulk_fast = replace
            if bulk_fast:
                print("  Dropping archive indexes & constraints for bulk load...")
                t0 = time.time()
                _drop_archive_indexes_and_constraints(cur)
                print(f"    Done ({time.time() - t0:.1f}s)")

            # Insert with type casting
            select_expr = """
                    s.forecast_ck,
                    s.dmdunit,
                    s.dmdgroup,
                    s.loc,
                    s.fcstdate::date,
                    s.startdate::date,
                    s.lag::integer,
                    CASE WHEN s.execution_lag IN ('', 'null', 'none', 'None', 'NA')
                         THEN NULL ELSE s.execution_lag::integer END,
                    CASE WHEN s.basefcst_pref IN ('', 'null', 'none', 'None', 'NA')
                         THEN NULL ELSE s.basefcst_pref::numeric END,
                    CASE WHEN s.tothist_dmd IN ('', 'null', 'none', 'None', 'NA')
                         THEN NULL ELSE s.tothist_dmd::numeric END,
                    s.model_id,
                    CASE WHEN s.timeframe IN ('', 'null', 'none', 'None', 'NA')
                         THEN NULL ELSE s.timeframe END
            """
            if bulk_fast:
                insert_sql = f"""
                    INSERT INTO {_ARCHIVE_TABLE}
                        (forecast_ck, dmdunit, dmdgroup, loc,
                         fcstdate, startdate, lag, execution_lag,
                         basefcst_pref, tothist_dmd, model_id, timeframe)
                    SELECT {select_expr}
                    FROM _stg_archive s
                    WHERE s._row_id > %s AND s._row_id <= %s
                """
            else:
                insert_sql = f"""
                    INSERT INTO {_ARCHIVE_TABLE}
                        (forecast_ck, dmdunit, dmdgroup, loc,
                         fcstdate, startdate, lag, execution_lag,
                         basefcst_pref, tothist_dmd, model_id, timeframe)
                    SELECT {select_expr}
                    FROM _stg_archive s
                    WHERE s._row_id > %s AND s._row_id <= %s
                    ON CONFLICT (forecast_ck, model_id, lag) DO UPDATE SET
                        basefcst_pref = EXCLUDED.basefcst_pref,
                        tothist_dmd = EXCLUDED.tothist_dmd,
                        execution_lag = EXCLUDED.execution_lag,
                        timeframe = EXCLUDED.timeframe
                """

            loaded_total = 0
            t_insert = time.time()
            for batch_start in range(0, staged_count, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, staged_count)
                cur.execute(insert_sql, (batch_start, batch_end))
                loaded_total += cur.rowcount
                elapsed = time.time() - t_insert
                rate = loaded_total / elapsed if elapsed > 0 else 0
                print(f"    Batch {batch_end:,}/{staged_count:,} — {loaded_total:,} loaded ({elapsed:.0f}s, {rate:,.0f} rows/s)")

            print(f"  Inserted {loaded_total:,} archive rows in {time.time() - t_insert:.1f}s")

            if bulk_fast:
                _recreate_archive_indexes_and_constraints(cur)

        conn.commit()
    print("Archive load complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load backtest predictions into Postgres")
    parser.add_argument("--input", type=str, default="data/backtest/backtest_predictions.csv",
                        help="Predictions CSV path")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Filter to specific model_id (default: load all from CSV)")
    parser.add_argument("--replace", action="store_true",
                        help="Delete existing rows for model_id(s) before inserting")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    db = get_db_conn()

    csv_path = ROOT / args.input
    if not csv_path.exists():
        print(f"Error: Input file not found: {csv_path}")
        sys.exit(1)

    # Peek at CSV to determine model_ids
    sample = pd.read_csv(csv_path, nrows=100)
    csv_model_ids = sample["model_id"].unique().tolist()
    if args.model_id:
        csv_model_ids = [args.model_id]
    print(f"Loading predictions from {csv_path}")
    print(f"  Model IDs: {csv_model_ids}")

    col_list = ", ".join(LOAD_COLS)

    with psycopg.connect(**db) as conn:
        with conn.cursor() as cur:
            # Session-level tuning for bulk load
            cur.execute("SET synchronous_commit = off")
            cur.execute("SET work_mem = '256MB'")
            cur.execute("SET maintenance_work_mem = '512MB'")

            # Step 1: Create temp staging table (with row_id for batching)
            cur.execute(f"""
                CREATE TEMP TABLE _stg_backtest (
                    _row_id SERIAL,
                    {', '.join(f'{c} TEXT' for c in LOAD_COLS)}
                ) ON COMMIT DROP
            """)

            # Step 2: Stream CSV into staging with progress
            copy_sql = f"COPY _stg_backtest ({col_list}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)"
            print("  Streaming CSV to staging table...")
            t0 = time.time()
            bytes_read = 0
            file_size = csv_path.stat().st_size
            last_pct = -1
            with cur.copy(copy_sql) as copy, csv_path.open("r", encoding="utf-8") as f:
                while chunk := f.read(1024 * 1024):
                    copy.write(chunk)
                    bytes_read += len(chunk)
                    pct = int(bytes_read * 100 / file_size)
                    if pct >= last_pct + 10:
                        last_pct = pct
                        print(f"    COPY progress: {pct}% ({bytes_read / (1024*1024):.0f} MB / {file_size / (1024*1024):.0f} MB)")
            print(f"  Staged {bytes_read / (1024*1024):.0f} MB in {time.time() - t0:.1f}s")

            # Filter by model_id if specified
            if args.model_id:
                cur.execute(
                    "DELETE FROM _stg_backtest WHERE model_id != %s",
                    (args.model_id,),
                )

            # Count staged rows
            cur.execute("SELECT COUNT(*) FROM _stg_backtest")
            staged_count = cur.fetchone()[0]
            print(f"  Staged rows: {staged_count:,}")

            if staged_count == 0:
                print("  No rows to load.")
                conn.rollback()
                return

            # Step 3: Delete existing rows if --replace
            if args.replace:
                t0 = time.time()
                for mid in csv_model_ids:
                    cur.execute(
                        f"DELETE FROM {_TABLE} WHERE model_id = %s",
                        (mid,),
                    )
                    deleted = cur.rowcount
                    print(f"  Deleted {deleted:,} existing rows for model_id='{mid}' ({time.time() - t0:.1f}s)")

            # Step 4: Drop indexes/constraints for fast bulk insert (--replace only)
            bulk_fast = args.replace
            if bulk_fast:
                print("  Dropping indexes & constraints for bulk load...")
                t0 = time.time()
                _drop_indexes_and_constraints(cur)
                print(f"    Done ({time.time() - t0:.1f}s)")

            # Step 5: Batched INSERT with type casting
            print(f"  Inserting {staged_count:,} rows in batches of {BATCH_SIZE:,}...")
            select_expr = """
                    s.forecast_ck,
                    s.dmdunit,
                    s.dmdgroup,
                    s.loc,
                    s.fcstdate::date,
                    s.startdate::date,
                    s.lag::integer,
                    CASE WHEN s.execution_lag IN ('', 'null', 'none', 'None', 'NA')
                         THEN NULL ELSE s.execution_lag::integer END,
                    CASE WHEN s.basefcst_pref IN ('', 'null', 'none', 'None', 'NA')
                         THEN NULL ELSE s.basefcst_pref::numeric END,
                    CASE WHEN s.tothist_dmd IN ('', 'null', 'none', 'None', 'NA')
                         THEN NULL ELSE s.tothist_dmd::numeric END,
                    s.model_id
            """
            if bulk_fast:
                insert_sql = f"""
                    INSERT INTO {_TABLE}
                        (forecast_ck, dmdunit, dmdgroup, loc,
                         fcstdate, startdate, lag, execution_lag,
                         basefcst_pref, tothist_dmd, model_id)
                    SELECT {select_expr}
                    FROM _stg_backtest s
                    WHERE s._row_id > %s AND s._row_id <= %s
                """
            else:
                insert_sql = f"""
                    INSERT INTO {_TABLE}
                        (forecast_ck, dmdunit, dmdgroup, loc,
                         fcstdate, startdate, lag, execution_lag,
                         basefcst_pref, tothist_dmd, model_id)
                    SELECT {select_expr}
                    FROM _stg_backtest s
                    WHERE s._row_id > %s AND s._row_id <= %s
                    ON CONFLICT (forecast_ck, model_id) DO UPDATE SET
                        basefcst_pref = EXCLUDED.basefcst_pref,
                        tothist_dmd = EXCLUDED.tothist_dmd,
                        execution_lag = EXCLUDED.execution_lag,
                        modified_ts = NOW()
                """

            loaded_total = 0
            t_insert = time.time()
            for batch_start in range(0, staged_count, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, staged_count)
                cur.execute(insert_sql, (batch_start, batch_end))
                loaded_total += cur.rowcount
                elapsed = time.time() - t_insert
                rate = loaded_total / elapsed if elapsed > 0 else 0
                print(f"    Batch {batch_end:,}/{staged_count:,} — {loaded_total:,} loaded ({elapsed:.0f}s, {rate:,.0f} rows/s)")

            print(f"  Inserted {loaded_total:,} rows in {time.time() - t_insert:.1f}s")

            # Step 6: Rebuild indexes/constraints
            if bulk_fast:
                _recreate_indexes_and_constraints(cur)

        conn.commit()

    # Step 7: Refresh materialized views
    print("Refreshing materialized views...")
    t0 = time.time()
    with psycopg.connect(**db) as conn:
        with conn.cursor() as cur:
            cur.execute("SET maintenance_work_mem = '512MB'")
            cur.execute("REFRESH MATERIALIZED VIEW agg_forecast_monthly")
            print(f"  agg_forecast_monthly refreshed ({time.time() - t0:.1f}s)")
            t1 = time.time()
            cur.execute("REFRESH MATERIALIZED VIEW agg_accuracy_by_dim")
            print(f"  agg_accuracy_by_dim refreshed ({time.time() - t1:.1f}s)")
        conn.commit()
    print(f"  All forecast views refreshed in {time.time() - t0:.1f}s")

    # Step 8: Load archive table (all lags)
    archive_path = csv_path.parent / "backtest_predictions_all_lags.csv"
    _load_archive(db, archive_path, csv_model_ids, args.replace, args.model_id)

    # Step 9: Refresh archive accuracy slice view (depends on backtest_lag_archive)
    print("Refreshing agg_accuracy_lag_archive...")
    t0 = time.time()
    with psycopg.connect(**db) as conn:
        with conn.cursor() as cur:
            cur.execute("SET maintenance_work_mem = '512MB'")
            cur.execute("REFRESH MATERIALIZED VIEW agg_accuracy_lag_archive")
        conn.commit()
    print(f"  agg_accuracy_lag_archive refreshed in {time.time() - t0:.1f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
