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


def _resolve_input_files(
    args_input: str | None,
    args_model: str | None,
    args_all: bool,
    backtest_dir: Path,
) -> list[Path]:
    """Return list of prediction CSVs to load based on CLI flags.

    Priority order:
      --all            → scan data/backtest/*/backtest_predictions.csv
      --model MODEL_ID → data/backtest/<MODEL_ID>/backtest_predictions.csv
      --input PATH     → explicit path (backward-compatible)
    """
    if args_all:
        found = sorted(backtest_dir.glob("*/backtest_predictions.csv"))
        if not found:
            print(f"No backtest_predictions.csv files found under {backtest_dir}/*/")
            print("  Run a backtest first: make backtest-lgbm-cluster")
            sys.exit(1)
        return found

    if args_model:
        p = backtest_dir / args_model / "backtest_predictions.csv"
        if not p.exists():
            available = [d.name for d in backtest_dir.iterdir() if d.is_dir() and
                         (d / "backtest_predictions.csv").exists()] if backtest_dir.exists() else []
            print(f"Error: No predictions found for model '{args_model}' at {p}")
            if available:
                print(f"  Available models: {available}")
            else:
                print("  No backtest output directories found — run a backtest first.")
            sys.exit(1)
        return [p]

    if args_input:
        p = ROOT / args_input
        if not p.exists():
            print(f"Error: Input file not found: {p}")
            sys.exit(1)
        return [p]

    print("Error: Specify one of:")
    print("  --model MODEL_ID     load a specific model  (e.g. --model lgbm_cluster)")
    print("  --all                load all models from data/backtest/*/")
    print("  --input PATH         explicit CSV path (legacy)")
    sys.exit(1)


def _load_one(db: dict, csv_path: Path, replace: bool, model_id_filter: str | None) -> None:
    """Load one backtest_predictions.csv + its sibling all_lags CSV into Postgres.

    Refreshes all 5 materialized views after each model load.
    The archive CSV is expected at csv_path.parent / backtest_predictions_all_lags.csv.
    """
    # Peek at CSV to determine model_ids present
    sample = pd.read_csv(csv_path, nrows=100)
    csv_model_ids = sample["model_id"].unique().tolist()
    if model_id_filter:
        csv_model_ids = [model_id_filter]

    print(f"\n{'='*60}")
    print(f"Loading {csv_path}")
    print(f"  Model IDs in file: {csv_model_ids}")

    col_list = ", ".join(LOAD_COLS)

    with psycopg.connect(**db) as conn:
        with conn.cursor() as cur:
            cur.execute("SET synchronous_commit = off")
            cur.execute("SET work_mem = '256MB'")
            cur.execute("SET maintenance_work_mem = '512MB'")

            cur.execute(f"""
                CREATE TEMP TABLE _stg_backtest (
                    _row_id SERIAL,
                    {', '.join(f'{c} TEXT' for c in LOAD_COLS)}
                ) ON COMMIT DROP
            """)

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

            if model_id_filter:
                cur.execute("DELETE FROM _stg_backtest WHERE model_id != %s", (model_id_filter,))

            cur.execute("SELECT COUNT(*) FROM _stg_backtest")
            staged_count = cur.fetchone()[0]
            print(f"  Staged rows: {staged_count:,}")

            if staged_count == 0:
                print("  No rows to load.")
                conn.rollback()
                return

            if replace:
                t0 = time.time()
                for mid in csv_model_ids:
                    cur.execute(f"DELETE FROM {_TABLE} WHERE model_id = %s", (mid,))
                    deleted = cur.rowcount
                    print(f"  Deleted {deleted:,} existing rows for model_id='{mid}' ({time.time() - t0:.1f}s)")

            bulk_fast = replace
            if bulk_fast:
                print("  Dropping indexes & constraints for bulk load...")
                t0 = time.time()
                _drop_indexes_and_constraints(cur)
                print(f"    Done ({time.time() - t0:.1f}s)")

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

            if bulk_fast:
                _recreate_indexes_and_constraints(cur)

        conn.commit()

    # Refresh forecast materialized views
    print("  Refreshing forecast materialized views...")
    t0 = time.time()
    with psycopg.connect(**db) as conn:
        with conn.cursor() as cur:
            cur.execute("SET maintenance_work_mem = '512MB'")
            cur.execute("REFRESH MATERIALIZED VIEW agg_forecast_monthly")
            print(f"    agg_forecast_monthly refreshed ({time.time() - t0:.1f}s)")
            t1 = time.time()
            cur.execute("REFRESH MATERIALIZED VIEW agg_accuracy_by_dim")
            print(f"    agg_accuracy_by_dim refreshed ({time.time() - t1:.1f}s)")
            t2 = time.time()
            cur.execute("REFRESH MATERIALIZED VIEW agg_dfu_coverage")
            print(f"    agg_dfu_coverage refreshed ({time.time() - t2:.1f}s)")
        conn.commit()
    print(f"  Forecast views refreshed in {time.time() - t0:.1f}s")

    # Load archive (sibling file lives in the same model subdirectory)
    archive_path = csv_path.parent / "backtest_predictions_all_lags.csv"
    _load_archive(db, archive_path, csv_model_ids, replace, model_id_filter)

    # Refresh archive accuracy views
    print("  Refreshing archive accuracy views...")
    t0 = time.time()
    with psycopg.connect(**db) as conn:
        with conn.cursor() as cur:
            cur.execute("SET maintenance_work_mem = '512MB'")
            cur.execute("REFRESH MATERIALIZED VIEW agg_accuracy_lag_archive")
            cur.execute("REFRESH MATERIALIZED VIEW agg_dfu_coverage_lag_archive")
        conn.commit()
    print(f"  Archive views refreshed in {time.time() - t0:.1f}s")
    print(f"  Done: {csv_path.parent.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load backtest predictions into Postgres",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load one model (output written to data/backtest/lgbm_cluster/)
  %(prog)s --model lgbm_cluster --replace

  # Load all models that have been run (scans data/backtest/*/)
  %(prog)s --all --replace

  # Backward-compatible explicit path
  %(prog)s --input data/backtest/lgbm_cluster/backtest_predictions.csv --replace
        """,
    )
    parser.add_argument(
        "--model", type=str, default=None, metavar="MODEL_ID",
        help="Load from data/backtest/<MODEL_ID>/backtest_predictions.csv "
             "(e.g. lgbm_cluster, catboost_cluster, xgboost_cluster)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Load every model found under data/backtest/*/backtest_predictions.csv",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Explicit CSV path (backward-compatible; prefer --model or --all)",
    )
    parser.add_argument(
        "--model-id", type=str, default=None,
        help="Filter rows to this model_id when using --input (legacy flag)",
    )
    parser.add_argument(
        "--replace", action="store_true",
        help="Delete existing rows for the model_id(s) before inserting (recommended)",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    db = get_db_conn()

    backtest_dir = ROOT / "data" / "backtest"
    csv_files = _resolve_input_files(args.input, args.model, args.all, backtest_dir)

    model_labels = [f.parent.name for f in csv_files]
    print(f"Loading {len(csv_files)} model(s): {model_labels}")

    for csv_path in csv_files:
        _load_one(db, csv_path, args.replace, args.model_id)

    print(f"\n{'='*60}")
    print(f"All done. Loaded: {model_labels}")


if __name__ == "__main__":
    main()
