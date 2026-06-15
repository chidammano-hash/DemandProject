"""
Load external ML forecast CSVs into fact_external_forecast_monthly and backtest_lag_archive.

These are backtest results from an external system with lags 0-4 already computed.
Unlike the standard external forecast (which stores only execution-lag), these
load ALL lags — matching the behaviour of internal LGBM/CatBoost/XGBoost backtests.

Usage:
    python scripts/etl/load_ext_ml_forecasts.py --model ext_lgbm --replace
    python scripts/etl/load_ext_ml_forecasts.py --all --replace
    python scripts/etl/load_ext_ml_forecasts.py --input data/input/df_ml_lgbm_l2_extract.csv --model-id ext_lgbm --replace
"""

import argparse
import sys
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.db import get_db_params
from common.core.etl_helpers import (
    drop_forecast_archive_indexes_and_constraints,
    drop_forecast_indexes_and_constraints,
    recreate_forecast_archive_indexes_and_constraints,
    recreate_forecast_indexes_and_constraints,
)
from common.core.utils import load_config
from common.services.perf_profiler import profiled_section

BATCH_SIZE = 2_000_000

_TABLE = "fact_external_forecast_monthly"
_ARCHIVE_TABLE = "backtest_lag_archive"

# Index/constraint specs + drop/recreate live in common/core/etl_helpers.py
# (US3, shared with load_backtest_forecasts.py) — imported above.

# Staging table column names match the CSV headers exactly (uppercase).
_STG_COLS = ["DFU", "STARTDATE", "PREDICTED_ORDERS", "FORECASTDATE", "ACTUAL_ORDERS", "FILE", "LAG"]


# ---------------------------------------------------------------------------
# Pure helpers — importable by unit tests without DB
# ---------------------------------------------------------------------------


def _normalize_date(date_str: str) -> str | None:
    """Return the month-start ISO string for a date string, or None if invalid.

    Accepts "YYYY-MM-DD" or "YYYY-MM-15" style inputs and returns "YYYY-MM-01".
    Returns None for empty, null, 'none', 'NA', or unparseable values.

    Examples
    --------
    >>> _normalize_date("2025-05-15")
    '2025-05-01'
    >>> _normalize_date("2025-05-01")
    '2025-05-01'
    >>> _normalize_date("")
    >>> _normalize_date("null")
    """
    if not date_str:
        return None
    cleaned = date_str.strip().lower()
    if cleaned in ("", "null", "none", "na", "nan"):
        return None
    try:
        # Accept YYYY-MM-DD; take only the first 10 chars to strip time components
        part = date_str.strip()[:10]
        year, month, _ = part.split("-")
        month_int = int(month)
        if not 1 <= month_int <= 12:
            return None
        return f"{int(year):04d}-{month_int:02d}-01"
    except (ValueError, AttributeError):
        return None


def _build_stg_insert_sql(model_id: str, bulk_fast: bool) -> str:
    """Return the INSERT SQL that promotes rows from _stg_ext_ml into fact_external_forecast_monthly.

    Parameters
    ----------
    model_id:
        Passed as %s placeholder in the returned SQL (the caller must supply it
        as the third bind parameter, after the two batch-range %s values).
    bulk_fast:
        When True the INSERT omits the ON CONFLICT clause for maximum throughput.
        When False an upsert (ON CONFLICT DO UPDATE) is generated.

    The returned SQL contains three %s placeholders:
        1. %s — model_id literal
        2. %s — batch_start row id (exclusive lower bound)
        3. %s — batch_end row id (inclusive upper bound)
    """
    select_expr = r"""
                d.item_id || '_' || d.customer_group || '_' || d.loc
                    || '_' || date_trunc('month', s."FORECASTDATE"::date)::date::text
                    || '_' || date_trunc('month', s."STARTDATE"::date)::date::text  AS forecast_ck,
                d.item_id,
                d.customer_group,
                d.loc,
                date_trunc('month', s."FORECASTDATE"::date)::date                  AS fcstdate,
                date_trunc('month', s."STARTDATE"::date)::date                     AS startdate,
                s."LAG"::integer                                                    AS lag,
                d.execution_lag                                                     AS execution_lag,
                CASE WHEN lower(trim(s."PREDICTED_ORDERS")) IN ('', 'null', 'none', 'na')
                     THEN NULL ELSE s."PREDICTED_ORDERS"::numeric END               AS basefcst_pref,
                CASE WHEN lower(trim(s."ACTUAL_ORDERS")) IN ('', 'null', 'none', 'na')
                     THEN NULL ELSE s."ACTUAL_ORDERS"::numeric END                  AS tothist_dmd,
                %s                                                                  AS model_id
    """
    if bulk_fast:
        return f"""
            INSERT INTO {_TABLE}
                (forecast_ck, item_id, customer_group, loc,
                 fcstdate, startdate, lag, execution_lag,
                 basefcst_pref, tothist_dmd, model_id)
            SELECT {select_expr}
            FROM _stg_ext_ml s
            JOIN dim_sku d ON d.sku_ck = trim(s."DFU")
            WHERE s._row_id > %s
              AND s._row_id <= %s
              AND s."LAG"::integer = d.execution_lag
              AND trim(s."FORECASTDATE") != ''
              AND trim(s."STARTDATE") != ''
        """
    return f"""
        INSERT INTO {_TABLE}
            (forecast_ck, item_id, customer_group, loc,
             fcstdate, startdate, lag, execution_lag,
             basefcst_pref, tothist_dmd, model_id)
        SELECT {select_expr}
        FROM _stg_ext_ml s
        JOIN dim_sku d ON d.sku_ck = trim(s."DFU")
        WHERE s._row_id > %s
          AND s._row_id <= %s
          AND s."LAG"::integer = d.execution_lag
          AND trim(s."FORECASTDATE") != ''
          AND trim(s."STARTDATE") != ''
        ON CONFLICT (forecast_ck, model_id) DO UPDATE SET
            basefcst_pref  = EXCLUDED.basefcst_pref,
            tothist_dmd    = EXCLUDED.tothist_dmd,
            execution_lag  = EXCLUDED.execution_lag,
            modified_ts    = NOW()
    """


def _build_archive_insert_sql(model_id: str, bulk_fast: bool) -> str:
    """Return the INSERT SQL that promotes rows from _stg_ext_ml into backtest_lag_archive.

    Parameters
    ----------
    model_id:
        Passed as %s placeholder — the caller supplies it as the third bind parameter.
    bulk_fast:
        When True skips ON CONFLICT for maximum throughput.

    The returned SQL contains three %s placeholders:
        1. %s — model_id literal
        2. %s — batch_start row id (exclusive lower bound)
        3. %s — batch_end row id (inclusive upper bound)
    """
    select_expr = r"""
                d.item_id || '_' || d.customer_group || '_' || d.loc
                    || '_' || date_trunc('month', s."FORECASTDATE"::date)::date::text
                    || '_' || date_trunc('month', s."STARTDATE"::date)::date::text  AS forecast_ck,
                d.item_id,
                d.customer_group,
                d.loc,
                date_trunc('month', s."FORECASTDATE"::date)::date                  AS fcstdate,
                date_trunc('month', s."STARTDATE"::date)::date                     AS startdate,
                s."LAG"::integer                                                    AS lag,
                NULL::integer                                                       AS execution_lag,
                CASE WHEN lower(trim(s."PREDICTED_ORDERS")) IN ('', 'null', 'none', 'na')
                     THEN NULL ELSE s."PREDICTED_ORDERS"::numeric END               AS basefcst_pref,
                CASE WHEN lower(trim(s."ACTUAL_ORDERS")) IN ('', 'null', 'none', 'na')
                     THEN NULL ELSE s."ACTUAL_ORDERS"::numeric END                  AS tothist_dmd,
                %s                                                                  AS model_id,
                NULL::text                                                          AS timeframe
    """
    if bulk_fast:
        return f"""
            INSERT INTO {_ARCHIVE_TABLE}
                (forecast_ck, item_id, customer_group, loc,
                 fcstdate, startdate, lag, execution_lag,
                 basefcst_pref, tothist_dmd, model_id, timeframe)
            SELECT {select_expr}
            FROM _stg_ext_ml s
            JOIN dim_sku d ON d.sku_ck = trim(s."DFU")
            WHERE s._row_id > %s
              AND s._row_id <= %s
              AND s."LAG"::integer BETWEEN 0 AND 4
              AND trim(s."FORECASTDATE") != ''
              AND trim(s."STARTDATE") != ''
        """
    return f"""
        INSERT INTO {_ARCHIVE_TABLE}
            (forecast_ck, item_id, customer_group, loc,
             fcstdate, startdate, lag, execution_lag,
             basefcst_pref, tothist_dmd, model_id, timeframe)
        SELECT {select_expr}
        FROM _stg_ext_ml s
        JOIN dim_sku d ON d.sku_ck = trim(s."DFU")
        WHERE s._row_id > %s
          AND s._row_id <= %s
          AND s."LAG"::integer BETWEEN 0 AND 4
          AND trim(s."FORECASTDATE") != ''
          AND trim(s."STARTDATE") != ''
        ON CONFLICT (forecast_ck, model_id, lag) DO UPDATE SET
            basefcst_pref  = EXCLUDED.basefcst_pref,
            tothist_dmd    = EXCLUDED.tothist_dmd,
            execution_lag  = EXCLUDED.execution_lag,
            timeframe      = EXCLUDED.timeframe
    """


# ---------------------------------------------------------------------------
# Table loaders
# ---------------------------------------------------------------------------


def _load_to_main_table(conn, cur, staged_count: int, model_id: str, bulk_fast: bool, batch_size: int) -> None:
    """Promote staged rows from _stg_ext_ml into fact_external_forecast_monthly."""
    if bulk_fast:
        print("  Dropping indexes & constraints for bulk load...")
        t0 = time.time()
        drop_forecast_indexes_and_constraints(cur)
        print(f"    Done ({time.time() - t0:.1f}s)")

    insert_sql = _build_stg_insert_sql(model_id, bulk_fast)

    print(f"  Inserting {staged_count:,} rows into {_TABLE} in batches of {batch_size:,}...")
    with profiled_section("insert_main_batches"):
        loaded_total = 0
        t_insert = time.time()
        for batch_start in range(0, staged_count, batch_size):
            batch_end = min(batch_start + batch_size, staged_count)
            cur.execute(insert_sql, (model_id, batch_start, batch_end))
            loaded_total += cur.rowcount
            elapsed = time.time() - t_insert
            rate = loaded_total / elapsed if elapsed > 0 else 0
            print(
                f"    Batch {batch_end:,}/{staged_count:,} — "
                f"{loaded_total:,} loaded ({elapsed:.0f}s, {rate:,.0f} rows/s)"
            )
        print(f"  Inserted {loaded_total:,} rows into {_TABLE} in {time.time() - t_insert:.1f}s")

    if bulk_fast:
        with profiled_section("recreate_main_indexes"):
            recreate_forecast_indexes_and_constraints(cur)


def _load_to_archive_table(conn, cur, staged_count: int, model_id: str, bulk_fast: bool, batch_size: int) -> None:
    """Promote staged rows from _stg_ext_ml into backtest_lag_archive."""
    if bulk_fast:
        print("  Dropping archive indexes & constraints for bulk load...")
        t0 = time.time()
        drop_forecast_archive_indexes_and_constraints(cur)
        print(f"    Done ({time.time() - t0:.1f}s)")

    insert_sql = _build_archive_insert_sql(model_id, bulk_fast)

    print(f"  Inserting {staged_count:,} rows into {_ARCHIVE_TABLE} in batches of {batch_size:,}...")
    with profiled_section("insert_archive_batches"):
        loaded_total = 0
        t_insert = time.time()
        for batch_start in range(0, staged_count, batch_size):
            batch_end = min(batch_start + batch_size, staged_count)
            cur.execute(insert_sql, (model_id, batch_start, batch_end))
            loaded_total += cur.rowcount
            elapsed = time.time() - t_insert
            rate = loaded_total / elapsed if elapsed > 0 else 0
            print(
                f"    Batch {batch_end:,}/{staged_count:,} — "
                f"{loaded_total:,} loaded ({elapsed:.0f}s, {rate:,.0f} rows/s)"
            )
        print(f"  Inserted {loaded_total:,} rows into {_ARCHIVE_TABLE} in {time.time() - t_insert:.1f}s")

    if bulk_fast:
        with profiled_section("recreate_archive_indexes"):
            recreate_forecast_archive_indexes_and_constraints(cur)


# ---------------------------------------------------------------------------
# Core load function
# ---------------------------------------------------------------------------


def _load_one(db: dict, csv_path: Path, model_id: str, replace: bool, batch_size: int) -> None:
    """Load one external ML forecast CSV into fact_external_forecast_monthly and backtest_lag_archive.

    Steps
    -----
    1. Connect and create a temp staging table (_stg_ext_ml).
    2. COPY the CSV into staging (streamed with progress output).
    3. Log staged row count.
    4. If replace: DELETE existing rows for model_id from both target tables.
    5. Promote staging rows into fact_external_forecast_monthly.
    6. Promote staging rows into backtest_lag_archive.
    7. COMMIT.
    8. Refresh downstream materialized views.
    """
    print(f"\n{'='*60}")
    print(f"Loading {csv_path}")
    print(f"  model_id: {model_id}")

    col_list = ", ".join(f'"{c}"' for c in _STG_COLS)
    copy_sql = (
        f'COPY _stg_ext_ml ({col_list}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)'
    )

    with psycopg.connect(**db) as conn:
        with conn.cursor() as cur:
            cur.execute("SET synchronous_commit = off")
            cur.execute("SET work_mem = '256MB'")
            cur.execute("SET maintenance_work_mem = '512MB'")

            # Create temp staging table with uppercase column names matching CSV headers.
            stg_col_defs = ", ".join(f'"{c}" TEXT' for c in _STG_COLS)
            cur.execute(f"""
                CREATE TEMP TABLE _stg_ext_ml (
                    _row_id SERIAL,
                    {stg_col_defs}
                ) ON COMMIT DROP
            """)

            # Stream CSV into staging with progress reporting.
            print("  Streaming CSV to staging table...")
            with profiled_section("stage_csv"):
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
                            print(
                                f"    COPY progress: {pct}% "
                                f"({bytes_read / (1024 * 1024):.0f} MB "
                                f"/ {file_size / (1024 * 1024):.0f} MB)"
                            )
                print(f"  Staged {bytes_read / (1024 * 1024):.0f} MB in {time.time() - t0:.1f}s")

            cur.execute("SELECT COUNT(*) FROM _stg_ext_ml")
            staged_count: int = cur.fetchone()[0]
            print(f"  Staged rows: {staged_count:,}")

            if staged_count == 0:
                print("  No rows to load.")
                conn.rollback()
                return

            # Delete existing records when --replace is active.
            if replace:
                with profiled_section("delete_existing_rows"):
                    t0 = time.time()
                    cur.execute(f"DELETE FROM {_TABLE} WHERE model_id = %s", (model_id,))
                    deleted_main = cur.rowcount
                    print(
                        f"  Deleted {deleted_main:,} existing rows from {_TABLE} "
                        f"for model_id='{model_id}' ({time.time() - t0:.1f}s)"
                    )
                    t1 = time.time()
                    cur.execute(f"DELETE FROM {_ARCHIVE_TABLE} WHERE model_id = %s", (model_id,))
                    deleted_archive = cur.rowcount
                    print(
                        f"  Deleted {deleted_archive:,} existing rows from {_ARCHIVE_TABLE} "
                        f"for model_id='{model_id}' ({time.time() - t1:.1f}s)"
                    )

            bulk_fast = replace

            _load_to_main_table(conn, cur, staged_count, model_id, bulk_fast, batch_size)
            _load_to_archive_table(conn, cur, staged_count, model_id, bulk_fast, batch_size)

        conn.commit()
        print(f"  Committed. model_id='{model_id}' loaded successfully.")

    # Refresh downstream materialized views (separate connection after commit).
    # All five MVs have unique indexes (sql/119_concurrent_mv_refresh.sql), so
    # CONCURRENTLY refresh is safe and avoids taking AccessExclusive locks
    # against the read-side. Note: REFRESH MATERIALIZED VIEW CONCURRENTLY
    # requires the MV to already be populated; for first-time refresh after
    # CREATE MATERIALIZED VIEW WITH NO DATA, drop the CONCURRENTLY keyword.
    print("  Refreshing forecast materialized views...")
    with profiled_section("refresh_forecast_views"):
        t0 = time.time()
        with psycopg.connect(**db) as conn:
            with conn.cursor() as cur:
                cur.execute("SET maintenance_work_mem = '512MB'")

                t1 = time.time()
                cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY agg_forecast_monthly")
                print(f"    agg_forecast_monthly refreshed ({time.time() - t1:.1f}s)")

                t1 = time.time()
                cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY agg_accuracy_by_dim")
                print(f"    agg_accuracy_by_dim refreshed ({time.time() - t1:.1f}s)")

                t1 = time.time()
                cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY agg_dfu_coverage")
                print(f"    agg_dfu_coverage refreshed ({time.time() - t1:.1f}s)")

                t1 = time.time()
                cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY agg_accuracy_lag_archive")
                print(f"    agg_accuracy_lag_archive refreshed ({time.time() - t1:.1f}s)")

                t1 = time.time()
                cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY agg_dfu_coverage_lag_archive")
                print(f"    agg_dfu_coverage_lag_archive refreshed ({time.time() - t1:.1f}s)")

            conn.commit()
        print(f"  All views refreshed in {time.time() - t0:.1f}s")

    print(f"  Done: model_id='{model_id}'")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load external ML forecast CSVs into Postgres",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load one model by ID (path resolved from config/forecasting/ext_ml_forecasts.yaml)
  %(prog)s --model ext_lgbm --replace

  # Load all four models defined in config
  %(prog)s --all --replace

  # Explicit path + model-id (backward-compatible)
  %(prog)s --input data/input/df_ml_lgbm_l2_extract.csv --model-id ext_lgbm --replace
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL_ID",
        help="model_id key from config/forecasting/ext_ml_forecasts.yaml (e.g. ext_lgbm)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Load all models defined in config/forecasting/ext_ml_forecasts.yaml",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        metavar="PATH",
        help="Explicit CSV path (backward-compatible; prefer --model or --all)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        dest="model_id",
        help="model_id to use when --input is supplied",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing rows for the model_id before inserting (recommended)",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    db = get_db_params()

    cfg = load_config("ext_ml_forecasts.yaml")
    models_cfg: dict = cfg.get("models", {})
    batch_size: int = int(cfg.get("batch_size", BATCH_SIZE))

    # Build the list of (csv_path, model_id) pairs to process.
    jobs: list[tuple[Path, str]] = []

    if args.all:
        if not models_cfg:
            print("Error: No models defined in config/forecasting/ext_ml_forecasts.yaml")
            sys.exit(1)
        for mid, mcfg in models_cfg.items():
            p = ROOT / mcfg["input_file"]
            if not p.exists():
                print(f"Warning: Input file not found for model '{mid}': {p} — skipping")
                continue
            jobs.append((p, mid))
        if not jobs:
            print("Error: None of the configured input files were found.")
            sys.exit(1)

    elif args.model:
        if args.model not in models_cfg:
            available = list(models_cfg.keys())
            print(f"Error: model '{args.model}' not found in config. Available: {available}")
            sys.exit(1)
        mcfg = models_cfg[args.model]
        p = ROOT / mcfg["input_file"]
        if not p.exists():
            print(f"Error: Input file not found: {p}")
            sys.exit(1)
        jobs.append((p, args.model))

    elif args.input:
        if not args.model_id:
            print("Error: --model-id is required when using --input")
            sys.exit(1)
        p = ROOT / args.input
        if not p.exists():
            print(f"Error: Input file not found: {p}")
            sys.exit(1)
        jobs.append((p, args.model_id))

    else:
        parser.print_help()
        print("\nError: Specify one of --model MODEL_ID, --all, or --input PATH --model-id MODEL_ID")
        sys.exit(1)

    model_labels = [mid for _, mid in jobs]
    print(f"Loading {len(jobs)} model(s): {model_labels}")

    for csv_path, model_id in jobs:
        _load_one(db, csv_path, model_id, args.replace, batch_size)

    print(f"\n{'='*60}")
    print(f"All done. Loaded: {model_labels}")


if __name__ == "__main__":
    main()
