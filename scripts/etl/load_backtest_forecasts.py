"""
Load backtest predictions into Postgres.

Loads execution-lag predictions into fact_external_forecast_monthly (main table)
and all-lag predictions into backtest_lag_archive (archive table).

Uses COPY + temp table pattern (same as load_dataset_postgres.py).
Supports --replace to delete existing rows for a model_id before inserting.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.constants import FORECAST_QTY_COL
from common.core.db import get_db_params
from common.core.etl_helpers import (
    drop_forecast_archive_indexes_and_constraints,
    drop_forecast_indexes_and_constraints,
    perf_setting,
    recreate_forecast_archive_indexes_and_constraints,
    recreate_forecast_indexes_and_constraints,
)
from common.services.perf_profiler import profiled_section

logger = logging.getLogger(__name__)

LOAD_COLS = [
    "forecast_ck", "item_id", "customer_group", "loc",
    "fcstdate", "startdate", "lag", "execution_lag",
    FORECAST_QTY_COL, "tothist_dmd", "model_id",
]

BATCH_SIZE = perf_setting("batch_size", 2_000_000)

_TABLE = "fact_external_forecast_monthly"

# ── Archive table constants ────────────────────────────────────────────────
_ARCHIVE_TABLE = "backtest_lag_archive"
ARCHIVE_COLS = [
    "forecast_ck", "item_id", "customer_group", "loc",
    "fcstdate", "startdate", "lag", "execution_lag",
    FORECAST_QTY_COL, "tothist_dmd", "model_id", "timeframe",
]

# Index/constraint specs + drop/recreate for both the main forecast table and
# the archive table live in common/core/etl_helpers.py (US3, shared with
# load_ext_ml_forecasts.py) — imported above as drop_forecast_*/recreate_forecast_*.


def _load_archive(db: dict, archive_path: Path, model_ids: list[str], replace: bool, model_id_filter: str | None, skip_index_ops: bool = False) -> None:
    """Load all-lags CSV into backtest_lag_archive table."""
    if not archive_path.exists():
        logger.info(f"\nArchive file not found: {archive_path} — skipping archive load")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"Loading archive from {archive_path}")
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
            logger.info("  Streaming archive CSV to staging table...")
            with profiled_section("archive_stage_csv"):
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
                            logger.info(f"    COPY progress: {pct}% ({bytes_read / (1024*1024):.0f} MB / {file_size / (1024*1024):.0f} MB)")
                logger.info(f"  Staged {bytes_read / (1024*1024):.0f} MB in {time.time() - t0:.1f}s")

            if model_id_filter:
                cur.execute("DELETE FROM _stg_archive WHERE model_id != %s", (model_id_filter,))

            cur.execute("SELECT COUNT(*) FROM _stg_archive")
            staged_count = cur.fetchone()[0]
            logger.info(f"  Staged archive rows: {staged_count:,}")

            if staged_count == 0:
                logger.info("  No archive rows to load.")
                conn.rollback()
                return

            # Delete existing if --replace
            if replace:
                with profiled_section("archive_delete_existing"):
                    t0 = time.time()
                    for mid in model_ids:
                        cur.execute(f"DELETE FROM {_ARCHIVE_TABLE} WHERE model_id = %s", (mid,))
                        deleted = cur.rowcount
                        logger.info(f"  Deleted {deleted:,} existing archive rows for model_id='{mid}' ({time.time() - t0:.1f}s)")

            use_plain_insert = skip_index_ops or replace
            if not skip_index_ops and replace:
                logger.info("  Dropping archive indexes & constraints for bulk load...")
                t0 = time.time()
                drop_forecast_archive_indexes_and_constraints(cur)
                logger.info(f"    Done ({time.time() - t0:.1f}s)")

            # Insert with type casting
            select_expr = """
                    s.forecast_ck,
                    s.item_id,
                    s.customer_group,
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
            if use_plain_insert:
                insert_sql = f"""
                    INSERT INTO {_ARCHIVE_TABLE}
                        (forecast_ck, item_id, customer_group, loc,
                         fcstdate, startdate, lag, execution_lag,
                         basefcst_pref, tothist_dmd, model_id, timeframe)
                    SELECT {select_expr}
                    FROM _stg_archive s
                    WHERE s._row_id > %s AND s._row_id <= %s
                """
            else:
                insert_sql = f"""
                    INSERT INTO {_ARCHIVE_TABLE}
                        (forecast_ck, item_id, customer_group, loc,
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

            with profiled_section("archive_insert_batches"):
                loaded_total = 0
                t_insert = time.time()
                for batch_start in range(0, staged_count, BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, staged_count)
                    cur.execute(insert_sql, (batch_start, batch_end))
                    loaded_total += cur.rowcount
                    elapsed = time.time() - t_insert
                    rate = loaded_total / elapsed if elapsed > 0 else 0
                    logger.info(f"    Batch {batch_end:,}/{staged_count:,} — {loaded_total:,} loaded ({elapsed:.0f}s, {rate:,.0f} rows/s)")

                logger.info(f"  Inserted {loaded_total:,} archive rows in {time.time() - t_insert:.1f}s")

            if not skip_index_ops and replace:
                with profiled_section("archive_recreate_indexes"):
                    recreate_forecast_archive_indexes_and_constraints(cur)

        conn.commit()
    logger.info("Archive load complete.")


def _resolve_input_files(
    args_input: str | None,
    args_model: str | None,
    args_all: bool,
    backtest_dir: Path,
    args_models: list[str] | None = None,
) -> list[Path]:
    """Return list of prediction CSVs to load based on CLI flags.

    Priority order:
      --all               → scan data/backtest/*/backtest_predictions.csv
      --models M1 M2 ...  → load specific models
      --model MODEL_ID    → data/backtest/<MODEL_ID>/backtest_predictions.csv
      --input PATH        → explicit path (backward-compatible)
    """
    if args_models:
        found = []
        for model_id in args_models:
            p = backtest_dir / model_id / "backtest_predictions.csv"
            if not p.exists():
                logger.info(f"Warning: No predictions found for model '{model_id}' at {p} — skipping")
                continue
            found.append(p)
        if not found:
            logger.info(f"Error: No prediction files found for models: {args_models}")
            sys.exit(1)
        return found

    if args_all:
        # Only load from canonical model directories — skip auxiliary dirs
        # like lgbm_cluster_baseline_20260322, logs, tuning_archive that
        # contain stale CSVs which would overwrite real backtest data.
        _CANONICAL_MODEL_DIRS = {"lgbm_cluster", "catboost_cluster", "xgboost_cluster", "chronos"}
        found = sorted(
            p for p in backtest_dir.glob("*/backtest_predictions.csv")
            if p.parent.name in _CANONICAL_MODEL_DIRS
        )
        if not found:
            logger.info(f"No backtest_predictions.csv files found under {backtest_dir}/*/")
            logger.info(f"  (looked in canonical dirs: {sorted(_CANONICAL_MODEL_DIRS)})")
            logger.info("  Run a backtest first: make backtest-lgbm-cluster")
            sys.exit(1)
        return found

    if args_model:
        p = backtest_dir / args_model / "backtest_predictions.csv"
        if not p.exists():
            available = [d.name for d in backtest_dir.iterdir() if d.is_dir() and
                         (d / "backtest_predictions.csv").exists()] if backtest_dir.exists() else []
            logger.info(f"Error: No predictions found for model '{args_model}' at {p}")
            if available:
                logger.info(f"  Available models: {available}")
            else:
                logger.info("  No backtest output directories found — run a backtest first.")
            sys.exit(1)
        return [p]

    if args_input:
        p = ROOT / args_input
        if not p.exists():
            logger.info(f"Error: Input file not found: {p}")
            sys.exit(1)
        return [p]

    logger.info("Error: Specify one of:")
    logger.info("  --model MODEL_ID     load a specific model  (e.g. --model lgbm_cluster)")
    logger.info("  --all                load all models from data/backtest/*/")
    logger.info("  --input PATH         explicit CSV path (legacy)")
    sys.exit(1)


def _load_one(
    db: dict,
    csv_path: Path,
    replace: bool,
    model_id_filter: str | None,
    skip_index_ops: bool = False,
    main_only: bool = False,
    archive_only: bool = False,
) -> None:
    """Load one backtest_predictions.csv + its sibling all_lags CSV into Postgres.

    Args:
        skip_index_ops: When True, skip index drop/create and MV refresh.
            Used by --bulk mode which handles these once for all models.
        main_only: Load only fact_external_forecast_monthly, skip archive.
        archive_only: Load only backtest_lag_archive, skip main table.
    """
    # Peek at CSV to determine model_ids present
    sample = pd.read_csv(csv_path, nrows=100)
    csv_model_ids = sample["model_id"].unique().tolist()
    if model_id_filter:
        csv_model_ids = [model_id_filter]

    logger.info(f"\n{'='*60}")
    logger.info(f"Loading {csv_path}")
    logger.info(f"  Model IDs in file: {csv_model_ids}")

    col_list = ", ".join(LOAD_COLS)

    if archive_only:
        # Skip main table entirely — jump to archive load
        archive_path = csv_path.parent / "backtest_predictions_all_lags.csv"
        _load_archive(db, archive_path, csv_model_ids, replace, model_id_filter,
                      skip_index_ops=skip_index_ops)
        if not skip_index_ops:
            logger.info("  Refreshing archive accuracy views...")
            with profiled_section("refresh_archive_views"):
                t0 = time.time()
                with psycopg.connect(**db) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SET maintenance_work_mem = '512MB'")
                        cur.execute("REFRESH MATERIALIZED VIEW agg_accuracy_lag_archive")
                        cur.execute("REFRESH MATERIALIZED VIEW agg_dfu_coverage_lag_archive")
                    conn.commit()
                logger.info(f"  Archive views refreshed in {time.time() - t0:.1f}s")
        logger.info(f"  Done (archive only): {csv_path.parent.name}")
        return

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
            logger.info("  Streaming CSV to staging table...")
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
                            logger.info(f"    COPY progress: {pct}% ({bytes_read / (1024*1024):.0f} MB / {file_size / (1024*1024):.0f} MB)")
                logger.info(f"  Staged {bytes_read / (1024*1024):.0f} MB in {time.time() - t0:.1f}s")

            if model_id_filter:
                cur.execute("DELETE FROM _stg_backtest WHERE model_id != %s", (model_id_filter,))

            cur.execute("SELECT COUNT(*) FROM _stg_backtest")
            staged_count = cur.fetchone()[0]
            logger.info(f"  Staged rows: {staged_count:,}")

            if staged_count == 0:
                logger.info("  No rows to load.")
                conn.rollback()
                return

            if replace:
                with profiled_section("delete_existing_rows"):
                    t0 = time.time()
                    for mid in csv_model_ids:
                        cur.execute(f"DELETE FROM {_TABLE} WHERE model_id = %s", (mid,))
                        deleted = cur.rowcount
                        logger.info(f"  Deleted {deleted:,} existing rows for model_id='{mid}' ({time.time() - t0:.1f}s)")

            # Use plain INSERT when indexes are dropped (bulk or skip_index_ops),
            # ON CONFLICT upsert when unique constraint is present.
            use_plain_insert = skip_index_ops or replace
            if not skip_index_ops and replace:
                logger.info("  Dropping indexes & constraints for bulk load...")
                t0 = time.time()
                drop_forecast_indexes_and_constraints(cur)
                logger.info(f"    Done ({time.time() - t0:.1f}s)")

            logger.info(f"  Inserting {staged_count:,} rows in batches of {BATCH_SIZE:,}...")
            select_expr = """
                    s.forecast_ck,
                    s.item_id,
                    s.customer_group,
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
            if use_plain_insert:
                insert_sql = f"""
                    INSERT INTO {_TABLE}
                        (forecast_ck, item_id, customer_group, loc,
                         fcstdate, startdate, lag, execution_lag,
                         basefcst_pref, tothist_dmd, model_id)
                    SELECT {select_expr}
                    FROM _stg_backtest s
                    WHERE s._row_id > %s AND s._row_id <= %s
                """
            else:
                insert_sql = f"""
                    INSERT INTO {_TABLE}
                        (forecast_ck, item_id, customer_group, loc,
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

            with profiled_section("insert_batches"):
                loaded_total = 0
                t_insert = time.time()
                for batch_start in range(0, staged_count, BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, staged_count)
                    cur.execute(insert_sql, (batch_start, batch_end))
                    loaded_total += cur.rowcount
                    elapsed = time.time() - t_insert
                    rate = loaded_total / elapsed if elapsed > 0 else 0
                    logger.info(f"    Batch {batch_end:,}/{staged_count:,} — {loaded_total:,} loaded ({elapsed:.0f}s, {rate:,.0f} rows/s)")

                logger.info(f"  Inserted {loaded_total:,} rows in {time.time() - t_insert:.1f}s")

            if not skip_index_ops and replace:
                with profiled_section("recreate_indexes"):
                    recreate_forecast_indexes_and_constraints(cur)

        conn.commit()

    # Load archive (sibling file lives in the same model subdirectory)
    if not main_only:
        archive_path = csv_path.parent / "backtest_predictions_all_lags.csv"
        _load_archive(db, archive_path, csv_model_ids, replace, model_id_filter,
                      skip_index_ops=skip_index_ops)

    if skip_index_ops:
        logger.info(f"  Done (bulk mode — skipping MV refresh): {csv_path.parent.name}")
        return

    # Refresh forecast materialized views
    logger.info("  Refreshing forecast materialized views...")
    with profiled_section("refresh_forecast_views"):
        t0 = time.time()
        with psycopg.connect(**db) as conn:
            with conn.cursor() as cur:
                cur.execute("SET maintenance_work_mem = '512MB'")
                cur.execute("REFRESH MATERIALIZED VIEW agg_forecast_monthly")
                logger.info(f"    agg_forecast_monthly refreshed ({time.time() - t0:.1f}s)")
                t1 = time.time()
                cur.execute("REFRESH MATERIALIZED VIEW agg_accuracy_by_dim")
                logger.info(f"    agg_accuracy_by_dim refreshed ({time.time() - t1:.1f}s)")
                t2 = time.time()
                cur.execute("REFRESH MATERIALIZED VIEW agg_dfu_coverage")
                logger.info(f"    agg_dfu_coverage refreshed ({time.time() - t2:.1f}s)")
            conn.commit()
        logger.info(f"  Forecast views refreshed in {time.time() - t0:.1f}s")

    # Refresh archive accuracy views
    logger.info("  Refreshing archive accuracy views...")
    with profiled_section("refresh_archive_views"):
        t0 = time.time()
        with psycopg.connect(**db) as conn:
            with conn.cursor() as cur:
                cur.execute("SET maintenance_work_mem = '512MB'")
                cur.execute("REFRESH MATERIALIZED VIEW agg_accuracy_lag_archive")
                cur.execute("REFRESH MATERIALIZED VIEW agg_dfu_coverage_lag_archive")
            conn.commit()
        logger.info(f"  Archive views refreshed in {time.time() - t0:.1f}s")
    logger.info(f"  Done: {csv_path.parent.name}")


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
        help="Load a single model from data/backtest/<MODEL_ID>/",
    )
    parser.add_argument(
        "--models", nargs="+", default=None, metavar="MODEL_ID",
        help="Load multiple models: --models lgbm_cluster chronos catboost_cluster",
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
    parser.add_argument(
        "--bulk", action="store_true",
        help="With multiple models + --replace: drop/recreate indexes ONCE "
             "(~4x faster than per-model index management)",
    )
    parser.add_argument(
        "--main-only", action="store_true",
        help="Load only fact_external_forecast_monthly (skip archive)",
    )
    parser.add_argument(
        "--archive-only", action="store_true",
        help="Load only backtest_lag_archive (skip main table)",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    db = get_db_params()

    backtest_dir = ROOT / "data" / "backtest"
    csv_files = _resolve_input_files(args.input, args.model, args.all, backtest_dir, args.models)

    model_labels = [f.parent.name for f in csv_files]
    logger.info(f"Loading {len(csv_files)} model(s): {model_labels}")

    if args.bulk and len(csv_files) >= 1 and args.replace:
        # Bulk mode: drop indexes ONCE, load all models, recreate ONCE,
        # refresh MVs ONCE.  ~4x faster than per-model index management.
        load_main = not args.archive_only
        load_archive = not args.main_only

        logger.info("\n[bulk] Dropping indexes & constraints once for all models...")
        with psycopg.connect(**db) as conn:
            with conn.cursor() as cur:
                if load_main:
                    drop_forecast_indexes_and_constraints(cur)
                if load_archive:
                    drop_forecast_archive_indexes_and_constraints(cur)
            conn.commit()
        logger.info("[bulk] Indexes dropped. Loading models without per-model index ops...")

        for csv_path in csv_files:
            _load_one(db, csv_path, args.replace, args.model_id, skip_index_ops=True,
                      main_only=args.main_only, archive_only=args.archive_only)

        logger.info("\n[bulk] Recreating indexes & constraints once for all models...")
        with psycopg.connect(**db) as conn:
            with conn.cursor() as cur:
                cur.execute("SET maintenance_work_mem = '512MB'")
                if load_main:
                    recreate_forecast_indexes_and_constraints(cur)
                if load_archive:
                    recreate_forecast_archive_indexes_and_constraints(cur)
            conn.commit()

        main_mvs = ["agg_forecast_monthly", "agg_accuracy_by_dim", "agg_dfu_coverage"]
        archive_mvs = ["agg_accuracy_lag_archive", "agg_dfu_coverage_lag_archive"]
        mvs = []
        if load_main:
            mvs.extend(main_mvs)
        if load_archive:
            mvs.extend(archive_mvs)

        logger.info("[bulk] Refreshing materialized views once...")
        with profiled_section("bulk_refresh_views"):
            t0 = time.time()
            with psycopg.connect(**db) as conn:
                with conn.cursor() as cur:
                    cur.execute("SET maintenance_work_mem = '512MB'")
                    for mv in mvs:
                        t1 = time.time()
                        cur.execute(f"REFRESH MATERIALIZED VIEW {mv}")
                        logger.info(f"    {mv} refreshed ({time.time() - t1:.1f}s)")
                conn.commit()
            logger.info(f"  All views refreshed in {time.time() - t0:.1f}s")
    else:
        for csv_path in csv_files:
            _load_one(db, csv_path, args.replace, args.model_id,
                      main_only=args.main_only, archive_only=args.archive_only)

    logger.info(f"\n{'='*60}")
    logger.info(f"All done. Loaded: {model_labels}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
