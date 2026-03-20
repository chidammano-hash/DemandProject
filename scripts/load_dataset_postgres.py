import argparse
import logging
import time
from pathlib import Path
import sys

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.db import get_db_params
from common.domain_specs import DOMAIN_SPECS, DomainSpec, get_spec
from common.sql_helpers import (
    NULL_SQL,
    EXTERNAL_MODEL_ID,
    HASH_CHUNK_SIZE,
    MV_REFRESH_ARCHIVE,
    _elapsed,
    qident,
    typed_expr_sets,
    business_key_expr,
)

logger = logging.getLogger(__name__)

# Unmatched DFU warning threshold (E5)
_UNMATCHED_DFU_WARN_PCT = 10.0

# PG session tuning defaults (M6) — can be overridden via medallion_config.yaml
_PG_WORK_MEM = "512MB"
_PG_MAINTENANCE_WORK_MEM = "1GB"


def _get_all_indexes(cur, table: str) -> list[tuple[str, str]]:
    """Return list of (index_name, index_def) for ALL indexes on *table*,
    excluding only the primary key."""
    cur.execute("""
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = %s
          AND indexname NOT LIKE '%%_pkey'
        ORDER BY indexname;
    """, (table,))
    return cur.fetchall()


def _get_unique_constraints(cur, table: str) -> list[tuple[str, str, list[str]]]:
    """Return list of (constraint_name, constraint_type, [columns]) for UNIQUE
    constraints on *table* (excludes PK)."""
    cur.execute("""
        SELECT con.conname,
               con.contype::text,
               array_agg(att.attname ORDER BY u.pos)
        FROM pg_constraint con
        JOIN pg_class rel ON rel.oid = con.conrelid
        JOIN LATERAL unnest(con.conkey) WITH ORDINALITY AS u(attnum, pos) ON true
        JOIN pg_attribute att ON att.attrelid = con.conrelid AND att.attnum = u.attnum
        WHERE rel.relname = %s
          AND con.contype = 'u'
        GROUP BY con.conname, con.contype;
    """, (table,))
    return [(r[0], r[1], r[2]) for r in cur.fetchall()]


def _drop_indexes(cur, indexes: list[tuple[str, str]]) -> None:
    for idx_name, _ in indexes:
        cur.execute(f"DROP INDEX IF EXISTS {qident(idx_name)};")


def _drop_unique_constraints(cur, table: str,
                             constraints: list[tuple[str, str, list[str]]]) -> None:
    for con_name, _, _ in constraints:
        cur.execute(
            f"ALTER TABLE {qident(table)} DROP CONSTRAINT IF EXISTS {qident(con_name)};"
        )


def _recreate_indexes(cur, indexes: list[tuple[str, str]]) -> None:
    for _, idx_def in indexes:
        cur.execute(idx_def + ";")


def _recreate_unique_constraints(cur, table: str,
                                 constraints: list[tuple[str, str, list[str]]]) -> None:
    for con_name, _, cols in constraints:
        col_list = ", ".join(qident(c) for c in cols)
        cur.execute(
            f"ALTER TABLE {qident(table)} ADD CONSTRAINT {qident(con_name)} "
            f"UNIQUE ({col_list});"
        )


def _resolve_forecast_execution_lag(cur, stg_table: str) -> tuple[int, int]:
    """For forecast domain: update staging execution_lag from dim_dfu.

    The normalized CSV sets execution_lag = lag for every row (since the source
    file doesn't have an execution_lag column).  We resolve the actual DFU
    execution lag from dim_dfu so we can filter to execution-lag rows only
    for the main table INSERT.

    IMPORTANT: This MUST be called AFTER the archive load, because the archive
    needs the untouched staging data where each row's execution_lag = its own lag.

    Returns (matched_rows, unmatched_rows).
    """
    # Check if dim_dfu exists
    cur.execute("""
        SELECT EXISTS(
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'dim_dfu' AND table_schema = 'public'
        )
    """)
    if not cur.fetchone()[0]:
        logger.info("dim_dfu table not found -- defaulting execution_lag to 0")
        cur.execute(f'UPDATE {qident(stg_table)} SET "execution_lag" = \'0\'')
        return (0, cur.rowcount)

    # Update staging execution_lag from dim_dfu (all rows for matched DFUs)
    cur.execute(f"""
        UPDATE {qident(stg_table)} s
        SET "execution_lag" = COALESCE(d.execution_lag, 0)::text
        FROM dim_dfu d
        WHERE d.dfu_ck = trim(s."dmdunit") || '_' || trim(s."dmdgroup") || '_' || trim(s."loc")
    """)
    matched_rows = cur.rowcount

    # Default execution_lag to 0 for DFUs not in dim_dfu
    cur.execute(f"""
        UPDATE {qident(stg_table)} s
        SET "execution_lag" = '0'
        WHERE NOT EXISTS (
            SELECT 1 FROM dim_dfu d
            WHERE d.dfu_ck = trim(s."dmdunit") || '_' || trim(s."dmdgroup") || '_' || trim(s."loc")
        )
    """)
    unmatched_rows = cur.rowcount

    # E5: warn if unmatched DFU percentage exceeds threshold
    total = (matched_rows or 0) + (unmatched_rows or 0)
    if total > 0:
        unmatched_pct = ((unmatched_rows or 0) / total) * 100
        if unmatched_pct > _UNMATCHED_DFU_WARN_PCT:
            logger.warning(
                "%.1f%% of forecast rows (%d/%d) could not match a DFU in dim_dfu "
                "(defaulted to execution_lag=0). Threshold: %.1f%%",
                unmatched_pct, unmatched_rows, total, _UNMATCHED_DFU_WARN_PCT,
            )

    return (matched_rows, unmatched_rows)


def _load_forecast_archive(cur, stg_table: str, stg_alias: str) -> int:
    """Load ALL forecast rows from staging into backtest_lag_archive.

    This preserves all 5 lags (0-4) for multi-lag accuracy analysis while
    the main table holds only execution-lag rows.
    """
    archive_table = "backtest_lag_archive"

    # Delete existing external rows from archive
    cur.execute(
        f"DELETE FROM {archive_table} WHERE model_id = %s",
        [EXTERNAL_MODEL_ID],
    )
    deleted = cur.rowcount
    if deleted:
        logger.info("Deleted %s existing '%s' archive rows", f"{deleted:,}", EXTERNAL_MODEL_ID)

    # Build forecast_ck expression (same separator as FORECAST_SPEC)
    ck_expr = (
        f"trim({stg_alias}.\"dmdunit\") || '_' || trim({stg_alias}.\"dmdgroup\") || '_' || "
        f"trim({stg_alias}.\"loc\") || '_' || trim({stg_alias}.\"fcstdate\") || '_' || "
        f"trim({stg_alias}.\"startdate\")"
    )

    insert_sql = f"""
        INSERT INTO {archive_table}
            (forecast_ck, dmdunit, dmdgroup, loc, fcstdate, startdate,
             lag, execution_lag, basefcst_pref, tothist_dmd, model_id, timeframe)
        SELECT
            {ck_expr},
            {stg_alias}."dmdunit",
            {stg_alias}."dmdgroup",
            {stg_alias}."loc",
            {stg_alias}."fcstdate"::date,
            {stg_alias}."startdate"::date,
            {stg_alias}."lag"::integer,
            CASE WHEN lower(trim({stg_alias}."execution_lag")) IN ({NULL_SQL})
                 THEN NULL ELSE {stg_alias}."execution_lag"::integer END,
            CASE WHEN lower(trim({stg_alias}."basefcst_pref")) IN ({NULL_SQL})
                 THEN NULL ELSE {stg_alias}."basefcst_pref"::numeric END,
            CASE WHEN lower(trim({stg_alias}."tothist_dmd")) IN ({NULL_SQL})
                 THEN NULL ELSE {stg_alias}."tothist_dmd"::numeric END,
            {stg_alias}."model_id",
            NULL
        FROM {qident(stg_table)} {stg_alias}
        ON CONFLICT (forecast_ck, model_id, lag) DO UPDATE SET
            basefcst_pref = EXCLUDED.basefcst_pref,
            tothist_dmd   = EXCLUDED.tothist_dmd,
            execution_lag = EXCLUDED.execution_lag
    """
    cur.execute(insert_sql)
    return cur.rowcount


def _run_medallion_pipeline(spec: DomainSpec, csv_path: Path,
                            apply_fixes: bool = False) -> None:
    """Run the full medallion pipeline: Bronze -> Silver -> Gold."""
    from common.medallion import (
        create_batch, complete_batch, fail_batch, file_hash,
        ingest_bronze, promote_to_silver, run_silver_dq_gate,
        apply_silver_fixes, promote_to_gold, write_lineage,
    )

    t_total = time.time()
    logger.info("=" * 60)
    logger.info("Medallion pipeline: %s -> bronze -> silver -> gold", spec.name)
    logger.info("CSV: %s", csv_path.name)
    if apply_fixes:
        logger.info("DQ auto-fixes: ENABLED")
    logger.info("=" * 60)

    if not csv_path.exists():
        logger.info("SKIPPED -- %s not found.", csv_path.name)
        logger.info("Run 'make normalize-all' first to generate clean CSVs.")
        return

    db = get_db_params()
    src_hash = file_hash(csv_path)

    with psycopg.connect(**db) as conn, conn.cursor() as cur:
        batch_id = create_batch(
            cur, spec.name, source_file=csv_path.name,
            source_hash=src_hash,
        )
        conn.commit()
        logger.info("[1/6] Created batch %s", batch_id)

        try:
            # Phase 1: Bronze ingest
            logger.info("[2/6] Bronze ingest: COPY -> bronze_%s ...", spec.name)
            t0 = time.time()
            bronze_count = ingest_bronze(cur, spec, csv_path, batch_id)
            conn.commit()
            logger.info("       %s rows ingested (%s)", f"{bronze_count:,}", _elapsed(t0))

            # Phase 2: Silver promotion (type cast + dedup)
            logger.info("[3/6] Silver promotion: bronze -> silver_%s ...", spec.name)
            t0 = time.time()
            silver_count, quarantine_count = promote_to_silver(cur, spec, batch_id)
            conn.commit()
            logger.info("       %s rows promoted (%s)", f"{silver_count:,}", _elapsed(t0))

            # Phase 3: DQ gate checks
            logger.info("[4/6] DQ gate checks ...")
            t0 = time.time()
            gate = run_silver_dq_gate(cur, spec, batch_id)
            conn.commit()
            logger.info("       Pass rate: %s%% (%s passed, %s quarantined) (%s)",
                        gate['pass_rate'], f"{gate.get('passed_count', 0):,}",
                        f"{gate['quarantined']:,}", _elapsed(t0))
            if not gate["passed"]:
                logger.warning("Below min pass rate %s%%", gate['min_pass_rate'])

            # Phase 4: Auto-fixes (optional)
            fix_result = {"fixes_applied": 0}
            if apply_fixes:
                logger.info("[4b/6] Applying DQ auto-fixes ...")
                t0 = time.time()
                fix_result = apply_silver_fixes(cur, spec, batch_id)
                conn.commit()
                logger.info("       %d fixes applied (%s)", fix_result['fixes_applied'], _elapsed(t0))

            # Phase 5: Gold promotion
            logger.info("[5/6] Gold promotion: silver -> %s ...", spec.table)
            t0 = time.time()
            gold = promote_to_gold(cur, spec, batch_id)
            conn.commit()
            logger.info("       %s rows -> %s (%s)", f"{gold['gold_count']:,}", gold['gold_table'], _elapsed(t0))
            if gold.get("original_count"):
                logger.info("       %s rows -> fact_sales_monthly_original", f"{gold['original_count']:,}")

            # Phase 6: Lineage
            logger.info("[6/6] Writing lineage records ...")
            t0 = time.time()
            lineage_count = write_lineage(cur, spec, batch_id)
            conn.commit()
            logger.info("       %s lineage rows (%s)", f"{lineage_count:,}", _elapsed(t0))

            # Complete batch
            complete_batch(
                cur, batch_id,
                row_count_in=bronze_count,
                row_count_out=gold["gold_count"],
                quarantined=gate["quarantined"],
            )
            conn.commit()

        except Exception as exc:
            fail_batch(cur, batch_id, str(exc))
            conn.commit()
            raise

    total = _elapsed(t_total)
    logger.info("=" * 60)
    logger.info("Medallion load complete: %s", spec.name)
    logger.info("  Bronze: %s rows", f"{bronze_count:,}")
    logger.info("  Silver: %s rows (gate: %s%%)", f"{silver_count:,}", gate['pass_rate'])
    logger.info("  Gold:   %s rows", f"{gold['gold_count']:,}")
    if fix_result["fixes_applied"]:
        logger.info("  Fixes:  %d", fix_result['fixes_applied'])
    logger.info("  Time:   %s", total)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# L1 / S1: Legacy (non-medallion) load extracted into its own function
# ---------------------------------------------------------------------------

def _run_legacy_load(
    spec: DomainSpec,
    csv_path: Path,
    no_dedup: bool,
    fast_mode: bool,
    replace_mode: bool,
    skip_archive: bool,
) -> None:
    """Run the legacy (pre-medallion) direct COPY load pipeline (L1)."""
    from common.medallion import fail_batch, create_batch, complete_batch

    csv_size_mb = csv_path.stat().st_size / (1024 ** 2) if csv_path.exists() else 0
    db = get_db_params()

    target_cols = [spec.ck_field, *spec.columns]
    stg_table = f"stg_{spec.table}_{spec.name}"
    src_alias = "s"
    stg_alias = "d"

    load_seq_col = "_load_seq"
    create_stage_sql = (
        f"CREATE TEMP TABLE {qident(stg_table)} ("
        + f"{qident(load_seq_col)} bigserial, "
        + ", ".join([f"{qident(c)} text" for c in spec.columns])
        + ") ON COMMIT DROP;"
    )

    copy_sql = (
        f"COPY {qident(stg_table)} ("
        + ", ".join([qident(c) for c in spec.columns])
        + ") FROM STDIN WITH (FORMAT CSV, HEADER TRUE)"
    )

    key_col = "_ck"
    select_exprs = [
        f"{src_alias}.{qident(key_col)} AS {qident(spec.ck_field)}",
        *[
            f"{typed_expr_sets(c, spec.int_fields, spec.float_fields, spec.date_fields, src_alias)} AS {qident(c)}"
            for c in spec.columns
        ],
    ]

    truncate_sql = f"TRUNCATE TABLE {qident(spec.table)};"

    if no_dedup:
        insert_sql = (
            f"INSERT INTO {qident(spec.table)} ("
            + ", ".join([qident(c) for c in target_cols])
            + ") SELECT "
            + ", ".join(select_exprs)
            + " FROM (SELECT *, "
            + business_key_expr(spec, stg_alias)
            + " AS "
            + qident(key_col)
            + " FROM "
            + qident(stg_table)
            + " "
            + stg_alias
            + f") {src_alias};"
        )
    else:
        insert_sql = (
            f"INSERT INTO {qident(spec.table)} ("
            + ", ".join([qident(c) for c in target_cols])
            + ") SELECT "
            + ", ".join(select_exprs)
            + " FROM (SELECT DISTINCT ON ("
            + qident(key_col)
            + ") * FROM (SELECT *, "
            + business_key_expr(spec, stg_alias)
            + " AS "
            + qident(key_col)
            + " FROM "
            + qident(stg_table)
            + " "
            + stg_alias
            + ") x ORDER BY "
            + qident(key_col)
            + ", "
            + qident(load_seq_col)
            + f" DESC) {src_alias};"
        )

    # ---- Header ----
    t_total = time.time()
    mode_flags = []
    if replace_mode:
        mode_flags.append("replace external only")
    if skip_archive:
        mode_flags.append("skip-archive")
    if no_dedup:
        mode_flags.append("no-dedup")
    if fast_mode:
        mode_flags.append("fast")
    mode_label = f" [{', '.join(mode_flags)}]" if mode_flags else ""
    logger.info("=" * 60)
    logger.info("Loading %s -> %s%s", spec.name, spec.table, mode_label)
    logger.info("CSV: %s (%s MB)", csv_path.name, f"{csv_size_mb:,.0f}")
    logger.info("=" * 60)

    saved_indexes: list[tuple[str, str]] = []
    saved_constraints: list[tuple[str, str, list[str]]] = []

    is_forecast = spec.name == "forecast"

    with psycopg.connect(**db) as conn, conn.cursor() as cur:
        # E1: wrap entire legacy load in try/except
        try:
            # ---- Phase 1: Session tuning (fast mode) ----
            if fast_mode:
                logger.info("[1/6] Tuning session for bulk load ...")
                cur.execute(f"SET work_mem = '{_PG_WORK_MEM}';")
                cur.execute(f"SET maintenance_work_mem = '{_PG_MAINTENANCE_WORK_MEM}';")
                cur.execute("SET synchronous_commit = 'off';")
                logger.info("       work_mem=%s, maintenance_work_mem=%s, synchronous_commit=off",
                            _PG_WORK_MEM, _PG_MAINTENANCE_WORK_MEM)
            else:
                logger.info("[1/6] Session defaults (use --fast for tuned bulk load)")

            # ---- Phase 2: Create staging table + COPY ----
            logger.info("[2/6] COPY %s -> staging table ...", csv_path.name)
            t0 = time.time()
            cur.execute(create_stage_sql)
            with cur.copy(copy_sql) as copy, csv_path.open("r", encoding="utf-8", newline="") as f:
                bytes_read = 0
                while chunk := f.read(HASH_CHUNK_SIZE):
                    copy.write(chunk)
                    bytes_read += len(chunk)
                    if bytes_read % (100 * HASH_CHUNK_SIZE) < HASH_CHUNK_SIZE:
                        pct = (bytes_read / (csv_size_mb * 1024 * 1024) * 100) if csv_size_mb > 0 else 0
                        logger.info("       %s MB copied (%.0f%%) ...", f"{bytes_read / (1024**2):,.0f}", pct)
            t_copy = time.time() - t0
            rate_mb = (bytes_read / (1024 ** 2)) / t_copy if t_copy > 0 else 0
            logger.info("       Done in %s (%s MB/s)", _elapsed(t0), f"{rate_mb:,.0f}")

            # ---- Phase 3: Count staging rows ----
            load_archive = is_forecast and not skip_archive
            total_phases = 8 if is_forecast else 6

            logger.info("[3/%d] Counting staging rows ...", total_phases)
            t0 = time.time()
            cur.execute(f"SELECT count(*) FROM {qident(stg_table)};")
            stg_rows = cur.fetchone()[0]
            logger.info("       %s rows in staging (%s)", f"{stg_rows:,}", _elapsed(t0))

            # ---- Phase 3b (forecast only): Load ALL lags into archive ----
            archive_count = 0
            if load_archive:
                logger.info("[3b/%d] Loading ALL lags -> backtest_lag_archive (before staging mutation) ...", total_phases)
                t0 = time.time()
                archive_count = _load_forecast_archive(cur, stg_table, stg_alias)
                logger.info("       Inserted %s archive rows in %s", f"{archive_count:,}", _elapsed(t0))
                logger.info("       Archive preserves each row's original lag as execution_lag")
            elif is_forecast and skip_archive:
                logger.info("[3b/%d] Skipping archive load (--skip-archive)", total_phases)

            # ---- Phase 3c (forecast only): Resolve execution lag from dim_dfu ----
            if is_forecast:
                logger.info("[3c/%d] Resolving execution lag from dim_dfu ...", total_phases)
                t0 = time.time()
                matched, unmatched = _resolve_forecast_execution_lag(cur, stg_table)
                logger.info("       Matched %s rows from dim_dfu, defaulted %s rows to lag 0 (%s)",
                            f"{matched:,}", f"{unmatched:,}", _elapsed(t0))
                # Add WHERE clause to main INSERT: keep only execution-lag rows
                insert_sql = insert_sql.rstrip(";") + (
                    f" WHERE {src_alias}.\"lag\" = {src_alias}.\"execution_lag\";"
                )
                logger.info("       Main table will receive execution-lag rows only")

            # ---- Phase 4: Clear target rows ----
            if replace_mode:
                logger.info("[4/%d] DELETE model_id='%s' from %s ...", total_phases, EXTERNAL_MODEL_ID, spec.table)
                t0 = time.time()
                cur.execute(
                    f"DELETE FROM {qident(spec.table)} WHERE model_id = %s",
                    [EXTERNAL_MODEL_ID],
                )
                deleted = cur.rowcount
                logger.info("       Deleted %s external rows (%s)", f"{deleted:,}", _elapsed(t0))
            else:
                logger.info("[4/%d] TRUNCATE %s", total_phases, spec.table)
                t0 = time.time()
                cur.execute(truncate_sql)
            if fast_mode:
                logger.info("       + dropping indexes & constraints ...")
                saved_constraints = _get_unique_constraints(cur, spec.table)
                if saved_constraints:
                    _drop_unique_constraints(cur, spec.table, saved_constraints)
                    for con_name, _, cols in saved_constraints:
                        logger.info("         - UNIQUE constraint %s (%s)", con_name, ', '.join(cols))
                saved_indexes = _get_all_indexes(cur, spec.table)
                if saved_indexes:
                    _drop_indexes(cur, saved_indexes)
                    for idx_name, _ in saved_indexes:
                        logger.info("         - index %s", idx_name)
                total_dropped = len(saved_constraints) + len(saved_indexes)
                if total_dropped:
                    logger.info("       Truncated + dropped %d indexes/constraints (%s)", total_dropped, _elapsed(t0))
                else:
                    logger.info("       Truncated (no indexes to drop) (%s)", _elapsed(t0))
            elif not replace_mode:
                logger.info("       (%s)", _elapsed(t0))

            # ---- Phase 5: INSERT into bare table ----
            logger.info("[5/%d] INSERT -> %s ...", total_phases, spec.table)
            t0 = time.time()
            dedup_label = "no dedup" if no_dedup else "with dedup sort"
            extra_label = " (execution-lag only)" if is_forecast else ""
            logger.info("       Inserting %s rows (%s%s) ...", f"{stg_rows:,}", dedup_label, extra_label)
            cur.execute(insert_sql)
            row_count = cur.rowcount
            t_insert = time.time() - t0
            rate_rows = row_count / t_insert if t_insert > 0 else 0
            logger.info("       Inserted %s rows in %s (%s rows/s)", f"{row_count:,}", _elapsed(t0), f"{rate_rows:,.0f}")

            # ---- Phase 6: Recreate indexes + constraints (fast mode) ----
            total_rebuild = len(saved_indexes) + len(saved_constraints)
            if fast_mode and total_rebuild > 0:
                logger.info("[6/%d] Recreating %d indexes/constraints ...", total_phases, total_rebuild)
                t0 = time.time()
                step = 0
                for con_name, _, cols in saved_constraints:
                    step += 1
                    t_idx = time.time()
                    _recreate_unique_constraints(cur, spec.table, [(con_name, 'u', cols)])
                    logger.info("       [%d/%d] UNIQUE %s (%s)", step, total_rebuild, con_name, _elapsed(t_idx))
                for idx_name, idx_def in saved_indexes:
                    step += 1
                    t_idx = time.time()
                    cur.execute(idx_def + ";")
                    logger.info("       [%d/%d] %s (%s)", step, total_rebuild, idx_name, _elapsed(t_idx))
                logger.info("       All indexes rebuilt in %s", _elapsed(t0))
            else:
                logger.info("[6/%d] No index rebuild needed", total_phases)

            # ---- Commit ----
            logger.info("Committing ...")
            conn.commit()

            # ---- Phase 8 (forecast only): Refresh archive views (D5) ----
            if load_archive:
                logger.info("[8/%d] Refreshing archive accuracy views ...", total_phases)
                t0 = time.time()
                for mv in MV_REFRESH_ARCHIVE:
                    cur.execute(f"REFRESH MATERIALIZED VIEW {mv}")
                conn.commit()
                logger.info("       %s refreshed (%s)", ' + '.join(MV_REFRESH_ARCHIVE), _elapsed(t0))
            elif is_forecast and skip_archive:
                logger.info("[8/%d] Skipping archive view refresh (--skip-archive)", total_phases)

        except Exception as exc:
            # E1: fail gracefully on legacy load error
            logger.error("Legacy load failed for %s: %s", spec.name, exc)
            raise

    # ---- Summary ----
    entity_type = "fact" if spec.table.startswith("fact_") else "dimension"
    total = _elapsed(t_total)
    logger.info("=" * 60)
    logger.info("Done: loaded %s rows into %s table %s", f"{row_count:,}", entity_type, spec.table)
    if is_forecast:
        logger.info("  Main table (execution-lag): %s rows", f"{row_count:,}")
        if skip_archive:
            logger.info("  Archive: skipped (--skip-archive)")
        else:
            logger.info("  Archive (all lags):         %s rows", f"{archive_count:,}")
    logger.info("Total time: %s", total)
    logger.info("=" * 60)


def main() -> None:
    allowed = ", ".join(sorted(DOMAIN_SPECS))
    parser = argparse.ArgumentParser(description="Load normalized dataset CSV into Postgres")
    parser.add_argument("--dataset", required=True, help=allowed)
    parser.add_argument("--no-dedup", action="store_true",
                        help="Skip DISTINCT ON dedup (faster for large clean datasets)")
    parser.add_argument("--fast", action="store_true",
                        help="Optimize for large datasets: drop indexes during load, "
                             "increase work_mem, implies --no-dedup")
    parser.add_argument("--replace", action="store_true",
                        help="(forecast only) Replace only model_id='external' rows "
                             "instead of truncating the whole table. Preserves backtest data.")
    parser.add_argument("--skip-archive", action="store_true",
                        help="(forecast only) Skip loading all lags into backtest_lag_archive. "
                             "Loads only the execution-lag row into the main table.")
    parser.add_argument("--medallion", action="store_true",
                        help="Use medallion pipeline (Bronze -> Silver -> Gold) with DQ gating")
    parser.add_argument("--apply-fixes", action="store_true",
                        help="(medallion only) Apply auto-fix strategies during silver DQ")
    args = parser.parse_args()

    no_dedup = args.no_dedup or args.fast
    fast_mode = args.fast
    replace_mode = args.replace
    skip_archive = args.skip_archive
    medallion_mode = args.medallion
    apply_fixes = args.apply_fixes

    spec = get_spec(args.dataset)

    if replace_mode and spec.name != "forecast":
        parser.error("--replace is only supported for --dataset forecast")
    if skip_archive and spec.name != "forecast":
        parser.error("--skip-archive is only supported for --dataset forecast")
    if replace_mode and fast_mode:
        parser.error("--replace and --fast are mutually exclusive")
    if apply_fixes and not medallion_mode:
        parser.error("--apply-fixes requires --medallion")
    if medallion_mode and (fast_mode or replace_mode or skip_archive):
        parser.error("--medallion cannot be combined with --fast, --replace, or --skip-archive")

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    csv_path = root / "data" / spec.clean_file

    # Medallion pipeline: use separate code path
    if medallion_mode:
        _run_medallion_pipeline(spec, csv_path, apply_fixes=apply_fixes)
        return

    # Legacy load (L1: extracted into _run_legacy_load)
    _run_legacy_load(spec, csv_path, no_dedup, fast_mode, replace_mode, skip_archive)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
