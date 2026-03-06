import argparse
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


NULL_SQL = "'', 'null', 'none', 'na', 'n/a'"


def _elapsed(t0: float) -> str:
    """Format elapsed time as human-readable string."""
    dt = time.time() - t0
    if dt < 60:
        return f"{dt:.1f}s"
    m, s = divmod(dt, 60)
    return f"{int(m)}m {s:.0f}s"


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def typed_expr(
    field: str,
    int_fields: set[str],
    float_fields: set[str],
    date_fields: set[str],
    src_alias: str,
) -> str:
    col = f"{src_alias}.{qident(field)}"
    if field in int_fields:
        return (
            f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL "
            f"ELSE {col}::integer END"
        )
    if field in float_fields:
        return (
            f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL "
            f"ELSE {col}::numeric END"
        )
    if field in date_fields:
        return (
            f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL "
            f"ELSE {col}::date END"
        )
    return col


def business_key_expr(spec: DomainSpec, src_alias: str) -> str:
    cols = [f"trim({src_alias}.{qident(f)})" for f in spec.key_fields]
    if len(cols) == 1:
        return cols[0]
    sep = (spec.business_key_separator or "-").replace("'", "''")
    return f" || '{sep}' || ".join(cols)


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
        print("       dim_dfu table not found — defaulting execution_lag to 0")
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
    return (matched_rows, unmatched_rows)


def _load_forecast_archive(cur, stg_table: str, stg_alias: str) -> int:
    """Load ALL forecast rows from staging into backtest_lag_archive.

    This preserves all 5 lags (0-4) for multi-lag accuracy analysis while
    the main table holds only execution-lag rows.
    """
    archive_table = "backtest_lag_archive"

    # Delete existing external rows from archive
    cur.execute(f"DELETE FROM {archive_table} WHERE model_id = 'external'")
    deleted = cur.rowcount
    if deleted:
        print(f"       Deleted {deleted:,} existing 'external' archive rows")

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
    args = parser.parse_args()

    no_dedup = args.no_dedup or args.fast
    fast_mode = args.fast
    replace_mode = args.replace
    skip_archive = args.skip_archive

    spec = get_spec(args.dataset)

    if replace_mode and spec.name != "forecast":
        parser.error("--replace is only supported for --dataset forecast")
    if skip_archive and spec.name != "forecast":
        parser.error("--skip-archive is only supported for --dataset forecast")
    if replace_mode and fast_mode:
        parser.error("--replace and --fast are mutually exclusive")

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    csv_path = root / "data" / spec.clean_file
    csv_size_mb = csv_path.stat().st_size / (1024 ** 2) if csv_path.exists() else 0

    db = get_db_params()

    target_cols = [spec.ck_field, *spec.columns]
    stg_table = f"stg_{spec.table}_{spec.name}"
    src_alias = "s"

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
            f"{typed_expr(c, spec.int_fields, spec.float_fields, spec.date_fields, src_alias)} AS {qident(c)}"
            for c in spec.columns
        ],
    ]

    truncate_sql = f"TRUNCATE TABLE {qident(spec.table)};"
    stg_alias = "d"

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
    print(f"\n{'='*60}")
    print(f"Loading {spec.name} → {spec.table}{mode_label}")
    print(f"CSV: {csv_path.name} ({csv_size_mb:,.0f} MB)")
    print(f"{'='*60}\n")

    saved_indexes: list[tuple[str, str]] = []
    saved_constraints: list[tuple[str, str, list[str]]] = []

    with psycopg.connect(**db) as conn, conn.cursor() as cur:
        # ---- Phase 1: Session tuning (fast mode) ----
        if fast_mode:
            print("[1/6] Tuning session for bulk load ...", flush=True)
            cur.execute("SET work_mem = '512MB';")
            cur.execute("SET maintenance_work_mem = '1GB';")
            cur.execute("SET synchronous_commit = 'off';")
            print("       work_mem=512MB, maintenance_work_mem=1GB, synchronous_commit=off\n", flush=True)
        else:
            print("[1/6] Session defaults (use --fast for tuned bulk load)\n", flush=True)

        # ---- Phase 2: Create staging table + COPY ----
        print(f"[2/6] COPY {csv_path.name} → staging table ...", flush=True)
        t0 = time.time()
        cur.execute(create_stage_sql)
        with cur.copy(copy_sql) as copy, csv_path.open("r", encoding="utf-8", newline="") as f:
            bytes_read = 0
            while chunk := f.read(1024 * 1024):
                copy.write(chunk)
                bytes_read += len(chunk)
                if bytes_read % (100 * 1024 * 1024) < 1024 * 1024:
                    pct = (bytes_read / (csv_size_mb * 1024 * 1024) * 100) if csv_size_mb > 0 else 0
                    print(f"       {bytes_read / (1024**2):,.0f} MB copied ({pct:.0f}%) ...", flush=True)
        t_copy = time.time() - t0
        rate_mb = (bytes_read / (1024 ** 2)) / t_copy if t_copy > 0 else 0
        print(f"       Done in {_elapsed(t0)} ({rate_mb:,.0f} MB/s)\n", flush=True)

        # ---- Phase 3: Count staging rows ----
        is_forecast = spec.name == "forecast"
        load_archive = is_forecast and not skip_archive
        total_phases = 8 if is_forecast else 6

        print(f"[3/{total_phases}] Counting staging rows ...", flush=True)
        t0 = time.time()
        cur.execute(f"SELECT count(*) FROM {qident(stg_table)};")
        stg_rows = cur.fetchone()[0]
        print(f"       {stg_rows:,} rows in staging ({_elapsed(t0)})\n", flush=True)

        # ---- Phase 3b (forecast only): Load ALL lags into archive BEFORE staging mutation ----
        archive_count = 0
        if load_archive:
            print(f"[3b/{total_phases}] Loading ALL lags → backtest_lag_archive (before staging mutation) ...", flush=True)
            t0 = time.time()
            archive_count = _load_forecast_archive(cur, stg_table, stg_alias)
            print(f"       Inserted {archive_count:,} archive rows in {_elapsed(t0)}")
            print(f"       Archive preserves each row's original lag as execution_lag\n", flush=True)
        elif is_forecast and skip_archive:
            print(f"[3b/{total_phases}] Skipping archive load (--skip-archive)\n", flush=True)

        # ---- Phase 3c (forecast only): Resolve execution lag from dim_dfu ----
        if is_forecast:
            print(f"[3c/{total_phases}] Resolving execution lag from dim_dfu ...", flush=True)
            t0 = time.time()
            matched, unmatched = _resolve_forecast_execution_lag(cur, stg_table)
            print(f"       Matched {matched:,} rows from dim_dfu, "
                  f"defaulted {unmatched:,} rows to lag 0 ({_elapsed(t0)})")
            # Add WHERE clause to main INSERT: keep only execution-lag rows
            insert_sql = insert_sql.rstrip(";") + (
                f" WHERE {src_alias}.\"lag\" = {src_alias}.\"execution_lag\";"
            )
            print(f"       Main table will receive execution-lag rows only\n", flush=True)

        # ---- Phase 4: Clear target rows ----
        if replace_mode:
            # Replace mode: only delete external rows (preserve backtest data)
            print(f"[4/{total_phases}] DELETE model_id='external' from {spec.table} ...", flush=True)
            t0 = time.time()
            cur.execute(f"DELETE FROM {qident(spec.table)} WHERE model_id = 'external'")
            deleted = cur.rowcount
            print(f"       Deleted {deleted:,} external rows ({_elapsed(t0)})\n", flush=True)
        else:
            # Full mode: truncate entire table
            print(f"[4/{total_phases}] TRUNCATE {spec.table}", end="", flush=True)
            t0 = time.time()
            cur.execute(truncate_sql)
        if fast_mode:
            print(f" + dropping indexes & constraints ...", flush=True)
            # Save index/constraint definitions before dropping
            saved_constraints = _get_unique_constraints(cur, spec.table)
            if saved_constraints:
                _drop_unique_constraints(cur, spec.table, saved_constraints)
                for con_name, _, cols in saved_constraints:
                    print(f"         - UNIQUE constraint {con_name} ({', '.join(cols)})")
            saved_indexes = _get_all_indexes(cur, spec.table)
            if saved_indexes:
                _drop_indexes(cur, saved_indexes)
                for idx_name, _ in saved_indexes:
                    print(f"         - index {idx_name}")
            total_dropped = len(saved_constraints) + len(saved_indexes)
            if total_dropped:
                print(f"       Truncated + dropped {total_dropped} indexes/constraints ({_elapsed(t0)})\n", flush=True)
            else:
                print(f"       Truncated (no indexes to drop) ({_elapsed(t0)})\n", flush=True)
        elif not replace_mode:
            print(f" ({_elapsed(t0)})\n", flush=True)

        # ---- Phase 5: INSERT into bare table ----
        print(f"[5/{total_phases}] INSERT → {spec.table} ...", flush=True)
        t0 = time.time()
        dedup_label = "no dedup" if no_dedup else "with dedup sort"
        extra_label = " (execution-lag only)" if is_forecast else ""
        print(f"       Inserting {stg_rows:,} rows ({dedup_label}{extra_label}) ...", flush=True)
        cur.execute(insert_sql)
        row_count = cur.rowcount
        t_insert = time.time() - t0
        rate_rows = row_count / t_insert if t_insert > 0 else 0
        print(f"       Inserted {row_count:,} rows in {_elapsed(t0)} ({rate_rows:,.0f} rows/s)\n", flush=True)

        # ---- Phase 6: Recreate indexes + constraints (fast mode) ----
        total_rebuild = len(saved_indexes) + len(saved_constraints)
        if fast_mode and total_rebuild > 0:
            print(f"[6/{total_phases}] Recreating {total_rebuild} indexes/constraints ...", flush=True)
            t0 = time.time()
            step = 0
            # Recreate unique constraints first
            for con_name, _, cols in saved_constraints:
                step += 1
                t_idx = time.time()
                _recreate_unique_constraints(cur, spec.table, [(con_name, 'u', cols)])
                print(f"       [{step}/{total_rebuild}] UNIQUE {con_name} ({_elapsed(t_idx)})", flush=True)
            # Recreate regular indexes
            for idx_name, idx_def in saved_indexes:
                step += 1
                t_idx = time.time()
                cur.execute(idx_def + ";")
                print(f"       [{step}/{total_rebuild}] {idx_name} ({_elapsed(t_idx)})", flush=True)
            print(f"       All indexes rebuilt in {_elapsed(t0)}\n", flush=True)
        else:
            print(f"[6/{total_phases}] No index rebuild needed\n", flush=True)

        # ---- Phase 7: (archive already loaded in Phase 3b) ----

        # ---- Commit ----
        print("Committing ...", flush=True)
        conn.commit()

        # ---- Phase 8 (forecast only): Refresh archive views ----
        if load_archive:
            print(f"[8/{total_phases}] Refreshing archive accuracy views ...", flush=True)
            t0 = time.time()
            cur.execute("REFRESH MATERIALIZED VIEW agg_accuracy_lag_archive")
            cur.execute("REFRESH MATERIALIZED VIEW agg_dfu_coverage_lag_archive")
            conn.commit()
            print(f"       agg_accuracy_lag_archive + agg_dfu_coverage_lag_archive refreshed ({_elapsed(t0)})\n", flush=True)
        elif is_forecast and skip_archive:
            print(f"[8/{total_phases}] Skipping archive view refresh (--skip-archive)\n", flush=True)

    # ---- Summary ----
    entity_type = "fact" if spec.table.startswith("fact_") else "dimension"
    total = _elapsed(t_total)
    print(f"\n{'='*60}")
    print(f"Done: loaded {row_count:,} rows into {entity_type} table {spec.table}")
    if is_forecast:
        print(f"  Main table (execution-lag): {row_count:,} rows")
        if skip_archive:
            print(f"  Archive: skipped (--skip-archive)")
        else:
            print(f"  Archive (all lags):         {archive_count:,} rows")
    print(f"Total time: {total}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
