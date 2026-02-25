import argparse
import os
import time
from pathlib import Path
import sys

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def main() -> None:
    allowed = ", ".join(sorted(DOMAIN_SPECS))
    parser = argparse.ArgumentParser(description="Load normalized dataset CSV into Postgres")
    parser.add_argument("--dataset", required=True, help=allowed)
    parser.add_argument("--no-dedup", action="store_true",
                        help="Skip DISTINCT ON dedup (faster for large clean datasets)")
    parser.add_argument("--fast", action="store_true",
                        help="Optimize for large datasets: drop indexes during load, "
                             "increase work_mem, implies --no-dedup")
    args = parser.parse_args()

    no_dedup = args.no_dedup or args.fast
    fast_mode = args.fast

    spec = get_spec(args.dataset)

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    csv_path = root / "data" / spec.clean_file
    csv_size_mb = csv_path.stat().st_size / (1024 ** 2) if csv_path.exists() else 0

    db = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5440")),
        "dbname": os.getenv("POSTGRES_DB", "demand_mvp"),
        "user": os.getenv("POSTGRES_USER", "demand"),
        "password": os.getenv("POSTGRES_PASSWORD", "demand"),
    }

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
        print("[3/6] Counting staging rows ...", flush=True)
        t0 = time.time()
        cur.execute(f"SELECT count(*) FROM {qident(stg_table)};")
        stg_rows = cur.fetchone()[0]
        print(f"       {stg_rows:,} rows in staging ({_elapsed(t0)})\n", flush=True)

        # ---- Phase 4: TRUNCATE + drop indexes/constraints (fast mode) ----
        # TRUNCATE first so index drops are instant (no data to de-index)
        print(f"[4/6] TRUNCATE {spec.table}", end="", flush=True)
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
        else:
            print(f" ({_elapsed(t0)})\n", flush=True)

        # ---- Phase 5: INSERT into bare table ----
        print(f"[5/6] INSERT → {spec.table} ...", flush=True)
        t0 = time.time()
        dedup_label = "no dedup" if no_dedup else "with dedup sort"
        print(f"       Inserting {stg_rows:,} rows ({dedup_label}) ...", flush=True)
        cur.execute(insert_sql)
        row_count = cur.rowcount
        t_insert = time.time() - t0
        rate_rows = row_count / t_insert if t_insert > 0 else 0
        print(f"       Inserted {row_count:,} rows in {_elapsed(t0)} ({rate_rows:,.0f} rows/s)\n", flush=True)

        # ---- Phase 6: Recreate indexes + constraints (fast mode) ----
        total_rebuild = len(saved_indexes) + len(saved_constraints)
        if fast_mode and total_rebuild > 0:
            print(f"[6/6] Recreating {total_rebuild} indexes/constraints ...", flush=True)
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
            print("[6/6] No index rebuild needed\n", flush=True)

        # ---- Commit ----
        print("Committing ...", flush=True)
        conn.commit()

    # ---- Summary ----
    entity_type = "fact" if spec.table.startswith("fact_") else "dimension"
    total = _elapsed(t_total)
    print(f"\n{'='*60}")
    print(f"Done: loaded {row_count:,} rows into {entity_type} table {spec.table}")
    print(f"Total time: {total}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
