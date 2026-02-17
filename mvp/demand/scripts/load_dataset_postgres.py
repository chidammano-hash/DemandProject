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


def main() -> None:
    allowed = ", ".join(sorted(DOMAIN_SPECS))
    parser = argparse.ArgumentParser(description="Load normalized dataset CSV into Postgres")
    parser.add_argument("--dataset", required=True, help=allowed)
    parser.add_argument("--no-dedup", action="store_true",
                        help="Skip DISTINCT ON dedup (faster for large clean datasets)")
    args = parser.parse_args()

    spec = get_spec(args.dataset)

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    csv_path = root / "data" / spec.clean_file

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

    if args.no_dedup:
        # Fast path: skip DISTINCT ON sort — assumes no duplicates in clean CSV
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

    with psycopg.connect(**db) as conn, conn.cursor() as cur:
        t0 = time.time()
        cur.execute(create_stage_sql)
        print(f"  COPY {csv_path.name} → staging ...", flush=True)
        with cur.copy(copy_sql) as copy, csv_path.open("r", encoding="utf-8", newline="") as f:
            bytes_read = 0
            while chunk := f.read(1024 * 1024):
                copy.write(chunk)
                bytes_read += len(chunk)
                if bytes_read % (100 * 1024 * 1024) < 1024 * 1024:
                    print(f"    {bytes_read / (1024**2):.0f} MB copied ...", flush=True)
        t_copy = time.time() - t0
        print(f"  COPY done in {t_copy:.1f}s", flush=True)

        cur.execute(truncate_sql)
        dedup_label = "INSERT (no dedup)" if args.no_dedup else "INSERT (with dedup sort)"
        print(f"  {dedup_label} → {spec.table} ...", flush=True)
        t1 = time.time()
        cur.execute(insert_sql)
        t_insert = time.time() - t1
        print(f"  INSERT done in {t_insert:.1f}s", flush=True)
        conn.commit()

    entity_type = "fact" if spec.table.startswith("fact_") else "dimension"
    print(f"Truncate+loaded {entity_type} table {spec.table} from {csv_path.name}")


if __name__ == "__main__":
    main()
