import argparse
import os
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
        cur.execute(create_stage_sql)
        with cur.copy(copy_sql) as copy, csv_path.open("r", encoding="utf-8", newline="") as f:
            while chunk := f.read(1024 * 1024):
                copy.write(chunk)

        cur.execute(truncate_sql)
        cur.execute(insert_sql)
        conn.commit()

    entity_type = "fact" if spec.table.startswith("fact_") else "dimension"
    print(f"Truncate+loaded {entity_type} table {spec.table} from {csv_path.name}")


if __name__ == "__main__":
    main()
