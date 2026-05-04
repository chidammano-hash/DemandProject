"""
Remove forecast data by time bucket from Postgres and refresh materialized views.

Usage:
  uv run python scripts/clean_forecasts_by_date.py --list
  uv run python scripts/clean_forecasts_by_date.py --before 2025-04-01 --model external
  uv run python scripts/clean_forecasts_by_date.py --after 2025-06-01
  uv run python scripts/clean_forecasts_by_date.py --between 2024-01-01 2024-07-01
  uv run python scripts/clean_forecasts_by_date.py --months 2024-03 2024-06 2024-09
  uv run python scripts/clean_forecasts_by_date.py --months 2025-01 --model external
  uv run python scripts/clean_forecasts_by_date.py --before 2025-01-01 --date-column fcstdate
  uv run python scripts/clean_forecasts_by_date.py --before 2025-04-01 --model external --dry-run
  uv run python scripts/clean_forecasts_by_date.py --before 2025-04-01 --forecast-only
  uv run python scripts/clean_forecasts_by_date.py --before 2025-04-01 --archive-only
"""

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.db import get_db_params
from common.utils import _ts

VALID_DATE_COLUMNS = ("startdate", "fcstdate")

TABLES_FORECAST = ["fact_external_forecast_monthly"]
TABLES_ARCHIVE = ["backtest_lag_archive"]
TABLES_ALL = TABLES_FORECAST + TABLES_ARCHIVE

REFRESH_VIEWS = [
    "agg_forecast_monthly",
    "agg_accuracy_by_dim",
    "agg_dfu_coverage",
    "agg_accuracy_lag_archive",
    "agg_dfu_coverage_lag_archive",
]


def parse_date(value: str) -> date:
    """Parse a date string into a month-start date.

    Accepts YYYY-MM-DD, YYYY-MM, or MM/DD/YYYY.
    Always normalizes to day=1 (month-start).
    """
    for fmt in ("%Y-%m-%d", "%Y-%m", "%m/%d/%Y"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(day=1).date()
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse date '{value}'. Use YYYY-MM-DD, YYYY-MM, or MM/DD/YYYY format."
    )


def build_where_clause(
    date_column: str,
    before: date | None = None,
    after: date | None = None,
    between: tuple[date, date] | None = None,
    months: list[date] | None = None,
    model: str | None = None,
) -> tuple[str, list]:
    """Build a parameterized WHERE clause and params list."""
    if date_column not in VALID_DATE_COLUMNS:
        raise ValueError(
            f"Invalid date column '{date_column}'. Must be one of {VALID_DATE_COLUMNS}"
        )

    conditions: list[str] = []
    params: list = []

    if before is not None:
        conditions.append(f"{date_column} < %s")
        params.append(before)
    elif after is not None:
        conditions.append(f"{date_column} >= %s")
        params.append(after)
    elif between is not None:
        conditions.append(f"{date_column} >= %s AND {date_column} < %s")
        params.extend([between[0], between[1]])
    elif months is not None:
        placeholders = ", ".join(["%s"] * len(months))
        conditions.append(f"{date_column} IN ({placeholders})")
        params.extend(months)
    else:
        raise ValueError("Must specify one of --before, --after, --between, or --months")

    if model is not None:
        conditions.append("model_id = %s")
        params.append(model)

    where_sql = "WHERE " + " AND ".join(conditions)
    return where_sql, params


def list_by_date(conn: psycopg.Connection, date_column: str = "startdate") -> None:
    """Show row counts grouped by model_id and month for both tables."""
    for table in TABLES_ALL:
        print(f"\n── {table} (by {date_column}) ──")
        rows = conn.execute(
            f"SELECT model_id, date_trunc('month', {date_column})::date AS month, "
            f"COUNT(*) AS cnt "
            f"FROM {table} "
            f"GROUP BY 1, 2 ORDER BY 1, 2"
        ).fetchall()
        if rows:
            current_model = None
            for model_id, month, cnt in rows:
                if model_id != current_model:
                    print(f"  {model_id}:")
                    current_model = model_id
                print(f"    {month}  {cnt:>12,} rows")
        else:
            print("  (empty)")
    print()


def clean_by_date(
    conn: psycopg.Connection,
    where_sql: str,
    params: list,
    tables: list[str],
    dry_run: bool = False,
) -> None:
    """Delete rows matching the WHERE clause from specified tables and refresh views."""
    print(f"[{_ts()}] Filter: {where_sql}")
    print(f"[{_ts()}] Params: {params}")
    print(f"[{_ts()}] Tables: {', '.join(tables)}")
    if dry_run:
        print(f"[{_ts()}] DRY RUN — no rows will be deleted\n")

    total_deleted = 0

    for table in tables:
        cnt = conn.execute(
            f"SELECT COUNT(*) FROM {table} {where_sql}", params
        ).fetchone()[0]

        if cnt == 0:
            print(f"  {table}: no matching rows — skipping")
            continue

        if dry_run:
            print(f"  {table}: would delete {cnt:,} rows")
        else:
            t0 = time.time()
            conn.execute(f"DELETE FROM {table} {where_sql}", params)
            conn.commit()
            print(f"  {table}: deleted {cnt:,} rows ({time.time() - t0:.1f}s)")

        total_deleted += cnt

    if total_deleted == 0:
        print(f"\n[{_ts()}] Nothing to clean.")
        return

    if dry_run:
        print(f"\n[{_ts()}] DRY RUN total: {total_deleted:,} rows would be deleted")
        return

    print(f"\n[{_ts()}] Deleted total: {total_deleted:,} rows")

    print(f"[{_ts()}] Refreshing materialized views...")
    for view in REFRESH_VIEWS:
        try:
            t0 = time.time()
            conn.execute(f"REFRESH MATERIALIZED VIEW {view}")
            conn.commit()
            print(f"  {view} ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"  {view} — skipped ({e})")
            conn.rollback()

    print(f"\n[{_ts()}] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove forecast data by time bucket from Postgres"
    )

    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--before", type=str, metavar="DATE",
        help="Delete rows where date_column < DATE (e.g., 2025-04-01 or 2025-04)",
    )
    date_group.add_argument(
        "--after", type=str, metavar="DATE",
        help="Delete rows where date_column >= DATE",
    )
    date_group.add_argument(
        "--between", type=str, nargs=2, metavar=("START", "END"),
        help="Delete rows where date_column >= START and < END",
    )
    date_group.add_argument(
        "--months", type=str, nargs="+", metavar="MONTH",
        help="Delete specific month(s) (e.g., 2024-03 2024-06 2024-09)",
    )

    parser.add_argument(
        "--model", type=str, metavar="MODEL",
        help="Filter by model_id (e.g., external, lgbm_global). Omit for all models.",
    )
    parser.add_argument(
        "--date-column", type=str, choices=["startdate", "fcstdate"],
        default="startdate",
        help="Which date column to filter on (default: startdate)",
    )

    scope_group = parser.add_mutually_exclusive_group()
    scope_group.add_argument(
        "--forecast-only", action="store_true",
        help="Only clean fact_external_forecast_monthly (skip archive)",
    )
    scope_group.add_argument(
        "--archive-only", action="store_true",
        help="Only clean backtest_lag_archive (skip forecast)",
    )

    parser.add_argument(
        "--list", action="store_true", dest="list_only",
        help="List row counts by model and month, then exit",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be deleted without actually deleting",
    )

    args = parser.parse_args()

    with psycopg.connect(**get_db_params()) as conn:
        if args.list_only:
            list_by_date(conn, args.date_column)
            return

        if not (args.before or args.after or args.between or args.months):
            parser.print_help()
            print("\nError: Must specify --before, --after, --between, or --months (or use --list)")
            sys.exit(1)

        before = parse_date(args.before) if args.before else None
        after = parse_date(args.after) if args.after else None
        between_dates = None
        if args.between:
            between_dates = (parse_date(args.between[0]), parse_date(args.between[1]))
            if between_dates[0] >= between_dates[1]:
                print(
                    f"Error: START date ({between_dates[0]}) must be before "
                    f"END date ({between_dates[1]})"
                )
                sys.exit(1)
        month_dates = None
        if args.months:
            month_dates = [parse_date(m) for m in args.months]

        where_sql, params = build_where_clause(
            date_column=args.date_column,
            before=before,
            after=after,
            between=between_dates,
            months=month_dates,
            model=args.model,
        )

        if args.forecast_only:
            tables = TABLES_FORECAST
        elif args.archive_only:
            tables = TABLES_ARCHIVE
        else:
            tables = TABLES_ALL

        clean_by_date(conn, where_sql, params, tables, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
