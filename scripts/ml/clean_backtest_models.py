"""
Remove backtest model predictions from Postgres and refresh materialized views.

Usage:
  uv run python scripts/ml/clean_backtest_models.py lgbm_cluster mstl
  uv run python scripts/ml/clean_backtest_models.py --all-backtest   # remove all non-external models
  uv run python scripts/ml/clean_backtest_models.py --list           # show model_id row counts
  uv run python scripts/ml/clean_backtest_models.py --dry-run lgbm_cluster  # preview without deleting
"""

import argparse
import sys
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.db import get_db_params  # noqa: E402 — after CLI path bootstrap
from common.core.mv_refresh import refresh_for_tables  # noqa: E402 — after sys.path bootstrap
from common.core.utils import _ts  # noqa: E402 — after CLI path bootstrap


def list_models(conn: psycopg.Connection) -> None:
    """Show row counts per model_id in both tables."""
    print("\n── fact_external_forecast_monthly ──")
    rows = conn.execute(
        "SELECT model_id, COUNT(*) AS cnt FROM fact_external_forecast_monthly GROUP BY 1 ORDER BY 1"
    ).fetchall()
    if rows:
        for model_id, cnt in rows:
            print(f"  {model_id:<30s} {cnt:>12,} rows")
    else:
        print("  (empty)")

    print("\n── backtest_lag_archive ──")
    rows = conn.execute(
        "SELECT model_id, COUNT(*) AS cnt FROM backtest_lag_archive GROUP BY 1 ORDER BY 1"
    ).fetchall()
    if rows:
        for model_id, cnt in rows:
            print(f"  {model_id:<30s} {cnt:>12,} rows")
    else:
        print("  (empty)")
    print()


def clean_models(
    conn: psycopg.Connection,
    model_ids: list[str],
    dry_run: bool = False,
) -> None:
    """Delete rows for given model_ids from both tables and refresh views."""
    if not model_ids:
        print("No model_ids specified. Nothing to do.")
        return

    print(f"[{_ts()}] Models to clean: {', '.join(model_ids)}")
    if dry_run:
        print(f"[{_ts()}] DRY RUN — no rows will be deleted\n")

    total_forecast = 0
    total_archive = 0

    for mid in model_ids:
        # Count rows first
        fc = conn.execute(
            "SELECT COUNT(*) FROM fact_external_forecast_monthly WHERE model_id = %s", (mid,)
        ).fetchone()[0]
        ac = conn.execute(
            "SELECT COUNT(*) FROM backtest_lag_archive WHERE model_id = %s", (mid,)
        ).fetchone()[0]

        if fc == 0 and ac == 0:
            print(f"  {mid}: no rows found — skipping")
            continue

        if dry_run:
            print(f"  {mid}: would delete {fc:,} forecast + {ac:,} archive rows")
        else:
            t0 = time.time()
            conn.execute(
                "DELETE FROM fact_external_forecast_monthly WHERE model_id = %s", (mid,)
            )
            conn.execute(
                "DELETE FROM backtest_lag_archive WHERE model_id = %s", (mid,)
            )
            conn.commit()
            print(f"  {mid}: deleted {fc:,} forecast + {ac:,} archive rows ({time.time() - t0:.1f}s)")

        total_forecast += fc
        total_archive += ac

    if total_forecast == 0 and total_archive == 0:
        print(f"\n[{_ts()}] Nothing to clean.")
        return

    if dry_run:
        print(f"\n[{_ts()}] DRY RUN total: {total_forecast:,} forecast + {total_archive:,} archive rows")
        return

    print(f"\n[{_ts()}] Deleted total: {total_forecast:,} forecast + {total_archive:,} archive rows")

    # Refresh every MV reading the tables rows were deleted from
    print(f"[{_ts()}] Refreshing materialized views...")
    refresh_for_tables(["fact_external_forecast_monthly", "backtest_lag_archive"])

    print(f"\n[{_ts()}] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove backtest model predictions from Postgres"
    )
    parser.add_argument(
        "models", nargs="*",
        help="model_id values to remove (e.g., lgbm_cluster mstl)"
    )
    parser.add_argument(
        "--all-backtest", action="store_true",
        help="Remove ALL non-external model predictions (everything except model_id='external')"
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_only",
        help="List model_id row counts and exit"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be deleted without actually deleting"
    )
    args = parser.parse_args()

    with psycopg.connect(**get_db_params()) as conn:
        if args.list_only:
            list_models(conn)
            return

        if args.all_backtest:
            # Find all non-external model_ids
            rows = conn.execute(
                "SELECT DISTINCT model_id FROM fact_external_forecast_monthly WHERE model_id != 'external' ORDER BY 1"
            ).fetchall()
            model_ids = [r[0] for r in rows]
            if not model_ids:
                print("No backtest models found.")
                return
            print(f"Found {len(model_ids)} backtest models: {', '.join(model_ids)}")
        else:
            model_ids = args.models

        if not model_ids:
            parser.print_help()
            sys.exit(1)

        clean_models(conn, model_ids, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
