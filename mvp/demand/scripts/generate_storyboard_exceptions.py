"""Generate Demand Planner Storyboard exceptions — Feature 40.

Scans forecast, inventory health, and accuracy data to detect exceptions:
  - forecast_bias: sustained over/under-forecast over trailing N months
  - stockout_risk: days of supply below threshold
  - accuracy_drop: WAPE degradation vs baseline period
  - excess_risk: days of supply above excess threshold
  - model_drift: champion model instability (placeholder, requires champion history)
  - new_item: items with insufficient forecast history

Writes detected exceptions to the exception_queue table.

Usage:
    uv run python scripts/generate_storyboard_exceptions.py
    uv run python scripts/generate_storyboard_exceptions.py --month 2026-03
    uv run python scripts/generate_storyboard_exceptions.py --dry-run
    uv run python scripts/generate_storyboard_exceptions.py --type stockout_risk
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.db import get_db_params
from common.planning_date import get_planning_date
from common.exception_engine import run_exception_detection


def _load_config() -> dict:
    """Load exception_config.yaml from the config directory."""
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "exception_config.yaml"
    )
    with open(config_path) as f:
        full = yaml.safe_load(f)
    return full.get("exception_engine", full)


def _parse_month(month_str: str | None) -> datetime.date:
    """Parse YYYY-MM string to first-of-month date. Defaults to current month."""
    if not month_str:
        today = get_planning_date()
        return today.replace(day=1)
    try:
        dt = datetime.datetime.strptime(month_str, "%Y-%m")
        return dt.date().replace(day=1)
    except ValueError:
        raise ValueError(f"Invalid month format '{month_str}'. Expected YYYY-MM.")


def run(
    month_str: str | None = None,
    dry_run: bool = False,
    exception_type: str | None = None,
) -> dict:
    """Entry point called by CLI and API POST /storyboard/generate."""
    import psycopg

    config = _load_config()

    # Filter to a single type if requested
    if exception_type:
        all_types = config.get("exception_types", [])
        if exception_type not in all_types:
            raise ValueError(
                f"Unknown exception_type '{exception_type}'. "
                f"Valid: {all_types}"
            )
        config = dict(config)
        config["exception_types"] = [exception_type]

    month_start = _parse_month(month_str)
    params = get_db_params()

    conn = psycopg.connect(**params, autocommit=False)
    try:
        result = run_exception_detection(
            conn=conn,
            config=config,
            month_start=month_start,
            dry_run=dry_run,
        )
    finally:
        conn.close()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Demand Planner Storyboard exceptions (Feature 40)."
    )
    parser.add_argument(
        "--month",
        default=None,
        help="Target month in YYYY-MM format (default: current month)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Detect exceptions but do not write to DB. Prints a sample.",
    )
    parser.add_argument(
        "--type",
        default=None,
        dest="exception_type",
        help="Only detect one exception type (e.g. stockout_risk)",
    )
    args = parser.parse_args()

    result = run(
        month_str=args.month,
        dry_run=args.dry_run,
        exception_type=args.exception_type,
    )

    if result.get("dry_run"):
        print(f"[DRY RUN] Detected {result['detected']} exceptions.")
        sample = result.get("sample", [])
        if sample:
            print("Sample (top 5 by severity):")
            for exc in sample:
                print(
                    f"  [{exc['exception_type']:20s}] "
                    f"sev={exc['severity']:.3f}  "
                    f"{exc['item_no']} @ {exc['loc']}"
                )
                print(f"    {exc['headline']}")
    else:
        print(
            f"Done: detected={result['detected']}, "
            f"inserted={result['inserted']}, "
            f"skipped_dedupe={result['skipped_dedupe']}"
        )


if __name__ == "__main__":
    main()
