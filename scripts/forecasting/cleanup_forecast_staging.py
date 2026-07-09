"""Delete old forecast staging generations only after bounded-archive reconciliation."""
from __future__ import annotations

import argparse
import logging
from datetime import date

import psycopg

from common.core.db import get_db_params
from common.core.planning_date import get_planning_date
from common.services.forecast_snapshot import (
    cleanup_reconciliation_issues,
    missing_required_lags,
)

logger = logging.getLogger(__name__)


def _parse_month(value: str) -> date:
    try:
        parsed = date.fromisoformat(f"{value}-01" if len(value) == 7 else value)
    except ValueError as exc:
        raise ValueError("generation must be YYYY-MM or YYYY-MM-DD") from exc
    return parsed.replace(day=1)


def _target_generations(cur: psycopg.Cursor, generation: date | None) -> list[date]:
    if generation is not None:
        return [generation]
    cur.execute(
        """SELECT DISTINCT forecast_month_generated
           FROM fact_production_forecast_staging
           WHERE forecast_month_generated < %s
           ORDER BY forecast_month_generated""",
        (get_planning_date().replace(day=1),),
    )
    return [row[0] for row in cur.fetchall()]


def _expected_contender_counts(cur: psycopg.Cursor, generation: date) -> dict[tuple[str, str], int]:
    cur.execute(
        """SELECT snapshot_role, contender_rank
           FROM forecast_snapshot_roster
           WHERE record_month = %s
           ORDER BY snapshot_role DESC, contender_rank NULLS FIRST""",
        (generation,),
    )
    roster = cur.fetchall()
    if roster.count(("champion", None)) != 1 or [row[1] for row in roster if row[0] == "contender"] != [1, 2, 3]:
        raise ValueError("snapshot roster must contain champion and contender ranks 1, 2, and 3")
    cur.execute(
        """SELECT r.model_id, r.generation_run_id, COUNT(s.run_id)
           FROM forecast_snapshot_roster r
           LEFT JOIN fact_production_forecast_staging s
             ON s.model_id = r.model_id
            AND s.run_id = r.generation_run_id
            AND s.forecast_month_generated = r.record_month
            AND s.forecast_month >= r.record_month
            AND s.forecast_month < r.record_month + INTERVAL '6 months'
           WHERE r.record_month = %s
             AND r.snapshot_role = 'contender'
           GROUP BY r.model_id, r.generation_run_id""",
        (generation,),
    )
    counts = {(str(model_id), str(run_id)): int(count) for model_id, run_id, count in cur.fetchall()}
    if len(counts) != 3 or any(count <= 0 for count in counts.values()):
        raise ValueError("selected contender staging rows are incomplete")
    return counts


def _archived_contender_counts(cur: psycopg.Cursor, generation: date) -> dict[tuple[str, str], int]:
    cur.execute(
        """SELECT model_id, run_id, COUNT(*)
           FROM fact_forecast_snapshot
           WHERE record_month = %s
             AND model_id <> 'champion'
           GROUP BY model_id, run_id""",
        (generation,),
    )
    return {(str(model_id), str(run_id)): int(count) for model_id, run_id, count in cur.fetchall()}


def _validate_lag_coverage(cur: psycopg.Cursor, generation: date) -> None:
    """Require every selected contender source and archive to cover lags 0..5."""
    cur.execute(
        """SELECT r.model_id,
                   ((EXTRACT(YEAR FROM s.forecast_month) - EXTRACT(YEAR FROM %s::date)) * 12
                    + (EXTRACT(MONTH FROM s.forecast_month) - EXTRACT(MONTH FROM %s::date)))::integer,
                   COUNT(*)
            FROM forecast_snapshot_roster r
            JOIN fact_production_forecast_staging s
              ON s.model_id = r.model_id
             AND s.run_id = r.generation_run_id
             AND s.forecast_month_generated = r.record_month
             AND s.forecast_month >= r.record_month
             AND s.forecast_month < r.record_month + INTERVAL '6 months'
            WHERE r.record_month = %s AND r.snapshot_role = 'contender'
            GROUP BY r.model_id, 2""",
        (generation, generation, generation),
    )
    staging: dict[str, dict[int, int]] = {}
    for model_id, lag, count in cur.fetchall():
        staging.setdefault(str(model_id), {})[int(lag)] = int(count)

    cur.execute(
        """SELECT model_id, lag, COUNT(*)
           FROM fact_forecast_snapshot
           WHERE record_month = %s AND model_id <> 'champion'
           GROUP BY model_id, lag""",
        (generation,),
    )
    archived = {model_id: {} for model_id in staging}
    for model_id, lag, count in cur.fetchall():
        if str(model_id) in archived:
            archived[str(model_id)][int(lag)] = int(count)

    missing = missing_required_lags(staging) | missing_required_lags(archived)
    if missing:
        detail = "; ".join(f"{model}: {lags}" for model, lags in sorted(missing.items()))
        raise ValueError(f"selected contender lag coverage is incomplete: {detail}")


def _champion_archive_count(cur: psycopg.Cursor, generation: date) -> int:
    cur.execute(
        """SELECT COUNT(*)
           FROM fact_forecast_snapshot
           WHERE record_month = %s AND model_id = 'champion'""",
        (generation,),
    )
    return int(cur.fetchone()[0])


def cleanup_generation(cur: psycopg.Cursor, generation: date, *, dry_run: bool) -> int:
    """Validate one generation then delete all of its transient staging rows."""
    expected = _expected_contender_counts(cur, generation)
    archived = _archived_contender_counts(cur, generation)
    _validate_lag_coverage(cur, generation)
    issues = cleanup_reconciliation_issues(
        expected,
        archived,
        champion_archive_count=_champion_archive_count(cur, generation),
    )
    if issues:
        raise ValueError("archive reconciliation failed: " + "; ".join(issues))
    cur.execute(
        "SELECT COUNT(*) FROM fact_production_forecast_staging WHERE forecast_month_generated = %s",
        (generation,),
    )
    total = int(cur.fetchone()[0])
    if dry_run:
        logger.info("[DRY RUN] Would delete %d staging rows for %s", total, generation)
        return total
    cur.execute(
        "DELETE FROM fact_production_forecast_staging WHERE forecast_month_generated = %s",
        (generation,),
    )
    return cur.rowcount


def cleanup_staging(generation: date | None = None, *, dry_run: bool = False) -> int:
    """Clean one explicit generation or all safely eligible old generations."""
    with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
        deleted = 0
        for target in _target_generations(cur, generation):
            deleted += cleanup_generation(cur, target, dry_run=dry_run)
        if not dry_run:
            conn.commit()
    return deleted


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely clean forecast staging after snapshot archive")
    parser.add_argument("--generation", default=None, help="One generation to clean (YYYY-MM)")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without deleting")
    args = parser.parse_args()
    try:
        deleted = cleanup_staging(_parse_month(args.generation) if args.generation else None, dry_run=args.dry_run)
    except (psycopg.Error, ValueError):
        logger.exception("Failed to clean forecast staging")
        return 2
    logger.info("Forecast staging cleanup complete: %d rows", deleted)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
