"""Archive one bounded champion-plus-three live forecast snapshot."""
from __future__ import annotations

import argparse
import logging
from datetime import date
from typing import Any

import psycopg

from common.core.db import get_db_params
from common.core.mv_refresh import refresh_for_tables
from common.core.planning_date import get_planning_date
from common.services.cache import get_cache
from common.services.forecast_snapshot import (
    archive_snapshot_in_transaction,
    missing_required_lags,
)

logger = logging.getLogger(__name__)


def _parse_record_month(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        parsed = date.fromisoformat(f"{value}-01" if len(value) == 7 else value)
    except ValueError as exc:
        raise ValueError("record month must be YYYY-MM or YYYY-MM-DD") from exc
    return parsed.replace(day=1)


def _default_record_month(cur: psycopg.Cursor) -> date | None:
    cur.execute(
        """SELECT plan_version
           FROM model_promotion_log
           WHERE is_active = TRUE
           ORDER BY promoted_at DESC, id DESC
           LIMIT 1"""
    )
    row = cur.fetchone()
    if row is None:
        return None
    try:
        return date.fromisoformat(f"{row[0]}-01")
    except (TypeError, ValueError) as exc:
        raise ValueError("active release does not have a calendar plan version") from exc


def _active_promotion(cur: psycopg.Cursor, record_month: date) -> tuple[int, Any]:
    cur.execute(
        """SELECT id, production_run_id
           FROM model_promotion_log
           WHERE is_active = TRUE
             AND plan_version = TO_CHAR(%s::date, 'YYYY-MM')
           ORDER BY promoted_at DESC, id DESC
           LIMIT 1
           FOR UPDATE""",
        (record_month,),
    )
    row = cur.fetchone()
    if row is None or row[1] is None:
        raise ValueError("active release has no verifiable production run lineage")
    return int(row[0]), row[1]


def _roster_models(cur: psycopg.Cursor, record_month: date) -> list[dict[str, Any]]:
    cur.execute(
        """SELECT model_id, snapshot_role, contender_rank, generation_run_id
           FROM forecast_snapshot_roster
           WHERE record_month = %s
           ORDER BY CASE WHEN snapshot_role = 'champion' THEN 0 ELSE 1 END,
                    contender_rank NULLS FIRST""",
        (record_month,),
    )
    rows = [
        {
            "model_id": row[0],
            "snapshot_role": row[1],
            "contender_rank": row[2],
            "generation_run_id": str(row[3]) if row[3] else None,
        }
        for row in cur.fetchall()
    ]
    champions = [row for row in rows if row["snapshot_role"] == "champion"]
    contenders = [row for row in rows if row["snapshot_role"] == "contender"]
    if len(champions) != 1 or [row["contender_rank"] for row in contenders] != [1, 2, 3]:
        raise ValueError("snapshot roster must contain champion and contender ranks 1, 2, and 3")
    return rows


def _dry_run_counts(
    cur: psycopg.Cursor,
    record_month: date,
    production_run_id: Any,
) -> dict[str, dict[int, int]]:
    """Return source counts for only the frozen roster models."""
    counts: dict[str, dict[int, int]] = {}
    cur.execute(
        """SELECT s.model_id,
                  ((EXTRACT(YEAR FROM s.forecast_month) - EXTRACT(YEAR FROM %s::date)) * 12
                   + (EXTRACT(MONTH FROM s.forecast_month) - EXTRACT(MONTH FROM %s::date)))::integer,
                  COUNT(*)
           FROM fact_production_forecast_staging s
           JOIN forecast_snapshot_roster r
             ON r.record_month = %s
            AND r.snapshot_role = 'contender'
            AND r.model_id = s.model_id
            AND r.generation_run_id = s.run_id
           WHERE s.forecast_month_generated = %s
             AND s.generation_purpose = 'snapshot_contender'
             AND s.candidate_model_id = r.model_id
             AND s.forecast_month >= %s
             AND s.forecast_month < %s + INTERVAL '6 months'
           GROUP BY s.model_id, 2""",
        (record_month, record_month, record_month, record_month, record_month, record_month),
    )
    for model_id, lag, row_count in cur.fetchall():
        counts.setdefault(str(model_id), {})[int(lag)] = int(row_count)

    cur.execute(
        """SELECT ((EXTRACT(YEAR FROM forecast_month) - EXTRACT(YEAR FROM %s::date)) * 12
                   + (EXTRACT(MONTH FROM forecast_month) - EXTRACT(MONTH FROM %s::date)))::integer,
                  COUNT(*)
           FROM fact_production_forecast
           WHERE model_id = 'champion'
             AND plan_version = TO_CHAR(%s::date, 'YYYY-MM')
             AND run_id = %s::uuid
             AND forecast_month >= %s
             AND forecast_month < %s + INTERVAL '6 months'
           GROUP BY 1""",
        (
            record_month,
            record_month,
            record_month,
            str(production_run_id),
            record_month,
            record_month,
        ),
    )
    counts["champion"] = {int(lag): int(row_count) for lag, row_count in cur.fetchall()}
    missing = missing_required_lags(counts)
    if missing:
        detail = "; ".join(f"{model}: {lags}" for model, lags in sorted(missing.items()))
        raise ValueError(f"source archive is incomplete for required lags: {detail}")
    return counts


def _validate_archive(cur: psycopg.Cursor, record_month: date, roster: list[dict[str, Any]]) -> dict[str, dict[int, int]]:
    cur.execute(
        """SELECT model_id, lag, COUNT(*)
           FROM fact_forecast_snapshot
           WHERE record_month = %s
           GROUP BY model_id, lag""",
        (record_month,),
    )
    counts = {row["model_id"]: {} for row in roster}
    for model_id, lag, count in cur.fetchall():
        if model_id in counts:
            counts[model_id][int(lag)] = int(count)
    missing = missing_required_lags(counts)
    if missing:
        detail = "; ".join(f"{model}: {lags}" for model, lags in sorted(missing.items()))
        raise ValueError(f"archive is incomplete for required lags: {detail}")
    return counts


def archive_snapshot(record_month: date | None = None, *, dry_run: bool = False, overwrite: bool = False) -> dict[str, dict[int, int]]:
    """Archive the frozen contender roster and matching promoted champion."""
    with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
        record_month = record_month or _default_record_month(cur)
        if record_month is None:
            logger.info("No staging generation exists; nothing to archive")
            return {}
        if record_month > get_planning_date().replace(day=1):
            raise ValueError("record month cannot be after the planning month")
        roster = _roster_models(cur, record_month)
        promotion_id, production_run_id = _active_promotion(cur, record_month)
        if dry_run:
            return _dry_run_counts(cur, record_month, production_run_id)
        if overwrite:
            logger.warning(
                "Overwriting forecast snapshot values for %s from the same frozen roster",
                record_month,
            )
        checksum = archive_snapshot_in_transaction(
            cur,
            record_month=record_month,
            production_run_id=production_run_id,
            source_promotion_id=promotion_id,
            overwrite=overwrite,
        )
        cur.execute(
            """UPDATE model_promotion_log
               SET archive_checksum = %s, archived_at = NOW()
               WHERE id = %s""",
            (checksum, promotion_id),
        )
        counts = _validate_archive(cur, record_month, roster)
        conn.commit()
    refresh_for_tables(["fact_forecast_snapshot"])
    get_cache().invalidate("ds:fva_snapshot*")
    logger.info(
        "Archived forecast snapshot for %s with checksum %s",
        record_month,
        checksum,
    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive champion plus three frozen live forecast contenders")
    parser.add_argument("--record-month", default=None, help="Planning month (YYYY-MM); defaults to active promoted release")
    parser.add_argument("--dry-run", action="store_true", help="Report source counts without writing")
    parser.add_argument("--overwrite", action="store_true", help="Re-snapshot forecast values using the same frozen roster")
    args = parser.parse_args()
    try:
        archive_snapshot(_parse_record_month(args.record_month), dry_run=args.dry_run, overwrite=args.overwrite)
    except (psycopg.Error, ValueError):
        logger.exception("Failed to archive forecast snapshot")
        return 2
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
