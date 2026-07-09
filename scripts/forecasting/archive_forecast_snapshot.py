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
from common.services.forecast_snapshot import missing_required_lags

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
    cur.execute("SELECT MAX(forecast_month_generated) FROM fact_production_forecast_staging")
    value = cur.fetchone()[0]
    return value.replace(day=1) if value else None


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


def _upsert_suffix(overwrite: bool) -> str:
    if not overwrite:
        return "ON CONFLICT (record_month, model_id, item_id, loc, forecast_month) DO NOTHING"
    return """ON CONFLICT (record_month, model_id, item_id, loc, forecast_month) DO UPDATE SET
                horizon_months = EXCLUDED.horizon_months,
                forecast_qty = EXCLUDED.forecast_qty,
                forecast_qty_lower = EXCLUDED.forecast_qty_lower,
                forecast_qty_upper = EXCLUDED.forecast_qty_upper,
                source_model_id = EXCLUDED.source_model_id,
                cluster_id = EXCLUDED.cluster_id,
                plan_version = EXCLUDED.plan_version,
                run_id = EXCLUDED.run_id,
                generated_at = EXCLUDED.generated_at,
                archived_at = EXCLUDED.archived_at"""


def _archive_contenders(cur: psycopg.Cursor, record_month: date, overwrite: bool) -> int:
    sql = """INSERT INTO fact_forecast_snapshot
                 (record_month, model_id, item_id, loc, forecast_month, horizon_months,
                  forecast_qty, forecast_qty_lower, forecast_qty_upper, source_model_id,
                  cluster_id, plan_version, run_id, generated_at)
             SELECT r.record_month, s.model_id, s.item_id, s.loc, s.forecast_month,
                    s.horizon_months, s.forecast_qty, s.forecast_qty_lower,
                    s.forecast_qty_upper, NULL, s.cluster_id, NULL, s.run_id, s.generated_at
             FROM fact_production_forecast_staging s
             JOIN forecast_snapshot_roster r
               ON r.record_month = %s
              AND r.snapshot_role = 'contender'
              AND r.model_id = s.model_id
              AND r.generation_run_id = s.run_id
             WHERE s.forecast_month_generated = %s
               AND s.forecast_month >= %s
               AND s.forecast_month < %s + INTERVAL '6 months'
             """ + _upsert_suffix(overwrite)
    cur.execute(sql, (record_month, record_month, record_month, record_month))
    return cur.rowcount


def _archive_champion(cur: psycopg.Cursor, record_month: date, overwrite: bool) -> int:
    sql = """INSERT INTO fact_forecast_snapshot
                 (record_month, model_id, item_id, loc, forecast_month, horizon_months,
                  forecast_qty, forecast_qty_lower, forecast_qty_upper, source_model_id,
                  cluster_id, plan_version, run_id, generated_at)
             SELECT r.record_month, 'champion', p.item_id, p.loc, p.forecast_month,
                    p.horizon_months, p.forecast_qty, p.forecast_qty_lower,
                    p.forecast_qty_upper, p.source_model_id, p.cluster_id,
                    p.plan_version, p.run_id, p.generated_at
             FROM fact_production_forecast p
             JOIN forecast_snapshot_roster r
               ON r.record_month = %s
              AND r.model_id = 'champion'
             WHERE p.model_id = 'champion'
               AND p.plan_version = TO_CHAR(%s::date, 'YYYY-MM')
               AND p.forecast_month >= %s
               AND p.forecast_month < %s + INTERVAL '6 months'
             """ + _upsert_suffix(overwrite)
    cur.execute(sql, (record_month, record_month, record_month, record_month))
    return cur.rowcount


def _dry_run_counts(cur: psycopg.Cursor, record_month: date) -> dict[str, dict[int, int]]:
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
             AND forecast_month >= %s
             AND forecast_month < %s + INTERVAL '6 months'
           GROUP BY 1""",
        (record_month, record_month, record_month, record_month, record_month),
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
        if dry_run:
            return _dry_run_counts(cur, record_month)
        if overwrite:
            logger.warning(
                "Overwriting the immutable forecast snapshot for %s using its existing frozen roster",
                record_month,
            )
        contender_rows = _archive_contenders(cur, record_month, overwrite)
        champion_rows = _archive_champion(cur, record_month, overwrite)
        counts = _validate_archive(cur, record_month, roster)
        conn.commit()
    refresh_for_tables(["fact_forecast_snapshot"])
    get_cache().invalidate("ds:fva_snapshot*")
    logger.info(
        "Archived forecast snapshot for %s (%d contender rows, %d champion rows)",
        record_month,
        contender_rows,
        champion_rows,
    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive champion plus three frozen live forecast contenders")
    parser.add_argument("--record-month", default=None, help="Planning month (YYYY-MM); defaults to staging maximum")
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
