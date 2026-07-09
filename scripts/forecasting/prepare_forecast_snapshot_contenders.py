"""Freeze and generate the three non-champion forecasts kept for live FVA."""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import uuid
from datetime import date
from typing import Any

import psycopg

from common.core.db import get_db_params
from common.core.paths import PROJECT_ROOT
from common.core.planning_date import get_planning_date
from common.core.utils import get_forecastable_model_ids, load_forecast_pipeline_config
from common.services.forecast_snapshot import missing_required_lags, select_top_contenders

logger = logging.getLogger(__name__)


def _parse_record_month(value: str | None) -> date:
    if not value:
        return get_planning_date().replace(day=1)
    try:
        parsed = date.fromisoformat(f"{value}-01" if len(value) == 7 else value)
    except ValueError as exc:
        raise ValueError("record month must be YYYY-MM or YYYY-MM-DD") from exc
    return parsed.replace(day=1)


def _snapshot_config() -> dict[str, Any]:
    config = load_forecast_pipeline_config().get("forecast_snapshot", {})
    if config.get("lag_count") != 6 or config.get("contender_count") != 3:
        raise ValueError("forecast_snapshot must retain exactly three contenders at lags 0..5")
    if config.get("rank_metric") != "wape":
        raise ValueError("forecast_snapshot rank_metric must be wape")
    return config


def _load_existing_roster(cur: psycopg.Cursor, record_month: date) -> list[dict[str, Any]]:
    cur.execute(
        """SELECT model_id, snapshot_role, contender_rank, source_backtest_run_id,
                  rank_wape, generation_run_id
           FROM forecast_snapshot_roster
           WHERE record_month = %s
           ORDER BY CASE WHEN snapshot_role = 'champion' THEN 0 ELSE 1 END,
                    contender_rank NULLS FIRST""",
        (record_month,),
    )
    return [
        {
            "model_id": row[0],
            "snapshot_role": row[1],
            "contender_rank": row[2],
            "backtest_run_id": row[3],
            "wape": row[4],
            "generation_run_id": str(row[5]) if row[5] else None,
        }
        for row in cur.fetchall()
    ]


def _validate_existing_roster(rows: list[dict[str, Any]]) -> None:
    champion = [row for row in rows if row["snapshot_role"] == "champion"]
    contenders = [row for row in rows if row["snapshot_role"] == "contender"]
    if len(champion) != 1 or [row["contender_rank"] for row in contenders] != [1, 2, 3]:
        raise ValueError("existing snapshot roster is incomplete; repair it before continuing")


def _latest_runs(
    cur: psycopg.Cursor,
    model_ids: list[str],
    *,
    record_month: date | None = None,
) -> list[dict[str, Any]]:
    if record_month is None:
        query = """WITH ranked_runs AS (
                       SELECT b.id, b.model_id, b.wape, b.accuracy_pct, b.completed_at,
                              ROW_NUMBER() OVER (
                                  PARTITION BY b.model_id
                                  ORDER BY b.completed_at DESC NULLS LAST, b.id DESC
                              ) AS run_rank
                       FROM backtest_run b
                       WHERE b.status = 'completed'
                         AND b.is_loaded_to_db = TRUE
                         AND b.wape IS NOT NULL
                         AND b.model_id <> 'champion'
                         AND b.model_id = ANY(%s)
                   )
                   SELECT id, model_id, wape, accuracy_pct, completed_at
                   FROM ranked_runs
                   WHERE run_rank = 1"""
        params = (model_ids,)
    else:
        query = """WITH cutoffs AS (
                       SELECT model_id, MIN(generated_at) AS generated_at
                       FROM fact_production_forecast_staging
                       WHERE forecast_month_generated = %s
                         AND model_id = ANY(%s)
                       GROUP BY model_id
                   ), ranked_runs AS (
                       SELECT b.id, b.model_id, b.wape, b.accuracy_pct, b.completed_at,
                              ROW_NUMBER() OVER (
                                  PARTITION BY b.model_id
                                  ORDER BY b.completed_at DESC NULLS LAST, b.id DESC
                              ) AS run_rank
                       FROM backtest_run b
                       JOIN cutoffs c ON c.model_id = b.model_id
                       WHERE b.status = 'completed'
                         AND b.is_loaded_to_db = TRUE
                         AND b.wape IS NOT NULL
                         AND b.model_id <> 'champion'
                         AND b.model_id = ANY(%s)
                         AND b.completed_at <= c.generated_at
                   )
                   SELECT id, model_id, wape, accuracy_pct, completed_at
                   FROM ranked_runs
                   WHERE run_rank = 1"""
        params = (record_month, model_ids, model_ids)
    cur.execute(query, params)
    return [
        {
            "backtest_run_id": row[0],
            "model_id": row[1],
            "wape": row[2],
            "accuracy_pct": row[3],
            "completed_at": row[4],
        }
        for row in cur.fetchall()
    ]


def _staging_run_ids(cur: psycopg.Cursor, record_month: date, model_ids: list[str]) -> dict[str, str]:
    cur.execute(
        """SELECT model_id, ARRAY_AGG(DISTINCT run_id)
           FROM fact_production_forecast_staging
           WHERE forecast_month_generated = %s
             AND model_id = ANY(%s)
             AND forecast_month >= %s
             AND forecast_month < %s + INTERVAL '6 months'
           GROUP BY model_id""",
        (record_month, model_ids, record_month, record_month),
    )
    run_ids: dict[str, str] = {}
    for model_id, values in cur.fetchall():
        if len(values) != 1:
            raise ValueError(f"{model_id} does not have one frozen staging run for {record_month:%Y-%m}")
        run_ids[str(model_id)] = str(values[0])
    if set(run_ids) != set(model_ids):
        missing = sorted(set(model_ids) - set(run_ids))
        raise ValueError(f"staging is missing selected contender models: {', '.join(missing)}")
    return run_ids


def _insert_roster(
    cur: psycopg.Cursor,
    *,
    record_month: date,
    contenders: list[dict[str, Any]],
) -> None:
    cur.execute(
        """INSERT INTO forecast_snapshot_roster
               (record_month, model_id, snapshot_role)
           VALUES (%s, 'champion', 'champion')""",
        (record_month,),
    )
    for contender in contenders:
        cur.execute(
            """INSERT INTO forecast_snapshot_roster
                   (record_month, model_id, snapshot_role, contender_rank,
                    source_backtest_run_id, rank_wape, generation_run_id)
               VALUES (%s, %s, 'contender', %s, %s, %s, %s::uuid)""",
            (
                record_month,
                contender["model_id"],
                contender["contender_rank"],
                contender["backtest_run_id"],
                contender["wape"],
                contender["generation_run_id"],
            ),
        )


def _verify_staged_lags(cur: psycopg.Cursor, record_month: date, contenders: list[dict[str, Any]]) -> None:
    run_ids = [row["generation_run_id"] for row in contenders]
    cur.execute(
        """SELECT model_id,
                  ((EXTRACT(YEAR FROM forecast_month) - EXTRACT(YEAR FROM %s::date)) * 12
                   + (EXTRACT(MONTH FROM forecast_month) - EXTRACT(MONTH FROM %s::date)))::integer,
                  COUNT(*)
           FROM fact_production_forecast_staging
           WHERE forecast_month_generated = %s
             AND model_id = ANY(%s)
             AND run_id = ANY(%s::uuid[])
             AND forecast_month >= %s
             AND forecast_month < %s + INTERVAL '6 months'
           GROUP BY model_id, 2""",
        (
            record_month,
            record_month,
            record_month,
            [row["model_id"] for row in contenders],
            run_ids,
            record_month,
            record_month,
        ),
    )
    counts = {row["model_id"]: {} for row in contenders}
    for model_id, lag, count in cur.fetchall():
        if str(model_id) in counts:
            counts[str(model_id)][int(lag)] = int(count)
    missing = missing_required_lags(counts)
    if missing:
        detail = "; ".join(f"{model}: {lags}" for model, lags in sorted(missing.items()))
        raise ValueError(f"selected contenders are missing required snapshot lags: {detail}")


def prepare_contenders(record_month: date, *, dry_run: bool = False, from_existing_staging: bool = False) -> list[dict[str, Any]]:
    """Freeze the roster and ensure its three contender runs cover lags 0..5."""
    _snapshot_config()
    with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
        existing = _load_existing_roster(cur, record_month)
        if existing:
            _validate_existing_roster(existing)
            contenders = [row for row in existing if row["snapshot_role"] == "contender"]
        else:
            forecastable_model_ids = [model_id for model_id in get_forecastable_model_ids() if model_id != "champion"]
            selected = select_top_contenders(
                _latest_runs(
                    cur,
                    forecastable_model_ids,
                    record_month=record_month if from_existing_staging else None,
                )
            )
            if from_existing_staging:
                run_ids = _staging_run_ids(cur, record_month, [row["model_id"] for row in selected])
            else:
                run_ids = {row["model_id"]: str(uuid.uuid4()) for row in selected}
            for row in selected:
                row["generation_run_id"] = run_ids[row["model_id"]]
            contenders = selected
            if dry_run:
                return contenders
            _insert_roster(cur, record_month=record_month, contenders=contenders)
            conn.commit()

    if from_existing_staging:
        with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
            _verify_staged_lags(cur, record_month, contenders)
        return contenders

    for contender in contenders:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "forecasting" / "generate_production_forecasts.py"),
            "--model-id",
            str(contender["model_id"]),
            "--horizon",
            "6",
            "--run-id",
            str(contender["generation_run_id"]),
        ]
        if dry_run:
            logger.info("[DRY RUN] Would run %s", " ".join(command))
            continue
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    if not dry_run:
        with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
            _verify_staged_lags(cur, record_month, contenders)
    return contenders


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze and generate the three forecast snapshot contenders")
    parser.add_argument("--record-month", default=None, help="Planning month (YYYY-MM); defaults to planning date")
    parser.add_argument("--dry-run", action="store_true", help="Show selected contenders without writing or generating")
    parser.add_argument(
        "--from-existing-staging",
        action="store_true",
        help="Bootstrap a historical roster from already-generated staging rows without inference",
    )
    args = parser.parse_args()
    try:
        contenders = prepare_contenders(
            _parse_record_month(args.record_month),
            dry_run=args.dry_run,
            from_existing_staging=args.from_existing_staging,
        )
    except (OSError, psycopg.Error, subprocess.CalledProcessError, ValueError):
        logger.exception("Failed to prepare forecast snapshot contenders")
        return 2
    logger.info("Prepared forecast snapshot contenders: %s", [row["model_id"] for row in contenders])
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
