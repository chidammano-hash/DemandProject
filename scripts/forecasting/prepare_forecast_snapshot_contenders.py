"""Freeze and generate the three non-champion forecasts kept for live FVA."""

from __future__ import annotations

import argparse
import logging
import os
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
from common.services.champion_lineage import (
    GovernedChampionLineageError,
    load_active_governed_champion_lineage,
)
from common.services.cluster_lineage import load_promoted_cluster_population
from common.services.forecast_generation import (
    GENERATOR_CONTRACT_METADATA_KEY,
    GENERATOR_CONTRACT_VERSION,
    reserve_generation_run,
)
from common.services.forecast_snapshot import missing_required_lags, select_top_contenders
from common.services.forecast_snapshot_validation import (
    SnapshotContenderStaleError,
    validate_ready_snapshot_contender,
)
from common.services.sales_lineage import load_completed_sales_lineage

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


def _delete_recoverable_roster(
    cur: psycopg.Cursor,
    record_month: date,
    rows: list[dict[str, Any]],
) -> None:
    """Remove only a repairable, unpublished current-month roster.

    Incomplete pre-reservation rosters and complete-but-stale ready rosters can
    be replaced only before any snapshot or active release exists for the month.
    """
    planning_month = get_planning_date().replace(day=1)
    if record_month != planning_month:
        raise ValueError("a historical snapshot roster cannot be replaced")
    cur.execute(
        """SELECT
               EXISTS (
                   SELECT 1 FROM fact_forecast_snapshot
                   WHERE record_month = %s
               ),
               EXISTS (
                   SELECT 1 FROM model_promotion_log
                   WHERE is_active = TRUE
                     AND plan_version = TO_CHAR(%s::date, 'YYYY-MM')
               )""",
        (record_month, record_month),
    )
    safety_state = cur.fetchone()
    has_snapshot, has_active_release = safety_state or (False, False)
    if has_snapshot or has_active_release:
        raise ValueError("the snapshot roster cannot be replaced after archive or publish")
    cur.execute(
        """DELETE FROM forecast_snapshot_roster
           WHERE record_month = %s""",
        (record_month,),
    )
    if cur.rowcount != len(rows):
        raise ValueError("the snapshot roster changed during recovery")
    logger.warning(
        "Replacing %d repairable, unpublished roster row(s) for %s",
        len(rows),
        record_month,
    )


def _latest_runs(
    cur: psycopg.Cursor,
    model_ids: list[str],
) -> list[dict[str, Any]]:
    lineage = _current_governed_lineage(cur)
    governed_run_ids = lineage["backtest_run_ids"]
    if set(governed_run_ids) != set(model_ids):
        raise SnapshotContenderStaleError(
            "The governed champion audit does not contain the exact current model roster"
        )
    cur.execute(
        """SELECT id, model_id, wape, accuracy_pct, completed_at
           FROM backtest_run
           WHERE id = ANY(%s)
             AND model_id = ANY(%s)
             AND status = 'completed'
             AND is_loaded_to_db = TRUE
             AND wape IS NOT NULL""",
        (list(governed_run_ids.values()), model_ids),
    )
    rows = [
        {
            "backtest_run_id": row[0],
            "model_id": row[1],
            "wape": row[2],
            "accuracy_pct": row[3],
            "completed_at": row[4],
        }
        for row in cur.fetchall()
    ]
    observed = {str(row["model_id"]): int(row["backtest_run_id"]) for row in rows}
    if observed != governed_run_ids:
        raise SnapshotContenderStaleError(
            "The exact governed five-model backtest runs are not release-ready"
        )
    return rows


def _current_governed_lineage(cur: psycopg.Cursor) -> dict[str, Any]:
    """Require the active champion audit to match current sales and clusters."""
    try:
        lineage = load_active_governed_champion_lineage(cur)
        sales = load_completed_sales_lineage(cur.connection)
        clusters = load_promoted_cluster_population(cur.connection)
    except (GovernedChampionLineageError, RuntimeError, ValueError) as exc:
        raise SnapshotContenderStaleError(
            "Current governed champion lineage is unavailable; run model-refresh"
        ) from exc
    if (
        lineage["source_sales_batch_id"] != sales.batch_id
        or lineage["data_checksum"] != sales.source_hash
        or lineage["cluster_experiment_id"] != clusters.experiment_id
        or lineage["cluster_assignment_count"] != clusters.assignment_count
        or lineage["cluster_assignment_checksum"] != clusters.assignment_checksum
    ):
        raise SnapshotContenderStaleError(
            "The active champion was selected on stale inputs; run model-refresh"
        )
    return lineage


def _validate_existing_roster_backtests(
    cur: psycopg.Cursor,
    rows: list[dict[str, Any]],
) -> None:
    lineage = _current_governed_lineage(cur)
    expected = lineage["backtest_run_ids"]
    for row in rows:
        if row["snapshot_role"] != "contender":
            continue
        if expected.get(str(row["model_id"])) != int(row["backtest_run_id"] or 0):
            raise SnapshotContenderStaleError(
                "Snapshot contender ranking does not match the active governed backtests"
            )


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


def _reserve_contender_runs(
    cur: psycopg.Cursor,
    record_month: date,
    contenders: list[dict[str, Any]],
) -> None:
    """Reserve every FK target before inserting the frozen roster."""
    for contender in contenders:
        status = reserve_generation_run(
            cur,
            run_id=contender["generation_run_id"],
            generation_purpose="snapshot_contender",
            requested_model_id=str(contender["model_id"]),
            record_month=record_month,
            horizon_months=6,
            created_by="forecast-snapshot-preparer",
        )
        if status != "generating":
            raise ValueError(f"{contender['model_id']} snapshot reservation is already {status}")


def _verify_staged_lags(
    cur: psycopg.Cursor, record_month: date, contenders: list[dict[str, Any]]
) -> None:
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
             AND generation_purpose = 'snapshot_contender'
             AND candidate_model_id = model_id
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


def _contender_requires_generation(
    cur: psycopg.Cursor,
    record_month: date,
    contender: dict[str, Any],
) -> bool:
    """Return whether a frozen contender still needs its immutable run.

    A completed run is reused on retry.  Any partially persisted or mismatched
    manifest fails closed because the same UUID must never be overwritten with
    a different forecast payload.
    """
    model_id = str(contender["model_id"])
    run_id = str(contender["generation_run_id"])
    cur.execute(
        """SELECT generation_purpose, requested_model_id,
                  forecast_month_generated, run_status, metadata
           FROM forecast_generation_run
           WHERE run_id = %s::uuid""",
        (run_id,),
    )
    manifest = cur.fetchone()
    if manifest is None:
        return True

    identity = (str(manifest[0]), str(manifest[1]), manifest[2])
    expected = ("snapshot_contender", model_id, record_month)
    if identity != expected:
        raise ValueError(f"{model_id} snapshot generation manifest has a different identity")
    status = str(manifest[3])
    metadata = manifest[4] if isinstance(manifest[4], dict) else {}
    if status != "ready":
        if (
            status == "invalid"
            and metadata.get(GENERATOR_CONTRACT_METADATA_KEY)
            != GENERATOR_CONTRACT_VERSION
        ):
            raise SnapshotContenderStaleError(
                f"{model_id} snapshot generation manifest uses an outdated generator contract"
            )
        if status not in {"generating", "invalid"}:
            raise ValueError(
                f"{model_id} snapshot generation manifest is {manifest[3]}, not reusable"
            )
        cur.execute(
            """SELECT EXISTS (
                   SELECT 1 FROM fact_production_forecast_staging
                   WHERE run_id = %s::uuid
               )""",
            (run_id,),
        )
        if bool(cur.fetchone()[0]):
            raise ValueError(
                f"{model_id} incomplete generation already has staged partial evidence"
            )
        return True

    validate_ready_snapshot_contender(
        cur,
        run_id=run_id,
        model_id=model_id,
        record_month=record_month,
        source_backtest_run_id=int(contender["backtest_run_id"]),
    )
    return False


def prepare_contenders(
    record_month: date,
    *,
    dry_run: bool = False,
    from_existing_staging: bool = False,
) -> list[dict[str, Any]]:
    """Freeze the roster and ensure its three contender runs cover lags 0..5."""
    _snapshot_config()
    pending: list[dict[str, Any]] | None = None
    with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
        existing = _load_existing_roster(cur, record_month)
        if existing:
            try:
                _validate_existing_roster(existing)
            except ValueError:
                if from_existing_staging or dry_run:
                    raise
                _delete_recoverable_roster(cur, record_month, existing)
                existing = []
        if existing:
            try:
                _validate_existing_roster_backtests(cur, existing)
            except SnapshotContenderStaleError:
                if from_existing_staging or dry_run:
                    raise
                _delete_recoverable_roster(cur, record_month, existing)
                existing = []
        if existing:
            contenders = [row for row in existing if row["snapshot_role"] == "contender"]
            if not from_existing_staging:
                try:
                    pending = [
                        contender
                        for contender in contenders
                        if _contender_requires_generation(
                            cur,
                            record_month,
                            contender,
                        )
                    ]
                except SnapshotContenderStaleError:
                    if dry_run:
                        raise
                    _delete_recoverable_roster(
                        cur,
                        record_month,
                        existing,
                    )
                    existing = []
        if not existing:
            if from_existing_staging:
                raise ValueError(
                    "existing-staging recovery requires an original frozen roster; "
                    "legacy staging must not be relabeled with hindsight"
                )
            forecastable_model_ids = [
                model_id for model_id in get_forecastable_model_ids() if model_id != "champion"
            ]
            selected = select_top_contenders(_latest_runs(cur, forecastable_model_ids))
            run_ids = {row["model_id"]: str(uuid.uuid4()) for row in selected}
            for row in selected:
                row["generation_run_id"] = run_ids[row["model_id"]]
            contenders = selected
            if dry_run:
                return contenders
            _reserve_contender_runs(cur, record_month, contenders)
            _insert_roster(cur, record_month=record_month, contenders=contenders)
            conn.commit()
            pending = list(contenders)

    if from_existing_staging:
        with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
            _verify_staged_lags(cur, record_month, contenders)
        return contenders

    if pending is None:
        with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
            pending = [
                contender
                for contender in contenders
                if _contender_requires_generation(cur, record_month, contender)
            ]

    for contender in pending:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "forecasting" / "generate_production_forecasts.py"),
            "--model-id",
            str(contender["model_id"]),
            "--horizon",
            "6",
            "--run-id",
            str(contender["generation_run_id"]),
            "--generation-purpose",
            "snapshot_contender",
        ]
        if dry_run:
            logger.info("[DRY RUN] Would run %s", " ".join(command))
            continue
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )

    reused_count = len(contenders) - len(pending)
    if reused_count:
        logger.info(
            "Reused %d complete snapshot contender run(s); generating %d",
            reused_count,
            len(pending),
        )

    if not dry_run:
        with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
            _verify_staged_lags(cur, record_month, contenders)
    return contenders


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze and generate the three forecast snapshot contenders"
    )
    parser.add_argument(
        "--record-month",
        default=None,
        help="Planning month (YYYY-MM); defaults to planning date",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show selected contenders without writing or generating",
    )
    parser.add_argument(
        "--from-existing-staging",
        action="store_true",
        help="Use only an original complete frozen roster registered by migration 203",
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
    logger.info(
        "Prepared forecast snapshot contenders: %s",
        [row["model_id"] for row in contenders],
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
