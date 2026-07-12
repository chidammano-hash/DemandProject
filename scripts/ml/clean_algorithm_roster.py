"""Remove retired forecasting algorithms from database facts and artifacts.

Dry-run is the default. Pass ``--execute`` to commit deletions.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import psycopg
from psycopg import sql

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.db import get_db_params  # noqa: E402
from common.core.mv_refresh import refresh_for_tables  # noqa: E402
from common.core.utils import get_algorithm_roster  # noqa: E402

logger = logging.getLogger(__name__)
MODEL_TABLES = ("forecast_snapshot_roster", "backtest_lag_archive", "backtest_run", "fact_ai_champion_forecast",
                "fact_candidate_forecast", "fact_external_forecast_monthly",
                "fact_forecast_snapshot", "fact_production_forecast",
                "fact_production_forecast_staging")
REFERENCE_IDS = frozenset({"external", "ceiling"})


def retained_ids(table: str) -> frozenset[str]:
    active = frozenset(get_algorithm_roster(stage="forecast")) | {"champion"}
    return active | REFERENCE_IDS if table == "fact_external_forecast_monthly" else active


def clean_database(conn: psycopg.Connection, *, execute: bool) -> dict[str, int]:
    affected: dict[str, int] = {}
    for table in MODEL_TABLES:
        keep = sorted(retained_ids(table))
        predicate = "model_id IS NOT NULL AND NOT (model_id = ANY(%s))"
        statement = "DELETE FROM {} WHERE " + predicate if execute else "SELECT COUNT(*) FROM {} WHERE " + predicate
        query = sql.SQL(statement).format(sql.Identifier(table))
        result = conn.execute(query, (keep,))
        affected[table] = result.rowcount if execute else int(result.fetchone()[0])
    if execute:
        conn.commit()
    return affected


def clean_artifacts(*, execute: bool) -> list[Path]:
    keep = set(get_algorithm_roster(stage="forecast")) | {"champion"}
    removed: list[Path] = []
    for parent in (ROOT / "data/backtest", ROOT / "data/models"):
        if parent.exists():
            for child in parent.iterdir():
                if child.is_dir() and child.name not in keep:
                    removed.append(child)
                    if execute:
                        shutil.rmtree(child)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="commit deletions; otherwise preview only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with psycopg.connect(**get_db_params()) as conn:
        counts = clean_database(conn, execute=args.execute)
    artifacts = clean_artifacts(execute=args.execute)
    action = "Deleted" if args.execute else "Would delete"
    for table, count in counts.items():
        logger.info("%s %s rows from %s", action, count, table)
    logger.info("%s %s artifact directories", action, len(artifacts))
    if args.execute:
        refresh_for_tables(list(MODEL_TABLES))


if __name__ == "__main__":
    main()
