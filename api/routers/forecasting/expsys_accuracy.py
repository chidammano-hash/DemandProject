"""Expert System Backtest accuracy API.

Serves pre-computed lag accuracy results from the ExpSys backtest run.
Primary source: data/expert_system_backtest/accuracy_report.json (written by the script).
Status endpoint: reports checkpoints completed and DB row counts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import psycopg
from fastapi import APIRouter, HTTPException, Query

from common.core.db import get_db_params

router = APIRouter(prefix="/expsys", tags=["expsys"])
logger = logging.getLogger(__name__)

from common.core.paths import PROJECT_ROOT as _PROJECT_ROOT
_OUTPUT_DIR = _PROJECT_ROOT / "data" / "expert_system_backtest"
_REPORT_PATH = _OUTPUT_DIR / "accuracy_report.json"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/lag-accuracy")
def get_lag_accuracy(model_id: str = Query(default="ExpSys")) -> dict[str, Any]:
    """Return pre-computed lag accuracy report for the Expert System backtest.

    Returns the JSON written by run_expert_system_backtest.py after all
    10 timeframes complete.  Shape::

        {
          "by_lag": {
            "0": {"accuracy_pct": 74.1, "wape": 25.9, "n_dfus": 12345,
                  "n_dfu_months": 123450, "per_segment": {"smooth_high": 82.3, ...}},
            ...
            "4": {...}
          },
          "execution_lag": {"accuracy_pct": 73.8, ...}
        }
    """
    if _REPORT_PATH.exists():
        return json.loads(_REPORT_PATH.read_text())

    # Fall back to querying the DB if no file report exists
    try:
        db = get_db_params()
        with psycopg.connect(**db) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT b.lag,
                           100 - (SUM(ABS(b.basefcst_pref - s.qty))
                                  / NULLIF(ABS(SUM(s.qty)), 0) * 100) AS accuracy_pct,
                           SUM(ABS(b.basefcst_pref - s.qty))
                                  / NULLIF(ABS(SUM(s.qty)), 0) * 100 AS wape,
                           COUNT(DISTINCT b.item_id || b.customer_group || b.loc) AS n_dfus,
                           COUNT(*) AS n_dfu_months
                    FROM backtest_lag_archive b
                    JOIN fact_sales_monthly s
                      ON s.item_id = b.item_id
                     AND s.customer_group = b.customer_group
                     AND s.loc = b.loc
                     AND s.startdate::date = b.startdate::date
                    WHERE b.model_id = %s
                    GROUP BY b.lag
                    ORDER BY b.lag
                    """,
                    (model_id,),
                )
                rows = cur.fetchall()
    except psycopg.Error as exc:
        logger.exception("DB query failed for lag accuracy")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No ExpSys accuracy data found. Run `make expsys-backtest` first.",
        )

    by_lag: dict[str, Any] = {}
    for lag, acc, wape, n_dfus, n_months in rows:
        by_lag[str(lag)] = {
            "accuracy_pct": round(float(acc or 0), 2),
            "wape": round(float(wape or 0), 2),
            "n_dfus": int(n_dfus),
            "n_dfu_months": int(n_months),
            "per_segment": {},
        }
    return {"by_lag": by_lag, "execution_lag": None}


@router.get("/status")
def get_status() -> dict[str, Any]:
    """Return run status: checkpoints completed, DB row count, last report timestamp."""
    checkpoint_labels: list[str] = []
    if _OUTPUT_DIR.exists():
        checkpoints = sorted(_OUTPUT_DIR.glob("tf_*_predictions.parquet"))
        checkpoint_labels = [
            c.stem.replace("_predictions", "").replace("tf_", "")
            for c in checkpoints
        ]

    has_report = _REPORT_PATH.exists()
    report_mtime: str | None = None
    if has_report:
        import datetime
        report_mtime = datetime.datetime.fromtimestamp(
            _REPORT_PATH.stat().st_mtime
        ).isoformat()

    db_row_count = 0
    try:
        db = get_db_params()
        with psycopg.connect(**db) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM backtest_lag_archive WHERE model_id = 'ExpSys'"
                )
                db_row_count = int(cur.fetchone()[0])  # type: ignore[index]
    except psycopg.Error:
        pass  # DB unavailable — return 0

    return {
        "has_report": has_report,
        "report_updated_at": report_mtime,
        "completed_timeframes": len(checkpoint_labels),
        "checkpoint_labels": checkpoint_labels,
        "db_row_count": db_row_count,
    }
