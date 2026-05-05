"""Nightly drift detector for promoted models.

Gen-4 Stream G / AI-9 (optional scaffold).

For each promoted model:
  1. Load a baseline window (e.g. last training slice) of realized
     demand and the corresponding forecast.
  2. Load the current window.
  3. Compute PSI (on demand) and rolling WAPE (on forecast error).
  4. Write one ``fact_drift_signal`` row per metric. Rows with
     ``threshold_breached = TRUE`` become the retrain trigger.

Usage:
    python scripts/ml/detect_drift.py [--window-months 3] [--psi-threshold 0.2]

This script is intentionally thin — it wraps ``common.ai.drift`` helpers
and the DB writes. Full trigger orchestration (scheduling the retrain)
belongs in the Stream H closed-loop orchestrator.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow "python scripts/ml/detect_drift.py" from the repo root.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.ai.drift import psi_signal, wape_signal
from common.core.db import get_db_params

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW_MONTHS: int = 3
_DEFAULT_PSI_THRESHOLD: float = 0.20
_DEFAULT_WAPE_THRESHOLD: float = 0.30  # 30 % rolling-WAPE breach cutoff
_DEFAULT_WAPE_WINDOW: int = 3          # months of rolling comparison


def _fetch_demand_series(cursor, window_months: int) -> tuple[list[float], list[float]]:
    """Return (baseline, current) flat demand lists."""
    cursor.execute(
        """
        SELECT qty_shipped
          FROM agg_sales_monthly
         WHERE month_start >= (CURRENT_DATE - INTERVAL '1 month' * %s)
         ORDER BY month_start
        """,
        (window_months,),
    )
    current = [float(r[0] or 0.0) for r in cursor.fetchall()]

    cursor.execute(
        """
        SELECT qty_shipped
          FROM agg_sales_monthly
         WHERE month_start <  (CURRENT_DATE - INTERVAL '1 month' * %s)
           AND month_start >= (CURRENT_DATE - INTERVAL '1 month' * %s)
         ORDER BY month_start
        """,
        (window_months, window_months * 2),
    )
    baseline = [float(r[0] or 0.0) for r in cursor.fetchall()]
    return baseline, current


def _fetch_promoted_model(cursor) -> str | None:
    cursor.execute(
        """
        SELECT model_id FROM model_promotion_log
         WHERE is_active = TRUE
         ORDER BY promoted_at DESC LIMIT 1
        """
    )
    row = cursor.fetchone()
    return row[0] if row else None


def _fetch_actuals_and_forecasts(cursor, window_months: int) -> tuple[list[float], list[float]]:
    """Return aligned (actuals, forecasts) for the last window_months months."""
    cursor.execute(
        """
        SELECT pf.forecast_qty, COALESCE(s.qty_shipped, 0.0)
          FROM fact_production_forecast pf
     LEFT JOIN agg_sales_monthly s
            ON s.item_id = pf.item_id
           AND s.loc = pf.loc
           AND s.month_start = pf.forecast_month
         WHERE pf.forecast_month >= (CURRENT_DATE - INTERVAL '1 month' * %s)
           AND pf.forecast_month <  CURRENT_DATE
         ORDER BY pf.forecast_month
        """,
        (window_months,),
    )
    rows = cursor.fetchall()
    forecasts = [float(r[0] or 0.0) for r in rows]
    actuals = [float(r[1] or 0.0) for r in rows]
    return actuals, forecasts


def _persist(cursor, signal) -> None:
    cursor.execute(
        """
        INSERT INTO fact_drift_signal
            (model_id, metric, value, baseline, threshold,
             threshold_breached, window_label, details)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        """,
        (
            signal.model_id,
            signal.metric,
            signal.value,
            signal.baseline,
            signal.threshold,
            signal.threshold_breached,
            signal.window_label,
            None if signal.details is None else _json(signal.details),
        ),
    )


def _json(obj) -> str:
    import json
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-months", type=int, default=_DEFAULT_WINDOW_MONTHS)
    parser.add_argument("--psi-threshold", type=float, default=_DEFAULT_PSI_THRESHOLD)
    parser.add_argument("--wape-threshold", type=float, default=_DEFAULT_WAPE_THRESHOLD)
    parser.add_argument("--wape-window", type=int, default=_DEFAULT_WAPE_WINDOW)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import psycopg  # local import keeps module importable without driver
    with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
        model_id = _fetch_promoted_model(cur)
        if not model_id:
            logger.info("No active promoted model; skipping drift detection")
            return 0

        baseline, current = _fetch_demand_series(cur, args.window_months)
        if baseline and current:
            signal = psi_signal(
                model_id, "demand_qty", baseline, current,
                threshold=args.psi_threshold,
                window_label=f"current vs prior {args.window_months}m",
            )
            _persist(cur, signal)
            logger.info(
                "PSI demand_qty model=%s value=%.4f breached=%s",
                model_id, signal.value, signal.threshold_breached,
            )

        actuals, forecasts = _fetch_actuals_and_forecasts(cur, args.window_months)
        if actuals and forecasts and len(actuals) >= args.wape_window:
            signal = wape_signal(
                model_id, actuals, forecasts,
                window=args.wape_window,
                threshold=args.wape_threshold,
                window_label=f"last {args.window_months}m",
            )
            _persist(cur, signal)
            logger.info(
                "rolling_wape model=%s value=%.4f breached=%s",
                model_id, signal.value, signal.threshold_breached,
            )

        conn.commit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
