"""Fit causal elasticities from sales + external signals.

Gen-4 Roadmap Cross-cutting #3. Scaffold implementation:

  * reads `fact_sales_monthly` (sales) and `fact_external_signal`
    (price/promo/weather signals) via a LEFT JOIN on item_id+loc+month
  * runs a simple linear regression (common.ai.causal.fit_linear_elasticities)
  * inserts coefficients into `fact_causal_elasticity`

The script deliberately keeps its dependency surface small: numpy,
pandas, sklearn. When EconML integration is approved, swap the
`fit_linear_elasticities` call in `run_fit` for the EconML backend.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from typing import Any

# Keep top-of-file imports light — DB and ML deps are imported in main().
logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit causal elasticities")
    p.add_argument("--dry-run", action="store_true", help="Compute coefs but don't INSERT")
    p.add_argument("--item-id", default=None, help="Restrict to a single item_id")
    p.add_argument("--loc", default=None, help="Restrict to a single location")
    p.add_argument("--history-months", type=int, default=None)
    return p.parse_args(argv)


def _load_dataset(cursor: Any, cfg: dict[str, Any], item_id: str | None, loc: str | None):
    """Pull a wide (target + features) dataframe.

    Uses pandas.read_sql-style approach but goes through psycopg cursor
    so tests can stub the SQL calls without a live DB.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("fit_elasticity requires pandas") from exc

    months = int(cfg.get("history_months", 36))
    filters = []
    params: list[Any] = [months]
    if item_id:
        filters.append("s.item_id = %s")
        params.append(item_id)
    if loc:
        filters.append("s.loc = %s")
        params.append(loc)
    where = (" AND " + " AND ".join(filters)) if filters else ""

    # Join sales + external signals by item_id/loc/month.
    cursor.execute(
        f"""
        SELECT s.item_id, s.loc, s.month AS event_month,
               s.qty AS target_qty,
               MAX(CASE WHEN x.signal_kind = 'unit_price' THEN (x.value->>'value')::float END) AS unit_price,
               MAX(CASE WHEN x.signal_kind = 'promo_flag' THEN (x.value->>'value')::float END) AS promo_flag,
               MAX(CASE WHEN x.signal_kind = 'temp_anomaly' THEN (x.value->>'value')::float END) AS temp_anomaly
        FROM fact_sales_monthly s
        LEFT JOIN fact_external_signal x
               ON x.item_id = s.item_id
              AND COALESCE(x.loc, s.loc) = s.loc
              AND date_trunc('month', x.event_ts) = s.month
        WHERE s.month >= (CURRENT_DATE - (%s::int || ' months')::interval)
          {where}
        GROUP BY s.item_id, s.loc, s.month, s.qty
        ORDER BY s.item_id, s.loc, s.month
        """,
        params,
    )
    rows = cursor.fetchall() or []
    cols = ["item_id", "loc", "event_month", "target_qty", "unit_price", "promo_flag", "temp_anomaly"]
    return pd.DataFrame(rows, columns=cols)


def _apply_transform(series, transform: str):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy required") from exc
    if transform == "log1p":
        return np.log1p(series.astype(float).clip(lower=0))
    if transform == "zscore":
        std = series.std() or 1.0
        return (series - series.mean()) / std
    return series.astype(float)


def run_fit(cursor: Any, cfg: dict[str, Any], *, dry_run: bool, item_id: str | None, loc: str | None) -> int:
    """Main worker. Returns number of coefficients written."""
    from common.ai.causal import fit_linear_elasticities

    df = _load_dataset(cursor, cfg, item_id, loc)
    if df.empty:
        logger.warning("fit_elasticity: empty training frame; nothing to do")
        return 0

    features_cfg = cfg.get("features", [])
    feature_names: list[str] = []
    feature_cols: list[str] = []
    for f in features_cfg:
        col = f["column_source"].split(":", 1)[-1] if f["column_source"].startswith("signal:") else f["column_source"]
        if col not in df.columns:
            logger.warning("feature %s missing column %s; skipping", f["name"], col)
            continue
        feature_names.append(f["name"])
        feature_cols.append(col)
        df[col] = _apply_transform(df[col].fillna(0.0), f.get("transform", "none"))

    if not feature_cols:
        logger.error("No usable features resolved; aborting")
        return 0

    df = df.dropna(subset=["target_qty"] + feature_cols)
    min_obs = int(cfg.get("min_obs", 24))
    if len(df) < min_obs:
        logger.warning("only %s rows available (< min_obs=%s); skipping", len(df), min_obs)
        return 0

    results = fit_linear_elasticities(
        df[feature_cols].to_numpy(),
        df["target_qty"].to_numpy(),
        feature_names,
    )

    run_id = f"{cfg.get('run_id_prefix', 'elast')}_{uuid.uuid4().hex[:12]}"
    written = 0
    for r in results:
        logger.info("coef %s=%.4f p=%s n=%s", r.feature, r.coef, r.p_value, r.n_obs)
        if dry_run:
            continue
        cursor.execute(
            """
            INSERT INTO fact_causal_elasticity
                (item_id, loc, feature, coef, p_value, std_err, n_obs, method, run_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (item_id, loc, r.feature, r.coef, r.p_value, r.std_err, r.n_obs, r.method, run_id),
        )
        written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)

    from common.utils import load_config
    from common.db import get_db_params
    import psycopg

    cfg = load_config("elasticity_config")
    if args.history_months:
        cfg["history_months"] = args.history_months

    params = get_db_params()
    try:
        with psycopg.connect(**params) as conn, conn.cursor() as cur:
            n = run_fit(cur, cfg, dry_run=args.dry_run, item_id=args.item_id, loc=args.loc)
            if not args.dry_run:
                conn.commit()
    except psycopg.Error as exc:
        logger.exception("fit_elasticity failed: %s", exc)
        return 1

    logger.info("fit_elasticity wrote %s coefficients", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
