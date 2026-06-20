"""Foundation-model quantile bridge: FM forecast draws -> SS empirical-quantile.

Gen-4 Stream G / AI-2 Phase 1.

Reads the champion foundation model's quantile forecasts from
``fact_candidate_forecast`` and exposes them as a numpy array shaped
``(n_months, n_samples)`` suitable for
``scripts/run_ss_simulation.empirical_quantile_ss`` and
``common.twin.state.TwinState.simulate`` (as a ``demand_pool`` override).

Design notes:
    - This module does NOT run Chronos inference. It only plumbs already-
      written quantile rows.
    - The shape contract for downstream SS: ``(n_months, n_samples)``.
      Each row is one forecast horizon month; each column is a sample
      draw from the FM's predictive distribution. Callers typically
      flatten rows 0..H-1 (one lead-time worth) into a 1-D demand pool.
    - When the candidate forecast only stores point + lower/upper
      quantiles (P10/P50/P90) we reconstruct ``n_samples`` by sampling
      from a piecewise-linear CDF over the stored quantile grid. The
      algorithm is intentionally simple — full draw retrieval is a TODO
      once the DB schema is extended to store sample-level draws.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from common.core.utils import load_forecast_pipeline_config

logger = logging.getLogger(__name__)

# Default quantile grid when we have to reconstruct from P10/P50/P90.
_DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class FMQuantileForecast:
    """Container for FM quantile rows read from fact_candidate_forecast."""

    item_id: str
    loc: str
    model_id: str
    # Shape (n_months, n_quantile_levels) of stored quantiles in level order.
    quantile_matrix: np.ndarray
    # The quantile levels (length = quantile_matrix.shape[1]).
    quantile_levels: tuple[float, ...]
    # The corresponding forecast_month for each row (length = n_months).
    months: tuple[str, ...]

    def to_sample_array(
        self, n_samples: int = 1000, *, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Return a ``(n_months, n_samples)`` array of demand draws.

        Samples uniformly over the CDF and inverts using the stored
        quantile grid via linear interpolation. Values below level 0 or
        above level 1 are clipped to the outer stored quantiles (no
        tail extrapolation — that would require a parametric assumption).
        """
        rng = rng or np.random.default_rng(42)
        levels = np.asarray(self.quantile_levels, dtype=float)
        out = np.empty((self.quantile_matrix.shape[0], n_samples), dtype=float)
        u = rng.uniform(0.0, 1.0, size=n_samples)
        for i, row in enumerate(self.quantile_matrix):
            # Clip u to the stored grid so np.interp returns the edge
            # values rather than extrapolating linearly outside.
            u_clipped = np.clip(u, levels[0], levels[-1])
            out[i, :] = np.interp(u_clipped, levels, row)
        return out


def _resolve_fm_model_id(fm_config: dict[str, Any] | None = None) -> str:
    """Return the champion FM model_id from config, e.g. 'chronos2_enriched'."""
    cfg = fm_config
    if cfg is None:
        try:
            pipeline_cfg = load_forecast_pipeline_config()
        except FileNotFoundError:
            logger.warning("forecast_pipeline_config.yaml missing; defaulting fm model_id")
            return "chronos2_enriched"
        cfg = pipeline_cfg.get("fm_spine", {}) or {}
    return str(cfg.get("model_id", "chronos2_enriched"))


def _resolve_quantile_levels(
    fm_config: dict[str, Any] | None = None,
) -> tuple[float, ...]:
    cfg = fm_config
    if cfg is None:
        try:
            pipeline_cfg = load_forecast_pipeline_config()
        except FileNotFoundError:
            return _DEFAULT_QUANTILES
        cfg = pipeline_cfg.get("fm_spine", {}) or {}
    qs = cfg.get("quantiles") or list(_DEFAULT_QUANTILES)
    return tuple(float(x) for x in qs)


def load_fm_quantile_forecast(
    cursor: Any,
    item_id: str,
    loc: str,
    *,
    fm_config: dict[str, Any] | None = None,
    horizon_months: int | None = None,
) -> FMQuantileForecast | None:
    """Read champion FM quantile rows for a DFU and return a matrix.

    Args:
        cursor: psycopg3 cursor owned by the caller.
        item_id, loc: the DFU identifiers.
        fm_config: optional override of ``fm_spine`` block (keys: model_id,
                   quantiles). Defaults to forecast_pipeline_config.yaml.
        horizon_months: if provided, LIMIT rows to the first N months.

    Returns:
        FMQuantileForecast, or None when no rows exist. Caller decides
        how to degrade (e.g. fall back to the sales bootstrap in
        ``TwinState``).
    """
    model_id = _resolve_fm_model_id(fm_config)
    levels = _resolve_quantile_levels(fm_config)

    # fact_candidate_forecast currently stores P10/P50/P90 in three
    # columns: forecast_qty_lower, forecast_qty, forecast_qty_upper. We
    # treat those as the stored quantile grid. If the schema grows a
    # proper quantile JSONB column, swap this query.
    sql = """
        SELECT forecast_month,
               forecast_qty_lower,
               forecast_qty,
               forecast_qty_upper
          FROM fact_candidate_forecast
         WHERE item_id = %s
           AND loc = %s
           AND model_id = %s
         ORDER BY forecast_month
    """
    if horizon_months is not None:
        if horizon_months <= 0:
            raise ValueError("horizon_months must be positive when supplied")
        sql += " LIMIT %s"
        cursor.execute(sql, (item_id, loc, model_id, horizon_months))
    else:
        cursor.execute(sql, (item_id, loc, model_id))

    rows = cursor.fetchall()
    if not rows:
        logger.info(
            "No FM quantile rows for item_id=%s loc=%s model_id=%s",
            item_id, loc, model_id,
        )
        return None

    months: list[str] = []
    q_rows: list[list[float]] = []
    for r in rows:
        month, lower, point, upper = r[0], r[1], r[2], r[3]
        # When lower/upper are NULL we approximate the spread as +/- 20%
        # of the point forecast. Keeps the pipeline moving even for
        # model rows that didn't emit full quantiles.
        p = float(point)
        lo = float(lower) if lower is not None else p * 0.8
        hi = float(upper) if upper is not None else p * 1.2
        months.append(str(month))
        # Clip to non-negative (demand >= 0) and enforce monotonic non-decreasing
        # quantiles (P10 <= P50 <= P90). A quantile model can emit crossed
        # quantiles, and the +/-20% fallback can cross near 0; a non-monotone grid
        # makes np.interp produce an invalid (non-monotone) inverse CDF in
        # to_sample_array, distorting the safety-stock demand draws. sorted()
        # restores monotonicity (Chernozhukov quantile rearrangement).
        q_rows.append(sorted([max(0.0, lo), max(0.0, p), max(0.0, hi)]))

    matrix = np.asarray(q_rows, dtype=float)
    # The stored grid is always (P10, P50, P90) regardless of fm_config —
    # we overlay the configured levels only if the grid sizes match.
    grid_levels = levels if matrix.shape[1] == len(levels) else _DEFAULT_QUANTILES
    return FMQuantileForecast(
        item_id=item_id,
        loc=loc,
        model_id=model_id,
        quantile_matrix=matrix,
        quantile_levels=grid_levels,
        months=tuple(months),
    )


def fm_demand_pool(
    cursor: Any,
    item_id: str,
    loc: str,
    *,
    n_samples: int = 1000,
    horizon_months: int | None = None,
    rng: np.random.Generator | None = None,
    fm_config: dict[str, Any] | None = None,
) -> np.ndarray | None:
    """Convenience: return a flattened 1-D demand pool from FM quantiles.

    Flattens the ``(n_months, n_samples)`` matrix into a single demand
    pool of size ``n_months * n_samples`` suitable for
    ``TwinState.simulate(scenario={"demand_pool": ...})``.

    Returns None when the FM has no candidate rows — callers should fall
    back to the sales-history bootstrap.
    """
    fc = load_fm_quantile_forecast(
        cursor, item_id, loc,
        fm_config=fm_config,
        horizon_months=horizon_months,
    )
    if fc is None:
        return None
    samples = fc.to_sample_array(n_samples=n_samples, rng=rng)
    return samples.reshape(-1)


__all__ = [
    "FMQuantileForecast",
    "load_fm_quantile_forecast",
    "fm_demand_pool",
]
