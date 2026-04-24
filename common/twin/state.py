"""Twin state loader and Monte Carlo horizon simulation.

Extracted from ``scripts/run_ss_simulation.py`` (safety stock Monte Carlo)
so the same state model can drive the inventory projection and exception
orchestrator. The public entry point is :class:`TwinState`.

Design notes:
    - State is fetched once via :meth:`TwinState.from_db` and then
      simulated in-memory many times with different scenarios.
    - ``simulate`` returns a NumPy array of end-of-horizon stock levels so
      callers can compute any KPI (stockout %, fill rate, P10/P50/P90).
    - No GPU path here yet — CPU-only so the scaffold stays dependency
      light. The ss simulation script keeps its CuPy backend for now.

TODO(gen-4): Plug empirical quantile forecasts from the FM spine
(Chronos-2) into ``demand_pool`` as a sample draw, bypassing the sales
history bootstrap. That unblocks Cross-cutting #2 empirical SS quantiles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Demand history lookback in months — kept as a named constant to avoid a
# magic number. The ss simulation has historically used 24 months of sales.
_DEFAULT_LOOKBACK_MONTHS: int = 24
# Days per month used to convert monthly qty to a daily rate.
_DAYS_PER_MONTH: float = 30.44
# Number of empirical lead-time draws when only lt_mean/lt_std are known.
_LT_POOL_SAMPLES: int = 100
# Fallback lead time mean/std when no profile row is found.
_LT_FALLBACK_MEAN_DAYS: float = 14.0
_LT_FALLBACK_STD_DAYS: float = 0.0


@dataclass
class TwinState:
    """In-memory twin state for a single item-location pair.

    Attributes:
        item_id: Item identifier.
        loc: Location code.
        on_hand: Current on-hand units.
        demand_pool: Daily demand observations bootstrapped from sales
            history. Monte Carlo draws sample from this pool.
        lt_pool: Lead time observations (in days), already clipped to
            ``>= 1``.
        lt_distribution: Label describing the lead-time distribution
            (``"empirical"`` | ``"constant"``).
        demand_mean: Mean of ``demand_pool`` (convenience for callers).
        demand_std: Std dev of ``demand_pool`` (convenience for callers).
        lt_mean_days: Mean of ``lt_pool``.
        lt_std_days: Std dev of ``lt_pool``.
        extras: Free-form dict for consumer-specific extensions (e.g.
            safety stock targets already cached on disk).
    """

    item_id: str
    loc: str
    on_hand: float
    demand_pool: np.ndarray
    lt_pool: np.ndarray
    lt_distribution: str = "empirical"
    demand_mean: float = 0.0
    demand_std: float = 0.0
    lt_mean_days: float = _LT_FALLBACK_MEAN_DAYS
    lt_std_days: float = _LT_FALLBACK_STD_DAYS
    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_db(
        cls,
        conn: Any,
        item_id: str,
        loc: str,
        *,
        lookback_months: int = _DEFAULT_LOOKBACK_MONTHS,
        rng: np.random.Generator | None = None,
        use_fm_quantiles: bool = False,
        fm_n_samples: int = 1000,
    ) -> "TwinState":
        """Hydrate a TwinState from Postgres.

        Joins ``fact_sales_monthly`` for demand history,
        ``dim_item_lead_time_profile`` for lead time, and
        ``fact_safety_stock_targets`` for the latest on-hand (proxied via
        analytical SS when on-hand is not tracked directly).

        Caller owns the transaction.
        """
        rng = rng or np.random.default_rng(42)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT qty_shipped FROM fact_sales_monthly
                 WHERE item_id = %s AND loc = %s AND type = 1
                 ORDER BY startdate DESC
                 LIMIT %s
                """,
                (item_id, loc, lookback_months),
            )
            demand_rows = cur.fetchall()
            if not demand_rows:
                raise ValueError(f"No demand history found for {item_id}/{loc}")
            daily_demand_obs = [
                float(r[0]) / _DAYS_PER_MONTH
                for r in demand_rows
                if r[0] and r[0] > 0
            ]
            if not daily_demand_obs:
                raise ValueError(
                    f"No positive demand observations for {item_id}/{loc}"
                )

            cur.execute(
                """
                SELECT lt_mean_days, lt_std_days
                  FROM dim_item_lead_time_profile
                 WHERE item_id = %s AND loc = %s
                 ORDER BY computed_at DESC
                 LIMIT 1
                """,
                (item_id, loc),
            )
            lt_row = cur.fetchone()
            if lt_row is not None:
                lt_mean = float(lt_row[0])
                lt_std = float(lt_row[1]) if lt_row[1] else 0.0
                lt_pool = rng.normal(lt_mean, lt_std, _LT_POOL_SAMPLES).clip(1.0)
                lt_distribution = "empirical"
            else:
                lt_mean = _LT_FALLBACK_MEAN_DAYS
                lt_std = _LT_FALLBACK_STD_DAYS
                lt_pool = np.full(_LT_POOL_SAMPLES, _LT_FALLBACK_MEAN_DAYS)
                lt_distribution = "constant"

            cur.execute(
                """
                SELECT ss_combined FROM fact_safety_stock_targets
                 WHERE item_id = %s AND loc = %s AND policy_version = 'v1'
                 LIMIT 1
                """,
                (item_id, loc),
            )
            ss_row = cur.fetchone()
            analytical_ss = float(ss_row[0]) if ss_row and ss_row[0] else 0.0

        demand_arr = np.asarray(daily_demand_obs, dtype=float)

        # Optional: overlay FM-quantile draws on the demand pool. We keep
        # the sales-history bootstrap as fallback when the FM has no rows.
        if use_fm_quantiles:
            # Lazy import so tests that don't need the FM don't drag
            # load_forecast_pipeline_config's YAML read.
            from common.ml.fm_quantile_bridge import fm_demand_pool

            with conn.cursor() as cur2:
                fm_pool = fm_demand_pool(
                    cur2, item_id, loc,
                    n_samples=fm_n_samples,
                    rng=rng,
                )
            if fm_pool is not None and fm_pool.size > 0:
                # Convert monthly FM draws to a daily-rate pool so the
                # downstream simulator stays consistent.
                demand_arr = np.asarray(fm_pool, dtype=float) / _DAYS_PER_MONTH
                logger.info(
                    "TwinState using FM quantile demand pool (size=%s) for %s/%s",
                    demand_arr.size, item_id, loc,
                )

        return cls(
            item_id=item_id,
            loc=loc,
            on_hand=analytical_ss,
            demand_pool=demand_arr,
            lt_pool=np.asarray(lt_pool, dtype=float),
            lt_distribution=lt_distribution,
            demand_mean=float(demand_arr.mean()),
            demand_std=float(demand_arr.std()),
            lt_mean_days=float(lt_mean),
            lt_std_days=float(lt_std),
            extras={"analytical_ss": analytical_ss},
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        item_id: str | None = None,
        loc: str | None = None,
        scenario: dict[str, Any] | None = None,
        n_iter: int = 10_000,
        *,
        random_seed: int = 42,
    ) -> np.ndarray:
        """Run a Monte Carlo horizon simulation.

        Draws ``n_iter`` lead-time samples from ``lt_pool`` and daily
        demand samples from ``demand_pool``. Returns a 1-D NumPy array of
        end-of-horizon stock levels (``on_hand + extra_stock - demand``).

        ``scenario`` accepts optional overrides:

        - ``"extra_stock"`` (float): additional units to simulate on top
          of the twin's current ``on_hand`` (e.g. a proposed SS).
        - ``"horizon_days"`` (int): override the horizon; defaults to the
          maximum lead-time draw so the pattern matches the existing ss
          simulation.
        - ``"demand_pool"`` / ``"lt_pool"`` (array-like): replace either
          pool for this run (useful to inject FM quantile draws).

        Signature accepts ``item_id`` / ``loc`` purely so callers can
        state what they think they're simulating; a mismatch raises. The
        state was already bound at load time.
        """
        if item_id is not None and item_id != self.item_id:
            raise ValueError(
                f"TwinState bound to item_id={self.item_id!r}, called with {item_id!r}"
            )
        if loc is not None and loc != self.loc:
            raise ValueError(
                f"TwinState bound to loc={self.loc!r}, called with {loc!r}"
            )
        if n_iter <= 0:
            raise ValueError("n_iter must be positive")

        scenario = scenario or {}
        extra_stock = float(scenario.get("extra_stock", 0.0))
        demand_pool = np.asarray(
            scenario.get("demand_pool", self.demand_pool), dtype=float
        )
        lt_pool = np.asarray(scenario.get("lt_pool", self.lt_pool), dtype=float)
        if demand_pool.size == 0 or lt_pool.size == 0:
            raise ValueError("demand_pool and lt_pool must be non-empty")

        rng = np.random.default_rng(random_seed)
        lt_sim = rng.choice(lt_pool, size=n_iter, replace=True)
        lt_ints = np.maximum(1, np.round(lt_sim).astype(int))
        horizon = int(scenario.get("horizon_days", lt_ints.max()))
        if horizon < 1:
            raise ValueError("horizon_days must be >= 1")

        all_demand = rng.choice(demand_pool, size=(n_iter, horizon), replace=True)
        cumsum = np.cumsum(all_demand, axis=1)
        # Clip each iter's lead-time index to horizon so callers can force
        # a shorter horizon without error.
        effective_lt = np.minimum(lt_ints, horizon)
        demand_during_lt = cumsum[np.arange(n_iter), effective_lt - 1]

        end_of_horizon_stock = (self.on_hand + extra_stock) - demand_during_lt
        return end_of_horizon_stock


__all__ = ["TwinState"]
