"""Model-aware confidence-interval calibration for production forecasts.

Residuals are calibrated at the same ``(item_id, loc, startdate)`` grain as
the production plan.  Customer-group forecasts and actuals are summed before
the monthly error is computed; summing already-computed group errors would
overstate uncertainty whenever group errors offset one another.

Explicit model runs use only that model's backtest residuals.  Champion runs
use the exact rows tied to the single active, results-promoted champion
experiment.  Missing lineage or evidence fails closed instead of silently
pooling residuals from unrelated models.
"""

import logging
import math

import numpy as np
import pandas as pd

from common.core.constants import FORECAST_QTY_COL

logger = logging.getLogger(__name__)

CALIBRATED_MODEL_IDS = frozenset({"lgbm_cluster", "chronos2_enriched", "mstl", "nbeats", "nhits"})
_SUPPORTED_INTERVAL_MODEL_IDS = CALIBRATED_MODEL_IDS | {"champion"}
_SIGMA_COLUMNS = ["item_id", "loc", "sigma", "n_months"]


def _empty_sigma_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_SIGMA_COLUMNS)


def _validate_interval_model_id(requested_model_id: str) -> None:
    if requested_model_id not in _SUPPORTED_INTERVAL_MODEL_IDS:
        supported = ", ".join(sorted(_SUPPORTED_INTERVAL_MODEL_IDS))
        raise ValueError(
            "unsupported confidence-interval model "
            f"'{requested_model_id}'; expected one of: {supported}"
        )


def compute_dfu_sigma(residuals: pd.DataFrame) -> pd.DataFrame:
    """Compute item/location RMSE after monthly customer-group aggregation.

    ``residuals`` must retain ``startdate`` so every customer group's forecast
    and actual can first be summed to the production month grain.

    Args:
        residuals: DataFrame with item_id, loc, startdate, basefcst_pref,
            and tothist_dmd. customer_group may be present but is not grouped.

    Returns:
        DataFrame with [item_id, loc, sigma, n_months]
    """
    required = {"item_id", "loc", "startdate", FORECAST_QTY_COL, "tothist_dmd"}
    missing = required - set(residuals.columns)
    if missing:
        raise ValueError(
            f"residual calibration is missing required columns: {', '.join(sorted(missing))}"
        )
    if residuals.empty:
        return _empty_sigma_frame()

    monthly = residuals.groupby(["item_id", "loc", "startdate"], as_index=False).agg(
        forecast_qty=(FORECAST_QTY_COL, "sum"),
        actual_qty=("tothist_dmd", "sum"),
        forecast_count=(FORECAST_QTY_COL, "count"),
        actual_count=("tothist_dmd", "count"),
        row_count=(FORECAST_QTY_COL, "size"),
    )
    monthly = monthly.loc[
        (monthly["forecast_count"] == monthly["row_count"])
        & (monthly["actual_count"] == monthly["row_count"])
    ].copy()
    monthly["residual_sq"] = (monthly["forecast_qty"] - monthly["actual_qty"]) ** 2

    grouped = (
        monthly.groupby(["item_id", "loc"])
        .agg(
            rmse_sum=("residual_sq", "sum"),
            n_months=("residual_sq", "count"),
        )
        .reset_index()
    )

    grouped["sigma"] = np.sqrt(grouped["rmse_sum"] / grouped["n_months"])
    return grouped[_SIGMA_COLUMNS]


def compute_cluster_sigma(dfu_sigma: pd.DataFrame, cluster_map: dict) -> dict[str, float]:
    """
    Compute pooled sigma per cluster as DFU-count-weighted mean.

    Args:
        dfu_sigma: DataFrame with [item_id, loc, sigma, n_months]
        cluster_map: {(item_id, loc): cluster_label}

    Returns:
        {cluster_label: pooled_sigma}
    """
    if dfu_sigma.empty:
        return {}

    df = dfu_sigma.copy()
    keys = list(zip(df["item_id"], df["loc"], strict=True))
    df["cluster"] = pd.Series(keys).map(cluster_map).fillna("unknown").values

    result = {}
    for cluster, group in df.groupby("cluster"):
        if cluster == "unknown":
            continue
        weights = group["n_months"].values
        sigmas = group["sigma"].values
        total_weight = weights.sum()
        if total_weight > 0:
            result[cluster] = float(np.average(sigmas, weights=weights))

    return result


def _load_active_champion_route(conn) -> tuple[int, str]:
    """Return the single active results-promoted experiment and its lag mode."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT experiment_id, lag_mode
               FROM champion_experiment
               WHERE is_promoted = TRUE
                 AND is_results_promoted = TRUE
               ORDER BY promoted_at DESC NULLS LAST, experiment_id DESC"""
        )
        rows = cur.fetchall()

    if len(rows) != 1:
        raise RuntimeError(
            "confidence-interval calibration requires exactly one active promoted "
            f"champion result; found {len(rows)}"
        )
    experiment_id, raw_lag_mode = rows[0]
    lag_mode = str(raw_lag_mode).strip().lower()
    if lag_mode not in {"execution", "all", "0", "1", "2", "3", "4"}:
        raise RuntimeError(
            f"active champion experiment {experiment_id} has unsupported lag mode '{raw_lag_mode}'"
        )
    return int(experiment_id), lag_mode


def _load_explicit_model_sigma(conn, requested_model_id: str, lag: int) -> list[tuple]:
    """Load one model's residual RMSE after production-grain aggregation."""
    with conn.cursor() as cur:
        cur.execute(
            """WITH item_location_month AS (
                   SELECT item_id,
                          loc,
                          startdate,
                          SUM(basefcst_pref) AS forecast_qty,
                          SUM(tothist_dmd) AS actual_qty
                   FROM backtest_lag_archive
                   WHERE model_id = %s
                     AND lag = %s
                   GROUP BY item_id, loc, startdate
                   HAVING COUNT(*) = COUNT(basefcst_pref)
                      AND COUNT(*) = COUNT(tothist_dmd)
               )
               SELECT item_id,
                      loc,
                      SQRT(AVG(POWER(forecast_qty - actual_qty, 2.0))) AS sigma,
                      COUNT(*)::int AS n_months
               FROM item_location_month
               GROUP BY item_id, loc""",
            (requested_model_id, lag),
        )
        return cur.fetchall()


def _load_champion_sigma(conn, experiment_id: int, lag_mode: str, lag: int) -> list[tuple]:
    """Load residuals for the active champion's exact persisted routes."""
    if lag_mode == "execution":
        with conn.cursor() as cur:
            cur.execute(
                """WITH item_location_month AS (
                       SELECT item_id,
                              loc,
                              startdate,
                              SUM(basefcst_pref) AS forecast_qty,
                              SUM(tothist_dmd) AS actual_qty
                       FROM fact_external_forecast_monthly
                       WHERE model_id = 'champion'
                         AND champion_experiment_id = %s
                         AND lag = execution_lag
                       GROUP BY item_id, loc, startdate
                       HAVING COUNT(*) = COUNT(basefcst_pref)
                          AND COUNT(*) = COUNT(tothist_dmd)
                   )
                   SELECT item_id,
                          loc,
                          SQRT(AVG(POWER(forecast_qty - actual_qty, 2.0))) AS sigma,
                          COUNT(*)::int AS n_months
                   FROM item_location_month
                   GROUP BY item_id, loc""",
                (experiment_id,),
            )
            return cur.fetchall()

    selected_lag = lag if lag_mode == "all" else int(lag_mode)
    with conn.cursor() as cur:
        cur.execute(
            """WITH item_location_month AS (
                   SELECT item_id,
                          loc,
                          startdate,
                          SUM(basefcst_pref) AS forecast_qty,
                          SUM(tothist_dmd) AS actual_qty
                   FROM fact_external_forecast_monthly
                   WHERE model_id = 'champion'
                     AND champion_experiment_id = %s
                     AND lag = %s
                   GROUP BY item_id, loc, startdate
                   HAVING COUNT(*) = COUNT(basefcst_pref)
                      AND COUNT(*) = COUNT(tothist_dmd)
               )
               SELECT item_id,
                      loc,
                      SQRT(AVG(POWER(forecast_qty - actual_qty, 2.0))) AS sigma,
                      COUNT(*)::int AS n_months
               FROM item_location_month
               GROUP BY item_id, loc""",
            (experiment_id, selected_lag),
        )
        return cur.fetchall()


def _load_dfu_sigma_aggregated(
    conn,
    requested_model_id: str,
    lag: int = 0,
) -> pd.DataFrame:
    """Load model/routing-specific RMSE at the production item/location grain."""
    _validate_interval_model_id(requested_model_id)
    if requested_model_id == "champion":
        experiment_id, lag_mode = _load_active_champion_route(conn)
        rows = _load_champion_sigma(conn, experiment_id, lag_mode, lag)
    else:
        rows = _load_explicit_model_sigma(conn, requested_model_id, lag)

    if not rows:
        return _empty_sigma_frame()
    return pd.DataFrame(rows, columns=_SIGMA_COLUMNS)


def build_sigma_lookup(
    conn,
    config: dict,
    cluster_map: dict,
    *,
    requested_model_id: str,
) -> dict[tuple, float]:
    """Build a model-aware ``{(item_id, loc): sigma}`` lookup.

    Level 1: DFU-level RMSE (when n_months >= min_residual_months)
    Level 2: Cluster-level pooled sigma
    Level 3: Global median of observed DFU sigmas

    Applies sigma_floor and sigma_cap (cap_multiplier * median(cluster_sigmas)).

    Uses SQL-aggregated RMSE per DFU to avoid loading millions of raw residual rows.

    Args:
        conn: psycopg3 connection
        config: dict with key ``confidence_interval`` containing all CI params
        cluster_map: {(item_id, loc): cluster_label}
        requested_model_id: explicit active model or ``champion``

    Returns:
        {(item_id, loc): capped_sigma}
    """
    _validate_interval_model_id(requested_model_id)
    ci_cfg = config["confidence_interval"]
    configured_models = set(ci_cfg["source_model_ids"])
    if requested_model_id != "champion" and requested_model_id not in configured_models:
        raise ValueError(
            f"confidence-interval model '{requested_model_id}' is not enabled in "
            "confidence_interval.source_model_ids"
        )
    residual_lag = ci_cfg.get("residual_lag", 0)
    min_months = ci_cfg.get("min_residual_months", 6)
    sigma_floor = ci_cfg.get("sigma_floor", 1.0)
    cap_multiplier = ci_cfg.get("sigma_cap_multiplier", 3.0)

    logger.info(
        "Loading production-grain residual sigma for %s at lag %d",
        requested_model_id,
        residual_lag,
    )
    dfu_sigma_df = _load_dfu_sigma_aggregated(
        conn,
        requested_model_id,
        residual_lag,
    )
    if dfu_sigma_df.empty:
        raise RuntimeError(
            f"{requested_model_id} has no residual evidence for confidence-interval "
            f"calibration at lag {residual_lag}"
        )
    logger.info("Loaded sigma for %d DFUs", len(dfu_sigma_df))

    # psycopg returns NUMERIC aggregates as Decimal, which leaves pandas with
    # object dtype. Normalize at the database boundary before NumPy finite
    # checks or weighted cluster calculations.
    dfu_sigma_df = dfu_sigma_df.copy()
    dfu_sigma_df["sigma"] = pd.to_numeric(dfu_sigma_df["sigma"], errors="coerce")
    dfu_sigma_df["n_months"] = pd.to_numeric(
        dfu_sigma_df["n_months"], errors="coerce"
    )
    valid_evidence = (
        np.isfinite(dfu_sigma_df["sigma"].to_numpy(dtype=float))
        & np.isfinite(dfu_sigma_df["n_months"].to_numpy(dtype=float))
        & (dfu_sigma_df["n_months"].to_numpy(dtype=float) > 0)
    )
    invalid_count = int((~valid_evidence).sum())
    if invalid_count:
        logger.warning("Discarded %d invalid residual sigma rows", invalid_count)
    dfu_sigma_df = dfu_sigma_df.loc[valid_evidence].reset_index(drop=True)
    if dfu_sigma_df.empty:
        raise RuntimeError(
            f"{requested_model_id} residual evidence contains no finite sigma values"
        )

    cluster_sigmas = compute_cluster_sigma(dfu_sigma_df, cluster_map)

    finite_sigmas = dfu_sigma_df.loc[np.isfinite(dfu_sigma_df["sigma"]), "sigma"].astype(float)
    if finite_sigmas.empty:
        raise RuntimeError(
            f"{requested_model_id} residual evidence contains no finite sigma values"
        )
    global_sigma = float(np.median(finite_sigmas))

    # Sigma cap: cap_multiplier * median of cluster sigmas (must not be below floor)
    sigma_cap = max(cap_multiplier * global_sigma, sigma_floor)

    def _apply_guards(sigma: float) -> float:
        return float(np.clip(sigma, sigma_floor, sigma_cap))

    # Pre-index dfu_sigma_df for O(1) per-DFU lookup (avoids O(N²) DataFrame scans)
    sigma_index: dict[tuple, tuple] = {}
    if not dfu_sigma_df.empty:
        sigma_index = {
            (item_id, loc): (float(sigma), int(n_months))
            for item_id, loc, sigma, n_months in zip(
                dfu_sigma_df["item_id"],
                dfu_sigma_df["loc"],
                dfu_sigma_df["sigma"],
                dfu_sigma_df["n_months"],
                strict=True,
            )
        }

    lookup: dict[tuple, float] = {}
    n_dfu_level = 0
    n_cluster_level = 0
    n_global_level = 0

    # All DFUs in cluster_map need a sigma
    for (item_id, loc), cluster in cluster_map.items():
        # Check DFU-level sigma via O(1) index lookup
        dfu_entry = sigma_index.get((item_id, loc))

        if dfu_entry is not None and dfu_entry[1] >= min_months:
            sigma = _apply_guards(dfu_entry[0])
            n_dfu_level += 1
        elif cluster in cluster_sigmas:
            sigma = _apply_guards(cluster_sigmas[cluster])
            n_cluster_level += 1
        else:
            sigma = _apply_guards(global_sigma)
            n_global_level += 1

        lookup[(item_id, loc)] = sigma

    logger.info(
        "Sigma lookup built: %d DFUs (%d DFU-level, %d cluster-fallback, %d global-fallback)",
        len(lookup),
        n_dfu_level,
        n_cluster_level,
        n_global_level,
    )
    return lookup


def compute_ci_bounds(
    point_forecast: float,
    sigma: float,
    horizon: int,
    z_lower: float,
    z_upper: float,
    scaling: str = "sqrt",
) -> tuple[float, float]:
    """
    Compute (lower, upper) CI bounds for a single point forecast at a given horizon.

    horizon_scale:
        "sqrt"   -> sqrt(horizon)   # random walk uncertainty growth
        "linear" -> horizon         # linear uncertainty growth
        "none"   -> 1.0             # constant width

    lower = max(0, round(point_forecast - z_lower * sigma * scale, 2))
    upper =        round(point_forecast + z_upper * sigma * scale, 2)

    Guarantees: lower >= 0, upper >= point_forecast

    Args:
        point_forecast: the ML point forecast value
        sigma: per-DFU forecast error sigma (RMSE)
        horizon: forecast horizon in months (1 = T+1, 2 = T+2, ...)
        z_lower: Z-score for lower bound (e.g. 1.282 for P10)
        z_upper: Z-score for upper bound (e.g. 1.282 for P90)
        scaling: "sqrt" | "linear" | "none"

    Returns:
        (lower, upper) tuple
    """
    h = max(1, horizon)
    if scaling == "sqrt":
        scale = math.sqrt(h)
    elif scaling == "linear":
        scale = float(h)
    else:
        scale = 1.0

    lower = max(0.0, round(point_forecast - z_lower * sigma * scale, 2))
    upper = round(max(point_forecast, point_forecast + z_upper * sigma * scale), 2)
    return lower, upper
