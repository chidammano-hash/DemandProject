"""Forecast confidence interval (CI) band computation.

Derives per-DFU forecast uncertainty (sigma) from champion model backtest
residuals stored in backtest_lag_archive. Applies Z-score-based CI bands
with configurable horizon scaling (sqrt, linear, none).

Three-level sigma fallback:
  1. DFU-level RMSE (when n_months >= min_residual_months)
  2. Cluster-level pooled sigma (weighted mean across DFUs in cluster)
  3. Global median sigma (final fallback)
"""

import math
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_champion_residuals(conn, source_model_ids: list[str], lag: int = 0) -> pd.DataFrame:
    """
    Load backtest residuals for specified models at the given lag.

    Queries backtest_lag_archive for rows where:
      - model_id IN source_model_ids
      - lag = lag param
      - both basefcst_pref (forecast) and tothist_dmd (actual) are NOT NULL

    Returns DataFrame with columns:
      item_id (str), loc (str), startdate, basefcst_pref (float), tothist_dmd (float), model_id (str)

    Uses psycopg3 %s placeholders.
    """
    if not source_model_ids:
        return pd.DataFrame(columns=["item_id", "loc", "startdate", "basefcst_pref", "tothist_dmd", "model_id"])

    placeholders = ", ".join(["%s"] * len(source_model_ids))
    sql = f"""
        SELECT item_id, loc, startdate, basefcst_pref, tothist_dmd, model_id
        FROM backtest_lag_archive
        WHERE model_id IN ({placeholders})
          AND lag = %s
          AND basefcst_pref IS NOT NULL
          AND tothist_dmd IS NOT NULL
          AND tothist_dmd > 0
    """
    params = source_model_ids + [lag]
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    if not rows:
        return pd.DataFrame(columns=["item_id", "loc", "startdate", "basefcst_pref", "tothist_dmd", "model_id"])

    return pd.DataFrame(rows, columns=["item_id", "loc", "startdate", "basefcst_pref", "tothist_dmd", "model_id"])


def compute_dfu_sigma(residuals: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RMSE per DFU from backtest residuals.

    residual_i = basefcst_pref_i - tothist_dmd_i
    sigma_dfu = sqrt(mean(residual_i^2))

    Args:
        residuals: DataFrame with [item_id, loc, basefcst_pref, tothist_dmd]

    Returns:
        DataFrame with [item_id, loc, sigma, n_months]
    """
    if residuals.empty:
        return pd.DataFrame(columns=["item_id", "loc", "sigma", "n_months"])

    df = residuals.copy()
    df["residual_sq"] = (df["basefcst_pref"] - df["tothist_dmd"]) ** 2

    grouped = df.groupby(["item_id", "loc"]).agg(
        rmse_sum=("residual_sq", "sum"),
        n_months=("residual_sq", "count"),
    ).reset_index()

    grouped["sigma"] = np.sqrt(grouped["rmse_sum"] / grouped["n_months"])
    return grouped[["item_id", "loc", "sigma", "n_months"]]


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
    keys = list(zip(df["item_id"], df["loc"]))
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


def _load_dfu_sigma_aggregated(conn, source_model_ids: list[str], lag: int = 0) -> pd.DataFrame:
    """
    Compute per-DFU RMSE (sigma) and month count directly via SQL aggregation.

    Avoids loading millions of raw residual rows by pushing GROUP BY to the DB.
    Returns DataFrame with columns: [item_id, loc, sigma, n_months].

    Uses psycopg3 %s placeholders.
    """
    if not source_model_ids:
        return pd.DataFrame(columns=["item_id", "loc", "sigma", "n_months"])

    placeholders = ", ".join(["%s"] * len(source_model_ids))
    sql = f"""
        SELECT
            item_id,
            loc,
            SQRT(SUM(POWER(basefcst_pref - tothist_dmd, 2.0)) / COUNT(*)) AS sigma,
            COUNT(*)::int AS n_months
        FROM backtest_lag_archive
        WHERE model_id IN ({placeholders})
          AND lag = %s
          AND basefcst_pref IS NOT NULL
          AND tothist_dmd IS NOT NULL
          AND tothist_dmd > 0
        GROUP BY item_id, loc
    """
    params = source_model_ids + [lag]
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    if not rows:
        return pd.DataFrame(columns=["item_id", "loc", "sigma", "n_months"])

    return pd.DataFrame(rows, columns=["item_id", "loc", "sigma", "n_months"])


def build_sigma_lookup(conn, config: dict, cluster_map: dict) -> dict[tuple, float]:
    """
    Build {(item_id, loc): sigma} lookup with three-level fallback.

    Level 1: DFU-level RMSE (when n_months >= min_residual_months)
    Level 2: Cluster-level pooled sigma
    Level 3: Global median of all cluster sigmas

    Applies sigma_floor and sigma_cap (cap_multiplier * median(cluster_sigmas)).

    Uses SQL-aggregated RMSE per DFU to avoid loading millions of raw residual rows.

    Args:
        conn: psycopg3 connection
        config: dict with key 'confidence_interval' containing all CI params
        cluster_map: {(item_id, loc): cluster_label}

    Returns:
        {(item_id, loc): capped_sigma}
    """
    ci_cfg = config.get("confidence_interval", {})
    source_model_ids = ci_cfg.get("source_model_ids", ["lgbm_cluster", "catboost_cluster", "xgboost_cluster"])
    residual_lag = ci_cfg.get("residual_lag", 0)
    min_months = ci_cfg.get("min_residual_months", 6)
    sigma_floor = ci_cfg.get("sigma_floor", 1.0)
    cap_multiplier = ci_cfg.get("sigma_cap_multiplier", 3.0)

    logger.info("Loading aggregated DFU sigma for models: %s at lag %d", source_model_ids, residual_lag)
    dfu_sigma_df = _load_dfu_sigma_aggregated(conn, source_model_ids, residual_lag)
    logger.info("Loaded sigma for %d DFUs", len(dfu_sigma_df))

    cluster_sigmas = compute_cluster_sigma(dfu_sigma_df, cluster_map)

    # Global fallback: median of all cluster sigmas
    if cluster_sigmas:
        global_sigma = float(np.median(list(cluster_sigmas.values())))
    else:
        global_sigma = max(sigma_floor, 10.0)  # absolute fallback

    # Sigma cap: cap_multiplier * median of cluster sigmas (must not be below floor)
    sigma_cap = max(cap_multiplier * global_sigma, sigma_floor)

    def _apply_guards(sigma: float) -> float:
        return float(np.clip(sigma, sigma_floor, sigma_cap))

    # Pre-index dfu_sigma_df for O(1) per-DFU lookup (avoids O(N²) DataFrame scans)
    sigma_index: dict[tuple, tuple] = {}
    if not dfu_sigma_df.empty:
        sigma_index = dict(zip(
            zip(dfu_sigma_df["item_id"], dfu_sigma_df["loc"]),
            zip(dfu_sigma_df["sigma"].astype(float), dfu_sigma_df["n_months"].astype(int)),
        ))

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
        len(lookup), n_dfu_level, n_cluster_level, n_global_level,
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
