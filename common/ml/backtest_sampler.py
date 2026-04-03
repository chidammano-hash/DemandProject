"""Stratified DFU sampling for fast backtest iteration.

Full backtest: 50K DFUs, ~30 min, 2.73M predictions.
Sampled backtest: 5K DFUs, ~3 min, ~273K predictions.
Expected accuracy deviation: +/-1.5% of full run.

All DB access uses psycopg3 with ``%s`` placeholders and explicit connection
management via ``common.db.get_db_params``.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any

import numpy as np
import pandas as pd
import psycopg

from common.core.db import get_db_params
from common.core.utils import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_sampling_config() -> dict[str, Any]:
    """Load sampling config.

    Reads from ``forecast_pipeline_config.yaml`` ``backtest_sampling`` section
    first, falls back to ``backtest_sampling_config.yaml`` ``sampling`` section.
    """
    try:
        from common.utils import load_forecast_pipeline_config
        pipeline_cfg = load_forecast_pipeline_config()
        sampling = pipeline_cfg.get("backtest_sampling", {})
        if sampling:
            return sampling
    except (ImportError, FileNotFoundError):
        pass
    # Legacy fallback
    cfg = load_config("backtest_sampling_config.yaml")
    return cfg.get("sampling", {})


# ---------------------------------------------------------------------------
# 1. Compute cluster strata
# ---------------------------------------------------------------------------


def compute_cluster_strata(conn: psycopg.Connection) -> dict[int, dict[str, Any]]:
    """Query dim_sku for ml_cluster assignments and fact_sales_monthly for demand stats.

    For each cluster: count DFUs, compute mean_demand, cv, zero_pct.

    Parameters
    ----------
    conn : psycopg.Connection
        An open psycopg3 connection.

    Returns
    -------
    dict[int, dict]
        ``{cluster_id: {n_dfus, mean_demand, cv, zero_pct, sku_cks}}``
    """
    sql = """
        WITH dfu_demand AS (
            SELECT
                d.sku_ck,
                COALESCE(NULLIF(d.ml_cluster, ''), 'unassigned') AS ml_cluster,
                AVG(s.qty)                                       AS avg_qty,
                STDDEV(s.qty)                                    AS std_qty,
                SUM(CASE WHEN s.qty = 0 THEN 1 ELSE 0 END)::float
                    / GREATEST(COUNT(*), 1)                      AS zero_pct
            FROM dim_sku d
            INNER JOIN fact_sales_monthly s
                ON d.item_id = s.item_id
               AND d.customer_group = s.customer_group
               AND d.loc = s.loc
            WHERE s.qty IS NOT NULL
            GROUP BY d.sku_ck, d.ml_cluster
        )
        SELECT
            ml_cluster,
            COUNT(*)                            AS n_dfus,
            AVG(avg_qty)                        AS mean_demand,
            CASE
                WHEN AVG(avg_qty) > 0
                THEN AVG(std_qty) / AVG(avg_qty)
                ELSE 0
            END                                 AS cv,
            AVG(zero_pct)                       AS zero_pct,
            ARRAY_AGG(sku_ck)                   AS sku_cks
        FROM dfu_demand
        GROUP BY ml_cluster
        ORDER BY ml_cluster
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    strata: dict[int, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        cluster_label = row[0]
        strata[idx] = {
            "cluster_label": cluster_label,
            "n_dfus": int(row[1]),
            "mean_demand": float(row[2]) if row[2] is not None else 0.0,
            "cv": float(row[3]) if row[3] is not None else 0.0,
            "zero_pct": float(row[4]) if row[4] is not None else 0.0,
            "sku_cks": list(row[5]) if row[5] else [],
        }

    logger.info(
        "Computed strata for %d clusters covering %d total DFUs",
        len(strata),
        sum(s["n_dfus"] for s in strata.values()),
    )
    return strata


# ---------------------------------------------------------------------------
# 2. Stratified sample
# ---------------------------------------------------------------------------


def _allocate_proportional(
    strata: dict[int, dict[str, Any]],
    target_n: int,
    min_per_cluster: int,
) -> dict[int, int]:
    """Allocate samples proportionally to cluster size."""
    total = sum(s["n_dfus"] for s in strata.values())
    if total == 0:
        return {}

    allocation: dict[int, int] = {}
    for cid, s in strata.items():
        raw = target_n * (s["n_dfus"] / total)
        allocation[cid] = max(min_per_cluster, round(raw))
    return allocation


def _allocate_equal(
    strata: dict[int, dict[str, Any]],
    target_n: int,
    min_per_cluster: int,
) -> dict[int, int]:
    """Allocate samples equally across clusters."""
    n_clusters = len(strata)
    if n_clusters == 0:
        return {}

    per_cluster = max(min_per_cluster, target_n // n_clusters)
    return {cid: per_cluster for cid in strata}


def _allocate_sqrt(
    strata: dict[int, dict[str, Any]],
    target_n: int,
    min_per_cluster: int,
) -> dict[int, int]:
    """Allocate proportionally to sqrt(cluster_size)."""
    sqrt_sizes = {cid: math.sqrt(s["n_dfus"]) for cid, s in strata.items()}
    total_sqrt = sum(sqrt_sizes.values())
    if total_sqrt == 0:
        return {}

    allocation: dict[int, int] = {}
    for cid, sq in sqrt_sizes.items():
        raw = target_n * (sq / total_sqrt)
        allocation[cid] = max(min_per_cluster, round(raw))
    return allocation


_ALLOCATORS = {
    "proportional": _allocate_proportional,
    "equal": _allocate_equal,
    "sqrt": _allocate_sqrt,
}


def stratified_sample(
    conn: psycopg.Connection,
    target_n: int = 5000,
    method: str = "proportional",
    seed: int | None = None,
) -> list[str]:
    """Perform stratified DFU sampling across ml_cluster strata.

    Parameters
    ----------
    conn : psycopg.Connection
        An open psycopg3 connection.
    target_n : int
        Target number of DFUs to sample.
    method : str
        Allocation method: ``"proportional"``, ``"equal"``, or ``"sqrt"``.
    seed : int | None
        Random seed for reproducibility. If None, uses config default.

    Returns
    -------
    list[str]
        Sampled ``sku_ck`` values.
    """
    cfg = _load_sampling_config()
    if seed is None:
        seed = cfg.get("seed", 42)
    min_per_cluster = cfg.get("min_per_cluster", 10)

    allocator = _ALLOCATORS.get(method)
    if allocator is None:
        raise ValueError(
            f"Unknown sampling method '{method}'. "
            f"Choose from: {', '.join(_ALLOCATORS)}"
        )

    strata = compute_cluster_strata(conn)
    if not strata:
        logger.warning("No strata found — returning empty sample")
        return []

    allocation = allocator(strata, target_n, min_per_cluster)

    rng = random.Random(seed)
    sampled: list[str] = []

    for cid, n_to_sample in allocation.items():
        sku_cks = strata[cid]["sku_cks"]
        n_available = len(sku_cks)
        actual_n = min(n_to_sample, n_available)
        sampled.extend(rng.sample(sku_cks, actual_n))

    logger.info(
        "Stratified sample: target=%d, method=%s, actual=%d across %d clusters",
        target_n, method, len(sampled), len(strata),
    )
    return sampled


# ---------------------------------------------------------------------------
# 3. Filter backtest data
# ---------------------------------------------------------------------------


def filter_backtest_data(
    sales_df: pd.DataFrame,
    dfu_attrs: pd.DataFrame,
    sampled_skus: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter sales_df and dfu_attrs to only sampled SKUs.

    Parameters
    ----------
    sales_df : pd.DataFrame
        Sales data with ``sku_ck`` column.
    dfu_attrs : pd.DataFrame
        DFU attribute data with ``sku_ck`` column.
    sampled_skus : list[str]
        List of sampled ``sku_ck`` values from :func:`stratified_sample`.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(filtered_sales, filtered_attrs)``
    """
    sku_set = set(sampled_skus)

    filtered_sales = sales_df[sales_df["sku_ck"].isin(sku_set)].copy()
    filtered_attrs = dfu_attrs[dfu_attrs["sku_ck"].isin(sku_set)].copy()

    logger.info(
        "Filtered backtest data: %d -> %d sales rows, %d -> %d DFUs",
        len(sales_df), len(filtered_sales),
        len(dfu_attrs), len(filtered_attrs),
    )
    return filtered_sales, filtered_attrs


# ---------------------------------------------------------------------------
# 4. Estimate accuracy deviation
# ---------------------------------------------------------------------------


def estimate_accuracy_deviation(
    sample_size: int,
    total_size: int,
    n_clusters: int,
) -> float:
    """Statistical estimate of expected deviation from full backtest.

    Based on stratified sampling theory. The standard error of a stratified
    mean is reduced by the factor ``sqrt(1 - f)`` (finite population
    correction) where ``f = n / N``, and further improved by stratification
    which captures between-cluster variance.

    Parameters
    ----------
    sample_size : int
        Number of DFUs in the sample.
    total_size : int
        Total DFUs in the population.
    n_clusters : int
        Number of strata (clusters) used.

    Returns
    -------
    float
        Expected +/- percentage point deviation (95% CI).
    """
    if sample_size <= 0 or total_size <= 0:
        return 0.0

    # Finite population correction
    f = min(sample_size / total_size, 1.0)
    fpc = math.sqrt(max(1.0 - f, 0.0))

    # Base SE from simple random sampling (~30% typical CV for accuracy)
    base_cv = 0.30
    base_se = base_cv / math.sqrt(sample_size)

    # Stratification efficiency: reduces between-cluster variance.
    # With k strata, effective SE is reduced by approx sqrt(k) factor
    # on the between-cluster component. Use a conservative 0.7 efficiency.
    strat_efficiency = 0.7 if n_clusters > 1 else 1.0

    se = base_se * fpc * strat_efficiency

    # 95% CI = ~1.96 * SE, expressed as percentage points
    deviation = 1.96 * se * 100.0

    return round(deviation, 2)


# ---------------------------------------------------------------------------
# 5. Validate sample representativeness
# ---------------------------------------------------------------------------


def validate_sample_representativeness(
    sampled_stats: dict[int, dict[str, Any]],
    full_stats: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Compare demand distribution of sample vs full population per cluster.

    For each cluster present in both ``sampled_stats`` and ``full_stats``,
    computes:
    - KS statistic (Kolmogorov-Smirnov) approximation based on mean/CV
    - Mean ratio (sampled / full)
    - CV ratio (sampled / full)
    - Whether the sample is representative (KS < threshold)

    Parameters
    ----------
    sampled_stats : dict[int, dict]
        Per-cluster stats from the sampled population.
    full_stats : dict[int, dict]
        Per-cluster stats from the full population.

    Returns
    -------
    dict[int, dict]
        ``{cluster_id: {ks_stat, mean_ratio, cv_ratio, is_representative}}``
    """
    cfg = _load_sampling_config()
    threshold = cfg.get("representativeness_threshold", 0.05)

    results: dict[int, dict[str, Any]] = {}

    for cid in full_stats:
        full = full_stats[cid]
        sampled = sampled_stats.get(cid)

        if sampled is None:
            results[cid] = {
                "ks_stat": 1.0,
                "mean_ratio": 0.0,
                "cv_ratio": 0.0,
                "is_representative": False,
            }
            continue

        full_mean = full.get("mean_demand", 0.0)
        sampled_mean = sampled.get("mean_demand", 0.0)
        full_cv = full.get("cv", 0.0)
        sampled_cv = sampled.get("cv", 0.0)

        mean_ratio = (sampled_mean / full_mean) if full_mean > 0 else 0.0
        cv_ratio = (sampled_cv / full_cv) if full_cv > 0 else 0.0

        # Approximate KS stat from distribution parameter differences.
        # A proper KS test requires raw data; here we use a proxy based on
        # how far the sample mean and CV deviate from the population.
        mean_dev = abs(mean_ratio - 1.0)
        cv_dev = abs(cv_ratio - 1.0)
        ks_stat = max(mean_dev, cv_dev)

        results[cid] = {
            "ks_stat": round(ks_stat, 4),
            "mean_ratio": round(mean_ratio, 4),
            "cv_ratio": round(cv_ratio, 4),
            "is_representative": ks_stat < threshold,
        }

    return results
