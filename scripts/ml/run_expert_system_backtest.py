#!/usr/bin/env python3
"""Expert System Backtest (ExpSys).

Full-population backtesting using segment-assigned algorithms.
Assigns one algorithm per demand archetype (nbeats / chronos / mstl),
runs 10 expanding-window timeframes, records accuracy at lags 0–4
and at execution lag, then loads results to PostgreSQL.

Usage
-----
    uv run python -m scripts.ml.run_expert_system_backtest
    uv run python -m scripts.ml.run_expert_system_backtest --replace
    uv run python -m scripts.ml.run_expert_system_backtest --skip-load
    uv run python -m scripts.ml.run_expert_system_backtest --config config/forecasting/expert_system_backtest.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.ml.expert_panel.demand_classifier import classify_demand
from common.ml.expert_panel.statistical_models import run_statistical_models
from common.ml.expert_panel.tree_models import run_tree_models
from common.ml.expert_panel.dl_models import run_dl_models
from common.ml.expert_panel.foundation_models import run_foundation_models
from common.ml.expert_panel.statistical_upgrades import run_statistical_upgrades
from common.core.constants import FORECAST_QTY_COL
from common.ml.feature_engineering import build_feature_matrix
from common.core.db import get_db_params
from common.core.planning_date import get_planning_date
from common.ml.backtest_framework import generate_timeframes
from common.core.utils import load_config

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "ExpSys"
_MAX_LAG = 4  # DB constraint: CHECK (lag BETWEEN 0 AND 4)

# Algorithms that run on CPU only — safe to run in background threads concurrently
# with GPU-bound algorithms (nbeats / chronos use MPS/CUDA; mstl/rolling_mean do not).
_GPU_ALGOS: frozenset[str] = frozenset({"nbeats", "nhits", "chronos"})
_CPU_ALGOS: frozenset[str] = frozenset({"mstl", "tsb", "imapa", "adida", "autoces", "dynamic_theta", "holt_winters", "simple_es", "croston_sba", "theta", "rolling_mean"})
_TREE_ALGOS: frozenset[str] = frozenset({"lgbm_cluster", "catboost_cluster", "xgboost_cluster"})

_ARCHIVE_COLS = [
    "forecast_ck", "item_id", "customer_group", "loc",
    "fcstdate", "startdate", "lag", "execution_lag",
    FORECAST_QTY_COL, "tothist_dmd", "model_id", "timeframe",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_full_population(
    loc_filter: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load sales history, DFU attributes, and item attributes.

    Args:
        loc_filter: When set, restricts to DFUs at this location only.

    Returns:
        sales_df  — columns: sku_ck, item_id, customer_group, loc, startdate, qty
        dfu_attrs — columns: sku_ck, item_id, customer_group, loc, execution_lag,
                             ml_cluster, region, brand, abc_vol, ...
        item_attrs — columns: item_id, category, brand_name, class, sub_class, ...
    """
    db = get_db_params()
    planning_cutoff = get_planning_date().replace(day=1)

    loc_clause = "AND loc = %s" if loc_filter else ""
    loc_params: tuple = (planning_cutoff, loc_filter) if loc_filter else (planning_cutoff,)

    logger.info(
        "Loading sales (cutoff=%s%s)...",
        planning_cutoff,
        f", loc={loc_filter}" if loc_filter else "",
    )
    with psycopg.connect(**db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT item_id || '_' || customer_group || '_' || loc AS sku_ck,
                       item_id, customer_group, loc, startdate,
                       COALESCE(qty, 0) AS qty
                FROM fact_sales_monthly
                WHERE startdate < %s {loc_clause}
                ORDER BY sku_ck, startdate
                """,
                loc_params,
            )
            sales_df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])

        with conn.cursor() as cur:
            dfu_where = "WHERE loc = %s" if loc_filter else ""
            dfu_params = (loc_filter,) if loc_filter else ()
            cur.execute(
                f"""
                SELECT sku_ck, item_id, customer_group, loc,
                       COALESCE(execution_lag, 0) AS execution_lag,
                       COALESCE(ml_cluster, '0') AS ml_cluster,
                       region, brand, abc_vol,
                       seasonality_profile, variability_class, abc_xyz_segment
                FROM dim_sku
                {dfu_where}
                """,
                dfu_params,
            )
            dfu_attrs = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT item_id, category, brand_name,
                       class, sub_class, case_weight, bpc, item_proof
                FROM dim_item
                """
            )
            item_attrs = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])

    sales_df["startdate"] = pd.to_datetime(sales_df["startdate"])
    sales_df["qty"] = sales_df["qty"].astype(float)  # psycopg3 returns Decimal; feature_engineering needs float
    logger.info(
        "Loaded %d sales rows, %d DFUs, %d DFU-attrs, %d item-attrs",
        len(sales_df), sales_df["sku_ck"].nunique(), len(dfu_attrs), len(item_attrs),
    )
    return sales_df, dfu_attrs, item_attrs


# ---------------------------------------------------------------------------
# Classification + assignment
# ---------------------------------------------------------------------------

def classify_and_assign(
    sales_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Classify DFUs into archetypes and attach primary algorithm assignment.

    Returns classification_df with an extra `assigned_algorithm` column.
    """
    class_cfg = cfg.get("demand_classification", {})
    classification = classify_demand(
        sales_df,
        adi_threshold=class_cfg.get("adi_threshold", 1.32),
        cv2_threshold=class_cfg.get("cv2_threshold", 0.49),
        high_volume_percentile=class_cfg.get("high_volume_percentile", 90),
        min_history_months=class_cfg.get("min_history_months", 6),
        cv2_volatile_threshold=class_cfg.get("cv2_volatile_threshold"),
        smooth_short_history_months=class_cfg.get("smooth_short_history_months"),
    )

    assignment_cfg = cfg.get("assignment", {})
    # Only archetype→algo mappings; exclude special keys like "fallbacks"
    arch_to_algo = {
        k: v for k, v in assignment_cfg.items()
        if isinstance(v, str) and k not in ("default",)
    }
    classification["assigned_algorithm"] = classification["archetype"].map(arch_to_algo)
    unassigned = classification["assigned_algorithm"].isna().sum()
    if unassigned:
        default_algo = assignment_cfg.get("fallbacks", {}).get("default", ["rolling_mean"])[0]
        logger.warning("%d DFUs have no assignment; defaulting to %s", unassigned, default_algo)
        classification["assigned_algorithm"] = classification["assigned_algorithm"].fillna(default_algo)

    for archetype, algo in sorted(arch_to_algo.items()):
        n = (classification["archetype"] == archetype).sum()
        logger.info("  %s → %s  (%d DFUs)", archetype, algo, n)

    # C10: History-length sub-routing within insufficient segment.
    # DFUs with very short history (n_periods <= threshold) route to a simpler
    # algorithm — any learned model will overfit or fail on 1–2 observations.
    insufficient_short_max = class_cfg.get("insufficient_short_max_periods")
    if insufficient_short_max is not None:
        insufficient_short_algo = assignment_cfg.get("insufficient_short")
        if insufficient_short_algo:
            short_mask = (
                (classification["archetype"] == "insufficient")
                & (classification["n_periods"] <= insufficient_short_max)
            )
            if short_mask.any():
                classification.loc[short_mask, "assigned_algorithm"] = insufficient_short_algo
                logger.info(
                    "  insufficient_short (n_periods≤%d) → %s  (%d DFUs)",
                    insufficient_short_max, insufficient_short_algo, short_mask.sum(),
                )

    return classification


# ---------------------------------------------------------------------------
# Rolling-mean fallback (inline, no library dependency)
# ---------------------------------------------------------------------------

def _rolling_mean_preds(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    window: int = 6,
    min_history: int = 3,
) -> pd.DataFrame:
    """Compute rolling-mean forecasts for all DFUs in sales_df."""
    rows = []
    for sku_ck, grp in sales_df.groupby("sku_ck", sort=False):
        qty_sorted = grp.sort_values("startdate")["qty"]
        if len(qty_sorted) < min_history:
            val = float(qty_sorted.mean()) if len(qty_sorted) > 0 else 0.0
        else:
            val = float(qty_sorted.tail(window).mean())
        val = max(val, 0.0)
        for month in predict_months:
            rows.append({"sku_ck": sku_ck, "startdate": month,
                         FORECAST_QTY_COL: val, "algorithm_id": "rolling_mean"})
    if not rows:
        return pd.DataFrame(columns=["sku_ck", "startdate", FORECAST_QTY_COL, "algorithm_id"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _trim_leading_zeros(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Remove pre-launch structural zeros before the first positive demand month.

    Leading zeros are data artifacts (system recording "no product" as zero),
    not real demand observations. They corrupt ADI, CV², seasonal indices, and
    rolling-mean baselines for all newly launched or re-introduced items.
    """
    parts: list[pd.DataFrame] = []
    for _sku, grp in sales_df.groupby("sku_ck", sort=False):
        grp_sorted = grp.sort_values("startdate")
        nonzero = grp_sorted["qty"] > 0
        if nonzero.any():
            parts.append(grp_sorted.loc[nonzero.idxmax():])
        else:
            parts.append(grp_sorted)
    if not parts:
        return sales_df.iloc[:0].copy()
    result = pd.concat(parts, ignore_index=True)
    trimmed = len(sales_df) - len(result)
    if trimmed > 0:
        logger.info("Trimmed %d leading-zero rows across %d DFUs", trimmed, len(parts))
    return result


# ---------------------------------------------------------------------------
# Per-timeframe runner
# ---------------------------------------------------------------------------

def _run_model(
    model_id: str,
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Dispatch to the correct model runner."""
    empty = pd.DataFrame(columns=["sku_ck", "startdate", FORECAST_QTY_COL, "algorithm_id"])
    if sales_df.empty:
        return empty
    params = cfg.get(model_id, {})
    if model_id in ("nbeats", "nhits"):
        return run_dl_models(sales_df, predict_months, {model_id: params})
    if model_id == "chronos":
        return run_foundation_models(sales_df, predict_months, {model_id: params})
    if model_id in ("mstl", "tsb", "imapa", "adida", "autoces", "dynamic_theta"):
        return run_statistical_upgrades(
            sales_df, predict_months, {model_id: params},
            n_workers=cfg.get("experiment", {}).get("n_workers", 8),
        )
    if model_id in ("holt_winters", "simple_es", "croston_sba", "auto_arima", "theta"):
        return run_statistical_models(
            sales_df, predict_months, {model_id: params},
            n_workers=cfg.get("experiment", {}).get("n_workers", 8),
        )
    if model_id == "rolling_mean":
        rm_params = cfg.get("rolling_mean", {})
        return _rolling_mean_preds(
            sales_df, predict_months,
            window=rm_params.get("window", 6),
            min_history=rm_params.get("min_history", 3),
        )
    logger.warning("Unknown model_id '%s'; skipping", model_id)
    return empty


def _run_cascade_group(
    primary_algo: str,
    sku_list: list[str],
    train_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    fallback_map: dict[str, list[str]],
    default_fallbacks: list[str],
    cfg: dict[str, Any],
    tf_label: str,
    *,
    cpu_safe: bool = False,
) -> pd.DataFrame:
    """Run the full fallback cascade for one algorithm group.

    Args:
        cpu_safe: When True, strip GPU algorithms from the cascade.  Use this
            when calling from a background thread to avoid MPS/CUDA access
            from a non-main thread.  The next CPU fallback (rolling_mean)
            handles the remaining DFUs instead.
    """
    empty = pd.DataFrame(columns=["sku_ck", "startdate", FORECAST_QTY_COL, "algorithm_id"])
    target_skus = set(sku_list)
    cascade = [primary_algo] + fallback_map.get(primary_algo, default_fallbacks)
    # Deduplicate while preserving order
    seen: set[str] = set()
    cascade = [a for a in cascade if not (a in seen or seen.add(a))]  # type: ignore[func-returns-value]
    if "rolling_mean" not in cascade:
        cascade.append("rolling_mean")
    if cpu_safe:
        cascade = [a for a in cascade if a not in _GPU_ALGOS]

    parts: list[pd.DataFrame] = []
    covered: set[str] = set()
    for algo in cascade:
        remaining = target_skus - covered
        if not remaining:
            break
        subset_sales = train_df[train_df["sku_ck"].isin(remaining)]
        preds = _run_model(algo, subset_sales, predict_months, cfg)
        if not preds.empty:
            parts.append(preds)
            newly_covered = preds["sku_ck"].nunique()
            covered.update(preds["sku_ck"].unique())
            if algo != primary_algo:
                logger.info(
                    "  tf=%s %s fallback→%s: %d DFUs",
                    tf_label, primary_algo, algo, newly_covered,
                )

    # Bug fix: DFUs with no history in train_df get zero predictions from all
    # cascade algorithms. Produce a naive zero forecast so they are not dropped
    # from accuracy computation (prevents selection-effect bias in results).
    if uncovered := target_skus - covered:
        naive_rows = []
        for sku_ck in uncovered:
            for month in predict_months:
                naive_rows.append({
                    "sku_ck": sku_ck,
                    "startdate": month,
                    FORECAST_QTY_COL: 0.0,
                    "algorithm_id": "naive_zero",
                })
        if naive_rows:
            parts.append(pd.DataFrame(naive_rows))
            logger.info(
                "tf=%s %s: %d DFUs had no history → naive_zero forecast",
                tf_label, primary_algo, len(uncovered),
            )
    return pd.concat(parts, ignore_index=True) if parts else empty


def run_timeframe(
    tf: dict[str, Any],
    sales_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    cfg: dict[str, Any],
    grid: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run ExpSys backtest for one timeframe.

    For each assigned algorithm, runs on its DFU group then applies the
    fallback cascade for any DFUs that produced no predictions.

    Returns DataFrame with columns:
        sku_ck, startdate, basefcst_pref, algorithm_id, timeframe_label, train_end
    """
    exp_cfg = cfg.get("experiment", {})
    horizon = exp_cfg.get("forecast_horizon", 6)
    max_lags = exp_cfg.get("max_lags", _MAX_LAG)

    train_end = tf["train_end"]
    predict_start = tf["predict_start"]
    # Only generate as many months as needed for lags 0..max_lags
    predict_months = pd.date_range(
        start=predict_start, periods=min(max_lags + 1, horizon), freq="MS"
    ).tolist()

    train_df = sales_df[sales_df["startdate"] <= train_end].copy()

    # C7: Trim pre-launch structural zeros to improve normalization and seasonal fits
    preproc_cfg = cfg.get("preprocessing", {})
    if preproc_cfg.get("trim_leading_zeros", False):
        train_df = _trim_leading_zeros(train_df)

    assignment_cfg = cfg.get("assignment", {})
    fallback_map: dict[str, list[str]] = assignment_cfg.get("fallbacks", {})
    default_fallbacks: list[str] = fallback_map.get("default", ["rolling_mean"])

    # Group DFUs by their primary algorithm
    algo_to_skus: dict[str, list[str]] = (
        classification_df.groupby("assigned_algorithm")["sku_ck"]
        .apply(list)
        .to_dict()
    )

    all_parts: list[pd.DataFrame] = []
    tree_covered: set[str] = set()
    tf_label = tf["label"]

    # --- Tree model pre-pass (lgbm/catboost/xgboost) ---------------------------
    # Tree models train on the feature grid rather than raw sales, so they run
    # outside the cascade mechanism. DFUs covered here are excluded from the
    # regular cascade; any uncovered DFUs fall through to their cascade fallback.
    tree_groups = {algo: skus for algo, skus in algo_to_skus.items() if algo in _TREE_ALGOS}
    if tree_groups and grid is not None:
        tree_model_name_map = {
            "lgbm_cluster": "lgbm",
            "catboost_cluster": "catboost",
            "xgboost_cluster": "xgboost",
        }
        enabled_tree = {
            tree_model_name_map[algo]: cfg.get(algo, {})
            for algo in tree_groups
            if algo in tree_model_name_map
        }
        tree_skus = {sku for skus in tree_groups.values() for sku in skus}
        tree_clf = classification_df[classification_df["sku_ck"].isin(tree_skus)]
        try:
            tree_preds = run_tree_models(
                grid=grid,
                train_end=train_end,
                predict_months=predict_months,
                enabled_models=enabled_tree,
                classification_df=tree_clf,
            )
            if not tree_preds.empty:
                all_parts.append(tree_preds)
                tree_covered = set(tree_preds["sku_ck"].unique())
                logger.info("tf=%s tree models: %d DFUs covered", tf_label, len(tree_covered))
        except Exception:
            logger.exception("tf=%s tree model pre-pass failed; falling through to cascade", tf_label)

    # Remove tree-covered DFUs from cascade groups; uncovered ones use their fallback
    for algo in list(tree_groups):
        remaining = [s for s in algo_to_skus.get(algo, []) if s not in tree_covered]
        if remaining:
            # Route uncovered tree DFUs through the tree algo's fallback cascade
            fallbacks = fallback_map.get(algo, default_fallbacks)
            first_fallback = fallbacks[0] if fallbacks else "rolling_mean"
            algo_to_skus[first_fallback] = algo_to_skus.get(first_fallback, []) + remaining
            logger.info(
                "tf=%s %d uncovered tree DFUs rerouted to %s", tf_label, len(remaining), first_fallback,
            )
        del algo_to_skus[algo]

    # Separate groups into CPU-only (mstl, rolling_mean) and GPU-bound (nbeats, chronos).
    # CPU groups are launched in background threads so they overlap with GPU execution.
    # cpu_safe=True strips GPU algorithms from the CPU thread's fallback cascade to avoid
    # MPS/CUDA access from non-main threads (PyTorch MPS is not guaranteed thread-safe).
    cpu_groups = [(algo, skus) for algo, skus in algo_to_skus.items() if algo in _CPU_ALGOS]
    gpu_groups = [(algo, skus) for algo, skus in algo_to_skus.items() if algo not in _CPU_ALGOS]

    cascade_kwargs = dict(
        train_df=train_df,
        predict_months=predict_months,
        fallback_map=fallback_map,
        default_fallbacks=default_fallbacks,
        cfg=cfg,
        tf_label=tf_label,
    )

    with ThreadPoolExecutor(max_workers=max(1, len(cpu_groups))) as cpu_pool:
        # Submit CPU-bound groups to background threads
        cpu_futures = {
            cpu_pool.submit(
                _run_cascade_group, algo, skus, cpu_safe=True, **cascade_kwargs
            ): algo
            for algo, skus in cpu_groups
        }

        # GPU-bound groups run in the main thread (MPS is single-process / single-stream)
        for primary_algo, sku_list in gpu_groups:
            result = _run_cascade_group(primary_algo, sku_list, **cascade_kwargs)
            if not result.empty:
                all_parts.append(result)

        # Collect CPU results (GPU work above may already be done by now)
        for fut in as_completed(cpu_futures):
            algo = cpu_futures[fut]
            try:
                result = fut.result()
                if not result.empty:
                    all_parts.append(result)
            except Exception:
                logger.exception("CPU group '%s' failed in background thread", algo)

    if not all_parts:
        return pd.DataFrame(
            columns=["sku_ck", "startdate", FORECAST_QTY_COL, "algorithm_id",
                     "timeframe_label", "train_end"]
        )

    result = pd.concat(all_parts, ignore_index=True)
    # Bug fix: Sort deterministically before dedup so thread ordering doesn't
    # affect which prediction wins. Prefer non-naive algorithms over naive_zero.
    _algo_priority = result["algorithm_id"].map(
        lambda a: 1 if a == "naive_zero" else 0
    )
    result = (
        result.assign(_prio=_algo_priority)
        .sort_values(["sku_ck", "startdate", "_prio"])
        .drop(columns=["_prio"])
        .drop_duplicates(subset=["sku_ck", "startdate"], keep="first")
        .reset_index(drop=True)
    )
    result["timeframe_label"] = tf["label"]
    result["train_end"] = train_end

    logger.info(
        "Timeframe %s: %d predictions across %d DFUs",
        tf["label"], len(result), result["sku_ck"].nunique(),
    )
    return result


# ---------------------------------------------------------------------------
# Parallel timeframe worker (top-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------

def _timeframe_worker(
    tf: dict[str, Any],
    sales_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    dfu_attrs: pd.DataFrame,
    cfg: dict[str, Any],
    output_dir: Path,
    grid: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Run one timeframe: load checkpoint if available, else compute and save."""
    label = tf["label"]
    checkpoint = output_dir / f"tf_{label}_predictions.parquet"

    if checkpoint.exists():
        logger.info("Timeframe %s: resuming from checkpoint", label)
        return pd.read_parquet(checkpoint)

    logger.info(
        "Timeframe %s — train_end=%s  predict_start=%s",
        label,
        tf["train_end"].strftime("%Y-%m"),
        tf["predict_start"].strftime("%Y-%m"),
    )
    preds = run_timeframe(tf, sales_df, classification_df, cfg, grid=grid)

    if preds.empty:
        logger.warning("Timeframe %s produced no predictions; skipping", label)
        return None

    preds = add_lag_columns(preds, dfu_attrs)
    preds.to_parquet(checkpoint, index=False)
    logger.info("Timeframe %s: checkpoint saved (%d rows)", label, len(preds))
    return preds


# ---------------------------------------------------------------------------
# Lag columns
# ---------------------------------------------------------------------------

def add_lag_columns(
    predictions_df: pd.DataFrame,
    dfu_attrs: pd.DataFrame,
) -> pd.DataFrame:
    """Add fcstdate, lag, execution_lag, item_id, customer_group, loc, forecast_ck."""
    df = predictions_df.copy()
    df["startdate"] = pd.to_datetime(df["startdate"])
    df["train_end"] = pd.to_datetime(df["train_end"])

    # fcstdate = first month of the prediction window = train_end + 1 month
    df["fcstdate"] = (
        (df["train_end"] + pd.DateOffset(months=1))
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    # lag = months(startdate − fcstdate)  [0 = first predicted month]
    df["lag"] = (
        (df["startdate"].dt.year - df["fcstdate"].dt.year) * 12
        + (df["startdate"].dt.month - df["fcstdate"].dt.month)
    )

    # Keep only lags 0–MAX_LAG (DB constraint)
    df = df[(df["lag"] >= 0) & (df["lag"] <= _MAX_LAG)].copy()

    # Attach DFU metadata
    df = df.merge(
        dfu_attrs[["sku_ck", "item_id", "customer_group", "loc", "execution_lag"]],
        on="sku_ck",
        how="left",
    )
    df["execution_lag"] = df["execution_lag"].fillna(0).astype(int)

    # Build forecast_ck matching the existing loader pattern
    df["forecast_ck"] = (
        df["item_id"].astype(str) + "_"
        + df["customer_group"].astype(str) + "_"
        + df["loc"].astype(str) + "_"
        + df["fcstdate"].dt.strftime("%Y%m%d")
    )

    return df


# ---------------------------------------------------------------------------
# Accuracy reporting
# ---------------------------------------------------------------------------

def compute_lag_accuracy(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    classification_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute WAPE/accuracy at each lag (0–4) and at DFU execution lag."""
    seg_map = classification_df.set_index("sku_ck")["archetype"].to_dict()
    results: dict[str, Any] = {"by_lag": {}, "execution_lag": {}}

    merged = predictions_df.merge(
        actuals_df[["sku_ck", "startdate", "qty"]], on=["sku_ck", "startdate"], how="inner"
    )
    if merged.empty:
        logger.warning("No overlap between predictions and actuals for accuracy computation")
        return results

    merged["abs_err"] = (merged[FORECAST_QTY_COL] - merged["qty"]).abs()
    merged["archetype"] = merged["sku_ck"].map(seg_map).fillna("unknown")

    for lag in range(_MAX_LAG + 1):
        lag_df = merged[merged["lag"] == lag]
        if lag_df.empty:
            continue
        wape = float(lag_df["abs_err"].sum()) / max(abs(float(lag_df["qty"].sum())), 1.0) * 100
        per_seg = {}
        for seg, grp in lag_df.groupby("archetype"):
            seg_wape = float(grp["abs_err"].sum()) / max(abs(float(grp["qty"].sum())), 1.0) * 100
            per_seg[str(seg)] = round(100.0 - seg_wape, 2)
        results["by_lag"][lag] = {
            "accuracy_pct": round(100.0 - wape, 2),
            "wape": round(wape, 2),
            "n_dfus": int(lag_df["sku_ck"].nunique()),
            "n_dfu_months": len(lag_df),
            "per_segment": per_seg,
        }

    # Execution-lag cut: each DFU evaluated at its own assigned lag
    exec_df = merged[merged["lag"] == merged["execution_lag"]]
    if not exec_df.empty:
        wape = float(exec_df["abs_err"].sum()) / max(abs(float(exec_df["qty"].sum())), 1.0) * 100
        per_seg = {}
        for seg, grp in exec_df.groupby("archetype"):
            seg_wape = float(grp["abs_err"].sum()) / max(abs(float(grp["qty"].sum())), 1.0) * 100
            per_seg[str(seg)] = round(100.0 - seg_wape, 2)
        results["execution_lag"] = {
            "accuracy_pct": round(100.0 - wape, 2),
            "wape": round(wape, 2),
            "n_dfus": int(exec_df["sku_ck"].nunique()),
            "n_dfu_months": len(exec_df),
            "per_segment": per_seg,
        }

    return results


def _print_accuracy_report(accuracy: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("EXPERT SYSTEM BACKTEST — LAG ACCURACY REPORT")
    print("=" * 72)
    print(f"\n  {'Lag':<10} {'Accuracy':>10} {'WAPE':>8} {'DFUs':>8} {'DFU-months':>12}")
    print("  " + "-" * 52)
    for lag, stats in sorted(accuracy.get("by_lag", {}).items()):
        print(
            f"  Lag {lag:<6} {stats['accuracy_pct']:>9.2f}% {stats['wape']:>7.2f}%"
            f" {stats['n_dfus']:>8,} {stats['n_dfu_months']:>12,}"
        )
    exec_stats = accuracy.get("execution_lag")
    if exec_stats and exec_stats.get("n_dfus", 0) > 0:
        print("  " + "-" * 52)
        print(
            f"  {'ExecLag':<10} {exec_stats['accuracy_pct']:>9.2f}% {exec_stats['wape']:>7.2f}%"
            f" {exec_stats['n_dfus']:>8,} {exec_stats['n_dfu_months']:>12,}"
        )

    all_lags = sorted(accuracy.get("by_lag", {}).keys())
    segments = sorted({
        seg
        for s in accuracy["by_lag"].values()
        for seg in s.get("per_segment", {})
    })
    if segments and all_lags:
        col_w = 10
        print(f"\n  Per-segment accuracy by lag:")
        header = f"  {'Segment':<22}" + "".join(f"{'Lag ' + str(l):>{col_w}}" for l in all_lags)
        if exec_stats:
            header += f"{'ExecLag':>{col_w}}"
        print(header)
        for seg in segments:
            row = f"  {seg:<22}"
            for lag in all_lags:
                val = accuracy["by_lag"][lag].get("per_segment", {}).get(seg, float("nan"))
                row += f"{val:>{col_w - 1}.1f}%" if val == val else f"{'n/a':>{col_w}}"
            if exec_stats:
                val = exec_stats.get("per_segment", {}).get(seg, float("nan"))
                row += f"{val:>{col_w - 1}.1f}%" if val == val else f"{'n/a':>{col_w}}"
            print(row)

    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# DB loading
# ---------------------------------------------------------------------------

def load_archive(
    predictions_df: pd.DataFrame,
    model_id: str,
    tothist_map: dict[str, float],
    replace: bool,
    db: dict[str, Any],
) -> None:
    """Upsert all lag 0–4 predictions into backtest_lag_archive."""
    df = predictions_df.copy()
    df["model_id"] = model_id
    df["tothist_dmd"] = df["sku_ck"].map(tothist_map).fillna(0.0)
    df["timeframe"] = df.get("timeframe_label", pd.Series(dtype=str))
    df["fcstdate"] = df["fcstdate"].dt.strftime("%Y-%m-%d")
    df["startdate"] = df["startdate"].dt.strftime("%Y-%m-%d")

    archive = df[_ARCHIVE_COLS].copy()

    if archive.empty:
        logger.warning("Archive is empty — skipping backtest_lag_archive load")
        return

    logger.info("Loading %d rows into backtest_lag_archive (model_id=%s)...", len(archive), model_id)
    col_list = ", ".join(_ARCHIVE_COLS)
    placeholders = ", ".join(["%s"] * len(_ARCHIVE_COLS))
    update_set = "basefcst_pref = EXCLUDED.basefcst_pref, tothist_dmd = EXCLUDED.tothist_dmd, " \
                 "execution_lag = EXCLUDED.execution_lag, timeframe = EXCLUDED.timeframe"

    with psycopg.connect(**db) as conn:
        with conn.cursor() as cur:
            cur.execute("SET work_mem = '256MB'")
            if replace:
                cur.execute("DELETE FROM backtest_lag_archive WHERE model_id = %s", (model_id,))
                logger.info("Deleted existing rows for model_id=%s", model_id)

            batch_size = 50_000
            n_loaded = 0
            for start in range(0, len(archive), batch_size):
                batch = archive.iloc[start : start + batch_size]
                rows = [tuple(r) for r in batch.itertuples(index=False)]
                cur.executemany(
                    f"INSERT INTO backtest_lag_archive ({col_list}) VALUES ({placeholders}) "
                    f"ON CONFLICT (forecast_ck, model_id, lag) DO UPDATE SET {update_set}",
                    rows,
                )
                n_loaded += len(batch)
                logger.info("  archive: %d / %d rows", n_loaded, len(archive))
        conn.commit()

    logger.info("backtest_lag_archive load complete: %d rows", len(archive))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_backtest(
    cfg: dict[str, Any],
    replace: bool,
    skip_load: bool,
    workers: int | None = None,
) -> None:
    exp_cfg = cfg.get("experiment", {})
    if workers is not None:
        exp_cfg = {**exp_cfg, "parallel_timeframes": workers}
        cfg = {**cfg, "experiment": exp_cfg}
    n_timeframes = exp_cfg.get("n_timeframes", 10)
    model_id = exp_cfg.get("algorithm_id", _DEFAULT_MODEL_ID)
    output_dir = Path(exp_cfg.get("output_dir", "data/expert_system_backtest"))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())

    logger.info("ExpSys backtest starting. run_id=%s", run_id)

    loc_filter: str | None = exp_cfg.get("loc_filter")

    # 1. Full-population data
    sales_df, dfu_attrs, item_attrs = load_full_population(loc_filter=loc_filter)
    tothist_map: dict[str, float] = sales_df.groupby("sku_ck")["qty"].sum().to_dict()

    # 2. Classify + assign
    logger.info("Classifying DFUs into archetypes...")
    classification_df = classify_and_assign(sales_df, cfg)

    # 2b. Build feature grid if tree models are assigned
    assignment_vals = {v for v in cfg.get("assignment", {}).values() if isinstance(v, str)}
    needs_grid = bool(_TREE_ALGOS & assignment_vals)
    grid: pd.DataFrame | None = None
    if needs_grid:
        all_months = sorted(sales_df["startdate"].unique())
        logger.info("Building feature grid for tree models (%d DFUs × %d months)...",
                    sales_df["sku_ck"].nunique(), len(all_months))
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, all_months)

    # 3. Generate timeframes
    earliest = sales_df["startdate"].min()
    latest = sales_df["startdate"].max()
    timeframes = generate_timeframes(earliest, latest, n=n_timeframes)
    logger.info(
        "Generated %d timeframes: train_end range %s → %s",
        n_timeframes,
        timeframes[0]["train_end"].strftime("%Y-%m"),
        timeframes[-1]["train_end"].strftime("%Y-%m"),
    )

    # 4. Per-timeframe backtest with checkpointing (serial or parallel)
    all_predictions: list[pd.DataFrame] = []
    parallel = exp_cfg.get("parallel_timeframes", 1)

    if parallel > 1:
        # Scale mstl's internal thread count to avoid CPU over-subscription
        base_n = exp_cfg.get("n_workers", 8)
        scaled_n = max(1, base_n // parallel)
        cfg_run: dict[str, Any] = {**cfg, "experiment": {**exp_cfg, "n_workers": scaled_n}}
        logger.info(
            "Parallel timeframes=%d — mstl n_workers scaled %d → %d",
            parallel, base_n, scaled_n,
        )
        # Warn when GPU models share MPS across processes
        assignment_vals = {v for v in cfg.get("assignment", {}).values() if isinstance(v, str)}
        gpu_algos = {"nbeats", "chronos"} & assignment_vals
        if gpu_algos:
            logger.warning(
                "GPU models (%s) will run in %d parallel processes — "
                "ensure sufficient MPS/VRAM memory",
                ", ".join(sorted(gpu_algos)), parallel,
            )
    else:
        cfg_run = cfg

    if parallel > 1:
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(
                    _timeframe_worker,
                    tf, sales_df, classification_df, dfu_attrs, cfg_run, output_dir, grid,
                ): tf["label"]
                for tf in timeframes
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        all_predictions.append(result)
                except Exception:
                    logger.exception("Timeframe %s failed", label)
    else:
        for tf in timeframes:
            result = _timeframe_worker(
                tf, sales_df, classification_df, dfu_attrs, cfg_run, output_dir, grid,
            )
            if result is not None:
                all_predictions.append(result)

    if not all_predictions:
        logger.error("No predictions generated across any timeframe — aborting")
        return

    all_preds = pd.concat(all_predictions, ignore_index=True)
    logger.info(
        "All timeframes complete: %d rows, %d DFUs, %d timeframes",
        len(all_preds),
        all_preds["sku_ck"].nunique(),
        all_preds["timeframe_label"].nunique() if "timeframe_label" in all_preds.columns else 0,
    )

    # 5. Accuracy
    actuals_df = sales_df[["sku_ck", "startdate", "qty"]].copy()
    actuals_df["qty"] = actuals_df["qty"].astype(float)
    logger.info("Computing lag accuracy...")
    accuracy = compute_lag_accuracy(all_preds, actuals_df, classification_df)
    _print_accuracy_report(accuracy)

    report_path = output_dir / "accuracy_report.json"
    report_path.write_text(json.dumps(accuracy, indent=2, default=str))
    logger.info("Accuracy report saved to %s", report_path)

    if skip_load:
        logger.info("--skip-load: skipping DB loading")
        return

    db = get_db_params()

    # 6. Load backtest_lag_archive (all lags 0–4)
    load_archive(all_preds, model_id, tothist_map, replace=replace, db=db)

    logger.info("ExpSys backtest complete. run_id=%s", run_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Expert System Backtest (ExpSys)")
    parser.add_argument(
        "--config", type=Path, default=Path("config/forecasting/expert_system_backtest.yaml"),
        help="Path to config YAML (default: config/forecasting/expert_system_backtest.yaml)",
    )
    parser.add_argument(
        "--replace", action="store_true",
        help="Delete existing ExpSys rows in backtest_lag_archive before loading",
    )
    parser.add_argument(
        "--skip-load", action="store_true",
        help="Compute and print accuracy but skip all DB loading",
    )
    parser.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help="Number of parallel timeframe workers (overrides config parallel_timeframes; 1=serial)",
    )
    parser.add_argument(
        "--loc", type=str, default=None, metavar="LOC",
        help="Restrict backtest to all DFUs at this location (e.g. 1401-BULK)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.config.exists():
        import yaml
        cfg = yaml.safe_load(args.config.read_text())
    else:
        cfg = load_config("expert_system_backtest.yaml")

    if args.loc:
        cfg.setdefault("experiment", {})["loc_filter"] = args.loc

    run_backtest(cfg, replace=args.replace, skip_load=args.skip_load, workers=args.workers)


if __name__ == "__main__":
    import multiprocessing
    # Must be set before any torch/NeuralForecast import to prevent
    # fork()-after-torch-import SIGSEGV on macOS (same pattern as adv expert panel)
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
