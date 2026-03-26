"""Shared backtest orchestration framework.

Provides common logic for all backtest scripts:
- Timeframe generation
- Data loading from Postgres
- Execution-lag assignment and forecast_ck construction
- All-lag expansion for archive tables
- Output saving (CSV + metadata JSON)
- Accuracy computation
- MLflow logging
- Per-cluster adaptive hyperparameter profiles

Model-specific scripts implement only the training/prediction functions.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import psycopg

from common.constants import (
    ARCHIVE_COLS,
    CAT_FEATURES,
    LAG_RANGE,
    MAX_ARCHIVE_LAG,
    MIN_TRAINING_MONTHS,
    NUMERIC_ITEM_FEATURES,
    NUMERIC_SKU_FEATURES,
    OUTPUT_COLS,
)
from common.db import get_db_params
from common.metrics import compute_accuracy_metrics
from common.mlflow_utils import log_backtest_run
from common.planning_date import get_planning_date
from common.utils import _ts, load_config

logger = logging.getLogger(__name__)

# ── Per-cluster adaptive hyperparameter profiles ─────────────────────────────

# Profile priority order — first match wins
_PROFILE_PRIORITY = [
    "sparse_intermittent",
    "low_volume_volatile",
    "high_volume_stable",
    "seasonal_dominant",
    "default",
]


def compute_cluster_demand_stats(
    train_df: pd.DataFrame,
    cluster_id: Any,
) -> dict[str, float]:
    """Compute demand characteristics for a single cluster from training data.

    Returns a dict with:
    - mean_demand: mean of non-zero qty values
    - cv_demand: coefficient of variation (std / mean) of qty
    - zero_demand_pct: fraction of rows with qty == 0
    - seasonal_amplitude: std of monthly means / overall mean (proxy for seasonality)
    """
    cluster_data = train_df[train_df["ml_cluster"] == cluster_id]
    qty = cluster_data["qty"] if "qty" in cluster_data.columns else pd.Series(dtype=float)

    if len(qty) == 0:
        return {
            "mean_demand": 0.0,
            "cv_demand": 0.0,
            "zero_demand_pct": 1.0,
            "seasonal_amplitude": 0.0,
        }

    nonzero_qty = qty[qty > 0]
    mean_demand = float(nonzero_qty.mean()) if len(nonzero_qty) > 0 else 0.0

    overall_mean = float(qty.mean())
    overall_std = float(qty.std()) if len(qty) > 1 else 0.0
    cv_demand = (overall_std / overall_mean) if overall_mean > 0 else 0.0

    zero_demand_pct = float((qty == 0).sum()) / len(qty)

    # Seasonal amplitude: variability of monthly means relative to overall mean
    seasonal_amplitude = 0.0
    if "startdate" in cluster_data.columns and overall_mean > 0:
        monthly_means = cluster_data.groupby(
            cluster_data["startdate"].dt.month
        )["qty"].mean()
        if len(monthly_means) > 1:
            seasonal_amplitude = float(monthly_means.std()) / overall_mean

    return {
        "mean_demand": mean_demand,
        "cv_demand": cv_demand,
        "zero_demand_pct": zero_demand_pct,
        "seasonal_amplitude": seasonal_amplitude,
    }


def _matches_profile(
    stats: dict[str, float],
    criteria: dict[str, float],
) -> bool:
    """Check if cluster stats satisfy all match criteria for a profile."""
    if not criteria:
        return True  # empty criteria = always matches (used by default profile)

    for key, threshold in criteria.items():
        if key.endswith("_min"):
            stat_key = key[:-4]  # strip _min suffix
            if stats.get(stat_key, 0.0) < threshold:
                return False
        elif key.endswith("_max"):
            stat_key = key[:-4]  # strip _max suffix
            if stats.get(stat_key, 0.0) > threshold:
                return False

    return True


def resolve_cluster_params(
    cluster_id: Any,
    cluster_stats: dict[str, float],
    base_params: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Resolve hyperparameters for a cluster based on its demand characteristics.

    Loads cluster_tuning_profiles.yaml, matches against profiles in priority
    order, and returns base_params merged with the matching profile's overrides.

    Args:
        cluster_id: Cluster label (for logging).
        cluster_stats: Output from ``compute_cluster_demand_stats()``.
        base_params: Default model hyperparameters.

    Returns:
        Tuple of (resolved_params, matched_profile_name).
        Falls back to (base_params, "none") if profiles are disabled or no match.
    """
    cfg = load_config("cluster_tuning_profiles.yaml")

    if not cfg.get("enabled", False):
        return base_params, "none"

    profiles = cfg.get("cluster_profiles", {})
    if not profiles:
        return base_params, "none"

    for profile_name in _PROFILE_PRIORITY:
        profile = profiles.get(profile_name)
        if profile is None:
            continue

        criteria = profile.get("match_criteria", {})
        overrides = profile.get("overrides", {})

        if _matches_profile(cluster_stats, criteria):
            if not overrides:
                # default profile or profile with no overrides
                logger.debug(
                    "Cluster '%s' matched profile '%s' (no overrides)",
                    cluster_id, profile_name,
                )
                return base_params, profile_name

            resolved = {**base_params, **overrides}
            logger.info(
                "Cluster '%s' matched profile '%s': overrides=%s",
                cluster_id, profile_name, overrides,
            )
            return resolved, profile_name

    return base_params, "none"


# ── Timeframe generation ─────────────────────────────────────────────────────


def generate_timeframes(
    earliest: pd.Timestamp,
    latest: pd.Timestamp,
    n: int = 10,
) -> list[dict]:
    """Generate N expanding-window timeframes.

    For timeframe i (A=0 .. J=9):
      train_end   = latest - (N - i) months
      predict     = [train_end + 1 month, latest]
    """
    timeframes = []
    for i in range(n):
        train_end = latest - pd.DateOffset(months=(n - i))
        train_end = train_end.normalize()  # midnight
        predict_start = train_end + pd.DateOffset(months=1)
        label = chr(ord("A") + i)
        timeframes.append({
            "label": label,
            "index": i,
            "train_start": earliest,
            "train_end": train_end,
            "predict_start": predict_start,
            "predict_end": latest,
        })
    return timeframes


# ── Data loading ─────────────────────────────────────────────────────────────


def load_backtest_data(
    db: dict[str, Any],
    include_item_attrs: bool = True,
    algo_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load sales, DFU attributes, and item attributes from Postgres.

    Sales are capped at the planning date (first of month) to ensure
    no future data leaks into backtesting.

    Returns (sales_df, dfu_attrs, item_attrs).
    """
    t1 = time.time()
    planning_cutoff = get_planning_date().replace(day=1)
    with psycopg.connect(**db) as conn:
        # Prefer uncorrected sales for accuracy (medallion dual-track)
        sales_table = "fact_sales_monthly"
        try:
            with conn.cursor() as _cur:
                _cur.execute("SELECT count(*) AS n FROM fact_sales_monthly_original LIMIT 1")
                _cnt_cols = [d[0] for d in _cur.description]
                _cnt = pd.DataFrame(_cur.fetchall(), columns=_cnt_cols)
            if _cnt.iloc[0]["n"] > 0:
                sales_table = "fact_sales_monthly_original"
        except Exception:
            pass  # Table doesn't exist or is empty, use default

        with conn.cursor() as _cur:
            _cur.execute(f"""
                SELECT d.sku_ck, s.item_id, s.customer_group, s.loc, s.startdate, s.qty
                FROM {sales_table} s
                INNER JOIN dim_sku d
                    ON d.item_id = s.item_id AND d.customer_group = s.customer_group AND d.loc = s.loc
                WHERE s.qty IS NOT NULL
                  AND s.startdate <= %s
                ORDER BY d.sku_ck, s.startdate
            """, (planning_cutoff,))
            _cols = [d[0] for d in _cur.description]
            sales_df = pd.DataFrame(_cur.fetchall(), columns=_cols)

        with conn.cursor() as _cur:
            _cur.execute("""
                SELECT sku_ck, item_id, customer_group, loc,
                       execution_lag, total_lt, ml_cluster,
                       brand, region, abc_vol
                FROM dim_sku
            """)
            _cols = [d[0] for d in _cur.description]
            dfu_attrs = pd.DataFrame(_cur.fetchall(), columns=_cols)

        if include_item_attrs:
            with conn.cursor() as _cur:
                _cur.execute("""
                    SELECT DISTINCT i.item_id AS item_id,
                           i.case_weight, i.item_proof, i.bpc
                    FROM dim_item i
                    INNER JOIN dim_sku d ON i.item_id = d.item_id
                """)
                _cols = [d[0] for d in _cur.description]
                item_attrs = pd.DataFrame(_cur.fetchall(), columns=_cols)
        else:
            item_attrs = pd.DataFrame()

    sales_df["startdate"] = pd.to_datetime(sales_df["startdate"])
    sales_df["qty"] = pd.to_numeric(sales_df["qty"], errors="coerce").fillna(0)
    for col in NUMERIC_SKU_FEATURES:
        if col in dfu_attrs.columns:
            dfu_attrs[col] = pd.to_numeric(dfu_attrs[col], errors="coerce").fillna(0)
    for col in NUMERIC_ITEM_FEATURES:
        if col in item_attrs.columns:
            item_attrs[col] = pd.to_numeric(item_attrs[col], errors="coerce").fillna(0)

    # Only keep DFUs that have sales
    dfus_with_sales = set(sales_df["sku_ck"].unique())
    dfu_attrs = dfu_attrs[dfu_attrs["sku_ck"].isin(dfus_with_sales)]

    # Apply cluster override if provided (for cluster experiments)
    cluster_override_path = algo_config.get("cluster_override_path") if algo_config else None
    if cluster_override_path:
        override_df = pd.read_csv(cluster_override_path, usecols=["sku_ck", "cluster_label"])
        override_map = dict(zip(override_df["sku_ck"], override_df["cluster_label"]))
        original_clusters = dfu_attrs["ml_cluster"].copy()
        dfu_attrs["ml_cluster"] = dfu_attrs["sku_ck"].map(override_map).fillna(dfu_attrs["ml_cluster"])
        n_remapped = int((dfu_attrs["ml_cluster"] != original_clusters).sum())
        print(f"  [{_ts()}] Cluster override applied: {len(override_map):,} entries from {cluster_override_path}, "
              f"{n_remapped:,} DFUs remapped")

    print(f"  [{_ts()}] Sales: {len(sales_df):,} rows, {len(dfus_with_sales):,} DFUs ({time.time() - t1:.1f}s)")
    print(f"  [{_ts()}] DFU attrs: {len(dfu_attrs):,}, Item attrs: {len(item_attrs):,}")

    return sales_df, dfu_attrs, item_attrs


# ── Execution-lag assignment ─────────────────────────────────────────────────


def assign_execution_lag(
    pred_df: pd.DataFrame,
    execution_lag_map: dict[str, int],
) -> pd.DataFrame:
    """Assign each prediction its DFU's execution lag and compute fcstdate.

    Only one row per prediction — at the DFU's execution lag.
    fcstdate = startdate - execution_lag months.
    """
    t0 = time.time()
    result = pred_df.copy()

    # Execution lag from DFU dimension
    result["execution_lag"] = result["sku_ck"].map(execution_lag_map).fillna(0).astype(int)
    result["lag"] = result["execution_lag"]
    # Group by unique lag values to minimize DateOffset calls
    for lag_val in result["lag"].unique():
        mask = result["lag"] == lag_val
        result.loc[mask, "fcstdate"] = result.loc[mask, "startdate"] - pd.DateOffset(months=int(lag_val))

    # Build forecast_ck (vectorized string concat via str.cat)
    result["forecast_ck"] = (
        result["item_id"].astype(str).str.cat([
            result["customer_group"].astype(str),
            result["loc"].astype(str),
            result["fcstdate"].dt.strftime("%Y-%m-%d"),
            result["startdate"].dt.strftime("%Y-%m-%d"),
        ], sep="_")
    )

    print(f"  [{_ts()}] Execution-lag assignment done ({time.time() - t0:.1f}s)")
    return result


# ── Natural lag assignment (archive) ─────────────────────────────────────────


def assign_natural_lags(
    pred_df: pd.DataFrame,
    timeframes: list[dict],
    max_lag: int,
    execution_lag_map: dict[str, int],
) -> pd.DataFrame:
    """Assign each prediction its natural forecast lag based on timeframe.

    The natural lag is the number of months between the timeframe's first
    predict month and the demand month (startdate).  This represents the
    true forecast horizon — how far ahead the model was predicting.

        lag = months_between(startdate, train_end) - 1

    For example, with 10 timeframes (A-J) predicting demand month Feb 2026:
      - Timeframe J (train_end = Jan 2026) → lag 0  (1-month-ahead)
      - Timeframe I (train_end = Dec 2025) → lag 1  (2-month-ahead)
      - Timeframe H (train_end = Nov 2025) → lag 2  (3-month-ahead)
      - Timeframe G (train_end = Oct 2025) → lag 3  (4-month-ahead)
      - Timeframe F (train_end = Sep 2025) → lag 4  (5-month-ahead)

    Each lag uses a genuinely different prediction because the model was
    trained on different data cutoffs.  Only keeps 0 <= lag <= max_lag.
    """
    t0 = time.time()

    # Map timeframe_idx → train_end
    tf_map = {tf["index"]: pd.Timestamp(tf["train_end"]) for tf in timeframes}

    df = pred_df.copy()
    df["_train_end"] = df["timeframe_idx"].map(tf_map)

    # Compute natural lag: months between train_end and startdate, minus 1
    # (predict_start = train_end + 1 month, so lag 0 = first predict month)
    df["lag"] = (
        (df["startdate"].dt.year * 12 + df["startdate"].dt.month)
        - (df["_train_end"].dt.year * 12 + df["_train_end"].dt.month)
        - 1
    )

    # Filter to valid lag range (0 .. max_lag)
    df = df[(df["lag"] >= 0) & (df["lag"] <= max_lag)].copy()

    # Assign execution_lag from DFU dimension
    df["execution_lag"] = df["sku_ck"].map(execution_lag_map).fillna(0).astype(int)

    # Compute fcstdate = startdate - lag months (vectorized per lag value)
    for lag_val in range(max_lag + 1):
        mask = df["lag"] == lag_val
        if mask.any():
            df.loc[mask, "fcstdate"] = df.loc[mask, "startdate"] - pd.DateOffset(months=lag_val)

    # Build forecast_ck (vectorized string concat via str.cat)
    df["forecast_ck"] = (
        df["item_id"].astype(str).str.cat([
            df["customer_group"].astype(str),
            df["loc"].astype(str),
            df["fcstdate"].dt.strftime("%Y-%m-%d"),
            df["startdate"].dt.strftime("%Y-%m-%d"),
        ], sep="_")
    )

    # Drop helper column
    df = df.drop(columns=["_train_end"])

    print(f"  [{_ts()}] Natural lag assignment (0-{max_lag}) done: {len(df):,} rows ({time.time() - t0:.1f}s)")
    return df


# ── Post-processing: combine, dedup, attach actuals ─────────────────────────


def postprocess_predictions(
    all_predictions: list[pd.DataFrame],
    sales_df: pd.DataFrame,
    exec_lag_map: dict[str, int],
    timeframes: list[dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combine timeframe predictions, assign execution lag, dedup, attach actuals.

    Args:
        all_predictions: list of DataFrames from each timeframe (carry timeframe_idx)
        sales_df: sales data for attaching actuals
        exec_lag_map: sku_ck → execution_lag mapping from dim_sku
        timeframes: list of timeframe dicts (with train_end).  When provided,
            the archive uses natural lags computed from the timeframe's training
            cutoff — each lag gets a genuinely different prediction.

    Returns (output_df, archive_df, combined_raw).
    """
    combined = pd.concat(all_predictions, ignore_index=True)
    print(f"  [{_ts()}] Total raw predictions: {len(combined):,}")

    # Ensure startdate is datetime
    combined["startdate"] = pd.to_datetime(combined["startdate"])

    # Assign execution lag and compute fcstdate (one row per prediction)
    print(f"  [{_ts()}] Assigning execution lag per DFU...")
    expanded = assign_execution_lag(combined, exec_lag_map)
    print(f"  [{_ts()}] Rows after execution-lag assignment: {len(expanded):,}")

    # Deduplicate: for same (forecast_ck, model_id), keep latest timeframe
    expanded = expanded.sort_values("timeframe_idx")
    expanded = expanded.drop_duplicates(subset=["forecast_ck", "model_id"], keep="last")
    print(f"  [{_ts()}] After dedup: {len(expanded):,}")

    # Attach actuals via merge (vectorized — not row-by-row apply)
    print(f"  [{_ts()}] Attaching actuals...")
    t1 = time.time()
    actuals = sales_df.drop_duplicates(subset=["sku_ck", "startdate"])[["sku_ck", "startdate", "qty"]].rename(
        columns={"qty": "tothist_dmd"}
    )
    expanded = expanded.merge(actuals, on=["sku_ck", "startdate"], how="left")
    print(f"  [{_ts()}] Actuals attached ({time.time() - t1:.1f}s)")

    # ── All-lags archive ──────────────────────────────────────────────────
    print(f"  [{_ts()}] Generating all-lags archive (lag 0-{MAX_ARCHIVE_LAG})...")

    if timeframes is not None:
        # Natural lags: each prediction gets its true forecast horizon based
        # on the gap between the timeframe's training cutoff and the demand
        # month.  Different lags = different predictions from different
        # timeframes = genuinely different accuracy per horizon.
        archive_expanded = assign_natural_lags(
            combined, timeframes, MAX_ARCHIVE_LAG, exec_lag_map,
        )
    else:
        # Legacy fallback (no timeframe metadata available) — duplicates the
        # same prediction across all lags.  This path should only be hit if
        # postprocess_predictions is called without timeframes.
        logger.warning(
            "postprocess_predictions called without timeframes — "
            "archive will have identical predictions across all lags"
        )
        archive_expanded = _expand_to_all_lags_legacy(
            combined, MAX_ARCHIVE_LAG, exec_lag_map,
        )

    # Deduplicate: each (DFU, startdate, lag) should map to exactly one
    # timeframe with natural lags, but keep dedup as safety net.
    archive_expanded = archive_expanded.sort_values("timeframe_idx")
    archive_expanded = archive_expanded.drop_duplicates(
        subset=["forecast_ck", "model_id", "lag"], keep="last"
    )
    print(f"  [{_ts()}] Archive after dedup: {len(archive_expanded):,}")

    # Attach actuals
    archive_expanded = archive_expanded.merge(actuals, on=["sku_ck", "startdate"], how="left")

    return expanded, archive_expanded, combined


def _expand_to_all_lags_legacy(
    pred_df: pd.DataFrame,
    max_lag: int,
    execution_lag_map: dict[str, int],
) -> pd.DataFrame:
    """Legacy: duplicate each prediction to lag 0..max_lag (same basefcst_pref).

    Kept only as a fallback when timeframe metadata is unavailable.
    """
    t0 = time.time()
    dfs = []
    for lag in range(max_lag + 1):
        df = pred_df.copy()
        df["lag"] = lag
        df["fcstdate"] = df["startdate"] - pd.DateOffset(months=lag)
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)
    result["execution_lag"] = result["sku_ck"].map(execution_lag_map).fillna(0).astype(int)

    result["forecast_ck"] = (
        result["item_id"].astype(str).str.cat([
            result["customer_group"].astype(str),
            result["loc"].astype(str),
            result["fcstdate"].dt.strftime("%Y-%m-%d"),
            result["startdate"].dt.strftime("%Y-%m-%d"),
        ], sep="_")
    )

    print(f"  [{_ts()}] Legacy all-lag expansion (0-{max_lag}) done: {len(result):,} rows ({time.time() - t0:.1f}s)")
    return result


# ── Output saving ────────────────────────────────────────────────────────────


def save_backtest_output(
    output_df: pd.DataFrame,
    archive_df: pd.DataFrame,
    output_dir: Path,
    model_id: str,
    cluster_strategy: str,
    n_timeframes: int,
    model_params: dict[str, Any],
    model_params_key: str,
    timeframes: list[dict],
    earliest_month: pd.Timestamp,
    latest_month: pd.Timestamp,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path, Path, dict]:
    """Save predictions CSV, archive CSV, and metadata JSON.

    Writes into a model-scoped subdirectory: output_dir / model_id /
    This prevents multiple backtest runs from overwriting each other (PL-001).

    Returns (output_path, archive_path, meta_path, metadata_dict).
    """
    # Model-scoped subdirectory — each model_id gets its own folder
    model_dir = output_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # Select and order columns for fact_external_forecast_monthly
    out = output_df[OUTPUT_COLS].copy()
    out["fcstdate"] = out["fcstdate"].dt.strftime("%Y-%m-%d")
    out["startdate"] = out["startdate"].dt.strftime("%Y-%m-%d")

    output_path = model_dir / "backtest_predictions.csv"
    out.to_csv(output_path, index=False)
    print(f"  [{_ts()}] Saved {len(out):,} predictions to {output_path}")

    # Archive CSV
    arch = archive_df[ARCHIVE_COLS].copy()
    arch["fcstdate"] = arch["fcstdate"].dt.strftime("%Y-%m-%d")
    arch["startdate"] = arch["startdate"].dt.strftime("%Y-%m-%d")

    archive_path = model_dir / "backtest_predictions_all_lags.csv"
    arch.to_csv(archive_path, index=False)
    print(f"  [{_ts()}] Saved {len(arch):,} archive rows to {archive_path}")

    # Build metadata
    metadata = {
        "model_id": model_id,
        "cluster_strategy": cluster_strategy,
        "n_timeframes": n_timeframes,
        model_params_key: {k: v for k, v in model_params.items()},
        **(extra_metadata or {}),
        "n_predictions": len(out),
        "n_dfus": int(output_df["item_id"].nunique()),
        "date_range": {
            "earliest": str(earliest_month.date()),
            "latest": str(latest_month.date()),
        },
        "timeframes": [
            {
                "label": tf["label"],
                "train_end": str(tf["train_end"].date()),
                "predict_start": str(tf["predict_start"].date()),
                "predict_end": str(tf["predict_end"].date()),
            }
            for tf in timeframes
        ],
    }

    # Compute accuracy summary
    acc = compute_accuracy_metrics(
        pd.to_numeric(out["basefcst_pref"], errors="coerce"),
        pd.to_numeric(out["tothist_dmd"], errors="coerce"),
    )
    if acc["wape"] is not None:
        metadata["accuracy_at_execution_lag"] = acc
        print(f"\n  Accuracy at execution lag ({acc['n_rows']:,} rows):")
        print(f"    WAPE: {acc['wape']:.2f}%")
        print(f"    Bias: {acc['bias']:.4f}")
        print(f"    Accuracy: {acc['accuracy_pct']:.2f}%")

    meta_path = model_dir / "backtest_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  [{_ts()}] Saved metadata to {meta_path}")

    return output_path, archive_path, meta_path, metadata


# ── Feature importance ───────────────────────────────────────────────────────


def save_feature_importance(
    model: Any,
    feature_cols: list[str],
    output_dir: Path,
    importance_attr: str = "feature_importances_",
) -> Path | None:
    """Save feature importance CSV from a trained model. Returns path or None."""
    try:
        if hasattr(model, importance_attr):
            importances = getattr(model, importance_attr)
        elif hasattr(model, "get_feature_importance"):
            importances = model.get_feature_importance()
        else:
            return None

        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        imp_path = output_dir / "feature_importance.csv"
        importance.to_csv(imp_path, index=False)
        print(f"  [{_ts()}] Saved feature importance to {imp_path}")
        return imp_path
    except Exception:
        return None


# ── Tree-model backtest runner ───────────────────────────────────────────────

# Type alias for train-and-predict functions
TrainFn = Callable[
    [pd.DataFrame, pd.DataFrame, list[str], list[str], dict],
    tuple[pd.DataFrame, Any],
]

_PREDICT_META_COLS = ["sku_ck", "item_id", "customer_group", "loc", "startdate"]


def _fill_predict_nans(
    predict_data: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    """Fill NaN values in numeric feature columns of predict_data with 0."""
    for col in feature_cols:
        if col in predict_data.columns and col not in cat_cols:
            predict_data[col] = predict_data[col].fillna(0)
    return predict_data


def _predict_single_month(
    models: dict,
    predict_data: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Route a single-month batch through per-cluster models for recursive inference.

    Used by recursive multi-step inference (Feature 43). Each DFU row is
    routed to its cluster's model; DFUs with no matching cluster are skipped.

    Args:
        models: ``{cluster_label: model}`` dict from ``train_and_predict_per_cluster``.
        predict_data: Feature matrix for one month, all DFUs (must have ``ml_cluster``).
        feature_cols: Ordered list of feature columns passed to each model.
    """
    parts = []
    for cluster, group in predict_data.groupby("ml_cluster", observed=True):
        m = models.get(cluster)
        if m is None:
            continue
        # Models were trained with all feature_cols (including ml_cluster as a
        # categorical feature constant within each cluster partition). Pass the
        # same feature set to maintain feature alignment.
        preds = np.maximum(m.predict(group[feature_cols]), 0)
        r = group[_PREDICT_META_COLS].copy()
        r["basefcst_pref"] = preds
        parts.append(r)
    if not parts:
        return pd.DataFrame(columns=_PREDICT_META_COLS + ["basefcst_pref"])
    return pd.concat(parts, ignore_index=True)


def run_tree_backtest(
    *,
    model_id: str,
    n_timeframes: int,
    output_dir: Path,
    model_params: dict[str, Any],
    model_params_key: str,
    model_type_tag: str,
    train_fn_per_cluster: TrainFn,
    extra_metadata: dict[str, Any] | None = None,
    cat_dtype: str = "category",
    min_training_months: int = MIN_TRAINING_MONTHS,
    inline_tuner_fn: Callable[[Any, list[str], list[str], Any], dict[str, Any]] | None = None,
    feature_selector_fn: Callable[
        [Any, pd.DataFrame, list[str], list[str], int, pd.Timestamp],
        tuple[list[str], pd.DataFrame],
    ] | None = None,
    recursive: bool = False,
    model_persistence_fn: Callable[[Any, list[str], str], None] | None = None,
    algo_config: dict[str, Any] | None = None,
) -> None:
    """Run a complete tree-based per-cluster backtest (LGBM, CatBoost, XGBoost).

    All algorithms use per-cluster strategy. Options (recursive, SHAP, tuning)
    are passed via closures rather than CLI flags; see algorithm_config.yaml.
    """
    from common.feature_engineering import (
        build_feature_matrix,
        get_feature_columns,
        mask_future_sales,
        update_grid_incremental,
        update_grid_with_predictions,
    )

    cluster_strategy = "per_cluster"
    t_start = time.time()
    db = get_db_params()

    print(f"[{_ts()}] Backtest: strategy={cluster_strategy}, model_id={model_id}, "
          f"n_timeframes={n_timeframes}, recursive={recursive}")

    # ── Step 1: Load data ────────────────────────────────────────────────────
    print(f"\n[{_ts()}] Step 1: Loading data from Postgres...")
    sales_df, dfu_attrs, item_attrs = load_backtest_data(db, algo_config=algo_config)

    # Execution lag lookup
    exec_lag_map = dfu_attrs.set_index("sku_ck")["execution_lag"].fillna(0).astype(int).to_dict()

    # ── Step 2: Generate timeframes ──────────────────────────────────────────
    planning_dt = pd.Timestamp(get_planning_date())
    # Cap to first-of-planning-month so we only use complete months
    planning_cutoff = planning_dt.normalize().replace(day=1)
    latest_month = min(sales_df["startdate"].max(), planning_cutoff)
    earliest_month = sales_df["startdate"].min()
    # Filter out any sales beyond the planning date
    sales_df = sales_df[sales_df["startdate"] <= latest_month].copy()
    print(f"  [{_ts()}] Date range: {earliest_month.date()} → {latest_month.date()} "
          f"(planning date: {planning_dt.date()})")

    timeframes = generate_timeframes(earliest_month, latest_month, n_timeframes)
    print(f"\n[{_ts()}] Step 2: Generated {len(timeframes)} timeframes:")
    for tf in timeframes:
        print(f"  {tf['label']}: train [{tf['train_start'].date()} → {tf['train_end'].date()}], "
              f"predict [{tf['predict_start'].date()} → {tf['predict_end'].date()}]")

    all_months = sorted(sales_df["startdate"].unique())

    # ── Step 3: Build feature matrix ONCE ────────────────────────────────────
    print(f"\n[{_ts()}] Step 3: Building feature matrix (one-time)...")
    full_grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, all_months, cat_dtype=cat_dtype)
    feature_cols = get_feature_columns(full_grid)
    cat_cols = [c for c in CAT_FEATURES if c in feature_cols and c in full_grid.columns]
    print(f"  [{_ts()}] Features: {len(feature_cols)} columns, cat: {cat_cols}")

    # ── Step 4: Train & predict per timeframe ────────────────────────────────
    print(f"\n[{_ts()}] Step 4: Running {len(timeframes)} timeframe backtests...")
    all_predictions = []
    shap_timeframe_reports: list[pd.DataFrame] = []

    for ti, tf in enumerate(timeframes):
        label = tf["label"]
        train_end = tf["train_end"]
        predict_start = tf["predict_start"]
        predict_end = tf["predict_end"]
        tf_start = time.time()

        print(f"\n── Timeframe {label} ({ti + 1}/{len(timeframes)}) ──")

        predict_months = [m for m in all_months if predict_start <= m <= predict_end]
        if not predict_months:
            print(f"  [{_ts()}] No predict months — skipping")
            continue

        train_months = [m for m in all_months if earliest_month <= m <= train_end]
        if len(train_months) < min_training_months:
            print(f"  [{_ts()}] Insufficient training months ({len(train_months)}) — need {min_training_months} min — skipping")
            continue

        # Mask future sales and recompute lag/rolling features
        print(f"  [{_ts()}] Masking sales after {train_end.date()} and recomputing features...")
        t1 = time.time()
        masked_grid = mask_future_sales(full_grid, train_end)
        print(f"  [{_ts()}] Masking done ({time.time() - t1:.1f}s)")

        # Split train / predict
        train_mask = masked_grid["startdate"] <= train_end
        predict_mask = masked_grid["startdate"].isin(predict_months)

        # Drop rows with NaN in lag features (first few months of each DFU)
        train_data = masked_grid[train_mask].dropna(subset=[f"qty_lag_{lag}" for lag in LAG_RANGE])
        predict_data = masked_grid[predict_mask].copy()

        # Fill NaN lag features in predict data with 0 (skip categoricals)
        for col in feature_cols:
            if col in predict_data.columns and col not in cat_cols:
                predict_data[col] = predict_data[col].fillna(0)

        print(f"  [{_ts()}] Train: {len(train_data):,} rows, Predict: {len(predict_data):,} rows")

        if len(train_data) == 0 or len(predict_data) == 0:
            print(f"  [{_ts()}] Empty train or predict — skipping")
            continue

        # Resolve hyperparams: per-timeframe inline tuning (PL-002) or static defaults
        if inline_tuner_fn is not None:
            print(f"  [{_ts()}] Inline hyperparameter tuning (cutoff={train_end.date()})...")
            t_tune = time.time()
            effective_params = inline_tuner_fn(full_grid, feature_cols, cat_cols, train_end)
            print(f"  [{_ts()}] Inline tuning done ({time.time() - t_tune:.1f}s)")
        else:
            effective_params = model_params

        # ── Direct multi-output path (default) ───────────────────────────────
        if not recursive:
            preds, models = train_fn_per_cluster(
                train_data, predict_data, feature_cols, cat_cols, effective_params
            )

        # ── Recursive multi-step path (Feature 43) ───────────────────────────
        else:
            sorted_months = sorted(predict_months)
            first_predict = _fill_predict_nans(
                masked_grid[masked_grid["startdate"] == sorted_months[0]].copy(),
                feature_cols, cat_cols,
            )
            print(f"  [{_ts()}] Recursive: training on first month {sorted_months[0].date()}, "
                  f"then iterating {len(sorted_months)} months...")
            preds_first, models = train_fn_per_cluster(
                train_data, first_predict, feature_cols, cat_cols, effective_params
            )

        # ── SHAP feature selection + conditional retrain (Feature 42) ─────────
        # Runs after initial train_fn_per_cluster in both direct and recursive modes.
        # In recursive mode, updates models and preds_first (first month preds).
        effective_feature_cols = feature_cols
        effective_cat_cols = cat_cols
        if feature_selector_fn is not None:
            print(f"  [{_ts()}] SHAP feature selection (timeframe {label})...")
            t_shap = time.time()
            selected_features, shap_df = feature_selector_fn(
                models, train_data, feature_cols, cat_cols,
                tf["index"], train_end,
            )
            shap_timeframe_reports.append(shap_df)
            print(f"  [{_ts()}] SHAP done ({time.time() - t_shap:.1f}s)")

            # Retrain if SHAP dropped >= threshold of features (configurable via algorithm_config.yaml)
            retrain_threshold = algo_config.get("shap_retrain_threshold", 0.10) if algo_config else 0.10
            features_dropped = len(feature_cols) - len(selected_features)
            drop_pct = features_dropped / len(feature_cols) if feature_cols else 0
            if drop_pct >= retrain_threshold and set(selected_features) != set(feature_cols):
                print(
                    f"  [{_ts()}] Retraining with {len(selected_features)} SHAP-selected features "
                    f"(was {len(feature_cols)})..."
                )
                selected_cat_cols = [c for c in cat_cols if c in selected_features]
                effective_feature_cols = selected_features
                effective_cat_cols = selected_cat_cols

                # predict_data for SHAP retrain: first month only in recursive mode
                shap_predict_data = (
                    _fill_predict_nans(
                        masked_grid[masked_grid["startdate"] == sorted_months[0]].copy(),
                        selected_features, selected_cat_cols,
                    )
                    if recursive
                    else _fill_predict_nans(predict_data.copy(), selected_features, selected_cat_cols)
                )

                t_retrain = time.time()
                preds_retrain, models = train_fn_per_cluster(
                    train_data, shap_predict_data, selected_features, selected_cat_cols, effective_params
                )
                if recursive:
                    preds_first = preds_retrain
                else:
                    preds = preds_retrain
                print(f"  [{_ts()}] Retrain done ({time.time() - t_retrain:.1f}s)")
            else:
                print(f"  [{_ts()}] SHAP: all {len(feature_cols)} features retained")

        # ── Complete recursive loop for months 2+ ─────────────────────────────
        if recursive:
            all_month_preds = [preds_first]
            # Single copy of masked_grid; all subsequent updates are in-place
            # to avoid O(months) full-grid copies (~9.8M rows each).
            current_grid = masked_grid.copy()
            # Use incremental update (only touches affected months, ~10x faster)
            # Pass all_months (full grid months), not sorted_months (predict-only)
            update_grid_incremental(current_grid, sorted_months[0], preds_first, all_months)

            for month in sorted_months[1:]:
                month_data = _fill_predict_nans(
                    current_grid[current_grid["startdate"] == month].copy(),
                    effective_feature_cols, effective_cat_cols,
                )
                preds_month = _predict_single_month(models, month_data, effective_feature_cols)
                all_month_preds.append(preds_month)
                update_grid_incremental(current_grid, month, preds_month, all_months)
                print(f"    [{_ts()}] Recursive month {month.date()}: {len(preds_month):,} predictions")

            preds = pd.concat(all_month_preds, ignore_index=True)

        preds["model_id"] = model_id
        preds["timeframe"] = label
        preds["timeframe_idx"] = tf["index"]
        all_predictions.append(preds)

        # Persist the most recent timeframe's models for production inference (F1.1)
        if model_persistence_fn is not None and ti == len(timeframes) - 1:
            try:
                model_persistence_fn(models, effective_feature_cols, label)
            except Exception as exc:
                print(f"  [{_ts()}] Warning: model persistence failed: {exc}")

        print(f"  [{_ts()}] Timeframe {label} complete: {len(preds):,} predictions ({time.time() - tf_start:.1f}s)")

    if not all_predictions:
        print(f"\n[{_ts()}] No predictions generated. Check data range and timeframe count.")
        sys.exit(1)

    # ── Step 5: Combine, assign execution lag, attach actuals ────────────────
    print(f"\n[{_ts()}] Step 5: Combining predictions...")
    expanded, archive_expanded, combined = postprocess_predictions(
        all_predictions, sales_df, exec_lag_map, timeframes=timeframes,
    )

    # ── Step 6: Save output ──────────────────────────────────────────────────
    print(f"\n[{_ts()}] Step 6: Saving output...")
    # Merge recursive flag into extra_metadata for traceability
    _extra_meta = dict(extra_metadata or {})
    if recursive:
        _extra_meta["recursive"] = True
    output_path, archive_path, meta_path, metadata = save_backtest_output(
        output_df=expanded,
        archive_df=archive_expanded,
        output_dir=output_dir,
        model_id=model_id,
        cluster_strategy=cluster_strategy,
        n_timeframes=n_timeframes,
        model_params=model_params,
        model_params_key=model_params_key,
        timeframes=timeframes,
        earliest_month=earliest_month,
        latest_month=latest_month,
        extra_metadata=_extra_meta or None,
    )

    # ── Save SHAP outputs (Feature 42) ───────────────────────────────────────
    extra_artifact_paths: list[str] = []
    if feature_selector_fn is not None and shap_timeframe_reports:
        from common.shap_selector import save_shap_outputs
        print(f"\n[{_ts()}] Saving SHAP feature selection outputs...")
        _, shap_summary_path = save_shap_outputs(
            shap_timeframe_reports, output_path.parent, len(timeframes)
        )
        if shap_summary_path:
            extra_artifact_paths.append(str(shap_summary_path))

    # ── Step 7: MLflow logging ───────────────────────────────────────────────
    mlflow_params = {
        "n_timeframes": n_timeframes,
        "cluster_strategy": cluster_strategy,
        **{k: v for k, v in model_params.items() if not callable(v)},
    }

    log_backtest_run(
        model_type=model_type_tag,
        model_id=model_id,
        cluster_strategy=cluster_strategy,
        hyperparams=mlflow_params,
        metrics={
            "n_predictions": len(expanded),
            "n_dfus": int(expanded["item_id"].nunique()),
        },
        metadata=metadata,
        artifact_paths=[str(output_path), str(archive_path), str(meta_path)] + extra_artifact_paths,
    )

    elapsed = time.time() - t_start
    print(f"\n[{_ts()}] Backtest complete in {elapsed:.0f}s ({elapsed / 60:.1f}m)")
