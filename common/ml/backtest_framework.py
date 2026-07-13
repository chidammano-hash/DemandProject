"""Shared backtest orchestration framework.

Provides common logic for all backtest scripts:
- Timeframe generation
- Data loading from Postgres
- Execution-lag assignment and forecast_ck construction
- All-lag expansion for archive tables
- DFU cohort classification (cold-start, sparse, active)
- Output saving (CSV + metadata JSON)
- Accuracy computation (overall + per-cohort)
- MLflow logging
- Per-cluster adaptive hyperparameter profiles
- Per-step recursive accuracy reporting
- Noise injection for recursive training robustness

Model-specific scripts implement only the training/prediction functions.
"""

import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql

from common.core.constants import (
    ARCHIVE_COLS,
    CAT_FEATURES,
    FORECAST_QTY_COL,
    MAX_ARCHIVE_LAG,
    METADATA_COLS,
    MIN_TRAINING_MONTHS,
    NUMERIC_ITEM_FEATURES,
    NUMERIC_SKU_FEATURES,
    OUTPUT_COLS,
)
from common.core.db import get_db_params
from common.core.planning_date import get_planning_date
from common.core.utils import load_config
from common.ml.mlflow_utils import log_backtest_run
from common.services.forecast_population import resolve_forecast_sales_table
from common.services.metrics import compute_accuracy_metrics

logger = logging.getLogger(__name__)

# ── Per-cluster adaptive hyperparameter profiles ─────────────────────────────

# Profile priority order — first match wins
_PROFILE_PRIORITY = [
    "sparse_intermittent",
    "high_volume_stable",  # mean demand >= 50, low zeros — gets deeper trees
    "medium_volume_periodic",  # mean demand 5-100, low zeros
    "low_volume_volatile",
    "volatile_large_cluster",  # large (>300k rows) + high CV + mostly continuous
    "seasonal_dominant",
    "default",
]


def compute_cluster_demand_stats(
    train_df: pd.DataFrame,
    cluster_id: Any,
) -> dict[str, float]:
    """Compute demand characteristics for a single cluster from training data.

    All five keys are consumed by ``resolve_cluster_params`` via ``_matches_profile``,
    which reads them dynamically using the ``_min`` / ``_max`` suffix convention
    defined in ``config/forecasting/cluster_tuning_profiles.yaml``:

    - mean_demand:        mean of non-zero qty values
    - cv_demand:          coefficient of variation (std / mean) of qty
    - zero_demand_pct:    fraction of rows with qty == 0
    - seasonal_amplitude: std of monthly means / overall mean (proxy for seasonality)
    - n_rows:             total training row count — used by n_rows_min / n_rows_max
                          criteria to guard large-cluster profiles from matching
                          small clusters (e.g. sparse_intermittent, low_volume_volatile).

    The returned dict is also stored as ``meta["cluster_stats"]`` in the per-cluster
    training result for diagnostic purposes (visible in model metadata JSON).
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
        monthly_means = cluster_data.groupby(cluster_data["startdate"].dt.month)["qty"].mean()
        if len(monthly_means) > 1:
            seasonal_amplitude = float(monthly_means.std()) / overall_mean

    return {
        "mean_demand": mean_demand,
        "cv_demand": cv_demand,
        "zero_demand_pct": zero_demand_pct,
        "seasonal_amplitude": seasonal_amplitude,
        # Training row count — used by profile match_criteria (n_rows_min / n_rows_max)
        # to prevent large-cluster profiles from being overridden by small-cluster rules.
        "n_rows": float(len(qty)),
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

    # Phase 1: Exact cluster-name match (from per-cluster tuning)
    for profile_name, profile in profiles.items():
        if profile_name == "default":
            continue
        criteria = profile.get("match_criteria", {})
        if "cluster_name" in criteria and str(criteria["cluster_name"]) == str(cluster_id):
            overrides = profile.get("overrides", {})
            if not overrides:
                return base_params, profile_name
            resolved = {**base_params, **overrides}
            logger.info(
                "Cluster '%s': using tuned profile (cluster_name match)",
                cluster_id,
            )
            return resolved, profile_name

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
                    cluster_id,
                    profile_name,
                )
                return base_params, profile_name

            resolved = {**base_params, **overrides}
            logger.info(
                "Cluster '%s' matched profile '%s': overrides=%s",
                cluster_id,
                profile_name,
                overrides,
            )
            return resolved, profile_name

    logger.info("Cluster '%s': using global params (no profile match)", cluster_id)
    return base_params, "none"


def validate_cluster_tuning_profiles(db_params: dict[str, Any]) -> None:
    """Fail closed unless enabled tuning profiles match current clusters.

    Name-matched LightGBM overrides are unsafe after cluster membership or
    labels change. Backtests therefore require an exact label set, no active
    stale flags, and the same promoted cluster experiment stamped in YAML.
    """
    cfg = load_config("cluster_tuning_profiles.yaml")
    if not cfg.get("enabled", False):
        return

    try:
        with psycopg.connect(**db_params) as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT DISTINCT assignment.ml_cluster
                   FROM current_sku_cluster_assignment assignment
                   WHERE assignment.ml_cluster IS NOT NULL
                   ORDER BY assignment.ml_cluster"""
            )
            current_labels = {str(row[0]) for row in cur.fetchall()}
            cur.execute(
                """SELECT tuning.cluster_name
                   FROM cluster_tuning_profile_state tuning
                   WHERE tuning.stale = TRUE
                     AND EXISTS (
                         SELECT 1 FROM current_sku_cluster_assignment assignment
                         WHERE assignment.ml_cluster = tuning.cluster_name
                     )
                   ORDER BY tuning.cluster_name"""
            )
            stale_labels = {str(row[0]) for row in cur.fetchall()}
            cur.execute(
                "SELECT experiment_id FROM cluster_experiment "
                "WHERE is_promoted ORDER BY promoted_at DESC LIMIT 1"
            )
            promoted_rows = cur.fetchall()
            promoted_id = int(promoted_rows[0][0]) if promoted_rows else None
    except psycopg.Error as exc:
        raise RuntimeError("Current cluster tuning profiles could not be verified") from exc

    if stale_labels:
        raise RuntimeError(
            "Current cluster tuning profiles are stale for: "
            + ", ".join(sorted(stale_labels))
        )

    profile_labels = {
        str(criteria["cluster_name"])
        for profile in (cfg.get("cluster_profiles") or {}).values()
        if isinstance(profile, dict)
        and isinstance((criteria := profile.get("match_criteria")), dict)
        and criteria.get("cluster_name") is not None
    }
    missing = sorted(current_labels - profile_labels)
    if missing:
        retired = sorted(profile_labels - current_labels)
        raise RuntimeError(
            "Tuning profile labels do not exactly match current cluster labels "
            f"(missing={missing}, retired={retired})"
        )
    retired = sorted(profile_labels - current_labels)
    if retired:
        logger.warning(
            "Ignoring %d tuning profile(s) for retired cluster labels: %s",
            len(retired),
            ", ".join(retired),
        )

    yaml_experiment = (cfg.get("metadata") or {}).get("cluster_experiment_id")
    if promoted_id is None or yaml_experiment is None or int(yaml_experiment) != promoted_id:
        raise RuntimeError(
            "Tuning profiles do not match the promoted cluster experiment "
            f"(profiles={yaml_experiment}, promoted={promoted_id})"
        )


# ── Timeframe generation ─────────────────────────────────────────────────────


def _closed_month_cutoff(planning_date: pd.Timestamp) -> pd.Timestamp:
    """Return the first day of the last fully closed month."""
    planning_month = planning_date.normalize().replace(day=1)
    return planning_month - pd.DateOffset(months=1)


def generate_timeframes(
    earliest: pd.Timestamp,
    latest: pd.Timestamp,
    n: int = 10,
    embargo_months: int = 0,
) -> list[dict]:
    """Generate N expanding-window timeframes with optional embargo gap.

    For timeframe i (A=0 .. J=9):
      train_end     = latest - (N - i + embargo_months) months
      predict_start = train_end + 1 + embargo_months months

    The embargo creates a gap between the last training month and the first
    prediction month, matching the causality gap used during hyperparameter
    tuning (``gap_months`` in ``tuning.py``). Training cutoffs shift earlier
    by the embargo width so all N prediction windows remain non-empty. An
    embargo of 0 preserves the legacy behaviour.

    Args:
        earliest: First available month.
        latest: Last available month.
        n: Number of timeframes to generate.
        embargo_months: Number of months to skip between train_end and
            predict_start (default 0 = no gap beyond the natural 1-month
            offset).
    """
    timeframes = []
    for i in range(n):
        train_end = latest - pd.DateOffset(months=(n - i + embargo_months))
        train_end = train_end.normalize()  # midnight
        predict_start = train_end + pd.DateOffset(months=1 + embargo_months)
        label = chr(ord("A") + i)
        timeframes.append(
            {
                "label": label,
                "index": i,
                "train_start": earliest,
                "train_end": train_end,
                "predict_start": predict_start,
                "predict_end": latest,
            }
        )
    return timeframes


def _last_persistable_timeframe(timeframes: list[dict], all_months: list) -> int:
    """Index of the last timeframe with a non-empty predict window.

    Generated windows are valid under every supported embargo. This remains a
    defensive guard for custom or externally supplied timeframe lists.
    """
    return max(
        (
            ti
            for ti, tf in enumerate(timeframes)
            if any(tf["predict_start"] <= m <= tf["predict_end"] for m in all_months)
        ),
        default=len(timeframes) - 1,
    )


# ── Data loading ─────────────────────────────────────────────────────────────


def load_backtest_data(
    db: dict[str, Any],
    include_item_attrs: bool = True,
    algo_config: dict[str, Any] | None = None,
    include_customer_features: bool = False,
) -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    """Load sales, DFU attributes, and item attributes from Postgres.

    Sales are capped at the last fully closed month to ensure partial planning-
    month actuals cannot leak into backtesting.

    When include_customer_features=True, also loads customer_features_monthly
    and returns a 4-tuple: (sales_df, dfu_attrs, item_attrs, customer_features).
    Otherwise returns 3-tuple: (sales_df, dfu_attrs, item_attrs).
    """
    t1 = time.time()
    planning_cutoff = _closed_month_cutoff(pd.Timestamp(get_planning_date())).date()
    with psycopg.connect(**db) as conn:
        with conn.cursor() as _cur:
            sales_table = resolve_forecast_sales_table(_cur)

        with conn.cursor() as _cur:
            sales_query = sql.SQL(
                """
                SELECT d.sku_ck, s.item_id, s.customer_group, s.loc, s.startdate, s.qty
                FROM {} s
                INNER JOIN dim_sku d
                    ON d.item_id = s.item_id AND d.customer_group = s.customer_group AND d.loc = s.loc
                WHERE s.qty IS NOT NULL
                  AND s.type = 1
                  AND s.startdate <= %s
                ORDER BY d.sku_ck, s.startdate
                """
            ).format(sql.Identifier(sales_table))
            _cur.execute(
                sales_query,
                (planning_cutoff,),
            )
            _cols = [d[0] for d in _cur.description]
            sales_df = pd.DataFrame(_cur.fetchall(), columns=_cols)

        with conn.cursor() as _cur:
            _cur.execute("""
                SELECT d.sku_ck, d.item_id, d.customer_group, d.loc,
                       d.execution_lag, d.total_lt, ca.ml_cluster,
                       d.brand, d.region, d.abc_vol
                FROM dim_sku d
                LEFT JOIN current_sku_cluster_assignment ca
                       ON ca.sku_ck = d.sku_ck
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
        override_map = dict(zip(override_df["sku_ck"], override_df["cluster_label"], strict=True))
        original_clusters = dfu_attrs["ml_cluster"].copy()
        dfu_attrs["ml_cluster"] = (
            dfu_attrs["sku_ck"].map(override_map).fillna(dfu_attrs["ml_cluster"])
        )
        n_remapped = int((dfu_attrs["ml_cluster"] != original_clusters).sum())
        logger.info(
            "Cluster override applied: %s entries from %s, %s DFUs remapped",
            f"{len(override_map):,}",
            cluster_override_path,
            f"{n_remapped:,}",
        )

    # Restrict to specific cluster labels when requested (for experiments / quick runs)
    cluster_filter = algo_config.get("cluster_filter") if algo_config else None
    if cluster_filter:
        cluster_filter_str = [str(c) for c in cluster_filter]
        dfu_attrs = dfu_attrs[dfu_attrs["ml_cluster"].astype(str).isin(cluster_filter_str)].copy()
        dfus_in_filter = set(dfu_attrs["sku_ck"])
        sales_df = sales_df[sales_df["sku_ck"].isin(dfus_in_filter)].copy()
        logger.info(
            "Cluster filter applied: clusters=%s → %s DFUs, %s sales rows retained",
            cluster_filter,
            f"{len(dfu_attrs):,}",
            f"{len(sales_df):,}",
        )

    # Classify DFUs into cohorts based on sales history depth
    dfu_attrs = classify_dfu_cohorts(sales_df, dfu_attrs)

    logger.info(
        "Sales: %s rows, %s DFUs (%.1fs)",
        f"{len(sales_df):,}",
        f"{len(dfus_with_sales):,}",
        time.time() - t1,
    )
    logger.info("DFU attrs: %s, Item attrs: %s", f"{len(dfu_attrs):,}", f"{len(item_attrs):,}")

    if not include_customer_features:
        return sales_df, dfu_attrs, item_attrs

    # Load customer-derived features from customer_features_monthly
    customer_features = pd.DataFrame()
    try:
        with psycopg.connect(**db) as conn, conn.cursor() as _cur:
            _cur.execute("SELECT * FROM customer_features_monthly ORDER BY item_id, loc, startdate")
            _cols = [d[0] for d in _cur.description]
            customer_features = pd.DataFrame(_cur.fetchall(), columns=_cols)
        if not customer_features.empty:
            customer_features["startdate"] = pd.to_datetime(customer_features["startdate"])
            # Drop non-feature columns that would conflict
            for drop_col in ["load_ts"]:
                if drop_col in customer_features.columns:
                    customer_features = customer_features.drop(columns=[drop_col])
            logger.info("Customer features: %s rows", f"{len(customer_features):,}")
        else:
            logger.warning("customer_features_monthly is empty; enriched models will use zeros")
    except Exception as exc:
        logger.warning("Could not load customer_features_monthly: %s", exc)

    return sales_df, dfu_attrs, item_attrs, customer_features


# ── DFU cohort classification ───────────────────────────────────────────────


def classify_dfu_cohorts(
    sales_df: pd.DataFrame,
    dfu_attrs: pd.DataFrame,
    cold_start_threshold: int = 6,
    sparse_threshold: int = 12,
) -> pd.DataFrame:
    """Classify DFUs into cohorts based on months of sales history.

    Cohorts:
      - cold_start: < cold_start_threshold months of history (default < 6)
      - sparse: >= cold_start_threshold and < sparse_threshold (default 6-11)
      - active: >= sparse_threshold months of history (default >= 12)

    Adds a ``cohort`` column to dfu_attrs (in-place copy).

    Args:
        sales_df: Sales data with ``sku_ck`` and ``startdate`` columns.
        dfu_attrs: DFU attributes DataFrame with ``sku_ck`` column.
        cold_start_threshold: Months below which a DFU is cold-start.
        sparse_threshold: Months below which (but >= cold_start) a DFU is sparse.

    Returns:
        dfu_attrs with added ``cohort`` column.
    """
    # Determine the DFU key column name (sku_ck in restructured, dfu_ck legacy)
    dfu_key = "sku_ck" if "sku_ck" in sales_df.columns else "dfu_ck"

    dfu_month_counts = sales_df.groupby(dfu_key)["startdate"].nunique()

    # Map each DFU to its month count
    result = dfu_attrs.copy()
    attr_key = "sku_ck" if "sku_ck" in result.columns else "dfu_ck"
    result["_month_count"] = result[attr_key].map(dfu_month_counts).fillna(0).astype(int)

    # Classify
    conditions = [
        result["_month_count"] < cold_start_threshold,
        result["_month_count"] < sparse_threshold,
    ]
    choices = ["cold_start", "sparse"]
    result["cohort"] = np.select(conditions, choices, default="active")

    n_cold = int((result["cohort"] == "cold_start").sum())
    n_sparse = int((result["cohort"] == "sparse").sum())
    n_active = int((result["cohort"] == "active").sum())
    logger.info(
        "DFU cohorts: active=%d, sparse=%d, cold_start=%d (total=%d)",
        n_active,
        n_sparse,
        n_cold,
        len(result),
    )

    result.drop(columns=["_month_count"], inplace=True)
    return result


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
        result.loc[mask, "fcstdate"] = result.loc[mask, "startdate"] - pd.DateOffset(
            months=int(lag_val)
        )

    # Build forecast_ck (vectorized string concat via str.cat)
    result["forecast_ck"] = (
        result["item_id"]
        .astype(str)
        .str.cat(
            [
                result["customer_group"].astype(str),
                result["loc"].astype(str),
                result["fcstdate"].dt.strftime("%Y-%m-%d"),
                result["startdate"].dt.strftime("%Y-%m-%d"),
            ],
            sep="_",
        )
    )

    logger.info("Execution-lag assignment done (%.1fs)", time.time() - t0)
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
        df["item_id"]
        .astype(str)
        .str.cat(
            [
                df["customer_group"].astype(str),
                df["loc"].astype(str),
                df["fcstdate"].dt.strftime("%Y-%m-%d"),
                df["startdate"].dt.strftime("%Y-%m-%d"),
            ],
            sep="_",
        )
    )

    # Drop helper column
    df = df.drop(columns=["_train_end"])

    logger.info(
        "Natural lag assignment (0-%d) done: %s rows (%.1fs)",
        max_lag,
        f"{len(df):,}",
        time.time() - t0,
    )
    return df


# ── Checkpoint manager — survives OOM / crash across all backtests ─────────


class BacktestCheckpointer:
    """Incremental checkpoint manager for backtest timeframes.

    Saves each timeframe's predictions to a parquet file immediately after
    inference completes.  On re-run, completed timeframes are loaded from
    disk and skipped, so only unfinished work is retried.

    Usage in any backtest script::

        ckpt = BacktestCheckpointer(output_dir, model_id)
        for tf in timeframes:
            if ckpt.exists(tf["index"]):
                all_predictions.append(ckpt.load(tf["index"]))
                continue
            preds = ...  # run inference
            ckpt.save(preds, tf["index"])
            all_predictions.append(preds)
        ...
        ckpt.cleanup()   # after successful final save
    """

    def __init__(self, output_dir: Path, model_id: str, resume: bool = False) -> None:
        self._dir = Path(output_dir) / model_id / "_checkpoints"
        # Default: start fresh. Only resume if explicitly requested.
        if not resume and self._dir.exists():
            import shutil

            shutil.rmtree(self._dir, ignore_errors=True)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._existing: dict[int, Path] = {
            int(p.stem.split("_")[1]): p for p in self._dir.glob("tf_*.parquet")
        }
        if self._existing:
            logger.info(
                "Checkpoint: resuming — found %d saved timeframe(s)",
                len(self._existing),
            )

    def _path(self, idx: int) -> Path:
        return self._dir / f"tf_{idx:03d}.parquet"

    def exists(self, idx: int) -> bool:
        return idx in self._existing

    def load(self, idx: int) -> pd.DataFrame:
        df = pd.read_parquet(self._existing[idx])
        logger.info("  Checkpoint: loaded tf_%03d (%s rows)", idx, f"{len(df):,}")
        return df

    def save(self, df: pd.DataFrame, idx: int) -> None:
        path = self._path(idx)
        df.to_parquet(path, index=False)
        self._existing[idx] = path

    def load_all_existing(self) -> list[pd.DataFrame]:
        """Load all previously saved checkpoints in index order."""
        dfs = []
        for idx in sorted(self._existing.keys()):
            dfs.append(self.load(idx))
        return dfs

    def cleanup(self) -> None:
        """Remove checkpoint directory after successful completion."""
        import shutil

        if self._dir.exists():
            shutil.rmtree(self._dir, ignore_errors=True)
            logger.info("Checkpoint: cleaned up %s", self._dir)


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
    # Free the input list immediately — no longer needed
    all_predictions.clear()

    # Normalize column names: some scripts use 'timeframe_label' instead of 'timeframe'
    if "timeframe" not in combined.columns and "timeframe_label" in combined.columns:
        combined["timeframe"] = combined["timeframe_label"]
    logger.info("Total raw predictions: %s", f"{len(combined):,}")

    # Ensure startdate is datetime
    combined["startdate"] = pd.to_datetime(combined["startdate"])

    # Build actuals lookup once (small - one row per DFU x month)
    logger.info("Building actuals lookup...")
    actuals = sales_df.drop_duplicates(subset=["sku_ck", "startdate"])[
        ["sku_ck", "startdate", "qty"]
    ].rename(columns={"qty": "tothist_dmd"})

    # ── All-lags archive (compute BEFORE execution-lag to share `combined`) ──
    logger.info("Generating all-lags archive (lag 0-%d)...", MAX_ARCHIVE_LAG)

    if timeframes is not None:
        archive_expanded = assign_natural_lags(
            combined,
            timeframes,
            MAX_ARCHIVE_LAG,
            exec_lag_map,
        )
    else:
        logger.warning(
            "postprocess_predictions called without timeframes — "
            "archive will have identical predictions across all lags"
        )
        archive_expanded = _expand_to_all_lags_legacy(
            combined,
            MAX_ARCHIVE_LAG,
            exec_lag_map,
        )

    archive_expanded = archive_expanded.sort_values("timeframe_idx")
    archive_expanded = archive_expanded.drop_duplicates(
        subset=["forecast_ck", "model_id", "lag"], keep="last"
    )
    logger.info("Archive after dedup: %s", f"{len(archive_expanded):,}")
    archive_expanded = archive_expanded.merge(actuals, on=["sku_ck", "startdate"], how="left")

    # ── Execution-lag output (single row per prediction) ──────────────────
    logger.info("Assigning execution lag per DFU...")
    expanded = assign_execution_lag(combined, exec_lag_map)
    logger.info("Rows after execution-lag assignment: %s", f"{len(expanded):,}")

    expanded = expanded.sort_values("timeframe_idx")
    expanded = expanded.drop_duplicates(subset=["forecast_ck", "model_id"], keep="last")
    logger.info("After dedup: %s", f"{len(expanded):,}")

    logger.info("Attaching actuals...")
    t1 = time.time()
    expanded = expanded.merge(actuals, on=["sku_ck", "startdate"], how="left")
    logger.info("Actuals attached (%.1fs)", time.time() - t1)

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
        result["item_id"]
        .astype(str)
        .str.cat(
            [
                result["customer_group"].astype(str),
                result["loc"].astype(str),
                result["fcstdate"].dt.strftime("%Y-%m-%d"),
                result["startdate"].dt.strftime("%Y-%m-%d"),
            ],
            sep="_",
        )
    )

    logger.info(
        "Legacy all-lag expansion (0-%d) done: %s rows (%.1fs)",
        max_lag,
        f"{len(result):,}",
        time.time() - t0,
    )
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
    dfu_cohort_map: dict[str, str] | None = None,
) -> tuple[Path, Path, Path, dict]:
    """Save predictions CSV, archive CSV, and metadata JSON.

    Writes into a model-scoped subdirectory: output_dir / model_id /
    This prevents multiple backtest runs from overwriting each other (PL-001).

    Args:
        dfu_cohort_map: Optional mapping of DFU key -> cohort name
            (``"active"``, ``"sparse"``, ``"cold_start"``). When provided,
            per-cohort accuracy is computed and added to metadata.

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
    logger.info("Saved %s predictions to %s", f"{len(out):,}", output_path)

    # Archive CSV
    arch = archive_df[ARCHIVE_COLS].copy()
    arch["fcstdate"] = arch["fcstdate"].dt.strftime("%Y-%m-%d")
    arch["startdate"] = arch["startdate"].dt.strftime("%Y-%m-%d")

    archive_path = model_dir / "backtest_predictions_all_lags.csv"
    arch.to_csv(archive_path, index=False)
    logger.info("Saved %s archive rows to %s", f"{len(arch):,}", archive_path)

    # Build metadata
    metadata = {
        "model_id": model_id,
        "cluster_strategy": cluster_strategy,
        "n_timeframes": n_timeframes,
        model_params_key: dict(model_params),
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
        pd.to_numeric(out[FORECAST_QTY_COL], errors="coerce"),
        pd.to_numeric(out["tothist_dmd"], errors="coerce"),
    )
    if acc["wape"] is not None:
        metadata["accuracy_at_execution_lag"] = acc
        metadata["accuracy_overall"] = acc["accuracy_pct"]
        logger.info(
            "Accuracy at execution lag (%s rows): WAPE=%.2f%%, Bias=%.4f, Accuracy=%.2f%%",
            f"{acc['n_rows']:,}",
            acc["wape"],
            acc["bias"],
            acc["accuracy_pct"],
        )

    # Per-cohort accuracy breakdown
    if dfu_cohort_map:
        # Determine DFU key column — sku_ck in restructured codebase, dfu_ck in legacy
        dfu_key = "sku_ck" if "sku_ck" in output_df.columns else "dfu_ck"
        cohort_col = output_df[dfu_key].map(dfu_cohort_map).fillna("active")
        cohort_counts: dict[str, int] = {}
        cohort_accuracy: dict[str, float | None] = {}

        for cohort_name in ("active", "sparse", "cold_start"):
            mask = cohort_col == cohort_name
            cohort_counts[f"n_dfus_{cohort_name}"] = int(output_df.loc[mask, dfu_key].nunique())

            cohort_out = out[mask.values] if len(out) == len(mask) else out
            if cohort_counts[f"n_dfus_{cohort_name}"] > 0 and len(out) == len(mask):
                cohort_acc = compute_accuracy_metrics(
                    pd.to_numeric(cohort_out[FORECAST_QTY_COL], errors="coerce"),
                    pd.to_numeric(cohort_out["tothist_dmd"], errors="coerce"),
                )
                cohort_accuracy[f"accuracy_{cohort_name}"] = cohort_acc.get("accuracy_pct")
                logger.info(
                    "  Accuracy [%s] (%d DFUs): %s%%",
                    cohort_name,
                    cohort_counts[f"n_dfus_{cohort_name}"],
                    cohort_acc.get("accuracy_pct"),
                )
            else:
                cohort_accuracy[f"accuracy_{cohort_name}"] = None

        metadata.update(cohort_counts)
        metadata.update(cohort_accuracy)
        metadata["accuracy_population"] = "active_and_sparse"

    meta_path = model_dir / "backtest_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Saved metadata to %s", meta_path)

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

        importance = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)
        imp_path = output_dir / "feature_importance.csv"
        importance.to_csv(imp_path, index=False)
        logger.info("Saved feature importance to %s", imp_path)
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


# ── Recursive mode helpers ────────────────────────────────────────────────


def _model_feature_cols(feature_cols: list[str]) -> list[str]:
    """Return model input columns with metadata/target columns stripped."""
    return [col for col in feature_cols if col not in METADATA_COLS]


def _model_cat_cols(cat_cols: list[str], feature_cols: list[str]) -> list[str]:
    """Return categorical columns that are valid model inputs."""
    feature_set = set(feature_cols)
    return [col for col in cat_cols if col in feature_set and col not in METADATA_COLS]


def _inject_recursive_noise(
    qty_values: np.ndarray,
    noise_pct: float = 0.05,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Gaussian noise to qty values to simulate recursive prediction errors.

    Used during training to make the model robust to noisy lag inputs,
    reducing the distribution shift between training (real actuals as lags)
    and recursive inference (model predictions as lags).

    Args:
        qty_values: Array of quantity values (e.g. lag features).
        noise_pct: Standard deviation of noise as a fraction of the mean
            absolute value.  0.0 returns the original values unchanged.

    Returns:
        Array with additive Gaussian noise applied.
    """
    if noise_pct <= 0.0 or len(qty_values) == 0:
        return qty_values.copy()
    # Lag columns 2..N carry NaN for short-history DFUs (only qty_lag_1 is
    # guaranteed non-NaN at train time). np.abs(...).mean() over an array with
    # any NaN is NaN, and `scale <= 0.0` does NOT catch NaN (NaN <= 0 is False),
    # so the old code drew np.random.normal(0, NaN) and overwrote the ENTIRE
    # column with NaN — silently wiping all lag-2..N signal from recursive tree
    # training. Compute the scale over finite values only and perturb only the
    # finite entries, leaving missing lags as NaN (correctly "missing").
    finite = np.isfinite(qty_values)
    if not finite.any():
        return qty_values.copy()
    scale = noise_pct * np.abs(qty_values[finite]).mean()
    if not np.isfinite(scale) or scale <= 0.0:
        return qty_values.copy()
    out = qty_values.copy()
    normal = rng.normal if rng is not None else np.random.normal
    out[finite] = out[finite] + normal(0, scale, size=int(finite.sum()))
    return out


def _compute_step_wape(
    predictions: pd.DataFrame,
    actuals_lookup: dict[str, float],
) -> float | None:
    """Compute WAPE for a single recursive step.

    Args:
        predictions: DataFrame with ``sku_ck`` and ``basefcst_pref`` columns.
        actuals_lookup: Mapping from ``sku_ck`` to actual qty for this month.

    Returns:
        WAPE as a percentage, or None if no matching actuals.
    """
    if predictions.empty or not actuals_lookup:
        return None
    matched = predictions[predictions["sku_ck"].isin(actuals_lookup)]
    if matched.empty:
        return None
    fcst = matched[FORECAST_QTY_COL].values
    actual = matched["sku_ck"].map(actuals_lookup).values
    total_actual = np.abs(np.nansum(actual))
    if total_actual == 0:
        return None
    abs_error = np.nansum(np.abs(fcst - actual))
    return round(float(100.0 * abs_error / total_actual), 2)


def _log_timeframe_accuracy(
    preds: pd.DataFrame,
    sales_df: pd.DataFrame,
    predict_data: pd.DataFrame,
    label: str,
) -> None:
    """Log overall and per-cluster accuracy for a completed timeframe."""
    if preds.empty:
        return

    # Join predictions with actuals from sales_df
    predict_months = preds["startdate"].unique()
    actuals = sales_df[sales_df["startdate"].isin(predict_months)].drop_duplicates(
        subset=["sku_ck", "startdate"]
    )[["sku_ck", "startdate", "qty"]]
    merged = preds.merge(actuals, on=["sku_ck", "startdate"], how="inner")
    if merged.empty:
        logger.info("Timeframe %s accuracy: no actuals available", label)
        return

    # Overall timeframe accuracy
    total_actual = float(np.abs(merged["qty"].sum()))
    if total_actual > 0:
        total_error = float(np.abs(merged[FORECAST_QTY_COL] - merged["qty"]).sum())
        overall_wape = round(100.0 * total_error / total_actual, 2)
        overall_acc = round(100.0 - overall_wape, 2)
    else:
        overall_wape, overall_acc = 0.0, 0.0

    # Build sku_ck -> ml_cluster lookup from predict_data
    cluster_map = predict_data.drop_duplicates(subset="sku_ck")[["sku_ck", "ml_cluster"]]
    merged_c = merged.merge(cluster_map, on="sku_ck", how="left")

    # Per-cluster accuracy
    cluster_lines = []
    for cluster_label, grp in sorted(merged_c.groupby("ml_cluster", observed=True)):
        c_actual = float(np.abs(grp["qty"].sum()))
        if c_actual > 0:
            c_error = float(np.abs(grp[FORECAST_QTY_COL] - grp["qty"]).sum())
            c_wape = round(100.0 * c_error / c_actual, 2)
            c_acc = round(100.0 - c_wape, 2)
        else:
            c_wape, c_acc = 0.0, 0.0
        cluster_lines.append(
            f"    {cluster_label}: accuracy={c_acc:.1f}% wape={c_wape:.1f}% ({len(grp):,} rows)"
        )

    logger.info(
        "Timeframe %s accuracy: %.1f%% (wape=%.1f%%, %s matched rows)\n%s",
        label,
        overall_acc,
        overall_wape,
        f"{len(merged):,}",
        "\n".join(cluster_lines),
    )


def _compute_cluster_wape(
    preds_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    cluster_label: str,
    predict_data: pd.DataFrame,
) -> float | None:
    """Compute WAPE for a single cluster's predictions against actuals.

    Used as a safety check after SHAP retrain — if the retrained model is worse
    than the original, we revert to the original model.

    Returns WAPE as a percentage, or None if no matching actuals.
    """
    # Build cluster SKU set from predict_data
    cluster_skus = set(predict_data.loc[predict_data["ml_cluster"] == cluster_label, "sku_ck"])
    cluster_preds = preds_df[preds_df["sku_ck"].isin(cluster_skus)]
    if cluster_preds.empty:
        return None

    predict_months = cluster_preds["startdate"].unique()
    actuals = sales_df[
        (sales_df["startdate"].isin(predict_months)) & (sales_df["sku_ck"].isin(cluster_skus))
    ].drop_duplicates(subset=["sku_ck", "startdate"])[["sku_ck", "startdate", "qty"]]
    merged = cluster_preds.merge(actuals, on=["sku_ck", "startdate"], how="inner")
    if merged.empty:
        return None

    total_actual = float(np.abs(merged["qty"].sum()))
    if total_actual == 0:
        return None
    total_error = float(np.abs(merged[FORECAST_QTY_COL] - merged["qty"]).sum())
    return round(float(100.0 * total_error / total_actual), 2)


def _fill_predict_nans(
    predict_data: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    """Fill NaN values in numeric feature columns of predict_data with 0."""
    feature_cols = _model_feature_cols(feature_cols)
    cat_cols = _model_cat_cols(cat_cols, feature_cols)
    for col in feature_cols:
        if col in predict_data.columns and col not in cat_cols:
            if pd.api.types.is_numeric_dtype(predict_data[col]):
                predict_data[col] = predict_data[col].fillna(0)
    return predict_data


def _predict_single_month(
    models: dict,
    predict_data: pd.DataFrame,
    feature_cols: list[str],
    per_cluster_feature_cols: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Route a single-month batch through per-cluster models for recursive inference.

    Used by recursive multi-step inference (Feature 43). Each DFU row is
    routed to its cluster's model; DFUs with no matching cluster are skipped.

    Args:
        models: ``{cluster_label: model}`` dict from ``train_and_predict_per_cluster``.
        predict_data: Feature matrix for one month, all DFUs (must have ``ml_cluster``).
        feature_cols: Ordered list of feature columns passed to each model (default/fallback).
        per_cluster_feature_cols: Optional per-cluster feature column overrides from
            per-cluster SHAP selection. When provided, each cluster uses its own
            feature set; clusters not in the dict fall back to ``feature_cols``.
    """
    parts = []
    global_model = models.get("global")
    for cluster, group in predict_data.groupby("ml_cluster", observed=True):
        m = models.get(cluster) or global_model
        if m is None:
            continue
        # ml_cluster is in METADATA_COLS (not in feature_cols) — used for
        # partitioning only, never passed to models as an input feature.
        cluster_feats = (
            per_cluster_feature_cols.get(str(cluster), feature_cols)
            if per_cluster_feature_cols
            else feature_cols
        )
        cluster_feats = _model_feature_cols(cluster_feats)
        # For baseline models (e.g., _RollingMeanModel, _SeasonalNaiveModel)
        # that need sku_ck mapping, set the DFU keys before calling predict
        # on feature-only DataFrames.
        if hasattr(m, "_sku_cks"):
            m._sku_cks = group["sku_ck"].tolist()
        if hasattr(m, "_months"):
            m._months = pd.to_datetime(group["startdate"]).dt.month.tolist()
        preds = np.maximum(m.predict(group[cluster_feats]), 0)
        r = group[_PREDICT_META_COLS].copy()
        r[FORECAST_QTY_COL] = preds
        parts.append(r)
    if not parts:
        return pd.DataFrame(columns=[*_PREDICT_META_COLS, FORECAST_QTY_COL])
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
        tuple[list[str] | dict[str, list[str]], pd.DataFrame],
    ]
    | None = None,
    recursive: bool = False,
    model_persistence_fn: Callable[[Any, list[str] | dict[str, list[str]], str], None]
    | None = None,
    algo_config: dict[str, Any] | None = None,
    embargo_months: int = 0,
    resume: bool = False,
) -> None:
    """Run a complete LightGBM per-cluster backtest.

    All algorithms use per-cluster strategy. Options (recursive, SHAP, tuning)
    are passed via closures rather than CLI flags; see forecast_pipeline_config.yaml.
    """
    from common.ml.feature_engineering import (
        build_feature_matrix,
        get_feature_columns,
        mask_future_sales,
        update_grid_incremental,
    )

    cluster_strategy = (algo_config or {}).get("cluster_strategy", "per_cluster")
    t_start = time.time()
    db = get_db_params()

    logger.info(
        "Backtest: strategy=%s, model_id=%s, n_timeframes=%d, recursive=%s",
        cluster_strategy,
        model_id,
        n_timeframes,
        recursive,
    )

    # ── Step 1: Load data ────────────────────────────────────────────────────
    logger.info("Step 1: Loading data from Postgres...")
    include_customer_features = bool((algo_config or {}).get("customer_features", False))
    data_result = load_backtest_data(
        db,
        algo_config=algo_config,
        include_customer_features=include_customer_features,
    )
    if include_customer_features:
        sales_df, dfu_attrs, item_attrs, customer_features = data_result
    else:
        sales_df, dfu_attrs, item_attrs = data_result
        customer_features = None

    validate_cluster_tuning_profiles(db)

    # Execution lag lookup
    exec_lag_map = dfu_attrs.set_index("sku_ck")["execution_lag"].fillna(0).astype(int).to_dict()

    # DFU cohort map for per-cohort accuracy reporting
    dfu_cohort_map: dict[str, str] | None = None
    if "cohort" in dfu_attrs.columns:
        dfu_cohort_map = dfu_attrs.set_index("sku_ck")["cohort"].to_dict()

    # ── Step 2: Generate timeframes ──────────────────────────────────────────
    planning_dt = pd.Timestamp(get_planning_date())
    # Backtests only score fully closed actual months; the planning month is open.
    planning_cutoff = _closed_month_cutoff(planning_dt)
    latest_month = min(sales_df["startdate"].max(), planning_cutoff)
    earliest_month = sales_df["startdate"].min()
    # Filter out any sales beyond the planning date
    sales_df = sales_df[sales_df["startdate"] <= latest_month].copy()
    logger.info(
        "Date range: %s -> %s (planning date: %s)",
        earliest_month.date(),
        latest_month.date(),
        planning_dt.date(),
    )

    timeframes = generate_timeframes(
        earliest_month, latest_month, n_timeframes, embargo_months=embargo_months
    )
    if embargo_months:
        logger.info("Embargo gap: %d month(s) between train_end and predict_start", embargo_months)
    logger.info("Step 2: Generated %d timeframes:", len(timeframes))
    for tf in timeframes:
        logger.info(
            "  %s: train [%s -> %s], predict [%s -> %s]",
            tf["label"],
            tf["train_start"].date(),
            tf["train_end"].date(),
            tf["predict_start"].date(),
            tf["predict_end"].date(),
        )

    all_months = sorted(sales_df["startdate"].unique())

    # ── Step 3: Build feature matrix ONCE ────────────────────────────────────
    logger.info("Step 3: Building feature matrix (one-time)...")
    full_grid = build_feature_matrix(
        sales_df,
        dfu_attrs,
        item_attrs,
        all_months,
        cat_dtype=cat_dtype,
        customer_features=customer_features,
    )
    feature_cols = _model_feature_cols(get_feature_columns(full_grid))
    cat_cols = [c for c in CAT_FEATURES if c in feature_cols and c in full_grid.columns]
    cat_cols = _model_cat_cols(cat_cols, feature_cols)
    logger.info("Features: %d columns, cat: %s", len(feature_cols), cat_cols)

    # ── Checkpoint manager for incremental saves ──────────────────────────────
    ckpt = BacktestCheckpointer(output_dir, model_id, resume=resume)

    # ── Step 4: Train & predict per timeframe ────────────────────────────────
    logger.info("Step 4: Running %d timeframe backtests...", len(timeframes))
    all_predictions = []
    shap_timeframe_reports: list[pd.DataFrame] = []
    recursive_step_metrics: list[dict[str, Any]] = []

    # Resume from any existing checkpoints
    all_predictions.extend(ckpt.load_all_existing())

    # Persist production models from the last timeframe that actually has a
    # predict window (the final one is skipped under embargo — see helper).
    persist_ti = _last_persistable_timeframe(timeframes, all_months)

    for ti, tf in enumerate(timeframes):
        if ckpt.exists(tf["index"]):
            logger.info(
                "Timeframe %s (%d/%d) — checkpoint exists, skipping",
                tf["label"],
                ti + 1,
                len(timeframes),
            )
            continue
        label = tf["label"]
        train_end = tf["train_end"]
        predict_start = tf["predict_start"]
        predict_end = tf["predict_end"]
        tf_start = time.time()

        logger.info("Timeframe %s (%d/%d)", label, ti + 1, len(timeframes))

        predict_months = [m for m in all_months if predict_start <= m <= predict_end]
        if not predict_months:
            logger.info("No predict months -- skipping")
            continue

        train_months = [m for m in all_months if earliest_month <= m <= train_end]
        if len(train_months) < min_training_months:
            logger.info(
                "Insufficient training months (%d) -- need %d min -- skipping",
                len(train_months),
                min_training_months,
            )
            continue

        # Mask future sales and recompute lag/rolling features
        logger.info("Masking sales after %s and recomputing features...", train_end.date())
        t1 = time.time()
        masked_grid = mask_future_sales(full_grid, train_end)
        logger.info("Masking done (%.1fs)", time.time() - t1)

        # Split train / predict
        train_mask = masked_grid["startdate"] <= train_end
        predict_mask = masked_grid["startdate"].isin(predict_months)

        # Drop only rows missing the most recent lag (qty_lag_1).
        # LightGBM handles NaN natively by creating
        # a separate "missing" bin during histogram splits.  Requiring all 12
        # lags to be non-NaN was overly aggressive: a DFU with 8 months of
        # history would have NaN for lags 9-12 in ALL rows, losing the DFU
        # entirely from training.  Now we only need the first month to be
        # available, letting short-history DFUs contribute training data.
        train_data = masked_grid[train_mask].dropna(subset=["qty_lag_1"])
        predict_data = masked_grid[predict_mask].copy()

        # Fill NaN lag features in predict data with 0 (skip categoricals)
        for col in feature_cols:
            if col in predict_data.columns and col not in cat_cols:
                predict_data[col] = predict_data[col].fillna(0)

        logger.info(
            "Train: %s rows, Predict: %s rows", f"{len(train_data):,}", f"{len(predict_data):,}"
        )

        if len(train_data) == 0 or len(predict_data) == 0:
            logger.info("Empty train or predict -- skipping")
            continue

        # Resolve hyperparams: per-timeframe inline tuning (PL-002) or static defaults
        if inline_tuner_fn is not None:
            logger.info("Inline hyperparameter tuning (cutoff=%s)...", train_end.date())
            t_tune = time.time()
            effective_params = inline_tuner_fn(full_grid, feature_cols, cat_cols, train_end)
            logger.info("Inline tuning done (%.1fs)", time.time() - t_tune)
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
                feature_cols,
                cat_cols,
            )
            logger.info(
                "Recursive: training on first month %s, then iterating %d months...",
                sorted_months[0].date(),
                len(sorted_months),
            )

            # ── Noise injection (teacher forcing lite) ────────────────────────
            # Optionally perturb lag features in training data so the model
            # learns to be robust to the noisy inputs it will see during
            # recursive inference (where predictions replace true actuals).
            algo_cfg_noise = algo_config or {}
            noise_enabled = algo_cfg_noise.get("recursive_noise_enabled", False)
            noise_pct = algo_cfg_noise.get("recursive_noise_pct", 0.05)
            # Lag smoothing factor: blend prediction-derived lags with prior
            # lag values to dampen recursive error compounding (0 = disabled).
            lag_smooth = float(algo_cfg_noise.get("recursive_lag_smooth", 0.0))
            train_data_for_fit = train_data
            if noise_enabled and noise_pct > 0:
                lag_cols = [c for c in feature_cols if c.startswith("qty_lag_")]
                if lag_cols:
                    train_data_for_fit = train_data.copy()
                    for col in lag_cols:
                        train_data_for_fit[col] = _inject_recursive_noise(
                            train_data_for_fit[col].values, noise_pct
                        )
                    logger.info(
                        "Recursive noise injection: %.1f%% on %d lag cols",
                        noise_pct * 100,
                        len(lag_cols),
                    )
            if lag_smooth > 0:
                logger.info(
                    "Recursive lag smoothing: factor=%.2f (applied from step 3+)",
                    lag_smooth,
                )

            preds_first, models = train_fn_per_cluster(
                train_data_for_fit, first_predict, feature_cols, cat_cols, effective_params
            )

        # ── SHAP feature selection + conditional retrain (Feature 42) ─────────
        # Runs after initial train_fn_per_cluster in both direct and recursive modes.
        # In recursive mode, updates models and preds_first (first month preds).
        effective_feature_cols = feature_cols
        effective_cat_cols = cat_cols
        per_cluster_feature_cols: dict[str, list[str]] | None = None
        per_cluster_cat_cols: dict[str, list[str]] | None = None
        if feature_selector_fn is not None:
            logger.info("SHAP feature selection (timeframe %s)...", label)
            t_shap = time.time()
            selected_features, shap_df = feature_selector_fn(
                models,
                train_data,
                feature_cols,
                cat_cols,
                tf["index"],
                train_end,
            )
            shap_timeframe_reports.append(shap_df)
            logger.info("SHAP done (%.1fs)", time.time() - t_shap)

            # Retrain if SHAP dropped >= threshold of features (configurable via forecast_pipeline_config.yaml)
            retrain_threshold = (
                algo_config.get("shap_retrain_threshold", 0.10) if algo_config else 0.10
            )

            if isinstance(selected_features, dict):
                # ── Per-cluster feature selection ──────────────────────────────
                per_cluster_feature_cols = {}
                per_cluster_cat_cols = {}
                clusters_to_retrain: list[str] = []

                for cluster_label, sel_feats in selected_features.items():
                    sel_feats = _model_feature_cols(sel_feats)
                    features_dropped = len(feature_cols) - len(sel_feats)
                    drop_pct = features_dropped / len(feature_cols) if feature_cols else 0
                    if drop_pct >= retrain_threshold and set(sel_feats) != set(feature_cols):
                        per_cluster_feature_cols[cluster_label] = sel_feats
                        per_cluster_cat_cols[cluster_label] = [
                            c for c in cat_cols if c in sel_feats
                        ]
                        clusters_to_retrain.append(cluster_label)
                    else:
                        per_cluster_feature_cols[cluster_label] = feature_cols
                        per_cluster_cat_cols[cluster_label] = cat_cols

                if clusters_to_retrain:
                    logger.info(
                        "Retraining %d/%d clusters with per-cluster SHAP-selected features...",
                        len(clusters_to_retrain),
                        len(selected_features),
                    )
                    t_retrain = time.time()
                    reverted_clusters: list[str] = []
                    for cluster_label in clusters_to_retrain:
                        sel_feats = per_cluster_feature_cols[cluster_label]
                        sel_cats = per_cluster_cat_cols[cluster_label]
                        cluster_train = train_data[train_data["ml_cluster"] == cluster_label]

                        if recursive:
                            cluster_predict = _fill_predict_nans(
                                masked_grid[
                                    (masked_grid["startdate"] == sorted_months[0])
                                    & (masked_grid["ml_cluster"] == cluster_label)
                                ].copy(),
                                sel_feats,
                                sel_cats,
                            )
                        else:
                            cluster_predict = _fill_predict_nans(
                                predict_data[predict_data["ml_cluster"] == cluster_label].copy(),
                                sel_feats,
                                sel_cats,
                            )

                        # Save original model for safety check
                        original_model = models.get(cluster_label)

                        retrain_preds, retrain_models = train_fn_per_cluster(
                            cluster_train,
                            cluster_predict,
                            sel_feats,
                            sel_cats,
                            effective_params,
                        )

                        # ── Safety check: revert if retrained model is worse ──
                        # Compare WAPE of retrained predictions vs original predictions.
                        # If retrained is worse, keep original model and predictions.
                        original_preds_source = preds_first if recursive else preds
                        orig_wape = _compute_cluster_wape(
                            original_preds_source,
                            sales_df,
                            cluster_label,
                            predict_data,
                        )
                        retrain_wape = None
                        if not retrain_preds.empty:
                            retrain_wape = _compute_cluster_wape(
                                retrain_preds,
                                sales_df,
                                cluster_label,
                                predict_data,
                            )

                        if (
                            orig_wape is not None
                            and retrain_wape is not None
                            and retrain_wape > orig_wape
                        ):
                            # Retrained model is WORSE — revert to original
                            logger.warning(
                                "SHAP retrain safety: cluster '%s' reverted "
                                "(retrain_wape=%.1f%% > orig_wape=%.1f%%)",
                                cluster_label,
                                retrain_wape,
                                orig_wape,
                            )
                            per_cluster_feature_cols[cluster_label] = feature_cols
                            per_cluster_cat_cols[cluster_label] = cat_cols
                            models[cluster_label] = original_model
                            reverted_clusters.append(cluster_label)
                            continue

                        models[cluster_label] = retrain_models.get(
                            cluster_label, models.get(cluster_label)
                        )

                        # Merge retrained predictions
                        if not retrain_preds.empty:
                            if recursive:
                                retrain_keys = set(retrain_preds["sku_ck"])
                                preds_first = pd.concat(
                                    [
                                        preds_first[~preds_first["sku_ck"].isin(retrain_keys)],
                                        retrain_preds,
                                    ],
                                    ignore_index=True,
                                )
                            else:
                                retrain_keys = set(
                                    zip(
                                        retrain_preds["sku_ck"],
                                        retrain_preds["startdate"],
                                        strict=True,
                                    )
                                )
                                keep_mask = (
                                    pd.Series(
                                        [
                                            k not in retrain_keys
                                            for k in zip(
                                                preds["sku_ck"], preds["startdate"], strict=True
                                            )
                                        ]
                                    ).values
                                    if len(preds) > 0
                                    else []
                                )
                                preds = pd.concat(
                                    [
                                        preds[keep_mask] if len(preds) > 0 else preds,
                                        retrain_preds,
                                    ],
                                    ignore_index=True,
                                )

                    if reverted_clusters:
                        logger.info(
                            "Per-cluster retrain done (%.1fs): %d retrained, %d reverted (%s)",
                            time.time() - t_retrain,
                            len(clusters_to_retrain) - len(reverted_clusters),
                            len(reverted_clusters),
                            ", ".join(reverted_clusters),
                        )
                    else:
                        logger.info("Per-cluster retrain done (%.1fs)", time.time() - t_retrain)
                else:
                    logger.info(
                        "SHAP per-cluster: all %d clusters retained all features",
                        len(selected_features),
                    )

            else:
                # ── Legacy single-selection path (global strategy) ─────────────
                selected_features = _model_feature_cols(selected_features)
                features_dropped = len(feature_cols) - len(selected_features)
                drop_pct = features_dropped / len(feature_cols) if feature_cols else 0
                if drop_pct >= retrain_threshold and set(selected_features) != set(feature_cols):
                    logger.info(
                        "Retraining with %d SHAP-selected features (was %d)...",
                        len(selected_features),
                        len(feature_cols),
                    )
                    selected_cat_cols = [c for c in cat_cols if c in selected_features]
                    effective_feature_cols = selected_features
                    effective_cat_cols = selected_cat_cols

                    # predict_data for SHAP retrain: first month only in recursive mode
                    shap_predict_data = (
                        _fill_predict_nans(
                            masked_grid[masked_grid["startdate"] == sorted_months[0]].copy(),
                            selected_features,
                            selected_cat_cols,
                        )
                        if recursive
                        else _fill_predict_nans(
                            predict_data.copy(), selected_features, selected_cat_cols
                        )
                    )

                    # Save original state for safety check
                    original_models = dict(models)
                    original_preds_first = preds_first.copy() if recursive else None
                    original_preds = preds.copy() if not recursive and len(preds) > 0 else None

                    t_retrain = time.time()
                    preds_retrain, models_retrain = train_fn_per_cluster(
                        train_data,
                        shap_predict_data,
                        selected_features,
                        selected_cat_cols,
                        effective_params,
                    )

                    # ── Safety check: compare overall WAPE before/after retrain ──
                    orig_source = preds_first if recursive else preds
                    orig_overall_wape = (
                        _compute_cluster_wape(
                            orig_source,
                            sales_df,
                            "__all__",
                            predict_data,
                        )
                        if not orig_source.empty
                        else None
                    )
                    retrain_overall_wape = (
                        _compute_cluster_wape(
                            preds_retrain,
                            sales_df,
                            "__all__",
                            predict_data,
                        )
                        if not preds_retrain.empty
                        else None
                    )

                    # For __all__ cluster, _compute_cluster_wape won't find matches
                    # since it filters by ml_cluster. Use direct WAPE comparison:
                    if not preds_retrain.empty:
                        _r_months = preds_retrain["startdate"].unique()
                        _r_actuals = sales_df[
                            sales_df["startdate"].isin(_r_months)
                        ].drop_duplicates(subset=["sku_ck", "startdate"])[
                            ["sku_ck", "startdate", "qty"]
                        ]
                        _r_merged = preds_retrain.merge(
                            _r_actuals, on=["sku_ck", "startdate"], how="inner"
                        )
                        if not _r_merged.empty:
                            _r_total = float(np.abs(_r_merged["qty"].sum()))
                            if _r_total > 0:
                                retrain_overall_wape = round(
                                    100.0
                                    * float(
                                        np.abs(_r_merged[FORECAST_QTY_COL] - _r_merged["qty"]).sum()
                                    )
                                    / _r_total,
                                    2,
                                )

                    if not orig_source.empty:
                        _o_months = orig_source["startdate"].unique()
                        _o_actuals = sales_df[
                            sales_df["startdate"].isin(_o_months)
                        ].drop_duplicates(subset=["sku_ck", "startdate"])[
                            ["sku_ck", "startdate", "qty"]
                        ]
                        _o_merged = orig_source.merge(
                            _o_actuals, on=["sku_ck", "startdate"], how="inner"
                        )
                        if not _o_merged.empty:
                            _o_total = float(np.abs(_o_merged["qty"].sum()))
                            if _o_total > 0:
                                orig_overall_wape = round(
                                    100.0
                                    * float(
                                        np.abs(_o_merged[FORECAST_QTY_COL] - _o_merged["qty"]).sum()
                                    )
                                    / _o_total,
                                    2,
                                )

                    if (
                        orig_overall_wape is not None
                        and retrain_overall_wape is not None
                        and retrain_overall_wape > orig_overall_wape
                    ):
                        logger.warning(
                            "SHAP retrain safety: global retrain reverted "
                            "(retrain_wape=%.1f%% > orig_wape=%.1f%%)",
                            retrain_overall_wape,
                            orig_overall_wape,
                        )
                        effective_feature_cols = feature_cols
                        effective_cat_cols = cat_cols
                        models = original_models
                        if recursive and original_preds_first is not None:
                            preds_first = original_preds_first
                        elif original_preds is not None:
                            preds = original_preds
                    else:
                        models = models_retrain
                        if recursive:
                            preds_first = preds_retrain
                        else:
                            preds = preds_retrain
                    logger.info("Retrain done (%.1fs)", time.time() - t_retrain)
                else:
                    logger.info("SHAP: all %d features retained", len(feature_cols))

        # ── Complete recursive loop for months 2+ ─────────────────────────────
        if recursive:
            all_month_preds = [preds_first]
            # Single copy of masked_grid; all subsequent updates are in-place
            # to avoid O(months) full-grid copies (~9.8M rows each).
            current_grid = masked_grid.copy()
            # Use incremental update (only touches affected months, ~10x faster)
            # Pass all_months (full grid months), not sorted_months (predict-only)
            update_grid_incremental(current_grid, sorted_months[0], preds_first, all_months)

            # Build actuals lookup for per-step accuracy reporting.
            # Maps month -> {sku_ck -> qty} for months in the predict window.
            actuals_by_month: dict[pd.Timestamp, dict[str, float]] = {}
            for m in sorted_months:
                m_sales = sales_df[sales_df["startdate"] == m]
                if not m_sales.empty:
                    actuals_by_month[m] = (
                        m_sales.drop_duplicates(subset="sku_ck")
                        .set_index("sku_ck")["qty"]
                        .to_dict()
                    )

            # Per-step accuracy tracking
            step_metrics: list[dict[str, Any]] = []

            # Step 1 accuracy (first month)
            if sorted_months[0] in actuals_by_month:
                step1_wape = _compute_step_wape(preds_first, actuals_by_month[sorted_months[0]])
                step_metrics.append(
                    {
                        "step": 1,
                        "month": str(sorted_months[0].date()),
                        "wape": step1_wape,
                        "n_dfus": len(preds_first),
                    }
                )

            for step_idx, month in enumerate(sorted_months[1:], start=2):
                # When per-cluster SHAP is active, fill NaNs using the union of
                # all per-cluster feature sets so every cluster's columns are present.
                if per_cluster_feature_cols:
                    all_feats_union = sorted(set().union(*per_cluster_feature_cols.values()))
                    all_cats_union = [c for c in cat_cols if c in all_feats_union]
                    month_data = _fill_predict_nans(
                        current_grid[current_grid["startdate"] == month].copy(),
                        all_feats_union,
                        all_cats_union,
                    )
                else:
                    month_data = _fill_predict_nans(
                        current_grid[current_grid["startdate"] == month].copy(),
                        effective_feature_cols,
                        effective_cat_cols,
                    )
                preds_month = _predict_single_month(
                    models,
                    month_data,
                    effective_feature_cols,
                    per_cluster_feature_cols,
                )
                all_month_preds.append(preds_month)
                # Apply lag smoothing for step 3+. Step 2 uses raw prediction
                # (lag-1 for step 2 came from step 1 prediction, not from prior
                # lag history, so smoothing with a masked zero would be harmful).
                step_smooth = lag_smooth if step_idx > 2 else 0.0
                update_grid_incremental(
                    current_grid,
                    month,
                    preds_month,
                    all_months,
                    smooth_factor=step_smooth,
                )

                # Compute per-step accuracy if actuals available
                if month in actuals_by_month:
                    step_wape = _compute_step_wape(preds_month, actuals_by_month[month])
                    step_metrics.append(
                        {
                            "step": step_idx,
                            "month": str(month.date()),
                            "wape": step_wape,
                            "n_dfus": len(preds_month),
                        }
                    )

                logger.debug(
                    "Recursive step %d, month %s: %s predictions",
                    step_idx,
                    month.date(),
                    f"{len(preds_month):,}",
                )

            preds = pd.concat(all_month_preds, ignore_index=True)

            # Accumulate per-step metrics across timeframes
            if step_metrics:
                for sm in step_metrics:
                    sm["timeframe"] = label
                recursive_step_metrics.extend(step_metrics)
                logger.info(
                    "Timeframe %s: %d recursive steps tracked",
                    label,
                    len(step_metrics),
                )

        preds["model_id"] = model_id
        preds["timeframe"] = label
        preds["timeframe_idx"] = tf["index"]

        # ── Per-timeframe accuracy summary ───────────────────────────────────
        _log_timeframe_accuracy(preds, sales_df, predict_data, label)

        ckpt.save(preds, tf["index"])
        all_predictions.append(preds)

        # Persist the most-recent trainable timeframe's models for production
        # inference (F1.1). Under embargo the final timeframe has no predict
        # window and is skipped, so target the last persistable timeframe.
        if model_persistence_fn is not None and ti == persist_ti:
            try:
                model_persistence_fn(
                    models,
                    per_cluster_feature_cols
                    if per_cluster_feature_cols
                    else effective_feature_cols,
                    label,
                )
            except Exception as exc:
                logger.warning("Model persistence failed: %s", exc)

        logger.info(
            "Timeframe %s complete: %s predictions (%.1fs)",
            label,
            f"{len(preds):,}",
            time.time() - tf_start,
        )

    if not all_predictions:
        logger.error("No predictions generated. Check data range and timeframe count.")
        sys.exit(1)

    # ── Step 5: Combine, assign execution lag, attach actuals ────────────────
    logger.info("Step 5: Combining predictions...")
    expanded, archive_expanded, _combined = postprocess_predictions(
        all_predictions,
        sales_df,
        exec_lag_map,
        timeframes=timeframes,
    )

    # ── Step 6: Save output ──────────────────────────────────────────────────
    logger.info("Step 6: Saving output...")
    # Merge recursive flag into extra_metadata for traceability
    _extra_meta = dict(extra_metadata or {})
    if recursive:
        _extra_meta["recursive"] = True
        _cfg = algo_config or {}
        _extra_meta["recursive_noise_pct"] = _cfg.get("recursive_noise_pct", 0.05)
        _extra_meta["recursive_lag_smooth"] = _cfg.get("recursive_lag_smooth", 0.0)
        # Attach per-step accuracy metrics collected across all timeframes
        if recursive_step_metrics:
            _extra_meta["recursive_step_metrics"] = recursive_step_metrics
            wapes_with_values = [
                m["wape"] for m in recursive_step_metrics if m.get("wape") is not None
            ]
            _extra_meta["recursive_accuracy_degradation"] = {
                "step_1_wape": recursive_step_metrics[0]["wape"]
                if recursive_step_metrics
                else None,
                "last_step_wape": recursive_step_metrics[-1]["wape"]
                if recursive_step_metrics
                else None,
                "mean_wape": round(float(np.mean(wapes_with_values)), 2)
                if wapes_with_values
                else None,
            }
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
        dfu_cohort_map=dfu_cohort_map,
    )

    # ── Save SHAP outputs (Feature 42) ───────────────────────────────────────
    extra_artifact_paths: list[str] = []
    if feature_selector_fn is not None and shap_timeframe_reports:
        from common.ml.shap_selector import save_shap_outputs

        logger.info("Saving SHAP feature selection outputs...")
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
        artifact_paths=[
            str(output_path),
            str(archive_path),
            str(meta_path),
            *extra_artifact_paths,
        ],
    )

    ckpt.cleanup()

    elapsed = time.time() - t_start
    logger.info("Backtest complete in %.0fs (%.1fm)", elapsed, elapsed / 60)
