"""Execution-lag accuracy helpers for Advanced Expert Panel.

Natural lag formula (from spec 14):
    lag = months(startdate - train_end) - 1

- lag=0: 1-month-ahead (most recent timeframe)
- lag=4: 5-month-ahead (oldest timeframe)

Execution lag (DFU property): stored in dim_sku.execution_lag (0-4).
Production-relevant prediction = where natural_lag == execution_lag.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_natural_lag(startdate: pd.Timestamp, train_end: pd.Timestamp) -> int:
    """Compute natural forecast lag from prediction date and training cutoff.

    lag = months(startdate - train_end) - 1
    lag=0 means 1-month-ahead (train_end is 1 month before startdate).
    """
    return (startdate.year - train_end.year) * 12 + (startdate.month - train_end.month) - 1


def add_lag_columns(
    predictions_df: pd.DataFrame,
    tf_train_end_map: dict[int, pd.Timestamp],
    exec_lag_map: dict[str, int],
) -> pd.DataFrame:
    """Add natural_lag and execution_lag columns to predictions DataFrame.

    Args:
        predictions_df: DataFrame with columns [sku_ck, startdate, timeframe_idx, ...].
        tf_train_end_map: {timeframe_index: train_end} mapping from generate_timeframes().
        exec_lag_map: {sku_ck: execution_lag} mapping from dim_sku.

    Returns:
        predictions_df with two new columns added in-place.
    """
    if predictions_df.empty:
        predictions_df["natural_lag"] = pd.Series(dtype=int)
        predictions_df["execution_lag"] = pd.Series(dtype=int)
        return predictions_df

    def _calc_natural_lag(row: pd.Series) -> int:
        te = tf_train_end_map.get(int(row["timeframe_idx"]))
        if te is None:
            return -1
        sd = pd.Timestamp(row["startdate"])
        return compute_natural_lag(sd, te)

    predictions_df = predictions_df.copy()
    predictions_df["natural_lag"] = predictions_df.apply(_calc_natural_lag, axis=1)
    predictions_df["execution_lag"] = (
        predictions_df["sku_ck"].map(exec_lag_map).fillna(0).astype(int)
    )

    logger.info(
        "Lag columns added. natural_lag distribution: %s",
        predictions_df["natural_lag"].value_counts().sort_index().to_dict(),
    )
    return predictions_df


def wape_accuracy(predictions: pd.Series, actuals: pd.Series) -> tuple[float, float]:
    """Compute WAPE and accuracy from aligned prediction/actual series.

    WAPE = SUM(|F-A|) / max(|SUM(A)|, 1.0) * 100
    Accuracy = 100 - WAPE

    Returns:
        (wape, accuracy) both as percentages.
    """
    abs_errors = (predictions - actuals).abs().sum()
    sum_actuals = actuals.sum()
    wape = float(abs_errors) / max(abs(float(sum_actuals)), 1.0) * 100.0
    return round(wape, 4), round(100.0 - wape, 4)


def compute_lag_accuracy(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute WAPE/accuracy for a set of predictions vs actuals.

    Args:
        predictions_df: DataFrame with [sku_ck, startdate, basefcst_pref].
        actuals_df: DataFrame with [sku_ck, startdate, qty].

    Returns:
        Dict with keys: wape, accuracy, n_rows, n_dfus.
    """
    if predictions_df.empty or actuals_df.empty:
        return {"wape": float("nan"), "accuracy": float("nan"), "n_rows": 0, "n_dfus": 0}

    merged = predictions_df[["sku_ck", "startdate", "basefcst_pref"]].merge(
        actuals_df[["sku_ck", "startdate", "qty"]].rename(columns={"qty": "actual"}),
        on=["sku_ck", "startdate"],
        how="inner",
    )
    if merged.empty:
        return {"wape": float("nan"), "accuracy": float("nan"), "n_rows": 0, "n_dfus": 0}

    wape, accuracy = wape_accuracy(merged["basefcst_pref"], merged["actual"])
    return {
        "wape": wape,
        "accuracy": accuracy,
        "n_rows": len(merged),
        "n_dfus": int(merged["sku_ck"].nunique()),
    }


def compute_execution_lag_accuracy(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Compute execution-lag-matched accuracy per algorithm.

    Filters to rows where natural_lag == execution_lag (production-relevant predictions).

    Args:
        predictions_df: DataFrame with [sku_ck, startdate, basefcst_pref,
                        algorithm_id, natural_lag, execution_lag].
        actuals_df: DataFrame with [sku_ck, startdate, qty].

    Returns:
        {algorithm_id: {wape, accuracy, n_rows, n_dfus}} sorted by accuracy desc.
    """
    if predictions_df.empty:
        return {}
    if "natural_lag" not in predictions_df.columns or "execution_lag" not in predictions_df.columns:
        logger.warning("natural_lag or execution_lag columns missing from predictions")
        return {}

    exec_matched = predictions_df[
        predictions_df["natural_lag"] == predictions_df["execution_lag"]
    ]
    if exec_matched.empty:
        logger.warning("No execution-lag-matched predictions found")
        return {}

    result: dict[str, dict[str, Any]] = {}
    for algo_id, algo_preds in exec_matched.groupby("algorithm_id"):
        result[str(algo_id)] = compute_lag_accuracy(algo_preds, actuals_df)

    logger.info(
        "Execution-lag accuracy: %d algorithms, %d total matched rows",
        len(result), exec_matched.shape[0],
    )
    return result


def compute_per_lag_breakdown(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
) -> dict[int, dict[str, dict[str, Any]]]:
    """Compute per-lag (0-4) accuracy for all algorithms at each forecast horizon.

    ALL DFUs are measured at each lag (not just DFUs whose execution_lag equals that lag).
    This gives a cross-sectional view of model accuracy by forecast horizon.

    Args:
        predictions_df: DataFrame with [sku_ck, startdate, basefcst_pref,
                        algorithm_id, natural_lag].
        actuals_df: DataFrame with [sku_ck, startdate, qty].

    Returns:
        {lag_value: {algorithm_id: {wape, accuracy, n_rows, n_dfus}}}
    """
    if predictions_df.empty or "natural_lag" not in predictions_df.columns:
        return {}

    result: dict[int, dict[str, dict[str, Any]]] = {}
    for lag_val in range(5):
        lag_preds = predictions_df[predictions_df["natural_lag"] == lag_val]
        if lag_preds.empty:
            continue
        lag_algos: dict[str, dict[str, Any]] = {}
        for algo_id, algo_preds in lag_preds.groupby("algorithm_id"):
            lag_algos[str(algo_id)] = compute_lag_accuracy(algo_preds, actuals_df)
        if lag_algos:
            result[lag_val] = lag_algos
            logger.info(
                "Lag %d (%d-month-ahead): %d algorithms, best=%s",
                lag_val, lag_val + 1, len(lag_algos),
                max(lag_algos.items(), key=lambda x: x[1].get("accuracy", 0)
                    if not np.isnan(x[1].get("accuracy", float("nan"))) else 0)[0],
            )
    return result


def compute_monthly_accuracy(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    execution_lag_only: bool = False,
    require_all_lags: bool = False,
) -> dict[str, dict[str, Any]]:
    """Compute per-calendar-month WAPE/accuracy for each algorithm.

    Groups predictions by (algorithm_id, startdate month) and computes WAPE
    for each (algorithm, month) pair. Raw error sums are stored alongside
    metrics so rolling windows can be aggregated properly (not averaging WAPEs).

    Args:
        predictions_df: DataFrame with [sku_ck, startdate, basefcst_pref, algorithm_id].
            When execution_lag_only=True, must also contain [natural_lag, execution_lag].
        actuals_df: DataFrame with [sku_ck, startdate, qty].
        execution_lag_only: If True, only include rows where natural_lag == execution_lag
            (production-relevant predictions matched to each DFU's execution horizon).
        require_all_lags: If True, only include calendar months where all 5 natural lag
            values (0-4) are present in the predictions — ensuring full coverage before
            including a month in the report.

    Returns:
        Chronologically-sorted dict:
        {month_str: {algorithm_id: {wape, accuracy, n_rows, n_dfus,
                                    sum_abs_error, sum_actual}}}
        where month_str is "YYYY-MM".
    """
    if predictions_df.empty or actuals_df.empty:
        return {}

    preds = predictions_df.copy()

    if execution_lag_only:
        if "natural_lag" not in preds.columns or "execution_lag" not in preds.columns:
            logger.warning(
                "execution_lag_only=True but lag columns missing; "
                "call add_lag_columns() first. Falling back to all predictions."
            )
        else:
            preds = preds[preds["natural_lag"] == preds["execution_lag"]]
            if preds.empty:
                logger.warning("No execution-lag-matched predictions found for monthly accuracy")
                return {}

    if require_all_lags and "natural_lag" in preds.columns:
        preds["_month_str"] = pd.to_datetime(preds["startdate"]).dt.strftime("%Y-%m")
        full_coverage_months = (
            preds.groupby("_month_str")["natural_lag"]
            .apply(lambda x: set(x.unique()) >= {0, 1, 2, 3, 4})
        )
        keep_months = full_coverage_months[full_coverage_months].index.tolist()
        preds = preds[preds["_month_str"].isin(keep_months)].drop(columns=["_month_str"])
        if preds.empty:
            logger.warning("No months with full lag coverage (0-4); skipping monthly accuracy")
            return {}
        logger.info(
            "require_all_lags: %d months with full lag coverage retained", len(keep_months)
        )
    elif "_month_str" in preds.columns:
        preds = preds.drop(columns=["_month_str"])

    cols = ["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
    merged = preds[cols].merge(
        actuals_df[["sku_ck", "startdate", "qty"]].rename(columns={"qty": "actual"}),
        on=["sku_ck", "startdate"],
        how="inner",
    )
    if merged.empty:
        return {}

    merged["month_str"] = pd.to_datetime(merged["startdate"]).dt.strftime("%Y-%m")

    result: dict[str, dict[str, Any]] = {}
    for month_str, month_df in merged.groupby("month_str"):
        algo_metrics: dict[str, Any] = {}
        for algo_id, algo_df in month_df.groupby("algorithm_id"):
            sum_abs_err = float((algo_df["basefcst_pref"] - algo_df["actual"]).abs().sum())
            sum_actual = float(algo_df["actual"].sum())
            wape = sum_abs_err / max(abs(sum_actual), 1.0) * 100.0
            algo_metrics[str(algo_id)] = {
                "wape": round(wape, 4),
                "accuracy": round(100.0 - wape, 4),
                "n_rows": len(algo_df),
                "n_dfus": int(algo_df["sku_ck"].nunique()),
                "sum_abs_error": sum_abs_err,
                "sum_actual": sum_actual,
            }
        if algo_metrics:
            result[str(month_str)] = algo_metrics

    result = dict(sorted(result.items()))
    logger.info(
        "Monthly accuracy computed: %d months, %d algorithms%s",
        len(result),
        len({a for m in result.values() for a in m}),
        " (execution-lag-matched)" if execution_lag_only else "",
    )
    return result


def compute_rolling_window_accuracy(
    monthly_accuracy: dict[str, dict[str, Any]],
    window_months: int,
) -> dict[str, Any]:
    """Aggregate per-algorithm accuracy over the last N calendar months.

    Computes two overall accuracy measures:
    - ``accuracy``: proper WAPE-based accuracy (sum abs_error / sum actual across window)
    - ``mean_monthly_accuracy``: simple unweighted mean of per-month accuracy values,
      giving equal weight to each month regardless of volume

    Args:
        monthly_accuracy: Output of compute_monthly_accuracy().
        window_months: Number of most-recent months to include.

    Returns:
        {algorithm_id: {wape, accuracy, mean_monthly_accuracy, n_rows,
                        n_months, months_included}}
        Returns {} if monthly_accuracy is empty.
    """
    if not monthly_accuracy:
        return {}

    sorted_months = sorted(monthly_accuracy.keys())
    recent_months = sorted_months[-window_months:]

    all_algos: set[str] = set()
    for m in recent_months:
        all_algos.update(monthly_accuracy[m].keys())

    result: dict[str, Any] = {}
    for algo_id in sorted(all_algos):
        total_abs_err = 0.0
        total_actual = 0.0
        total_rows = 0
        monthly_accs: list[float] = []
        months_with_data: list[str] = []

        for m in recent_months:
            algo_data = monthly_accuracy.get(m, {}).get(algo_id)
            if algo_data is not None:
                total_abs_err += algo_data["sum_abs_error"]
                total_actual += algo_data["sum_actual"]
                total_rows += algo_data["n_rows"]
                monthly_accs.append(algo_data["accuracy"])
                months_with_data.append(m)

        if months_with_data:
            wape = total_abs_err / max(abs(total_actual), 1.0) * 100.0
            mean_monthly = float(np.mean(monthly_accs))
            result[algo_id] = {
                "wape": round(wape, 4),
                "accuracy": round(100.0 - wape, 4),
                "mean_monthly_accuracy": round(mean_monthly, 4),
                "n_rows": total_rows,
                "n_months": len(months_with_data),
                "months_included": months_with_data,
            }

    return result


def compute_overall_monthly_accuracy(
    monthly_accuracy: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute overall accuracy across ALL available months.

    Equivalent to compute_rolling_window_accuracy with window = all months.
    Returns both proper WAPE-based accuracy and unweighted mean-of-months accuracy.

    Args:
        monthly_accuracy: Output of compute_monthly_accuracy().

    Returns:
        {algorithm_id: {wape, accuracy, mean_monthly_accuracy, n_rows,
                        n_months, months_included}}
    """
    return compute_rolling_window_accuracy(monthly_accuracy, len(monthly_accuracy))
