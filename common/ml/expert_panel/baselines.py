"""Baseline forecasting models: Seasonal Naive, Rolling Mean, Rolling Median, Ridge Regression.

Seasonal Naive, Rolling Mean, and Rolling Median work per-DFU on raw time series.
Ridge Regression uses the feature matrix (same features as tree models).
"""

import logging
from typing import Any

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from common.core.constants import FORECAST_QTY_COL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seasonal Naive
# ---------------------------------------------------------------------------


def predict_seasonal_naive(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
) -> pd.DataFrame:
    """Seasonal Naive: forecast = same month last year.

    Args:
        sales_df: Training data with columns [sku_ck, startdate, qty].
        predict_months: Months to predict.

    Returns:
        DataFrame with columns: sku_ck, startdate, basefcst_pref, algorithm_id
        algorithm_id = 'seasonal_naive'
    """
    records: list[dict[str, Any]] = []

    if sales_df.empty:
        for m in predict_months:
            records.append(
                {"sku_ck": None, "startdate": m, FORECAST_QTY_COL: 0.0}
            )
        out = pd.DataFrame(records)
        out["algorithm_id"] = "seasonal_naive"
        return out

    sales = sales_df.copy()
    sales["startdate"] = pd.to_datetime(sales["startdate"])

    # Pivot to wide format: rows = sku_ck, columns = startdate, values = qty
    wide = sales.pivot_table(
        index="sku_ck", columns="startdate", values="qty", aggfunc="sum"
    )

    # Pre-compute per-DFU mean as fallback
    dfu_means: pd.Series = wide.mean(axis=1)

    for m in predict_months:
        m_ts = pd.Timestamp(m)
        lookback = m_ts - pd.DateOffset(months=12)

        if lookback in wide.columns:
            forecast_vals = wide[lookback].copy()
            # Fill DFUs missing at M-12 with their overall mean
            forecast_vals = forecast_vals.fillna(dfu_means)
        else:
            # No matching month last year for any DFU -> use mean
            forecast_vals = dfu_means.copy()

        forecast_vals = forecast_vals.fillna(0.0)
        month_df = pd.DataFrame(
            {"sku_ck": forecast_vals.index, "startdate": m_ts, FORECAST_QTY_COL: forecast_vals.values}
        )
        records.append(month_df)

    result = pd.concat(records, ignore_index=True)
    result[FORECAST_QTY_COL] = np.maximum(result[FORECAST_QTY_COL].values, 0.0)
    result["algorithm_id"] = "seasonal_naive"

    logger.info(
        "Seasonal Naive: produced %d predictions for %d months",
        len(result),
        len(predict_months),
    )
    return result


# ---------------------------------------------------------------------------
# Rolling Mean
# ---------------------------------------------------------------------------


def predict_rolling_mean(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    window: int = 6,
) -> pd.DataFrame:
    """Rolling Mean: forecast = mean of last ``window`` months.

    Args:
        sales_df: Training data with columns [sku_ck, startdate, qty].
        predict_months: Months to predict.
        window: Number of trailing months to average.

    Returns:
        DataFrame with columns: sku_ck, startdate, basefcst_pref, algorithm_id
        algorithm_id = 'rolling_mean'
    """
    if sales_df.empty:
        out = pd.DataFrame(
            {"sku_ck": pd.Series(dtype="object"), "startdate": pd.Series(dtype="datetime64[ns]"),
             FORECAST_QTY_COL: pd.Series(dtype="float64")}
        )
        out["algorithm_id"] = "rolling_mean"
        return out

    sales = sales_df.copy()
    sales["startdate"] = pd.to_datetime(sales["startdate"])

    # Sort descending by date so head(window) grabs the most recent months
    sales_sorted = sales.sort_values("startdate", ascending=False)

    # For each DFU, take last `window` months and compute mean
    rolling_means = (
        sales_sorted
        .groupby("sku_ck")
        .apply(lambda g: g.head(window)["qty"].mean(), include_groups=False)
        .reset_index()
        .rename(columns={0: FORECAST_QTY_COL})
    )
    rolling_means[FORECAST_QTY_COL] = rolling_means[FORECAST_QTY_COL].fillna(0.0)

    # Flat forecast: same value for every predict month
    frames: list[pd.DataFrame] = []
    for m in predict_months:
        month_df = rolling_means.copy()
        month_df["startdate"] = pd.Timestamp(m)
        frames.append(month_df)

    result = pd.concat(frames, ignore_index=True)
    result[FORECAST_QTY_COL] = np.maximum(result[FORECAST_QTY_COL].values, 0.0)
    result["algorithm_id"] = "rolling_mean"

    logger.info(
        "Rolling Mean (window=%d): produced %d predictions for %d months",
        window,
        len(result),
        len(predict_months),
    )
    return result


# ---------------------------------------------------------------------------
# Rolling Median
# ---------------------------------------------------------------------------


def predict_rolling_median(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    window: int = 6,
) -> pd.DataFrame:
    """Rolling Median: forecast = median of last ``window`` months.

    The outlier-robust sibling of :func:`predict_rolling_mean`: a single spike
    month (F-01) or a stale-high tail (F-07 level step) shifts the median far
    less than the mean, so this baseline tracks the typical level rather than
    chasing extremes.

    Args:
        sales_df: Training data with columns [sku_ck, startdate, qty].
        predict_months: Months to predict.
        window: Number of trailing months to take the median over.

    Returns:
        DataFrame with columns: sku_ck, startdate, basefcst_pref, algorithm_id
        algorithm_id = 'rolling_median'
    """
    if sales_df.empty:
        out = pd.DataFrame(
            {"sku_ck": pd.Series(dtype="object"), "startdate": pd.Series(dtype="datetime64[ns]"),
             FORECAST_QTY_COL: pd.Series(dtype="float64")}
        )
        out["algorithm_id"] = "rolling_median"
        return out

    sales = sales_df.copy()
    sales["startdate"] = pd.to_datetime(sales["startdate"])

    # Sort descending by date so head(window) grabs the most recent months
    sales_sorted = sales.sort_values("startdate", ascending=False)

    # For each DFU, take last `window` months and compute median
    rolling_medians = (
        sales_sorted
        .groupby("sku_ck")
        .apply(lambda g: g.head(window)["qty"].median(), include_groups=False)
        .reset_index()
        .rename(columns={0: FORECAST_QTY_COL})
    )
    rolling_medians[FORECAST_QTY_COL] = rolling_medians[FORECAST_QTY_COL].fillna(0.0)

    # Flat forecast: same value for every predict month
    frames: list[pd.DataFrame] = []
    for m in predict_months:
        month_df = rolling_medians.copy()
        month_df["startdate"] = pd.Timestamp(m)
        frames.append(month_df)

    result = pd.concat(frames, ignore_index=True)
    result[FORECAST_QTY_COL] = np.maximum(result[FORECAST_QTY_COL].values, 0.0)
    result["algorithm_id"] = "rolling_median"

    logger.info(
        "Rolling Median (window=%d): produced %d predictions for %d months",
        window,
        len(result),
        len(predict_months),
    )
    return result


# ---------------------------------------------------------------------------
# Ridge Regression
# ---------------------------------------------------------------------------


def _label_encode_categoricals(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label-encode categorical columns consistently across train and predict.

    Uses ``pd.factorize`` on the combined (train + predict) values so that
    both sets share the same integer mapping.

    Returns:
        Tuple of (encoded_train, encoded_predict) DataFrames.
    """
    train_out = train_df.copy()
    predict_out = predict_df.copy()

    for col in cat_cols:
        if col not in train_out.columns:
            continue
        combined = pd.concat(
            [train_out[col], predict_out[col]], ignore_index=True
        )
        codes, _ = pd.factorize(combined)
        n_train = len(train_out)
        train_out[col] = codes[:n_train]
        predict_out[col] = codes[n_train:]

    return train_out, predict_out


def predict_ridge(
    train_grid: pd.DataFrame,
    predict_grid: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    alpha: float = 100.0,
) -> pd.DataFrame:
    """Ridge Regression using the same feature matrix as tree models.

    Trains one Ridge model per ``ml_cluster`` (matching tree model strategy).
    Uses a ``RobustScaler → Ridge`` pipeline to prevent numerical instability
    caused by supply-chain features spanning very different magnitudes (e.g.
    0/1 flags vs. large quantity values).  RobustScaler is preferred over
    StandardScaler because supply-chain data frequently contains outliers.

    Args:
        train_grid: Training feature matrix with 'qty' target and 'ml_cluster' column.
        predict_grid: Prediction feature matrix (same features, qty masked to 0).
        feature_cols: List of feature column names.
        cat_cols: Categorical column names (will be label-encoded for Ridge).
        alpha: Ridge regularization strength. Defaults to 100.0 — a higher value
            is needed because RobustScaler normalises feature scale but the
            regularisation must still dominate when the condition number is large.

    Returns:
        DataFrame with columns: sku_ck, startdate, basefcst_pref, algorithm_id
        algorithm_id = 'ridge'
    """
    all_preds: list[pd.DataFrame] = []

    clusters = sorted(
        set(train_grid["ml_cluster"].unique()) | set(predict_grid["ml_cluster"].unique())
    )

    for cluster in clusters:
        train_mask = train_grid["ml_cluster"] == cluster
        pred_mask = predict_grid["ml_cluster"] == cluster

        train_c = train_grid.loc[train_mask].copy()
        pred_c = predict_grid.loc[pred_mask].copy()

        if train_c.empty:
            logger.warning(
                "Ridge: cluster %s has no training data, skipping", cluster
            )
            continue

        if pred_c.empty:
            logger.warning(
                "Ridge: cluster %s has no prediction rows, skipping", cluster
            )
            continue

        # Ensure feature columns exist; fill missing with 0
        for col in feature_cols:
            if col not in train_c.columns:
                train_c[col] = 0.0
            if col not in pred_c.columns:
                pred_c[col] = 0.0

        # Label-encode ALL categorical/object/string columns in features
        # (cat_cols from caller + any extra object/category dtype columns)
        obj_cols = set(
            train_c[feature_cols]
            .select_dtypes(include=["object", "string", "category"])
            .columns
        )
        effective_cat_cols = list(
            {c for c in cat_cols if c in feature_cols} | obj_cols
        )
        train_enc, pred_enc = _label_encode_categoricals(
            train_c, pred_c, effective_cat_cols
        )

        X_train = train_enc[feature_cols].values.astype(np.float64)
        y_train = train_enc["qty"].values.astype(np.float64)
        X_pred = pred_enc[feature_cols].values.astype(np.float64)

        # Replace NaN/inf with 0 before feeding into the scaler.
        # Supply-chain features can contain missing lag values that are
        # zero-filled rather than imputed; large sentinel values (e.g. 1e9)
        # are clipped to avoid poisoning the scaler's median/IQR estimates.
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)

        # Drop constant columns (IQR=0) to prevent RobustScaler from
        # producing inf, which causes overflow in the Ridge matmul.
        col_std = X_train.std(axis=0)
        active_mask = col_std > 0
        if not active_mask.all():
            X_train = X_train[:, active_mask]
            X_pred = X_pred[:, active_mask]

        try:
            # RobustScaler centres on median and scales by IQR, making it
            # resistant to the large quantity outliers typical in supply-chain
            # data.  The sanitize step clips any inf/nan that RobustScaler
            # produces when a feature column has IQR=0 (zero-variance column
            # within a cluster), preventing matmul overflow in Ridge.predict.
            # solver='lsqr' uses iterative least-squares instead of the
            # Cholesky direct solve, avoiding LinAlgWarning on ill-conditioned
            # matrices while remaining fast for this data size.
            model: Pipeline = Pipeline(
                [
                    ("scaler", RobustScaler()),
                    ("sanitize", FunctionTransformer(
                        lambda X: np.clip(
                            np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0),
                            -1e6, 1e6,
                        )
                    )),
                    ("ridge", Ridge(alpha=alpha, solver="lsqr")),
                ]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                model.fit(X_train, y_train)
                preds: np.ndarray = model.predict(X_pred)
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.warning(
                "Ridge: failed to fit cluster %s: %s", cluster, exc
            )
            preds = np.zeros(len(pred_c))

        # Guard against any NaN/inf that the solver may still produce (e.g.
        # degenerate clusters with zero-variance features) before clipping
        # negatives.  Use a finite large bound so outlier predictions are
        # capped rather than silently propagated downstream.
        preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
        preds = np.clip(preds, 0.0, 1e9)

        cluster_result = pred_c[["sku_ck", "startdate"]].copy()
        cluster_result[FORECAST_QTY_COL] = preds
        all_preds.append(cluster_result)

    if all_preds:
        result = pd.concat(all_preds, ignore_index=True)
    else:
        result = pd.DataFrame(
            {"sku_ck": pd.Series(dtype="object"),
             "startdate": pd.Series(dtype="datetime64[ns]"),
             FORECAST_QTY_COL: pd.Series(dtype="float64")}
        )

    result["algorithm_id"] = "ridge"

    logger.info(
        "Ridge (alpha=%.2f): produced %d predictions across %d clusters",
        alpha,
        len(result),
        len(clusters),
    )
    return result
