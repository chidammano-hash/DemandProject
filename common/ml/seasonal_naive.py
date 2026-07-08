"""Seasonal-naive model artifact used for sparse demand routing."""

from __future__ import annotations

import numpy as np
import pandas as pd

SEASONAL_DFU_KEY_COL = "_seasonal_dfu_key"


def make_dfu_key(item_id: object, customer_group: object, loc: object) -> str:
    """Build the full forecasting grain key used by seasonal-naive artifacts."""
    return f"{item_id}||{customer_group}||{loc}"


def add_dfu_key(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the seasonal-naive DFU key column populated."""
    out = df.copy()
    if SEASONAL_DFU_KEY_COL in out.columns:
        out[SEASONAL_DFU_KEY_COL] = out[SEASONAL_DFU_KEY_COL].astype(str)
        return out
    if {"item_id", "customer_group", "loc"}.issubset(out.columns):
        out[SEASONAL_DFU_KEY_COL] = [
            make_dfu_key(row.item_id, row.customer_group, row.loc)
            for row in out.itertuples(index=False)
        ]
    elif "sku_ck" in out.columns:
        out[SEASONAL_DFU_KEY_COL] = out["sku_ck"].astype(str)
    else:
        out[SEASONAL_DFU_KEY_COL] = "__unknown__"
    return out


class SeasonalNaiveModel:
    """Seasonal naive model compatible with the tree artifact predict contract.

    It stores the most recent historical value for each DFU/calendar-month pair
    and falls back to a trailing per-DFU mean when that pair has no history.
    Batch inference sets ``_sku_cks`` and ``_months`` immediately before calling
    ``predict`` because the model feature matrix does not include metadata.
    """

    def __init__(
        self,
        seasonal_map: dict[tuple[str, int], float],
        fallback_means: dict[str, float],
    ):
        self._seasonal_map = seasonal_map
        self._fallback_means = fallback_means
        self._sku_cks: list[str] | None = None
        self._months: list[int] | None = None

    def predict(self, X) -> np.ndarray:
        n_rows = len(X)
        if isinstance(X, pd.DataFrame) and "sku_ck" in X.columns and "startdate" in X.columns:
            keys = X["sku_ck"].astype(str).tolist()
            months = pd.to_datetime(X["startdate"]).dt.month.tolist()
        elif self._sku_cks is not None and self._months is not None:
            keys = self._sku_cks
            months = self._months
        elif self._sku_cks is not None:
            values = [self._fallback_means.get(key, 0.0) for key in self._sku_cks]
            return np.maximum(np.asarray(values, dtype=float), 0.0)
        else:
            return np.zeros(n_rows)

        values = [
            self._seasonal_map.get((key, month), self._fallback_means.get(key, 0.0))
            for key, month in zip(keys, months, strict=False)
        ]
        return np.maximum(np.asarray(values, dtype=float), 0.0)


def build_seasonal_naive_model(
    train_df: pd.DataFrame,
    *,
    window: int = 12,
) -> SeasonalNaiveModel:
    """Build a seasonal-naive artifact from historical demand rows."""
    if train_df.empty or "qty" not in train_df.columns:
        return SeasonalNaiveModel({}, {})

    train = add_dfu_key(train_df)
    train["_month"] = pd.to_datetime(train["startdate"]).dt.month
    train["_year"] = pd.to_datetime(train["startdate"]).dt.year
    train["_qty_non_negative"] = np.maximum(
        pd.to_numeric(train["qty"], errors="coerce").fillna(0.0).astype(float),
        0.0,
    )

    train_sorted_by_year = train.sort_values("_year", ascending=False)
    latest_by_key_month = (
        train_sorted_by_year
        .groupby([SEASONAL_DFU_KEY_COL, "_month"], sort=False)["_qty_non_negative"]
        .first()
    )
    seasonal_map = {
        (str(key), int(month)): float(value)
        for (key, month), value in latest_by_key_month.items()
    }

    train_sorted = train.sort_values("startdate", ascending=False)
    fallback_means = (
        train_sorted
        .groupby(SEASONAL_DFU_KEY_COL, sort=False)
        .apply(lambda grp: grp.head(window)["_qty_non_negative"].mean(), include_groups=False)
    )
    fallback_map = {
        str(key): float(value) if pd.notna(value) else 0.0
        for key, value in fallback_means.items()
    }
    return SeasonalNaiveModel(seasonal_map, fallback_map)
