"""Canonical calendar completion shared by forecasting model adapters."""

from __future__ import annotations

import pandas as pd


def select_bounded_history(
    sales_df: pd.DataFrame,
    *,
    history_end: pd.Timestamp,
    lookback_months: int,
) -> pd.DataFrame:
    """Return the exact inclusive monthly context window ending at a cutoff."""
    if lookback_months <= 0:
        raise ValueError("lookback_months must be positive")
    if "startdate" not in sales_df.columns:
        raise ValueError("Monthly sales input is missing column: startdate")
    normalized_end = pd.Timestamp(history_end).to_period("M").to_timestamp()
    history_start = normalized_end - pd.DateOffset(months=lookback_months - 1)
    bounded = sales_df.copy()
    bounded["startdate"] = (
        pd.to_datetime(bounded["startdate"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    if bounded["startdate"].isna().any():
        raise ValueError("Monthly sales input contains an invalid startdate")
    bounded = bounded[
        (bounded["startdate"] >= history_start)
        & (bounded["startdate"] <= normalized_end)
    ].copy()
    bounded.attrs = dict(sales_df.attrs)
    bounded.attrs["history_start"] = history_start
    bounded.attrs["history_end"] = normalized_end
    return bounded


def complete_monthly_history(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Densify each DFU from its first observation through the declared history end."""
    required = {"sku_ck", "startdate", "qty"}
    missing = required - set(sales_df.columns)
    if missing:
        raise ValueError(f"Monthly sales input is missing columns: {sorted(missing)}")
    if sales_df.empty:
        return pd.DataFrame(columns=["sku_ck", "startdate", "qty"])

    sales = sales_df[["sku_ck", "startdate", "qty"]].copy()
    sales["sku_ck"] = sales["sku_ck"].astype(str)
    sales["startdate"] = (
        pd.to_datetime(sales["startdate"]).dt.to_period("M").dt.to_timestamp()
    )
    sales["qty"] = pd.to_numeric(sales["qty"], errors="coerce").fillna(0.0)
    sales = sales.groupby(["sku_ck", "startdate"], as_index=False)["qty"].sum()

    history_end = pd.Timestamp(
        sales_df.attrs.get("history_end", sales["startdate"].max())
    ).to_period("M").to_timestamp()
    observed_end = sales["startdate"].max()
    if observed_end > history_end:
        raise ValueError(
            "Monthly sales input contains observations after the configured closed month"
        )

    observed_first = sales.groupby("sku_ck", as_index=False)["startdate"].min()
    if "first_sale_month" in sales_df.columns:
        declared_first = sales_df[["sku_ck", "first_sale_month"]].copy()
        declared_first["sku_ck"] = declared_first["sku_ck"].astype(str)
        declared_first["first_sale_month"] = (
            pd.to_datetime(declared_first["first_sale_month"], errors="coerce")
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        missing_first_sale = (
            declared_first.groupby("sku_ck")["first_sale_month"].count() == 0
        )
        if missing_first_sale.any():
            sample = ", ".join(
                missing_first_sale.index[missing_first_sale].astype(str)[:5]
            )
            raise ValueError(
                "Monthly target sku_ck(s) have no observed first sale: " f"{sample}"
            )
        first_month = (
            declared_first.groupby("sku_ck", as_index=False)["first_sale_month"]
            .min()
            .rename(columns={"first_sale_month": "startdate"})
        )
        first_month = first_month.merge(
            observed_first,
            on="sku_ck",
            how="right",
            suffixes=("_declared", "_observed"),
            validate="one_to_one",
        )
        first_month["startdate"] = first_month["startdate_declared"].combine_first(
            first_month["startdate_observed"]
        )
        first_month["startdate"] = first_month["startdate"].clip(
            lower=pd.Timestamp(sales["startdate"].min())
        )
        first_month = first_month[["sku_ck", "startdate"]]
    else:
        first_month = observed_first

    calendar = pd.date_range(first_month["startdate"].min(), history_end, freq="MS")
    grid = pd.MultiIndex.from_product(
        [first_month["sku_ck"], calendar], names=["sku_ck", "startdate"]
    ).to_frame(index=False)
    grid = grid.merge(
        first_month.rename(columns={"startdate": "first_month"}),
        on="sku_ck",
        how="left",
        validate="many_to_one",
    )
    grid = grid[grid["startdate"] >= grid["first_month"]].drop(columns="first_month")
    grid = grid.merge(
        sales,
        on=["sku_ck", "startdate"],
        how="left",
        validate="one_to_one",
    )
    grid["qty"] = grid["qty"].fillna(0.0).astype(float)
    result = grid.sort_values(["sku_ck", "startdate"]).reset_index(drop=True)
    result.attrs["history_end"] = history_end
    return result
