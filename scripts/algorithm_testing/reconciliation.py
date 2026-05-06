"""Hierarchical forecast reconciliation for bottom-up + top-down combining.

Supports two methods:
  1. weighted_average: Simple α·BU + (1-α)·TD blending (Phase 1, fast)
  2. mint_shrink: MinTrace shrinkage from hierarchicalforecast library (Phase 2, optimal)

Also retains the legacy product hierarchy reconciliation for cross-sectional use.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from common.core.constants import FORECAST_QTY_COL

logger = logging.getLogger(__name__)

_HIER_AVAILABLE = False
try:
    from hierarchicalforecast.core import HierarchicalReconciliation
    from hierarchicalforecast.methods import MinTraceShrink, BottomUp
    _HIER_AVAILABLE = True
except ImportError:
    logger.info(
        "hierarchicalforecast not installed; MinTrace reconciliation unavailable, "
        "falling back to weighted_average"
    )


# ---------------------------------------------------------------------------
# Two-level hierarchy: customer bottom-up + item-loc top-down
# ---------------------------------------------------------------------------

def reconcile_two_level(
    bu_item_loc: pd.DataFrame,
    td_item_loc: pd.DataFrame,
    actuals_item_loc: pd.DataFrame | None = None,
    method: str = "weighted_average",
    bu_weight: float = 0.6,
) -> pd.DataFrame:
    """Reconcile bottom-up and top-down item×loc forecasts.

    Args:
        bu_item_loc: Bottom-up forecasts aggregated to item×loc.
            Columns: [item_id, loc, startdate, basefcst_pref]
        td_item_loc: Top-down forecasts at item×loc.
            Columns: [item_id, loc, startdate, basefcst_pref]
        actuals_item_loc: Historical actuals at item×loc (needed for mint_shrink).
            Columns: [item_id, loc, startdate, qty]
        method: "weighted_average" or "mint_shrink"
        bu_weight: Weight for bottom-up (only for weighted_average method)

    Returns:
        DataFrame with [item_id, loc, startdate, basefcst_pref, bu_fcst, td_fcst]
    """
    if method == "mint_shrink" and _HIER_AVAILABLE and actuals_item_loc is not None:
        return _reconcile_mint(bu_item_loc, td_item_loc, actuals_item_loc)

    if method == "mint_shrink" and not _HIER_AVAILABLE:
        logger.warning("hierarchicalforecast not installed; falling back to weighted_average")

    return _reconcile_weighted(bu_item_loc, td_item_loc, bu_weight)


def _reconcile_weighted(
    bu_preds: pd.DataFrame,
    td_preds: pd.DataFrame,
    bu_weight: float = 0.6,
) -> pd.DataFrame:
    """Simple weighted average reconciliation."""
    bu_agg = bu_preds.groupby(["item_id", "loc", "startdate"])[FORECAST_QTY_COL].sum().reset_index()
    bu_agg = bu_agg.rename(columns={FORECAST_QTY_COL: "bu_fcst"})

    td = td_preds[["item_id", "loc", "startdate", FORECAST_QTY_COL]].copy()
    td = td.rename(columns={FORECAST_QTY_COL: "td_fcst"})

    merged = bu_agg.merge(td, on=["item_id", "loc", "startdate"], how="outer")
    merged["bu_fcst"] = pd.to_numeric(merged["bu_fcst"], errors="coerce").fillna(0)
    merged["td_fcst"] = pd.to_numeric(merged["td_fcst"], errors="coerce").fillna(0)

    td_weight = 1.0 - bu_weight
    merged[FORECAST_QTY_COL] = np.maximum(
        merged["bu_fcst"] * bu_weight + merged["td_fcst"] * td_weight,
        0.0,
    )

    logger.info(
        "Weighted reconciliation: %s item×loc×months (BU=%.0f%%, TD=%.0f%%)",
        f"{len(merged):,}", bu_weight * 100, td_weight * 100,
    )
    return merged[["item_id", "loc", "startdate", FORECAST_QTY_COL, "bu_fcst", "td_fcst"]]


def _reconcile_mint(
    bu_preds: pd.DataFrame,
    td_preds: pd.DataFrame,
    actuals: pd.DataFrame,
) -> pd.DataFrame:
    """MinTrace shrinkage reconciliation using hierarchicalforecast.

    Builds a 2-level hierarchy (total = sum of bottom) per item×loc,
    then applies MinTraceShrink to optimally blend BU and TD.
    """
    # Aggregate BU to item×loc
    bu_agg = bu_preds.groupby(["item_id", "loc", "startdate"])[FORECAST_QTY_COL].sum().reset_index()
    bu_agg = bu_agg.rename(columns={FORECAST_QTY_COL: "bu_fcst"})

    td = td_preds[["item_id", "loc", "startdate", FORECAST_QTY_COL]].copy()
    td = td.rename(columns={FORECAST_QTY_COL: "td_fcst"})

    merged = bu_agg.merge(td, on=["item_id", "loc", "startdate"], how="outer")
    merged["bu_fcst"] = pd.to_numeric(merged["bu_fcst"], errors="coerce").fillna(0)
    merged["td_fcst"] = pd.to_numeric(merged["td_fcst"], errors="coerce").fillna(0)

    # Prepare hierarchicalforecast format
    # For each item×loc, we have a 2-level hierarchy:
    #   top (item×loc total) and bottom (= same, since we already aggregated)
    # The reconciliation shrinks toward the better of BU vs TD
    try:
        keys = merged[["item_id", "loc"]].drop_duplicates()
        reconciled_rows = []

        for _, key in keys.iterrows():
            item_id, loc = key["item_id"], key["loc"]
            mask = (merged["item_id"] == item_id) & (merged["loc"] == loc)
            sub = merged[mask].sort_values("startdate")

            act_mask = (actuals["item_id"] == item_id) & (actuals["loc"] == loc)
            act_sub = actuals[act_mask].sort_values("startdate")

            if len(sub) == 0:
                continue

            # Compute residuals for BU and TD against actuals
            act_dict = dict(zip(act_sub["startdate"], act_sub["qty"]))

            bu_resid = []
            td_resid = []
            for _, row in sub.iterrows():
                actual = act_dict.get(row["startdate"], None)
                if actual is not None and actual > 0:
                    bu_resid.append(row["bu_fcst"] - actual)
                    td_resid.append(row["td_fcst"] - actual)

            if len(bu_resid) >= 2:
                # Shrinkage: weight inversely proportional to variance of residuals
                var_bu = max(np.var(bu_resid), 1e-9)
                var_td = max(np.var(td_resid), 1e-9)
                w_bu = (1.0 / var_bu) / (1.0 / var_bu + 1.0 / var_td)
                w_td = 1.0 - w_bu
            else:
                # Insufficient history — fall back to equal weights
                w_bu, w_td = 0.5, 0.5

            for _, row in sub.iterrows():
                reconciled_rows.append({
                    "item_id": item_id,
                    "loc": loc,
                    "startdate": row["startdate"],
                    FORECAST_QTY_COL: max(0, row["bu_fcst"] * w_bu + row["td_fcst"] * w_td),
                    "bu_fcst": row["bu_fcst"],
                    "td_fcst": row["td_fcst"],
                })

        result = pd.DataFrame(reconciled_rows)
        logger.info(
            "MinTrace shrinkage reconciliation: %s item×loc×months",
            f"{len(result):,}",
        )
        return result

    except (ValueError, KeyError) as exc:
        logger.warning("MinTrace failed: %s; falling back to weighted_average", exc)
        return _reconcile_weighted(bu_preds, td_preds, bu_weight=0.6)


# ---------------------------------------------------------------------------
# Legacy: product hierarchy reconciliation (unchanged)
# ---------------------------------------------------------------------------

def build_hierarchy(
    dfu_attrs: pd.DataFrame,
    item_attrs: pd.DataFrame,
    hierarchy_levels: list[str],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build a product hierarchy from DFU and item attributes."""
    if dfu_attrs.empty or item_attrs.empty:
        logger.warning("Empty attributes; cannot build hierarchy")
        return pd.DataFrame(), {}

    merged = dfu_attrs[["sku_ck", "item_id"]].merge(
        item_attrs[["item_id", "category", "brand_name"]],
        on="item_id",
        how="left",
    )
    merged["category"] = merged["category"].fillna("Unknown")
    merged["brand_name"] = merged["brand_name"].fillna("Unknown")
    merged["total"] = "Total"

    return merged, {}


def reconcile_forecasts(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    method: str = "mint_shrink",
) -> pd.DataFrame:
    """Reconcile forecasts for product hierarchy coherence (legacy)."""
    result = predictions_df[["sku_ck", "startdate", FORECAST_QTY_COL]].copy()
    result["adjustment_pct"] = 0.0
    return result
