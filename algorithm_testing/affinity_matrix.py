"""Affinity matrix builder: segment x algorithm WAPE matrix.

Computes per-segment, per-algorithm accuracy from backtest predictions,
producing the core decision matrix for the portfolio optimizer.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_affinity_matrix(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    classification_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the segment x algorithm WAPE matrix.

    Args:
        predictions_df: All predictions from all algorithms, all timeframes.
            Columns: sku_ck, startdate, basefcst_pref, algorithm_id
            May have multiple rows per (sku_ck, startdate) if from different timeframes.
        actuals_df: Actual demand values.
            Columns: sku_ck, startdate, qty (actual demand).
        classification_df: DFU demand archetype classification.
            Columns: sku_ck, archetype

    Returns:
        (affinity_matrix, detail_df)

        affinity_matrix: DataFrame with index=archetype, columns=algorithm_id,
            values=accuracy_pct (100 - WAPE). NaN where no data.
        detail_df: Per-(archetype, algorithm_id) with columns:
            archetype, algorithm_id, wape, accuracy_pct, bias, n_dfu_months, n_dfus,
            mean_abs_error, mean_actual
    """
    # --- Validate inputs -------------------------------------------------------
    for name, df, required in [
        ("predictions_df", predictions_df, {"sku_ck", "startdate", "basefcst_pref", "algorithm_id"}),
        ("actuals_df", actuals_df, {"sku_ck", "startdate", "qty"}),
        ("classification_df", classification_df, {"sku_ck", "archetype"}),
    ]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")

    if predictions_df.empty:
        logger.warning("Empty predictions_df; returning empty affinity matrix")
        return _empty_affinity_matrix(), _empty_detail_df()

    if actuals_df.empty:
        logger.warning("Empty actuals_df; returning empty affinity matrix")
        return _empty_affinity_matrix(), _empty_detail_df()

    # --- Step 1: Aggregate duplicate predictions (multiple timeframes) ---------
    preds = (
        predictions_df.groupby(["sku_ck", "startdate", "algorithm_id"], sort=False)["basefcst_pref"]
        .mean()
        .reset_index()
    )
    logger.info(
        "Predictions: %d rows after averaging duplicates (from %d raw rows)",
        len(preds),
        len(predictions_df),
    )

    # --- Step 2: Inner join with actuals ---------------------------------------
    merged = preds.merge(
        actuals_df[["sku_ck", "startdate", "qty"]],
        on=["sku_ck", "startdate"],
        how="inner",
    )
    if merged.empty:
        logger.warning("No matching (sku_ck, startdate) between predictions and actuals")
        return _empty_affinity_matrix(), _empty_detail_df()

    logger.info(
        "Matched %d prediction-actual pairs (%d DFUs, %d algorithms)",
        len(merged),
        merged["sku_ck"].nunique(),
        merged["algorithm_id"].nunique(),
    )

    # --- Step 3: Attach archetype via classification ---------------------------
    merged = merged.merge(
        classification_df[["sku_ck", "archetype"]],
        on="sku_ck",
        how="inner",
    )
    if merged.empty:
        logger.warning("No DFUs matched between predictions and classification")
        return _empty_affinity_matrix(), _empty_detail_df()

    # --- Step 4: Compute per-row absolute error --------------------------------
    merged["abs_error"] = np.abs(merged["basefcst_pref"] - merged["qty"])

    # --- Step 5: Group by (archetype, algorithm_id) and aggregate --------------
    grouped = merged.groupby(["archetype", "algorithm_id"], sort=False)

    detail = grouped.agg(
        sum_abs_error=("abs_error", "sum"),
        sum_actual=("qty", "sum"),
        sum_forecast=("basefcst_pref", "sum"),
        n_dfu_months=("abs_error", "count"),
        n_dfus=("sku_ck", "nunique"),
        mean_abs_error=("abs_error", "mean"),
        mean_actual=("qty", "mean"),
    ).reset_index()

    # WAPE = sum_abs_error / max(|sum_actual|, 1.0) * 100
    safe_denom = np.maximum(np.abs(detail["sum_actual"]), 1.0)
    detail["wape"] = detail["sum_abs_error"] / safe_denom * 100

    # Accuracy = max(100 - WAPE, 0)
    detail["accuracy_pct"] = np.maximum(100 - detail["wape"], 0.0)

    # Bias = (sum_forecast / max(sum_actual, 1.0)) - 1
    safe_actual = np.maximum(detail["sum_actual"], 1.0)
    detail["bias"] = (detail["sum_forecast"] / safe_actual) - 1

    # Select and order final columns
    detail_df = detail[
        [
            "archetype",
            "algorithm_id",
            "wape",
            "accuracy_pct",
            "bias",
            "n_dfu_months",
            "n_dfus",
            "mean_abs_error",
            "mean_actual",
        ]
    ].copy()

    # --- Step 6: Pivot to matrix -----------------------------------------------
    affinity_matrix = detail_df.pivot(
        index="archetype",
        columns="algorithm_id",
        values="accuracy_pct",
    )

    # --- Step 7: Log summary ---------------------------------------------------
    logger.info(
        "Affinity matrix: %d archetypes x %d algorithms",
        affinity_matrix.shape[0],
        affinity_matrix.shape[1],
    )
    for archetype in affinity_matrix.index:
        row = affinity_matrix.loc[archetype]
        valid = row.dropna()
        if not valid.empty:
            best_algo = valid.idxmax()
            best_acc = valid.max()
            logger.info(
                "  %s: best = %s (%.1f%% accuracy)",
                archetype,
                best_algo,
                best_acc,
            )

    return affinity_matrix, detail_df


def get_best_per_segment(affinity_matrix: pd.DataFrame) -> pd.DataFrame:
    """For each segment, find the best algorithm.

    Args:
        affinity_matrix: Output of build_affinity_matrix()[0].
            Index=archetype, columns=algorithm_id, values=accuracy_pct.

    Returns:
        DataFrame with columns: archetype, best_algorithm, accuracy_pct, margin_bps
        margin_bps = gap between best and second-best in basis points.
    """
    if affinity_matrix.empty:
        logger.warning("Empty affinity matrix; returning empty best-per-segment")
        return pd.DataFrame(
            columns=["archetype", "best_algorithm", "accuracy_pct", "margin_bps"]
        )

    rows: list[dict[str, object]] = []

    for archetype in affinity_matrix.index:
        row = affinity_matrix.loc[archetype].dropna().sort_values(ascending=False)

        if row.empty:
            continue

        best_algo = str(row.index[0])
        best_acc = float(row.iloc[0])

        # Margin in basis points (100 bps = 1 percentage point)
        if len(row) >= 2:
            second_best_acc = float(row.iloc[1])
            margin_bps = round((best_acc - second_best_acc) * 100, 1)
        else:
            margin_bps = float("nan")

        rows.append(
            {
                "archetype": archetype,
                "best_algorithm": best_algo,
                "accuracy_pct": round(best_acc, 2),
                "margin_bps": margin_bps,
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        logger.info(
            "Best per segment: %d segments, algorithms used: %s",
            len(result),
            sorted(result["best_algorithm"].unique()),
        )

    return result


def compute_ceiling_accuracy(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    classification_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute oracle/ceiling accuracy: best algorithm per DFU-month.

    For each (sku_ck, startdate), pick the algorithm with the lowest absolute
    error, then aggregate to per-segment accuracy.

    This is the theoretical upper bound -- if we could perfectly pick the best
    algorithm for each individual DFU-month.

    Args:
        predictions_df: All predictions (columns: sku_ck, startdate, basefcst_pref, algorithm_id).
        actuals_df: Actual demand (columns: sku_ck, startdate, qty).
        classification_df: DFU archetype labels (columns: sku_ck, archetype).

    Returns:
        DataFrame with columns: archetype, ceiling_accuracy, n_dfu_months
    """
    for name, df, required in [
        ("predictions_df", predictions_df, {"sku_ck", "startdate", "basefcst_pref", "algorithm_id"}),
        ("actuals_df", actuals_df, {"sku_ck", "startdate", "qty"}),
        ("classification_df", classification_df, {"sku_ck", "archetype"}),
    ]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")

    if predictions_df.empty or actuals_df.empty:
        logger.warning("Empty input; returning empty ceiling accuracy")
        return pd.DataFrame(columns=["archetype", "ceiling_accuracy", "n_dfu_months"])

    # Average duplicate timeframe predictions
    preds = (
        predictions_df.groupby(["sku_ck", "startdate", "algorithm_id"], sort=False)["basefcst_pref"]
        .mean()
        .reset_index()
    )

    # Join with actuals
    merged = preds.merge(
        actuals_df[["sku_ck", "startdate", "qty"]],
        on=["sku_ck", "startdate"],
        how="inner",
    )
    if merged.empty:
        logger.warning("No matching (sku_ck, startdate) for ceiling computation")
        return pd.DataFrame(columns=["archetype", "ceiling_accuracy", "n_dfu_months"])

    # Compute absolute error per prediction row
    merged["abs_error"] = np.abs(merged["basefcst_pref"] - merged["qty"])

    # For each (sku_ck, startdate), keep the row with the lowest abs_error (oracle pick)
    idx_best = merged.groupby(["sku_ck", "startdate"], sort=False)["abs_error"].idxmin()
    oracle = merged.loc[idx_best].copy()

    # Attach archetype
    oracle = oracle.merge(
        classification_df[["sku_ck", "archetype"]],
        on="sku_ck",
        how="inner",
    )

    if oracle.empty:
        logger.warning("No DFUs matched for ceiling computation")
        return pd.DataFrame(columns=["archetype", "ceiling_accuracy", "n_dfu_months"])

    # Aggregate per archetype: WAPE then accuracy
    agg = oracle.groupby("archetype", sort=False).agg(
        sum_abs_error=("abs_error", "sum"),
        sum_actual=("qty", "sum"),
        n_dfu_months=("abs_error", "count"),
    ).reset_index()

    safe_denom = np.maximum(np.abs(agg["sum_actual"]), 1.0)
    wape = agg["sum_abs_error"] / safe_denom * 100
    agg["ceiling_accuracy"] = np.maximum(100 - wape, 0.0)

    result = agg[["archetype", "ceiling_accuracy", "n_dfu_months"]].copy()

    logger.info(
        "Ceiling accuracy: %d archetypes, mean ceiling = %.1f%%",
        len(result),
        result["ceiling_accuracy"].mean(),
    )

    return result


def format_affinity_heatmap(
    affinity_matrix: pd.DataFrame,
    detail_df: pd.DataFrame,
) -> str:
    """Format the affinity matrix as a human-readable text table.

    Shows accuracy_pct values with the best per row highlighted with *.
    Includes sample sizes (n_dfu_months) in parentheses.

    Args:
        affinity_matrix: Index=archetype, columns=algorithm_id, values=accuracy_pct.
        detail_df: Detail dataframe with n_dfu_months per (archetype, algorithm_id).

    Returns:
        Formatted string table suitable for logging or display.
    """
    if affinity_matrix.empty:
        return "(empty affinity matrix)"

    # Build a lookup for n_dfu_months
    sample_lookup: dict[tuple[str, str], int] = {}
    if not detail_df.empty:
        for _, row in detail_df.iterrows():
            key = (str(row["archetype"]), str(row["algorithm_id"]))
            sample_lookup[key] = int(row["n_dfu_months"])

    algorithms = list(affinity_matrix.columns)
    archetypes = list(affinity_matrix.index)

    # Determine column widths
    archetype_width = max(len(str(a)) for a in archetypes) if archetypes else 10
    archetype_width = max(archetype_width, 10)

    col_widths: list[int] = []
    for algo in algorithms:
        # Width must fit: " *XX.X (NNNN) " or "     ---     "
        min_width = max(len(algo), 13)
        col_widths.append(min_width)

    # Header
    lines: list[str] = []
    header_parts = [f"{'':>{archetype_width}}"]
    for algo, cw in zip(algorithms, col_widths, strict=False):
        header_parts.append(f" {algo:^{cw}} ")
    lines.append("|".join(header_parts))

    # Separator
    sep_parts = ["-" * archetype_width]
    for cw in col_widths:
        sep_parts.append("-" * (cw + 2))
    lines.append("+".join(sep_parts))

    # Data rows
    for archetype in archetypes:
        row = affinity_matrix.loc[archetype]
        valid = row.dropna()
        best_algo = valid.idxmax() if not valid.empty else None

        row_parts = [f"{archetype:>{archetype_width}}"]
        for algo, cw in zip(algorithms, col_widths, strict=False):
            val = row.get(algo)
            if pd.isna(val):
                cell = "---"
            else:
                n_months = sample_lookup.get((archetype, algo), 0)
                marker = "*" if algo == best_algo else " "
                cell = f"{marker}{val:.1f} ({n_months})"
            row_parts.append(f" {cell:^{cw}} ")
        lines.append("|".join(row_parts))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _empty_affinity_matrix() -> pd.DataFrame:
    """Return an empty affinity matrix with proper structure."""
    return pd.DataFrame()


def _empty_detail_df() -> pd.DataFrame:
    """Return an empty detail DataFrame with the correct columns."""
    return pd.DataFrame(
        columns=[
            "archetype",
            "algorithm_id",
            "wape",
            "accuracy_pct",
            "bias",
            "n_dfu_months",
            "n_dfus",
            "mean_abs_error",
            "mean_actual",
        ]
    )
