"""Report generator for the Expert Panel test.

Produces CSV outputs, JSON summary, and a human-readable text report.
All outputs go to the configured results directory.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def save_all_outputs(
    output_dir: Path,
    golden_skus: list[str],
    classification_df: pd.DataFrame,
    all_predictions_df: pd.DataFrame,
    affinity_matrix: pd.DataFrame,
    affinity_detail: pd.DataFrame,
    assignments_df: pd.DataFrame,
    portfolio_stats: dict[str, Any],
    comparison: dict[str, Any],
    runtime_seconds: float,
    monthly_accuracy: dict[str, dict[str, Any]] | None = None,
    avg_3m: dict[str, Any] | None = None,
    avg_6m: dict[str, Any] | None = None,
    overall_monthly: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Save all experiment outputs to the results directory.

    Args:
        output_dir: Directory to write outputs.
        golden_skus: List of sampled sku_ck values.
        classification_df: DFU archetype classifications.
        all_predictions_df: All predictions from all algorithms.
        affinity_matrix: Segment x algorithm accuracy matrix.
        affinity_detail: Per-(segment, algorithm) detail.
        assignments_df: Portfolio assignments.
        portfolio_stats: Portfolio-level accuracy stats.
        comparison: Full comparison dict from compare_all().
        runtime_seconds: Total experiment runtime.

    Returns:
        Dict mapping output name to file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    # 1. Golden set SKUs
    golden_path = output_dir / "golden_set_skus.csv"
    golden_df = pd.DataFrame({"sku_ck": golden_skus})
    golden_df.to_csv(golden_path, index=False)
    logger.info("Saved %s (%d rows)", golden_path, len(golden_df))
    saved["golden_set_skus"] = golden_path

    # 2. Demand classification
    classification_path = output_dir / "demand_classification.csv"
    classification_df.to_csv(classification_path, index=False)
    logger.info("Saved %s (%d rows)", classification_path, len(classification_df))
    saved["demand_classification"] = classification_path

    # 3. Affinity matrix
    affinity_matrix_path = output_dir / "affinity_matrix.csv"
    affinity_matrix.to_csv(affinity_matrix_path)
    logger.info("Saved %s (%d rows)", affinity_matrix_path, len(affinity_matrix))
    saved["affinity_matrix"] = affinity_matrix_path

    # 4. Affinity detail
    affinity_detail_path = output_dir / "affinity_detail.csv"
    affinity_detail.to_csv(affinity_detail_path, index=False)
    logger.info("Saved %s (%d rows)", affinity_detail_path, len(affinity_detail))
    saved["affinity_detail"] = affinity_detail_path

    # 5. Portfolio assignments
    assignments_path = output_dir / "portfolio_assignments.csv"
    assignments_df.to_csv(assignments_path, index=False)
    logger.info("Saved %s (%d rows)", assignments_path, len(assignments_df))
    saved["portfolio_assignments"] = assignments_path

    # 6. Comparison summary JSON
    comparison_json_path = output_dir / "comparison_summary.json"
    comparison_json_path.write_text(
        json.dumps(comparison, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Saved %s", comparison_json_path)
    saved["comparison_summary"] = comparison_json_path

    # 7. Comparison per-segment CSV
    per_segment_path = output_dir / "comparison_per_segment.csv"
    per_segment_df = comparison.get("per_segment")
    if isinstance(per_segment_df, pd.DataFrame) and not per_segment_df.empty:
        per_segment_df.to_csv(per_segment_path, index=False)
        logger.info("Saved %s (%d rows)", per_segment_path, len(per_segment_df))
    else:
        pd.DataFrame().to_csv(per_segment_path, index=False)
        logger.info("Saved %s (empty)", per_segment_path)
    saved["comparison_per_segment"] = per_segment_path

    # 8. Experiment report
    report_path = output_dir / "experiment_report.txt"
    report_text = generate_report(
        classification_df=classification_df,
        affinity_matrix=affinity_matrix,
        assignments_df=assignments_df,
        portfolio_stats=portfolio_stats,
        comparison=comparison,
        runtime_seconds=runtime_seconds,
        monthly_accuracy=monthly_accuracy,
        avg_3m=avg_3m,
        avg_6m=avg_6m,
    )
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Saved %s", report_path)
    saved["experiment_report"] = report_path

    # 9. Monthly accuracy JSON (optional)
    if monthly_accuracy is not None:
        monthly_path = output_dir / "monthly_accuracy.json"
        import json
        monthly_path.write_text(
            json.dumps(
                {
                    "monthly": monthly_accuracy,
                    "avg_3m": avg_3m or {},
                    "avg_6m": avg_6m or {},
                    "overall": overall_monthly or {},
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        logger.info("Saved %s", monthly_path)
        saved["monthly_accuracy"] = monthly_path

    logger.info(
        "All outputs saved to %s (%d files)", output_dir, len(saved)
    )
    return saved


def generate_report(
    classification_df: pd.DataFrame,
    affinity_matrix: pd.DataFrame,
    assignments_df: pd.DataFrame,
    portfolio_stats: dict[str, Any],
    comparison: dict[str, Any],
    runtime_seconds: float,
    monthly_accuracy: dict[str, dict[str, Any]] | None = None,
    avg_3m: dict[str, Any] | None = None,
    avg_6m: dict[str, Any] | None = None,
    overall_monthly: dict[str, Any] | None = None,
) -> str:
    """Generate a human-readable experiment report.

    Returns:
        Multi-line string report.
    """
    lines: list[str] = []
    sep = "=" * 80

    # --- Header ---
    lines.append(sep)
    lines.append("EXPERT PANEL ALGORITHM SELECTION — EXPERIMENT REPORT")
    lines.append(sep)
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    minutes = int(runtime_seconds // 60)
    seconds = int(runtime_seconds % 60)
    lines.append(f"Runtime: {minutes}m {seconds:02d}s")

    n_golden = len(classification_df) if not classification_df.empty else 0
    lines.append(f"Golden Set: {n_golden:,} DFUs")
    lines.append("")

    # --- Section 1: Demand Classification ---
    lines.append(sep)
    lines.append("1. DEMAND CLASSIFICATION")
    lines.append(sep)
    lines.extend(_format_classification_table(classification_df))
    lines.append("")

    # --- Section 2: Affinity Matrix ---
    lines.append(sep)
    lines.append("2. AFFINITY MATRIX (Accuracy % by Segment x Algorithm)")
    lines.append(sep)
    lines.extend(_format_affinity_matrix_section(affinity_matrix))
    lines.append("")

    # --- Section 3: Portfolio Assignments ---
    lines.append(sep)
    lines.append("3. PORTFOLIO ASSIGNMENTS")
    lines.append(sep)
    lines.extend(
        _format_portfolio_section(assignments_df, portfolio_stats)
    )
    lines.append("")

    # --- Section 4: Comparison vs Baselines ---
    lines.append(sep)
    lines.append("4. COMPARISON VS BASELINES")
    lines.append(sep)
    lines.extend(_format_comparison_section(comparison))
    lines.append("")

    # --- Section 5: Per-Segment Comparison ---
    lines.append(sep)
    lines.append("5. PER-SEGMENT COMPARISON")
    lines.append(sep)
    lines.extend(_format_per_segment_section(comparison))
    lines.append("")

    # --- Section 6: Key Insights ---
    lines.append(sep)
    lines.append("6. KEY INSIGHTS")
    lines.append(sep)
    insights = _generate_insights(assignments_df, comparison)
    for insight in insights:
        lines.append(f"- {insight}")
    lines.append("")

    # --- Section 7: Monthly Accuracy (optional) ---
    if monthly_accuracy:
        lines.append(sep)
        lines.append(
            "7. MONTHLY ACCURACY — execution-lag-matched, full-coverage months only"
        )
        lines.append(sep)
        lines.extend(
            _format_monthly_accuracy_section(
                monthly_accuracy, avg_3m, avg_6m, overall_monthly
            )
        )
        lines.append("")

    lines.append(sep)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section formatters
# ---------------------------------------------------------------------------


def _format_classification_table(classification_df: pd.DataFrame) -> list[str]:
    """Format the demand classification summary table."""
    lines: list[str] = []
    if classification_df.empty:
        lines.append("(no classification data)")
        return lines

    # Group by archetype
    grouped = (
        classification_df.groupby("archetype", sort=False)
        .agg(
            n_dfus=("sku_ck", "count"),
            mean_demand=("mean_demand", "mean"),
            mean_adi=("adi", "mean"),
            mean_cv2=("cv2", "mean"),
        )
        .reset_index()
        .sort_values("n_dfus", ascending=False)
        .reset_index(drop=True)
    )

    total_dfus = grouped["n_dfus"].sum()

    header = (
        f"{'Archetype':<20}| {'DFUs':>6} | {'%':>6} "
        f"| {'Mean Demand':>12} | {'Mean ADI':>9} | {'Mean CV2':>9}"
    )
    lines.append(header)
    lines.append(
        "-" * 20 + "+" + "-" * 8 + "+" + "-" * 8
        + "+" + "-" * 14 + "+" + "-" * 11 + "+" + "-" * 10
    )

    for _, row in grouped.iterrows():
        pct = row["n_dfus"] / total_dfus * 100 if total_dfus > 0 else 0.0
        lines.append(
            f"{row['archetype']:<20}| {row['n_dfus']:>5,} | {pct:>5.1f}% "
            f"| {row['mean_demand']:>11,.0f} | {row['mean_adi']:>9.2f} | {row['mean_cv2']:>9.2f}"
        )

    return lines


def _format_affinity_matrix_section(affinity_matrix: pd.DataFrame) -> list[str]:
    """Format the affinity matrix with * marking the best per row."""
    lines: list[str] = []
    if affinity_matrix.empty:
        lines.append("(empty affinity matrix)")
        return lines

    algorithms = list(affinity_matrix.columns)
    archetypes = list(affinity_matrix.index)

    # Column widths
    arch_width = max(max(len(str(a)) for a in archetypes), 18)
    col_width = max(max(len(str(a)) for a in algorithms), 10)

    # Header
    header_parts = [f"{'Archetype':<{arch_width}}"]
    for algo in algorithms:
        header_parts.append(f" {algo:>{col_width}} ")
    lines.append("|".join(header_parts))

    sep_parts = ["-" * arch_width]
    for _ in algorithms:
        sep_parts.append("-" * (col_width + 2))
    lines.append("+".join(sep_parts))

    # Rows
    for archetype in archetypes:
        row = affinity_matrix.loc[archetype]
        valid = row.dropna()
        best_algo = valid.idxmax() if not valid.empty else None

        row_parts = [f"{archetype:<{arch_width}}"]
        for algo in algorithms:
            val = row.get(algo)
            if pd.isna(val):
                cell = "---"
            else:
                marker = "*" if algo == best_algo else " "
                cell = f"{marker}{val:.1f}%"
            row_parts.append(f" {cell:>{col_width}} ")
        lines.append("|".join(row_parts))

    lines.append("")
    lines.append("* = best algorithm for that segment")

    return lines


def _format_portfolio_section(
    assignments_df: pd.DataFrame,
    portfolio_stats: dict[str, Any],
) -> list[str]:
    """Format the portfolio assignments summary."""
    lines: list[str] = []

    if assignments_df.empty:
        lines.append("(no portfolio assignments)")
        return lines

    dist = portfolio_stats.get("algorithm_distribution", {})
    total_dfus = portfolio_stats.get("n_dfus", 0)

    # Count segments per algorithm
    segment_counts = (
        assignments_df.groupby("best_algorithm")["archetype"]
        .count()
        .to_dict()
    )

    header = (
        f"{'Algorithm':<22}| {'Segments':>9} | {'DFUs':>6} "
        f"| {'%':>6} | {'Accuracy%':>10}"
    )
    lines.append(header)
    lines.append(
        "-" * 22 + "+" + "-" * 11 + "+" + "-" * 8
        + "+" + "-" * 8 + "+" + "-" * 11
    )

    # Sort by n_dfus descending
    sorted_algos = sorted(
        dist.keys(),
        key=lambda a: dist[a].get("n_dfus", 0),
        reverse=True,
    )

    for algo in sorted_algos:
        info = dist[algo]
        n_segments = segment_counts.get(algo, 0)
        n_dfus = info.get("n_dfus", 0)
        pct = info.get("pct", 0.0)
        accuracy = info.get("accuracy", 0.0)
        lines.append(
            f"{algo:<22}| {n_segments:>9} | {n_dfus:>5,} "
            f"| {pct:>5.1f}% | {accuracy:>9.1f}%"
        )

    lines.append("")
    portfolio_acc = portfolio_stats.get("portfolio_accuracy", 0.0)
    portfolio_wape = portfolio_stats.get("portfolio_wape", 0.0)
    lines.append(f"Portfolio accuracy: {portfolio_acc:.1f}%")
    lines.append(f"Portfolio WAPE:     {portfolio_wape:.2f}%")
    lines.append(f"Total DFUs:         {total_dfus:,}")
    lines.append(f"Algorithms used:    {len(dist)}")

    return lines


def _format_comparison_section(comparison: dict[str, Any]) -> list[str]:
    """Format the comparison vs baselines table."""
    lines: list[str] = []

    baselines = comparison.get("baselines", {})
    panel = comparison.get("panel", {})

    if not baselines and not panel:
        lines.append("(no comparison data)")
        return lines

    header = f"{'':>22}| {'WAPE':>8} | {'Accuracy%':>10} | {'Bias':>7}"
    lines.append(header)
    lines.append("-" * 22 + "+" + "-" * 10 + "+" + "-" * 12 + "+" + "-" * 8)

    # Panel row
    panel_wape = panel.get("wape", 0.0)
    panel_acc = panel.get("accuracy", 0.0)
    panel_bias = panel.get("bias", 0.0)
    lines.append(
        f"{'Expert Panel':<22}| {panel_wape:>7.1f}% | {panel_acc:>9.1f}% | {panel_bias:>+7.2f}"
    )

    # Baseline rows and track lifts
    lifts: list[tuple[str, float]] = []
    for name, stats in baselines.items():
        display_name = _display_name(name)
        bl_wape = stats.get("wape", 0.0)
        bl_acc = stats.get("accuracy", 0.0)
        bl_bias = stats.get("bias", 0.0)
        lines.append(
            f"{display_name:<22}| {bl_wape:>7.1f}% | {bl_acc:>9.1f}% | {bl_bias:>+7.2f}"
        )
        lift_bps = (panel_acc - bl_acc) * 100
        lifts.append((display_name, lift_bps))

    lines.append("")
    for name, lift_bps in lifts:
        lift_pct = lift_bps / 100
        sign = "+" if lift_bps >= 0 else ""
        lines.append(
            f">>> LIFT vs {name + ':':<22} {sign}{lift_bps:,.0f} bps "
            f"({sign}{lift_pct:.1f}% accuracy)"
        )

    return lines


def _format_per_segment_section(comparison: dict[str, Any]) -> list[str]:
    """Format the per-segment comparison table."""
    lines: list[str] = []

    per_segment = comparison.get("per_segment")
    if not isinstance(per_segment, pd.DataFrame) or per_segment.empty:
        lines.append("(no per-segment comparison data)")
        return lines

    # Determine which baseline columns exist
    baseline_cols = [
        c for c in per_segment.columns
        if c not in ("archetype", "panel_accuracy", "best_lift")
        and c.endswith("_accuracy")
    ]

    # Build header
    header_parts = [f"{'Archetype':<20}", f" {'Panel':>7} "]
    for col in baseline_cols:
        display = col.replace("_accuracy", "").replace("_", " ").title()
        header_parts.append(f" {display:>10} ")
    header_parts.append(f" {'Best Lift':>10} ")
    lines.append("|".join(header_parts))

    sep_parts = ["-" * 20, "-" * 9]
    for _ in baseline_cols:
        sep_parts.append("-" * 12)
    sep_parts.append("-" * 12)
    lines.append("+".join(sep_parts))

    # Rows
    for _, row in per_segment.iterrows():
        row_parts = [f"{row['archetype']:<20}"]
        panel_val = row.get("panel_accuracy", 0.0)
        row_parts.append(f" {panel_val:>6.1f}% ")

        for col in baseline_cols:
            val = row.get(col, np.nan)
            if pd.isna(val):
                row_parts.append(f" {'---':>10} ")
            else:
                row_parts.append(f" {val:>9.1f}% ")
        best_lift = row.get("best_lift", 0.0)
        if pd.isna(best_lift):
            row_parts.append(f" {'---':>10} ")
        else:
            row_parts.append(f" {best_lift:>+9.1f}% ")
        lines.append("|".join(row_parts))

    return lines


def _generate_insights(
    assignments_df: pd.DataFrame,
    comparison: dict[str, Any],
) -> list[str]:
    """Auto-generate key insights from the results."""
    insights: list[str] = []

    if assignments_df.empty:
        insights.append("No assignments data available for insight generation.")
        return insights

    per_segment = comparison.get("per_segment")
    panel = comparison.get("panel", {})
    baselines = comparison.get("baselines", {})

    # --- Insight 1: Segments with biggest lift vs current champion ---
    if isinstance(per_segment, pd.DataFrame) and not per_segment.empty:
        champion_col = _find_column(per_segment, "champion")
        if champion_col is not None and "panel_accuracy" in per_segment.columns:
            per_segment = per_segment.copy()
            per_segment["_lift"] = per_segment["panel_accuracy"] - per_segment[champion_col]
            best_row = per_segment.loc[per_segment["_lift"].idxmax()]
            insights.append(
                f"Biggest lift vs champion: '{best_row['archetype']}' segment gains "
                f"+{best_row['_lift']:.1f}pp from Expert Panel assignment"
            )
            per_segment.drop(columns=["_lift"], inplace=True)

    # --- Insight 2: Statistical models winning over tree models ---
    statistical_algos = {
        "holt_winters", "simple_es", "croston_sba", "auto_arima",
        "theta", "seasonal_naive", "rolling_mean",
    }
    tree_algos = {"lgbm_cluster", "catboost_cluster", "xgboost_cluster"}

    stat_wins = assignments_df[
        assignments_df["best_algorithm"].isin(statistical_algos)
    ]
    if not stat_wins.empty:
        stat_segments = stat_wins["archetype"].tolist()
        stat_n_dfus = int(stat_wins["n_dfus"].sum())
        stat_algos_used = stat_wins["best_algorithm"].unique().tolist()
        insights.append(
            f"Statistical models ({', '.join(stat_algos_used)}) won "
            f"{len(stat_segments)} segment(s) covering {stat_n_dfus:,} DFUs: "
            f"{', '.join(stat_segments)}"
        )

    tree_wins = assignments_df[
        assignments_df["best_algorithm"].isin(tree_algos)
    ]
    if not tree_wins.empty:
        tree_segments = tree_wins["archetype"].tolist()
        tree_n_dfus = int(tree_wins["n_dfus"].sum())
        insights.append(
            f"Tree models retained {len(tree_segments)} segment(s) "
            f"covering {tree_n_dfus:,} DFUs: {', '.join(tree_segments)}"
        )

    # --- Insight 3: Ceiling gap (oracle vs portfolio) ---
    ceiling = comparison.get("ceiling_accuracy")
    portfolio_acc = panel.get("accuracy", 0.0)
    if ceiling is not None and not np.isnan(ceiling):
        gap = ceiling - portfolio_acc
        insights.append(
            f"Oracle ceiling accuracy: {ceiling:.1f}% vs portfolio {portfolio_acc:.1f}% "
            f"(gap = {gap:.1f}pp of unrealized potential)"
        )

    # --- Insight 4: Low-confidence segments ---
    low_conf = assignments_df[assignments_df["confidence"] == "low"]
    if not low_conf.empty:
        low_segments = low_conf["archetype"].tolist()
        insights.append(
            f"Low-confidence segments (few DFUs): {', '.join(low_segments)} "
            f"— results may be unreliable, consider expanding the golden set"
        )

    # --- Insight 5: Simplest high-impact change ---
    if not assignments_df.empty and baselines:
        # Find the single largest DFU block assigned to a non-default algorithm
        champion_ids = {"lgbm_cluster", "catboost_cluster", "xgboost_cluster"}
        non_champion = assignments_df[
            ~assignments_df["best_algorithm"].isin(champion_ids)
        ].copy()
        if not non_champion.empty:
            biggest = non_champion.loc[non_champion["n_dfus"].idxmax()]
            # Estimate lift from this single reassignment
            runner_up_acc = biggest.get("runner_up_accuracy", np.nan)
            best_acc = biggest.get("accuracy_pct", 0.0)
            if not np.isnan(runner_up_acc):
                single_lift_bps = int((best_acc - runner_up_acc) * 100)
                insights.append(
                    f"Simplest high-impact change: switching {int(biggest['n_dfus']):,} "
                    f"'{biggest['archetype']}' DFUs to '{biggest['best_algorithm']}' "
                    f"yields +{single_lift_bps} bps over the runner-up"
                )

    if not insights:
        insights.append("No actionable insights could be generated from the data.")

    return insights


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _format_monthly_accuracy_section(
    monthly_accuracy: dict[str, dict[str, Any]],
    avg_3m: dict[str, Any] | None,
    avg_6m: dict[str, Any] | None,
    overall_monthly: dict[str, Any] | None = None,
) -> list[str]:
    """Format per-calendar-month accuracy table.

    Columns: Month | Algo1 | Algo2 | ...  (sorted by algo name)
    Shows accuracy% per month, then footer rows:
      Overall (mean-of-months), Avg 3M, Avg 6M
    All rows use execution-lag-matched accuracy.
    """
    lines: list[str] = []
    if not monthly_accuracy:
        lines.append("(no monthly accuracy data)")
        return lines

    all_algos = sorted({a for m in monthly_accuracy.values() for a in m})
    if not all_algos:
        lines.append("(no algorithms in monthly accuracy data)")
        return lines

    month_width = 8
    col_width = max(max(len(a) for a in all_algos), 9)
    separator = "+".join(["-" * month_width] + ["-" * (col_width + 2)] * len(all_algos))

    # Header
    header_parts = [f"{'Month':<{month_width}}"]
    for algo in all_algos:
        header_parts.append(f" {algo:>{col_width}} ")
    lines.append("|".join(header_parts))
    lines.append(separator)

    # One row per month
    for month_str in sorted(monthly_accuracy.keys()):
        row_parts = [f"{month_str:<{month_width}}"]
        for algo in all_algos:
            data = monthly_accuracy[month_str].get(algo)
            if data is None:
                row_parts.append(f" {'---':>{col_width}} ")
            else:
                row_parts.append(f" {data['accuracy']:>{col_width - 1}.1f}% ")
        lines.append("|".join(row_parts))

    lines.append(separator)

    # Overall (mean-of-months) row
    overall_parts = [f"{'Overall':>{month_width}}"]
    for algo in all_algos:
        data = (overall_monthly or {}).get(algo)
        if data is None:
            overall_parts.append(f" {'---':>{col_width}} ")
        else:
            overall_parts.append(f" {data['mean_monthly_accuracy']:>{col_width - 1}.1f}% ")
    lines.append("|".join(overall_parts))

    # Avg 3M row
    avg3_parts = [f"{'Avg 3M':>{month_width}}"]
    for algo in all_algos:
        data = (avg_3m or {}).get(algo)
        if data is None:
            avg3_parts.append(f" {'---':>{col_width}} ")
        else:
            avg3_parts.append(f" {data['mean_monthly_accuracy']:>{col_width - 1}.1f}% ")
    lines.append("|".join(avg3_parts))

    # Avg 6M row
    avg6_parts = [f"{'Avg 6M':>{month_width}}"]
    for algo in all_algos:
        data = (avg_6m or {}).get(algo)
        if data is None:
            avg6_parts.append(f" {'---':>{col_width}} ")
        else:
            avg6_parts.append(f" {data['mean_monthly_accuracy']:>{col_width - 1}.1f}% ")
    lines.append("|".join(avg6_parts))

    lines.append("")
    lines.append("All rows: mean-of-monthly-accuracies (equal weight per month, exec-lag-matched)")

    if overall_monthly:
        best_overall = max(
            overall_monthly.items(), key=lambda x: x[1].get("mean_monthly_accuracy", 0)
        )
        lines.append(
            f"Best overall: {best_overall[0]} @ {best_overall[1]['mean_monthly_accuracy']:.1f}%"
        )
    if avg_3m:
        best_3m = max(avg_3m.items(), key=lambda x: x[1].get("mean_monthly_accuracy", 0))
        lines.append(
            f"Best Avg 3M:  {best_3m[0]} @ {best_3m[1]['mean_monthly_accuracy']:.1f}%"
        )
    if avg_6m:
        best_6m = max(avg_6m.items(), key=lambda x: x[1].get("mean_monthly_accuracy", 0))
        lines.append(
            f"Best Avg 6M:  {best_6m[0]} @ {best_6m[1]['mean_monthly_accuracy']:.1f}%"
        )

    return lines


def _display_name(key: str) -> str:
    """Convert a snake_case key to a human-readable display name."""
    return key.replace("_", " ").title()


def _find_column(df: pd.DataFrame, substring: str) -> str | None:
    """Find the first column containing the given substring."""
    for col in df.columns:
        if substring in col.lower():
            return col
    return None
