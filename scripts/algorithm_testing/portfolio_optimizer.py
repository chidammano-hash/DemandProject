"""Portfolio optimizer: assign best algorithm per demand segment.

Supports unconstrained (greedy) and constrained (max N algorithms) modes.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _naive_acc_per_segment(detail_df: pd.DataFrame) -> dict[str, float]:
    """Return seasonal_naive accuracy per archetype; 0.0 if not present."""
    naive_rows = detail_df[detail_df["algorithm_id"] == "seasonal_naive"]
    return dict(zip(naive_rows["archetype"], naive_rows["accuracy_pct"].clip(lower=0.0)))


def optimize_greedy(
    affinity_matrix: pd.DataFrame,
    detail_df: pd.DataFrame,
    min_segment_dfus: int = 30,
    min_dfu_coverage_pct: float = 0.5,
    coverage_weighted: bool = True,
    naive_floor: bool = True,
) -> pd.DataFrame:
    """Greedy assignment: pick best algorithm per segment.

    Args:
        affinity_matrix: archetype x algorithm accuracy matrix.
            Index = archetype names, columns = algorithm names, values = accuracy_pct.
        detail_df: Per-(archetype, algorithm) detail with n_dfus.
        min_segment_dfus: Mark segments with fewer DFUs as low confidence.
        min_dfu_coverage_pct: Coverage threshold below which accuracy is adjusted.
            Set to 0.0 to disable. Default 0.5 (50%).
        coverage_weighted: When True, algorithms with coverage < min_dfu_coverage_pct
            get coverage-adjusted accuracy = coverage*algo_acc + (1-coverage)*naive_acc
            instead of being excluded outright. Prevents low-coverage DL models from
            winning segments they were only tested on a biased subsample of.
        naive_floor: When True, any segment assigned an algorithm with accuracy
            below seasonal_naive's accuracy for that segment is reassigned to
            seasonal_naive. Prevents the portfolio from performing worse than naive.

    Returns:
        DataFrame with columns:
            archetype, best_algorithm, accuracy_pct, runner_up_algorithm,
            runner_up_accuracy, margin_bps, n_dfus, confidence
        confidence = 'high' if n_dfus >= min_segment_dfus, else 'low'
    """
    # Total DFUs per archetype = max n_dfus across algorithms (all algorithms
    # share the same DFU pool; the max is the closest to the true total)
    total_dfus_per_arch: dict[str, int] = (
        detail_df.groupby("archetype")["n_dfus"].max().to_dict()
        if not detail_df.empty
        else {}
    )
    naive_acc: dict[str, float] = _naive_acc_per_segment(detail_df) if (coverage_weighted or naive_floor) else {}

    records: list[dict[str, Any]] = []

    for archetype in affinity_matrix.index:
        row = affinity_matrix.loc[archetype].dropna()
        if row.empty:
            logger.warning("No valid accuracies for archetype=%s; skipping", archetype)
            continue

        if min_dfu_coverage_pct > 0.0:
            total = total_dfus_per_arch.get(archetype, 1)
            seg_naive_acc = naive_acc.get(archetype, 0.0)
            if coverage_weighted:
                # Soft penalty: adjust accuracy for low-coverage algorithms rather
                # than excluding them outright. adj_acc = cov*algo + (1-cov)*naive.
                adjusted: dict[str, float] = {}
                for algo in row.index:
                    mask = (detail_df["archetype"] == archetype) & (
                        detail_df["algorithm_id"] == algo
                    )
                    n = int(detail_df.loc[mask, "n_dfus"].iloc[0]) if mask.any() else 0
                    cov = n / max(total, 1)
                    if cov < min_dfu_coverage_pct:
                        adjusted[algo] = cov * float(row[algo]) + (1 - cov) * seg_naive_acc
                    else:
                        adjusted[algo] = float(row[algo])
                row = pd.Series(adjusted).dropna()
            else:
                # Original binary filter: exclude algorithms below coverage threshold
                eligible_algos = []
                for algo in row.index:
                    mask = (detail_df["archetype"] == archetype) & (
                        detail_df["algorithm_id"] == algo
                    )
                    n = int(detail_df.loc[mask, "n_dfus"].iloc[0]) if mask.any() else 0
                    if n / max(total, 1) >= min_dfu_coverage_pct:
                        eligible_algos.append(algo)
                if eligible_algos:
                    row = row[eligible_algos]
                else:
                    best_cov_algo = max(
                        row.index,
                        key=lambda a: int(
                            detail_df.loc[
                                (detail_df["archetype"] == archetype)
                                & (detail_df["algorithm_id"] == a),
                                "n_dfus",
                            ].iloc[0]
                            if (
                                (detail_df["archetype"] == archetype)
                                & (detail_df["algorithm_id"] == a)
                            ).any()
                            else 0
                        ),
                    )
                    logger.warning(
                        "Archetype %s: no algorithm meets %.0f%% coverage; "
                        "falling back to best-coverage algorithm (%s)",
                        archetype, min_dfu_coverage_pct * 100, best_cov_algo,
                    )
                    row = row[[best_cov_algo]]

        sorted_row = row.sort_values(ascending=False)
        best_algo = sorted_row.index[0]
        best_acc = float(sorted_row.iloc[0])

        if len(sorted_row) >= 2:
            runner_up_algo = sorted_row.index[1]
            runner_up_acc = float(sorted_row.iloc[1])
        else:
            runner_up_algo = None
            runner_up_acc = np.nan

        margin_bps = int((best_acc - (runner_up_acc if not np.isnan(runner_up_acc) else best_acc)) * 100)

        # Look up n_dfus from detail_df
        mask = (detail_df["archetype"] == archetype) & (detail_df["algorithm_id"] == best_algo)
        matched = detail_df.loc[mask, "n_dfus"]
        n_dfus = int(matched.iloc[0]) if not matched.empty else 0

        confidence = "high" if n_dfus >= min_segment_dfus else "low"

        records.append(
            {
                "archetype": archetype,
                "best_algorithm": best_algo,
                "accuracy_pct": best_acc,
                "runner_up_algorithm": runner_up_algo,
                "runner_up_accuracy": runner_up_acc,
                "margin_bps": margin_bps,
                "n_dfus": n_dfus,
                "confidence": confidence,
            }
        )

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values("accuracy_pct", ascending=False).reset_index(drop=True)

    # Naive floor: reassign segments where assigned algorithm is worse than naive
    if naive_floor and not result.empty and naive_acc:
        for i, row in result.iterrows():
            seg = row["archetype"]
            seg_naive = naive_acc.get(seg, np.nan)
            if not np.isnan(seg_naive) and row["accuracy_pct"] < seg_naive:
                logger.info(
                    "Naive floor applied: %s → seasonal_naive (%.1f%% > %.1f%% from %s)",
                    seg, seg_naive, row["accuracy_pct"], row["best_algorithm"],
                )
                result.at[i, "best_algorithm"] = "seasonal_naive"
                result.at[i, "accuracy_pct"] = seg_naive

    logger.info(
        "Greedy optimization: %d segments assigned, %d high-confidence",
        len(result),
        (result["confidence"] == "high").sum() if not result.empty else 0,
    )
    return result


def optimize_constrained(
    affinity_matrix: pd.DataFrame,
    detail_df: pd.DataFrame,
    max_algorithms: int = 6,
    min_segment_dfus: int = 30,
    min_dfu_coverage_pct: float = 0.5,
    coverage_weighted: bool = True,
    naive_floor: bool = True,
) -> pd.DataFrame:
    """Constrained assignment: minimize portfolio WAPE with max N algorithms.

    Uses a greedy set-cover approach:
    1. Start with the single algorithm that gives the best DFU-weighted accuracy
    2. Iteratively add the algorithm that provides the largest marginal improvement
    3. Stop when max_algorithms reached or marginal improvement < 0.001% portfolio WAPE

    Args:
        affinity_matrix: archetype x algorithm accuracy matrix.
        detail_df: Per-(archetype, algorithm) detail with n_dfus, wape.
        max_algorithms: Maximum number of distinct algorithms allowed.
        min_segment_dfus: Mark segments below threshold as low confidence.
        min_dfu_coverage_pct: Coverage threshold below which WAPE is adjusted.
            Set to 0.0 to disable. Default 0.5.
        coverage_weighted: When True, algorithms below the coverage threshold get
            adj_wape = coverage*algo_wape + (1-coverage)*naive_wape instead of inf.
            Prevents biased subsamples from winning segments. Default True.
        naive_floor: When True, any segment assigned an algorithm worse than naive
            (by affinity accuracy) is reassigned to seasonal_naive. Default True.

    Returns:
        Same schema as optimize_greedy, plus:
            selected_algorithms: list of the algorithms in the portfolio
    """
    algorithms = list(affinity_matrix.columns)
    segments = list(affinity_matrix.index)

    # Build n_dfus lookup: segment -> n_dfus (sum across algorithms for the segment)
    segment_dfus: dict[str, int] = {}
    for seg in segments:
        seg_mask = detail_df["archetype"] == seg
        seg_total = detail_df.loc[seg_mask, "n_dfus"]
        # Use the max n_dfus across algorithms (each algorithm sees the same DFUs in a segment)
        segment_dfus[seg] = int(seg_total.max()) if not seg_total.empty else 0

    # Precompute naive WAPE per segment for coverage adjustment and naive floor
    naive_wape_per_seg: dict[str, float] = {}
    naive_acc_per_seg: dict[str, float] = {}
    if coverage_weighted or naive_floor:
        naive_acc_per_seg = _naive_acc_per_segment(detail_df)
        for seg, acc in naive_acc_per_seg.items():
            naive_wape_per_seg[seg] = max(0.0, 100.0 - acc)

    # Build WAPE lookup: (segment, algorithm) -> wape
    # Low-coverage algorithms are penalized via coverage-weighted WAPE rather than
    # excluded with inf, preventing biased subsamples from winning segments.
    wape_lookup: dict[tuple[str, str], float] = {}
    for _, row in detail_df.iterrows():
        archetype_str = row["archetype"]
        algo_str = row["algorithm_id"]
        key = (archetype_str, algo_str)
        raw_wape = float(row["wape"]) if not pd.isna(row.get("wape")) else np.inf
        if min_dfu_coverage_pct > 0.0:
            algo_n = int(row.get("n_dfus", 0))
            total_n = segment_dfus.get(archetype_str, 1)
            coverage = algo_n / max(total_n, 1)
            if coverage < min_dfu_coverage_pct:
                if coverage_weighted:
                    # Expected WAPE if algorithm covers `coverage` fraction and
                    # naive covers the rest — realistic deployment estimate.
                    seg_naive_wape = naive_wape_per_seg.get(archetype_str, 100.0)
                    raw_wape = coverage * raw_wape + (1 - coverage) * seg_naive_wape
                else:
                    raw_wape = np.inf  # original binary exclusion
        wape_lookup[key] = raw_wape

    total_dfus = sum(segment_dfus.values())
    if total_dfus == 0:
        logger.warning("No DFUs found; returning empty constrained result")
        return pd.DataFrame()

    def _portfolio_wape(selected: set[str]) -> float:
        """Compute portfolio WAPE given a set of selected algorithms."""
        weighted_sum = 0.0
        for seg in segments:
            best_wape = min(
                (wape_lookup.get((seg, algo), np.inf) for algo in selected),
                default=np.inf,
            )
            weighted_sum += segment_dfus.get(seg, 0) * best_wape
        return weighted_sum / total_dfus

    # Greedy set cover
    selected: set[str] = set()
    best_wape = np.inf

    for iteration in range(max_algorithms):
        best_candidate = None
        best_candidate_wape = best_wape

        for algo in algorithms:
            if algo in selected:
                continue
            candidate_wape = _portfolio_wape(selected | {algo})
            if candidate_wape < best_candidate_wape:
                best_candidate_wape = candidate_wape
                best_candidate = algo

        if best_candidate is None:
            logger.info("No more candidate algorithms to add at iteration %d", iteration)
            break

        improvement = best_wape - best_candidate_wape
        if iteration > 0 and improvement < 0.001:
            logger.info(
                "Stopping early at iteration %d: marginal improvement=%.4f%% < 0.001%%",
                iteration,
                improvement,
            )
            break

        selected.add(best_candidate)
        best_wape = best_candidate_wape
        logger.info(
            "Iteration %d: added '%s', portfolio WAPE=%.2f%%",
            iteration,
            best_candidate,
            best_wape,
        )

    selected_list = sorted(selected)
    logger.info("Selected %d algorithms: %s", len(selected_list), selected_list)

    # Assign each segment to its best algorithm within the selected set
    records: list[dict[str, Any]] = []
    for seg in segments:
        seg_wapes = {
            algo: wape_lookup.get((seg, algo), np.inf)
            for algo in selected
        }
        best_algo = min(seg_wapes, key=seg_wapes.get)  # type: ignore[arg-type]
        best_acc = 100.0 - seg_wapes[best_algo]

        # Find runner-up within selected set
        remaining = {a: w for a, w in seg_wapes.items() if a != best_algo}
        if remaining:
            runner_up_algo = min(remaining, key=remaining.get)  # type: ignore[arg-type]
            runner_up_acc = 100.0 - remaining[runner_up_algo]
        else:
            runner_up_algo = None
            runner_up_acc = np.nan

        margin_bps = int((best_acc - (runner_up_acc if not np.isnan(runner_up_acc) else best_acc)) * 100)

        n_dfus = segment_dfus.get(seg, 0)
        confidence = "high" if n_dfus >= min_segment_dfus else "low"

        records.append(
            {
                "archetype": seg,
                "best_algorithm": best_algo,
                "accuracy_pct": best_acc,
                "runner_up_algorithm": runner_up_algo,
                "runner_up_accuracy": runner_up_acc,
                "margin_bps": margin_bps,
                "n_dfus": n_dfus,
                "confidence": confidence,
                "selected_algorithms": selected_list,
            }
        )

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values("accuracy_pct", ascending=False).reset_index(drop=True)

    # Naive floor: reassign segments where assigned algorithm is worse than naive
    if naive_floor and not result.empty and naive_acc_per_seg:
        for i, row in result.iterrows():
            seg = row["archetype"]
            seg_naive_acc = naive_acc_per_seg.get(seg, np.nan)
            if not np.isnan(seg_naive_acc) and row["accuracy_pct"] < seg_naive_acc:
                logger.info(
                    "Naive floor applied: %s → seasonal_naive (%.1f%% > %.1f%% from %s)",
                    seg, seg_naive_acc, row["accuracy_pct"], row["best_algorithm"],
                )
                result.at[i, "best_algorithm"] = "seasonal_naive"
                result.at[i, "accuracy_pct"] = seg_naive_acc

    return result


def compute_portfolio_accuracy(
    assignments: pd.DataFrame,
    detail_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute overall portfolio accuracy from assignments.

    Args:
        assignments: Output of optimize_greedy or optimize_constrained.
        detail_df: Per-(archetype, algorithm) detail with wape, n_dfus, n_dfu_months.

    Returns:
        {
            'portfolio_wape': float,
            'portfolio_accuracy': float,
            'n_dfus': int,
            'n_dfu_months': int,
            'n_algorithms': int,
            'algorithm_distribution': {algo: {'n_dfus': int, 'pct': float, 'accuracy': float}}
        }
    """
    if assignments.empty:
        return {
            "portfolio_wape": 0.0,
            "portfolio_accuracy": 0.0,
            "n_dfus": 0,
            "n_dfu_months": 0,
            "n_algorithms": 0,
            "algorithm_distribution": {},
        }

    total_wape_weighted = 0.0
    total_dfu_months = 0
    total_dfus = 0
    algo_stats: dict[str, dict[str, float]] = {}

    for _, row in assignments.iterrows():
        archetype = row["archetype"]
        algo = row["best_algorithm"]

        mask = (detail_df["archetype"] == archetype) & (detail_df["algorithm_id"] == algo)
        matched = detail_df.loc[mask]

        if matched.empty:
            logger.warning(
                "No detail_df match for archetype=%s, algorithm=%s", archetype, algo
            )
            continue

        detail_row = matched.iloc[0]
        wape = float(detail_row.get("wape", 0.0))
        n_dfu_months = int(detail_row.get("n_dfu_months", 0))
        n_dfus = int(detail_row.get("n_dfus", 0))

        total_wape_weighted += wape * n_dfu_months
        total_dfu_months += n_dfu_months
        total_dfus += n_dfus

        if algo not in algo_stats:
            algo_stats[algo] = {"n_dfus": 0, "wape_weighted": 0.0, "dfu_months": 0}
        algo_stats[algo]["n_dfus"] += n_dfus
        algo_stats[algo]["wape_weighted"] += wape * n_dfu_months
        algo_stats[algo]["dfu_months"] += n_dfu_months

    portfolio_wape = total_wape_weighted / total_dfu_months if total_dfu_months > 0 else 0.0
    portfolio_accuracy = 100.0 - portfolio_wape

    algorithm_distribution: dict[str, dict[str, Any]] = {}
    for algo, stats in algo_stats.items():
        algo_accuracy = 100.0 - (stats["wape_weighted"] / stats["dfu_months"]) if stats["dfu_months"] > 0 else 0.0
        algorithm_distribution[algo] = {
            "n_dfus": int(stats["n_dfus"]),
            "pct": round(stats["n_dfus"] / total_dfus * 100, 1) if total_dfus > 0 else 0.0,
            "accuracy": round(algo_accuracy, 2),
        }

    return {
        "portfolio_wape": round(portfolio_wape, 4),
        "portfolio_accuracy": round(portfolio_accuracy, 2),
        "n_dfus": total_dfus,
        "n_dfu_months": total_dfu_months,
        "n_algorithms": len(algo_stats),
        "algorithm_distribution": algorithm_distribution,
    }


def format_portfolio_summary(
    assignments: pd.DataFrame,
    portfolio_stats: dict[str, Any],
) -> str:
    """Format portfolio assignments as a human-readable summary.

    Args:
        assignments: Output of optimize_greedy or optimize_constrained.
        portfolio_stats: Output of compute_portfolio_accuracy.

    Returns:
        Multi-line string summarizing the portfolio.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("PORTFOLIO OPTIMIZATION SUMMARY")
    lines.append("=" * 60)

    n_dfus = portfolio_stats.get("n_dfus", 0)
    n_algos = portfolio_stats.get("n_algorithms", 0)
    accuracy = portfolio_stats.get("portfolio_accuracy", 0.0)
    wape = portfolio_stats.get("portfolio_wape", 0.0)

    lines.append(f"Total DFUs:        {n_dfus:,}")
    lines.append(f"Algorithms used:   {n_algos}")
    lines.append(f"Portfolio accuracy: {accuracy:.2f}%")
    lines.append(f"Portfolio WAPE:     {wape:.4f}%")
    lines.append("")

    # Algorithm distribution
    dist = portfolio_stats.get("algorithm_distribution", {})
    if dist:
        lines.append("-" * 60)
        lines.append(f"{'Algorithm':<25} {'DFUs':>8} {'Share':>8} {'Accuracy':>10}")
        lines.append("-" * 60)
        for algo in sorted(dist, key=lambda a: dist[a]["n_dfus"], reverse=True):
            info = dist[algo]
            lines.append(
                f"{algo:<25} {info['n_dfus']:>8,} {info['pct']:>7.1f}% {info['accuracy']:>9.2f}%"
            )
        lines.append("")

    # Per-segment assignments
    if not assignments.empty:
        lines.append("-" * 60)
        lines.append(f"{'Segment':<22} {'Algorithm':<20} {'Acc%':>7} {'Conf':>6}")
        lines.append("-" * 60)
        for _, row in assignments.iterrows():
            archetype = row.get("archetype", "")
            algo = row.get("best_algorithm", "")
            acc = row.get("accuracy_pct", 0.0)
            conf = row.get("confidence", "")
            lines.append(f"{archetype:<22} {algo:<20} {acc:>6.2f}% {conf:>6}")

    lines.append("=" * 60)
    return "\n".join(lines)
