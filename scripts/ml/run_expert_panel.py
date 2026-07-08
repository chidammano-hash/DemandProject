#!/usr/bin/env python3
"""Expert Panel Algorithm Selection — self-contained test.

Usage:
    python -m common.ml.expert_panel.run_expert_panel
    python -m common.ml.expert_panel.run_expert_panel --n-dfus 1000 --n-timeframes 3

Tests 12 algorithms across 8 demand archetypes on a stratified golden set.
Compares the optimal algorithm mix vs Seasonal Naive, External Forecast,
and Current Tree Champion baselines.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[2]  # scripts/ml/<file>.py -> project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.ml.expert_panel.affinity_matrix import (  # noqa: E402
    build_affinity_matrix,
    compute_ceiling_accuracy,
    format_affinity_heatmap,
)
from common.ml.expert_panel.baselines import (  # noqa: E402
    predict_ridge,
    predict_rolling_mean,
    predict_seasonal_naive,
)
from common.ml.expert_panel.comparison import (  # noqa: E402
    compare_all,
    compute_portfolio_predictions,
    format_comparison_summary,
)
from common.ml.expert_panel.demand_classifier import classify_demand, get_segment_summary  # noqa: E402
from common.core.constants import FORECAST_QTY_COL  # noqa: E402
from common.services.perf_profiler import profiled_section  # noqa: E402
from common.ml.expert_panel.golden_set import (  # noqa: E402
    create_golden_set,
    create_loc_golden_set,
    load_existing_predictions,
    load_external_forecast,
    load_golden_set_data,
)
from common.ml.expert_panel.portfolio_optimizer import (  # noqa: E402
    compute_portfolio_accuracy,
    format_portfolio_summary,
    optimize_constrained,
    optimize_greedy,
)
from common.ml.expert_panel.dfu_accuracy_matrix import build_dfu_accuracy_matrix  # noqa: E402
from common.ml.expert_panel.hybrid_ensemble import compute_hybrid_predictions  # noqa: E402
from common.ml.expert_panel.meta_router import train_meta_router  # noqa: E402
from common.ml.expert_panel.statistical_models import run_statistical_models  # noqa: E402
from common.ml.expert_panel.tree_models import run_tree_models  # noqa: E402
from common.ml.expert_panel.lag_accuracy import (  # noqa: E402
    add_lag_columns,
    compute_monthly_accuracy,
    compute_overall_monthly_accuracy,
    compute_rolling_window_accuracy,
)
from common.ml.backtest_framework import generate_timeframes  # noqa: E402
from common.ml.feature_engineering import (  # noqa: E402
    build_feature_matrix,
    get_feature_columns,
    mask_future_sales,
)

logger = logging.getLogger(__name__)


def load_experiment_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load experiment configuration from YAML."""
    if config_path is None:
        config_path = ROOT / "common" / "ml" / "expert_panel" / "expert_panel_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _timeframe_predict_months(tf: dict[str, Any]) -> list[pd.Timestamp]:
    """Extract sorted list of monthly timestamps from a timeframe dict.

    ``generate_timeframes`` returns dicts with ``predict_start`` and
    ``predict_end`` as Timestamps.  This helper expands that range into
    month-start timestamps.
    """
    return list(pd.date_range(tf["predict_start"], tf["predict_end"], freq="MS"))


def save_all_outputs(
    output_dir: Path,
    config: dict[str, Any],
    golden_skus: list[str],
    classification_df: pd.DataFrame,
    all_predictions_df: pd.DataFrame,
    affinity_matrix: pd.DataFrame,
    affinity_detail: pd.DataFrame,
    assignments_df: pd.DataFrame,
    portfolio_stats: dict[str, Any],
    comparison: dict[str, Any],
    runtime_seconds: float,
    monthly_accuracy: Any = None,
    avg_3m: Any = None,
    avg_6m: Any = None,
    overall_monthly: Any = None,
) -> list[Path]:
    """Save all experiment artifacts to disk.

    Returns list of saved file paths.
    """
    saved: list[Path] = []

    # Golden set SKUs
    golden_path = output_dir / "golden_set_skus.csv"
    pd.DataFrame({"sku_ck": golden_skus}).to_csv(golden_path, index=False)
    saved.append(golden_path)
    logger.info("Saved golden set to %s", golden_path)

    # Classification
    cls_path = output_dir / "classification.csv"
    classification_df.to_csv(cls_path, index=False)
    saved.append(cls_path)
    logger.info("Saved classification to %s", cls_path)

    # All predictions
    preds_path = output_dir / "all_predictions.parquet"
    all_predictions_df.to_parquet(preds_path, index=False)
    saved.append(preds_path)
    logger.info("Saved predictions to %s (%d rows)", preds_path, len(all_predictions_df))

    # Affinity matrix
    aff_path = output_dir / "affinity_matrix.csv"
    affinity_matrix.to_csv(aff_path)
    saved.append(aff_path)

    aff_detail_path = output_dir / "affinity_detail.csv"
    affinity_detail.to_csv(aff_detail_path, index=False)
    saved.append(aff_detail_path)

    # Assignments
    assign_path = output_dir / "assignments.csv"
    assignments_df.to_csv(assign_path, index=False)
    saved.append(assign_path)

    # Portfolio stats and comparison as JSON
    stats_path = output_dir / "portfolio_stats.json"
    stats_path.write_text(json.dumps(_make_serializable(portfolio_stats), indent=2))
    saved.append(stats_path)

    comp_path = output_dir / "comparison.json"
    comp_path.write_text(json.dumps(_make_serializable(comparison), indent=2))
    saved.append(comp_path)

    # Metadata
    exp_cfg = config["experiment"]
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "runtime_seconds": round(runtime_seconds, 1),
                "n_dfus": len(golden_skus),
                "n_predictions": len(all_predictions_df),
                "n_algorithms": int(all_predictions_df["algorithm_id"].nunique())
                if not all_predictions_df.empty
                else 0,
                "n_archetypes": int(classification_df["archetype"].nunique())
                if not classification_df.empty
                else 0,
                "loc_filter": exp_cfg.get("loc_filter"),
            },
            indent=2,
        )
    )
    saved.append(meta_path)

    # Monthly accuracy diagnostics (per-month + rolling 3m/6m + overall per-algo)
    if monthly_accuracy is not None:
        ma_path = output_dir / "monthly_accuracy.json"
        ma_path.write_text(
            json.dumps(
                _make_serializable(
                    {
                        "monthly_accuracy": monthly_accuracy,
                        "avg_3m": avg_3m,
                        "avg_6m": avg_6m,
                        "overall_monthly": overall_monthly,
                    }
                ),
                indent=2,
            )
        )
        saved.append(ma_path)

    logger.info("Saved %d output files to %s", len(saved), output_dir)
    return saved


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return obj


def generate_report(
    classification_df: pd.DataFrame,
    affinity_matrix: pd.DataFrame,
    assignments_df: pd.DataFrame,
    portfolio_stats: dict[str, Any],
    comparison: dict[str, Any],
    runtime_seconds: float,
    monthly_accuracy: Any = None,
    avg_3m: Any = None,
    avg_6m: Any = None,
    overall_monthly: Any = None,
) -> str:
    """Generate a plain-text experiment summary report."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("EXPERT PANEL ALGORITHM SELECTION — EXPERIMENT REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Summary stats
    n_dfus = len(classification_df) if not classification_df.empty else 0
    n_archetypes = classification_df["archetype"].nunique() if not classification_df.empty else 0
    lines.append(f"DFUs tested:     {n_dfus:,}")
    lines.append(f"Archetypes:      {n_archetypes}")
    lines.append(f"Runtime:         {runtime_seconds / 60:.1f} minutes")
    lines.append("")

    # Archetype distribution
    lines.append("-" * 40)
    lines.append("DEMAND ARCHETYPE DISTRIBUTION")
    lines.append("-" * 40)
    if not classification_df.empty:
        dist = classification_df["archetype"].value_counts().sort_index()
        for arch, count in dist.items():
            pct = 100.0 * count / n_dfus if n_dfus else 0
            lines.append(f"  {arch:<30s} {count:>5d}  ({pct:5.1f}%)")
    lines.append("")

    # Affinity matrix (text representation)
    lines.append("-" * 40)
    lines.append("AFFINITY MATRIX (accuracy % per segment x algorithm)")
    lines.append("-" * 40)
    if not affinity_matrix.empty:
        lines.append(affinity_matrix.round(1).to_string())
    lines.append("")

    # Portfolio assignments
    lines.append("-" * 40)
    lines.append("PORTFOLIO ASSIGNMENTS")
    lines.append("-" * 40)
    if not assignments_df.empty:
        for _, row in assignments_df.iterrows():
            conf = row.get("confidence", "?")
            lines.append(
                f"  {row.get('archetype', '?'):<30s} -> {row.get('best_algorithm', '?'):<20s} "
                f"acc={row.get('accuracy_pct', 0):.1f}%  ({conf})"
            )
    lines.append("")

    # Portfolio accuracy
    lines.append("-" * 40)
    lines.append("PORTFOLIO ACCURACY")
    lines.append("-" * 40)
    p_acc = portfolio_stats.get("portfolio_accuracy", 0)
    p_wape = portfolio_stats.get("portfolio_wape", 0)
    n_algos = portfolio_stats.get("n_algorithms", 0)
    lines.append(f"  Overall accuracy:   {p_acc:.1f}%")
    lines.append(f"  Overall WAPE:       {p_wape:.1f}%")
    lines.append(f"  Algorithms used:    {n_algos}")
    lines.append("")

    # Monthly accuracy (mean per algorithm across months)
    if overall_monthly:
        lines.append("-" * 40)
        lines.append("MONTHLY ACCURACY (mean per algorithm)")
        lines.append("-" * 40)
        try:
            ranked = sorted(
                overall_monthly.items(),
                key=lambda kv: kv[1].get("mean_monthly_accuracy", 0),
                reverse=True,
            )
            for algo, stats in ranked:
                mma = stats.get("mean_monthly_accuracy", 0) if isinstance(stats, dict) else 0
                lines.append(f"  {algo!s:<30s} {mma:5.1f}%")
        except (AttributeError, TypeError):
            lines.append("  (unavailable)")
        lines.append("")

    # Comparison vs baselines
    lines.append("-" * 40)
    lines.append("COMPARISON VS BASELINES")
    lines.append("-" * 40)

    portfolio_metrics = comparison.get("portfolio", {})
    baselines = comparison.get("baselines", {})
    lift = comparison.get("lift", {})

    lines.append(f"  Portfolio:          {portfolio_metrics.get('accuracy', 0):.1f}%")

    for name, metrics in baselines.items():
        if metrics is not None:
            lines.append(f"  {name:<22s}{metrics.get('accuracy', 0):.1f}%")

    lines.append("")
    for key, val in lift.items():
        if val is not None:
            lines.append(f"  Lift {key}: {val:+d} bps")

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    return "\n".join(lines)


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run the full Expert Panel experiment.

    Args:
        config: Experiment configuration from config.yaml.

    Returns:
        Full comparison results dict.
    """
    t0 = time.perf_counter()
    exp = config["experiment"]
    output_dir = Path(ROOT) / exp["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    loc_filter: str | None = exp.get("loc_filter")

    # -- Step 1: Build golden set --------------------------------------------
    with profiled_section("step_1_golden_set"):
        t1 = time.perf_counter()
        if loc_filter:
            logger.info("Step 1/10: Using all DFUs at loc='%s' (no sampling)...", loc_filter)
            golden_skus = create_loc_golden_set(loc=loc_filter, output_dir=output_dir)
        else:
            logger.info("Step 1/10: Sampling golden set (%d DFUs)...", exp["n_dfus"])
            golden_skus = create_golden_set(
                n_dfus=exp["n_dfus"],
                sampling_method=exp["sampling_method"],
                seed=exp["seed"],
                output_dir=output_dir,
            )
        logger.info("  Golden set: %d DFUs in %.1fs", len(golden_skus), time.perf_counter() - t1)

    # -- Step 2: Load data ---------------------------------------------------
    with profiled_section("step_2_load_data"):
        logger.info("Step 2/10: Loading data for golden set...")
        t2 = time.perf_counter()
        sales_df, dfu_attrs, item_attrs = load_golden_set_data(golden_skus)
        logger.info(
            "  Loaded %d sales rows, %d DFU attrs, %d item attrs in %.1fs",
            len(sales_df),
            len(dfu_attrs),
            len(item_attrs),
            time.perf_counter() - t2,
        )

    # -- Step 3: Classify demand ---------------------------------------------
    with profiled_section("step_3_classify_demand"):
        logger.info("Step 3/10: Classifying demand archetypes...")
        t3 = time.perf_counter()
        dc = config["demand_classification"]
        classification_df = classify_demand(
            sales_df,
            adi_threshold=dc["adi_threshold"],
            cv2_threshold=dc["cv2_threshold"],
            high_volume_percentile=dc["high_volume_percentile"],
            min_history_months=dc["min_history_months"],
        )
        seg_summary = get_segment_summary(classification_df)
        logger.info(
            "  Classified %d DFUs into %d archetypes in %.1fs",
            len(classification_df),
            classification_df["archetype"].nunique(),
            time.perf_counter() - t3,
        )
    for _, row in seg_summary.iterrows():
        logger.info(
            "    %-25s %5d DFUs  mean_demand=%.0f",
            row.get("archetype", "?"),
            row.get("n_dfus", 0),
            row.get("mean_demand", 0),
        )

    # -- Step 4: Generate timeframes -----------------------------------------
    with profiled_section("step_4_timeframes"):
        logger.info("Step 4/10: Generating %d timeframes...", exp["n_timeframes"])
        all_months = sorted(sales_df["startdate"].unique())
        earliest = pd.Timestamp(min(all_months))
        latest = pd.Timestamp(max(all_months))
        timeframes = generate_timeframes(earliest, latest, n=exp["n_timeframes"])
        for tf in timeframes:
            predict_months = _timeframe_predict_months(tf)
            logger.info(
                "  Timeframe %s: train_end=%s, predict=%s..%s (%d months)",
                tf["label"],
                tf["train_end"].strftime("%Y-%m-%d"),
                predict_months[0].strftime("%Y-%m-%d") if predict_months else "?",
                predict_months[-1].strftime("%Y-%m-%d") if predict_months else "?",
                len(predict_months),
            )

    # -- Step 5: Build feature matrix (for tree models + Ridge) --------------
    with profiled_section("step_5_feature_matrix"):
        logger.info("Step 5/10: Building feature matrix...")
        t5 = time.perf_counter()
        grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, all_months)
        feature_cols = get_feature_columns(grid)
        logger.info(
            "  Built grid: %d rows x %d features in %.1fs",
            len(grid),
            len(feature_cols),
            time.perf_counter() - t5,
        )

    # -- Step 6-7: Run all algorithms per timeframe --------------------------
    with profiled_section("step_6_7_algorithms"):
        logger.info("Step 6-7/10: Running algorithms across %d timeframes...", len(timeframes))
        all_predictions: list[pd.DataFrame] = []
        stat_config = {
            k: v for k, v in config["statistical_models"].items() if v.get("enabled", False)
        }
        tree_config = {k: v for k, v in config["tree_models"].items() if v.get("enabled", False)}
        baseline_config = config["baselines"]

        for tf_idx, tf in enumerate(timeframes):
            with profiled_section(f"timeframe_{tf['label']}"):
                tf_start = time.perf_counter()
                train_end: pd.Timestamp = tf["train_end"]
                predict_months = _timeframe_predict_months(tf)

                if not predict_months:
                    logger.warning("  Timeframe %s: no predict months, skipping", tf["label"])
                    continue

                logger.info(
                    "  Timeframe %s (%d/%d): train_end=%s, predicting %d months...",
                    tf["label"],
                    tf_idx + 1,
                    len(timeframes),
                    train_end.strftime("%Y-%m-%d"),
                    len(predict_months),
                )

                # Filter training sales (up to train_end inclusive)
                train_sales = sales_df[sales_df["startdate"] <= train_end].copy()

                # 6a. Statistical models (parallel per-DFU)
                if stat_config:
                    with profiled_section("statistical_models"):
                        logger.info("    Running %d statistical models...", len(stat_config))
                        ts = time.perf_counter()
                        stat_preds = run_statistical_models(
                            train_sales,
                            predict_months,
                            stat_config,
                            n_workers=exp.get("n_workers", 8),
                        )
                        stat_preds["timeframe_idx"] = tf_idx
                        all_predictions.append(stat_preds)
                        logger.info(
                            "    Statistical: %d predictions in %.1fs",
                            len(stat_preds),
                            time.perf_counter() - ts,
                        )

                # 6b. Baselines — Seasonal Naive
                if baseline_config.get("seasonal_naive", {}).get("enabled", False):
                    with profiled_section("seasonal_naive"):
                        ts = time.perf_counter()
                        naive_preds = predict_seasonal_naive(train_sales, predict_months)
                        naive_preds["timeframe_idx"] = tf_idx
                        all_predictions.append(naive_preds)
                        logger.info(
                            "    Seasonal Naive: %d predictions in %.1fs",
                            len(naive_preds),
                            time.perf_counter() - ts,
                        )

                # 6b. Baselines — Rolling Mean
                if baseline_config.get("rolling_mean", {}).get("enabled", False):
                    with profiled_section("rolling_mean"):
                        ts = time.perf_counter()
                        rm_preds = predict_rolling_mean(
                            train_sales,
                            predict_months,
                            window=baseline_config["rolling_mean"].get("window", 6),
                        )
                        rm_preds["timeframe_idx"] = tf_idx
                        all_predictions.append(rm_preds)
                        logger.info(
                            "    Rolling Mean: %d predictions in %.1fs",
                            len(rm_preds),
                            time.perf_counter() - ts,
                        )

                # 6c. Tree models (per-cluster, using masked grid)
                if tree_config:
                    with profiled_section("tree_models"):
                        logger.info("    Running %d tree models...", len(tree_config))
                        ts = time.perf_counter()
                        masked_grid = mask_future_sales(grid.copy(), train_end)
                        tree_preds = run_tree_models(
                            masked_grid,
                            train_end,
                            predict_months,
                            tree_config,
                            classification_df=classification_df,
                        )
                        tree_preds["timeframe_idx"] = tf_idx
                        all_predictions.append(tree_preds)
                        logger.info(
                            "    Tree models: %d predictions in %.1fs",
                            len(tree_preds),
                            time.perf_counter() - ts,
                        )

                # 6d. Ridge Regression
                if baseline_config.get("ridge", {}).get("enabled", False):
                    with profiled_section("ridge"):
                        ts = time.perf_counter()
                        masked_grid_ridge = mask_future_sales(grid.copy(), train_end)
                        train_grid = masked_grid_ridge[masked_grid_ridge["startdate"] <= train_end]
                        pred_grid = masked_grid_ridge[
                            masked_grid_ridge["startdate"].isin(predict_months)
                        ]
                        cat_cols = [
                            c
                            for c in feature_cols
                            if c in ["ml_cluster", "region", "brand", "abc_vol"]
                        ]
                        ridge_preds = predict_ridge(
                            train_grid,
                            pred_grid,
                            feature_cols,
                            cat_cols,
                            alpha=baseline_config["ridge"].get("alpha", 1.0),
                        )
                        ridge_preds["timeframe_idx"] = tf_idx
                        all_predictions.append(ridge_preds)
                        logger.info(
                            "    Ridge: %d predictions in %.1fs",
                            len(ridge_preds),
                            time.perf_counter() - ts,
                        )

                logger.info(
                    "  Timeframe %s complete in %.1fs",
                    tf["label"],
                    time.perf_counter() - tf_start,
                )

    # Combine all predictions
    if all_predictions:
        all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    else:
        all_predictions_df = pd.DataFrame(
            columns=["sku_ck", "startdate", FORECAST_QTY_COL, "algorithm_id", "timeframe_idx"],
        )
    logger.info(
        "Total predictions: %d rows across %d algorithms",
        len(all_predictions_df),
        all_predictions_df["algorithm_id"].nunique() if not all_predictions_df.empty else 0,
    )

    # -- Step 8: Load comparison baselines from DB ---------------------------
    with profiled_section("step_8_load_comparison_baselines"):
        logger.info("Step 8/10: Loading external forecast and existing predictions...")
        t8 = time.perf_counter()
        # Collect all predict months across timeframes
        all_predict_months_set: set[pd.Timestamp] = set()
        for tf in timeframes:
            all_predict_months_set.update(_timeframe_predict_months(tf))
        all_predict_ts = sorted(all_predict_months_set)

        external_df = load_external_forecast(golden_skus, all_predict_ts)
        existing_df = load_existing_predictions(
            golden_skus,
            all_predict_ts,
            config["comparison"]["champion_model_ids"],
        )
        logger.info(
            "  External forecast: %d rows, Existing predictions: %d rows in %.1fs",
            len(external_df) if external_df is not None and not external_df.empty else 0,
            len(existing_df) if existing_df is not None and not existing_df.empty else 0,
            time.perf_counter() - t8,
        )

    # -- Step 9: Build affinity matrix + optimize ----------------------------
    with profiled_section("step_9_affinity_optimize"):
        logger.info("Step 9/10: Building affinity matrix and optimizing portfolio...")
        t9 = time.perf_counter()

        # Actuals for the predict months
        actuals_df = sales_df[sales_df["startdate"].isin(all_predict_ts)][
            ["sku_ck", "startdate", "qty"]
        ].copy()

        affinity_matrix, affinity_detail = build_affinity_matrix(
            all_predictions_df,
            actuals_df,
            classification_df,
        )
        logger.info(
            "  Affinity matrix: %d segments x %d algorithms",
            affinity_matrix.shape[0],
            affinity_matrix.shape[1],
        )
        logger.info("\n%s", format_affinity_heatmap(affinity_matrix, affinity_detail))

        # Portfolio optimization
        opt_config = config.get("portfolio_optimizer", {})
        max_algos = opt_config.get("max_algorithms")
        min_seg = opt_config.get("min_segment_dfus", 30)

        min_cov = opt_config.get("min_dfu_coverage_pct", 0.5)
        cov_weighted = opt_config.get("coverage_weighted", True)
        n_floor = opt_config.get("naive_floor", True)
        if max_algos:
            assignments_df = optimize_constrained(
                affinity_matrix,
                affinity_detail,
                max_algorithms=max_algos,
                min_segment_dfus=min_seg,
                min_dfu_coverage_pct=min_cov,
                coverage_weighted=cov_weighted,
                naive_floor=n_floor,
            )
        else:
            assignments_df = optimize_greedy(
                affinity_matrix,
                affinity_detail,
                min_segment_dfus=min_seg,
                min_dfu_coverage_pct=min_cov,
                coverage_weighted=cov_weighted,
                naive_floor=n_floor,
            )

        portfolio_stats = compute_portfolio_accuracy(assignments_df, affinity_detail)
        logger.info("\n%s", format_portfolio_summary(assignments_df, portfolio_stats))

        # Ceiling accuracy (theoretical upper bound)
        ceiling_df = compute_ceiling_accuracy(all_predictions_df, actuals_df, classification_df)
        logger.info(
            "  Ceiling accuracy computed for %d segments in %.1fs",
            len(ceiling_df),
            time.perf_counter() - t9,
        )

    # -- Step 9b: Per-DFU hybrid ensemble ------------------------------------
    hybrid_cfg = config.get("hybrid_ensemble", {})
    hybrid_preds = pd.DataFrame(columns=["sku_ck", "startdate", FORECAST_QTY_COL, "algorithm_id"])
    dfu_accuracy_matrix = pd.DataFrame(
        columns=["sku_ck", "algorithm_id", "wape", "accuracy_pct", "n_months"]
    )
    hybrid_routing_df = pd.DataFrame(columns=["sku_ck", "predicted_algorithm", "confidence"])
    if hybrid_cfg.get("enabled", True) and not all_predictions_df.empty:
        with profiled_section("step_9b_hybrid_ensemble"):
            logger.info("Step 9b/10: Building per-DFU hybrid ensemble...")
            t9b = time.perf_counter()
            try:
                dfu_accuracy_matrix = build_dfu_accuracy_matrix(
                    all_predictions_df,
                    actuals_df,
                    min_n_months=hybrid_cfg.get("min_n_months", 2),
                )
                meta_model = train_meta_router(
                    dfu_accuracy_matrix,
                    dfu_attrs,
                    classification_df,
                    hybrid_config=hybrid_cfg,
                )
                hybrid_preds = compute_hybrid_predictions(
                    all_predictions_df=all_predictions_df,
                    dfu_accuracy_matrix=dfu_accuracy_matrix,
                    dfu_attrs=dfu_attrs,
                    classification_df=classification_df,
                    meta_model=meta_model,
                    blend_top_k=hybrid_cfg.get("blend_top_k", 3),
                    confidence_threshold=hybrid_cfg.get("confidence_threshold", 0.6),
                )
                from common.ml.expert_panel.meta_router import predict_meta_router  # noqa: E402

                hybrid_routing_df = predict_meta_router(meta_model, dfu_attrs, classification_df)
                logger.info(
                    "  Hybrid ensemble: %d predictions, %d DFUs in %.1fs",
                    len(hybrid_preds),
                    hybrid_preds["sku_ck"].nunique() if not hybrid_preds.empty else 0,
                    time.perf_counter() - t9b,
                )
            except (ValueError, RuntimeError, KeyError) as exc:
                logger.warning("  Hybrid ensemble skipped: %s", exc)

    # -- Step 10: Compare and report -----------------------------------------
    with profiled_section("step_10_compare_report"):
        logger.info("Step 10/10: Comparing portfolio vs baselines...")
        t10 = time.perf_counter()

        portfolio_preds = compute_portfolio_predictions(
            all_predictions_df,
            assignments_df,
            classification_df,
        )

        # Get naive predictions specifically for comparison
        naive_preds_all = all_predictions_df[all_predictions_df["algorithm_id"] == "seasonal_naive"]

        # Handle empty DataFrames for optional baselines
        ext_for_compare = (
            external_df if (external_df is not None and not external_df.empty) else None
        )
        exist_for_compare = (
            existing_df if (existing_df is not None and not existing_df.empty) else None
        )

        comparison = compare_all(
            portfolio_predictions=portfolio_preds,
            naive_predictions=naive_preds_all,
            external_predictions=ext_for_compare,
            existing_predictions=exist_for_compare,
            actuals_df=actuals_df,
            classification_df=classification_df,
            all_predictions_df=all_predictions_df,
        )

        logger.info("\n%s", format_comparison_summary(comparison))

        # Inject hybrid ensemble metrics into the comparison dict
        if not hybrid_preds.empty:
            from common.ml.expert_panel.comparison import compute_baseline_accuracy  # noqa: E402
            import numpy as np  # noqa: E402 (already imported at top but guard for clarity)

            hybrid_metrics = compute_baseline_accuracy(
                hybrid_preds, actuals_df, "hybrid", classification_df
            )
            comparison["baselines"]["hybrid"] = hybrid_metrics
            port_acc = comparison["portfolio"]["accuracy_pct"]
            hyb_acc = hybrid_metrics["accuracy_pct"]
            naive_acc = comparison["baselines"]["seasonal_naive"]["accuracy_pct"]
            if not (np.isnan(hyb_acc) or np.isnan(port_acc)):
                comparison["lift"]["hybrid_vs_portfolio_bps"] = round((hyb_acc - port_acc) * 100)
            if not (np.isnan(hyb_acc) or np.isnan(naive_acc)):
                comparison["lift"]["hybrid_vs_naive_bps"] = round((hyb_acc - naive_acc) * 100)
            logger.info(
                "  Hybrid ensemble: Accuracy=%.2f%%  WAPE=%.2f%%  "
                "vs Portfolio=%+d bps  vs Naive=%+d bps",
                hyb_acc,
                hybrid_metrics["wape"],
                comparison["lift"].get("hybrid_vs_portfolio_bps", 0),
                comparison["lift"].get("hybrid_vs_naive_bps", 0),
            )

        logger.info("  Comparison complete in %.1fs", time.perf_counter() - t10)

    # Add execution-lag columns to combined predictions
    tf_train_end_map = {idx: tf["train_end"] for idx, tf in enumerate(timeframes)}
    exec_lag_map: dict[str, int] = (
        dfu_attrs.set_index("sku_ck")["execution_lag"].to_dict()
        if "execution_lag" in dfu_attrs.columns
        else {}
    )
    if exec_lag_map and not all_predictions_df.empty:
        all_predictions_df = add_lag_columns(all_predictions_df, tf_train_end_map, exec_lag_map)

    # Monthly accuracy — execution-lag-matched, only months with full lag coverage
    monthly_accuracy = compute_monthly_accuracy(
        all_predictions_df,
        actuals_df,
        execution_lag_only=True,
        require_all_lags=True,
    )
    avg_3m = compute_rolling_window_accuracy(monthly_accuracy, 3)
    avg_6m = compute_rolling_window_accuracy(monthly_accuracy, 6)
    overall_monthly = compute_overall_monthly_accuracy(monthly_accuracy)
    if monthly_accuracy:
        logger.info(
            "Monthly accuracy (exec-lag): %d months | Best overall: %s (%.1f%% mean-monthly)",
            len(monthly_accuracy),
            max(overall_monthly.items(), key=lambda x: x[1]["mean_monthly_accuracy"])[0]
            if overall_monthly
            else "n/a",
            max(v["mean_monthly_accuracy"] for v in overall_monthly.values())
            if overall_monthly
            else 0.0,
        )

    # Save hybrid artifacts alongside the standard outputs
    if not dfu_accuracy_matrix.empty:
        dfu_acc_path = output_dir / "dfu_accuracy_matrix.csv"
        dfu_accuracy_matrix.to_csv(dfu_acc_path, index=False)
        logger.info("Saved DFU accuracy matrix to %s", dfu_acc_path)
    if not hybrid_routing_df.empty:
        routing_path = output_dir / "hybrid_assignments.csv"
        hybrid_routing_df.to_csv(routing_path, index=False)
        logger.info("Saved hybrid assignments to %s", routing_path)

    # Save outputs
    runtime = time.perf_counter() - t0
    save_all_outputs(
        output_dir=output_dir,
        config=config,
        golden_skus=golden_skus,
        classification_df=classification_df,
        all_predictions_df=all_predictions_df,
        affinity_matrix=affinity_matrix,
        affinity_detail=affinity_detail,
        assignments_df=assignments_df,
        portfolio_stats=portfolio_stats,
        comparison=comparison,
        runtime_seconds=runtime,
        monthly_accuracy=monthly_accuracy,
        avg_3m=avg_3m,
        avg_6m=avg_6m,
        overall_monthly=overall_monthly,
    )

    report_text = generate_report(
        classification_df,
        affinity_matrix,
        assignments_df,
        portfolio_stats,
        comparison,
        runtime,
        monthly_accuracy=monthly_accuracy,
        avg_3m=avg_3m,
        avg_6m=avg_6m,
        overall_monthly=overall_monthly,
    )
    report_path = output_dir / "experiment_report.txt"
    report_path.write_text(report_text)
    logger.info("Report saved to %s", report_path)
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE in %.1f minutes", runtime / 60)
    logger.info("=" * 60)

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Expert Panel Algorithm Selection Test")
    parser.add_argument("--config", type=Path, help="Path to config.yaml")
    parser.add_argument("--n-dfus", type=int, help="Override number of DFUs")
    parser.add_argument("--n-timeframes", type=int, help="Override number of timeframes")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument(
        "--loc",
        type=str,
        help="Run on all DFUs at a specific location (e.g. 1401-BULK). "
        "Overrides --n-dfus and sampling; uses every DFU at that loc.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_experiment_config(args.config)

    # Apply CLI overrides
    if args.n_dfus:
        config["experiment"]["n_dfus"] = args.n_dfus
    if args.n_timeframes:
        config["experiment"]["n_timeframes"] = args.n_timeframes
    if args.seed:
        config["experiment"]["seed"] = args.seed
    if args.loc:
        config["experiment"]["loc_filter"] = args.loc

    loc_filter = config["experiment"].get("loc_filter")
    logger.info("=" * 60)
    logger.info("EXPERT PANEL ALGORITHM SELECTION TEST")
    logger.info("=" * 60)
    if loc_filter:
        logger.info(
            "Loc filter: %s | Timeframes: %d | Seed: %d",
            loc_filter,
            config["experiment"]["n_timeframes"],
            config["experiment"]["seed"],
        )
    else:
        logger.info(
            "DFUs: %d | Timeframes: %d | Seed: %d",
            config["experiment"]["n_dfus"],
            config["experiment"]["n_timeframes"],
            config["experiment"]["seed"],
        )

    run_experiment(config)


if __name__ == "__main__":
    main()
