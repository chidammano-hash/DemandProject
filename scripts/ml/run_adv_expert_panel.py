#!/usr/bin/env python3
"""Advanced Expert Panel Algorithm Selection — extended test.

Extends the base Expert Panel (common.ml.expert_panel) with:
- 6 statistical upgrades (AutoCES, DynamicTheta, IMAPA, TSB, ADIDA, MSTL)
- 8 deep learning models (N-BEATS, N-HiTS, TFT, DeepAR, TiDE, TCN, PatchTST, iTransformer)
- 5 foundation models (Chronos, TimesFM, Moirai, TimeGPT, Lag-Llama)
- 2 DL baselines (DLinear, NLinear)
- Cross-sectional hierarchical reconciliation

Usage:
    python -m common.ml.expert_panel.run_adv_expert_panel
    python -m common.ml.expert_panel.run_adv_expert_panel --n-dfus 1000 --n-timeframes 3
"""

import argparse
import json
import logging
import multiprocessing
import os
import sys
import time

# macOS fork() safety: torch loaded in parent process causes SIGSEGV in
# ProcessPoolExecutor children. Force 'spawn' start method and set the
# macOS ObjC fork-safety env var before anything else imports torch.
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[2]  # scripts/ml/<file>.py -> project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Base common.ml.expert_panel modules (reused) ---
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
from common.ml.expert_panel.demand_classifier import (  # noqa: E402
    classify_demand,
    get_segment_summary,
)
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
from common.ml.expert_panel.meta_router import predict_meta_router, train_meta_router  # noqa: E402
from common.ml.expert_panel.statistical_models import run_statistical_models  # noqa: E402
from common.ml.expert_panel.tree_models import run_tree_models  # noqa: E402

# --- Advanced modules (NEW) ---
from common.ml.expert_panel.dl_baselines import (  # noqa: E402
    predict_dlinear,
    predict_nlinear,
)
from common.ml.expert_panel.dl_models import run_dl_models  # noqa: E402
from common.ml.expert_panel.foundation_models import run_foundation_models  # noqa: E402
from common.ml.expert_panel.lag_accuracy import (  # noqa: E402
    add_lag_columns,
    compute_monthly_accuracy,
    compute_overall_monthly_accuracy,
    compute_rolling_window_accuracy,
)
from common.ml.expert_panel.statistical_upgrades import run_statistical_upgrades  # noqa: E402

# --- Common utilities ---
from common.core.constants import FORECAST_QTY_COL  # noqa: E402
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
        config_path = ROOT / "common" / "ml" / "expert_panel" / "adv_expert_panel_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _timeframe_predict_months(tf: dict[str, Any]) -> list[pd.Timestamp]:
    """Extract sorted list of monthly timestamps from a timeframe dict."""
    return list(pd.date_range(tf["predict_start"], tf["predict_end"], freq="MS"))


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


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run the full Advanced Expert Panel experiment.

    14-step pipeline:
    1. Sample golden set
    2. Load data
    3. Classify demand
    4. Generate timeframes
    5. Build feature matrix
    6-11. Per-timeframe loop (all algorithm groups):
        6. Base statistical models
        7. Base tree models + baselines (tree: per-cluster, not DL/FM)
        8. Statistical upgrades
        9. DL baselines (DLinear, NLinear)
        10. Deep learning models (per timeframe, not per cluster)  [NEW]
        11. Foundation models (zero-shot, per timeframe)  [NEW]
    12. Load comparison baselines from DB
    13. Build affinity matrix + optimize portfolio
    13b. Per-DFU hybrid ensemble
    14. Compare and report
    """
    t0 = time.time()
    exp = config["experiment"]
    output_dir = Path(ROOT) / exp["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    loc_filter: str | None = exp.get("loc_filter")

    # -- Step 1: Build golden set ------------------------------------------
    t1 = time.time()
    if loc_filter:
        logger.info("Step 1/14: Using all DFUs at loc='%s' (no sampling)...", loc_filter)
        golden_skus = create_loc_golden_set(loc=loc_filter, output_dir=output_dir)
    else:
        logger.info("Step 1/14: Sampling golden set (%d DFUs)...", exp["n_dfus"])
        golden_skus = create_golden_set(
            n_dfus=exp["n_dfus"],
            sampling_method=exp["sampling_method"],
            seed=exp["seed"],
            output_dir=output_dir,
        )
    logger.info("  Golden set: %d DFUs in %.1fs", len(golden_skus), time.time() - t1)

    # -- Step 2: Load data -------------------------------------------------
    logger.info("Step 2/14: Loading data for golden set...")
    t2 = time.time()
    sales_df, dfu_attrs, item_attrs = load_golden_set_data(golden_skus)
    logger.info(
        "  Loaded %d sales rows, %d DFU attrs, %d item attrs in %.1fs",
        len(sales_df), len(dfu_attrs), len(item_attrs), time.time() - t2,
    )

    # -- Step 3: Classify demand -------------------------------------------
    logger.info("Step 3/14: Classifying demand archetypes...")
    t3 = time.time()
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
        time.time() - t3,
    )
    for _, row in seg_summary.iterrows():
        logger.info(
            "    %-25s %5d DFUs  mean_demand=%.0f",
            row.get("archetype", "?"),
            row.get("n_dfus", 0),
            row.get("mean_demand", 0),
        )

    # -- Step 4: Generate timeframes ---------------------------------------
    logger.info("Step 4/14: Generating %d timeframes...", exp["n_timeframes"])
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

    # -- Step 5: Build feature matrix --------------------------------------
    logger.info("Step 5/14: Building feature matrix...")
    t5 = time.time()
    grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, all_months)
    feature_cols = get_feature_columns(grid)
    logger.info(
        "  Built grid: %d rows x %d features in %.1fs",
        len(grid), len(feature_cols), time.time() - t5,
    )

    # -- Steps 6-11: Per-timeframe algorithms ------------------------------
    logger.info("Steps 6-11/14: Running per-timeframe algorithms...")
    all_predictions: list[pd.DataFrame] = []

    stat_config = {
        k: v for k, v in config.get("statistical_models", {}).items()
        if v.get("enabled", False)
    }
    tree_config = {
        k: v for k, v in config.get("tree_models", {}).items()
        if v.get("enabled", False)
    }
    baseline_config = config.get("baselines", {})
    stat_upgrade_config = {
        k: v for k, v in config.get("statistical_upgrades", {}).items()
        if v.get("enabled", False)
    }
    dl_config = {
        k: v for k, v in config.get("deep_learning", {}).items()
        if v.get("enabled", False)
    }
    fm_config = {
        k: v for k, v in config.get("foundation_models", {}).items()
        if v.get("enabled", False)
    }

    for tf_idx, tf in enumerate(timeframes):
        tf_start = time.time()
        train_end: pd.Timestamp = tf["train_end"]
        predict_months = _timeframe_predict_months(tf)

        if not predict_months:
            logger.warning("  Timeframe %s: no predict months, skipping", tf["label"])
            continue

        logger.info(
            "  Timeframe %s (%d/%d): train_end=%s, predicting %d months...",
            tf["label"], tf_idx + 1, len(timeframes),
            train_end.strftime("%Y-%m-%d"), len(predict_months),
        )

        train_sales = sales_df[sales_df["startdate"] <= train_end].copy()

        # Step 6: Base statistical models
        if stat_config:
            logger.info("    [6] Running %d base statistical models...", len(stat_config))
            ts = time.time()
            stat_preds = run_statistical_models(
                train_sales, predict_months, stat_config,
                n_workers=exp.get("n_workers", 8),
            )
            stat_preds["timeframe_idx"] = tf_idx
            all_predictions.append(stat_preds)
            logger.info("    Base statistical: %d preds in %.1fs", len(stat_preds), time.time() - ts)

        # Step 7: Base tree models + baselines
        if baseline_config.get("seasonal_naive", {}).get("enabled", False):
            ts = time.time()
            naive_preds = predict_seasonal_naive(train_sales, predict_months)
            naive_preds["timeframe_idx"] = tf_idx
            all_predictions.append(naive_preds)
            logger.info("    Seasonal Naive: %d preds in %.1fs", len(naive_preds), time.time() - ts)

        if baseline_config.get("rolling_mean", {}).get("enabled", False):
            ts = time.time()
            rm_preds = predict_rolling_mean(
                train_sales, predict_months,
                window=baseline_config["rolling_mean"].get("window", 6),
            )
            rm_preds["timeframe_idx"] = tf_idx
            all_predictions.append(rm_preds)
            logger.info("    Rolling Mean: %d preds in %.1fs", len(rm_preds), time.time() - ts)

        if tree_config:
            logger.info("    [7] Running %d tree models...", len(tree_config))
            ts = time.time()
            masked_grid = mask_future_sales(grid.copy(), train_end)
            tree_preds = run_tree_models(
                masked_grid, train_end, predict_months, tree_config,
                classification_df=classification_df,
            )
            tree_preds["timeframe_idx"] = tf_idx
            all_predictions.append(tree_preds)
            logger.info("    Tree models: %d preds in %.1fs", len(tree_preds), time.time() - ts)

        if baseline_config.get("ridge", {}).get("enabled", False):
            ts = time.time()
            masked_grid_ridge = mask_future_sales(grid.copy(), train_end)
            train_grid = masked_grid_ridge[masked_grid_ridge["startdate"] <= train_end]
            pred_grid = masked_grid_ridge[masked_grid_ridge["startdate"].isin(predict_months)]
            cat_cols = [c for c in feature_cols if c in ["ml_cluster", "region", "brand", "abc_vol"]]
            ridge_preds = predict_ridge(
                train_grid, pred_grid, feature_cols, cat_cols,
                alpha=baseline_config["ridge"].get("alpha", 1.0),
            )
            ridge_preds["timeframe_idx"] = tf_idx
            all_predictions.append(ridge_preds)
            logger.info("    Ridge: %d preds in %.1fs", len(ridge_preds), time.time() - ts)

        # Step 8: Statistical upgrades (NEW)
        if stat_upgrade_config:
            logger.info("    [8] Running %d statistical upgrades...", len(stat_upgrade_config))
            ts = time.time()
            upgrade_preds = run_statistical_upgrades(
                train_sales, predict_months, stat_upgrade_config,
                n_workers=exp.get("n_workers", 8),
            )
            upgrade_preds["timeframe_idx"] = tf_idx
            all_predictions.append(upgrade_preds)
            logger.info("    Statistical upgrades: %d preds in %.1fs", len(upgrade_preds), time.time() - ts)

        # Step 9: DL baselines (NEW)
        if baseline_config.get("dlinear", {}).get("enabled", False):
            ts = time.time()
            dlinear_preds = predict_dlinear(train_sales, predict_months)
            dlinear_preds["timeframe_idx"] = tf_idx
            all_predictions.append(dlinear_preds)
            logger.info("    DLinear: %d preds in %.1fs", len(dlinear_preds), time.time() - ts)

        if baseline_config.get("nlinear", {}).get("enabled", False):
            ts = time.time()
            nlinear_preds = predict_nlinear(train_sales, predict_months)
            nlinear_preds["timeframe_idx"] = tf_idx
            all_predictions.append(nlinear_preds)
            logger.info("    NLinear: %d preds in %.1fs", len(nlinear_preds), time.time() - ts)

        # Step 10: Deep learning models (per timeframe, not per cluster)
        if dl_config:
            logger.info("    [10] Running %d deep learning models...", len(dl_config))
            ts = time.time()
            dl_preds = run_dl_models(train_sales, predict_months, dl_config)
            if not dl_preds.empty:
                dl_preds["timeframe_idx"] = tf_idx
                all_predictions.append(dl_preds)
            logger.info("    DL models: %d preds in %.1fs", len(dl_preds), time.time() - ts)

        # Step 11: Foundation models (zero-shot, per timeframe)
        if fm_config:
            logger.info("    [11] Running %d foundation models...", len(fm_config))
            ts = time.time()
            fm_preds = run_foundation_models(train_sales, predict_months, fm_config)
            if not fm_preds.empty:
                fm_preds["timeframe_idx"] = tf_idx
                all_predictions.append(fm_preds)
            logger.info("    Foundation models: %d preds in %.1fs", len(fm_preds), time.time() - ts)

        logger.info(
            "  Timeframe %s complete in %.1fs",
            tf["label"], time.time() - tf_start,
        )

    # Combine all predictions
    if all_predictions:
        all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    else:
        all_predictions_df = pd.DataFrame(
            columns=["sku_ck", "startdate", FORECAST_QTY_COL, "algorithm_id", "timeframe_idx"]
        )
    logger.info(
        "Total predictions: %d rows across %d algorithms",
        len(all_predictions_df),
        all_predictions_df["algorithm_id"].nunique() if not all_predictions_df.empty else 0,
    )

    # -- Step 12: Load comparison baselines from DB ------------------------
    logger.info("Step 12/14: Loading external forecast and existing predictions...")
    t12 = time.time()
    all_predict_months_set: set[pd.Timestamp] = set()
    for tf in timeframes:
        all_predict_months_set.update(_timeframe_predict_months(tf))
    all_predict_ts = sorted(all_predict_months_set)

    external_df = load_external_forecast(golden_skus, all_predict_ts)
    existing_df = load_existing_predictions(
        golden_skus, all_predict_ts,
        config["comparison"]["champion_model_ids"],
    )
    logger.info(
        "  External: %d rows, Existing: %d rows in %.1fs",
        len(external_df) if external_df is not None and not external_df.empty else 0,
        len(existing_df) if existing_df is not None and not existing_df.empty else 0,
        time.time() - t12,
    )

    # -- Step 13: Build affinity matrix + optimize -------------------------
    logger.info("Step 13/14: Building affinity matrix and optimizing portfolio...")
    t13 = time.time()

    actuals_df = sales_df[
        sales_df["startdate"].isin(all_predict_ts)
    ][["sku_ck", "startdate", "qty"]].copy()

    affinity_matrix, affinity_detail = build_affinity_matrix(
        all_predictions_df, actuals_df, classification_df,
    )
    logger.info(
        "  Affinity matrix: %d segments x %d algorithms",
        affinity_matrix.shape[0], affinity_matrix.shape[1],
    )
    logger.info("\n%s", format_affinity_heatmap(affinity_matrix, affinity_detail))

    opt_config = config.get("portfolio_optimizer", {})
    max_algos = opt_config.get("max_algorithms")
    min_seg = opt_config.get("min_segment_dfus", 30)

    min_cov = opt_config.get("min_dfu_coverage_pct", 0.5)
    cov_weighted = opt_config.get("coverage_weighted", True)
    n_floor = opt_config.get("naive_floor", True)
    if max_algos:
        assignments_df = optimize_constrained(
            affinity_matrix, affinity_detail,
            max_algorithms=max_algos, min_segment_dfus=min_seg,
            min_dfu_coverage_pct=min_cov,
            coverage_weighted=cov_weighted, naive_floor=n_floor,
        )
    else:
        assignments_df = optimize_greedy(
            affinity_matrix, affinity_detail, min_segment_dfus=min_seg,
            min_dfu_coverage_pct=min_cov,
            coverage_weighted=cov_weighted, naive_floor=n_floor,
        )

    portfolio_stats = compute_portfolio_accuracy(assignments_df, affinity_detail)
    logger.info("\n%s", format_portfolio_summary(assignments_df, portfolio_stats))

    ceiling_df = compute_ceiling_accuracy(all_predictions_df, actuals_df, classification_df)
    logger.info(
        "  Ceiling accuracy computed for %d segments in %.1fs",
        len(ceiling_df), time.time() - t13,
    )

    # -- Step 13b: Per-DFU hybrid ensemble ---------------------------------
    hybrid_cfg = config.get("hybrid_ensemble", {})
    hybrid_preds = pd.DataFrame(
        columns=["sku_ck", "startdate", FORECAST_QTY_COL, "algorithm_id"]
    )
    dfu_accuracy_matrix = pd.DataFrame(
        columns=["sku_ck", "algorithm_id", "wape", "accuracy_pct", "n_months"]
    )
    hybrid_routing_df = pd.DataFrame(
        columns=["sku_ck", "predicted_algorithm", "confidence"]
    )
    if hybrid_cfg.get("enabled", True) and not all_predictions_df.empty:
        logger.info("Step 13b/14: Building per-DFU hybrid ensemble...")
        t13b = time.time()
        try:
            dfu_accuracy_matrix = build_dfu_accuracy_matrix(
                all_predictions_df, actuals_df,
                min_n_months=hybrid_cfg.get("min_n_months", 2),
            )
            meta_model = train_meta_router(
                dfu_accuracy_matrix, dfu_attrs, classification_df,
                n_estimators=hybrid_cfg.get("meta_n_estimators", 300),
                learning_rate=hybrid_cfg.get("meta_learning_rate", 0.05),
                num_leaves=hybrid_cfg.get("meta_num_leaves", 31),
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
            hybrid_routing_df = predict_meta_router(meta_model, dfu_attrs, classification_df)
            logger.info(
                "  Hybrid ensemble: %d predictions, %d DFUs in %.1fs",
                len(hybrid_preds),
                hybrid_preds["sku_ck"].nunique() if not hybrid_preds.empty else 0,
                time.time() - t13b,
            )
        except (ValueError, Exception) as exc:  # noqa: BLE001
            logger.warning("  Hybrid ensemble skipped: %s", exc)

    # -- Step 14: Compare and report ---------------------------------------
    logger.info("Step 14/14: Comparing portfolio vs baselines...")
    t14 = time.time()

    portfolio_preds = compute_portfolio_predictions(
        all_predictions_df, assignments_df, classification_df,
    )

    naive_preds_all = all_predictions_df[
        all_predictions_df["algorithm_id"] == "seasonal_naive"
    ]

    ext_for_compare = external_df if (external_df is not None and not external_df.empty) else None
    exist_for_compare = existing_df if (existing_df is not None and not existing_df.empty) else None

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
        import numpy as np  # noqa: E402
        hybrid_metrics = compute_baseline_accuracy(
            hybrid_preds, actuals_df, "hybrid", classification_df
        )
        comparison["baselines"]["hybrid"] = hybrid_metrics
        port_acc = comparison["portfolio"]["accuracy_pct"]
        hyb_acc = hybrid_metrics["accuracy_pct"]
        naive_acc = comparison["baselines"]["seasonal_naive"]["accuracy_pct"]
        if not (np.isnan(hyb_acc) or np.isnan(port_acc)):
            comparison["lift"]["hybrid_vs_portfolio_bps"] = round(
                (hyb_acc - port_acc) * 100
            )
        if not (np.isnan(hyb_acc) or np.isnan(naive_acc)):
            comparison["lift"]["hybrid_vs_naive_bps"] = round(
                (hyb_acc - naive_acc) * 100
            )
        logger.info(
            "  Hybrid ensemble: Accuracy=%.2f%%  WAPE=%.2f%%  "
            "vs Portfolio=%+d bps  vs Naive=%+d bps",
            hyb_acc,
            hybrid_metrics["wape"],
            comparison["lift"].get("hybrid_vs_portfolio_bps", 0),
            comparison["lift"].get("hybrid_vs_naive_bps", 0),
        )

    logger.info("  Comparison complete in %.1fs", time.time() - t14)

    # Add execution-lag columns to combined predictions
    tf_train_end_map = {idx: tf["train_end"] for idx, tf in enumerate(timeframes)}
    exec_lag_map: dict[str, int] = (
        dfu_attrs.set_index("sku_ck")["execution_lag"].to_dict()
        if "execution_lag" in dfu_attrs.columns
        else {}
    )
    if exec_lag_map and not all_predictions_df.empty:
        all_predictions_df = add_lag_columns(
            all_predictions_df, tf_train_end_map, exec_lag_map
        )

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
            if overall_monthly else "n/a",
            max(v["mean_monthly_accuracy"] for v in overall_monthly.values())
            if overall_monthly else 0.0,
        )

    # -- Save outputs ------------------------------------------------------
    runtime = time.time() - t0

    # Save core artifacts
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hybrid artifacts
    if not dfu_accuracy_matrix.empty:
        dfu_accuracy_matrix.to_csv(output_dir / "dfu_accuracy_matrix.csv", index=False)
        logger.info("Saved DFU accuracy matrix to %s", output_dir / "dfu_accuracy_matrix.csv")
    if not hybrid_routing_df.empty:
        hybrid_routing_df.to_csv(output_dir / "hybrid_assignments.csv", index=False)
        logger.info("Saved hybrid assignments to %s", output_dir / "hybrid_assignments.csv")

    pd.DataFrame({"sku_ck": golden_skus}).to_csv(output_dir / "golden_set_skus.csv", index=False)
    classification_df.to_csv(output_dir / "classification.csv", index=False)
    all_predictions_df.to_parquet(output_dir / "all_predictions.parquet", index=False)
    affinity_matrix.to_csv(output_dir / "affinity_matrix.csv")
    affinity_detail.to_csv(output_dir / "affinity_detail.csv", index=False)
    assignments_df.to_csv(output_dir / "assignments.csv", index=False)

    (output_dir / "portfolio_stats.json").write_text(
        json.dumps(_make_serializable(portfolio_stats), indent=2)
    )
    (output_dir / "comparison.json").write_text(
        json.dumps(_make_serializable(comparison), indent=2)
    )
    (output_dir / "monthly_accuracy.json").write_text(
        json.dumps(
            _make_serializable({
                "monthly": monthly_accuracy,
                "avg_3m": avg_3m,
                "avg_6m": avg_6m,
                "overall": overall_monthly,
            }),
            indent=2,
        )
    )

    # Algorithm summary
    algo_counts = (
        all_predictions_df["algorithm_id"].value_counts().to_dict()
        if not all_predictions_df.empty else {}
    )
    (output_dir / "metadata.json").write_text(json.dumps({
        "experiment": "adv_expert_panel_v1",
        "runtime_seconds": round(runtime, 1),
        "n_dfus": len(golden_skus),
        "n_predictions": len(all_predictions_df),
        "n_algorithms": int(all_predictions_df["algorithm_id"].nunique())
            if not all_predictions_df.empty else 0,
        "n_archetypes": int(classification_df["archetype"].nunique())
            if not classification_df.empty else 0,
        "loc_filter": exp.get("loc_filter"),
        "algorithm_prediction_counts": algo_counts,
        "new_algorithms": {
            "statistical_upgrades": list(stat_upgrade_config.keys()),
            "deep_learning": list(dl_config.keys()) if dl_config else [],
            "foundation_models": list(fm_config.keys()) if fm_config else [],
            "dl_baselines": [
                b for b in ["dlinear", "nlinear"]
                if baseline_config.get(b, {}).get("enabled", False)
            ],
        },
    }, indent=2))

    # Report
    report_lines = [
        "=" * 70,
        "ADVANCED EXPERT PANEL — EXPERIMENT REPORT",
        "=" * 70,
        "",
        f"DFUs tested:         {len(golden_skus):,}",
        f"Archetypes:          {classification_df['archetype'].nunique() if not classification_df.empty else 0}",
        f"Total algorithms:    {all_predictions_df['algorithm_id'].nunique() if not all_predictions_df.empty else 0}",
        f"  Base statistical:  {len(stat_config)}",
        f"  Tree models:       {len(tree_config)}",
        f"  Stat upgrades:     {len(stat_upgrade_config)}",
        f"  Deep learning:     {len(dl_config) if dl_config else 0}",
        f"  Foundation models: {len(fm_config) if fm_config else 0}",
        f"  DL baselines:      {sum(1 for b in ['dlinear', 'nlinear'] if baseline_config.get(b, {}).get('enabled', False))}",
        f"Runtime:             {runtime / 60:.1f} minutes",
        "",
    ]

    if not affinity_matrix.empty:
        report_lines.append("-" * 70)
        report_lines.append("AFFINITY MATRIX (accuracy % per segment x algorithm)")
        report_lines.append("-" * 70)
        report_lines.append(affinity_matrix.round(1).to_string())
        report_lines.append("")

    if not assignments_df.empty:
        report_lines.append("-" * 70)
        report_lines.append("PORTFOLIO ASSIGNMENTS")
        report_lines.append("-" * 70)
        for _, row in assignments_df.iterrows():
            report_lines.append(
                f"  {row.get('archetype', '?'):<30s} -> {row.get('best_algorithm', '?'):<20s} "
                f"acc={row.get('accuracy_pct', 0):.1f}%  ({row.get('confidence', '?')})"
            )
        report_lines.append("")

    report_lines.append("-" * 70)
    report_lines.append("PORTFOLIO ACCURACY")
    report_lines.append("-" * 70)
    report_lines.append(f"  Overall accuracy:   {portfolio_stats.get('portfolio_accuracy', 0):.1f}%")
    report_lines.append(f"  Overall WAPE:       {portfolio_stats.get('portfolio_wape', 0):.1f}%")
    report_lines.append(f"  Algorithms used:    {portfolio_stats.get('n_algorithms', 0)}")
    report_lines.append("")

    if monthly_accuracy:
        from common.ml.expert_panel.report import _format_monthly_accuracy_section
        report_lines.append("-" * 70)
        report_lines.append(
            "MONTHLY ACCURACY — execution-lag-matched, full-coverage months only"
        )
        report_lines.append("-" * 70)
        report_lines.extend(
            _format_monthly_accuracy_section(
                monthly_accuracy, avg_3m, avg_6m, overall_monthly
            )
        )
        report_lines.append("")

    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)
    (output_dir / "experiment_report.txt").write_text(report_text)
    logger.info("Report saved to %s", output_dir / "experiment_report.txt")

    logger.info("=" * 60)
    logger.info("ADVANCED EXPERIMENT COMPLETE in %.1f minutes", runtime / 60)
    logger.info("Output: %s", output_dir)
    logger.info("=" * 60)

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Advanced Expert Panel Algorithm Selection Test"
    )
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
    logger.info("ADVANCED EXPERT PANEL ALGORITHM SELECTION TEST")
    logger.info("=" * 60)
    if loc_filter:
        logger.info("Loc filter: %s | Timeframes: %d | Seed: %d",
                    loc_filter,
                    config["experiment"]["n_timeframes"],
                    config["experiment"]["seed"])
    else:
        logger.info(
            "DFUs: %d | Timeframes: %d | Seed: %d",
            config["experiment"]["n_dfus"],
            config["experiment"]["n_timeframes"],
            config["experiment"]["seed"],
        )
    logger.info(
        "Algorithms: base_stat=%d tree=%d stat_upgrades=%d dl=%d foundation=%d baselines=%d",
        sum(1 for v in config.get("statistical_models", {}).values() if v.get("enabled")),
        sum(1 for v in config.get("tree_models", {}).values() if v.get("enabled")),
        sum(1 for v in config.get("statistical_upgrades", {}).values() if v.get("enabled")),
        sum(1 for v in config.get("deep_learning", {}).values() if v.get("enabled")),
        sum(1 for v in config.get("foundation_models", {}).values() if v.get("enabled")),
        sum(1 for b in ["dlinear", "nlinear", "seasonal_naive", "rolling_mean", "ridge"]
            if config.get("baselines", {}).get(b, {}).get("enabled")),
    )

    run_experiment(config)


if __name__ == "__main__":
    main()
