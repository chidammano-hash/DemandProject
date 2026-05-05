"""Unified SKU feature computation pipeline.

Computes all time-series features (volume, trend, seasonality, periodicity,
intermittency, lifecycle) from fact_sales_monthly and writes to dim_sku.

Usage:
    python scripts/ml/compute_sku_features.py
    python scripts/ml/compute_sku_features.py --dry-run
    python scripts/ml/compute_sku_features.py --output-csv data/sku_features.csv
    python scripts/ml/compute_sku_features.py --workers 4 --time-window 24
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.db import get_db_params  # noqa: E402
from common.ml.sku_features.compute import (  # noqa: E402
    compute_all_sku_features,
    load_sales_from_db,
)
from common.ml.sku_features.persistence import write_features_to_dim_sku  # noqa: E402
from common.core.planning_date import get_planning_date  # noqa: E402
from common.services.perf_profiler import profiled_section  # noqa: E402
from common.core.utils import _ts, load_config  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config (used when config/forecasting/sku_features_config.yaml is absent)
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, object] = {
    "time_window_months": 36,
    "min_months_history": 1,
    "workers": 4,
    "classifiers": {
        "seasonality_profile": {
            "high_threshold": 0.4,
            "moderate_threshold": 0.15,
        },
        "variability_class": {
            "low_cv": 0.3,
            "moderate_cv": 0.6,
            "high_cv": 1.0,
        },
    },
}

# Backward-compat CSV path consumed by the clustering pipeline
_CLUSTERING_FEATURES_CSV = ROOT / "data" / "clustering_features.csv"


def _load_pipeline_config() -> dict:
    """Load config/forecasting/sku_features_config.yaml, falling back to built-in defaults."""
    try:
        cfg = load_config("sku_features_config")
        if cfg:
            logger.info("Loaded config/forecasting/sku_features_config.yaml")
            return cfg
    except Exception:
        logger.debug("No sku_features_config.yaml found — using defaults")
    return dict(_DEFAULTS)


def _classify_seasonality_profile(
    seasonality_strength: float,
    thresholds: dict[str, float],
) -> str:
    """Assign a seasonality profile label based on strength value.

    Returns one of: 'highly_seasonal', 'moderate_seasonal', 'non_seasonal'.
    """
    high = thresholds.get("high_threshold", 0.4)
    moderate = thresholds.get("moderate_threshold", 0.15)
    if seasonality_strength >= high:
        return "highly_seasonal"
    if seasonality_strength >= moderate:
        return "moderate_seasonal"
    return "non_seasonal"


def _classify_variability(
    cv_demand: float,
    thresholds: dict[str, float],
) -> str:
    """Assign a variability class label based on CV.

    Returns one of: 'low', 'moderate', 'high', 'erratic'.
    """
    low = thresholds.get("low_cv", 0.3)
    moderate = thresholds.get("moderate_cv", 0.6)
    high = thresholds.get("high_cv", 1.0)
    if cv_demand <= low:
        return "low"
    if cv_demand <= moderate:
        return "moderate"
    if cv_demand <= high:
        return "high"
    return "erratic"


def _apply_classifiers(feature_df, cfg: dict):
    """Apply classification columns to the feature DataFrame in-place.

    Adds ``seasonality_profile`` and ``variability_class`` columns derived
    from the computed time-series features using thresholds from config.
    """
    import pandas as pd  # local import to keep top-level lightweight

    classifiers = cfg.get("classifiers", _DEFAULTS["classifiers"])

    # --- seasonality_profile ---
    seas_thresholds = classifiers.get(
        "seasonality_profile",
        _DEFAULTS["classifiers"]["seasonality_profile"],  # type: ignore[index]
    )
    if "seasonality_strength" in feature_df.columns:
        feature_df["seasonality_profile"] = feature_df["seasonality_strength"].apply(
            lambda v: _classify_seasonality_profile(float(v) if pd.notna(v) else 0.0, seas_thresholds)
        )
    else:
        feature_df["seasonality_profile"] = "non_seasonal"

    # --- variability_class ---
    var_thresholds = classifiers.get(
        "variability_class",
        _DEFAULTS["classifiers"]["variability_class"],  # type: ignore[index]
    )
    if "cv_demand" in feature_df.columns:
        feature_df["variability_class"] = feature_df["cv_demand"].apply(
            lambda v: _classify_variability(float(v) if pd.notna(v) else 0.0, var_thresholds)
        )
    else:
        feature_df["variability_class"] = "low"

    return feature_df


def run_pipeline(
    *,
    dry_run: bool = False,
    output_csv: str | None = None,
    workers: int | None = None,
    time_window: int | None = None,
) -> dict:
    """Execute the full SKU feature computation pipeline.

    Parameters
    ----------
    dry_run : bool
        If True, compute features but skip DB writes.
    output_csv : str | None
        Optional path to write the feature DataFrame as CSV.
    workers : int | None
        Number of parallel workers (overrides config).
    time_window : int | None
        Lookback window in months (overrides config).

    Returns
    -------
    dict with summary statistics (skus_processed, features_count, elapsed_s).
    """
    t_start = time.time()
    cfg = _load_pipeline_config()

    # Resolve effective parameters — CLI flags override config
    effective_window = time_window or int(cfg.get("time_window_months", 36))
    effective_workers = workers or int(cfg.get("workers", 4))

    planning_date = get_planning_date()
    db_params = get_db_params()

    logger.info(
        "%s Starting SKU feature pipeline — window=%d months, workers=%d, "
        "planning_date=%s, dry_run=%s",
        _ts(), effective_window, effective_workers, planning_date, dry_run,
    )

    # ── Step 1: Load sales data ─────────────────────────────────────────────
    with profiled_section("load_sales_from_db"):
        logger.info("%s Loading sales data from fact_sales_monthly ...", _ts())
        sales_df = load_sales_from_db(
            db_params=db_params,
            time_window_months=effective_window,
        )
        logger.info(
            "%s Loaded %s sales records for %s unique SKUs",
            _ts(),
            f"{len(sales_df):,}",
            f"{sales_df['sku_ck'].nunique():,}" if "sku_ck" in sales_df.columns else "?",
        )

    # ── Step 2: Compute all time-series features ────────────────────────────
    with profiled_section("compute_all_sku_features"):
        logger.info("%s Computing time-series features (workers=%d) ...", _ts(), effective_workers)
        feature_df = compute_all_sku_features(
            sales_df=sales_df,
            workers=effective_workers,
        )
        logger.info(
            "%s Computed %d features for %s SKUs",
            _ts(),
            len([c for c in feature_df.columns if c != "sku_ck"]),
            f"{len(feature_df):,}",
        )

    # ── Step 3: Apply classifiers ───────────────────────────────────────────
    with profiled_section("apply_classifiers"):
        logger.info("%s Applying classifiers (seasonality_profile, variability_class) ...", _ts())
        feature_df = _apply_classifiers(feature_df, cfg)
        logger.info(
            "%s Seasonality profile distribution: %s",
            _ts(),
            feature_df["seasonality_profile"].value_counts().to_dict(),
        )
        logger.info(
            "%s Variability class distribution: %s",
            _ts(),
            feature_df["variability_class"].value_counts().to_dict(),
        )

    # ── Step 4: Write to dim_sku (unless --dry-run) ─────────────────────────
    if dry_run:
        logger.info("%s DRY RUN — skipping DB write to dim_sku", _ts())
    else:
        with profiled_section("write_features_to_dim_sku"):
            logger.info("%s Writing features to dim_sku (%s rows) ...", _ts(), f"{len(feature_df):,}")
            rows_updated = write_features_to_dim_sku(
                features_df=feature_df,
                db_params=db_params,
            )
            logger.info("%s Updated %s rows in dim_sku", _ts(), f"{rows_updated['updated']:,}")

    # ── Step 5: Write backward-compat clustering_features.csv ───────────────
    with profiled_section("write_clustering_csv"):
        _CLUSTERING_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(_CLUSTERING_FEATURES_CSV, index=False)
        logger.info(
            "%s Wrote backward-compat CSV: %s (%s rows)",
            _ts(), _CLUSTERING_FEATURES_CSV, f"{len(feature_df):,}",
        )

    # ── Step 6: Optional user-specified CSV output ──────────────────────────
    if output_csv:
        out_path = Path(output_csv)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(out_path, index=False)
        logger.info("%s Wrote CSV: %s", _ts(), out_path)

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    feature_cols = [c for c in feature_df.columns if c != "sku_ck"]
    summary = {
        "skus_processed": len(feature_df),
        "features_count": len(feature_cols),
        "elapsed_s": round(elapsed, 1),
        "dry_run": dry_run,
        "time_window_months": effective_window,
        "workers": effective_workers,
    }
    logger.info(
        "%s Pipeline complete — %s SKUs, %d features, %.1fs elapsed",
        _ts(), f"{summary['skus_processed']:,}", summary["features_count"], elapsed,
    )
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute unified SKU features from fact_sales_monthly → dim_sku",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute features but skip writing to dim_sku",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to write the feature DataFrame as CSV (e.g. data/sku_features.csv)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: from config or 4)",
    )
    parser.add_argument(
        "--time-window",
        type=int,
        default=None,
        help="Lookback window in months (default: from config or 36)",
    )
    args = parser.parse_args()

    summary = run_pipeline(
        dry_run=args.dry_run,
        output_csv=args.output_csv,
        workers=args.workers,
        time_window=args.time_window,
    )

    # Log summary
    logger.info("=" * 60)
    logger.info("  SKU Feature Computation — Summary")
    logger.info("=" * 60)
    logger.info("  SKUs processed:   %s", f"{summary['skus_processed']:,}")
    logger.info("  Features:         %s", summary["features_count"])
    logger.info("  Time window:      %s months", summary["time_window_months"])
    logger.info("  Workers:          %s", summary["workers"])
    logger.info("  Dry run:          %s", summary["dry_run"])
    logger.info("  Elapsed:          %.1fs", summary["elapsed_s"])
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()
