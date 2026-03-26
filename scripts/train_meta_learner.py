"""Train meta-learner classifier for champion model selection.

Uses ceiling (oracle) labels as ground truth: for each DFU-month, the
ceiling identifies which model actually had the lowest error. The meta-learner
learns to predict this from DFU features + recent model performance stats.

Strict temporal train/test split prevents data leakage — no random splitting.

Usage:
    python scripts/train_meta_learner.py \
        --config config/model_competition.yaml \
        [--model-type random_forest] \
        [--output data/champion/meta_learner.joblib]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
import yaml
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.db import get_db_params
from common.services.perf_profiler import profiled_section


def _load_monthly_errors(
    db: dict[str, Any], models: list[str], lag_mode: str,
) -> pd.DataFrame:
    placeholders = ",".join(["%s"] * len(models))
    params: list[Any] = list(models)
    if lag_mode == "execution":
        lag_cond = "lag::text = execution_lag::text"
    else:
        lag_cond = "lag = %s"
        params.append(int(lag_mode))

    sql = f"""
        SELECT item_id, customer_group, loc, startdate, model_id,
               basefcst_pref, tothist_dmd,
               ABS(basefcst_pref - tothist_dmd) AS abs_err
        FROM fact_external_forecast_monthly
        WHERE model_id IN ({placeholders})
          AND {lag_cond}
          AND basefcst_pref IS NOT NULL
          AND tothist_dmd IS NOT NULL
        ORDER BY item_id, customer_group, loc, model_id, startdate
    """
    with psycopg.connect(**db) as conn:
        df = pd.read_sql(sql, conn, params=params)
    df["startdate"] = pd.to_datetime(df["startdate"])
    for col in ["basefcst_pref", "tothist_dmd", "abs_err"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_dfu_features(db: dict[str, Any]) -> pd.DataFrame:
    sql = """
        SELECT item_id, customer_group, loc,
               ml_cluster, abc_vol, execution_lag, total_lt,
               brand, region,
               seasonality_profile, seasonality_strength,
               is_yearly_seasonal, peak_month, trough_month,
               peak_trough_ratio
        FROM dim_sku
    """
    with psycopg.connect(**db) as conn:
        df = pd.read_sql(sql, conn)
    for col in ["ml_cluster", "abc_vol", "brand", "region", "seasonality_profile"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    return df


# ---------------------------------------------------------------------------
# Training data construction
# ---------------------------------------------------------------------------

_DFU_COLS = ["item_id", "customer_group", "loc"]
_DFU_MONTH_COLS = ["item_id", "customer_group", "loc", "startdate"]
_DFU_MODEL_COLS = ["item_id", "customer_group", "loc", "model_id"]


def build_training_data(
    monthly_errors: pd.DataFrame,
    dfu_features: pd.DataFrame,
    models: list[str],
    performance_window: int = 6,
    min_prior_months: int = 3,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build (X, y) for meta-learner from ceiling labels + DFU features.

    Returns (features_df, target_series, feature_column_names).
    Each row is a DFU-month. Target is the ceiling-winning model_id.
    All features are strictly causal (use only prior data).
    """
    df = monthly_errors.sort_values(_DFU_MODEL_COLS + ["startdate"]).copy()

    # Step 1: Compute ceiling labels (ground truth)
    ranked = df.copy()
    ranked["_rank"] = ranked.groupby(_DFU_MONTH_COLS)["abs_err"].rank(
        method="first", ascending=True,
    )
    ceiling = ranked[ranked["_rank"] == 1][
        _DFU_MONTH_COLS + ["model_id"]
    ].rename(columns={"model_id": "ceiling_winner"})

    # Step 2: Compute per-model rolling performance (strictly prior)
    g = df.groupby(_DFU_MODEL_COLS, sort=False)
    df["_roll_abs_err"] = g["abs_err"].transform(
        lambda x: x.shift(1).rolling(window=performance_window, min_periods=1).sum()
    )
    df["_roll_actual"] = g["tothist_dmd"].transform(
        lambda x: x.shift(1).rolling(window=performance_window, min_periods=1).sum()
    )
    df["_prior_count"] = g["abs_err"].transform(
        lambda x: x.shift(1).expanding().count()
    )
    df["_roll_wape"] = df["_roll_abs_err"] / df["_roll_actual"].abs().clip(lower=1e-6)

    # Step 3: Filter rows with enough prior history
    eligible = df[df["_prior_count"] >= min_prior_months]
    dfu_months = eligible[_DFU_MONTH_COLS].drop_duplicates()

    # Step 4: Pivot per-model stats
    pivoted = dfu_months.copy()
    for model_id in models:
        model_data = df[df["model_id"] == model_id][
            _DFU_MONTH_COLS + ["_roll_wape"]
        ].rename(columns={"_roll_wape": f"roll_wape_{model_id}"})
        pivoted = pivoted.merge(model_data, on=_DFU_MONTH_COLS, how="left")

    # Step 5: Demand statistics (strictly prior)
    demand = df.drop_duplicates(subset=_DFU_MONTH_COLS).copy()
    demand = demand.sort_values(_DFU_COLS + ["startdate"])
    dg = demand.groupby(_DFU_COLS, sort=False)["tothist_dmd"]
    demand["mean_qty"] = dg.transform(lambda x: x.shift(1).expanding().mean())
    demand["std_qty"] = dg.transform(lambda x: x.shift(1).expanding().std())
    demand["cv_demand"] = demand["std_qty"] / demand["mean_qty"].abs().clip(lower=1e-6)

    pivoted = pivoted.merge(
        demand[_DFU_MONTH_COLS + ["mean_qty", "cv_demand"]],
        on=_DFU_MONTH_COLS, how="left",
    )

    # Step 6: Calendar features (Fourier terms replace legacy month_sin/cos)
    pivoted["month"] = pivoted["startdate"].dt.month
    pivoted["quarter"] = pivoted["startdate"].dt.quarter
    month_vals = pivoted["month"].values.astype(np.float64)
    for period in [12, 6, 4, 3]:
        angle = 2.0 * np.pi * month_vals / period
        pivoted[f"fourier_sin_{period}"] = np.sin(angle).astype(np.float32)
        pivoted[f"fourier_cos_{period}"] = np.cos(angle).astype(np.float32)

    # Step 7: Join DFU static features
    pivoted = pivoted.merge(dfu_features, on=_DFU_COLS, how="left")

    # Step 8: Join ceiling labels
    pivoted = pivoted.merge(ceiling, on=_DFU_MONTH_COLS, how="inner")

    # Identify feature columns (everything except keys and target)
    exclude = set(_DFU_MONTH_COLS + ["ceiling_winner"])
    feature_cols = [c for c in pivoted.columns if c not in exclude]

    return pivoted, pivoted["ceiling_winner"], feature_cols


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = "random_forest",
    **kwargs: Any,
) -> tuple[Any, float, dict]:
    """Train and evaluate the meta-learner classifier."""
    from sklearn.metrics import accuracy_score, classification_report

    if model_type == "xgboost":
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        # y_test encoding not needed — evaluation uses string labels
        clf = XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 8),
            learning_rate=kwargs.get("learning_rate", 0.1),
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train_enc)
        y_pred_enc = clf.predict(X_test)
        y_pred = le.inverse_transform(y_pred_enc)
    else:
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 15),
            min_samples_leaf=kwargs.get("min_samples_leaf", 10),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return clf, accuracy, report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train meta-learner for champion selection")
    parser.add_argument(
        "--config", type=str, default="config/model_competition.yaml",
    )
    parser.add_argument(
        "--model-type", type=str, default=None,
        help="Classifier type: random_forest or xgboost (overrides config)",
    )
    parser.add_argument(
        "--output", type=str, default="data/champion/meta_learner.joblib",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    db = get_db_params()

    # Load config
    config_path = ROOT / args.config
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    cfg = raw.get("competition", {})
    meta_cfg = cfg.get("meta_learner", {})

    models = cfg.get("models", [])
    lag_mode = str(cfg.get("lag", "execution"))
    min_rows = int(cfg.get("min_dfu_rows", 3))
    model_type = args.model_type or meta_cfg.get("model_type", "random_forest")
    performance_window = int(meta_cfg.get("performance_window", 6))
    test_months = int(meta_cfg.get("test_months", 3))

    print(f"Meta-Learner Training — {model_type}")
    print(f"  Models: {', '.join(models)}")
    print(f"  Performance window: {performance_window} months")
    print(f"  Test holdout: {test_months} months")
    print()

    # Load data
    print("Loading monthly errors...")
    t0 = time.time()
    with profiled_section("load_monthly_errors"):
        monthly_errors = _load_monthly_errors(db, models, lag_mode)
    print(f"  {len(monthly_errors):,} rows ({time.time() - t0:.1f}s)")

    print("Loading DFU features...")
    with profiled_section("load_dfu_features"):
        dfu_features = _load_dfu_features(db)
    print(f"  {len(dfu_features):,} DFUs")
    print()

    # Build training data
    print("Building training data (ceiling labels + features)...")
    t1 = time.time()
    with profiled_section("build_training_data"):
        features_df, target, feature_cols = build_training_data(
            monthly_errors, dfu_features, models,
            performance_window=performance_window,
            min_prior_months=min_rows,
        )
    print(f"  {len(features_df):,} samples, {len(feature_cols)} features ({time.time() - t1:.1f}s)")
    print(f"  Feature columns: {feature_cols}")
    print()

    if len(features_df) == 0:
        print("No training data. Aborting.")
        sys.exit(1)

    # Temporal train/test split (no data leakage — strictly temporal)
    all_months = sorted(features_df["startdate"].unique())
    if len(all_months) <= test_months:
        print(f"Not enough months ({len(all_months)}) for {test_months}-month holdout")
        sys.exit(1)

    cutoff = all_months[-(test_months)]
    train_mask = features_df["startdate"] < cutoff
    test_mask = features_df["startdate"] >= cutoff

    X_train = features_df.loc[train_mask, feature_cols].fillna(0)
    y_train = target[train_mask]
    X_test = features_df.loc[test_mask, feature_cols].fillna(0)
    y_test = target[test_mask]

    print(f"Train: {len(X_train):,} samples (months < {cutoff.date()})")
    print(f"Test:  {len(X_test):,} samples (months >= {cutoff.date()})")
    print("Target distribution (train):")
    for model_id, count in y_train.value_counts().items():
        print(f"  {model_id}: {count} ({100.0 * count / len(y_train):.1f}%)")
    print()

    # Train
    print(f"Training {model_type} classifier...")
    t2 = time.time()
    # Only pass classifier-relevant hyperparams (not config keys like test_months)
    clf_keys = {"n_estimators", "max_depth", "min_samples_leaf", "learning_rate"}
    clf_params = {k: v for k, v in meta_cfg.items() if k in clf_keys}
    with profiled_section("train_classifier"):
        clf, accuracy, report = train_classifier(
            X_train, y_train, X_test, y_test,
            model_type=model_type,
            **clf_params,
        )
    print(f"  Test accuracy: {accuracy:.4f} ({time.time() - t2:.1f}s)")
    print("  Per-class F1:")
    for cls, metrics in report.items():
        if isinstance(metrics, dict) and "f1-score" in metrics:
            print(f"    {cls}: F1={metrics['f1-score']:.3f}, "
                  f"precision={metrics['precision']:.3f}, "
                  f"recall={metrics['recall']:.3f}")
    print()

    # Feature importance
    if hasattr(clf, "feature_importances_"):
        importance = sorted(
            zip(feature_cols, clf.feature_importances_),
            key=lambda x: -x[1],
        )
        print("Top 15 features:")
        for feat, imp in importance[:15]:
            print(f"  {feat:<35s} {imp:.4f}")
        print()

    # Save model
    import joblib
    from sklearn.preprocessing import LabelEncoder

    with profiled_section("save_model_and_report"):
        le = LabelEncoder()
        le.fit(target)  # fit on all labels

        output_path = ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)

        meta_artifact = {
            "model": clf,
            "feature_columns": feature_cols,
            "label_encoder": le,
            "training_metadata": {
                "model_type": model_type,
                "test_accuracy": round(accuracy, 4),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_features": len(feature_cols),
                "test_months": test_months,
                "performance_window": performance_window,
                "train_cutoff": str(cutoff.date()),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "models": models,
            },
        }
        joblib.dump(meta_artifact, output_path)
        print(f"Model saved to {output_path}")

        # Save report
        report_path = output_path.parent / "meta_learner_report.json"
        with open(report_path, "w") as f:
            json.dump({
                "accuracy": round(accuracy, 4),
                "classification_report": report,
                "feature_importance": (
                    {feat: round(float(imp), 6) for feat, imp in importance[:30]}
                    if hasattr(clf, "feature_importances_") else {}
                ),
                **meta_artifact["training_metadata"],
            }, f, indent=2, default=str)
        print(f"Report saved to {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
