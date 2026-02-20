"""Shared MLflow logging for backtest scripts."""

import os
import time
from typing import Any


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log_backtest_run(
    model_type: str,
    model_id: str,
    cluster_strategy: str,
    hyperparams: dict[str, Any],
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    artifact_paths: list[str],
) -> None:
    """Log a backtest run to MLflow.

    Args:
        model_type: e.g. "lgbm_backtest", "catboost_backtest"
        model_id: e.g. "lgbm_global", "catboost_cluster"
        cluster_strategy: "global", "per_cluster", "transfer", "pooled"
        hyperparams: Model hyperparameters to log as params
        metrics: Dict of metric_name → value (n_predictions, n_dfus, etc.)
        metadata: Full metadata dict (checked for accuracy_at_execution_lag)
        artifact_paths: List of file paths to log as artifacts
    """
    try:
        import mlflow

        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5003")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("demand_backtest")

        with mlflow.start_run():
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("cluster_strategy", cluster_strategy)
            mlflow.set_tag("model_id", model_id)

            mlflow.log_params(hyperparams)
            mlflow.log_metrics(metrics)

            if "accuracy_at_execution_lag" in metadata:
                acc = metadata["accuracy_at_execution_lag"]
                if acc.get("wape"):
                    mlflow.log_metric("wape", acc["wape"])
                if acc.get("accuracy_pct"):
                    mlflow.log_metric("accuracy_pct", acc["accuracy_pct"])
                if acc.get("bias"):
                    mlflow.log_metric("bias", acc["bias"])

            for path in artifact_paths:
                mlflow.log_artifact(path)

            print(f"\n  [{_ts()}] Logged to MLflow: {mlflow.get_artifact_uri()}")
    except Exception as e:
        print(f"\n  [{_ts()}] MLflow logging skipped: {e}")
