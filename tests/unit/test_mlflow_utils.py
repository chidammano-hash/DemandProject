"""Tests for common/mlflow_utils.py — MLflow logging."""

from unittest.mock import MagicMock, patch

from common.ml.mlflow_utils import log_backtest_run


class TestLogBacktestRun:
    @patch.dict("sys.modules", {"mlflow": MagicMock()})
    def test_logs_experiment_and_tags(self):
        import sys
        mock_mlflow = sys.modules["mlflow"]
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_mlflow.get_artifact_uri.return_value = "file:///tmp"

        log_backtest_run(
            model_type="lgbm_backtest",
            model_id="lgbm_cluster",
            cluster_strategy="global",
            hyperparams={"n_estimators": 500},
            metrics={"n_predictions": 1000},
            metadata={},
            artifact_paths=[],
        )

        mock_mlflow.set_experiment.assert_called_once_with("demand_backtest")
        mock_mlflow.set_tag.assert_any_call("model_type", "lgbm_backtest")
        mock_mlflow.set_tag.assert_any_call("model_id", "lgbm_cluster")
        mock_mlflow.set_tag.assert_any_call("cluster_strategy", "global")

    @patch.dict("sys.modules", {"mlflow": MagicMock()})
    def test_logs_accuracy_metrics_from_metadata(self):
        import sys
        mock_mlflow = sys.modules["mlflow"]
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_mlflow.get_artifact_uri.return_value = "file:///tmp"

        metadata = {
            "accuracy_at_execution_lag": {
                "wape": 15.5,
                "accuracy_pct": 84.5,
                "bias": -0.02,
            }
        }

        log_backtest_run(
            model_type="lgbm_backtest",
            model_id="lgbm_cluster",
            cluster_strategy="global",
            hyperparams={},
            metrics={},
            metadata=metadata,
            artifact_paths=[],
        )

        mock_mlflow.log_metric.assert_any_call("wape", 15.5)
        mock_mlflow.log_metric.assert_any_call("accuracy_pct", 84.5)
        mock_mlflow.log_metric.assert_any_call("bias", -0.02)

    def test_graceful_when_mlflow_unavailable(self):
        """Should not raise when MLflow is unavailable — function catches exceptions internally."""
        # The function has a try/except that catches all exceptions
        # Even with mlflow installed, if the server is unreachable it catches the error
        # Just verify it doesn't raise
        log_backtest_run(
            model_type="lgbm_backtest",
            model_id="lgbm_cluster",
            cluster_strategy="global",
            hyperparams={},
            metrics={},
            metadata={},
            artifact_paths=[],
        )
        # No exception raised = pass
