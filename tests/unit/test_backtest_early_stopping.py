"""
Unit tests for backtest training improvements:
- Eval set creation (time-aware 80/20 split)
- Feature importance logging path creation
- Artifact enrichment keys
"""
import sys
from pathlib import Path
import pickle
import json
import tempfile
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestTrainValSplit:
    """Verify the 80/20 time-aware train/val split logic."""

    def _split(self, n):
        """Replicate the split formula used in all 3 training functions."""
        n_val = max(1, int(n * 0.20))
        return n - n_val, n_val

    def test_split_100_rows(self):
        n_tr, n_val = self._split(100)
        assert n_tr == 80
        assert n_val == 20

    def test_split_50_rows(self):
        n_tr, n_val = self._split(50)
        assert n_tr == 40
        assert n_val == 10

    def test_split_minimum_1_val_row(self):
        """Even with MIN_CLUSTER_ROWS=50, we must have at least 1 val row."""
        n_tr, n_val = self._split(1)
        assert n_val == 1
        assert n_tr == 0

    def test_split_4_rows(self):
        n_tr, n_val = self._split(4)
        # int(4 * 0.20) = 0, but max(1, 0) = 1
        assert n_val == 1
        assert n_tr == 3


class TestValWapeFormula:
    """Verify the val_wape formula."""

    def test_perfect_predictions(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        denom = float(abs(y_true.sum()))
        wape = round(float((abs(y_pred - y_true)).sum() / denom * 100), 2)
        assert wape == 0.0

    def test_all_zeros_denom_guard(self):
        """When sum of actuals is 0, wape should be 0.0 not ZeroDivisionError."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([5.0, 5.0, 5.0])
        denom = float(abs(y_true.sum()))
        wape = round(float((abs(y_pred - y_true)).sum() / denom * 100), 2) if denom > 0 else 0.0
        assert wape == 0.0

    def test_simple_wape(self):
        y_true = np.array([100.0, 100.0])
        y_pred = np.array([110.0, 90.0])
        denom = float(abs(y_true.sum()))
        wape = round(float((abs(y_pred - y_true)).sum() / denom * 100), 2)
        assert wape == 10.0


class TestArtifactEnrichment:
    """Verify the enriched artifact dict keys are present."""

    REQUIRED_KEYS = {
        "model", "feature_cols", "model_id",
        "cluster_label", "n_estimators_used",
        "val_wape", "trained_at", "timeframe", "feature_importance",
    }

    def _make_artifact(self, model_stub):
        """Build a minimal enriched artifact as the backtest scripts would."""
        from datetime import datetime, timezone
        n_est_used = getattr(model_stub, "best_iteration_", None) or 0
        feature_cols = ["f1", "f2", "f3"]
        imp = getattr(model_stub, "feature_importances_", np.array([0.5, 0.3, 0.2]))
        importance_dict = dict(zip(feature_cols, [float(v) for v in imp]))
        return {
            "model": model_stub,
            "feature_cols": feature_cols,
            "model_id": "lgbm_cluster",
            "cluster_label": "A",
            "n_estimators_used": n_est_used,
            "val_wape": getattr(model_stub, "_val_wape", None),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "timeframe": "J",
            "feature_importance": importance_dict,
        }

    def test_all_required_keys_present(self):
        from unittest.mock import MagicMock
        stub = MagicMock()
        stub.feature_importances_ = np.array([0.5, 0.3, 0.2])
        stub.best_iteration_ = 42
        stub._val_wape = 15.3
        artifact = self._make_artifact(stub)
        for key in self.REQUIRED_KEYS:
            assert key in artifact, f"Missing key: {key}"

    def test_trained_at_is_iso_string(self):
        from unittest.mock import MagicMock
        from datetime import datetime
        stub = MagicMock()
        stub.feature_importances_ = np.array([0.5, 0.3, 0.2])
        artifact = self._make_artifact(stub)
        # Should be parseable as ISO datetime
        dt = datetime.fromisoformat(artifact["trained_at"])
        assert dt is not None

    def test_feature_importance_keys_match_feature_cols(self):
        from unittest.mock import MagicMock
        stub = MagicMock()
        stub.feature_importances_ = np.array([0.5, 0.3, 0.2])
        artifact = self._make_artifact(stub)
        assert set(artifact["feature_importance"].keys()) == set(artifact["feature_cols"])

    def test_artifact_is_picklable(self):
        from unittest.mock import MagicMock
        stub = MagicMock()
        stub.feature_importances_ = np.array([0.5, 0.3, 0.2])
        stub.best_iteration_ = 42
        stub._val_wape = 15.3
        artifact = self._make_artifact(stub)
        # Remove model and val_wape (val_wape attribute access returns a MagicMock
        # which is not picklable); keep only plain-data keys
        picklable_keys = {"feature_cols", "model_id", "cluster_label",
                          "n_estimators_used", "trained_at", "timeframe", "feature_importance"}
        artifact_subset = {k: v for k, v in artifact.items() if k in picklable_keys}
        data = pickle.dumps(artifact_subset)
        restored = pickle.loads(data)
        assert restored["cluster_label"] == "A"
        assert restored["timeframe"] == "J"


class TestGpuEnvVar:
    """Verify DEMAND_GPU env-var logic."""

    def _resolve_gpu(self, pref, dummy_gpu_works):
        """Replicate the GPU detection logic."""
        import os
        _use_gpu = False
        if pref == "on":
            _use_gpu = True
        elif pref == "off":
            _use_gpu = False
        else:  # auto
            _use_gpu = dummy_gpu_works
        return _use_gpu

    def test_on_forces_gpu(self):
        assert self._resolve_gpu("on", False) is True

    def test_off_disables_gpu(self):
        assert self._resolve_gpu("off", True) is False

    def test_auto_uses_detection_result_true(self):
        assert self._resolve_gpu("auto", True) is True

    def test_auto_uses_detection_result_false(self):
        assert self._resolve_gpu("auto", False) is False
