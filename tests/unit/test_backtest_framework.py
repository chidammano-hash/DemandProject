"""Tests for common/backtest_framework.py — timeframe generation and utilities."""

import datetime
from unittest.mock import patch

import pytest
import pandas as pd
import numpy as np

from common.backtest_framework import generate_timeframes


class TestGenerateTimeframes:
    def test_returns_10_timeframes(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        assert len(tfs) == 10

    def test_labels_a_through_j(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        labels = [tf["label"] for tf in tfs]
        assert labels == list("ABCDEFGHIJ")

    def test_expanding_windows(self):
        """Each train_end should be later than the previous one."""
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        train_ends = [tf["train_end"] for tf in tfs]
        for i in range(1, len(train_ends)):
            assert train_ends[i] > train_ends[i - 1]

    def test_train_end_before_predict_start(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        for tf in tfs:
            assert tf["train_end"] < tf["predict_start"]

    def test_predict_end_equals_latest(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        for tf in tfs:
            assert tf["predict_end"] == latest

    def test_train_start_equals_earliest(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        for tf in tfs:
            assert tf["train_start"] == earliest

    def test_custom_n(self):
        earliest = pd.Timestamp("2022-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=5)
        assert len(tfs) == 5
        assert tfs[0]["label"] == "A"
        assert tfs[4]["label"] == "E"

    def test_indices_sequential(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        indices = [tf["index"] for tf in tfs]
        assert indices == list(range(10))

    def test_predict_start_is_one_month_after_train_end(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        for tf in tfs:
            expected = tf["train_end"] + pd.DateOffset(months=1)
            assert tf["predict_start"] == expected


class TestPredictSingleMonth:
    """Verify _predict_single_month feature alignment."""

    def test_predict_single_month_passes_all_feature_cols(self):
        """_predict_single_month must pass all feature_cols to model.predict(),
        including ml_cluster (since per-cluster models are trained with it)."""
        from unittest.mock import MagicMock
        from common.backtest_framework import _predict_single_month

        feature_cols = ["qty_lag_1", "qty_lag_2", "ml_cluster", "month"]

        predict_data = pd.DataFrame({
            "dfu_ck": ["A_B_C", "D_E_F"],
            "dmdunit": ["A", "D"],
            "dmdgroup": ["B", "E"],
            "loc": ["C", "F"],
            "startdate": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
            "qty_lag_1": [100.0, 200.0],
            "qty_lag_2": [90.0, 180.0],
            "ml_cluster": ["cluster_0", "cluster_0"],
            "month": [1, 1],
        })

        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([110.0, 210.0]))
        models = {"cluster_0": mock_model}

        result = _predict_single_month(models, predict_data, feature_cols)

        # Model should have been called with ALL feature_cols (including ml_cluster)
        call_args = mock_model.predict.call_args[0][0]
        assert list(call_args.columns) == feature_cols
        assert len(result) == 2

    def test_predict_single_month_routes_by_cluster(self):
        """Each cluster's rows go to the correct model."""
        from unittest.mock import MagicMock
        from common.backtest_framework import _predict_single_month

        feature_cols = ["qty_lag_1", "ml_cluster"]
        predict_data = pd.DataFrame({
            "dfu_ck": ["A", "B"],
            "dmdunit": ["A", "B"],
            "dmdgroup": ["G", "G"],
            "loc": ["L1", "L2"],
            "startdate": [pd.Timestamp("2025-01-01")] * 2,
            "qty_lag_1": [100.0, 200.0],
            "ml_cluster": ["c0", "c1"],
        })

        m0 = MagicMock()
        m0.predict = MagicMock(return_value=np.array([110.0]))
        m1 = MagicMock()
        m1.predict = MagicMock(return_value=np.array([220.0]))
        models = {"c0": m0, "c1": m1}

        result = _predict_single_month(models, predict_data, feature_cols)
        assert len(result) == 2
        assert m0.predict.call_count == 1
        assert m1.predict.call_count == 1

    def test_predict_single_month_missing_cluster_skips(self):
        """DFUs with no matching cluster model are silently skipped."""
        from unittest.mock import MagicMock
        from common.backtest_framework import _predict_single_month

        feature_cols = ["qty_lag_1", "ml_cluster"]
        predict_data = pd.DataFrame({
            "dfu_ck": ["A"],
            "dmdunit": ["A"],
            "dmdgroup": ["G"],
            "loc": ["L1"],
            "startdate": [pd.Timestamp("2025-01-01")],
            "qty_lag_1": [100.0],
            "ml_cluster": ["unknown_cluster"],
        })

        models = {}  # no models at all
        result = _predict_single_month(models, predict_data, feature_cols)
        assert len(result) == 0


class TestPlanningDateCap:
    """Verify that load_backtest_data and run_tree_backtest respect planning date."""

    def test_load_backtest_data_sql_includes_planning_cutoff(self):
        """The SQL query in load_backtest_data must filter by planning date."""
        from common.backtest_framework import load_backtest_data
        import inspect
        source = inspect.getsource(load_backtest_data)
        assert "planning_cutoff" in source
        assert "get_planning_date" in source

    def test_run_tree_backtest_caps_latest_month(self):
        """run_tree_backtest must cap latest_month at planning date."""
        from common.backtest_framework import run_tree_backtest
        import inspect
        source = inspect.getsource(run_tree_backtest)
        assert "planning_cutoff" in source or "planning_dt" in source
        assert "get_planning_date" in source

    def test_planning_date_returns_feb_2026(self):
        """With default config, planning date is 2026-02-24."""
        from common.planning_date import get_planning_date, _reset_cache
        _reset_cache()
        d = get_planning_date()
        assert d == datetime.date(2026, 2, 24)

    def test_planning_cutoff_is_first_of_month(self):
        """The planning cutoff for SQL queries should be first-of-month."""
        from common.planning_date import get_planning_date
        cutoff = get_planning_date().replace(day=1)
        assert cutoff.day == 1
        assert cutoff == datetime.date(2026, 2, 1)
