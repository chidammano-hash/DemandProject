"""Tests for common/backtest_framework.py — timeframe generation and utilities."""

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
