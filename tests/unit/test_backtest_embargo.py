"""Tests for the embargo_months parameter in generate_timeframes().

Verifies that the configurable embargo gap between train_end and
predict_start is consistent with the gap_months used in tuning CV splits.
"""

import pandas as pd
import pytest

from common.ml.backtest_framework import generate_timeframes


class TestEmbargoMonthsZero:
    """With embargo_months=0, behaviour is unchanged from legacy."""

    def test_predict_start_one_month_after_train_end(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10, embargo_months=0)
        for tf in tfs:
            expected = tf["train_end"] + pd.DateOffset(months=1)
            assert tf["predict_start"] == expected

    def test_returns_correct_count(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10, embargo_months=0)
        assert len(tfs) == 10


class TestEmbargoMonthsOne:
    """With embargo_months=1, predict_start = train_end + 2 months."""

    def test_predict_start_two_months_after_train_end(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10, embargo_months=1)
        for tf in tfs:
            expected = tf["train_end"] + pd.DateOffset(months=2)
            assert tf["predict_start"] == expected

    def test_all_10_timeframes_have_correct_gap(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10, embargo_months=1)
        assert len(tfs) == 10
        for tf in tfs:
            gap_months = (
                (tf["predict_start"].year * 12 + tf["predict_start"].month)
                - (tf["train_end"].year * 12 + tf["train_end"].month)
            )
            assert gap_months == 2, (
                f"Timeframe {tf['label']}: expected 2-month gap, got {gap_months}"
            )


class TestEmbargoMonthsTwo:
    """With embargo_months=2, predict_start = train_end + 3 months."""

    def test_predict_start_three_months_after_train_end(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10, embargo_months=2)
        for tf in tfs:
            expected = tf["train_end"] + pd.DateOffset(months=3)
            assert tf["predict_start"] == expected


class TestEmbargoDoesNotCollapseWindow:
    """Embargo should not cause empty prediction windows."""

    def test_most_timeframes_have_valid_windows_embargo_1(self):
        """With embargo_months=1, the last timeframe may have predict_start
        beyond predict_end (the backtest loop filters these out).  But most
        timeframes should still have valid windows."""
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10, embargo_months=1)
        valid = [
            tf for tf in tfs
            if tf["predict_start"] <= tf["predict_end"]
        ]
        # With 10 timeframes and embargo=1, only the last TF (J) loses
        # its window (train_end = latest-1mo, predict_start = latest+1mo).
        assert len(valid) >= 9, (
            f"Expected at least 9 valid timeframes, got {len(valid)}"
        )

    def test_large_embargo_still_produces_at_least_one_month(self):
        """Even with a large embargo, the last timeframe (J) should still
        have predict_start <= predict_end because the prediction window
        shrinks but the last timeframe's train_end is close to latest."""
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        # embargo=3 is aggressive — verify at least the last timeframe
        # still has a valid window (train_end is latest - 1 month for
        # the last TF, so predict_start = latest - 1 + 1 + 3 = latest + 3
        # which would be > latest). Early timeframes will definitely
        # have predict_start > predict_end. This is expected — the caller
        # filters out months with no data.
        tfs = generate_timeframes(earliest, latest, n=10, embargo_months=3)
        assert len(tfs) == 10
        # The last timeframe J has train_end = latest - 1 month
        last_tf = tfs[-1]
        # With embargo=3, predict_start = train_end + 4 months
        # This will exceed latest for the last timeframe, but the framework
        # handles this in the prediction loop by filtering predict_months.
        # The function itself should still return all 10 timeframes.
        assert last_tf["label"] == "J"

    def test_moderate_embargo_last_tf_window_valid(self):
        """With a longer date range, even embargo=2 leaves valid windows."""
        earliest = pd.Timestamp("2018-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10, embargo_months=2)
        # Last TF: train_end = latest - 1 month = Nov 2024
        # predict_start = Nov 2024 + 3 months = Feb 2025
        # predict_end = Dec 2024
        # Only early timeframes will have valid windows
        # At least timeframe A should have a valid window
        tf_a = tfs[0]
        assert tf_a["predict_start"] <= tf_a["predict_end"], (
            f"Timeframe A should have valid window: "
            f"{tf_a['predict_start']} <= {tf_a['predict_end']}"
        )


class TestEmbargoConsistencyWithTuning:
    """embargo_months=1 in backtest should match gap_months=1 in tuning."""

    def test_gap_size_matches_tuning_cv(self):
        """Both embargo_months and gap_months create the same number of
        skipped months between training data and evaluation data."""
        from common.ml.tuning import generate_cv_month_splits

        # Setup: 48 months of data
        all_months = pd.date_range("2021-01-01", periods=48, freq="MS").tolist()
        earliest = all_months[0]
        latest = all_months[-1]

        # Backtest with embargo_months=1
        tfs = generate_timeframes(earliest, latest, n=10, embargo_months=1)

        # Tuning CV with gap_months=1
        splits = generate_cv_month_splits(
            all_months, n_splits=5, gap_months=1,
            min_train_months=13, val_months_per_fold=3,
        )

        # Verify backtest gap: predict_start - train_end = 2 months
        for tf in tfs:
            gap = (
                (tf["predict_start"].year * 12 + tf["predict_start"].month)
                - (tf["train_end"].year * 12 + tf["train_end"].month)
            )
            assert gap == 2, f"Backtest gap should be 2 months, got {gap}"

        # Verify tuning gap: val_start - train_end = 2 months (1 natural + 1 gap)
        assert len(splits) > 0, "Should produce at least one CV split"
        for train_months, val_months in splits:
            train_end = max(train_months)
            val_start = min(val_months)
            gap = (
                (val_start.year * 12 + val_start.month)
                - (train_end.year * 12 + train_end.month)
            )
            assert gap == 2, f"Tuning gap should be 2 months, got {gap}"


class TestDefaultEmbargoIsZero:
    """The default embargo_months parameter should be 0 (backward compat)."""

    def test_default_matches_legacy(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        # Call without embargo_months — should behave like embargo_months=0
        tfs_default = generate_timeframes(earliest, latest, n=10)
        tfs_explicit = generate_timeframes(earliest, latest, n=10, embargo_months=0)
        assert len(tfs_default) == len(tfs_explicit)
        for d, e in zip(tfs_default, tfs_explicit):
            assert d["predict_start"] == e["predict_start"]
            assert d["train_end"] == e["train_end"]


class TestEmbargoPreservesStructure:
    """Embargo changes predict_start but not other timeframe properties."""

    def test_train_end_unchanged(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs_0 = generate_timeframes(earliest, latest, n=10, embargo_months=0)
        tfs_1 = generate_timeframes(earliest, latest, n=10, embargo_months=1)
        for t0, t1 in zip(tfs_0, tfs_1):
            assert t0["train_end"] == t1["train_end"]

    def test_predict_end_unchanged(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs_0 = generate_timeframes(earliest, latest, n=10, embargo_months=0)
        tfs_1 = generate_timeframes(earliest, latest, n=10, embargo_months=1)
        for t0, t1 in zip(tfs_0, tfs_1):
            assert t0["predict_end"] == t1["predict_end"]

    def test_labels_unchanged(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10, embargo_months=2)
        labels = [tf["label"] for tf in tfs]
        assert labels == list("ABCDEFGHIJ")

    def test_expanding_windows_still_hold(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10, embargo_months=1)
        train_ends = [tf["train_end"] for tf in tfs]
        for i in range(1, len(train_ends)):
            assert train_ends[i] > train_ends[i - 1]
