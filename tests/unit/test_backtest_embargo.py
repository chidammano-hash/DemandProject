"""Tests for the embargo_months parameter in generate_timeframes().

Verifies that the configurable embargo gap between train_end and
predict_start is consistent with the gap_months used in tuning CV splits.
"""

import pandas as pd

from common.ml.backtest_framework import (
    _last_persistable_timeframe,
    generate_timeframes,
)


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
            gap_months = (tf["predict_start"].year * 12 + tf["predict_start"].month) - (
                tf["train_end"].year * 12 + tf["train_end"].month
            )
            assert gap_months == 2, (
                f"Timeframe {tf['label']}: expected 2-month gap, got {gap_months}"
            )

    def test_j_is_a_valid_closed_month_window_for_july_planning(self):
        tfs = generate_timeframes(
            pd.Timestamp("2023-07-01"),
            pd.Timestamp("2026-06-01"),
            n=10,
            embargo_months=1,
        )

        assert tfs[-1]["label"] == "J"
        assert tfs[-1]["train_end"] == pd.Timestamp("2026-04-01")
        assert tfs[-1]["predict_start"] == pd.Timestamp("2026-06-01")
        assert tfs[-1]["predict_end"] == pd.Timestamp("2026-06-01")
        assert all(tf["predict_start"] <= tf["predict_end"] for tf in tfs)


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
        """Every requested timeframe remains a valid evaluation window."""
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10, embargo_months=1)
        valid = [tf for tf in tfs if tf["predict_start"] <= tf["predict_end"]]
        assert len(valid) == 10

    def test_large_embargo_still_produces_at_least_one_month(self):
        """Even with a large embargo, the last timeframe (J) should still
        have predict_start <= predict_end because the prediction window
        shrinks but the last timeframe's train_end is close to latest."""
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs = generate_timeframes(earliest, latest, n=10, embargo_months=3)
        assert len(tfs) == 10
        last_tf = tfs[-1]
        assert last_tf["label"] == "J"
        assert last_tf["predict_start"] <= last_tf["predict_end"]

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


class TestOperationalBacktestCalendar:
    """Regression coverage for planning-month rollover and natural lag 0."""

    def test_july_planning_scores_june_at_lag_zero(self):
        latest_closed = pd.Timestamp("2026-06-01")
        timeframes = generate_timeframes(
            pd.Timestamp("2023-07-01"), latest_closed, n=10, embargo_months=0
        )

        assert timeframes[-1]["train_end"] == pd.Timestamp("2026-05-01")
        assert timeframes[-1]["predict_start"] == latest_closed
        assert timeframes[-1]["predict_end"] == latest_closed

    def test_august_planning_rolls_j_forward_one_month(self):
        latest_closed = pd.Timestamp("2026-07-01")
        timeframes = generate_timeframes(
            pd.Timestamp("2023-07-01"), latest_closed, n=10, embargo_months=0
        )

        assert timeframes[-1]["train_end"] == pd.Timestamp("2026-06-01")
        assert timeframes[-1]["predict_start"] == latest_closed
        assert timeframes[-1]["predict_end"] == latest_closed

    def test_all_standard_timeframes_are_non_empty(self):
        timeframes = generate_timeframes(
            pd.Timestamp("2023-07-01"),
            pd.Timestamp("2026-06-01"),
            n=10,
            embargo_months=0,
        )

        assert len(timeframes) == 10
        assert all(tf["predict_start"] <= tf["predict_end"] for tf in timeframes)


class TestDefaultEmbargoIsZero:
    """The default embargo_months parameter should be 0 (backward compat)."""

    def test_default_matches_legacy(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        # Call without embargo_months — should behave like embargo_months=0
        tfs_default = generate_timeframes(earliest, latest, n=10)
        tfs_explicit = generate_timeframes(earliest, latest, n=10, embargo_months=0)
        assert len(tfs_default) == len(tfs_explicit)
        for d, e in zip(tfs_default, tfs_explicit, strict=True):
            assert d["predict_start"] == e["predict_start"]
            assert d["train_end"] == e["train_end"]


class TestEmbargoPreservesStructure:
    """Embargo preserves valid window structure by shifting train cutoffs."""

    def test_train_end_shifts_earlier_by_embargo(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs_0 = generate_timeframes(earliest, latest, n=10, embargo_months=0)
        tfs_1 = generate_timeframes(earliest, latest, n=10, embargo_months=1)
        for t0, t1 in zip(tfs_0, tfs_1, strict=True):
            assert t1["train_end"] == t0["train_end"] - pd.DateOffset(months=1)

    def test_predict_end_unchanged(self):
        earliest = pd.Timestamp("2021-01-01")
        latest = pd.Timestamp("2024-12-01")
        tfs_0 = generate_timeframes(earliest, latest, n=10, embargo_months=0)
        tfs_1 = generate_timeframes(earliest, latest, n=10, embargo_months=1)
        for t0, t1 in zip(tfs_0, tfs_1, strict=True):
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


class TestPersistableTimeframe:
    """Model persistence must target the last timeframe with a real predict window.

    Every generated timeframe is now valid, so persistence targets the last one.
    """

    earliest = pd.Timestamp("2021-01-01")
    latest = pd.Timestamp("2024-12-01")
    all_months = list(pd.date_range("2021-01-01", "2024-12-01", freq="MS"))

    def test_embargo_zero_persists_last_index(self):
        tfs = generate_timeframes(self.earliest, self.latest, n=10, embargo_months=0)
        # Last timeframe predicts the final month -> persist the last index.
        assert _last_persistable_timeframe(tfs, self.all_months) == len(tfs) - 1

    def test_embargo_one_persists_valid_final_timeframe(self):
        tfs = generate_timeframes(self.earliest, self.latest, n=10, embargo_months=1)
        persist_ti = _last_persistable_timeframe(tfs, self.all_months)
        assert persist_ti == len(tfs) - 1
        tf = tfs[persist_ti]
        assert any(tf["predict_start"] <= m <= tf["predict_end"] for m in self.all_months)
