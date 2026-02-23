"""Tests for common/constants.py."""

from common.constants import (
    CAT_FEATURES,
    NUMERIC_DFU_FEATURES,
    NUMERIC_ITEM_FEATURES,
    LAG_RANGE,
    ROLLING_WINDOWS,
    OUTPUT_COLS,
    ARCHIVE_COLS,
    METADATA_COLS,
    MAX_ARCHIVE_LAG,
    MIN_TRAINING_MONTHS,
    MIN_CLUSTER_ROWS,
)


class TestConstants:
    def test_lag_range_bounds(self):
        assert min(LAG_RANGE) == 1
        assert max(LAG_RANGE) == 12
        assert len(list(LAG_RANGE)) == 12

    def test_rolling_windows_sorted(self):
        assert ROLLING_WINDOWS == sorted(ROLLING_WINDOWS)
        assert ROLLING_WINDOWS == [3, 6, 12]

    def test_output_cols_non_empty(self):
        assert len(OUTPUT_COLS) > 0
        assert len(set(OUTPUT_COLS)) == len(OUTPUT_COLS), "No duplicates in OUTPUT_COLS"

    def test_archive_cols_non_empty(self):
        assert len(ARCHIVE_COLS) > 0
        assert len(set(ARCHIVE_COLS)) == len(ARCHIVE_COLS), "No duplicates in ARCHIVE_COLS"

    def test_archive_cols_superset_of_output_cols(self):
        """Archive cols should contain all output cols plus timeframe."""
        output_set = set(OUTPUT_COLS)
        archive_set = set(ARCHIVE_COLS)
        assert output_set.issubset(archive_set)
        assert "timeframe" in archive_set

    def test_cat_features_non_empty(self):
        assert len(CAT_FEATURES) > 0
        assert all(isinstance(f, str) for f in CAT_FEATURES)

    def test_numeric_features_non_empty(self):
        assert len(NUMERIC_DFU_FEATURES) > 0
        assert len(NUMERIC_ITEM_FEATURES) > 0

    def test_metadata_cols_is_set(self):
        assert isinstance(METADATA_COLS, set)
        assert "dfu_ck" in METADATA_COLS
        assert "startdate" in METADATA_COLS
        assert "qty" in METADATA_COLS

    def test_max_archive_lag(self):
        assert MAX_ARCHIVE_LAG == 4

    def test_min_training_months(self):
        assert MIN_TRAINING_MONTHS >= 1

    def test_min_cluster_rows(self):
        assert MIN_CLUSTER_ROWS >= 1
