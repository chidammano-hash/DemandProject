"""Characterization tests for scripts/etl/load_backtest_forecasts.py.

US1 pinned this loader's behavior; US3 moved its index/constraint management
into common/core/etl_helpers.py (covered by tests/unit/test_etl_helpers.py).
What remains here are the loader's own column/table constants.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.etl import load_backtest_forecasts as bt


class TestArchiveStreaming:
    """US9: the real archive owner streams in BATCH_SIZE chunks (not one giant
    INSERT). External forecasts skip the archive entirely (see load_dataset_postgres)."""

    def test_batch_size_is_positive_int(self):
        assert isinstance(bt.BATCH_SIZE, int)
        assert bt.BATCH_SIZE > 0

    def test_load_archive_loops_in_batches(self):
        import inspect
        src = inspect.getsource(bt._load_archive)
        assert "BATCH_SIZE" in src
        assert "range(" in src  # batched loop over staged rows


class TestConstants:
    def test_main_and_archive_tables(self):
        assert bt._TABLE == "fact_external_forecast_monthly"
        assert bt._ARCHIVE_TABLE == "backtest_lag_archive"

    def test_load_cols_lead_with_forecast_ck(self):
        assert bt.LOAD_COLS[0] == "forecast_ck"
        assert "model_id" in bt.LOAD_COLS

    def test_archive_cols_end_with_timeframe(self):
        assert bt.ARCHIVE_COLS[-1] == "timeframe"
        assert bt.ARCHIVE_COLS[0] == "forecast_ck"
