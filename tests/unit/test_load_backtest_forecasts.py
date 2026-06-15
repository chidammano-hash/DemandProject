"""Characterization tests for scripts/etl/load_backtest_forecasts.py.

US1 pinned this loader's behavior; US3 moved its index/constraint management
into common/core/etl_helpers.py (covered by tests/unit/test_etl_helpers.py).
What remains here are the loader's own column/table constants.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.etl import load_backtest_forecasts as bt


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
