"""Expert-system backtests must never bypass the release transaction."""

from pathlib import Path


def test_expert_system_backtest_does_not_write_production_forecast():
    source = Path("scripts/ml/run_expert_system_backtest.py").read_text()

    assert "def load_production_forecast(" not in source
    assert "INSERT INTO fact_production_forecast" not in source
    assert "DELETE FROM fact_production_forecast" not in source
