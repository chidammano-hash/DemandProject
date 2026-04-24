"""Tests for scripts/ml/fit_elasticity (run_fit fallback path)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from scripts.ml.fit_elasticity import run_fit


def _cfg():
    return {
        "min_obs": 5,
        "history_months": 36,
        "features": [
            {"name": "price_log", "column_source": "signal:unit_price", "transform": "log1p"},
            {"name": "promo_flag", "column_source": "signal:promo_flag", "transform": "none"},
        ],
        "run_id_prefix": "testrun",
    }


def _mock_rows(n=30):
    import numpy as np
    rng = np.random.default_rng(seed=1)
    rows = []
    for i in range(n):
        unit_price = float(10 + rng.normal(scale=1))
        promo = float(i % 3 == 0)
        qty = 50.0 - 2.0 * unit_price + 5.0 * promo + float(rng.normal(scale=0.5))
        rows.append(("ITEM1", "LOC1", f"2026-{(i % 12) + 1:02d}-01", qty, unit_price, promo, 0.0))
    return rows


def test_run_fit_fallback_writes_rows():
    cursor = MagicMock()
    cursor.fetchall.return_value = _mock_rows(40)
    n_written = run_fit(cursor, _cfg(), dry_run=False, item_id="ITEM1", loc="LOC1")
    assert n_written == 2   # one INSERT per feature resolved
    inserts = [c for c in cursor.execute.call_args_list
               if "INSERT INTO fact_causal_elasticity" in c.args[0]]
    assert len(inserts) == 2
    # Every INSERT gets 9 parameters (item_id, loc, feature, coef, p, se, n, method, run_id)
    for call in inserts:
        assert len(call.args[1]) == 9
        assert call.args[1][0] == "ITEM1"
        assert call.args[1][7] == "linear_regression"
        assert call.args[1][8].startswith("testrun_")


def test_run_fit_dry_run_inserts_nothing():
    cursor = MagicMock()
    cursor.fetchall.return_value = _mock_rows(40)
    n = run_fit(cursor, _cfg(), dry_run=True, item_id=None, loc=None)
    assert n == 0
    inserts = [c for c in cursor.execute.call_args_list
               if "INSERT INTO fact_causal_elasticity" in c.args[0]]
    assert inserts == []


def test_run_fit_empty_data_returns_zero():
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    assert run_fit(cursor, _cfg(), dry_run=False, item_id=None, loc=None) == 0


def test_run_fit_below_min_obs_skips():
    cursor = MagicMock()
    cursor.fetchall.return_value = _mock_rows(3)  # below min_obs=5
    assert run_fit(cursor, _cfg(), dry_run=False, item_id=None, loc=None) == 0
