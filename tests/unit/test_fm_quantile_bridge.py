"""Unit tests for common.ml.fm_quantile_bridge."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from common.ml.fm_quantile_bridge import (
    FMQuantileForecast,
    fm_demand_pool,
    load_fm_quantile_forecast,
)


def _rows(n_months: int = 3):
    """Build fake (month, p10, p50, p90) rows."""
    return [
        (f"2026-0{m}-01", 5.0 + m, 10.0 + m, 15.0 + m)
        for m in range(1, n_months + 1)
    ]


def test_to_sample_array_shape_and_values():
    matrix = np.array([[5.0, 10.0, 15.0], [6.0, 12.0, 18.0]])
    fc = FMQuantileForecast(
        item_id="I1", loc="L1", model_id="chronos2_enriched",
        quantile_matrix=matrix, quantile_levels=(0.1, 0.5, 0.9),
        months=("2026-01-01", "2026-02-01"),
    )
    samples = fc.to_sample_array(n_samples=200, rng=np.random.default_rng(0))
    assert samples.shape == (2, 200)
    # Drawn values lie within [min, max] of the stored quantiles
    assert samples[0].min() >= 5.0
    assert samples[0].max() <= 15.0


def test_load_fm_quantile_forecast_returns_none_when_empty():
    cur = MagicMock()
    cur.fetchall.return_value = []
    result = load_fm_quantile_forecast(
        cur, "I1", "L1",
        fm_config={"model_id": "chronos2_enriched", "quantiles": [0.1, 0.5, 0.9]},
    )
    assert result is None


def test_load_fm_quantile_forecast_happy_path():
    cur = MagicMock()
    cur.fetchall.return_value = _rows(3)
    result = load_fm_quantile_forecast(
        cur, "I1", "L1",
        fm_config={"model_id": "chronos2_enriched", "quantiles": [0.1, 0.5, 0.9]},
    )
    assert result is not None
    assert result.quantile_matrix.shape == (3, 3)
    # Stored lower/upper values should come through unchanged.
    assert result.quantile_matrix[0, 0] == pytest.approx(6.0)  # 5 + 1
    assert result.quantile_matrix[0, 2] == pytest.approx(16.0)


def test_load_fm_quantile_forecast_handles_null_bounds():
    cur = MagicMock()
    cur.fetchall.return_value = [("2026-01-01", None, 10.0, None)]
    result = load_fm_quantile_forecast(
        cur, "I1", "L1",
        fm_config={"model_id": "chronos2_enriched", "quantiles": [0.1, 0.5, 0.9]},
    )
    # NULL bounds -> +/- 20% fill
    assert result.quantile_matrix[0, 0] == pytest.approx(8.0)
    assert result.quantile_matrix[0, 2] == pytest.approx(12.0)


def test_load_fm_quantile_forecast_clips_negative():
    cur = MagicMock()
    cur.fetchall.return_value = [("2026-01-01", -5.0, -1.0, 2.0)]
    result = load_fm_quantile_forecast(
        cur, "I1", "L1",
        fm_config={"model_id": "chronos2_enriched", "quantiles": [0.1, 0.5, 0.9]},
    )
    # Negative demand clipped to zero
    assert result.quantile_matrix[0, 0] == 0.0
    assert result.quantile_matrix[0, 1] == 0.0


def test_fm_demand_pool_returns_flat_array():
    cur = MagicMock()
    cur.fetchall.return_value = _rows(2)
    pool = fm_demand_pool(
        cur, "I1", "L1",
        n_samples=50,
        fm_config={"model_id": "chronos2_enriched", "quantiles": [0.1, 0.5, 0.9]},
    )
    assert pool is not None
    # 2 months * 50 samples
    assert pool.shape == (100,)


def test_fm_demand_pool_none_when_no_rows():
    cur = MagicMock()
    cur.fetchall.return_value = []
    pool = fm_demand_pool(
        cur, "I1", "L1",
        fm_config={"model_id": "chronos2_enriched", "quantiles": [0.1, 0.5, 0.9]},
    )
    assert pool is None


def test_load_fm_quantile_forecast_invalid_horizon():
    cur = MagicMock()
    with pytest.raises(ValueError):
        load_fm_quantile_forecast(
            cur, "I1", "L1", horizon_months=0,
            fm_config={"model_id": "chronos2_enriched", "quantiles": [0.1, 0.5, 0.9]},
        )
