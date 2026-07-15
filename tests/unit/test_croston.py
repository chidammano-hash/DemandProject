from __future__ import annotations

import numpy as np
import pytest

from common.ml.croston import croston_forecast


def test_croston_sba_forecast_updates_demand_size_and_interval() -> None:
    forecast = croston_forecast(
        np.array([0.0, 10.0, 0.0, 0.0, 20.0]),
        horizon=3,
        params={
            "alpha": 0.2,
            "variant": "sba",
            "recursive": False,
            "recursive_damping": 0.5,
        },
    )

    # z=12, p=2.2 after the second non-zero observation; SBA applies
    # the (1 - alpha/2) bias correction.
    assert forecast.tolist() == pytest.approx([4.9090909] * 3)


def test_croston_sba_recursive_forecast_damps_latest_demand_toward_rate() -> None:
    forecast = croston_forecast(
        np.array([0.0, 10.0, 0.0, 0.0, 20.0]),
        horizon=3,
        params={
            "alpha": 0.2,
            "variant": "sba",
            "recursive": True,
            "recursive_damping": 0.5,
        },
    )

    # The SBA rate is 4.9090909. Each month recursively damps the prior
    # projected demand toward that long-run intermittent-demand rate.
    assert forecast.tolist() == pytest.approx([12.45454545, 8.68181818, 6.79545455])
    assert len(set(forecast)) == 3


def test_croston_all_zero_history_returns_zero_forecast() -> None:
    forecast = croston_forecast(
        np.zeros(18),
        horizon=4,
        params={
            "alpha": 0.1,
            "variant": "sba",
            "recursive": True,
            "recursive_damping": 0.5,
        },
    )

    assert forecast.tolist() == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (
            {
                "alpha": 0.0,
                "variant": "sba",
                "recursive": True,
                "recursive_damping": 0.5,
            },
            "alpha",
        ),
        (
            {
                "alpha": 0.1,
                "variant": "unknown",
                "recursive": True,
                "recursive_damping": 0.5,
            },
            "variant",
        ),
        (
            {
                "alpha": 0.1,
                "variant": "sba",
                "recursive": True,
                "recursive_damping": 1.0,
            },
            "recursive damping",
        ),
    ],
)
def test_croston_rejects_invalid_configuration(
    params: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        croston_forecast(np.array([1.0, 0.0]), horizon=2, params=params)
