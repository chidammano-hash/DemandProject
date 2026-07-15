from __future__ import annotations

import numpy as np
import pytest

from common.ml.croston import croston_forecast
from common.ml.customer_forecast_rules import (
    CROSTON_ROUTE_ID,
    MOVING_AVERAGE_ROUTE_ID,
    SEASONAL_REPEAT_ROUTE_ID,
    CustomerForecastRuleParameters,
    forecast_customer_demand,
    parse_customer_forecast_rule_parameters,
    select_customer_forecast_route,
)


@pytest.fixture
def rule_params() -> CustomerForecastRuleParameters:
    return CustomerForecastRuleParameters(
        recent_demand_lookback_months=6,
        moving_average_window_months=3,
        repeat_history_lookback_months=12,
        repeat_history_min_demand_months=9,
    )


def test_recent_demand_route_takes_precedence_over_dense_history(
    rule_params: CustomerForecastRuleParameters,
) -> None:
    history = np.arange(1.0, 13.0)

    route = select_customer_forecast_route(
        history,
        demand_started_within_recent_window=True,
        params=rule_params,
    )

    assert route == MOVING_AVERAGE_ROUTE_ID


@pytest.mark.parametrize(
    ("positive_months", "expected_route"),
    [
        pytest.param(9, SEASONAL_REPEAT_ROUTE_ID, id="nine-positive-months"),
        pytest.param(8, CROSTON_ROUTE_ID, id="eight-positive-months"),
    ],
)
def test_trailing_twelve_demand_density_selects_repeat_or_croston(
    rule_params: CustomerForecastRuleParameters,
    positive_months: int,
    expected_route: str,
) -> None:
    history = np.zeros(18, dtype=float)
    history[-positive_months:] = 1.0

    route = select_customer_forecast_route(
        history,
        demand_started_within_recent_window=False,
        params=rule_params,
    )

    assert route == expected_route


def test_seasonal_repeat_requires_a_complete_twelve_month_cycle(
    rule_params: CustomerForecastRuleParameters,
) -> None:
    route = select_customer_forecast_route(
        np.ones(9, dtype=float),
        demand_started_within_recent_window=False,
        params=rule_params,
    )

    assert route == CROSTON_ROUTE_ID


@pytest.mark.parametrize(
    ("moving_average_months", "repeat_months"),
    [
        pytest.param(4, 12, id="route-id-requires-three-month-average"),
        pytest.param(3, 11, id="profile-requires-twelve-month-repeat"),
    ],
)
def test_rule_parameter_parser_enforces_canonical_route_windows(
    moving_average_months: int,
    repeat_months: int,
) -> None:
    with pytest.raises(ValueError, match="parameters are invalid"):
        parse_customer_forecast_rule_parameters(
            {
                "recent_demand_lookback_months": 6,
                "moving_average_window_months": moving_average_months,
                "repeat_history_lookback_months": repeat_months,
                "repeat_history_min_demand_months": 9,
            }
        )


def test_three_month_moving_average_rolls_forecasts_recursively(
    rule_params: CustomerForecastRuleParameters,
) -> None:
    history = np.array([0.0, 0.0, 1.0, 2.0, 4.0, 6.0])

    forecast = forecast_customer_demand(
        history,
        horizon=3,
        route_model_id=MOVING_AVERAGE_ROUTE_ID,
        rule_params=rule_params,
        croston_params={
            "alpha": 0.1,
            "variant": "sba",
            "recursive": True,
            "recursive_damping": 0.5,
        },
    )

    assert forecast == pytest.approx(
        [
            4.0,
            14.0 / 3.0,
            44.0 / 9.0,
        ]
    )


def test_seasonal_repeat_cycles_the_last_twelve_actual_months(
    rule_params: CustomerForecastRuleParameters,
) -> None:
    history = np.arange(1.0, 19.0)

    forecast = forecast_customer_demand(
        history,
        horizon=18,
        route_model_id=SEASONAL_REPEAT_ROUTE_ID,
        rule_params=rule_params,
        croston_params={
            "alpha": 0.1,
            "variant": "sba",
            "recursive": True,
            "recursive_damping": 0.5,
        },
    )

    assert forecast.tolist() == [*np.arange(7.0, 19.0), *np.arange(7.0, 13.0)]


def test_croston_route_uses_the_configured_recursive_sba_forecast(
    rule_params: CustomerForecastRuleParameters,
) -> None:
    history = np.array([4.0, 0.0, 0.0, 6.0, 0.0, 0.0])
    croston_params = {
        "alpha": 0.1,
        "variant": "sba",
        "recursive": True,
        "recursive_damping": 0.5,
    }

    forecast = forecast_customer_demand(
        history,
        horizon=4,
        route_model_id=CROSTON_ROUTE_ID,
        rule_params=rule_params,
        croston_params=croston_params,
    )

    assert forecast == pytest.approx(croston_forecast(history, horizon=4, params=croston_params))
