from __future__ import annotations

import numpy as np
import pytest

from common.ml.croston import croston_forecast
from common.ml.customer_forecast_rules import (
    CROSTON_ROUTE_ID,
    CUSTOMER_FORECAST_ROUTE_IDS,
    MOVING_AVERAGE_ROUTE_ID,
    SEASONAL_REPEAT_ROUTE_ID,
    CustomerForecastRuleParameters,
    forecast_customer_demand,
    parse_customer_forecast_rule_parameters,
    select_customer_forecast_route,
)

_STATISTICAL_PARAMS = {
    "tsb_demand_alpha": 0.5,
    "tsb_probability_alpha": 0.5,
    "adida_alpha": 0.5,
    "ses_alpha": 0.5,
    "holt_level_alpha": 0.8,
    "holt_trend_alpha": 0.2,
    "holt_damping": 0.8,
}


@pytest.fixture
def rule_params() -> CustomerForecastRuleParameters:
    return CustomerForecastRuleParameters(
        recent_demand_lookback_months=6,
        moving_average_window_months=3,
        trailing_average_window_months=6,
        minimum_positive_demand_months=3,
        repeat_history_lookback_months=12,
        seasonal_min_history_months=24,
        seasonal_min_wape_improvement_pct=5.0,
        intermittent_adi_threshold=1.32,
        lumpy_cv2_threshold=0.49,
        decay_gap_adi_multiplier=1.5,
        declining_occurrence_ratio=0.5,
        trend_relative_change_threshold=0.15,
    )


def test_customer_router_exposes_only_the_canonical_v2_routes() -> None:
    assert set(CUSTOMER_FORECAST_ROUTE_IDS) == {
        "moving_average_3",
        "trailing_average_6",
        "seasonal_repeat_12",
        "tsb",
        "adida",
        "croston",
        "ses",
        "holt_damped",
    }


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


def test_fewer_than_three_positive_observations_uses_trailing_average(
    rule_params: CustomerForecastRuleParameters,
) -> None:
    history = np.array([8.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    route = select_customer_forecast_route(
        history,
        demand_started_within_recent_window=False,
        params=rule_params,
    )

    assert route == "trailing_average_6"


@pytest.mark.parametrize(
    ("history_months", "validated", "should_repeat"),
    [
        pytest.param(24, True, True, id="enough-history-and-validated"),
        pytest.param(24, False, False, id="enough-history-without-validation"),
        pytest.param(23, True, False, id="validated-without-two-full-cycles"),
    ],
)
def test_seasonal_repeat_requires_two_cycles_and_validation_evidence(
    rule_params: CustomerForecastRuleParameters,
    history_months: int,
    validated: bool,
    should_repeat: bool,
) -> None:
    cycle = np.array([8.0, 10.0, 9.0, 11.0, 12.0, 10.0, 8.0, 7.0, 9.0, 11.0, 13.0, 10.0])
    history = np.resize(cycle, history_months)

    route = select_customer_forecast_route(
        history,
        demand_started_within_recent_window=False,
        params=rule_params,
        seasonal_repeat_validated=validated,
    )

    assert (route == SEASONAL_REPEAT_ROUTE_ID) is should_repeat


def test_intermittent_decay_routes_to_tsb(rule_params: CustomerForecastRuleParameters) -> None:
    history = np.zeros(18, dtype=float)
    history[[0, 3, 6]] = [10.0, 8.0, 9.0]

    route = select_customer_forecast_route(
        history,
        demand_started_within_recent_window=False,
        params=rule_params,
    )

    assert route == "tsb"


@pytest.mark.parametrize(
    ("positive_values", "expected_route"),
    [
        pytest.param([1.0, 20.0, 1.0, 20.0], "adida", id="stable-lumpy-demand"),
        pytest.param([10.0, 11.0, 9.0, 10.0], CROSTON_ROUTE_ID, id="stable-smooth-demand"),
    ],
)
def test_stable_intermittent_series_routes_by_positive_demand_cv2(
    rule_params: CustomerForecastRuleParameters,
    positive_values: list[float],
    expected_route: str,
) -> None:
    history = np.zeros(12, dtype=float)
    history[[1, 4, 7, 10]] = positive_values

    route = select_customer_forecast_route(
        history,
        demand_started_within_recent_window=False,
        params=rule_params,
    )

    assert route == expected_route


def test_regular_trending_series_routes_to_damped_holt(
    rule_params: CustomerForecastRuleParameters,
) -> None:
    route = select_customer_forecast_route(
        np.linspace(10.0, 30.0, 12),
        demand_started_within_recent_window=False,
        params=rule_params,
    )

    assert route == "holt_damped"


def test_regular_level_series_routes_to_ses(rule_params: CustomerForecastRuleParameters) -> None:
    route = select_customer_forecast_route(
        np.tile(np.array([10.0, 11.0, 9.0, 10.0]), 3),
        demand_started_within_recent_window=False,
        params=rule_params,
    )

    assert route == "ses"


def test_adi_uses_effective_exposure_not_prelaunch_zero_padding(
    rule_params: CustomerForecastRuleParameters,
) -> None:
    history = np.array([0.0] * 11 + [10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 0.0])

    route = select_customer_forecast_route(
        history,
        demand_started_within_recent_window=False,
        params=rule_params,
        effective_history_months=7,
    )

    assert route == "ses"


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
                "trailing_average_window_months": 6,
                "minimum_positive_demand_months": 3,
                "repeat_history_lookback_months": repeat_months,
                "seasonal_min_history_months": 24,
                "seasonal_min_wape_improvement_pct": 5.0,
                "intermittent_adi_threshold": 1.32,
                "lumpy_cv2_threshold": 0.49,
                "decay_gap_adi_multiplier": 1.5,
                "declining_occurrence_ratio": 0.5,
                "trend_relative_change_threshold": 0.15,
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
        statistical_params=_STATISTICAL_PARAMS,
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
        statistical_params=_STATISTICAL_PARAMS,
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
        statistical_params=_STATISTICAL_PARAMS,
    )

    assert forecast == pytest.approx(croston_forecast(history, horizon=4, params=croston_params))


def test_trailing_six_month_average_is_a_flat_calendar_rate(
    rule_params: CustomerForecastRuleParameters,
) -> None:
    forecast = forecast_customer_demand(
        np.array([100.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]),
        horizon=3,
        route_model_id="trailing_average_6",
        rule_params=rule_params,
        croston_params={
            "alpha": 0.1,
            "variant": "sba",
            "recursive": True,
            "recursive_damping": 0.5,
        },
        statistical_params=_STATISTICAL_PARAMS,
    )

    assert forecast == pytest.approx([7.0, 7.0, 7.0])


def test_ses_forecast_uses_the_smoothed_terminal_level(
    rule_params: CustomerForecastRuleParameters,
) -> None:
    forecast = forecast_customer_demand(
        np.array([10.0, 14.0, 8.0]),
        horizon=3,
        route_model_id="ses",
        rule_params=rule_params,
        croston_params={
            "alpha": 0.1,
            "variant": "sba",
            "recursive": True,
            "recursive_damping": 0.5,
        },
        statistical_params=_STATISTICAL_PARAMS,
    )

    assert forecast == pytest.approx([10.0, 10.0, 10.0])


def test_tsb_probability_decays_recursively_without_new_occurrences(
    rule_params: CustomerForecastRuleParameters,
) -> None:
    forecast = forecast_customer_demand(
        np.array([4.0, 0.0, 0.0]),
        horizon=4,
        route_model_id="tsb",
        rule_params=rule_params,
        croston_params={
            "alpha": 0.1,
            "variant": "sba",
            "recursive": True,
            "recursive_damping": 0.5,
        },
        statistical_params=_STATISTICAL_PARAMS,
    )

    assert np.all(forecast > 0.0)
    assert forecast[1:] == pytest.approx(forecast[:-1] * 0.5)


def test_adida_aggregates_smooths_and_disaggregates_to_a_monthly_rate(
    rule_params: CustomerForecastRuleParameters,
) -> None:
    forecast = forecast_customer_demand(
        np.array([3.0, 0.0, 0.0, 9.0, 0.0, 0.0]),
        horizon=5,
        route_model_id="adida",
        rule_params=rule_params,
        croston_params={
            "alpha": 0.1,
            "variant": "sba",
            "recursive": True,
            "recursive_damping": 0.5,
        },
        statistical_params=_STATISTICAL_PARAMS,
    )

    assert forecast == pytest.approx([2.0, 2.0, 2.0, 2.0, 2.0])


def test_damped_holt_forecast_extends_trend_with_shrinking_increments(
    rule_params: CustomerForecastRuleParameters,
) -> None:
    forecast = forecast_customer_demand(
        np.array([10.0, 12.0, 14.0, 16.0, 18.0]),
        horizon=4,
        route_model_id="holt_damped",
        rule_params=rule_params,
        croston_params={
            "alpha": 0.1,
            "variant": "sba",
            "recursive": True,
            "recursive_damping": 0.5,
        },
        statistical_params=_STATISTICAL_PARAMS,
    )

    increments = np.diff(forecast)
    assert np.all(increments > 0.0)
    assert np.all(increments[1:] < increments[:-1])
