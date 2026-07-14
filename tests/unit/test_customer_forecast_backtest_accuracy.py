from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock
from uuid import UUID

import pandas as pd
import pytest

from common.services.customer_forecast_backtest import CustomerBacktestSettings
from common.services.customer_forecast_backtest_accuracy import (
    AccuracyMetrics,
    BacktestAccuracyComparison,
    calculate_accuracy_metrics,
    compare_backtest_accuracy,
    load_backtest_accuracy_frame,
    persist_backtest_accuracy,
)


def test_accuracy_metrics_use_canonical_wape_bias_and_accuracy_formulas() -> None:
    metrics = calculate_accuracy_metrics(
        actuals=[Decimal("100"), Decimal("50")],
        forecasts=[Decimal("120"), Decimal("70")],
    )

    assert isinstance(metrics, AccuracyMetrics)
    assert metrics.observations == 2
    assert metrics.actual_qty == Decimal("150")
    assert metrics.absolute_error == Decimal("40")
    assert metrics.mae == Decimal("20")
    assert metrics.wape_pct == pytest.approx(26.6667, abs=1e-4)
    assert metrics.accuracy_pct == pytest.approx(73.3333, abs=1e-4)
    assert metrics.bias_pct == pytest.approx(26.6667, abs=1e-4)


def test_common_cohort_accuracy_compares_champion_bottom_up_and_blend() -> None:
    rows = pd.DataFrame.from_records(
        [
            {
                "item_id": "ITEM-1",
                "loc": "LOC-1",
                "forecast_month": date(2026, 5, 1),
                "actual_qty": Decimal("100"),
                "champion_qty": Decimal("120"),
                "normalized_customer_qty": Decimal("80"),
                "blended_qty": Decimal("100"),
            },
            {
                "item_id": "ITEM-2",
                "loc": "LOC-2",
                "forecast_month": date(2026, 6, 1),
                "actual_qty": Decimal("50"),
                "champion_qty": Decimal("70"),
                "normalized_customer_qty": Decimal("40"),
                "blended_qty": Decimal("55"),
            },
            {
                # This row must be excluded from every model so metrics share one cohort.
                "item_id": "ITEM-3",
                "loc": "LOC-3",
                "forecast_month": date(2026, 6, 1),
                "actual_qty": Decimal("100"),
                "champion_qty": Decimal("100"),
                "normalized_customer_qty": None,
                "blended_qty": Decimal("100"),
            },
        ]
    )

    result = compare_backtest_accuracy(
        rows,
        settings=CustomerBacktestSettings(
            enabled=True,
            lookback_months=2,
            min_train_months=6,
            horizon_months=1,
            batch_size=10_000,
            min_common_months=2,
            min_common_dfus=2,
            max_wape_degradation_pct=0.0,
        ),
    )

    assert isinstance(result, BacktestAccuracyComparison)
    assert result.common_rows == 2
    assert result.common_dfus == 2
    assert result.common_months == 2

    champion = result.champion
    assert isinstance(champion, AccuracyMetrics)
    assert champion.actual_qty == pytest.approx(150.0)
    assert champion.wape_pct == pytest.approx(26.6667, abs=1e-4)
    assert champion.bias_pct == pytest.approx(26.6667, abs=1e-4)
    assert champion.accuracy_pct == pytest.approx(73.3333, abs=1e-4)

    bottom_up = result.customer_bottom_up
    assert isinstance(bottom_up, AccuracyMetrics)
    assert bottom_up.wape_pct == pytest.approx(20.0)
    assert bottom_up.bias_pct == pytest.approx(-20.0)
    assert bottom_up.accuracy_pct == pytest.approx(80.0)

    blend = result.customer_bottom_up_blend
    assert isinstance(blend, AccuracyMetrics)
    assert blend.wape_pct == pytest.approx(3.3333, abs=1e-4)
    assert blend.bias_pct == pytest.approx(3.3333, abs=1e-4)
    assert blend.accuracy_pct == pytest.approx(96.6667, abs=1e-4)
    assert result.gate_passed is True


def test_blend_wape_delta_is_preserved_when_champion_wape_is_zero() -> None:
    rows = pd.DataFrame.from_records(
        [
            {
                "item_id": "ITEM-1",
                "loc": "LOC-1",
                "forecast_month": date(2026, 6, 1),
                "actual_qty": Decimal("100"),
                "champion_qty": Decimal("100"),
                "normalized_customer_qty": Decimal("120"),
                "blended_qty": Decimal("110"),
            }
        ]
    )

    result = compare_backtest_accuracy(
        rows,
        settings=CustomerBacktestSettings(
            enabled=True,
            lookback_months=1,
            min_train_months=6,
            horizon_months=1,
            batch_size=10_000,
            min_common_months=1,
            min_common_dfus=1,
            max_wape_degradation_pct=0.0,
        ),
    )

    assert result.champion.wape_pct == 0.0
    assert result.customer_bottom_up_blend.wape_pct == 10.0
    assert result.blend_wape_degradation_pct == 10.0
    assert result.gate_passed is False


def test_equal_exact_error_totals_do_not_fail_from_binary_float_noise() -> None:
    rows = pd.DataFrame.from_records(
        [
            {
                "item_id": "ITEM-1",
                "loc": "LOC-1",
                "forecast_month": date(2026, 5, 1),
                "actual_qty": Decimal("9.9064"),
                "champion_qty": Decimal("8.1050"),
                "normalized_customer_qty": Decimal("11.9469"),
                "blended_qty": Decimal("11.9469"),
            },
            {
                "item_id": "ITEM-2",
                "loc": "LOC-2",
                "forecast_month": date(2026, 6, 1),
                "actual_qty": Decimal("1.2429"),
                "champion_qty": Decimal("3.2834"),
                "normalized_customer_qty": Decimal("3.0443"),
                "blended_qty": Decimal("3.0443"),
            },
        ]
    )

    result = compare_backtest_accuracy(
        rows,
        settings=CustomerBacktestSettings(
            enabled=True,
            lookback_months=2,
            min_train_months=6,
            horizon_months=1,
            batch_size=10_000,
            min_common_months=2,
            min_common_dfus=2,
            max_wape_degradation_pct=0.0,
        ),
    )

    assert result.champion.absolute_error == pytest.approx(3.8419)
    assert result.customer_bottom_up_blend.absolute_error == pytest.approx(3.8419)
    assert result.blend_wape_degradation_pct == 0.0
    assert result.gate_passed is True


def test_wape_gate_compares_metrics_at_persisted_six_decimal_precision() -> None:
    rows = pd.DataFrame.from_records(
        [
            {
                "item_id": "ITEM-1",
                "loc": "LOC-1",
                "forecast_month": date(2026, 6, 1),
                "actual_qty": Decimal("100"),
                "champion_qty": Decimal("100.99999951"),
                "normalized_customer_qty": Decimal("101.00000049"),
                "blended_qty": Decimal("101.00000049"),
            }
        ]
    )

    result = compare_backtest_accuracy(
        rows,
        settings=CustomerBacktestSettings(
            enabled=True,
            lookback_months=1,
            min_train_months=6,
            horizon_months=1,
            batch_size=10_000,
            min_common_months=1,
            min_common_dfus=1,
            max_wape_degradation_pct=0.0,
        ),
    )

    assert result.champion.wape_pct == pytest.approx(0.99999951)
    assert result.customer_bottom_up_blend.wape_pct == pytest.approx(1.00000049)
    assert result.blend_wape_degradation_pct == 0.0
    assert result.gate_passed is True


def test_load_backtest_accuracy_frame_uses_exact_common_cohort_columns() -> None:
    cursor = MagicMock()
    cursor.description = [
        ("item_id",),
        ("loc",),
        ("forecast_month",),
        ("actual_qty",),
        ("normalized_customer_qty",),
        ("champion_qty",),
        ("blended_qty",),
    ]
    cursor.fetchall.return_value = [("ITEM-1", "LOC-1", date(2026, 6, 1), 100, 90, 110, 100)]
    run_id = UUID("00000000-0000-0000-0000-000000000501")

    frame = load_backtest_accuracy_frame(cursor, run_id)

    assert frame.to_dict("records") == [
        {
            "item_id": "ITEM-1",
            "loc": "LOC-1",
            "forecast_month": date(2026, 6, 1),
            "actual_qty": 100,
            "normalized_customer_qty": 90,
            "champion_qty": 110,
            "blended_qty": 100,
        }
    ]
    sql, params = cursor.execute.call_args.args
    assert "FROM customer_bottom_up_backtest_component" in sql
    assert params == (str(run_id),)


def test_persist_backtest_accuracy_writes_metrics_and_gate_contract() -> None:
    cursor = MagicMock()
    run_id = UUID("00000000-0000-0000-0000-000000000501")
    metric = AccuracyMetrics(
        observations=2,
        actual_qty=150.0,
        absolute_error=30.0,
        mae=15.0,
        wape_pct=20.0,
        accuracy_pct=80.0,
        bias_pct=2.0,
    )
    comparison = BacktestAccuracyComparison(
        common_months=2,
        common_dfus=2,
        common_rows=2,
        champion=metric,
        customer_bottom_up=metric,
        customer_bottom_up_blend=metric,
        blend_wape_degradation_pct=0.0,
        gate_passed=True,
        gate_reason="passed",
    )
    settings = CustomerBacktestSettings(
        enabled=True,
        lookback_months=2,
        min_train_months=6,
        horizon_months=1,
        batch_size=10_000,
        min_common_months=2,
        min_common_dfus=2,
        max_wape_degradation_pct=0.0,
    )

    persist_backtest_accuracy(
        cursor,
        run_id=run_id,
        evaluation_start=date(2026, 5, 1),
        evaluation_end=date(2026, 6, 1),
        comparison=comparison,
        settings=settings,
    )

    sql, params = cursor.execute.call_args.args
    assert "INSERT INTO customer_bottom_up_backtest_accuracy" in sql
    assert params[:7] == (
        str(run_id),
        date(2026, 5, 1),
        date(2026, 6, 1),
        2,
        2,
        2,
        150.0,
    )
    assert params[-5:] == (2, 2, 0.0, True, "passed")
