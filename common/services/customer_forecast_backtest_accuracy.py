"""Accuracy calculation and persistence for customer forecast backtests."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Protocol
from uuid import UUID

import numpy as np
import pandas as pd

_PERSISTED_METRIC_SCALE = Decimal("0.000001")


class _AccuracyGateSettings(Protocol):
    @property
    def min_common_months(self) -> int: ...

    @property
    def min_common_dfus(self) -> int: ...

    @property
    def max_wape_degradation_pct(self) -> float: ...


@dataclass(frozen=True)
class AccuracyMetrics:
    observations: int
    actual_qty: float
    absolute_error: float
    mae: float
    wape_pct: float | None
    accuracy_pct: float | None
    bias_pct: float | None


@dataclass(frozen=True)
class BacktestAccuracyComparison:
    common_months: int
    common_dfus: int
    common_rows: int
    champion: AccuracyMetrics
    customer_bottom_up: AccuracyMetrics
    customer_bottom_up_blend: AccuracyMetrics
    blend_wape_degradation_pct: float | None
    gate_passed: bool
    gate_reason: str


def calculate_accuracy_metrics(
    actuals: Iterable[float],
    forecasts: Iterable[float],
) -> AccuracyMetrics:
    actual = np.asarray(list(actuals), dtype=float)
    forecast = np.asarray(list(forecasts), dtype=float)
    if actual.ndim != 1 or forecast.ndim != 1 or len(actual) != len(forecast):
        raise ValueError("Accuracy inputs must be equal-length one-dimensional series")
    if not np.isfinite(actual).all() or not np.isfinite(forecast).all():
        raise ValueError("Accuracy inputs must be finite")
    if (actual < 0).any() or (forecast < 0).any():
        raise ValueError("Accuracy inputs must be non-negative")
    observations = len(actual)
    actual_qty = float(actual.sum())
    errors = forecast - actual
    absolute_error = float(np.abs(errors).sum())
    mae = absolute_error / observations if observations else 0.0
    if actual_qty > 0:
        wape_pct = 100.0 * absolute_error / actual_qty
        accuracy_pct = max(0.0, 100.0 - wape_pct)
        bias_pct = 100.0 * float(errors.sum()) / actual_qty
    else:
        wape_pct = None
        accuracy_pct = None
        bias_pct = None
    return AccuracyMetrics(
        observations=observations,
        actual_qty=actual_qty,
        absolute_error=absolute_error,
        mae=mae,
        wape_pct=wape_pct,
        accuracy_pct=accuracy_pct,
        bias_pct=bias_pct,
    )


def compare_backtest_accuracy(
    frame: pd.DataFrame,
    settings: _AccuracyGateSettings,
) -> BacktestAccuracyComparison:
    """Compare all three forecasts on one exact, non-null historical cohort."""
    required = {
        "item_id",
        "loc",
        "forecast_month",
        "actual_qty",
        "normalized_customer_qty",
        "champion_qty",
        "blended_qty",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Customer backtest comparison is missing columns: {sorted(missing)}")
    common = frame.dropna(
        subset=[
            "actual_qty",
            "normalized_customer_qty",
            "champion_qty",
            "blended_qty",
        ]
    )
    champion = calculate_accuracy_metrics(common["actual_qty"], common["champion_qty"])
    customer = calculate_accuracy_metrics(common["actual_qty"], common["normalized_customer_qty"])
    blend = calculate_accuracy_metrics(common["actual_qty"], common["blended_qty"])
    common_months = int(common["forecast_month"].nunique())
    common_dfus = int(common[["item_id", "loc"]].drop_duplicates().shape[0])
    if champion.wape_pct is None or blend.wape_pct is None:
        degradation = None
    else:
        persisted_champion_wape = Decimal(str(champion.wape_pct)).quantize(
            _PERSISTED_METRIC_SCALE,
            rounding=ROUND_HALF_UP,
        )
        persisted_blend_wape = Decimal(str(blend.wape_pct)).quantize(
            _PERSISTED_METRIC_SCALE,
            rounding=ROUND_HALF_UP,
        )
        degradation = float(persisted_blend_wape - persisted_champion_wape)

    failures: list[str] = []
    if common_months < settings.min_common_months:
        failures.append("insufficient common closed months")
    if common_dfus < settings.min_common_dfus:
        failures.append("insufficient common warehouse-item coverage")
    if degradation is None or degradation > settings.max_wape_degradation_pct:
        failures.append("blended WAPE exceeds the approved champion degradation threshold")
    return BacktestAccuracyComparison(
        common_months=common_months,
        common_dfus=common_dfus,
        common_rows=len(common),
        champion=champion,
        customer_bottom_up=customer,
        customer_bottom_up_blend=blend,
        blend_wape_degradation_pct=degradation,
        gate_passed=not failures,
        gate_reason="passed" if not failures else "; ".join(failures),
    )


def load_backtest_accuracy_frame(cur: Any, run_id: UUID) -> pd.DataFrame:
    cur.execute(
        """SELECT item_id, loc, forecast_month, actual_qty,
                  normalized_customer_qty, champion_qty, blended_qty
           FROM customer_bottom_up_backtest_component
           WHERE backtest_run_id = %s::uuid
           ORDER BY item_id, loc, forecast_month""",
        (str(run_id),),
    )
    columns = [description[0] for description in cur.description]
    return pd.DataFrame.from_records(cur.fetchall(), columns=columns)


def persist_backtest_accuracy(
    cur: Any,
    *,
    run_id: UUID,
    evaluation_start: date,
    evaluation_end: date,
    comparison: BacktestAccuracyComparison,
    settings: _AccuracyGateSettings,
) -> None:
    champion = comparison.champion
    customer = comparison.customer_bottom_up
    blend = comparison.customer_bottom_up_blend
    cur.execute(
        """INSERT INTO customer_bottom_up_backtest_accuracy
               (backtest_run_id, evaluation_start, evaluation_end,
                common_months, common_dfus, common_rows, actual_qty,
                customer_absolute_error, customer_mae, customer_wape_pct,
                customer_accuracy_pct, customer_bias_pct,
                champion_absolute_error, champion_mae, champion_wape_pct,
                champion_accuracy_pct, champion_bias_pct,
                blend_absolute_error, blend_mae, blend_wape_pct,
                blend_accuracy_pct, blend_bias_pct,
                blend_wape_degradation_pct, min_common_months,
                min_common_dfus, max_wape_degradation_pct,
                gate_passed, gate_reason)
           VALUES (%s::uuid, %s, %s, %s, %s, %s, %s,
                   %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                   %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (
            str(run_id),
            evaluation_start,
            evaluation_end,
            comparison.common_months,
            comparison.common_dfus,
            comparison.common_rows,
            champion.actual_qty,
            customer.absolute_error,
            customer.mae,
            customer.wape_pct,
            customer.accuracy_pct,
            customer.bias_pct,
            champion.absolute_error,
            champion.mae,
            champion.wape_pct,
            champion.accuracy_pct,
            champion.bias_pct,
            blend.absolute_error,
            blend.mae,
            blend.wape_pct,
            blend.accuracy_pct,
            blend.bias_pct,
            comparison.blend_wape_degradation_pct,
            settings.min_common_months,
            settings.min_common_dfus,
            settings.max_wape_degradation_pct,
            comparison.gate_passed,
            comparison.gate_reason,
        ),
    )
