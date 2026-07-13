"""Calendar contract shared by direct forecast adapters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class ForecastOutputWindow:
    """Evaluated output dates and the full model-inference calendar behind them."""

    output_months: tuple[pd.Timestamp, ...]
    inference_months: tuple[pd.Timestamp, ...]
    output_offset: int

    @property
    def inference_horizon(self) -> int:
        """Number of steps the model must produce before output slicing."""
        return len(self.inference_months)


def build_forecast_output_window(
    predict_months: Sequence[object],
    *,
    history_end: object,
    adapter_name: str,
) -> ForecastOutputWindow:
    """Include any embargo gap in inference, then expose only evaluated dates."""
    if not adapter_name.strip():
        raise ValueError("Forecast adapter_name must be non-empty")
    if not predict_months:
        raise ValueError(f"{adapter_name} prediction requires at least one output month")

    months = tuple(
        pd.Timestamp(month).to_period("M").to_timestamp() for month in predict_months
    )
    if any(pd.isna(month) for month in months):
        raise ValueError(f"{adapter_name} output months contain an invalid date")
    expected = tuple(pd.date_range(months[0], periods=len(months), freq="MS"))
    if months != expected:
        raise ValueError(
            f"{adapter_name} output months must be unique, sorted, and contiguous"
        )

    normalized_history_end = pd.Timestamp(history_end)
    if pd.isna(normalized_history_end):
        raise ValueError(f"{adapter_name} history_end is invalid")
    normalized_history_end = normalized_history_end.to_period("M").to_timestamp()
    first_distance = (
        (months[0].year - normalized_history_end.year) * 12
        + months[0].month
        - normalized_history_end.month
    )
    if first_distance <= 0:
        raise ValueError(f"{adapter_name} output months must start after history_end")

    output_offset = first_distance - 1
    inference_months = tuple(
        pd.date_range(
            normalized_history_end + pd.DateOffset(months=1),
            periods=output_offset + len(months),
            freq="MS",
        )
    )
    return ForecastOutputWindow(
        output_months=months,
        inference_months=inference_months,
        output_offset=output_offset,
    )
