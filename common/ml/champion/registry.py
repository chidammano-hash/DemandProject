"""Strategy registry and shared column constants for champion strategies."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from common.core.constants import FORECAST_QTY_COL

# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, Callable[..., pd.DataFrame]] = {}


def register_strategy(name: str):
    """Decorator to register a strategy function."""
    def decorator(fn: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        STRATEGY_REGISTRY[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Shared column constants
# ---------------------------------------------------------------------------

_DFU_COLS = ["item_id", "customer_group", "loc"]
_DFU_MONTH_COLS = ["item_id", "customer_group", "loc", "startdate"]
_DFU_MODEL_COLS = ["item_id", "customer_group", "loc", "model_id"]

_OUTPUT_COLS = [
    "item_id", "customer_group", "loc", "startdate",
    "model_id", "prior_wape", FORECAST_QTY_COL, "tothist_dmd",
    # Per-DFU-month blend composition for blended champions: a list of
    # {"model": <id>, "weight": <float 0-1>} dicts. None for single-model picks.
    # Lets the UI show the champion mix, e.g. "champion (40% NBEATS, 35% LGBM)".
    "source_mix",
]
