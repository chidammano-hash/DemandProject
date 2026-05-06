"""Lightweight clustering feature constants.

Extracted from ``training.py`` so that consumers (e.g. API routers) can import
the constant lists without paying the ~1s cost of pulling in matplotlib,
sklearn, and scipy at module-import time.

Keep this module dependency-free — only stdlib types, no third-party imports.
"""
from __future__ import annotations

# ── Feature groups ──────────────────────────────────────────────────────────
# Core features that drive business-meaningful clusters for tree models.
# Covers volume, trend, seasonality, periodicity, intermittency, and lifecycle.
CORE_FEATURES = [
    # Volume (log-transformed)
    "mean_demand",
    "cv_demand",
    "iqr_demand",
    # Trend (scale-invariant)
    "trend_slope_norm",
    "trend_r2",
    "cagr",
    # Seasonality
    "seasonal_amplitude",
    "seasonal_r2",
    "yoy_correlation",
    # Periodicity
    "periodicity_strength",
    # Intermittency
    "zero_demand_pct",
    "adi",
    # Lifecycle
    "months_available",
    "recency_ratio",
]

# Features that get log1p-transformed (highly skewed, spans orders of magnitude)
LOG_TRANSFORM_FEATURES = [
    "mean_demand",
    "median_demand",
    "std_demand",
    "total_demand",
    "max_demand",
    "iqr_demand",
    "adi",
]

__all__ = ["CORE_FEATURES", "LOG_TRANSFORM_FEATURES"]
