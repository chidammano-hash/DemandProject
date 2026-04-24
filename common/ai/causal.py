"""Causal elasticity fit helpers.

Gen-4 Roadmap Cross-cutting #3. Shared helpers used by
`scripts/ml/fit_elasticity.py` and tests. The heavy script logic lives
in the script; this module keeps the algorithmic core pure and
dependency-light so tests can exercise it without I/O.

Design notes
------------
The initial implementation uses sklearn.LinearRegression on a joined
(sales + external-signal) dataframe. Coefficients from the linear model
are shipped as "elasticities" with p-values computed via the classical
OLS formulas (t-test on coef / SE).

A TODO marker is left for swapping in EconML's `LinearDML` once the
EconML dependency is approved.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ElasticityResult:
    """One fitted coefficient."""
    feature: str
    coef: float
    std_err: float | None
    p_value: float | None
    n_obs: int
    method: str = "linear_regression"


def _erf(x: float) -> float:
    """Abramowitz-Stegun erf approximation (no scipy dependency)."""
    # Constants
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return sign * y


def _two_sided_p_from_t(t_stat: float, df: int) -> float:
    """Approximate two-sided p-value for a t-statistic.

    Uses the normal approximation for df >= 30 (standard rule of thumb);
    below that falls back to the same approximation — good enough for
    the scaffold, which is what this module is.
    """
    if df <= 0 or not math.isfinite(t_stat):
        return float("nan")
    z = abs(t_stat)
    # Two-sided p = 2 * (1 - Phi(|t|)) with Phi via erf.
    return 2.0 * (1.0 - 0.5 * (1.0 + _erf(z / math.sqrt(2.0))))


def fit_linear_elasticities(
    X: Any,
    y: Any,
    feature_names: list[str],
) -> list[ElasticityResult]:
    """Fit OLS, return one ElasticityResult per feature.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    feature_names : list[str]

    Imports numpy + sklearn lazily so the module can be imported without
    them (e.g. during lightweight API startup).
    """
    try:
        import numpy as np
        from sklearn.linear_model import LinearRegression
    except ImportError as exc:
        raise RuntimeError(
            "fit_linear_elasticities requires numpy + scikit-learn; "
            "install them or swap in a different backend."
        ) from exc

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()

    if X_arr.ndim != 2 or X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"shape mismatch: X={X_arr.shape} y={y_arr.shape}")
    if X_arr.shape[1] != len(feature_names):
        raise ValueError(
            f"feature_names length {len(feature_names)} != X cols {X_arr.shape[1]}"
        )

    n, k = X_arr.shape
    if n <= k + 1:
        logger.warning("fit_linear_elasticities: n=%s <= k+1=%s, p-values will be NaN", n, k + 1)

    model = LinearRegression()
    model.fit(X_arr, y_arr)
    coefs = model.coef_
    resid = y_arr - model.predict(X_arr)
    df = max(n - k - 1, 0)
    sse = float((resid ** 2).sum())
    sigma_sq = sse / df if df > 0 else float("nan")

    # Standard errors via (X'X)^-1 sigma^2.
    std_errs: list[float | None] = [None] * k
    p_values: list[float | None] = [None] * k
    try:
        # Augment X with intercept for SE calculation.
        Xi = np.hstack([np.ones((n, 1)), X_arr])
        xtx_inv = np.linalg.pinv(Xi.T @ Xi)
        # coefs positions: 0=intercept, 1..k = feature coefs.
        for i in range(k):
            var_i = sigma_sq * xtx_inv[i + 1, i + 1]
            if math.isfinite(var_i) and var_i > 0:
                se = math.sqrt(var_i)
                std_errs[i] = se
                t_stat = float(coefs[i]) / se if se > 0 else float("nan")
                p_values[i] = _two_sided_p_from_t(t_stat, df)
    except (ValueError, ArithmeticError) as exc:
        logger.warning("SE computation failed: %s (leaving std_err/p_value None)", exc)

    return [
        ElasticityResult(
            feature=feature_names[i],
            coef=float(coefs[i]),
            std_err=std_errs[i],
            p_value=p_values[i],
            n_obs=n,
            method="linear_regression",
        )
        for i in range(k)
    ]


__all__ = [
    "ElasticityResult",
    "fit_linear_elasticities",
]
