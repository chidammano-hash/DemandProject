"""SKU feature classifiers — derived classification logic.

Provides pure-function classifiers that turn computed time-series features
into categorical labels (seasonality profile, variability class).

Thresholds are aligned with the existing pipeline defaults in
``config/forecast_domain_config.yaml`` but are passed as parameters so
callers can override from config.
"""

from __future__ import annotations


def classify_seasonality_profile(
    features: dict,
    *,
    threshold_low: float = 0.15,
    threshold_medium: float = 0.35,
    threshold_high: float = 0.70,
    yoy_correlation_gate: float = 0.40,
    seasonal_r2_gate: float = 0.30,
) -> str:
    """Classify a SKU's seasonality profile.

    Uses three signals from ``compute_time_series_features()`` output:
      - ``seasonal_amplitude`` — ratio of (max-min monthly mean) to overall mean
      - ``seasonal_r2`` — OLS seasonal dummy R-squared (STL-lite)
      - ``yoy_correlation`` — mean pairwise year-over-year Pearson correlation

    Classification tiers:
      - ``"strong"`` — amplitude >= high threshold AND confirmation passes
      - ``"moderate"`` — amplitude >= medium threshold AND confirmation passes
      - ``"low"`` — amplitude >= low threshold (no confirmation required)
      - ``"none"`` — below all thresholds

    Confirmation requires at least one of:
      - ``yoy_correlation >= yoy_correlation_gate``
      - ``seasonal_r2 >= seasonal_r2_gate``

    Parameters
    ----------
    features:
        Dict of feature name -> value, as returned by
        ``compute_time_series_features().to_dict()``.
    threshold_low:
        Minimum seasonal_amplitude to qualify as ``"low"`` seasonality.
    threshold_medium:
        Minimum seasonal_amplitude to qualify as ``"moderate"`` (with confirmation).
    threshold_high:
        Minimum seasonal_amplitude to qualify as ``"strong"`` (with confirmation).
    yoy_correlation_gate:
        Year-over-year correlation threshold for confirmation.
    seasonal_r2_gate:
        Seasonal R-squared threshold for confirmation.

    Returns
    -------
    One of ``"none"``, ``"low"``, ``"moderate"``, ``"strong"``.
    """
    amplitude = float(features.get("seasonal_amplitude", 0.0) or 0.0)
    yoy_corr = float(features.get("yoy_correlation", 0.0) or 0.0)
    s_r2 = float(features.get("seasonal_r2", 0.0) or 0.0)

    has_confirmation = (
        yoy_corr >= yoy_correlation_gate
        or s_r2 >= seasonal_r2_gate
    )

    if amplitude >= threshold_high and has_confirmation:
        return "strong"
    if amplitude >= threshold_medium and has_confirmation:
        return "moderate"
    if amplitude >= threshold_low:
        return "low"
    return "none"


def classify_variability_class(
    cv_demand: float,
    zero_demand_pct: float,
    *,
    cv_low: float = 0.30,
    cv_medium: float = 0.80,
    cv_high: float = 1.50,
    intermittency_threshold: float = 0.30,
) -> str:
    """Classify a SKU's demand variability class.

    Uses the coefficient of variation (CV) and the fraction of zero-demand
    months to assign one of four classes.  The logic mirrors
    ``scripts/compute_demand_variability.py`` but operates on pre-computed
    feature values rather than raw time-series data.

    Classification rules:
      - ``"lumpy"`` — intermittency >= threshold OR CV >= high
      - ``"erratic"`` — CV >= medium  (high variability, not intermittent)
      - ``"intermittent"`` — intermittency >= threshold AND CV < medium
      - ``"smooth"`` — CV < low

    The four-class taxonomy follows the Syntetos-Boylan framework:
      - **smooth**: low variability, regular demand
      - **erratic**: high variability but regular timing
      - **intermittent**: low variability but sporadic timing
      - **lumpy**: both high variability and sporadic timing

    Parameters
    ----------
    cv_demand:
        Coefficient of variation of demand (std / mean).
    zero_demand_pct:
        Fraction of months with zero demand (0.0 to 1.0).
    cv_low:
        Upper bound for ``"smooth"`` class.
    cv_medium:
        Upper bound for ``"erratic"`` when below this.
    cv_high:
        CV threshold that forces ``"lumpy"`` regardless of intermittency.
    intermittency_threshold:
        Zero-demand fraction that triggers intermittent/lumpy classification.

    Returns
    -------
    One of ``"smooth"``, ``"erratic"``, ``"intermittent"``, ``"lumpy"``.
    """
    is_intermittent = zero_demand_pct >= intermittency_threshold
    is_high_cv = cv_demand >= cv_high

    # Lumpy: extreme CV or both high variability and intermittent
    if is_high_cv or (is_intermittent and cv_demand >= cv_medium):
        return "lumpy"

    # Erratic: high variability but regular timing
    if cv_demand >= cv_medium:
        return "erratic"

    # Intermittent: sporadic demand but low variability per occurrence
    if is_intermittent:
        return "intermittent"

    # Smooth: low variability, regular demand
    if cv_demand < cv_low:
        return "smooth"

    # Middle ground: moderate CV, regular timing — still erratic
    return "erratic"
