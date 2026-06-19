"""Pure safety-stock formula functions — IPfeature3 / IPfeature11.

These functions are unit-testable in isolation (no DB, no CLI, no config IO).
They were extracted verbatim from ``scripts/inventory/compute_safety_stock.py``
so the formulas can be imported and tested without importing a CLI script.

Formula reference:
    sigma_D_daily  = demand_std_monthly / sqrt(30.44)
    D_avg_daily    = demand_mean_monthly / 30.44

    SS_demand   = Z * sqrt(LT_mean_days * sigma_D_daily^2)
    SS_lt       = Z * D_avg_daily * lt_std_days
    SS_combined = Z * sqrt(LT_mean_days * sigma_D_daily^2 + D_avg_daily^2 * lt_std_days^2)

    ROP = D_avg_daily * LT_mean_days + SS_combined
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import date

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DAYS_PER_MONTH: float = 30.44


# ---------------------------------------------------------------------------
# Individual SS formula functions (expected by unit tests in test_safety_stock.py)
# ---------------------------------------------------------------------------

def compute_ss_demand(
    z_score: float,
    sigma_demand: float,
    lt_mean_days: float,
) -> float:
    """Demand variability component of safety stock.

    SS_demand = Z * sqrt(LT_mean_days * sigma_demand^2)

    Args:
        z_score:       Z-score for the target service level.
        sigma_demand:  Daily demand standard deviation (units/day).
        lt_mean_days:  Mean lead time in days.

    Returns:
        SS_demand in units.
    """
    variance_term = lt_mean_days * (sigma_demand ** 2)
    return z_score * math.sqrt(variance_term) if variance_term > 0 else 0.0


def compute_ss_lt(
    z_score: float,
    avg_daily_demand: float,
    lt_std_days: float,
) -> float:
    """Lead time variability component of safety stock.

    SS_lt = Z * avg_daily_demand * lt_std_days

    Args:
        z_score:           Z-score for the target service level.
        avg_daily_demand:  Mean daily demand (units/day).
        lt_std_days:       Standard deviation of lead time in days.

    Returns:
        SS_lt in units.
    """
    return z_score * avg_daily_demand * lt_std_days


def compute_ss_combined(
    z_score: float,
    sigma_demand: float,
    lt_mean_days: float,
    avg_daily_demand: float,
    lt_std_days: float,
    yield_std_days: float = 0.0,
) -> float:
    """Combined (uncorrelated) safety stock using the full formula.

    SS_combined = Z * sqrt(LT_mean * sigma_D^2 + D_avg^2 * (lt_std^2 + yield_std^2))

    The yield term (Gen-4 SC-2) is added under the uncorrelated-variance
    assumption: supplier-yield variability — measured as stddev of actual vs
    scheduled delivery — adds to the effective lead-time uncertainty.

    Args:
        z_score:           Z-score for the target service level.
        sigma_demand:      Daily demand standard deviation (units/day).
        lt_mean_days:      Mean lead time in days.
        avg_daily_demand:  Mean daily demand (units/day).
        lt_std_days:       Standard deviation of lead time in days.
        yield_std_days:    Supplier-yield std (days) from mv_supplier_po_performance.
                           Default 0 (feature disabled / data missing).

    Returns:
        SS_combined in units.
    """
    demand_var = lt_mean_days * (sigma_demand ** 2)
    lt_var = (avg_daily_demand ** 2) * ((lt_std_days ** 2) + (yield_std_days ** 2))
    return z_score * math.sqrt(demand_var + lt_var)


def compute_reorder_point_periodic(
    ss_combined: float,
    avg_daily_demand: float,
    lt_mean_days: float,
    review_cycle_days: float,
) -> float:
    """Gen-4 SC-2: ROP for periodic-review policies protects (LT + R/2).

    Pure LT protection understates exposure because the next inspection may
    be up to R days away. Industry-standard formula:

        ROP_periodic = D_avg * (LT_mean + R/2) + SS_combined

    Args:
        ss_combined:       Safety stock quantity.
        avg_daily_demand:  Mean daily demand (units/day).
        lt_mean_days:      Mean lead time (days).
        review_cycle_days: Review interval R (days).

    Returns:
        Periodic-review reorder point in units.
    """
    protection_days = lt_mean_days + (review_cycle_days / 2.0)
    return avg_daily_demand * protection_days + ss_combined


def compute_reorder_point(
    ss_combined: float,
    avg_daily_demand: float,
    lt_mean_days: float,
) -> float:
    """Reorder point = cycle demand during lead time + safety stock.

    ROP = avg_daily_demand * lt_mean_days + ss_combined

    Args:
        ss_combined:       Safety stock quantity.
        avg_daily_demand:  Mean daily demand (units/day).
        lt_mean_days:      Mean lead time in days.

    Returns:
        Reorder point in units.
    """
    return avg_daily_demand * lt_mean_days + ss_combined


def compute_ss_coverage(
    current_on_hand: float,
    ss_combined: float,
) -> float | None:
    """Safety stock coverage ratio.

    ss_coverage = current_on_hand / ss_combined

    Returns None when ss_combined = 0 (no safety stock required).
    """
    if ss_combined == 0.0:
        return None
    return current_on_hand / ss_combined


def get_z_score(service_level: float, z_table: dict[str, float]) -> float:
    """Look up the Z-score for a given service level from the config z_table.

    Uses string key lookup to avoid float comparison fragility (YAML keys
    like 0.98 may not round-trip exactly through float conversion).
    Tries an exact match first, then the closest key.
    Falls back to 1.645 (95%) if nothing is found.
    """
    # Exact match via string key
    sl_str = str(service_level)
    if sl_str in z_table:
        return float(z_table[sl_str])
    # Closest key
    if z_table:
        closest = min(z_table.keys(), key=lambda k: abs(float(k) - service_level))
        return float(z_table[closest])
    return 1.645  # hard fallback


def classify_xyz(
    demand_cv: float | None,
    xyz_thresholds: dict[str, float] | None = None,
) -> str | None:
    """Derive XYZ classification from demand coefficient of variation.

    X = stable (CV < x_max), Y = moderate (CV < y_max), Z = volatile (CV >= y_max).
    Returns None when demand_cv is not available.

    Args:
        demand_cv:       Coefficient of variation (std / mean). None if unavailable.
        xyz_thresholds:  Dict with 'x_max' and 'y_max' keys. Defaults to 0.3/0.8.

    Returns:
        'X', 'Y', 'Z', or None.
    """
    if demand_cv is None:
        return None
    x_max = 0.3
    y_max = 0.8
    if xyz_thresholds:
        x_max = float(xyz_thresholds.get("x_max", 0.3))
        y_max = float(xyz_thresholds.get("y_max", 0.8))
    if demand_cv < x_max:
        return "X"
    if demand_cv < y_max:
        return "Y"
    return "Z"


def detect_outliers(
    demand_history: list[float],
    method: str = "mad",
    threshold: float = 3.0,
    max_outlier_pct: float = 0.20,
) -> tuple[float, bool]:
    """Flag demand outliers using Median Absolute Deviation.

    Args:
        demand_history:   List of monthly demand values.
        method:           Detection method ('mad' only for now).
        threshold:        MAD multiplier to flag a month as outlier.
        max_outlier_pct:  Fraction of months that triggers 'volatile' flag.

    Returns:
        Tuple of (outlier_pct, is_volatile).
        outlier_pct is the fraction of months flagged as outliers.
        is_volatile is True when outlier_pct exceeds max_outlier_pct.
    """
    if not demand_history or len(demand_history) < 3:
        return 0.0, False

    arr = np.array(demand_history, dtype=float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))

    if mad == 0.0:
        return 0.0, False  # no variability — no outliers

    outlier_flags = np.abs(arr - median) > threshold * mad
    outlier_pct = float(outlier_flags.sum()) / len(arr)
    return outlier_pct, outlier_pct > max_outlier_pct


def compute_seasonal_factors(
    demand_history: list[tuple[date, float]],
    min_history_months: int = 24,
) -> dict[int, float]:
    """Compute monthly seasonal multipliers from demand history.

    For each calendar month (1-12), compute:
        factor = avg_demand_for_month / overall_avg_demand

    Args:
        demand_history:     List of (date, qty) tuples ordered by date ascending.
        min_history_months: Minimum number of data points required to compute
                            seasonal factors. If fewer, returns all 1.0 (no seasonality).

    Returns:
        dict {month_num: factor} where factor=1.0 means average,
        >1.0 means above-average (peak), <1.0 means below-average (trough).
        If insufficient data, returns all 1.0 (no seasonality).
    """
    _NO_SEASONALITY: dict[int, float] = dict.fromkeys(range(1, 13), 1.0)

    if len(demand_history) < min_history_months:
        return _NO_SEASONALITY.copy()

    monthly_sums: dict[int, list[float]] = defaultdict(list)
    for dt, qty in demand_history:
        monthly_sums[dt.month].append(qty)

    monthly_avgs: dict[int, float] = {}
    for m in range(1, 13):
        vals = monthly_sums.get(m, [])
        monthly_avgs[m] = sum(vals) / len(vals) if vals else 0.0

    overall_avg = sum(monthly_avgs.values()) / 12
    if overall_avg <= 0:
        return _NO_SEASONALITY.copy()

    return {m: monthly_avgs[m] / overall_avg for m in range(1, 13)}


def apply_seasonal_adjustment(
    ss_combined: float,
    seasonal_factor: float,
    dampening: float = 0.5,
) -> float:
    """Apply seasonal adjustment to base safety stock.

    Scales ss_combined by sqrt(seasonal_factor) since safety stock scales
    with the square root of demand. Blends with the base value using dampening
    to prevent extreme swings.

    Args:
        ss_combined:     Base safety stock quantity (post guard-rail).
        seasonal_factor: Monthly demand ratio (>1.0 = peak, <1.0 = trough).
        dampening:       Blend factor (0.0 = no seasonal, 1.0 = full seasonal).
                         Default 0.5 means 50% seasonal-adjusted + 50% base.

    Returns:
        Seasonally adjusted safety stock quantity.
    """
    if seasonal_factor <= 0:
        return ss_combined
    ss_scaled = ss_combined * math.sqrt(seasonal_factor)
    return dampening * ss_scaled + (1.0 - dampening) * ss_combined


def get_service_level(
    abc_vol: str | None,
    service_levels: dict[str, float],
    demand_cv: float | None = None,
    service_level_matrix: dict[str, float] | None = None,
    xyz_thresholds: dict[str, float] | None = None,
    is_peak_season: bool = False,
    is_trough_season: bool = False,
    intermittency_ratio: float = 0.0,
    adjustments: dict[str, float] | None = None,
) -> tuple[float, str | None, str | None, str | None]:
    """Return (service_level, xyz_class, abc_xyz_segment, sl_adjustment_reason) with dynamic adjustments.

    When service_level_matrix is provided and demand_cv is available, uses
    the 9-cell ABC x XYZ matrix for differentiated service levels. Otherwise
    falls back to the ABC-only lookup.

    Dynamic adjustments (seasonal peak/trough, intermittency) are applied
    additively on top of the base SL, then clamped to [sl_floor, sl_ceiling].

    Args:
        abc_vol:               ABC volume classification ('A', 'B', 'C', or None).
        service_levels:        ABC-only service level dict (fallback).
        demand_cv:             Demand coefficient of variation. None if unavailable.
        service_level_matrix:  ABC x XYZ matrix dict (e.g. {'AX': 0.99, ...}).
        xyz_thresholds:        XYZ classification thresholds.
        is_peak_season:        True when seasonal_factor > 1.15 for the current month.
        is_trough_season:      True when seasonal_factor < 0.85 for the current month.
        intermittency_ratio:   Fraction of zero-demand months (from dim_sku). 0.0 if unavailable.
        adjustments:           Service level adjustments config dict (from YAML). None = no adjustments.

    Returns:
        Tuple of (service_level, xyz_class, abc_xyz_segment, sl_adjustment_reason).
        xyz_class and abc_xyz_segment are None when matrix lookup is not possible.
        sl_adjustment_reason is None when no dynamic adjustments were applied.
    """
    xyz_class = classify_xyz(demand_cv, xyz_thresholds)

    # Try matrix lookup when both ABC and XYZ are available
    if service_level_matrix and abc_vol and xyz_class:
        segment = f"{abc_vol.upper()}{xyz_class}"
        if segment in service_level_matrix:
            base_sl = float(service_level_matrix[segment])
        else:
            # Segment not in matrix — fall through to ABC-only
            base_sl = None
    else:
        base_sl = None

    if base_sl is None:
        # Fallback to ABC-only lookup
        if abc_vol and abc_vol.upper() in service_levels:
            base_sl = float(service_levels[abc_vol.upper()])
        else:
            base_sl = float(service_levels.get("default", 0.95))

    # Still compute xyz_class/segment for informational purposes even on fallback
    segment = f"{abc_vol.upper()}{xyz_class}" if abc_vol and xyz_class else None

    # -- Apply dynamic adjustments (additive on top of base SL) -----------------
    if not adjustments:
        return base_sl, xyz_class, segment, None

    sl = base_sl
    reasons: list[str] = []

    # Seasonal adjustment
    if is_peak_season:
        boost = adjustments.get("seasonal_peak_boost", 0.0)
        if boost != 0.0:
            sl += boost
            reasons.append(f"seasonal_peak_boost({boost:+.2f})")
    elif is_trough_season:
        relax = adjustments.get("seasonal_trough_relax", 0.0)
        if relax != 0.0:
            sl += relax
            reasons.append(f"seasonal_trough_relax({relax:+.2f})")

    # Intermittency adjustment
    if intermittency_ratio > 0.5:
        relax = adjustments.get("intermittent_relax", 0.0)
        if relax != 0.0:
            sl += relax
            reasons.append(f"intermittent_relax({relax:+.2f})")

    # Clamp to floor/ceiling
    sl_floor = adjustments.get("sl_floor", 0.80)
    sl_ceiling = adjustments.get("sl_ceiling", 0.995)
    sl = max(sl_floor, min(sl, sl_ceiling))

    reason_str = ", ".join(reasons) if reasons else None
    return sl, xyz_class, segment, reason_str


def compute_ss_components(
    z: float,
    demand_mean_monthly: float,
    demand_std_monthly: float,
    lt_mean_days: float,
    lt_std_days: float,
    yield_std_days: float = 0.0,
) -> dict[str, float | None]:
    """Compute safety stock components.

    Args:
        z:                    Z-score for the target service level.
        demand_mean_monthly:  Mean monthly demand (units).
        demand_std_monthly:   Std dev of monthly demand (units).
        lt_mean_days:         Mean lead time in days.
        lt_std_days:          Std dev of lead time in days.
        yield_std_days:       Gen-4 SC-2: supplier-yield std (days) from
                              mv_supplier_po_performance.stddev_lead_time_days.
                              0 when unavailable / disabled. Folded into the
                              lt-variance term under uncorrelated assumption.

    Returns dict with:
        avg_daily_demand, sigma_d_daily,
        ss_demand_only, ss_lt_only, ss_combined, ss_method,
        ss_yield (new)
    """
    avg_daily = demand_mean_monthly / DAYS_PER_MONTH if DAYS_PER_MONTH else 0.0
    sigma_d_daily = demand_std_monthly / math.sqrt(DAYS_PER_MONTH) if demand_std_monthly else 0.0

    demand_variance_term = lt_mean_days * (sigma_d_daily ** 2)
    # Fold supplier-yield variance into the LT-variance term (uncorrelated).
    lt_variance_term = (avg_daily ** 2) * ((lt_std_days ** 2) + (yield_std_days ** 2))

    ss_demand = z * math.sqrt(demand_variance_term) if demand_variance_term >= 0 else 0.0
    ss_lt = z * avg_daily * lt_std_days if lt_std_days else 0.0
    ss_yield = z * avg_daily * yield_std_days if yield_std_days else 0.0
    ss_combined = z * math.sqrt(demand_variance_term + lt_variance_term)

    # Determine method label
    if demand_mean_monthly == 0.0 and demand_std_monthly == 0.0:
        ss_method = "demand_only"  # zero-demand; formula returns 0
    elif lt_std_days == 0.0:
        ss_method = "demand_only"
    else:
        ss_method = "combined"

    return {
        "avg_daily_demand": avg_daily,
        "sigma_d_daily": sigma_d_daily,
        "ss_demand_only": ss_demand,
        "ss_lt_only": ss_lt,
        "ss_yield_only": ss_yield,
        "ss_combined": ss_combined,
        "ss_method": ss_method,
    }


def apply_guard_rails(
    ss_combined: float = 0.0,
    avg_daily_demand: float = 0.0,
    min_ss_days: float = 3.0,
    max_ss_days: float = 120.0,
    # Aliases used by unit tests (ss_days, min_days, max_days)
    ss_days: float | None = None,
    min_days: float | None = None,
    max_days: float | None = None,
    # ABC-aware guard rails (new — optional for backward compat)
    abc_vol: str | None = None,
    guard_rails_config: dict | None = None,
) -> tuple[float, bool, float, float]:
    """Clamp safety stock between min_ss_days and max_ss_days of supply.

    Can be called either with qty (ss_combined) or days (ss_days).
    When ss_days is provided it is converted to qty via ss_days * avg_daily_demand.
    The return value is a tuple of (clamped_qty, was_clamped, min_qty, max_qty).

    When guard_rails_config is provided, uses ABC-specific bounds.
    Falls back to global min_ss_days/max_ss_days when config is absent.

    For zero-demand items with guard_rails_config, applies zero_demand_min_units floor.
    Without config, zero-demand items pass through unchanged.
    """
    # Resolve ABC-specific bounds from config when available
    if guard_rails_config and abc_vol:
        abc_cfg = guard_rails_config.get(
            abc_vol.upper(),
            guard_rails_config.get("default", {}),
        )
        _min = float(abc_cfg.get("min_ss_days", min_ss_days))
        _max = float(abc_cfg.get("max_ss_days", max_ss_days))
    elif guard_rails_config:
        default_cfg = guard_rails_config.get("default", {})
        _min = float(default_cfg.get("min_ss_days", min_ss_days))
        _max = float(default_cfg.get("max_ss_days", max_ss_days))
    else:
        # Backward compat: resolve from explicit params / aliases
        _min = min_days if min_days is not None else min_ss_days
        _max = max_days if max_days is not None else max_ss_days

    _ss_qty: float
    if ss_days is not None:
        _ss_qty = ss_days * avg_daily_demand
    else:
        _ss_qty = ss_combined

    if avg_daily_demand <= 0.0:
        # Zero-demand floor from config
        if guard_rails_config:
            floor = float(guard_rails_config.get("zero_demand_min_units", 0))
            clamped = max(_ss_qty, floor)
            was_clamped = clamped != _ss_qty
            return clamped, was_clamped, floor, floor
        return _ss_qty, False, 0.0, 0.0

    min_qty = _min * avg_daily_demand
    max_qty = _max * avg_daily_demand
    clamped = max(min_qty, min(max_qty, _ss_qty))
    was_clamped = abs(clamped - _ss_qty) > 1e-9
    return clamped, was_clamped, min_qty, max_qty


def compute_position_metrics(
    ss_combined: float,
    avg_daily_demand: float,
    lt_mean_days: float,
    current_qty_on_hand: float,
) -> dict[str, float | bool | None]:
    """Derive ROP, coverage, gap, and is_below_ss from SS output.

    Args:
        ss_combined:         Recommended safety stock quantity.
        avg_daily_demand:    Average daily demand (units/day).
        lt_mean_days:        Mean lead time in days.
        current_qty_on_hand: Latest on-hand quantity.

    Returns dict with reorder_point, target_dos_min, ss_coverage, ss_gap, is_below_ss.
    """
    reorder_point = avg_daily_demand * lt_mean_days + ss_combined

    target_dos_min: float | None
    if avg_daily_demand > 0:
        target_dos_min = ss_combined / avg_daily_demand
        current_dos: float | None = current_qty_on_hand / avg_daily_demand
    else:
        target_dos_min = None
        current_dos = None

    ss_coverage: float | None
    if ss_combined > 0:
        ss_coverage = current_qty_on_hand / ss_combined
    else:
        ss_coverage = None  # no SS required; avoid divide-by-zero

    ss_gap = current_qty_on_hand - ss_combined
    is_below_ss = current_qty_on_hand < ss_combined

    return {
        "reorder_point": reorder_point,
        "target_dos_min": target_dos_min,
        "current_dos": current_dos,
        "ss_coverage": ss_coverage,
        "ss_gap": ss_gap,
        "is_below_ss": is_below_ss,
    }
