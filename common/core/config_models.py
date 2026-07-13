"""Pydantic validation models for YAML configuration files.

Register models here so that ``load_config()`` in ``common/utils.py``
validates the raw YAML dict before returning it.  Configs without a
registered model are returned as-is (backward compatible).

Usage
-----
To add validation for a new config file:

1. Define a Pydantic ``BaseModel`` subclass below.
2. Call ``register_config_model("my_config.yaml", MyConfigModel)``
   at module level.

The registry is imported by ``common/utils.py`` at validation time.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_config_validators: dict[str, type[BaseModel]] = {}


def register_config_model(name: str, model: type[BaseModel]) -> None:
    """Register a Pydantic model for config-file validation.

    Parameters
    ----------
    name : str
        Config file name with extension, e.g. ``"planning_config.yaml"``.
    model : type[BaseModel]
        Pydantic model class used to validate the top-level YAML dict.
    """
    _config_validators[name] = model


# ---------------------------------------------------------------------------
# Pydantic models for critical configs
# ---------------------------------------------------------------------------


class PlanningConfig(BaseModel):
    """Validates ``planning_config.yaml``."""

    planning: PlanningInner = Field(default_factory=lambda: PlanningInner())


class PlanningInner(BaseModel):
    """Inner ``planning:`` block of ``planning_config.yaml``."""

    planning_date: str | None = None
    use_system_date: bool = True


# Re-order so PlanningInner is defined before PlanningConfig uses it.
PlanningConfig.model_rebuild()


class XyzThresholds(BaseModel):
    """XYZ classification thresholds based on demand CV."""

    x_max: float = Field(default=0.3, ge=0.0, le=1.0)  # CV < x_max → X (stable)
    y_max: float = Field(default=0.8, ge=0.0, le=1.0)  # CV < y_max → Y (moderate), else Z


class OutlierDetectionConfig(BaseModel):
    """Outlier detection settings for demand history."""

    enabled: bool = True                        # Enable MAD-based outlier detection
    method: str = Field(default="mad", pattern=r"^(mad)$")  # Detection method
    threshold: float = Field(default=3.0, ge=1.0)  # MAD multiplier for outlier flag
    max_outlier_pct: float = Field(default=0.20, ge=0.0, le=1.0)  # Pct above which item is volatile
    volatile_sl_boost: float = Field(default=0.02, ge=0.0, le=0.10)  # SL boost for volatile items


class AbcGuardRailBounds(BaseModel):
    """Min/max safety stock bounds in days of supply for an ABC class."""

    min_ss_days: float = Field(default=3, ge=0)   # Minimum SS in days of supply
    max_ss_days: float = Field(default=120, ge=1)  # Maximum SS in days of supply


class GuardRailsConfig(BaseModel):
    """ABC-specific safety stock guard rails with outlier detection."""

    model_config = {"extra": "allow"}  # Allow A, B, C, default as extra keys

    A: AbcGuardRailBounds = Field(default_factory=lambda: AbcGuardRailBounds(min_ss_days=5, max_ss_days=60))
    B: AbcGuardRailBounds = Field(default_factory=lambda: AbcGuardRailBounds(min_ss_days=3, max_ss_days=90))
    C: AbcGuardRailBounds = Field(default_factory=lambda: AbcGuardRailBounds(min_ss_days=1, max_ss_days=120))
    default: AbcGuardRailBounds = Field(default_factory=lambda: AbcGuardRailBounds(min_ss_days=3, max_ss_days=120))
    zero_demand_min_units: float = Field(default=3, ge=0)  # Absolute floor for zero-demand items
    outlier_detection: OutlierDetectionConfig = Field(default_factory=OutlierDetectionConfig)


class ServiceLevelAdjustmentsConfig(BaseModel):
    """Dynamic service level adjustments applied on top of the base ABC-XYZ matrix."""

    seasonal_peak_boost: float = Field(default=0.02, ge=-0.10, le=0.10)  # SL boost during peak months
    seasonal_trough_relax: float = Field(default=-0.01, ge=-0.10, le=0.10)  # SL adjustment during trough months
    intermittent_relax: float = Field(default=-0.02, ge=-0.10, le=0.10)  # SL adjustment for intermittent items
    sl_floor: float = Field(default=0.80, ge=0.50, le=1.0)  # Absolute minimum service level
    sl_ceiling: float = Field(default=0.995, ge=0.80, le=1.0)  # Absolute maximum service level


class SafetyStockInner(BaseModel):
    """Inner ``safety_stock:`` block of ``safety_stock_config.yaml``."""

    default_method: str = Field(
        default="combined", pattern=r"^(combined|demand_only)$"
    )
    policy_version: str = "v1"
    service_levels: dict[str, float] = Field(default_factory=dict)
    z_table: dict[float, float] = Field(default_factory=dict)
    min_ss_days: int = Field(default=3, ge=0)
    max_ss_days: int = Field(default=120, ge=1)
    min_demand_months: int = Field(default=3, ge=1)
    lt_std_fallback_pct: float = Field(default=0.20, ge=0.0, le=1.0)
    use_demand_variability: bool = True
    use_lt_variability: bool = True
    batch_size: int = Field(default=1000, ge=1)
    # ABC x XYZ service level matrix (optional — falls back to ABC-only)
    service_level_matrix: dict[str, float] | None = None
    # XYZ classification thresholds (optional — defaults to x_max=0.3, y_max=0.8)
    xyz_thresholds: XyzThresholds | None = None
    # ABC-specific guard rails with outlier detection (optional — falls back to global min/max)
    guard_rails: GuardRailsConfig | None = None
    # Dynamic service level adjustments (optional — no adjustments when absent)
    service_level_adjustments: ServiceLevelAdjustmentsConfig | None = None


class SafetyStockConfig(BaseModel):
    """Top-level schema for ``safety_stock_config.yaml``."""

    safety_stock: SafetyStockInner = Field(
        default_factory=lambda: SafetyStockInner()
    )


# ---------------------------------------------------------------------------
# Register models
# ---------------------------------------------------------------------------

register_config_model("planning_config.yaml", PlanningConfig)
register_config_model("safety_stock_config.yaml", SafetyStockConfig)
