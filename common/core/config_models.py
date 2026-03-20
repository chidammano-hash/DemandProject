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

from typing import Optional

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


class AlgorithmModelConfig(BaseModel):
    """Validates a single algorithm entry inside ``algorithm_config.yaml``."""

    enabled: bool = True
    model_id: str = ""
    cluster_strategy: str = Field(
        default="per_cluster", pattern=r"^(per_cluster|global)$"
    )
    shap_select: bool = False
    shap_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    shap_top_n: Optional[int] = None
    shap_sample_size: int = Field(default=500, ge=1)
    recursive: bool = False
    tune_inline: bool = False
    params_file: Optional[str] = None

    class Config:
        extra = "allow"  # allow model-specific hyper-params


class AlgorithmConfig(BaseModel):
    """Top-level schema for ``algorithm_config.yaml``."""

    backtest: dict = Field(default_factory=dict)
    algorithms: dict[str, AlgorithmModelConfig] = Field(default_factory=dict)


class PlanningConfig(BaseModel):
    """Validates ``planning_config.yaml``."""

    planning: PlanningInner = Field(default_factory=lambda: PlanningInner())


class PlanningInner(BaseModel):
    """Inner ``planning:`` block of ``planning_config.yaml``."""

    planning_date: Optional[str] = None
    use_system_date: bool = True


# Re-order so PlanningInner is defined before PlanningConfig uses it.
PlanningConfig.model_rebuild()


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


class SafetyStockConfig(BaseModel):
    """Top-level schema for ``safety_stock_config.yaml``."""

    safety_stock: SafetyStockInner = Field(
        default_factory=lambda: SafetyStockInner()
    )


# ---------------------------------------------------------------------------
# Register models
# ---------------------------------------------------------------------------

register_config_model("algorithm_config.yaml", AlgorithmConfig)
register_config_model("planning_config.yaml", PlanningConfig)
register_config_model("safety_stock_config.yaml", SafetyStockConfig)
