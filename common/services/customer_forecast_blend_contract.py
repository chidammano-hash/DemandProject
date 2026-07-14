"""Configuration and deterministic math for customer bottom-up blending."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from common.core.constants import CUSTOMER_BOTTOM_UP_BLEND_MODEL_ID
from common.core.utils import load_forecast_pipeline_config

CUSTOMER_BLEND_LINEAGE_METADATA_KEY = "customer_bottom_up_blend"
CUSTOMER_BLEND_CONTRACT_VERSION = "warehouse-item-sales-normalized-v1"

_QUANTITY_SCALE = Decimal("0.0001")
_WEIGHT_SCALE = Decimal("0.000001")


@dataclass(frozen=True)
class CustomerBlendSettings:
    """Validated configuration that affects the blend payload."""

    enabled: bool
    model_id: str
    customer_weight: Decimal
    champion_weight: Decimal
    normalization_method: str
    normalization_lookback_months: int
    normalization_min_demand_qty: Decimal
    normalization_min_ratio: Decimal
    normalization_max_ratio: Decimal
    output_population: str
    missing_customer_policy: str
    customer_only_policy: str
    interval_method: str
    promotion_enabled: bool
    promotion_reason: str

    def as_lineage(self) -> dict[str, Any]:
        return {
            "contract_version": CUSTOMER_BLEND_CONTRACT_VERSION,
            "model_id": self.model_id,
            "customer_weight": str(self.customer_weight),
            "champion_weight": str(self.champion_weight),
            "normalization": {
                "method": self.normalization_method,
                "lookback_months": self.normalization_lookback_months,
                "min_demand_qty": str(self.normalization_min_demand_qty),
                "min_ratio": str(self.normalization_min_ratio),
                "max_ratio": str(self.normalization_max_ratio),
            },
            "coverage": {
                "output_population": self.output_population,
                "missing_customer": self.missing_customer_policy,
                "customer_only": self.customer_only_policy,
            },
            "interval": {"method": self.interval_method},
            "promotion": {
                "enabled": self.promotion_enabled,
                "reason": self.promotion_reason,
            },
        }


@dataclass(frozen=True)
class BlendResult:
    """One deterministic bottom-up/champion blend calculation."""

    raw_customer_demand_qty: Decimal | None
    normalized_customer_qty: Decimal | None
    champion_qty: Decimal
    blended_qty: Decimal
    lower_bound: Decimal | None
    upper_bound: Decimal | None
    fulfillment_ratio: Decimal | None
    effective_customer_weight: Decimal
    coverage_status: str
    interval_method: str


def _decimal(value: object, *, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (ArithmeticError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not result.is_finite():
        raise ValueError(f"{label} must be finite")
    return result


def validate_blend_settings(raw: dict[str, Any]) -> CustomerBlendSettings:
    """Validate the complete, no-default blend policy from pipeline config."""
    try:
        settings = raw["bottom_up_blend"] if "bottom_up_blend" in raw else raw
        normalization = settings["normalization"]
        coverage = settings["coverage"]
        interval = settings["interval"]
        promotion = settings["promotion"]
        parsed = CustomerBlendSettings(
            enabled=settings["enabled"],
            model_id=str(settings["model_id"]),
            customer_weight=_decimal(settings["customer_weight"], label="customer_weight"),
            champion_weight=_decimal(settings["champion_weight"], label="champion_weight"),
            normalization_method=str(normalization["method"]),
            normalization_lookback_months=int(normalization["lookback_months"]),
            normalization_min_demand_qty=_decimal(
                normalization["min_demand_qty"], label="normalization.min_demand_qty"
            ),
            normalization_min_ratio=_decimal(
                normalization["min_ratio"], label="normalization.min_ratio"
            ),
            normalization_max_ratio=_decimal(
                normalization["max_ratio"], label="normalization.max_ratio"
            ),
            output_population=str(coverage["output_population"]),
            missing_customer_policy=str(coverage["missing_customer"]),
            customer_only_policy=str(coverage["customer_only"]),
            interval_method=str(interval["method"]),
            promotion_enabled=promotion["enabled"],
            promotion_reason=str(promotion["reason"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Customer bottom-up blend settings are incomplete") from exc

    if not isinstance(parsed.enabled, bool) or not isinstance(parsed.promotion_enabled, bool):
        raise ValueError("Customer bottom-up blend enablement values must be boolean")
    if parsed.model_id != CUSTOMER_BOTTOM_UP_BLEND_MODEL_ID:
        raise ValueError("Customer bottom-up blend model_id is invalid")
    if (
        parsed.customer_weight <= 0
        or parsed.champion_weight < 0
        or parsed.customer_weight + parsed.champion_weight != Decimal("1")
    ):
        raise ValueError(
            "Customer weight must be positive, champion weight non-negative, "
            "and weights must sum to 1"
        )
    if any(
        weight != weight.quantize(_WEIGHT_SCALE)
        for weight in (parsed.customer_weight, parsed.champion_weight)
    ):
        raise ValueError("Customer and champion blend weights must use at most 6 decimal places")
    if (
        parsed.normalization_method != "historical_fulfillment_ratio"
        or parsed.normalization_lookback_months <= 0
        or parsed.normalization_min_demand_qty <= 0
        or parsed.normalization_min_ratio < 0
        or parsed.normalization_max_ratio > 1
        or parsed.normalization_min_ratio > parsed.normalization_max_ratio
    ):
        raise ValueError("Customer bottom-up target normalization settings are invalid")
    if (
        parsed.output_population != "active_champion"
        or parsed.missing_customer_policy != "champion_fallback"
        or parsed.customer_only_policy != "exclude"
    ):
        raise ValueError("Customer bottom-up coverage policy is invalid")
    if parsed.interval_method != "champion_width_shift":
        raise ValueError("Customer bottom-up interval policy is invalid")
    if not parsed.promotion_reason.strip():
        raise ValueError("Customer bottom-up promotion policy needs a reason")
    return parsed


def get_customer_blend_settings() -> CustomerBlendSettings:
    config = load_forecast_pipeline_config()
    try:
        customer = config["customer_forecast"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Customer forecast configuration is unavailable") from exc
    if not isinstance(customer, dict):
        raise ValueError("Customer forecast configuration is invalid")
    return validate_blend_settings(customer)


def customer_blend_config_checksum(settings: CustomerBlendSettings) -> str:
    payload = json.dumps(settings.as_lineage(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def normalize_customer_quantity(
    customer_demand_qty: Decimal | int | float | str,
    fulfillment_ratio: Decimal | int | float | str,
) -> Decimal:
    """Convert ordered customer demand to the shipped-sales production target."""
    quantity = _decimal(customer_demand_qty, label="customer_demand_qty")
    ratio = _decimal(fulfillment_ratio, label="fulfillment_ratio")
    if quantity < 0 or ratio < 0 or ratio > 1:
        raise ValueError("Customer quantity and fulfillment ratio must be within bounds")
    return (quantity * ratio).quantize(_QUANTITY_SCALE, rounding=ROUND_HALF_UP)


def blend_customer_and_champion(
    *,
    customer_demand_qty: Decimal | int | float | str | None,
    fulfillment_ratio: Decimal | int | float | str | None,
    champion_qty: Decimal | int | float | str,
    champion_lower: Decimal | int | float | str | None,
    champion_upper: Decimal | int | float | str | None,
    customer_weight: Decimal | int | float | str,
    champion_weight: Decimal | int | float | str,
) -> BlendResult:
    """Blend one row, falling back to champion when bottom-up is unusable."""
    top = _decimal(champion_qty, label="champion_qty")
    customer_share = _decimal(customer_weight, label="customer_weight")
    champion_share = _decimal(champion_weight, label="champion_weight")
    if top < 0 or customer_share < 0 or champion_share < 0:
        raise ValueError("Forecast quantities and weights must be non-negative")
    if customer_share + champion_share != Decimal("1"):
        raise ValueError("Customer and champion blend weights must sum to 1")

    raw_customer: Decimal | None = None
    ratio: Decimal | None = None
    normalized: Decimal | None = None
    if customer_demand_qty is not None:
        raw_customer = _decimal(customer_demand_qty, label="customer_demand_qty")
    if fulfillment_ratio is not None:
        ratio = _decimal(fulfillment_ratio, label="fulfillment_ratio")
    if raw_customer is not None and ratio is not None:
        normalized = normalize_customer_quantity(raw_customer, ratio)

    if normalized is None:
        blended = top.quantize(_QUANTITY_SCALE, rounding=ROUND_HALF_UP)
        coverage_status = "champion_fallback"
        effective_customer_weight = Decimal("0")
    else:
        blended = (customer_share * normalized + champion_share * top).quantize(
            _QUANTITY_SCALE, rounding=ROUND_HALF_UP
        )
        coverage_status = "blended"
        effective_customer_weight = customer_share

    lower = _decimal(champion_lower, label="champion_lower") if champion_lower is not None else None
    upper = _decimal(champion_upper, label="champion_upper") if champion_upper is not None else None
    if lower is not None and (lower < 0 or lower > top):
        raise ValueError("Champion lower bound is invalid")
    if upper is not None and upper < top:
        raise ValueError("Champion upper bound is invalid")
    if lower is not None and upper is not None:
        shifted_lower = max(Decimal("0"), blended - (top - lower)).quantize(
            _QUANTITY_SCALE, rounding=ROUND_HALF_UP
        )
        shifted_upper = (blended + (upper - top)).quantize(_QUANTITY_SCALE, rounding=ROUND_HALF_UP)
        interval_method = (
            "champion_width_shift" if normalized is not None else "champion_passthrough"
        )
    else:
        shifted_lower = None
        shifted_upper = None
        interval_method = "none"

    return BlendResult(
        raw_customer_demand_qty=raw_customer,
        normalized_customer_qty=normalized,
        champion_qty=top,
        blended_qty=blended,
        lower_bound=shifted_lower,
        upper_bound=shifted_upper,
        fulfillment_ratio=ratio,
        effective_customer_weight=effective_customer_weight,
        coverage_status=coverage_status,
        interval_method=interval_method,
    )
