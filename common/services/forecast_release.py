"""Pure quality calculations for forecast release readiness.

The API owns the SQL that creates the exact common DFU-month cohort. Keeping
the policy math here makes the planner-readiness decision independently
testable and prevents downstream surfaces from drifting to different formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReleaseReadinessThresholds:
    """Provisional thresholds configured under ``champion.release_readiness``."""

    min_relative_wape_lift_vs_naive_pct: float
    min_accuracy_delta_vs_external_pct_points: float
    max_abs_bias_pct: float
    min_current_plan_coverage_frac: float
    min_common_cohort_coverage_frac: float
    min_common_cohort_closed_months: int
    min_common_cohort_dfus: int
    min_common_cohort_actual_volume: float


@dataclass(frozen=True)
class ReleaseQualityMetrics:
    """Quality metrics computed on one exact champion/naive/external cohort."""

    dfu_months: int
    dfus: int
    champion_observations: int
    champion_dfus: int
    closed_months: int
    actual_volume: float
    champion_wape_pct: float | None
    champion_bias_pct: float | None
    naive_wape_pct: float | None
    external_wape_pct: float | None

    @property
    def common_observation_coverage_frac(self) -> float | None:
        if self.champion_observations <= 0:
            return None
        return self.dfu_months / self.champion_observations

    @property
    def common_dfu_coverage_frac(self) -> float | None:
        if self.champion_dfus <= 0:
            return None
        return self.dfus / self.champion_dfus

    @property
    def relative_wape_lift_vs_naive_pct(self) -> float | None:
        if (
            self.champion_wape_pct is None
            or self.naive_wape_pct is None
            or self.naive_wape_pct <= 0
        ):
            return None
        return 100.0 * (self.naive_wape_pct - self.champion_wape_pct) / self.naive_wape_pct

    @property
    def accuracy_delta_vs_external_pct_points(self) -> float | None:
        """Accuracy(champion) minus accuracy(external), expressed in points."""
        if self.champion_wape_pct is None or self.external_wape_pct is None:
            return None
        return self.external_wape_pct - self.champion_wape_pct


def _check(
    check_id: str,
    *,
    passed: bool,
    value: float | int | None,
    threshold: float | int | str,
    message: str,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": "pass" if passed else "block",
        "value": round(value, 4) if isinstance(value, float) else value,
        "threshold": threshold,
        "message": message,
    }


def evaluate_quality_checks(
    metrics: ReleaseQualityMetrics,
    thresholds: ReleaseReadinessThresholds,
) -> list[dict[str, Any]]:
    """Evaluate forecast quality on one common DFU-month population."""
    has_common_cohort = (
        metrics.dfu_months > 0
        and metrics.champion_wape_pct is not None
        and metrics.naive_wape_pct is not None
        and metrics.external_wape_pct is not None
        and metrics.champion_bias_pct is not None
    )
    lift = metrics.relative_wape_lift_vs_naive_pct
    external_delta = metrics.accuracy_delta_vs_external_pct_points
    bias = metrics.champion_bias_pct
    cohort_coverage = metrics.common_observation_coverage_frac

    cohort_coverage_passed = (
        has_common_cohort
        and cohort_coverage is not None
        and cohort_coverage >= thresholds.min_common_cohort_coverage_frac
    )
    lift_passed = (
        has_common_cohort
        and lift is not None
        and lift >= thresholds.min_relative_wape_lift_vs_naive_pct
    )
    external_delta_passed = (
        has_common_cohort
        and external_delta is not None
        and external_delta >= thresholds.min_accuracy_delta_vs_external_pct_points
    )
    bias_passed = (
        has_common_cohort and bias is not None and abs(bias) <= thresholds.max_abs_bias_pct
    )
    months_passed = (
        has_common_cohort and metrics.closed_months >= thresholds.min_common_cohort_closed_months
    )
    dfus_passed = has_common_cohort and metrics.dfus >= thresholds.min_common_cohort_dfus
    actual_volume_passed = (
        has_common_cohort and metrics.actual_volume >= thresholds.min_common_cohort_actual_volume
    )

    return [
        _check(
            "common_cohort",
            passed=has_common_cohort,
            value=metrics.dfu_months,
            threshold="> 0 DFU-months",
            message=(
                "Champion, seasonal naive, and external forecasts are measured on "
                "the same DFU-months."
                if has_common_cohort
                else "No measurable champion, seasonal-naive, and external common cohort exists."
            ),
        ),
        _check(
            "common_cohort_coverage",
            passed=cohort_coverage_passed,
            value=cohort_coverage,
            threshold=thresholds.min_common_cohort_coverage_frac,
            message=(
                "Common-cohort observations cover the valid champion population."
                if cohort_coverage_passed
                else "Common-cohort coverage is below the planner-readiness threshold."
            ),
        ),
        _check(
            "common_cohort_months",
            passed=months_passed,
            value=metrics.closed_months,
            threshold=thresholds.min_common_cohort_closed_months,
            message=(
                "The common cohort spans the required closed-month window."
                if months_passed
                else "The common cohort does not span enough closed months."
            ),
        ),
        _check(
            "common_cohort_dfus",
            passed=dfus_passed,
            value=metrics.dfus,
            threshold=thresholds.min_common_cohort_dfus,
            message=(
                "The common cohort contains enough DFUs for portfolio evidence."
                if dfus_passed
                else "The common cohort contains too few DFUs for a release decision."
            ),
        ),
        _check(
            "common_cohort_actual_volume",
            passed=actual_volume_passed,
            value=metrics.actual_volume,
            threshold=thresholds.min_common_cohort_actual_volume,
            message=(
                "The common cohort contains measurable actual demand."
                if actual_volume_passed
                else "The common cohort has insufficient actual demand for stable metrics."
            ),
        ),
        _check(
            "lift_vs_naive",
            passed=lift_passed,
            value=lift,
            threshold=thresholds.min_relative_wape_lift_vs_naive_pct,
            message=(
                "Champion WAPE improvement versus seasonal naive meets policy."
                if lift_passed
                else "Champion WAPE improvement versus seasonal naive is below policy."
            ),
        ),
        _check(
            "delta_vs_external",
            passed=external_delta_passed,
            value=external_delta,
            threshold=thresholds.min_accuracy_delta_vs_external_pct_points,
            message=(
                "Champion accuracy meets or exceeds the external forecast on the common cohort."
                if external_delta_passed
                else "Champion accuracy trails the external forecast on the common cohort."
            ),
        ),
        _check(
            "champion_bias",
            passed=bias_passed,
            value=bias,
            threshold=f"abs <= {thresholds.max_abs_bias_pct}",
            message=(
                "Volume-weighted champion bias is within policy."
                if bias_passed
                else "Volume-weighted champion bias exceeds policy on the common cohort."
            ),
        ),
    ]
