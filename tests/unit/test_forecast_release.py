"""Unit tests for forecast release quality gates."""

from common.services.forecast_release import (
    ReleaseQualityMetrics,
    ReleaseReadinessThresholds,
    evaluate_quality_checks,
)


def _thresholds() -> ReleaseReadinessThresholds:
    return ReleaseReadinessThresholds(
        min_relative_wape_lift_vs_naive_pct=10.0,
        min_accuracy_delta_vs_external_pct_points=0.0,
        max_abs_bias_pct=5.0,
        min_current_plan_coverage_frac=0.95,
        min_common_cohort_coverage_frac=0.95,
        min_common_cohort_closed_months=6,
        min_common_cohort_dfus=1_000,
        min_common_cohort_actual_volume=1.0,
    )


def test_quality_checks_pass_on_one_common_cohort() -> None:
    metrics = ReleaseQualityMetrics(
        dfu_months=1_200,
        dfus=1_000,
        champion_observations=1_200,
        champion_dfus=1_000,
        closed_months=6,
        actual_volume=50_000.0,
        champion_wape_pct=18.0,
        champion_bias_pct=-2.0,
        naive_wape_pct=24.0,
        external_wape_pct=19.0,
    )

    checks = evaluate_quality_checks(metrics, _thresholds())

    assert {check["id"] for check in checks} == {
        "common_cohort",
        "common_cohort_coverage",
        "common_cohort_months",
        "common_cohort_dfus",
        "common_cohort_actual_volume",
        "lift_vs_naive",
        "delta_vs_external",
        "champion_bias",
    }
    assert all(check["status"] == "pass" for check in checks)
    lift = next(check for check in checks if check["id"] == "lift_vs_naive")
    assert lift["value"] == 25.0


def test_quality_checks_block_regression_and_excess_bias() -> None:
    metrics = ReleaseQualityMetrics(
        dfu_months=45_812,
        dfus=12_023,
        champion_observations=55_000,
        champion_dfus=13_000,
        closed_months=6,
        actual_volume=500_000.0,
        champion_wape_pct=29.38,
        champion_bias_pct=-6.2,
        naive_wape_pct=33.56,
        external_wape_pct=26.50,
    )

    checks = evaluate_quality_checks(metrics, _thresholds())

    by_id = {check["id"]: check for check in checks}
    assert by_id["lift_vs_naive"]["status"] == "pass"
    assert by_id["common_cohort_coverage"]["status"] == "block"
    assert by_id["delta_vs_external"]["status"] == "block"
    assert "trails" in by_id["delta_vs_external"]["message"]
    assert by_id["champion_bias"]["status"] == "block"


def test_quality_checks_block_when_common_cohort_is_empty() -> None:
    metrics = ReleaseQualityMetrics(
        dfu_months=0,
        dfus=0,
        champion_observations=0,
        champion_dfus=0,
        closed_months=0,
        actual_volume=0.0,
        champion_wape_pct=None,
        champion_bias_pct=None,
        naive_wape_pct=None,
        external_wape_pct=None,
    )

    checks = evaluate_quality_checks(metrics, _thresholds())

    assert all(check["status"] == "block" for check in checks)
    assert checks[0]["id"] == "common_cohort"


def test_quality_checks_block_an_unrepresentative_sample() -> None:
    metrics = ReleaseQualityMetrics(
        dfu_months=120,
        dfus=80,
        champion_observations=120,
        champion_dfus=80,
        closed_months=2,
        actual_volume=0.0,
        champion_wape_pct=5.0,
        champion_bias_pct=0.0,
        naive_wape_pct=20.0,
        external_wape_pct=10.0,
    )

    checks = evaluate_quality_checks(metrics, _thresholds())
    by_id = {check["id"]: check for check in checks}

    assert by_id["common_cohort_months"]["status"] == "block"
    assert by_id["common_cohort_dfus"]["status"] == "block"
    assert by_id["common_cohort_actual_volume"]["status"] == "block"
