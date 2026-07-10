"""Typed response contract for the forecast release readiness router."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

ForecastReleaseCheckId = Literal[
    "readiness_policy",
    "common_cohort",
    "common_cohort_coverage",
    "common_cohort_months",
    "common_cohort_dfus",
    "common_cohort_actual_volume",
    "lift_vs_naive",
    "delta_vs_external",
    "champion_bias",
    "actual_alignment",
    "active_promotion_state",
    "champion_results_lineage",
    "cluster_lineage",
    "cluster_assignments",
    "sales_freshness",
    "tuning_freshness",
    "current_plan_version",
    "current_plan_coverage",
    "release_integrity",
    "outgoing_archive",
]


class ForecastReleaseCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: ForecastReleaseCheckId
    status: Literal["pass", "block"]
    value: Any
    threshold: Any
    message: str


class ForecastReleaseQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lookback_months: int
    first_month: str | None
    last_month: str | None
    dfu_months: int
    dfus: int
    closed_months: int
    actual_volume: float
    champion_observations: int
    champion_dfus: int
    common_observation_coverage_frac: float | None
    common_dfu_coverage_frac: float | None
    champion_wape_pct: float | None
    champion_accuracy_pct: float | None
    champion_bias_pct: float | None
    naive_wape_pct: float | None
    external_wape_pct: float | None
    relative_wape_lift_vs_naive_pct: float | None
    accuracy_delta_vs_external_pct_points: float | None


class ForecastReleaseLineage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_promotion_id: int | None
    active_promotion_count: int
    champion_results_promoted: bool
    results_promoted_at: str | None
    champion_rows_modified_at: str | None
    results_promoted_experiment_count: int
    champion_cluster_experiment_id: int | None
    current_cluster_experiment_id: int | None
    promoted_cluster_experiment_count: int
    matches: bool
    cluster_assignment_count: int
    stale_tuning_profiles: int


class ForecastReleaseFreshness(BaseModel):
    model_config = ConfigDict(extra="forbid")

    release_promoted_at: str | None
    release_generated_at: str | None
    latest_sales_load: str | None
    fresh: bool


class ForecastReleaseCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eligible_dfus: int
    complete_plan_dfus: int
    covered_eligible_dfus: int
    current_plan_rows: int
    coverage_frac: float | None
    forecast_start: str | None
    forecast_end: str | None
    required_end: str
    minimum_history_months: int


class ForecastReleaseIntegrity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_ids: int
    invalid_quantity_rows: int
    missing_source_rows: int
    invalid_interval_rows: int
    confidence_interval_rows: int
    confidence_interval_coverage_frac: float | None
    minimum_confidence_interval_coverage_frac: float
    valid: bool


class ForecastReleaseArchive(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_plan_version: str | None
    outgoing_promotion_id: int | None
    outgoing_plan_version: str | None
    outgoing_promoted_at: str | None
    replacement_at: str | None
    staging_record_month: str | None
    models: int
    roster_rows: int
    champion_roster_rows: int
    contender_ranks: int
    model_lag_pairs: int
    minimum_rows: int
    champion_run_ids: int
    lineage_mismatches: int
    complete: bool


class ForecastReleaseNextAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tab: Literal["integration", "fva", "dataQuality", "clusters", "lgbmTuning"]
    pipeline: str | None
    label: str
    reason: str


class ForecastReleaseReadinessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready: bool
    policy_enabled: bool
    release_version: str
    planning_month: str
    champion_experiment_id: int | None
    quality: ForecastReleaseQuality
    lineage: ForecastReleaseLineage
    freshness: ForecastReleaseFreshness
    coverage: ForecastReleaseCoverage
    release_integrity: ForecastReleaseIntegrity
    archive: ForecastReleaseArchive
    checks: list[ForecastReleaseCheck]
    next_action: ForecastReleaseNextAction | None
