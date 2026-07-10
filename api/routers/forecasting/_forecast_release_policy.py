"""Policy helpers owned by the forecast release readiness router."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any


def as_optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def add_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 + months
    return date(month_index // 12, month_index % 12 + 1, 1)


def is_calendar_plan_version(value: str | None) -> bool:
    """Return whether a plan version is a valid YYYY-MM archive key."""
    return bool(value and re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value))


def build_gate_check(
    check_id: str,
    *,
    passed: bool,
    value: Any,
    threshold: Any,
    message: str,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": "pass" if passed else "block",
        "value": value,
        "threshold": threshold,
        "message": message,
    }


def build_active_promotion_check(
    active_promotion_id: int | None,
    active_promotion_count: int,
    active_plan_version: str | None,
) -> dict[str, Any]:
    """Fail closed when the production control record is absent or ambiguous."""
    valid = (
        active_promotion_count == 1
        and active_promotion_id is not None
        and active_plan_version is not None
    )
    return build_gate_check(
        "active_promotion_state",
        passed=valid,
        value={
            "active_promotion_id": active_promotion_id,
            "active_promotion_count": active_promotion_count,
            "plan_version": active_plan_version,
        },
        threshold="exactly one active promotion with a plan version",
        message=(
            "Exactly one versioned promotion owns the active production release."
            if valid
            else "The active production promotion is missing, ambiguous, or unversioned."
        ),
    )


def build_results_lineage_check(
    champion_experiment_id: int | None,
    champion_results_promoted: bool,
    results_promoted_count: int,
    results_promoted_at: datetime | None,
    champion_rows_modified_at: datetime | None,
) -> dict[str, Any]:
    """Fence explicit results promotion against later champion rewrites."""
    control_matches = (
        champion_experiment_id is not None
        and champion_results_promoted
        and results_promoted_count == 1
        and results_promoted_at is not None
    )
    rows_unchanged = (
        control_matches
        and champion_rows_modified_at is not None
        and champion_rows_modified_at <= results_promoted_at
    )
    return build_gate_check(
        "champion_results_lineage",
        passed=rows_unchanged,
        value={
            "active_champion_experiment_id": champion_experiment_id,
            "results_promoted": champion_results_promoted,
            "results_promoted_experiment_count": results_promoted_count,
            "results_promoted_at": (
                results_promoted_at.isoformat() if results_promoted_at else None
            ),
            "champion_rows_modified_at": (
                champion_rows_modified_at.isoformat() if champion_rows_modified_at else None
            ),
        },
        threshold="sole explicit results promotion newer than every champion row",
        message=(
            "Explicit results promotion is unique and no champion rows were rewritten later."
            if rows_unchanged
            else (
                "Champion quality rows were rewritten after explicit results promotion."
                if control_matches and champion_rows_modified_at is not None
                else "The active release does not match one explicit results promotion."
            )
        ),
    )


def build_cluster_lineage_check(
    champion_cluster_id: int | None,
    current_cluster_id: int | None,
    promoted_cluster_count: int,
    required: bool,
) -> tuple[dict[str, Any], bool]:
    """Require one unambiguous promoted clustering generation."""
    matches = (
        champion_cluster_id is not None
        and current_cluster_id is not None
        and promoted_cluster_count == 1
        and champion_cluster_id == current_cluster_id
    )
    check = build_gate_check(
        "cluster_lineage",
        passed=matches or not required,
        value={
            "champion_cluster_experiment_id": champion_cluster_id,
            "current_cluster_experiment_id": current_cluster_id,
            "promoted_cluster_experiment_count": promoted_cluster_count,
        },
        threshold=("one matching non-null experiment" if required else "not required"),
        message=(
            "Cluster lineage is not required by policy."
            if not required
            else (
                "The active release matches the sole promoted cluster experiment."
                if matches
                else "Champion cluster lineage is missing, ambiguous, or not current."
            )
        ),
    )
    return check, matches


def next_release_action(
    checks: list[dict[str, Any]],
) -> dict[str, str | None] | None:
    """Choose one honest, non-executing destination for the highest-risk blocker."""
    blocked = {check["id"] for check in checks if check["status"] == "block"}
    if "readiness_policy" in blocked:
        return None
    if blocked & {"active_promotion_state", "champion_results_lineage"}:
        return {
            "tab": "lgbmTuning",
            "pipeline": None,
            "label": "Review forecast promotion",
            "reason": "Production or champion-results lineage is missing, ambiguous, or stale.",
        }
    if blocked & {
        "common_cohort",
        "common_cohort_coverage",
        "common_cohort_months",
        "common_cohort_dfus",
        "common_cohort_actual_volume",
        "actual_alignment",
    }:
        return {
            "tab": "dataQuality",
            "pipeline": None,
            "label": "Review forecast data quality",
            "reason": "The comparison cohort is incomplete or internally inconsistent.",
        }
    if blocked & {"cluster_lineage", "cluster_assignments"}:
        return {
            "tab": "clusters",
            "pipeline": None,
            "label": "Review promoted clusters",
            "reason": "Cluster lineage or promoted assignments are incomplete.",
        }
    if blocked & {
        "lift_vs_naive",
        "delta_vs_external",
        "champion_bias",
        "tuning_freshness",
    }:
        return {
            "tab": "integration",
            "pipeline": "model-refresh",
            "label": "Open Workflows for model refresh",
            "reason": "Forecast evidence or model lineage is stale or below policy.",
        }
    if "sales_freshness" in blocked:
        return {
            "tab": "integration",
            "pipeline": "forecast-publish",
            "label": "Open Workflows for forecast publish",
            "reason": "The active release must be regenerated after the latest sales load.",
        }
    if "outgoing_archive" in blocked:
        return {
            "tab": "fva",
            "pipeline": None,
            "label": "Review archive evidence",
            "reason": "The outgoing plan lacks complete pre-replacement archive evidence.",
        }
    if blocked & {"current_plan_version", "current_plan_coverage", "release_integrity"}:
        return {
            "tab": "integration",
            "pipeline": "forecast-publish",
            "label": "Open Workflows for forecast publish",
            "reason": "No complete champion plan is available for the planning month.",
        }
    return None
