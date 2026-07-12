"""Planner-facing forecast release readiness contract.

The endpoint intentionally evaluates the current planning month as one release
version.  Quality comparisons use an exact common DFU-month cohort, while
lineage, data freshness, production coverage, and outgoing archive checks make
the operational handoff explicit instead of presenting disconnected metrics.
"""

from __future__ import annotations

import logging
from typing import Any

import psycopg
from fastapi import APIRouter, HTTPException

from api.core import get_read_only_conn
from api.routers.forecasting._forecast_release_models import (
    ForecastReleaseReadinessResponse,
)
from api.routers.forecasting._forecast_release_policy import (
    add_months,
    as_optional_float,
    build_active_promotion_check,
    build_cluster_lineage_check,
    build_gate_check,
    build_results_lineage_check,
    is_calendar_plan_version,
    next_release_action,
)
from common.core.planning_date import get_planning_date
from common.core.utils import load_forecast_pipeline_config
from common.services.cache import cached_sync
from common.services.forecast_release import (
    ReleaseQualityMetrics,
    ReleaseReadinessThresholds,
    evaluate_quality_checks,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forecast-release", tags=["forecast-release"])


@router.get("/readiness", response_model=ForecastReleaseReadinessResponse)
@cached_sync(ttl=60, group="forecast_release")
def get_forecast_release_readiness() -> dict[str, Any]:
    """Return whether the active release meets the planner evidence policy."""
    config = load_forecast_pipeline_config()
    gate_config = config["champion"]["release_readiness"]
    lookback_months = int(gate_config["lookback_months"])
    lag_count = int(config["forecast_snapshot"]["lag_count"])
    contender_count = int(config["forecast_snapshot"]["contender_count"])
    archive_model_count = contender_count + 1
    active_window_months = int(config["forecast_snapshot"]["active_window_months"])
    cold_start_min_months = int(config["production_forecast"]["cold_start_min_months"])
    thresholds = ReleaseReadinessThresholds(
        min_relative_wape_lift_vs_naive_pct=float(
            gate_config["min_relative_wape_lift_vs_naive_pct"]
        ),
        min_accuracy_delta_vs_external_pct_points=float(
            gate_config["min_accuracy_delta_vs_external_pct_points"]
        ),
        max_abs_bias_pct=float(gate_config["max_abs_bias_pct"]),
        min_current_plan_coverage_frac=float(gate_config["min_current_plan_coverage_frac"]),
        min_common_cohort_coverage_frac=float(gate_config["min_common_cohort_coverage_frac"]),
        min_common_cohort_closed_months=int(gate_config["min_common_cohort_closed_months"]),
        min_common_cohort_dfus=int(gate_config["min_common_cohort_dfus"]),
        min_common_cohort_actual_volume=float(gate_config["min_common_cohort_actual_volume"]),
    )
    min_ci_coverage_frac = float(gate_config["min_confidence_interval_coverage_frac"])

    planning_month = get_planning_date().replace(day=1)
    release_version = planning_month.strftime("%Y-%m")
    release_end = add_months(planning_month, lag_count - 1)

    # The aggregate drops customer_group/lag; score the execution-lag DFU grain here.
    quality_sql = """WITH active_champion AS (
                         SELECT champion_experiment_id
                         FROM model_promotion_log
                         WHERE is_active = TRUE
                         ORDER BY promoted_at DESC, id DESC
                         LIMIT 1
                     ), champion_keys AS (
                         SELECT f.item_id, f.customer_group, f.loc, f.startdate,
                                f.lag, f.tothist_dmd::numeric AS actual_qty
                         FROM fact_external_forecast_monthly f
                         JOIN dim_sku d
                           ON d.item_id = f.item_id
                          AND d.customer_group = f.customer_group
                          AND d.loc = f.loc
                         JOIN active_champion active
                           ON active.champion_experiment_id = f.champion_experiment_id
                         WHERE f.model_id = 'champion'
                           AND f.lag = COALESCE(d.execution_lag, 0)
                           AND f.lag BETWEEN 0 AND 4
                           AND f.basefcst_pref IS NOT NULL
                           AND f.tothist_dmd IS NOT NULL
                           AND f.startdate >= %s::date - (%s * INTERVAL '1 month')
                           AND f.startdate < %s::date
                     ), required_prior_months AS (
                         SELECT DISTINCT startdate - INTERVAL '12 months' AS startdate
                         FROM champion_keys
                     ), sales_by_dfu AS (
                         SELECT sales.item_id, sales.customer_group, sales.loc,
                                sales.startdate, SUM(sales.qty)::numeric AS qty
                         FROM fact_sales_monthly sales
                         JOIN required_prior_months required
                           ON required.startdate = sales.startdate
                         WHERE sales.type = 1
                         GROUP BY 1, 2, 3, 4
                     ), scored AS (
                         SELECT f.item_id, f.customer_group, f.loc, f.startdate,
                                f.lag, f.model_id,
                                f.basefcst_pref::numeric AS forecast_qty,
                                f.tothist_dmd::numeric AS actual_qty
                         FROM fact_external_forecast_monthly f
                         JOIN dim_sku d
                           ON d.item_id = f.item_id
                          AND d.customer_group = f.customer_group
                          AND d.loc = f.loc
                         WHERE f.model_id IN ('champion', 'external')
                           AND (
                               f.model_id <> 'champion'
                               OR f.champion_experiment_id =
                                  (SELECT champion_experiment_id FROM active_champion)
                           )
                           AND f.lag = COALESCE(d.execution_lag, 0)
                           AND f.lag BETWEEN 0 AND 4
                           AND f.basefcst_pref IS NOT NULL
                           AND f.tothist_dmd IS NOT NULL
                           AND f.startdate >= %s::date - (%s * INTERVAL '1 month')
                           AND f.startdate < %s::date
                         UNION ALL
                         SELECT keys.item_id, keys.customer_group, keys.loc, keys.startdate,
                                keys.lag, 'seasonal_naive' AS model_id,
                                COALESCE(prior.qty, 0)::numeric AS forecast_qty,
                                keys.actual_qty
                         FROM champion_keys keys
                         LEFT JOIN sales_by_dfu prior
                           ON prior.item_id = keys.item_id
                          AND prior.customer_group = keys.customer_group
                          AND prior.loc = keys.loc
                          AND prior.startdate = keys.startdate - INTERVAL '12 months'
                     ), key_quality AS (
                         SELECT item_id, customer_group, loc, startdate, lag,
                                COUNT(DISTINCT model_id) AS model_count,
                                MIN(actual_qty) AS min_actual,
                                MAX(actual_qty) AS max_actual
                         FROM scored
                         GROUP BY item_id, customer_group, loc, startdate, lag
                         HAVING COUNT(*) = 3 AND COUNT(DISTINCT model_id) = 3
                     ), common_keys AS (
                         SELECT item_id, customer_group, loc, startdate, lag
                         FROM key_quality
                         WHERE min_actual = max_actual
                     ), champion_population AS (
                         SELECT COUNT(*)::bigint AS observations,
                                COUNT(DISTINCT (item_id, customer_group, loc))::bigint AS dfus
                         FROM scored
                         WHERE model_id = 'champion'
                     ), model_metrics AS (
                         SELECT s.model_id,
                                COUNT(*)::bigint AS observations,
                                COUNT(DISTINCT (s.item_id, s.customer_group, s.loc))::bigint AS dfus,
                                COUNT(DISTINCT s.startdate)::integer AS closed_months,
                                MIN(s.startdate) AS first_month,
                                MAX(s.startdate) AS last_month,
                                SUM(s.actual_qty)::numeric AS actual_volume,
                                100.0 * SUM(ABS(s.forecast_qty - s.actual_qty))
                                    / NULLIF(ABS(SUM(s.actual_qty)), 0) AS wape_pct,
                                100.0 * (SUM(s.forecast_qty)
                                    / NULLIF(SUM(s.actual_qty), 0) - 1.0) AS bias_pct
                         FROM scored s
                         JOIN common_keys c
                           USING (item_id, customer_group, loc, startdate, lag)
                         GROUP BY s.model_id
                     )
                     SELECT
                         MAX(observations) FILTER (WHERE model_id = 'champion'),
                         MAX(dfus) FILTER (WHERE model_id = 'champion'),
                         MAX(wape_pct) FILTER (WHERE model_id = 'champion'),
                         MAX(bias_pct) FILTER (WHERE model_id = 'champion'),
                         MAX(wape_pct) FILTER (WHERE model_id = 'seasonal_naive'),
                         MAX(wape_pct) FILTER (WHERE model_id = 'external'),
                         MAX(first_month) FILTER (WHERE model_id = 'champion'),
                         MAX(last_month) FILTER (WHERE model_id = 'champion'),
                         (SELECT COUNT(*) FROM key_quality
                          WHERE min_actual IS DISTINCT FROM max_actual),
                         (SELECT observations FROM champion_population),
                         (SELECT dfus FROM champion_population),
                         MAX(closed_months) FILTER (WHERE model_id = 'champion'),
                         MAX(actual_volume) FILTER (WHERE model_id = 'champion')
                     FROM model_metrics"""

    state_sql = """WITH champion_lineage AS (
                     SELECT experiment_id, cluster_experiment_id, is_results_promoted,
                            results_promoted_at
                         FROM champion_experiment
                     ), promoted_cluster AS (
                     SELECT experiment_id, COUNT(*) OVER ()::integer AS promoted_count
                         FROM cluster_experiment
                         WHERE is_promoted = TRUE
                         ORDER BY promoted_at DESC
                         LIMIT 1
                     ), promotion_state AS (
                         SELECT COUNT(*)::integer AS active_count
                         FROM model_promotion_log
                         WHERE is_active = TRUE
                     ), active_promotion AS (
                         SELECT id, plan_version, promoted_at, champion_experiment_id
                         FROM model_promotion_log
                         WHERE is_active = TRUE
                         ORDER BY promoted_at DESC, id DESC
                         LIMIT 1
                     ), outgoing_promotion AS (
                         SELECT previous.id, previous.plan_version, previous.promoted_at
                         FROM model_promotion_log previous
                         CROSS JOIN active_promotion active
                         WHERE (previous.promoted_at, previous.id)
                               < (active.promoted_at, active.id)
                         ORDER BY previous.promoted_at DESC, previous.id DESC
                         LIMIT 1
                     )
                     SELECT active.id,
                            state.active_count,
                            active.champion_experiment_id,
                            champion.cluster_experiment_id,
                            champion.is_results_promoted,
                            champion.results_promoted_at,
                            (SELECT COUNT(*) FROM champion_experiment
                             WHERE is_results_promoted = TRUE),
                            (SELECT MAX(modified_ts)
                             FROM fact_external_forecast_monthly
                             WHERE model_id = 'champion'),
                            cluster.experiment_id,
                            COALESCE(cluster.promoted_count, 0),
                            (SELECT COUNT(*) FROM current_sku_cluster_assignment),
                            (SELECT MAX(completed_at) FROM audit_load_batch
                             WHERE domain = 'sales' AND status = 'completed'),
                            (SELECT COUNT(*) FROM cluster_tuning_profile_state tuning
                             WHERE tuning.stale = TRUE
                               AND EXISTS (
                                   SELECT 1 FROM current_sku_cluster_assignment assignment
                                   WHERE assignment.ml_cluster = tuning.cluster_name
                               )),
                            active.plan_version,
                            active.promoted_at,
                            (SELECT MIN(p.generated_at)
                             FROM fact_production_forecast p
                             WHERE p.model_id = 'champion'
                               AND p.plan_version = active.plan_version),
                            (SELECT MAX(forecast_month_generated)
                             FROM forecast_generation_run
                             WHERE generation_purpose = 'release_candidate'
                               AND run_status IN ('ready', 'promoted')),
                            outgoing.id,
                            outgoing.plan_version,
                            outgoing.promoted_at
                     FROM promotion_state state
                     LEFT JOIN active_promotion active ON TRUE
                     LEFT JOIN outgoing_promotion outgoing ON TRUE
                     LEFT JOIN champion_lineage champion
                       ON champion.experiment_id = active.champion_experiment_id
                     LEFT JOIN promoted_cluster cluster ON TRUE"""

    coverage_sql = """WITH eligible AS (
                            SELECT item_id, loc
                            FROM fact_sales_monthly
                            WHERE type = 1 AND startdate < %s
                            GROUP BY item_id, loc
                            HAVING COUNT(DISTINCT startdate) >= %s
                               AND MAX(startdate) >= %s::date - (%s * INTERVAL '1 month')
                        ), release_rows AS (
                            SELECT item_id, loc, forecast_month, forecast_qty,
                                   forecast_qty_lower, forecast_qty_upper,
                                   source_model_id, run_id
                            FROM fact_production_forecast
                            WHERE model_id = 'champion'
                              AND plan_version = %s
                              AND forecast_month >= %s
                              AND forecast_month < %s::date + (%s * INTERVAL '1 month')
                        ), complete_plan AS (
                            SELECT item_id, loc
                            FROM release_rows
                            GROUP BY item_id, loc
                            HAVING COUNT(DISTINCT forecast_month) = %s
                        ), release_integrity AS (
                            SELECT COUNT(DISTINCT run_id) AS run_ids,
                                   COUNT(*) FILTER (
                                       WHERE forecast_qty < 0
                                          OR forecast_qty_lower < 0
                                          OR forecast_qty_upper < 0
                                   ) AS invalid_quantity_rows,
                                   COUNT(*) FILTER (
                                       WHERE source_model_id IS NULL
                                   ) AS missing_source_rows,
                                   COUNT(*) FILTER (
                                       WHERE (forecast_qty_lower IS NULL)
                                             <> (forecast_qty_upper IS NULL)
                                          OR forecast_qty_lower > forecast_qty
                                          OR forecast_qty_upper < forecast_qty
                                   ) AS invalid_interval_rows,
                                   COUNT(*) FILTER (
                                       WHERE forecast_qty_lower IS NOT NULL
                                         AND forecast_qty_upper IS NOT NULL
                                   ) AS interval_rows
                            FROM release_rows
                        )
                        SELECT (SELECT COUNT(*) FROM eligible),
                               (SELECT COUNT(*) FROM complete_plan),
                               (SELECT COUNT(*) FROM eligible e
                                JOIN complete_plan p USING (item_id, loc)),
                               (SELECT COUNT(*) FROM release_rows),
                               (SELECT MIN(forecast_month) FROM release_rows),
                               (SELECT MAX(forecast_month) FROM release_rows),
                               (SELECT run_ids FROM release_integrity),
                               (SELECT invalid_quantity_rows FROM release_integrity),
                               (SELECT missing_source_rows FROM release_integrity),
                               (SELECT invalid_interval_rows FROM release_integrity),
                               (SELECT interval_rows FROM release_integrity)"""

    archive_sql = """WITH target AS (
                           SELECT TO_DATE(%s || '-01', 'YYYY-MM-DD') AS record_month,
                                  %s::timestamptz AS outgoing_promoted_at,
                                  %s::timestamptz AS replacement_at
                       ), roster AS (
                           SELECT r.model_id, r.snapshot_role, r.contender_rank,
                                  r.generation_run_id
                           FROM forecast_snapshot_roster r
                           JOIN target t ON t.record_month = r.record_month
                           WHERE r.selected_at <= t.replacement_at
                       ), snapshot_rows AS (
                           SELECT f.*, r.snapshot_role, r.generation_run_id
                           FROM fact_forecast_snapshot f
                           JOIN roster r ON r.model_id = f.model_id
                           JOIN target t ON t.record_month = f.record_month
                           WHERE f.archived_at >= t.outgoing_promoted_at
                             AND f.archived_at <= t.replacement_at
                       ), archived AS (
                           SELECT model_id, lag, COUNT(*) AS row_count
                           FROM snapshot_rows
                           GROUP BY model_id, lag
                       )
                       SELECT
                           (SELECT COUNT(*) FROM roster),
                           (SELECT COUNT(*) FROM roster
                            WHERE snapshot_role = 'champion'
                              AND model_id = 'champion'),
                           (SELECT COUNT(DISTINCT contender_rank) FROM roster
                            WHERE snapshot_role = 'contender'),
                           COUNT(DISTINCT model_id), COUNT(*), MIN(row_count),
                           (SELECT COUNT(DISTINCT run_id)
                            FROM snapshot_rows
                            WHERE model_id = 'champion'),
                           (SELECT COUNT(*)
                            FROM snapshot_rows s
                            CROSS JOIN target t
                            WHERE (s.snapshot_role = 'champion'
                                   AND (s.run_id IS NULL
                                        OR s.plan_version IS DISTINCT FROM
                                           TO_CHAR(t.record_month, 'YYYY-MM')))
                               OR (s.snapshot_role = 'contender'
                                   AND s.run_id IS DISTINCT FROM s.generation_run_id))
                       FROM archived"""

    try:
        with get_read_only_conn() as conn, conn.cursor() as cur:
            cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            cur.execute(
                quality_sql,
                (
                    planning_month,
                    lookback_months,
                    planning_month,
                    planning_month,
                    lookback_months,
                    planning_month,
                ),
            )
            quality_row = cur.fetchone()
            cur.execute(state_sql)
            state_row = cur.fetchone()
            cur.execute(
                coverage_sql,
                (
                    planning_month,
                    cold_start_min_months,
                    planning_month,
                    active_window_months,
                    release_version,
                    planning_month,
                    planning_month,
                    lag_count,
                    lag_count,
                ),
            )
            coverage_row = cur.fetchone()
            active_plan_version = state_row[13] if state_row else None
            outgoing_plan_version = state_row[18] if state_row else None
            archive_record_month = (
                outgoing_plan_version if is_calendar_plan_version(outgoing_plan_version) else None
            )
            cur.execute(
                archive_sql,
                (
                    archive_record_month,
                    state_row[19] if state_row else None,
                    state_row[14] if state_row else None,
                ),
            )
            archive_row = cur.fetchone()
    except psycopg.Error:
        logger.exception("Failed to evaluate forecast release readiness")
        raise HTTPException(
            status_code=500,
            detail="Failed to evaluate forecast release readiness",
        ) from None

    quality_metrics = ReleaseQualityMetrics(
        dfu_months=int(quality_row[0] or 0),
        dfus=int(quality_row[1] or 0),
        champion_observations=int(quality_row[9] or 0),
        champion_dfus=int(quality_row[10] or 0),
        closed_months=int(quality_row[11] or 0),
        actual_volume=float(quality_row[12] or 0),
        champion_wape_pct=as_optional_float(quality_row[2]),
        champion_bias_pct=as_optional_float(quality_row[3]),
        naive_wape_pct=as_optional_float(quality_row[4]),
        external_wape_pct=as_optional_float(quality_row[5]),
    )
    policy_enabled = bool(gate_config["enabled"])
    checks = [
        build_gate_check(
            "readiness_policy",
            passed=policy_enabled,
            value=policy_enabled,
            threshold=True,
            message=(
                "Forecast release readiness policy is enabled."
                if policy_enabled
                else "Forecast release readiness policy is disabled and fails closed."
            ),
        ),
        *evaluate_quality_checks(quality_metrics, thresholds),
    ]
    actual_mismatches = int(quality_row[8] or 0)
    actuals_aligned = actual_mismatches == 0 and quality_metrics.dfu_months > 0
    checks.append(
        build_gate_check(
            "actual_alignment",
            passed=actuals_aligned,
            value=actual_mismatches,
            threshold=0,
            message=(
                "Actual demand is identical for all three common-cohort series."
                if actuals_aligned
                else "Actual demand differs across model rows for otherwise matching keys."
            ),
        )
    )

    active_promotion_id = state_row[0] if state_row else None
    active_promotion_count = int(state_row[1] or 0) if state_row else 0
    champion_experiment_id = state_row[2] if state_row else None
    champion_cluster_id = state_row[3] if state_row else None
    champion_results_promoted = bool(state_row[4]) if state_row else False
    results_promoted_at = state_row[5] if state_row else None
    results_promoted_count = int(state_row[6] or 0) if state_row else 0
    champion_rows_modified_at = state_row[7] if state_row else None
    current_cluster_id = state_row[8] if state_row else None
    promoted_cluster_count = int(state_row[9] or 0) if state_row else 0
    cluster_assignment_count = int(state_row[10] or 0) if state_row else 0
    latest_sales_load = state_row[11] if state_row else None
    stale_tuning_profiles = int(state_row[12] or 0) if state_row else 0
    active_plan_version = state_row[13] if state_row else None
    release_promoted_at = state_row[14] if state_row else None
    release_generated_at = state_row[15] if state_row else None
    staging_record_month = state_row[16] if state_row else None
    outgoing_promotion_id = state_row[17] if state_row else None
    outgoing_plan_version = state_row[18] if state_row else None
    outgoing_promoted_at = state_row[19] if state_row else None

    checks.append(
        build_active_promotion_check(
            active_promotion_id, active_promotion_count, active_plan_version
        )
    )

    checks.append(
        build_results_lineage_check(
            champion_experiment_id,
            champion_results_promoted,
            results_promoted_count,
            results_promoted_at,
            champion_rows_modified_at,
        )
    )

    lineage_required = bool(gate_config["require_current_cluster_lineage"])
    cluster_check, lineage_matches = build_cluster_lineage_check(
        champion_cluster_id, current_cluster_id, promoted_cluster_count, lineage_required
    )
    checks.append(cluster_check)
    cluster_assignments_present = cluster_assignment_count > 0
    checks.append(
        build_gate_check(
            "cluster_assignments",
            passed=cluster_assignments_present,
            value=cluster_assignment_count,
            threshold="> 0 promoted SKU assignments",
            message=(
                "Promoted SKU cluster assignments are available."
                if cluster_assignments_present
                else "No promoted SKU cluster assignments are available."
            ),
        )
    )

    sales_fresh = (
        release_generated_at is not None
        and latest_sales_load is not None
        and release_generated_at >= latest_sales_load
    )
    freshness_required = bool(gate_config["require_fresh_sales"])
    sales_freshness_passed = sales_fresh or not freshness_required
    checks.append(
        build_gate_check(
            "sales_freshness",
            passed=sales_freshness_passed,
            value={
                "release_promoted_at": release_promoted_at.isoformat()
                if release_promoted_at
                else None,
                "release_generated_at": release_generated_at.isoformat()
                if release_generated_at
                else None,
                "latest_sales_load": latest_sales_load.isoformat() if latest_sales_load else None,
            },
            threshold="oldest active release row generated at or after latest sales load",
            message=(
                "Release generation freshness is not required by policy."
                if not freshness_required
                else (
                    "Every active release row was generated after the latest completed sales load."
                    if sales_fresh
                    else "The active release was generated before the latest completed sales load."
                )
            ),
        )
    )
    tuning_fresh = stale_tuning_profiles == 0
    checks.append(
        build_gate_check(
            "tuning_freshness",
            passed=tuning_fresh,
            value=stale_tuning_profiles,
            threshold=0,
            message=(
                "No per-cluster tuning profiles are marked stale."
                if tuning_fresh
                else f"{stale_tuning_profiles} per-cluster tuning profiles are stale."
            ),
        )
    )

    eligible_dfus = int(coverage_row[0] or 0)
    complete_plan_dfus = int(coverage_row[1] or 0)
    covered_eligible_dfus = int(coverage_row[2] or 0)
    current_plan_rows = int(coverage_row[3] or 0)
    current_plan_min_month = coverage_row[4]
    current_plan_max_month = coverage_row[5]
    release_run_ids = int(coverage_row[6] or 0)
    invalid_quantity_rows = int(coverage_row[7] or 0)
    missing_source_rows = int(coverage_row[8] or 0)
    invalid_interval_rows = int(coverage_row[9] or 0)
    interval_rows = int(coverage_row[10] or 0)
    interval_coverage_frac = interval_rows / current_plan_rows if current_plan_rows > 0 else None
    coverage_frac = covered_eligible_dfus / eligible_dfus if eligible_dfus > 0 else None
    current_version = active_plan_version == release_version
    checks.append(
        build_gate_check(
            "current_plan_version",
            passed=current_version,
            value=active_plan_version,
            threshold=release_version,
            message=(
                "The active production champion is versioned for the planning month."
                if current_version
                else "The active production champion is not versioned for the planning month."
            ),
        )
    )
    complete_horizon = (
        current_plan_min_month == planning_month and current_plan_max_month == release_end
    )
    current_plan_complete = (
        current_version
        and complete_horizon
        and coverage_frac is not None
        and coverage_frac >= thresholds.min_current_plan_coverage_frac
    )
    checks.append(
        build_gate_check(
            "current_plan_coverage",
            passed=current_plan_complete,
            value=round(coverage_frac, 4) if coverage_frac is not None else None,
            threshold=thresholds.min_current_plan_coverage_frac,
            message=(
                "Six-month eligible-DFU coverage meets the readiness threshold."
                if current_plan_complete
                else "The current release lacks complete six-month coverage for eligible DFUs."
            ),
        )
    )
    release_integrity_passed = (
        current_version
        and current_plan_rows > 0
        and release_run_ids == 1
        and invalid_quantity_rows == 0
        and missing_source_rows == 0
        and invalid_interval_rows == 0
        and interval_coverage_frac is not None
        and interval_coverage_frac >= min_ci_coverage_frac
    )
    checks.append(
        build_gate_check(
            "release_integrity",
            passed=release_integrity_passed,
            value={
                "run_ids": release_run_ids,
                "invalid_quantity_rows": invalid_quantity_rows,
                "missing_source_rows": missing_source_rows,
                "invalid_interval_rows": invalid_interval_rows,
                "confidence_interval_coverage_frac": round(interval_coverage_frac, 4)
                if interval_coverage_frac is not None
                else None,
            },
            threshold={
                "run_ids": 1,
                "invalid_rows": 0,
                "min_confidence_interval_coverage_frac": min_ci_coverage_frac,
            },
            message=(
                "The active plan is one coherent run with valid quantities, lineage, and intervals."
                if release_integrity_passed
                else "The active plan has mixed runs, invalid values, missing lineage, or insufficient intervals."
            ),
        )
    )

    archive_roster_rows = int(archive_row[0] or 0)
    archive_champion_rows = int(archive_row[1] or 0)
    archive_contender_ranks = int(archive_row[2] or 0)
    archive_models = int(archive_row[3] or 0)
    archive_model_lags = int(archive_row[4] or 0)
    archive_min_rows = int(archive_row[5] or 0)
    archive_champion_run_ids = int(archive_row[6] or 0)
    archive_lineage_mismatches = int(archive_row[7] or 0)
    archive_required = bool(gate_config["require_outgoing_archive"])
    archive_version_valid = outgoing_promotion_id is None or is_calendar_plan_version(
        outgoing_plan_version
    )
    archive_complete = outgoing_promotion_id is None or (
        archive_version_valid
        and archive_roster_rows == archive_model_count
        and archive_champion_rows == 1
        and archive_contender_ranks == contender_count
        and archive_models == archive_model_count
        and archive_model_lags == archive_model_count * lag_count
        and archive_min_rows > 0
        and archive_champion_run_ids == 1
        and archive_lineage_mismatches == 0
    )
    archive_passed = archive_complete or not archive_required
    checks.append(
        build_gate_check(
            "outgoing_archive",
            passed=archive_passed,
            value={
                "outgoing_promotion_id": outgoing_promotion_id,
                "outgoing_plan_version": outgoing_plan_version,
                "outgoing_promoted_at": outgoing_promoted_at.isoformat()
                if outgoing_promoted_at
                else None,
                "replacement_at": release_promoted_at.isoformat() if release_promoted_at else None,
                "roster_rows": archive_roster_rows,
                "champion_roster_rows": archive_champion_rows,
                "contender_ranks": archive_contender_ranks,
                "models": archive_models,
                "model_lag_pairs": archive_model_lags,
                "minimum_rows": archive_min_rows,
                "champion_run_ids": archive_champion_run_ids,
                "lineage_mismatches": archive_lineage_mismatches,
            },
            threshold=f"{archive_model_count} models x {lag_count} lags",
            message=(
                "Outgoing archive evidence is not required by policy."
                if not archive_required
                else (
                    "There is no preceding plan to archive."
                    if outgoing_promotion_id is None
                    else (
                        "The outgoing plan version is not a valid YYYY-MM archive key."
                        if not archive_version_valid
                        else (
                            "The outgoing plan has a complete, lineage-consistent bounded archive."
                            if archive_complete
                            else "The outgoing plan was not completely archived before replacement."
                        )
                    )
                )
            ),
        )
    )

    ready = policy_enabled and all(check["status"] == "pass" for check in checks)
    response_payload = {
        "ready": ready,
        "policy_enabled": policy_enabled,
        "release_version": release_version,
        "planning_month": planning_month.isoformat(),
        "champion_experiment_id": champion_experiment_id,
        "quality": {
            "lookback_months": lookback_months,
            "first_month": quality_row[6].isoformat() if quality_row[6] else None,
            "last_month": quality_row[7].isoformat() if quality_row[7] else None,
            "dfu_months": quality_metrics.dfu_months,
            "dfus": quality_metrics.dfus,
            "closed_months": quality_metrics.closed_months,
            "actual_volume": quality_metrics.actual_volume,
            "champion_observations": quality_metrics.champion_observations,
            "champion_dfus": quality_metrics.champion_dfus,
            "common_observation_coverage_frac": (
                round(quality_metrics.common_observation_coverage_frac, 4)
                if quality_metrics.common_observation_coverage_frac is not None
                else None
            ),
            "common_dfu_coverage_frac": (
                round(quality_metrics.common_dfu_coverage_frac, 4)
                if quality_metrics.common_dfu_coverage_frac is not None
                else None
            ),
            "champion_wape_pct": quality_metrics.champion_wape_pct,
            "champion_accuracy_pct": (
                100.0 - quality_metrics.champion_wape_pct
                if quality_metrics.champion_wape_pct is not None
                else None
            ),
            "champion_bias_pct": quality_metrics.champion_bias_pct,
            "naive_wape_pct": quality_metrics.naive_wape_pct,
            "external_wape_pct": quality_metrics.external_wape_pct,
            "relative_wape_lift_vs_naive_pct": (
                round(quality_metrics.relative_wape_lift_vs_naive_pct, 4)
                if quality_metrics.relative_wape_lift_vs_naive_pct is not None
                else None
            ),
            "accuracy_delta_vs_external_pct_points": (
                round(
                    quality_metrics.accuracy_delta_vs_external_pct_points,
                    4,
                )
                if quality_metrics.accuracy_delta_vs_external_pct_points is not None
                else None
            ),
        },
        "lineage": {
            "active_promotion_id": active_promotion_id,
            "active_promotion_count": active_promotion_count,
            "champion_results_promoted": champion_results_promoted,
            "results_promoted_at": results_promoted_at.isoformat() if results_promoted_at else None,
            "champion_rows_modified_at": champion_rows_modified_at.isoformat()
            if champion_rows_modified_at
            else None,
            "results_promoted_experiment_count": results_promoted_count,
            "champion_cluster_experiment_id": champion_cluster_id,
            "current_cluster_experiment_id": current_cluster_id,
            "promoted_cluster_experiment_count": promoted_cluster_count,
            "matches": lineage_matches,
            "cluster_assignment_count": cluster_assignment_count,
            "stale_tuning_profiles": stale_tuning_profiles,
        },
        "freshness": {
            "release_promoted_at": release_promoted_at.isoformat() if release_promoted_at else None,
            "release_generated_at": release_generated_at.isoformat()
            if release_generated_at
            else None,
            "latest_sales_load": latest_sales_load.isoformat() if latest_sales_load else None,
            "fresh": sales_fresh,
        },
        "coverage": {
            "eligible_dfus": eligible_dfus,
            "complete_plan_dfus": complete_plan_dfus,
            "covered_eligible_dfus": covered_eligible_dfus,
            "current_plan_rows": current_plan_rows,
            "coverage_frac": round(coverage_frac, 4) if coverage_frac is not None else None,
            "forecast_start": current_plan_min_month.isoformat()
            if current_plan_min_month
            else None,
            "forecast_end": current_plan_max_month.isoformat() if current_plan_max_month else None,
            "required_end": release_end.isoformat(),
            "minimum_history_months": cold_start_min_months,
        },
        "release_integrity": {
            "run_ids": release_run_ids,
            "invalid_quantity_rows": invalid_quantity_rows,
            "missing_source_rows": missing_source_rows,
            "invalid_interval_rows": invalid_interval_rows,
            "confidence_interval_rows": interval_rows,
            "confidence_interval_coverage_frac": round(interval_coverage_frac, 4)
            if interval_coverage_frac is not None
            else None,
            "minimum_confidence_interval_coverage_frac": min_ci_coverage_frac,
            "valid": release_integrity_passed,
        },
        "archive": {
            "active_plan_version": active_plan_version,
            "outgoing_promotion_id": outgoing_promotion_id,
            "outgoing_plan_version": outgoing_plan_version,
            "outgoing_promoted_at": outgoing_promoted_at.isoformat()
            if outgoing_promoted_at
            else None,
            "replacement_at": release_promoted_at.isoformat() if release_promoted_at else None,
            "staging_record_month": staging_record_month.isoformat()
            if staging_record_month
            else None,
            "models": archive_models,
            "roster_rows": archive_roster_rows,
            "champion_roster_rows": archive_champion_rows,
            "contender_ranks": archive_contender_ranks,
            "model_lag_pairs": archive_model_lags,
            "minimum_rows": archive_min_rows,
            "champion_run_ids": archive_champion_run_ids,
            "lineage_mismatches": archive_lineage_mismatches,
            "complete": archive_complete,
        },
        "checks": checks,
        "next_action": next_release_action(checks),
    }
    return ForecastReleaseReadinessResponse.model_validate(response_payload).model_dump(mode="json")
