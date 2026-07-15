"""Readiness and durable reservation policy for customer bottom-up blends."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from common.core.constants import CHAMPION_MODEL_ID
from common.services.customer_forecast import (
    customer_forecast_config_checksum,
    get_customer_forecast_settings,
)
from common.services.customer_forecast_blend_contract import (
    CUSTOMER_BLEND_LINEAGE_METADATA_KEY,
    CustomerBlendSettings,
    customer_blend_config_checksum,
    get_customer_blend_settings,
)
from common.services.customer_forecast_blend_evidence import (
    CustomerForecastPayloadStats,
    compute_customer_forecast_output_stats,
)
from common.services.forecast_generation import reserve_generation_run

_SUBMISSION_RECONCILIATION_GRACE_SECONDS = 300
_FROZEN_BLEND_LINEAGE_KEYS = (
    "config_checksum",
    "customer_run_id",
    "customer_config_checksum",
    "customer_source_checksum",
    "source_customer_demand_batch_id",
    "source_promotion_id",
    "source_run_id",
    "source_production_run_id",
    "source_production_checksum",
    "backtest_run_id",
    "backtest_config_checksum",
    "backtest_component_checksum",
    "backtest_component_rows",
    "backtest_gate",
)


def load_customer_blend_readiness(
    conn: Any,
    customer_run_id: UUID | None = None,
    *,
    require_backtest: bool = False,
    verify_evidence: bool = False,
) -> dict[str, Any]:
    """Resolve one completed customer run and the active governed champion."""
    if verify_evidence and not require_backtest:
        raise ValueError("Customer blend evidence verification requires a backtest")
    settings = get_customer_blend_settings()
    current_customer_settings = get_customer_forecast_settings()
    customer_output_stats: CustomerForecastPayloadStats | None = None
    backtest_component_stats: tuple[str, int] | None = None
    with conn.cursor() as cur:
        customer_sql = """SELECT run_id, planning_month, history_start, history_end,
                                 forecast_start, forecast_end, horizon_months,
                                 row_count, eligible_series, config_checksum,
                                 source_checksum, completed_at, model_id,
                                 source_customer_demand_batch_id,
                                 (
                                     SELECT batch_id
                                     FROM audit_load_batch
                                     WHERE domain = 'customer_demand'
                                       AND status = 'completed'
                                     ORDER BY completed_at DESC NULLS LAST, batch_id DESC
                                     LIMIT 1
                                 ),
                                 (
                                     SELECT source_batch_id
                                     FROM customer_demand_profile_refresh_state
                                     WHERE singleton_id = 1
                                 ),
                                 (
                                     SELECT COUNT(*)
                                     FROM audit_load_batch
                                     WHERE domain = 'customer_demand'
                                       AND status = 'running'
                                 )
                          FROM customer_forecast_run
                          WHERE run_status = 'completed'"""
        params: tuple[Any, ...] = ()
        if customer_run_id is not None:
            customer_sql += " AND run_id = %s::uuid"
            params = (str(customer_run_id),)
        customer_sql += " ORDER BY completed_at DESC, created_at DESC LIMIT 1"
        cur.execute(customer_sql, params)
        customer = cur.fetchone()
        cur.execute(
            """SELECT promotion.id, promotion.model_id, promotion.plan_version,
                      promotion.source_run_id, promotion.production_run_id,
                      promotion.production_checksum,
                      generation.forecast_month_generated,
                      generation.horizon_months,
                      generation.champion_experiment_id,
                      generation.cluster_experiment_id,
                      generation.source_sales_batch_id,
                      generation.routing_artifact_checksum,
                      generation.champion_results_checksum,
                      generation.metadata
               FROM model_promotion_log promotion
               LEFT JOIN forecast_generation_run generation
                 ON generation.run_id = promotion.source_run_id
               WHERE promotion.is_active = TRUE
               ORDER BY promotion.promoted_at DESC, promotion.id DESC
               LIMIT 1"""
        )
        source = cur.fetchone()
        backtest = None
        if customer is not None and source is not None and source[4] is not None:
            cur.execute(
                """SELECT backtest.run_id, backtest.config_checksum,
                          backtest.component_checksum, backtest.component_rows,
                          backtest.completed_at,
                          accuracy.gate_passed, accuracy.gate_reason,
                          accuracy.common_months, accuracy.common_dfus,
                          accuracy.champion_wape_pct,
                          accuracy.customer_wape_pct,
                          accuracy.blend_wape_pct,
                          accuracy.blend_wape_degradation_pct
                   FROM customer_forecast_backtest_run backtest
                   JOIN customer_bottom_up_backtest_accuracy accuracy
                     ON accuracy.backtest_run_id = backtest.run_id
                   WHERE backtest.run_status = 'completed'
                     AND backtest.customer_run_id = %s::uuid
                     AND backtest.source_promotion_id = %s
                     AND backtest.source_production_run_id = %s::uuid
                   ORDER BY backtest.completed_at DESC, backtest.created_at DESC
                   LIMIT 1""",
                (str(customer[0]), int(source[0]), str(source[4])),
            )
            backtest = cur.fetchone()
    source_metadata = (
        dict(source[13]) if source is not None and isinstance(source[13], dict) else {}
    )
    blockers: list[str] = []
    if not settings.enabled:
        blockers.append("Enable the customer bottom-up blend in configuration")
    if customer is None:
        blockers.append("Complete a customer forecast run before building a bottom-up blend")
    if source is None:
        blockers.append("Promote a governed champion before building a bottom-up blend")
    elif source[1] != CHAMPION_MODEL_ID:
        blockers.append("The active production release must be a governed champion")
    elif CUSTOMER_BLEND_LINEAGE_METADATA_KEY in source_metadata:
        blockers.append("Promote a fresh unblended champion before creating another customer blend")
    elif any(source[index] is None for index in (3, 4, 5, 6, 7, 8, 9, 10, 11, 12)):
        blockers.append("The active champion is missing immutable release lineage")

    if customer is not None:
        if any(customer[index] is None for index in (9, 10, 11)):
            blockers.append("The customer forecast run is missing immutable source lineage")
        if customer[13] is None:
            blockers.append(
                "Generate a new customer forecast bound to a completed customer-demand load"
            )
        elif int(customer[16] or 0) > 0:
            blockers.append("Wait for the active customer-demand load before building a blend")
        elif (
            customer[14] is None
            or customer[15] is None
            or int(customer[13]) != int(customer[14])
            or int(customer[13]) != int(customer[15])
        ):
            blockers.append(
                "Generate a new customer forecast from the latest refreshed customer-demand load"
            )
        if customer[12] != current_customer_settings["model_id"] or customer[
            9
        ] != customer_forecast_config_checksum(current_customer_settings):
            blockers.append("Generate a new rule-routed customer forecast with current configuration")
        if customer[4] != customer[1]:
            blockers.append("The customer forecast window does not start in its planning month")
    if customer is not None and source is not None and source[6] is not None:
        if customer[1] != source[6]:
            blockers.append("Customer and champion forecasts belong to different planning months")
    if require_backtest:
        if backtest is None:
            blockers.append("Complete the current customer bottom-up accuracy backtest")
        else:
            from common.services.customer_forecast_backtest import (
                customer_backtest_config_checksum,
                get_customer_backtest_settings,
            )

            expected_backtest_checksum = customer_backtest_config_checksum(
                get_customer_backtest_settings(),
                settings,
                current_customer_settings,
            )
            if backtest[1] != expected_backtest_checksum:
                blockers.append(
                    "Complete a new customer bottom-up accuracy backtest with current configuration"
                )
            if not bool(backtest[5]):
                blockers.append(f"Customer blend backtest gate failed: {backtest[6]}")

    if verify_evidence and not blockers and customer is not None and backtest is not None:
        with conn.cursor() as cur:
            customer_output_stats = compute_customer_forecast_output_stats(
                cur,
                customer[0],
            )
            from common.services.customer_forecast_backtest import (
                compute_customer_backtest_component_stats,
            )

            backtest_component_stats = compute_customer_backtest_component_stats(
                cur,
                backtest[0],
            )
        if customer_output_stats.row_count != int(
            customer[7]
        ) or customer_output_stats.series_count != int(customer[8]):
            blockers.append("The completed customer forecast output no longer matches its manifest")
        if backtest_component_stats != (backtest[2], int(backtest[3])):
            blockers.append(
                "The completed customer backtest evidence no longer matches its manifest"
            )

    return {
        "ready": not blockers,
        "blockers": blockers,
        "settings": settings,
        "customer_run_id": str(customer[0]) if customer is not None else None,
        "planning_month": customer[1] if customer is not None else None,
        "history_start": customer[2] if customer is not None else None,
        "history_end": customer[3] if customer is not None else None,
        "customer_forecast_start": customer[4] if customer is not None else None,
        "customer_forecast_end": customer[5] if customer is not None else None,
        "customer_horizon_months": int(customer[6]) if customer is not None else 0,
        "customer_row_count": int(customer[7]) if customer is not None else 0,
        "customer_series_count": int(customer[8]) if customer is not None else 0,
        "customer_config_checksum": customer[9] if customer is not None else None,
        "customer_source_checksum": customer[10] if customer is not None else None,
        "customer_output_checksum": (
            customer_output_stats.checksum if customer_output_stats is not None else None
        ),
        "customer_output_row_count": (
            customer_output_stats.row_count if customer_output_stats is not None else 0
        ),
        "customer_output_series_count": (
            customer_output_stats.series_count if customer_output_stats is not None else 0
        ),
        "customer_model_id": customer[12] if customer is not None else None,
        "source_customer_demand_batch_id": (
            int(customer[13]) if customer is not None and customer[13] is not None else None
        ),
        "profile_customer_demand_batch_id": (
            int(customer[15]) if customer is not None and customer[15] is not None else None
        ),
        "active_customer_demand_loads": (int(customer[16] or 0) if customer is not None else 0),
        "source_promotion_id": int(source[0]) if source is not None else None,
        "source_plan_version": source[2] if source is not None else None,
        "source_run_id": str(source[3]) if source is not None and source[3] else None,
        "source_production_run_id": (str(source[4]) if source is not None and source[4] else None),
        "source_production_checksum": source[5] if source is not None else None,
        "source_horizon_months": int(source[7]) if source is not None and source[7] else 0,
        "champion_experiment_id": int(source[8]) if source is not None and source[8] else None,
        "cluster_experiment_id": int(source[9]) if source is not None and source[9] else None,
        "source_sales_batch_id": int(source[10]) if source is not None and source[10] else None,
        "routing_artifact_checksum": source[11] if source is not None else None,
        "champion_results_checksum": source[12] if source is not None else None,
        "source_metadata": source_metadata,
        "promotion_enabled": settings.promotion_enabled,
        "promotion_reason": settings.promotion_reason,
        "backtest_run_id": str(backtest[0]) if backtest is not None else None,
        "backtest_config_checksum": backtest[1] if backtest is not None else None,
        "backtest_component_checksum": backtest[2] if backtest is not None else None,
        "backtest_component_rows": int(backtest[3]) if backtest is not None else 0,
        "backtest_completed_at": backtest[4] if backtest is not None else None,
        "backtest_gate_passed": bool(backtest[5]) if backtest is not None else False,
        "backtest_gate_reason": backtest[6] if backtest is not None else None,
        "backtest_common_months": int(backtest[7]) if backtest is not None else 0,
        "backtest_common_dfus": int(backtest[8]) if backtest is not None else 0,
        "champion_wape_pct": (
            float(backtest[9]) if backtest is not None and backtest[9] is not None else None
        ),
        "customer_wape_pct": (
            float(backtest[10]) if backtest is not None and backtest[10] is not None else None
        ),
        "blend_wape_pct": (
            float(backtest[11]) if backtest is not None and backtest[11] is not None else None
        ),
        "blend_wape_degradation_pct": (
            float(backtest[12]) if backtest is not None and backtest[12] is not None else None
        ),
    }


def reserve_customer_blend_generation(
    conn: Any,
    *,
    run_id: UUID,
    customer_run_id: UUID | None = None,
) -> dict[str, Any]:
    """Persist one recoverable blend manifest before its managed job is submitted."""
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtext('customer_bottom_up_blend_submission'))"
            )
            cur.execute(
                """UPDATE forecast_generation_run generation
                   SET run_status = CASE
                           WHEN job.status IN ('failed', 'cancelled', 'completed')
                           THEN 'invalid'
                           ELSE generation.run_status
                       END,
                       invalid_reason = CASE
                           WHEN job.status = 'cancelled' THEN 'managed job cancelled'
                           WHEN job.status = 'failed' THEN 'managed job failed'
                           WHEN job.status = 'completed'
                           THEN 'managed job completed without a ready manifest'
                           ELSE generation.invalid_reason
                       END,
                       completed_at = CASE
                           WHEN job.status IN ('failed', 'cancelled', 'completed')
                           THEN COALESCE(job.completed_at, NOW())
                           ELSE generation.completed_at
                       END
                   FROM job_history job
                   WHERE generation.run_status = 'generating'
                     AND generation.metadata ? %s
                     AND job.job_type = 'generate_customer_forecast_blend'
                     AND job.status IN ('failed', 'cancelled', 'completed')
                     AND job.params ->> 'run_id' = generation.run_id::text""",
                (CUSTOMER_BLEND_LINEAGE_METADATA_KEY,),
            )
            cur.execute(
                """UPDATE forecast_generation_run generation
                   SET run_status = 'invalid',
                       invalid_reason = 'job submission was not persisted',
                       completed_at = NOW()
                   WHERE generation.run_status = 'generating'
                     AND generation.metadata ? %s
                     AND generation.created_at
                         < NOW() - (%s * INTERVAL '1 second')
                     AND NOT EXISTS (
                         SELECT 1
                         FROM job_history job
                         WHERE job.job_type = 'generate_customer_forecast_blend'
                           AND job.params ->> 'run_id' = generation.run_id::text
                     )""",
                (
                    CUSTOMER_BLEND_LINEAGE_METADATA_KEY,
                    _SUBMISSION_RECONCILIATION_GRACE_SECONDS,
                ),
            )
            readiness = load_customer_blend_readiness(
                conn,
                customer_run_id,
                require_backtest=True,
            )
            if not readiness["ready"]:
                return readiness
            cur.execute(
                """SELECT run_id::text
                   FROM forecast_generation_run
                   WHERE run_status = 'generating'
                     AND metadata ? %s
                   ORDER BY created_at DESC
                   LIMIT 1""",
                (CUSTOMER_BLEND_LINEAGE_METADATA_KEY,),
            )
            active = cur.fetchone()
            if active is not None:
                return {
                    **readiness,
                    "ready": False,
                    "blockers": ["A customer bottom-up blend is already generating"],
                    "active_run_id": str(active[0]),
                }

            settings: CustomerBlendSettings = readiness["settings"]
            source_metadata = dict(readiness["source_metadata"])
            source_metadata[CUSTOMER_BLEND_LINEAGE_METADATA_KEY] = {
                **settings.as_lineage(),
                "config_checksum": customer_blend_config_checksum(settings),
                "status": "queued",
                "customer_run_id": readiness["customer_run_id"],
                "customer_config_checksum": readiness["customer_config_checksum"],
                "customer_source_checksum": readiness["customer_source_checksum"],
                "source_customer_demand_batch_id": readiness["source_customer_demand_batch_id"],
                "source_promotion_id": readiness["source_promotion_id"],
                "source_run_id": readiness["source_run_id"],
                "source_production_run_id": readiness["source_production_run_id"],
                "source_production_checksum": readiness["source_production_checksum"],
                "backtest_run_id": readiness["backtest_run_id"],
                "backtest_config_checksum": readiness["backtest_config_checksum"],
                "backtest_component_checksum": readiness["backtest_component_checksum"],
                "backtest_component_rows": readiness["backtest_component_rows"],
                "backtest_gate": {
                    "passed": readiness["backtest_gate_passed"],
                    "reason": readiness["backtest_gate_reason"],
                    "common_months": readiness["backtest_common_months"],
                    "common_dfus": readiness["backtest_common_dfus"],
                    "champion_wape_pct": readiness["champion_wape_pct"],
                    "customer_wape_pct": readiness["customer_wape_pct"],
                    "blend_wape_pct": readiness["blend_wape_pct"],
                    "blend_wape_degradation_pct": readiness["blend_wape_degradation_pct"],
                },
            }
            status = reserve_generation_run(
                cur,
                run_id=run_id,
                generation_purpose="release_candidate",
                requested_model_id=CHAMPION_MODEL_ID,
                record_month=readiness["planning_month"],
                horizon_months=int(readiness["source_horizon_months"]),
                created_by="customer-bottom-up-blend",
                metadata=source_metadata,
            )
            if status != "generating":
                raise ValueError(f"Customer blend run is already {status}")
    return readiness


def _validate_reserved_customer_blend_lineage(
    cur: Any,
    run_id: UUID,
    expected: dict[str, Any],
) -> None:
    """Fail closed when a queued blend's exact source identity has drifted."""
    cur.execute(
        """SELECT metadata -> %s
           FROM forecast_generation_run
           WHERE run_id = %s::uuid
             AND run_status = 'generating'""",
        (CUSTOMER_BLEND_LINEAGE_METADATA_KEY, str(run_id)),
    )
    row = cur.fetchone()
    recorded = row[0] if row is not None and isinstance(row[0], dict) else None
    if recorded is None or any(
        recorded.get(key) != expected.get(key) for key in _FROZEN_BLEND_LINEAGE_KEYS
    ):
        raise ValueError("Reserved customer blend lineage changed before generation")
