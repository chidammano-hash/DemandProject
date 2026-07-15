"""Governed customer bottom-up blend candidates at warehouse-item grain."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from psycopg.types.json import Jsonb

from common.core.constants import (
    CHAMPION_MODEL_ID,
    CUSTOMER_BOTTOM_UP_BLEND_MODEL_ID,
    CUSTOMER_BOTTOM_UP_MODEL_ID,
)
from common.services.customer_demand_lineage import customer_demand_snapshot_locked
from common.services.customer_forecast import _shift_month
from common.services.customer_forecast_blend_contract import (
    CUSTOMER_BLEND_LINEAGE_METADATA_KEY,
    CustomerBlendSettings,
    customer_blend_config_checksum,
)
from common.services.customer_forecast_blend_readiness import (
    _validate_reserved_customer_blend_lineage,
    load_customer_blend_readiness,
)
from common.services.forecast_generation import reserve_generation_run
from common.services.forecast_lineage import (
    ForecastPayloadStats,
    compute_production_payload_stats,
    compute_staging_payload_stats,
)


@dataclass(frozen=True)
class CustomerBlendGenerationResult:
    run_id: UUID
    bottom_up_staging_run_id: UUID
    customer_run_id: UUID
    source_promotion_id: int
    source_production_run_id: UUID
    row_count: int
    dfu_count: int
    blended_row_count: int
    fallback_row_count: int
    excluded_customer_dfu_count: int
    checksum: str


@dataclass(frozen=True)
class CustomerBottomUpShadowResult:
    run_id: UUID
    status: str
    stats: ForecastPayloadStats


@dataclass(frozen=True)
class CustomerBlendComponentLineage:
    customer_run_id: UUID
    backtest_run_id: UUID
    source_promotion_id: int
    source_production_run_id: UUID


CUSTOMER_BOTTOM_UP_STAGING_METADATA_KEY = "customer_bottom_up_staging"


def customer_bottom_up_shadow_run_id(blend_run_id: UUID | str) -> UUID:
    """Derive the immutable review-run identity paired with one blend draft."""
    return uuid5(
        NAMESPACE_URL,
        f"demand-project/customer-bottom-up-staging/{UUID(str(blend_run_id))}",
    )


def stage_customer_bottom_up_shadow(
    cur: Any,
    *,
    blend_run_id: UUID,
    planning_month: date,
    horizon_months: int,
    source_metadata: dict[str, Any],
    lineage: dict[str, Any],
    champion_experiment_id: int,
    cluster_experiment_id: int,
    source_sales_batch_id: int,
    routing_artifact_checksum: str,
    champion_results_checksum: str,
) -> CustomerBottomUpShadowResult:
    """Stage the normalized customer signal as immutable, non-promotable evidence."""
    shadow_run_id = customer_bottom_up_shadow_run_id(blend_run_id)
    shadow_lineage = {
        **lineage,
        "model_id": CUSTOMER_BOTTOM_UP_MODEL_ID,
        "source_blend_run_id": str(blend_run_id),
        "status": "generating",
    }
    metadata = {
        key: value
        for key, value in source_metadata.items()
        if key != CUSTOMER_BLEND_LINEAGE_METADATA_KEY
    }
    metadata[CUSTOMER_BOTTOM_UP_STAGING_METADATA_KEY] = shadow_lineage
    status = reserve_generation_run(
        cur,
        run_id=shadow_run_id,
        generation_purpose="shadow_candidate",
        requested_model_id=CUSTOMER_BOTTOM_UP_MODEL_ID,
        record_month=planning_month,
        horizon_months=horizon_months,
        created_by="customer-bottom-up-shadow",
        metadata=metadata,
    )
    if status != "generating":
        raise ValueError(f"Customer bottom-up shadow run is already {status}")

    cur.execute(
        """UPDATE forecast_generation_run
           SET champion_experiment_id = %s,
               cluster_experiment_id = %s,
               source_sales_batch_id = %s,
               routing_artifact_checksum = %s,
               champion_results_checksum = %s
           WHERE run_id = %s::uuid
             AND generation_purpose = 'shadow_candidate'
             AND run_status = 'generating'""",
        (
            champion_experiment_id,
            cluster_experiment_id,
            source_sales_batch_id,
            routing_artifact_checksum,
            champion_results_checksum,
            str(shadow_run_id),
        ),
    )
    if cur.rowcount != 1:
        raise ValueError("Customer bottom-up shadow manifest could not accept source lineage")

    cur.execute(
        """INSERT INTO fact_production_forecast_staging
               (model_id, candidate_model_id, generation_purpose,
                item_id, loc, forecast_month, forecast_month_generated,
                forecast_qty, forecast_qty_lower, forecast_qty_upper,
                cluster_id, horizon_months, is_recursive, lag_source,
                generated_at, run_id)
           SELECT %s, %s, 'shadow_candidate',
                  component.item_id, component.loc, component.forecast_month,
                  %s, ROUND(component.normalized_customer_qty, 2),
                  NULL, NULL, NULL, production.horizon_months,
                  FALSE, 'customer_demand', component.generated_at, %s::uuid
           FROM customer_bottom_up_blend_component component
           JOIN fact_production_forecast production
             ON production.run_id = component.source_production_run_id
            AND production.item_id = component.item_id
            AND production.loc = component.loc
            AND production.forecast_month = component.forecast_month
           WHERE component.run_id = %s::uuid
             AND component.normalized_customer_qty IS NOT NULL""",
        (
            CUSTOMER_BOTTOM_UP_MODEL_ID,
            CUSTOMER_BOTTOM_UP_MODEL_ID,
            planning_month,
            str(shadow_run_id),
            str(blend_run_id),
        ),
    )
    stats = compute_staging_payload_stats(cur, shadow_run_id)
    if stats.row_count <= 0 or stats.dfu_count <= 0:
        shadow_lineage.update({"status": "invalid", "row_count": 0, "dfu_count": 0})
        metadata[CUSTOMER_BOTTOM_UP_STAGING_METADATA_KEY] = shadow_lineage
        cur.execute(
            """UPDATE forecast_generation_run
               SET run_status = 'invalid', promotion_eligible = FALSE,
                   invalid_reason = 'no normalized customer rows were available',
                   artifact_checksum = %s, metadata = %s, completed_at = NOW()
               WHERE run_id = %s::uuid AND run_status = 'generating'""",
            (stats.checksum, Jsonb(metadata), str(shadow_run_id)),
        )
        if cur.rowcount != 1:
            raise ValueError("Customer bottom-up shadow manifest did not become invalid")
        return CustomerBottomUpShadowResult(shadow_run_id, "invalid", stats)

    shadow_lineage.update(
        {
            "status": "ready",
            "row_count": stats.row_count,
            "dfu_count": stats.dfu_count,
            "artifact_checksum": stats.checksum,
        }
    )
    metadata[CUSTOMER_BOTTOM_UP_STAGING_METADATA_KEY] = shadow_lineage
    cur.execute(
        """UPDATE forecast_generation_run
           SET run_status = 'ready', promotion_eligible = FALSE,
               row_count = %s, dfu_count = %s,
               candidate_model_count = %s, artifact_checksum = %s,
               metadata = %s, completed_at = NOW()
           WHERE run_id = %s::uuid
             AND generation_purpose = 'shadow_candidate'
             AND run_status = 'generating'""",
        (
            stats.row_count,
            stats.dfu_count,
            stats.source_model_count,
            stats.checksum,
            Jsonb(metadata),
            str(shadow_run_id),
        ),
    )
    if cur.rowcount != 1:
        raise ValueError("Customer bottom-up shadow manifest did not transition to ready")
    return CustomerBottomUpShadowResult(shadow_run_id, "ready", stats)


def compute_customer_blend_component_stats(
    cur: Any,
    run_id: UUID | str,
) -> tuple[str, int, int, int, int]:
    """Hash source and derived blend values as a commutative row-digest multiset."""
    cur.execute(
        """WITH canonical_rows AS (
                 SELECT item_id, loc, forecast_month,
                        (
                            'x' || ENCODE(
                                DIGEST(
                                    jsonb_build_array(
                                        run_id, customer_run_id, backtest_run_id,
                                        source_promotion_id,
                                        source_production_run_id,
                                        item_id, loc,
                                        TO_CHAR(forecast_month, 'YYYY-MM-DD'),
                                        raw_customer_demand_qty,
                                        normalized_customer_qty,
                                        champion_qty, blended_qty,
                                        blended_lower, blended_upper,
                                        fulfillment_ratio, customer_weight,
                                        champion_weight,
                                        effective_customer_weight,
                                        customer_series_count, coverage_status,
                                        interval_method
                                    )::text,
                                    'sha256'
                                ),
                                'hex'
                            )
                        )::bit(256) AS row_digest,
                        coverage_status
                 FROM customer_bottom_up_blend_component
                 WHERE run_id = %s::uuid
             ), counted_rows AS (
                 SELECT canonical_rows.*,
                        ROW_NUMBER() OVER (
                            PARTITION BY item_id, loc
                            ORDER BY forecast_month
                        ) = 1 AS first_dfu_row
                 FROM canonical_rows
             ), aggregate_stats AS (
                 SELECT COALESCE(
                            BIT_XOR(row_digest),
                            B'0'::bit(256)
                        ) AS payload_digest,
                        COUNT(*)::bigint AS row_count,
                        COUNT(*) FILTER (WHERE first_dfu_row)::bigint AS dfu_count,
                        COUNT(*) FILTER (
                            WHERE coverage_status = 'blended'
                        )::bigint AS blended_count,
                        COUNT(*) FILTER (
                            WHERE coverage_status <> 'blended'
                        )::bigint AS fallback_count
                 FROM counted_rows
             )
             SELECT ENCODE(
                        DIGEST(
                            jsonb_build_array(
                                'xor256-v1', payload_digest::text,
                                row_count, dfu_count,
                                blended_count, fallback_count
                            )::text,
                            'sha256'
                        ),
                        'hex'
                    ),
                    row_count::integer,
                    dfu_count::integer,
                    blended_count::integer,
                    fallback_count::integer
             FROM aggregate_stats""",
        (str(run_id),),
    )
    row = cur.fetchone()
    if row is None:
        raise ValueError("Customer blend checksum query returned no result")
    return (
        str(row[0]),
        int(row[1] or 0),
        int(row[2] or 0),
        int(row[3] or 0),
        int(row[4] or 0),
    )


def _count_excluded_customer_dfus(
    cur: Any,
    *,
    customer_run_id: UUID,
    source_production_run_id: UUID,
) -> int:
    cur.execute(
        """SELECT COUNT(*)::integer
           FROM (
               SELECT DISTINCT customer.item_id, customer.location_id
               FROM fact_customer_forecast customer
               WHERE customer.run_id = %s::uuid
               EXCEPT
               SELECT DISTINCT production.item_id, production.loc
               FROM fact_production_forecast production
               WHERE production.run_id = %s::uuid
           ) excluded""",
        (str(customer_run_id), str(source_production_run_id)),
    )
    row = cur.fetchone()
    return int(row[0] or 0) if row else 0


def load_customer_blend_component_lineage(
    cur: Any,
    run_id: UUID | str,
) -> CustomerBlendComponentLineage:
    """Load the one normalized source identity shared by every blend component."""
    cur.execute(
        """SELECT customer_run_id, backtest_run_id, source_promotion_id,
                  source_production_run_id
           FROM customer_bottom_up_blend_component
           WHERE run_id = %s::uuid
           GROUP BY customer_run_id, backtest_run_id, source_promotion_id,
                    source_production_run_id
           ORDER BY customer_run_id, backtest_run_id, source_promotion_id,
                    source_production_run_id
           LIMIT 2""",
        (str(run_id),),
    )
    rows = cur.fetchall()
    if len(rows) != 1:
        raise ValueError("Customer blend components do not share one source lineage")
    row = rows[0]
    return CustomerBlendComponentLineage(
        customer_run_id=UUID(str(row[0])),
        backtest_run_id=UUID(str(row[1])),
        source_promotion_id=int(row[2]),
        source_production_run_id=UUID(str(row[3])),
    )


@customer_demand_snapshot_locked
def generate_customer_bottom_up_blend(
    conn: Any,
    *,
    run_id: UUID,
    customer_run_id: UUID | None = None,
) -> CustomerBlendGenerationResult:
    """Aggregate, normalize, and blend one immutable release draft in SQL."""
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtext('customer_bottom_up_blend_generation'))"
            )
            readiness = load_customer_blend_readiness(
                conn,
                customer_run_id,
                require_backtest=True,
                verify_evidence=True,
            )
            if not readiness["ready"]:
                raise ValueError(str(readiness["blockers"][0]))
            settings: CustomerBlendSettings = readiness["settings"]
            resolved_customer_run_id = UUID(str(readiness["customer_run_id"]))
            source_promotion_id = int(readiness["source_promotion_id"])
            source_run_id = UUID(str(readiness["source_run_id"]))
            source_production_run_id = UUID(str(readiness["source_production_run_id"]))
            planning_month: date = readiness["planning_month"]
            horizon_months = int(readiness["source_horizon_months"])

            cur.execute(
                """SELECT promotion.id
                   FROM model_promotion_log promotion
                   JOIN forecast_generation_run generation
                     ON generation.run_id = promotion.source_run_id
                   WHERE promotion.id = %s
                     AND promotion.is_active = TRUE
                     AND promotion.model_id = 'champion'
                     AND promotion.source_run_id = %s::uuid
                     AND promotion.production_run_id = %s::uuid
                   FOR SHARE OF promotion, generation""",
                (source_promotion_id, str(source_run_id), str(source_production_run_id)),
            )
            if cur.fetchone() is None:
                raise ValueError("The active champion changed before blend generation")

            source_stats = compute_production_payload_stats(cur, source_production_run_id)
            if source_stats.checksum != readiness["source_production_checksum"]:
                raise ValueError("The active champion payload no longer matches its promotion")

            source_metadata = dict(readiness["source_metadata"])
            initial_lineage = {
                **settings.as_lineage(),
                "config_checksum": customer_blend_config_checksum(settings),
                "status": "backtest_qualified_candidate",
                "customer_run_id": str(resolved_customer_run_id),
                "customer_config_checksum": readiness["customer_config_checksum"],
                "customer_source_checksum": readiness["customer_source_checksum"],
                "source_customer_demand_batch_id": readiness["source_customer_demand_batch_id"],
                "customer_output_checksum": readiness["customer_output_checksum"],
                "customer_output_row_count": readiness["customer_output_row_count"],
                "customer_output_series_count": readiness["customer_output_series_count"],
                "source_promotion_id": source_promotion_id,
                "source_run_id": str(source_run_id),
                "source_production_run_id": str(source_production_run_id),
                "source_production_checksum": source_stats.checksum,
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
            source_metadata[CUSTOMER_BLEND_LINEAGE_METADATA_KEY] = initial_lineage
            status = reserve_generation_run(
                cur,
                run_id=run_id,
                generation_purpose="release_candidate",
                requested_model_id=CHAMPION_MODEL_ID,
                record_month=planning_month,
                horizon_months=horizon_months,
                created_by="customer-bottom-up-blend",
                metadata=source_metadata,
            )
            if status != "generating":
                raise ValueError(f"Customer blend run is already {status}")
            _validate_reserved_customer_blend_lineage(cur, run_id, initial_lineage)
            cur.execute(
                """UPDATE forecast_generation_run
                   SET champion_experiment_id = %s,
                       cluster_experiment_id = %s,
                       source_sales_batch_id = %s,
                       routing_artifact_checksum = %s,
                       champion_results_checksum = %s
                   WHERE run_id = %s::uuid AND run_status = 'generating'""",
                (
                    readiness["champion_experiment_id"],
                    readiness["cluster_experiment_id"],
                    readiness["source_sales_batch_id"],
                    readiness["routing_artifact_checksum"],
                    readiness["champion_results_checksum"],
                    str(run_id),
                ),
            )
            if cur.rowcount != 1:
                raise ValueError("Customer blend manifest could not accept source lineage")

            normalization_start = _shift_month(
                planning_month,
                -settings.normalization_lookback_months,
            )
            history_end = readiness["history_end"]
            cur.execute(
                """WITH customer_aggregate AS (
                         SELECT item_id, location_id AS loc, forecast_month,
                                SUM(forecast_qty)::numeric AS raw_customer_demand_qty,
                                COUNT(*)::integer AS customer_series_count
                         FROM fact_customer_forecast
                         WHERE run_id = %s::uuid
                         GROUP BY item_id, location_id, forecast_month
                     ), fulfillment AS (
                         SELECT item_id, location_id AS loc,
                                CASE
                                    WHEN SUM(demand_qty) >= %s
                                    THEN LEAST(
                                        %s::numeric,
                                        GREATEST(
                                            %s::numeric,
                                            SUM(sales_qty) / NULLIF(SUM(demand_qty), 0)
                                        )
                                    )
                                END AS fulfillment_ratio
                         FROM fact_customer_demand_monthly
                         WHERE startdate >= %s AND startdate <= %s
                         GROUP BY item_id, location_id
                     ), prepared AS (
                         SELECT production.item_id,
                                production.loc,
                                production.forecast_month,
                                production.forecast_qty::numeric AS champion_qty,
                                production.forecast_qty_lower::numeric AS champion_lower,
                                production.forecast_qty_upper::numeric AS champion_upper,
                                production.cluster_id,
                                production.horizon_months,
                                production.is_recursive,
                                production.lag_source,
                                customer.raw_customer_demand_qty,
                                customer.customer_series_count,
                                fulfillment.fulfillment_ratio,
                                CASE
                                    WHEN customer.raw_customer_demand_qty IS NOT NULL
                                     AND fulfillment.fulfillment_ratio IS NOT NULL
                                    THEN customer.raw_customer_demand_qty
                                         * fulfillment.fulfillment_ratio
                                END AS normalized_customer_qty
                         FROM fact_production_forecast production
                         LEFT JOIN customer_aggregate customer
                           ON customer.item_id = production.item_id
                          AND customer.loc = production.loc
                          AND customer.forecast_month = production.forecast_month
                         LEFT JOIN fulfillment
                           ON fulfillment.item_id = production.item_id
                          AND fulfillment.loc = production.loc
                         WHERE production.run_id = %s::uuid
                     ), derived AS (
                         SELECT prepared.*,
                                CASE
                                    WHEN normalized_customer_qty IS NOT NULL
                                    THEN %s::numeric * normalized_customer_qty
                                         + %s::numeric * champion_qty
                                    ELSE champion_qty
                                END AS blended_qty,
                                CASE
                                    WHEN normalized_customer_qty IS NOT NULL THEN 'blended'
                                    ELSE 'champion_fallback'
                                END AS coverage_status,
                                CASE WHEN normalized_customer_qty IS NOT NULL
                                     THEN %s::numeric ELSE 0::numeric END
                                    AS effective_customer_weight
                         FROM prepared
                     )
                     INSERT INTO customer_bottom_up_blend_component
                         (run_id, customer_run_id, backtest_run_id,
                          source_promotion_id,
                          source_production_run_id, item_id, loc, forecast_month,
                          raw_customer_demand_qty, normalized_customer_qty,
                          champion_qty, blended_qty, blended_lower, blended_upper,
                          fulfillment_ratio, customer_weight, champion_weight,
                          effective_customer_weight, customer_series_count,
                          coverage_status, interval_method, generated_at)
                     SELECT %s::uuid, %s::uuid, %s::uuid, %s, %s::uuid,
                            item_id, loc, forecast_month,
                            raw_customer_demand_qty, normalized_customer_qty,
                            champion_qty, blended_qty,
                            CASE
                                WHEN champion_lower IS NOT NULL
                                 AND champion_upper IS NOT NULL
                                THEN GREATEST(0, blended_qty - (champion_qty - champion_lower))
                            END,
                            CASE
                                WHEN champion_lower IS NOT NULL
                                 AND champion_upper IS NOT NULL
                                THEN blended_qty + (champion_upper - champion_qty)
                            END,
                            fulfillment_ratio, %s::numeric, %s::numeric,
                            effective_customer_weight,
                            COALESCE(customer_series_count, 0), coverage_status,
                            CASE
                                WHEN champion_lower IS NULL OR champion_upper IS NULL THEN 'none'
                                WHEN normalized_customer_qty IS NOT NULL
                                    THEN 'champion_width_shift'
                                ELSE 'champion_passthrough'
                            END,
                            NOW()
                     FROM derived""",
                (
                    str(resolved_customer_run_id),
                    str(settings.normalization_min_demand_qty),
                    str(settings.normalization_max_ratio),
                    str(settings.normalization_min_ratio),
                    normalization_start,
                    history_end,
                    str(source_production_run_id),
                    str(settings.customer_weight),
                    str(settings.champion_weight),
                    str(settings.customer_weight),
                    str(run_id),
                    str(resolved_customer_run_id),
                    readiness["backtest_run_id"],
                    source_promotion_id,
                    str(source_production_run_id),
                    str(settings.customer_weight),
                    str(settings.champion_weight),
                ),
            )

            component_checksum, row_count, dfu_count, blended_count, fallback_count = (
                compute_customer_blend_component_stats(cur, run_id)
            )
            if row_count <= 0 or dfu_count <= 0:
                raise ValueError("Customer blend generation produced no champion population")
            excluded_customer_dfus = _count_excluded_customer_dfus(
                cur,
                customer_run_id=resolved_customer_run_id,
                source_production_run_id=source_production_run_id,
            )
            cur.execute(
                """INSERT INTO fact_production_forecast_staging
                       (model_id, candidate_model_id, generation_purpose,
                        item_id, loc, forecast_month, forecast_month_generated,
                        forecast_qty, forecast_qty_lower, forecast_qty_upper,
                        cluster_id, horizon_months, is_recursive, lag_source,
                        generated_at, run_id)
                   SELECT %s, 'champion', 'release_candidate',
                          component.item_id, component.loc, component.forecast_month,
                          %s, ROUND(component.blended_qty, 2),
                          ROUND(component.blended_lower, 2),
                          ROUND(component.blended_upper, 2),
                          production.cluster_id, production.horizon_months,
                          production.is_recursive, production.lag_source,
                          component.generated_at, component.run_id
                   FROM customer_bottom_up_blend_component component
                   JOIN fact_production_forecast production
                     ON production.run_id = component.source_production_run_id
                    AND production.item_id = component.item_id
                    AND production.loc = component.loc
                    AND production.forecast_month = component.forecast_month
                   WHERE component.run_id = %s::uuid""",
                (
                    CUSTOMER_BOTTOM_UP_BLEND_MODEL_ID,
                    planning_month,
                    str(run_id),
                ),
            )
            staging_stats: ForecastPayloadStats = compute_staging_payload_stats(cur, run_id)
            if staging_stats.row_count != row_count or staging_stats.dfu_count != dfu_count:
                raise ValueError("Customer blend staging payload does not match its components")

            shadow = stage_customer_bottom_up_shadow(
                cur,
                blend_run_id=run_id,
                planning_month=planning_month,
                horizon_months=int(readiness["customer_horizon_months"]),
                source_metadata=source_metadata,
                lineage={
                    **initial_lineage,
                    "component_checksum": component_checksum,
                    "component_row_count": row_count,
                    "component_dfu_count": dfu_count,
                },
                champion_experiment_id=int(readiness["champion_experiment_id"]),
                cluster_experiment_id=int(readiness["cluster_experiment_id"]),
                source_sales_batch_id=int(readiness["source_sales_batch_id"]),
                routing_artifact_checksum=str(readiness["routing_artifact_checksum"]),
                champion_results_checksum=str(readiness["champion_results_checksum"]),
            )

            completed_lineage = {
                **initial_lineage,
                "component_checksum": component_checksum,
                "row_count": row_count,
                "dfu_count": dfu_count,
                "blended_row_count": blended_count,
                "fallback_row_count": fallback_count,
                "excluded_customer_dfu_count": excluded_customer_dfus,
                "bottom_up_staging_run_id": str(shadow.run_id),
                "bottom_up_staging_status": shadow.status,
                "bottom_up_staging_row_count": shadow.stats.row_count,
                "bottom_up_staging_dfu_count": shadow.stats.dfu_count,
                "bottom_up_staging_checksum": shadow.stats.checksum,
            }
            source_metadata[CUSTOMER_BLEND_LINEAGE_METADATA_KEY] = completed_lineage
            cur.execute(
                """UPDATE forecast_generation_run
                   SET run_status = 'ready', promotion_eligible = FALSE,
                       row_count = %s, dfu_count = %s,
                       candidate_model_count = %s,
                       artifact_checksum = %s, metadata = %s,
                       completed_at = NOW()
                   WHERE run_id = %s::uuid AND run_status = 'generating'""",
                (
                    staging_stats.row_count,
                    staging_stats.dfu_count,
                    staging_stats.source_model_count,
                    staging_stats.checksum,
                    Jsonb(source_metadata),
                    str(run_id),
                ),
            )
            if cur.rowcount != 1:
                raise ValueError("Customer blend manifest did not transition to ready")

    return CustomerBlendGenerationResult(
        run_id=run_id,
        bottom_up_staging_run_id=shadow.run_id,
        customer_run_id=resolved_customer_run_id,
        source_promotion_id=source_promotion_id,
        source_production_run_id=source_production_run_id,
        row_count=row_count,
        dfu_count=dfu_count,
        blended_row_count=blended_count,
        fallback_row_count=fallback_count,
        excluded_customer_dfu_count=excluded_customer_dfus,
        checksum=staging_stats.checksum,
    )
