"""Rolling customer rule-router backtests and warehouse-item blend evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any
from uuid import UUID

import pandas as pd
from psycopg import sql
from psycopg.types.json import Jsonb

from common.core.constants import FORECAST_QTY_COL
from common.core.sql_helpers import read_sql_chunked
from common.core.utils import load_forecast_pipeline_config
from common.services import customer_forecast_backtest_accuracy as backtest_accuracy
from common.services.customer_demand_lineage import customer_demand_snapshot_locked
from common.services.customer_forecast import (
    CustomerForecastWindow,
    _shift_month,
    get_customer_forecast_settings,
)
from common.services.customer_forecast_backtest_population import (
    compute_customer_backtest_source_population,
)
from common.services.customer_forecast_backtest_rules import (
    build_customer_rule_backtest_batch,
)
from common.services.customer_forecast_blend_contract import (
    CustomerBlendSettings,
    get_customer_blend_settings,
)
from common.services.customer_forecast_blend_readiness import load_customer_blend_readiness


@dataclass(frozen=True)
class CustomerBacktestSettings:
    enabled: bool
    lookback_months: int
    min_train_months: int
    horizon_months: int
    batch_size: int
    min_common_months: int
    min_common_dfus: int
    max_wape_degradation_pct: float

    def as_lineage(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "lookback_months": self.lookback_months,
            "min_train_months": self.min_train_months,
            "horizon_months": self.horizon_months,
            "batch_size": self.batch_size,
            "min_common_months": self.min_common_months,
            "min_common_dfus": self.min_common_dfus,
            "max_wape_degradation_pct": self.max_wape_degradation_pct,
        }


@dataclass(frozen=True)
class CustomerBacktestResult:
    run_id: UUID
    customer_run_id: UUID
    component_checksum: str
    component_rows: int
    comparison: backtest_accuracy.BacktestAccuracyComparison


def get_customer_backtest_settings() -> CustomerBacktestSettings:
    config = load_forecast_pipeline_config()
    try:
        raw = config["customer_forecast"]["backtest"]
        thresholds = raw["promotion_thresholds"]
        settings = CustomerBacktestSettings(
            enabled=raw["enabled"],
            lookback_months=int(raw["lookback_months"]),
            min_train_months=int(raw["min_train_months"]),
            horizon_months=int(raw["horizon_months"]),
            batch_size=int(raw["batch_size"]),
            min_common_months=int(thresholds["min_common_months"]),
            min_common_dfus=int(thresholds["min_common_dfus"]),
            max_wape_degradation_pct=float(thresholds["max_wape_degradation_pct"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Customer forecast backtest settings are incomplete") from exc
    if not isinstance(settings.enabled, bool):
        raise ValueError("Customer forecast backtest enabled setting must be boolean")
    if (
        settings.lookback_months <= 0
        or settings.min_train_months <= 0
        or settings.horizon_months != 1
        or settings.batch_size <= 0
        or settings.min_common_months <= 0
        or settings.min_common_dfus <= 0
        or settings.max_wape_degradation_pct != 0
    ):
        raise ValueError("Customer forecast backtest settings are invalid")
    customer = get_customer_forecast_settings()
    if settings.batch_size != int(customer["batch_size"]):
        raise ValueError("Customer backtest batch size must match customer forecasting")
    if settings.min_train_months + settings.lookback_months > int(customer["history_months"]):
        raise ValueError("Customer backtest needs more history than customer forecasting provides")
    return settings


def customer_backtest_config_checksum(
    settings: CustomerBacktestSettings,
    blend: CustomerBlendSettings,
    customer_settings: dict[str, Any],
) -> str:
    payload = {
        "backtest": settings.as_lineage(),
        "blend": blend.as_lineage(),
        "customer_model_id": customer_settings["model_id"],
        "customer_rule_params": customer_settings["rule_params"],
        "customer_croston_params": customer_settings["croston_params"],
        "recent_sales_lookback_months": customer_settings["recent_sales_lookback_months"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def compute_customer_backtest_component_stats(
    cur: Any,
    run_id: UUID | str,
) -> tuple[str, int]:
    """Hash historical components as a commutative row-digest multiset."""
    cur.execute(
        """WITH canonical_rows AS (
                 SELECT (
                            'x' || ENCODE(
                                DIGEST(
                                    jsonb_build_array(
                                        item_id, loc,
                                        TO_CHAR(forecast_origin, 'YYYY-MM-DD'),
                                        TO_CHAR(forecast_month, 'YYYY-MM-DD'),
                                        raw_customer_demand_qty,
                                        fulfillment_ratio,
                                        normalized_customer_qty, champion_qty,
                                        blended_qty, actual_qty,
                                        customer_weight, champion_weight,
                                        effective_customer_weight,
                                        customer_series_count, coverage_status
                                    )::text,
                                    'sha256'
                                ),
                                'hex'
                            )
                        )::bit(256) AS row_digest
                 FROM customer_bottom_up_backtest_component
                 WHERE backtest_run_id = %s::uuid
             ), aggregate_stats AS (
                 SELECT COALESCE(
                            BIT_XOR(row_digest),
                            B'0'::bit(256)
                        ) AS payload_digest,
                        COUNT(*)::bigint AS row_count
                 FROM canonical_rows
             )
             SELECT ENCODE(
                        DIGEST(
                            jsonb_build_array(
                                'xor256-v1', payload_digest::text, row_count
                            )::text,
                            'sha256'
                        ),
                        'hex'
                    ),
                    row_count::integer
             FROM aggregate_stats""",
        (str(run_id),),
    )
    row = cur.fetchone()
    if row is None:
        raise ValueError("Customer backtest component checksum is unavailable")
    return str(row[0]), int(row[1] or 0)


def _load_origin_causal_batch_history(
    conn: Any,
    *,
    route_batch_no: int,
    window: CustomerForecastWindow,
) -> pd.DataFrame:
    """Load one deterministic all-series partition without current-run selection bias."""
    return read_sql_chunked(
        conn,
        """SELECT series.item_id, series.location_id, series.customer_no,
                  history.startdate,
                  COALESCE(SUM(history.demand_qty), 0)::double precision AS demand_qty,
                  COALESCE(SUM(history.sales_qty), 0)::double precision AS sales_qty,
                  series.first_demand_month AS series_first_demand_month
           FROM temp_customer_backtest_series series
           LEFT JOIN fact_customer_demand_monthly history
             ON history.item_id = series.item_id
            AND history.location_id = series.location_id
            AND history.customer_no = series.customer_no
            AND history.startdate >= %s
            AND history.startdate < %s
           WHERE series.route_batch_no = %s
           GROUP BY series.item_id, series.location_id, series.customer_no,
                    history.startdate, series.first_demand_month
           ORDER BY series.item_id, series.location_id, series.customer_no,
                    history.startdate""",
        params=(window.history_start, window.forecast_start, route_batch_no),
    )


@customer_demand_snapshot_locked
def generate_customer_forecast_backtest(
    conn: Any,
    *,
    run_id: UUID,
    progress_callback: Callable[[int, int], None] | None = None,
) -> CustomerBacktestResult:
    """Build immutable rule-router, champion, and blended common-cohort evidence."""
    settings = get_customer_backtest_settings()
    customer_settings = get_customer_forecast_settings()
    blend_settings = get_customer_blend_settings()
    if not settings.enabled:
        raise ValueError("Customer forecast backtesting is disabled")

    with conn.cursor() as cur:
        cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        cur.execute(
            """SELECT customer_run_id, source_promotion_id,
                      source_production_run_id, planning_month,
                      evaluation_start, evaluation_end, config_checksum,
                      run_status, lookback_months, min_train_months,
                      horizon_months, batch_size, customer_model_id,
                      blend_model_id, source_series_count,
                      source_series_checksum, total_batches
               FROM customer_forecast_backtest_run
               WHERE run_id = %s::uuid""",
            (str(run_id),),
        )
        request = cur.fetchone()
    if request is None or request[7] not in {"queued", "generating", "failed"}:
        raise ValueError("Customer forecast backtest run is missing or not runnable")
    customer_run_id = UUID(str(request[0]))
    readiness = load_customer_blend_readiness(conn, customer_run_id)
    if not readiness["ready"]:
        raise ValueError(str(readiness["blockers"][0]))
    expected_checksum = customer_backtest_config_checksum(
        settings, blend_settings, customer_settings
    )
    if request[6] != expected_checksum:
        raise ValueError("Customer backtest configuration changed after submission")
    expected_evaluation_end = _shift_month(request[3], -1)
    expected_evaluation_start = _shift_month(
        expected_evaluation_end,
        1 - settings.lookback_months,
    )
    stored_contract = (
        request[4],
        request[5],
        int(request[8]),
        int(request[9]),
        int(request[10]),
        int(request[11]),
        str(request[12]),
        str(request[13]),
    )
    expected_contract = (
        expected_evaluation_start,
        expected_evaluation_end,
        settings.lookback_months,
        settings.min_train_months,
        settings.horizon_months,
        settings.batch_size,
        str(customer_settings["model_id"]),
        blend_settings.model_id,
    )
    if (
        stored_contract != expected_contract
        or int(request[14] or 0) <= 0
        or not request[15]
        or int(request[16] or 0) <= 0
    ):
        raise ValueError("Customer backtest request does not match its configured contract")
    if (
        int(request[1]) != int(readiness["source_promotion_id"])
        or UUID(str(request[2])) != UUID(str(readiness["source_production_run_id"]))
        or request[3] != readiness["planning_month"]
    ):
        raise ValueError("Customer backtest source champion changed after submission")

    window = CustomerForecastWindow(
        planning_month=readiness["planning_month"],
        history_start=readiness["history_start"],
        history_end=readiness["history_end"],
        forecast_start=readiness["customer_forecast_start"],
        forecast_end=readiness["customer_forecast_end"],
        history_months=int(customer_settings["history_months"]),
        horizon_months=int(customer_settings["horizon_months"]),
        forecast_months=tuple(
            _shift_month(readiness["planning_month"], offset)
            for offset in range(int(customer_settings["horizon_months"]))
        ),
    )
    evaluation_start: date = request[4]
    evaluation_end: date = request[5]

    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_xact_lock(hashtext('customer_forecast_backtest_generation'))"
        )
        cur.execute(
            """UPDATE customer_forecast_backtest_run
               SET run_status = 'generating', started_at = COALESCE(started_at, NOW()),
                   completed_at = NULL, error_summary = NULL
               WHERE run_id = %s::uuid
                 AND run_status IN ('queued', 'generating', 'failed')""",
            (str(run_id),),
        )
        if cur.rowcount != 1:
            raise ValueError("Customer forecast backtest run could not start")
        cur.execute(
            """SELECT promotion.id
               FROM model_promotion_log promotion
               WHERE promotion.id = %s AND promotion.is_active = TRUE
                 AND promotion.production_run_id = %s::uuid
                 AND promotion.model_id = 'champion'
               FOR SHARE""",
            (int(request[1]), str(request[2])),
        )
        if cur.fetchone() is None:
            raise ValueError("The source champion changed before customer backtesting")
        source_population = compute_customer_backtest_source_population(
            cur,
            planning_month=request[3],
            batch_size=settings.batch_size,
        )
        if (
            source_population.series_count != int(request[14])
            or source_population.checksum != request[15]
            or source_population.batch_count != int(request[16])
        ):
            raise ValueError("Customer backtest source population changed after submission")
        cur.execute(
            """CREATE TEMP TABLE temp_customer_backtest_series ON COMMIT DROP AS
               SELECT profile.item_id, profile.location_id, profile.customer_no,
                      profile.first_demand_month,
                      ((ROW_NUMBER() OVER (
                          ORDER BY profile.item_id, profile.location_id,
                                   profile.customer_no
                      ) - 1) / %s)::integer AS route_batch_no
               FROM mv_customer_demand_series_profile profile
               WHERE profile.first_month < %s""",
            (settings.batch_size, request[3]),
        )
        cur.execute(
            """CREATE INDEX temp_customer_backtest_series_route_idx
               ON temp_customer_backtest_series
                   (route_batch_no, item_id, location_id, customer_no)"""
        )
        cur.execute("ANALYZE temp_customer_backtest_series")
        cur.execute(
            """SELECT COUNT(*)::integer,
                      COUNT(DISTINCT route_batch_no)::integer
               FROM temp_customer_backtest_series"""
        )
        series_count, total_batches = cur.fetchone() or (0, 0)
        if int(series_count or 0) <= 0:
            raise ValueError("Customer backtesting has no historical source population")
        if (
            int(series_count or 0) != source_population.series_count
            or int(total_batches or 0) != source_population.batch_count
        ):
            raise ValueError("Customer backtest source population changed after submission")
        cur.execute(
            """CREATE TEMP TABLE temp_customer_backtest_raw (
                   item_id TEXT NOT NULL,
                   loc TEXT NOT NULL,
                   forecast_origin DATE NOT NULL,
                   forecast_month DATE NOT NULL,
                   raw_customer_demand_qty NUMERIC NOT NULL,
                   customer_series_count INTEGER NOT NULL,
                   PRIMARY KEY (item_id, loc, forecast_origin, forecast_month)
               ) ON COMMIT DROP"""
        )
        cur.execute(
            """CREATE TEMP TABLE temp_customer_backtest_batch (
                   item_id TEXT NOT NULL,
                   loc TEXT NOT NULL,
                   forecast_origin DATE NOT NULL,
                   forecast_month DATE NOT NULL,
                   raw_customer_demand_qty NUMERIC NOT NULL,
                   customer_series_count INTEGER NOT NULL
               ) ON COMMIT DROP"""
        )

        for route_batch_no in range(int(total_batches)):
            batch_history = _load_origin_causal_batch_history(
                conn,
                route_batch_no=route_batch_no,
                window=window,
            )
            rows = build_customer_rule_backtest_batch(
                batch_history,
                window,
                evaluation_months=settings.lookback_months,
                min_train_months=settings.min_train_months,
                recent_sales_lookback_months=int(customer_settings["recent_sales_lookback_months"]),
                rule_params=dict(customer_settings["rule_params"]),
                croston_params=dict(customer_settings["croston_params"]),
            )
            cur.execute("TRUNCATE temp_customer_backtest_batch")
            if not rows.empty:
                with cur.copy(
                    "COPY temp_customer_backtest_batch "
                    "(item_id, loc, forecast_origin, forecast_month, "
                    "raw_customer_demand_qty, customer_series_count) FROM STDIN"
                ) as copy:
                    for row in rows.itertuples(index=False):
                        copy.write_row(tuple(row))
                cur.execute(
                    """INSERT INTO temp_customer_backtest_raw
                           (item_id, loc, forecast_origin, forecast_month,
                            raw_customer_demand_qty, customer_series_count)
                       SELECT item_id, loc, forecast_origin, forecast_month,
                              raw_customer_demand_qty, customer_series_count
                       FROM temp_customer_backtest_batch
                       ON CONFLICT (item_id, loc, forecast_origin, forecast_month)
                       DO UPDATE SET
                           raw_customer_demand_qty =
                               temp_customer_backtest_raw.raw_customer_demand_qty
                               + EXCLUDED.raw_customer_demand_qty,
                           customer_series_count =
                               temp_customer_backtest_raw.customer_series_count
                               + EXCLUDED.customer_series_count"""
                )
            if progress_callback is not None:
                progress_callback(route_batch_no + 1, int(total_batches))

        normalization_start_offset = blend_settings.normalization_lookback_months
        cur.execute(
            sql.SQL(
                """WITH champion AS (
                     SELECT forecast.item_id, forecast.loc,
                            forecast.startdate AS forecast_month,
                            SUM(forecast.{forecast_qty})::numeric AS champion_qty
                     FROM fact_external_forecast_monthly forecast
                     WHERE forecast.model_id = 'champion'
                       AND forecast.champion_experiment_id = %s
                       AND forecast.lag = COALESCE(forecast.execution_lag, 0)
                       AND forecast.startdate >= %s
                       AND forecast.startdate <= %s
                       AND forecast.{forecast_qty} IS NOT NULL
                     GROUP BY forecast.item_id, forecast.loc, forecast.startdate
                 ), actual AS (
                     SELECT item_id, loc, startdate AS forecast_month,
                            SUM(qty)::numeric AS actual_qty
                     FROM fact_sales_monthly
                     WHERE type = 1 AND startdate >= %s AND startdate <= %s
                     GROUP BY item_id, loc, startdate
                 ), fulfillment AS (
                     SELECT raw.item_id, raw.loc, raw.forecast_month,
                            CASE WHEN SUM(history.demand_qty) >= %s
                                 THEN LEAST(%s::numeric, GREATEST(
                                     %s::numeric,
                                     SUM(history.sales_qty)
                                         / NULLIF(SUM(history.demand_qty), 0)
                                 ))
                            END AS fulfillment_ratio
                     FROM temp_customer_backtest_raw raw
                     LEFT JOIN fact_customer_demand_monthly history
                       ON history.item_id = raw.item_id
                      AND history.location_id = raw.loc
                      AND history.startdate >= raw.forecast_month
                          - (%s * INTERVAL '1 month')
                      AND history.startdate < raw.forecast_month
                     GROUP BY raw.item_id, raw.loc, raw.forecast_month
                 ), prepared AS (
                     SELECT champion.item_id, champion.loc,
                            (champion.forecast_month - INTERVAL '1 month')::date
                                AS forecast_origin,
                            champion.forecast_month, raw.raw_customer_demand_qty,
                            fulfillment.fulfillment_ratio,
                            CASE WHEN fulfillment.fulfillment_ratio IS NOT NULL
                                 THEN raw.raw_customer_demand_qty
                                      * fulfillment.fulfillment_ratio END
                                AS normalized_customer_qty,
                            champion.champion_qty, actual.actual_qty,
                            COALESCE(raw.customer_series_count, 0)
                                AS customer_series_count
                     FROM champion
                     JOIN actual USING (item_id, loc, forecast_month)
                     LEFT JOIN temp_customer_backtest_raw raw
                       USING (item_id, loc, forecast_month)
                     LEFT JOIN fulfillment USING (item_id, loc, forecast_month)
                 )
                 INSERT INTO customer_bottom_up_backtest_component
                     (backtest_run_id, item_id, loc, forecast_origin,
                      forecast_month, raw_customer_demand_qty, fulfillment_ratio,
                      normalized_customer_qty, champion_qty, blended_qty,
                      actual_qty, customer_weight, champion_weight,
                      effective_customer_weight, customer_series_count,
                      coverage_status)
                 SELECT %s::uuid, item_id, loc, forecast_origin, forecast_month,
                        raw_customer_demand_qty, fulfillment_ratio,
                        normalized_customer_qty, champion_qty,
                        ROUND(
                            CASE WHEN normalized_customer_qty IS NOT NULL
                                 THEN %s::numeric * normalized_customer_qty
                                      + %s::numeric * champion_qty
                                 ELSE champion_qty END,
                            4
                        ),
                        actual_qty, %s::numeric, %s::numeric,
                        CASE WHEN normalized_customer_qty IS NOT NULL
                             THEN %s::numeric ELSE 0::numeric END,
                        customer_series_count,
                        CASE WHEN normalized_customer_qty IS NOT NULL
                             THEN 'blended' ELSE 'champion_fallback' END
                 FROM prepared"""
            ).format(forecast_qty=sql.Identifier(FORECAST_QTY_COL)),
            (
                readiness["champion_experiment_id"],
                evaluation_start,
                evaluation_end,
                evaluation_start,
                evaluation_end,
                str(blend_settings.normalization_min_demand_qty),
                str(blend_settings.normalization_max_ratio),
                str(blend_settings.normalization_min_ratio),
                normalization_start_offset,
                str(run_id),
                str(blend_settings.customer_weight),
                str(blend_settings.champion_weight),
                str(blend_settings.customer_weight),
                str(blend_settings.champion_weight),
                str(blend_settings.customer_weight),
            ),
        )

        component_checksum, component_rows = compute_customer_backtest_component_stats(
            cur,
            run_id,
        )
        frame = backtest_accuracy.load_backtest_accuracy_frame(cur, run_id)
        comparison = backtest_accuracy.compare_backtest_accuracy(frame, settings)
        backtest_accuracy.persist_backtest_accuracy(
            cur,
            run_id=run_id,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            comparison=comparison,
            settings=settings,
        )
        cur.execute(
            """UPDATE customer_forecast_backtest_run
               SET run_status = 'completed', completed_batches = total_batches,
                   component_rows = %s, component_checksum = %s,
                   completed_at = NOW(), error_summary = NULL,
                   metadata = %s
               WHERE run_id = %s::uuid AND run_status = 'generating'
                 AND EXISTS (
                     SELECT 1
                     FROM customer_forecast_run customer
                     WHERE customer.run_id = %s::uuid
                       AND customer.source_customer_demand_batch_id IS NOT NULL
                       AND customer.source_customer_demand_batch_id = (
                           SELECT source_batch_id
                           FROM customer_demand_profile_refresh_state
                           WHERE singleton_id = 1
                       )
                       AND customer.source_customer_demand_batch_id = (
                           SELECT batch_id
                           FROM audit_load_batch
                           WHERE domain = 'customer_demand'
                             AND status = 'completed'
                           ORDER BY completed_at DESC NULLS LAST, batch_id DESC
                           LIMIT 1
                       )
                 )
                 AND NOT EXISTS (
                     SELECT 1
                     FROM audit_load_batch
                     WHERE domain = 'customer_demand'
                       AND status = 'running'
                 )""",
            (
                component_rows,
                component_checksum,
                Jsonb(
                    {
                        "customer_config_checksum": readiness["customer_config_checksum"],
                        "customer_source_checksum": readiness["customer_source_checksum"],
                        "champion_experiment_id": readiness["champion_experiment_id"],
                        "source_production_checksum": readiness["source_production_checksum"],
                    }
                ),
                str(run_id),
                str(customer_run_id),
            ),
        )
        if cur.rowcount != 1:
            raise ValueError("Customer forecast backtest did not transition to completed")

    return CustomerBacktestResult(
        run_id=run_id,
        customer_run_id=customer_run_id,
        component_checksum=component_checksum,
        component_rows=component_rows,
        comparison=comparison,
    )
