"""Deterministic lineage evidence for generated and promoted forecasts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class ForecastPayloadStats:
    """Canonical checksum and cardinalities for one immutable forecast payload."""

    checksum: str
    row_count: int
    dfu_count: int
    source_model_count: int


_STAGING_PAYLOAD_SQL = """WITH canonical_rows AS (
    SELECT item_id,
           loc,
           forecast_month,
           model_id AS source_model_id,
           jsonb_build_array(
               item_id,
               loc,
               TO_CHAR(forecast_month, 'YYYY-MM-DD'),
               forecast_qty,
               forecast_qty_lower,
               forecast_qty_upper,
               model_id,
               cluster_id,
               horizon_months,
               is_recursive,
               lag_source,
               TO_CHAR(
                   generated_at AT TIME ZONE 'UTC',
                   'YYYY-MM-DD"T"HH24:MI:SS.US'
               )
           )::text AS canonical_row
    FROM fact_production_forecast_staging
    WHERE run_id = %s::uuid
      AND (%s::date IS NULL OR forecast_month >= %s::date)
      AND (%s::date IS NULL OR forecast_month < %s::date)
)
SELECT ENCODE(
           DIGEST(
               COALESCE(
                   STRING_AGG(
                       canonical_row,
                       E'\n'
                       ORDER BY item_id, loc, forecast_month, source_model_id
                   ),
                   ''
               ),
               'sha256'
           ),
           'hex'
       ),
       COUNT(*)::integer,
       COUNT(DISTINCT (item_id, loc))::integer,
       COUNT(DISTINCT source_model_id)::integer
FROM canonical_rows"""


_PRODUCTION_PAYLOAD_SQL = """WITH canonical_rows AS (
    SELECT item_id,
           loc,
           forecast_month,
           source_model_id,
           jsonb_build_array(
               item_id,
               loc,
               TO_CHAR(forecast_month, 'YYYY-MM-DD'),
               forecast_qty,
               forecast_qty_lower,
               forecast_qty_upper,
               source_model_id,
               cluster_id,
               horizon_months,
               is_recursive,
               lag_source,
               TO_CHAR(
                   generated_at AT TIME ZONE 'UTC',
                   'YYYY-MM-DD"T"HH24:MI:SS.US'
               )
           )::text AS canonical_row
    FROM fact_production_forecast
    WHERE run_id = %s::uuid
      AND (%s::date IS NULL OR forecast_month >= %s::date)
      AND (%s::date IS NULL OR forecast_month < %s::date)
)
SELECT ENCODE(
           DIGEST(
               COALESCE(
                   STRING_AGG(
                       canonical_row,
                       E'\n'
                       ORDER BY item_id, loc, forecast_month, source_model_id
                   ),
                   ''
               ),
               'sha256'
           ),
           'hex'
       ),
       COUNT(*)::integer,
       COUNT(DISTINCT (item_id, loc))::integer,
       COUNT(DISTINCT source_model_id)::integer
FROM canonical_rows"""


_SNAPSHOT_PAYLOAD_SQL = """WITH canonical_rows AS (
    SELECT item_id,
           loc,
           forecast_month,
           source_model_id,
           jsonb_build_array(
               item_id,
               loc,
               TO_CHAR(forecast_month, 'YYYY-MM-DD'),
               forecast_qty,
               forecast_qty_lower,
               forecast_qty_upper,
               source_model_id,
               cluster_id,
               horizon_months,
               is_recursive,
               lag_source,
               TO_CHAR(
                   generated_at AT TIME ZONE 'UTC',
                   'YYYY-MM-DD"T"HH24:MI:SS.US'
               )
           )::text AS canonical_row
    FROM fact_forecast_snapshot
    WHERE record_month = %s
      AND model_id = 'champion'
      AND run_id = %s::uuid
)
SELECT ENCODE(
           DIGEST(
               COALESCE(
                   STRING_AGG(
                       canonical_row,
                       E'\n'
                       ORDER BY item_id, loc, forecast_month, source_model_id
                   ),
                   ''
               ),
               'sha256'
           ),
           'hex'
       ),
       COUNT(*)::integer,
       COUNT(DISTINCT (item_id, loc))::integer,
       COUNT(DISTINCT source_model_id)::integer
FROM canonical_rows"""


_CHAMPION_RESULTS_SQL = """WITH canonical_rows AS (
    SELECT item_id,
           customer_group,
           loc,
           startdate,
           lag,
           source_model_id,
           jsonb_build_array(
               item_id,
               customer_group,
               loc,
               TO_CHAR(startdate, 'YYYY-MM-DD'),
               lag,
               execution_lag,
               basefcst_pref,
               tothist_dmd,
               source_model_id
           )::text AS canonical_row
    FROM fact_external_forecast_monthly
    WHERE model_id = 'champion'
      AND champion_experiment_id = %s
)
SELECT ENCODE(
           DIGEST(
               COALESCE(
                   STRING_AGG(
                       canonical_row,
                       E'\n'
                       ORDER BY item_id, customer_group, loc, startdate, lag,
                                source_model_id
                   ),
                   ''
               ),
               'sha256'
           ),
           'hex'
       ),
       COUNT(*)::integer,
       COUNT(DISTINCT (item_id, customer_group, loc))::integer,
       COUNT(DISTINCT source_model_id)::integer
FROM canonical_rows"""


def _payload_params(
    run_id: UUID | str,
    start_month: date | None,
    end_month: date | None,
) -> tuple[Any, ...]:
    return (str(run_id), start_month, start_month, end_month, end_month)


def _stats_from_row(row: tuple[Any, ...] | None) -> ForecastPayloadStats:
    if row is None:
        raise ValueError("forecast checksum query returned no result")
    return ForecastPayloadStats(
        checksum=str(row[0]),
        row_count=int(row[1] or 0),
        dfu_count=int(row[2] or 0),
        source_model_count=int(row[3] or 0),
    )


def compute_staging_payload_stats(
    cur: Any,
    run_id: UUID | str,
    *,
    start_month: date | None = None,
    end_month: date | None = None,
) -> ForecastPayloadStats:
    """Hash one source run in stable business-key order."""
    cur.execute(
        _STAGING_PAYLOAD_SQL,
        _payload_params(run_id, start_month, end_month),
    )
    return _stats_from_row(cur.fetchone())


def compute_production_payload_stats(
    cur: Any,
    production_run_id: UUID | str,
    *,
    start_month: date | None = None,
    end_month: date | None = None,
) -> ForecastPayloadStats:
    """Hash one published run using the same canonical payload as staging."""
    cur.execute(
        _PRODUCTION_PAYLOAD_SQL,
        _payload_params(production_run_id, start_month, end_month),
    )
    return _stats_from_row(cur.fetchone())


def compute_snapshot_champion_payload_stats(
    cur: Any,
    *,
    record_month: date,
    production_run_id: UUID | str,
) -> ForecastPayloadStats:
    """Hash the archived champion rows tied to one outgoing release run."""
    cur.execute(
        _SNAPSHOT_PAYLOAD_SQL,
        (record_month, str(production_run_id)),
    )
    return _stats_from_row(cur.fetchone())


def compute_champion_results_stats(
    cur: Any,
    champion_experiment_id: int,
) -> ForecastPayloadStats:
    """Hash historical champion evidence stamped by one results promotion."""
    cur.execute(_CHAMPION_RESULTS_SQL, (champion_experiment_id,))
    return _stats_from_row(cur.fetchone())


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of exact artifact bytes without loading the file at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
