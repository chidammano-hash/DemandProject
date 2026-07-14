"""Immutable evidence helpers for customer-level forecast payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class CustomerForecastPayloadStats:
    """Canonical identity for one completed customer forecast payload."""

    checksum: str
    row_count: int
    series_count: int


def compute_customer_forecast_output_stats(
    cur: Any,
    run_id: UUID | str,
) -> CustomerForecastPayloadStats:
    """Hash one completed customer forecast as a commutative row-digest multiset."""
    cur.execute(
        """WITH canonical_rows AS (
                 SELECT item_id, location_id, customer_no, forecast_month,
                        (
                            'x' || ENCODE(
                                DIGEST(
                                    jsonb_build_array(
                                        run_id, batch_id, item_id, location_id,
                                        customer_no,
                                        TO_CHAR(forecast_month, 'YYYY-MM-DD'),
                                        forecast_qty, lower_bound, upper_bound,
                                        model_id,
                                        TO_CHAR(history_end, 'YYYY-MM-DD'),
                                        TO_CHAR(
                                            generated_at AT TIME ZONE 'UTC',
                                            'YYYY-MM-DD"T"HH24:MI:SS.US'
                                        )
                                    )::text,
                                    'sha256'
                                ),
                                'hex'
                            )
                        )::bit(256) AS row_digest
                 FROM fact_customer_forecast
                 WHERE run_id = %s::uuid
             ), counted_rows AS (
                 SELECT canonical_rows.*,
                        ROW_NUMBER() OVER (
                            PARTITION BY item_id, location_id, customer_no
                            ORDER BY forecast_month
                        ) = 1 AS first_series_row
                 FROM canonical_rows
             ), aggregate_stats AS (
                 SELECT COALESCE(
                            BIT_XOR(row_digest),
                            B'0'::bit(256)
                        ) AS payload_digest,
                        COUNT(*)::bigint AS row_count,
                        COUNT(*) FILTER (WHERE first_series_row)::bigint
                            AS series_count
                 FROM counted_rows
             )
             SELECT ENCODE(
                        DIGEST(
                            jsonb_build_array(
                                'xor256-v1', payload_digest::text,
                                row_count, series_count
                            )::text,
                            'sha256'
                        ),
                        'hex'
                    ),
                    row_count::integer,
                    series_count::integer
             FROM aggregate_stats""",
        (str(run_id),),
    )
    row = cur.fetchone()
    if row is None:
        raise ValueError("Customer forecast output checksum is unavailable")
    return CustomerForecastPayloadStats(
        checksum=str(row[0]),
        row_count=int(row[1] or 0),
        series_count=int(row[2] or 0),
    )
