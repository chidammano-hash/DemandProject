"""Immutable source-population identity for customer forecast backtests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class CustomerBacktestSourcePopulation:
    """Exact customer-series membership submitted to one backtest."""

    series_count: int
    batch_count: int
    checksum: str


def compute_customer_backtest_source_population(
    cur: Any,
    *,
    planning_month: date,
    batch_size: int,
) -> CustomerBacktestSourcePopulation:
    """Return a versioned, order-independent identity for the source cohort."""
    if batch_size <= 0:
        raise ValueError("Customer backtest batch size must be positive")
    cur.execute(
        """WITH canonical_rows AS (
                 SELECT (
                            'x' || ENCODE(
                                DIGEST(
                                    jsonb_build_array(
                                        item_id, location_id, customer_no
                                    )::text,
                                    'sha256'
                                ),
                                'hex'
                            )
                        )::bit(256) AS row_digest
                 FROM mv_customer_demand_series_profile
                 WHERE first_month < %s
             ), aggregate_stats AS (
                 SELECT COALESCE(
                            BIT_XOR(row_digest),
                            B'0'::bit(256)
                        ) AS population_digest,
                        COUNT(*)::bigint AS series_count
                 FROM canonical_rows
             )
             SELECT series_count::integer,
                    CEIL(series_count::numeric / %s)::integer,
                    ENCODE(
                        DIGEST(
                            jsonb_build_array(
                                'xor256-v1', population_digest::text,
                                series_count
                            )::text,
                            'sha256'
                        ),
                        'hex'
                    )
             FROM aggregate_stats""",
        (planning_month, batch_size),
    )
    row = cur.fetchone()
    if row is None:
        raise ValueError("Customer backtest source population is unavailable")
    return CustomerBacktestSourcePopulation(
        series_count=int(row[0] or 0),
        batch_count=int(row[1] or 0),
        checksum=str(row[2]),
    )
