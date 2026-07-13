"""Shared SQL contract for the production forecast population."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

FORECAST_SALES_TABLE = "fact_sales_monthly"
FORECAST_ORIGINAL_SALES_TABLE = "fact_sales_monthly_original"
_SUPPORTED_SALES_TABLES = frozenset(
    {FORECAST_SALES_TABLE, FORECAST_ORIGINAL_SALES_TABLE}
)


@dataclass(frozen=True)
class ForecastEligibilityCtes:
    """Composable eligibility CTEs and their parameter tuple."""

    sql: str
    params: tuple[Any, ...]


def build_forecast_eligibility_ctes(
    *,
    planning_month: date,
    min_history_months: int,
    active_window_months: int,
    item_id: str | None = None,
    loc: str | None = None,
    sales_table: str = FORECAST_SALES_TABLE,
) -> ForecastEligibilityCtes:
    """Build the one population definition shared by generation and promotion.

    Production is published at item/location grain but forecast inference runs
    at item/customer-group/location grain.  An item/location is eligible only
    when *every* active constituent customer group has enough history.  That
    prevents a candidate from appearing complete after silently dropping one
    group's demand.
    """
    if min_history_months <= 0:
        raise ValueError("min_history_months must be positive")
    if active_window_months <= 0:
        raise ValueError("active_window_months must be positive")
    if sales_table not in _SUPPORTED_SALES_TABLES:
        raise ValueError(f"Unsupported forecast sales table: {sales_table}")

    history_filters = [
        "sales.type = 1",
        "sales.qty IS NOT NULL",
        "sales.startdate < %s",
    ]
    history_params: list[Any] = [planning_month]
    if item_id is not None:
        history_filters.append("sales.item_id = %s")
        history_params.append(item_id)
    if loc is not None:
        history_filters.append("sales.loc = %s")
        history_params.append(loc)

    where_sql = " AND ".join(history_filters)
    ctes = f"""
        group_history AS (
            SELECT sales.item_id, sales.customer_group, sales.loc,
                   COUNT(DISTINCT sales.startdate) AS history_months,
                   MAX(sales.startdate) AS last_sale_month
            FROM {sales_table} sales
            WHERE {where_sql}
            GROUP BY sales.item_id, sales.customer_group, sales.loc
        ), active_customer_groups AS (
            SELECT history.item_id, history.customer_group, history.loc,
                   history.history_months
            FROM group_history history
            WHERE history.last_sale_month
                  >= %s::date - (%s * INTERVAL '1 month')
        ), eligible_item_locations AS (
            SELECT active.item_id, active.loc
            FROM active_customer_groups active
            GROUP BY active.item_id, active.loc
            HAVING BOOL_AND(
                active.history_months >= %s
                AND active.customer_group IS NOT NULL
                AND BTRIM(active.customer_group) <> ''
            )
        )
    """
    params = (
        *history_params,
        planning_month,
        active_window_months,
        min_history_months,
    )
    return ForecastEligibilityCtes(sql=ctes, params=params)


def resolve_forecast_sales_table(cursor: Any) -> str:
    """Require the immutable raw-sales mirror for the latest completed batch."""
    cursor.execute("SELECT to_regclass('fact_sales_monthly_original')")
    relation = cursor.fetchone()
    if not relation or relation[0] is None:
        raise RuntimeError(
            "The immutable original sales table is missing; apply the sales schema "
            "and run one canonical sales reload"
        )
    cursor.execute(
        """SELECT
               (SELECT COUNT(*) FROM fact_sales_monthly_original),
               (SELECT MAX(load_ts) FROM fact_sales_monthly_original),
               (SELECT row_count_out
                FROM audit_load_batch
                WHERE domain = 'sales' AND status = 'completed'
                  AND row_count_out > 0
                ORDER BY completed_at DESC NULLS LAST, batch_id DESC
                LIMIT 1),
               (SELECT started_at
                FROM audit_load_batch
                WHERE domain = 'sales' AND status = 'completed'
                  AND row_count_out > 0
                ORDER BY completed_at DESC NULLS LAST, batch_id DESC
                LIMIT 1),
               (SELECT source_file
                FROM audit_load_batch
                WHERE domain = 'sales' AND status = 'completed'
                  AND row_count_out > 0
                ORDER BY completed_at DESC NULLS LAST, batch_id DESC
                LIMIT 1)"""
    )
    state = cursor.fetchone()
    if not state or state[2] is None or state[3] is None:
        raise RuntimeError("No completed sales batch can prove the forecast source")
    original_rows, original_loaded_at, batch_rows, batch_started_at, source_file = state
    if not source_file or str(source_file).strip() == "safe_upsert":
        raise RuntimeError(
            "The latest completed sales batch did not synchronize the immutable "
            "forecast source; run one canonical sales reload"
        )
    if int(original_rows or 0) <= 0 or original_loaded_at is None:
        raise RuntimeError(
            "The immutable original sales table is empty; run one canonical sales reload"
        )
    if int(original_rows) != int(batch_rows):
        raise RuntimeError(
            "The immutable original sales row count does not match the latest completed batch"
        )
    # Rows are inserted before complete_batch() stamps completed_at, so compare
    # their load timestamp with the batch start rather than its later completion.
    if original_loaded_at < batch_started_at:
        raise RuntimeError(
            "The immutable original sales mirror is older than the latest completed batch"
        )
    return FORECAST_ORIGINAL_SALES_TABLE
