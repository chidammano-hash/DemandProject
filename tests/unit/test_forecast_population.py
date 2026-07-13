"""Shared production-forecast population contract tests."""

from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import pytest

from common.services.forecast_population import (
    build_forecast_eligibility_ctes,
    resolve_forecast_sales_table,
)


def test_eligibility_ctes_require_every_active_group_to_meet_history_floor():
    eligibility = build_forecast_eligibility_ctes(
        planning_month=date(2026, 7, 1),
        min_history_months=3,
        active_window_months=12,
    )
    sql = " ".join(eligibility.sql.split())

    assert "group_history AS" in sql
    assert "sales.qty IS NOT NULL" in sql
    assert "GROUP BY sales.item_id, sales.customer_group, sales.loc" in sql
    assert "active_customer_groups AS" in sql
    assert "eligible_item_locations AS" in sql
    assert "BOOL_AND(" in sql
    assert "active.history_months >= %s" in sql
    assert eligibility.params == (date(2026, 7, 1), date(2026, 7, 1), 12, 3)


def test_eligibility_ctes_parameterize_optional_scope_filters():
    eligibility = build_forecast_eligibility_ctes(
        planning_month=date(2026, 7, 1),
        min_history_months=25,
        active_window_months=12,
        item_id="ITEM-1",
        loc="LOC-1",
    )
    sql = " ".join(eligibility.sql.split())

    assert "sales.item_id = %s" in sql
    assert "sales.loc = %s" in sql
    assert "ITEM-1" not in sql
    assert "LOC-1" not in sql
    assert eligibility.params == (
        date(2026, 7, 1),
        "ITEM-1",
        "LOC-1",
        date(2026, 7, 1),
        12,
        25,
    )


def test_eligibility_ctes_use_validated_canonical_sales_source():
    eligibility = build_forecast_eligibility_ctes(
        planning_month=date(2026, 7, 1),
        min_history_months=3,
        active_window_months=12,
        sales_table="fact_sales_monthly_original",
    )

    assert "FROM fact_sales_monthly_original sales" in eligibility.sql
    with pytest.raises(ValueError, match="Unsupported forecast sales table"):
        build_forecast_eligibility_ctes(
            planning_month=date(2026, 7, 1),
            min_history_months=3,
            active_window_months=12,
            sales_table="fact_sales_monthly; DROP TABLE dim_sku",
        )


def test_sales_source_resolver_requires_original_to_match_latest_completed_batch():
    cursor = MagicMock()
    batch_started_at = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)
    mirror_loaded_at = datetime(2026, 7, 1, 8, 1, tzinfo=UTC)
    cursor.fetchone.side_effect = [
        ("fact_sales_monthly_original",),
        (120, mirror_loaded_at, 120, batch_started_at, "sku_lvl2_hist_clean.csv"),
    ]

    assert resolve_forecast_sales_table(cursor) == "fact_sales_monthly_original"


@pytest.mark.parametrize(
    ("relation", "state", "message"),
    [
        (None, None, "missing"),
        (
            "fact_sales_monthly_original",
            (
                0,
                None,
                120,
                datetime(2026, 7, 1, tzinfo=UTC),
                "sku_lvl2_hist_clean.csv",
            ),
            "empty",
        ),
        (
            "fact_sales_monthly_original",
            (
                119,
                datetime(2026, 7, 1, tzinfo=UTC),
                120,
                datetime(2026, 7, 1, tzinfo=UTC),
                "sku_lvl2_hist_clean.csv",
            ),
            "row count",
        ),
        (
            "fact_sales_monthly_original",
            (
                120,
                datetime(2026, 6, 30, tzinfo=UTC),
                120,
                datetime(2026, 7, 1, tzinfo=UTC),
                "sku_lvl2_hist_clean.csv",
            ),
            "older",
        ),
        (
            "fact_sales_monthly_original",
            (
                120,
                datetime(2026, 7, 1, tzinfo=UTC),
                None,
                None,
                None,
            ),
            "completed sales batch",
        ),
        (
            "fact_sales_monthly_original",
            (
                120,
                datetime(2026, 7, 1, tzinfo=UTC),
                120,
                datetime(2026, 7, 1, tzinfo=UTC),
                "safe_upsert",
            ),
            "canonical sales reload",
        ),
    ],
)
def test_sales_source_resolver_fails_closed_on_ambiguous_provenance(
    relation, state, message: str
):
    cursor = MagicMock()
    cursor.fetchone.side_effect = [(relation,), state]

    with pytest.raises(RuntimeError, match=message):
        resolve_forecast_sales_table(cursor)


@pytest.mark.parametrize(
    ("minimum", "active_window"),
    [(0, 12), (3, 0), (-1, 12), (3, -1)],
)
def test_eligibility_ctes_reject_non_positive_policy_values(minimum, active_window):
    with pytest.raises(ValueError, match="must be positive"):
        build_forecast_eligibility_ctes(
            planning_month=date(2026, 7, 1),
            min_history_months=minimum,
            active_window_months=active_window,
        )
