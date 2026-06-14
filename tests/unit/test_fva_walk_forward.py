# Spec 02-27 walk-forward + point-in-time correctness tests.
"""Walk-forward and point-in-time correctness tests for the AI Planner FVA backtest runner.

Spec: docs/specs/02-forecasting/27-ai-fva-backtest.md (§4.1 walk-forward loop,
§4.3 point-in-time correctness, §4.4 sampling, §4.5 evaluation horizon, §10
edge cases / leakage risk).

Other agents own: LLM mocking, schema/math, frontend, and E2E coverage.
This file restricts itself to:
    1. Date arithmetic helpers (month_floor / add_months / walk_back_months).
    2. Point-in-time correctness of build_dfu_context (strict < T actuals filter,
       baseline >= T lookup, 24-month lookback boundary, horizon-shortfall None).
    3. sample_dfus sampling stratification, limit_override, GREATEST(1, ...) guard.
    4. Walk-forward chronological + horizon invariants.
"""
from __future__ import annotations

from datetime import date
from itertools import pairwise
from unittest.mock import MagicMock

import pytest

from scripts.forecasting.run_ai_fva_backtest import (
    add_months,
    build_dfu_context,
    month_floor,
    sample_dfus,
    walk_back_months,
)

# ---------------------------------------------------------------------------
# 1. Date helpers — pure functions, no mocking
# ---------------------------------------------------------------------------


class TestMonthFloor:
    def test_first_of_month_unchanged(self):
        assert month_floor(date(2026, 5, 1)) == date(2026, 5, 1)

    def test_mid_month_floors_to_first(self):
        assert month_floor(date(2026, 5, 15)) == date(2026, 5, 1)

    def test_month_end_floors_to_first(self):
        assert month_floor(date(2026, 5, 31)) == date(2026, 5, 1)

    def test_leap_year_feb_29_floors_to_feb_1(self):
        # 2024 was a leap year.
        assert month_floor(date(2024, 2, 29)) == date(2024, 2, 1)

    def test_non_leap_year_feb_28_floors_to_feb_1(self):
        assert month_floor(date(2026, 2, 28)) == date(2026, 2, 1)

    @pytest.mark.parametrize("dst_day", [
        date(2026, 3, 8),   # US DST starts
        date(2026, 3, 9),
        date(2026, 11, 1),  # US DST ends
        date(2026, 11, 2),
    ])
    def test_dst_boundary_days_floor_correctly(self, dst_day):
        """month_floor uses date.replace(day=1) so DST is irrelevant —
        but assert it explicitly so a future timezone-aware refactor can't
        silently shift months by a day."""
        assert month_floor(dst_day) == date(dst_day.year, dst_day.month, 1)


class TestAddMonths:
    def test_add_zero_returns_first_of_input_month(self):
        assert add_months(date(2026, 5, 15), 0) == date(2026, 5, 1)

    def test_add_positive_within_year(self):
        assert add_months(date(2026, 1, 1), 3) == date(2026, 4, 1)

    def test_add_positive_crosses_year(self):
        assert add_months(date(2026, 11, 1), 3) == date(2027, 2, 1)

    def test_add_negative_within_year(self):
        assert add_months(date(2026, 5, 1), -3) == date(2026, 2, 1)

    def test_add_negative_crosses_year(self):
        assert add_months(date(2026, 2, 1), -3) == date(2025, 11, 1)

    def test_large_positive_offset(self):
        # 5 years forward.
        assert add_months(date(2026, 5, 1), 60) == date(2031, 5, 1)

    def test_large_negative_offset(self):
        # 5 years back.
        assert add_months(date(2026, 5, 1), -60) == date(2021, 5, 1)

    def test_jan_31_plus_one_month_lands_on_march_1_not_feb_28(self):
        # The spec for the runner's add_months is "floor then add" — so the
        # output is always the FIRST of the resulting month. The non-leap-year
        # corner case "Jan 31 + 1 month" therefore returns Feb 1 (because the
        # input is floored to Jan 1 before addition), NOT Feb 28.
        # This documents the implementation behaviour explicitly so a future
        # change that introduces month-end clamping breaks loudly.
        result = add_months(date(2026, 1, 31), 1)
        assert result == date(2026, 2, 1)
        # Confirm a calendar-day-preserving implementation that produced
        # Feb 28 would also fail this test.
        assert result != date(2026, 2, 28)

    def test_feb_29_leap_year_minus_one_month_is_jan_1(self):
        assert add_months(date(2024, 2, 29), -1) == date(2024, 1, 1)

    def test_dec_to_jan_year_boundary(self):
        assert add_months(date(2026, 12, 1), 1) == date(2027, 1, 1)

    def test_jan_to_dec_year_boundary(self):
        assert add_months(date(2026, 1, 1), -1) == date(2025, 12, 1)


class TestWalkBackMonths:
    def test_default_10_month_window_spec_example(self):
        """Spec §4.1 example: as_of=2026-05-01, window=10 → 2025-08..2026-05."""
        result = walk_back_months(date(2026, 5, 1), 10)
        assert result == [
            date(2025, 8, 1),
            date(2025, 9, 1),
            date(2025, 10, 1),
            date(2025, 11, 1),
            date(2025, 12, 1),
            date(2026, 1, 1),
            date(2026, 2, 1),
            date(2026, 3, 1),
            date(2026, 4, 1),
            date(2026, 5, 1),
        ]

    def test_returns_exactly_window_entries(self):
        for w in (1, 3, 6, 10, 24):
            result = walk_back_months(date(2026, 5, 1), w)
            assert len(result) == w, f"window={w} returned {len(result)}"

    def test_chronological_order(self):
        result = walk_back_months(date(2026, 5, 1), 10)
        assert result == sorted(result), "walk_back_months must be ascending"

    def test_ends_at_as_of_month(self):
        as_of = date(2026, 5, 1)
        result = walk_back_months(as_of, 10)
        assert result[-1] == as_of

    def test_all_entries_are_month_floored(self):
        # Pass a mid-month as_of; every entry should still be day=1.
        result = walk_back_months(date(2026, 5, 17), 10)
        assert all(d.day == 1 for d in result)

    def test_window_of_one_returns_just_as_of(self):
        assert walk_back_months(date(2026, 5, 1), 1) == [date(2026, 5, 1)]

    def test_walk_crosses_year_boundary(self):
        result = walk_back_months(date(2026, 2, 1), 4)
        assert result == [
            date(2025, 11, 1),
            date(2025, 12, 1),
            date(2026, 1, 1),
            date(2026, 2, 1),
        ]


# ---------------------------------------------------------------------------
# 2. Walk-forward invariants (cross-cutting)
# ---------------------------------------------------------------------------


class TestWalkForwardInvariants:
    def test_no_target_month_exceeds_t_plus_horizon(self):
        """For each month T in the walk-forward sequence, the AI-adjusted-forecast
        rows persisted for that month must have target_month <= T + horizon.

        We simulate the insert_adjusted_forecasts row construction directly,
        using the baseline iteration order from build_dfu_context (T+1..T+H),
        and assert the spec invariant.
        """
        as_of = date(2026, 5, 1)
        window = 10
        horizon = 3

        for month_t in walk_back_months(as_of, window):
            # Simulated baseline_forecast for T+1..T+H: month strings only.
            baseline_months = [add_months(month_t, h) for h in range(1, horizon + 1)]
            for target_month in baseline_months:
                # target_month must lie strictly after T and within T + horizon.
                assert target_month > month_t, (
                    f"target_month {target_month} not strictly after T={month_t}"
                )
                assert target_month <= add_months(month_t, horizon), (
                    f"target_month {target_month} exceeds T={month_t} + horizon={horizon}"
                )

    def test_walk_forward_months_strictly_increasing(self):
        """Each subsequent T must be exactly one calendar month after the prior."""
        result = walk_back_months(date(2026, 5, 1), 10)
        for prev, curr in pairwise(result):
            assert add_months(prev, 1) == curr, (
                f"non-contiguous step: {prev} -> {curr}"
            )


# ---------------------------------------------------------------------------
# 3. sample_dfus — DB-mocked, stratification + override behaviour
# ---------------------------------------------------------------------------


def _make_mock_conn(fetchall_rows):
    """Return a MagicMock psycopg connection whose cursor.fetchall() returns rows.

    sample_dfus uses ``with conn.cursor() as cur:`` so we wire __enter__/__exit__.
    """
    cursor = MagicMock()
    cursor.fetchall.return_value = list(fetchall_rows)
    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cursor


# Fixed dummy values for the baseline-coverage filter — the cursor is mocked,
# so concrete values don't matter; they only need to occupy the SQL params.
_BASELINE_KWARGS = {
    "baseline_model_id": "external",
    "baseline_window_start": date(2025, 1, 1),
    "baseline_window_end": date(2026, 1, 1),
}


class TestSampleDfus:
    def test_returns_list_of_item_loc_tuples(self):
        conn, _ = _make_mock_conn([("A", "L1"), ("B", "L2"), ("C", "L3")])
        result = sample_dfus(conn, {"default_mode": "stratified"}, **_BASELINE_KWARGS)
        assert result == [("A", "L1"), ("B", "L2"), ("C", "L3")]

    def test_empty_dim_sku_returns_empty_list_not_raises(self):
        """Spec §10 edge: cold/empty universe must not crash the runner."""
        conn, _ = _make_mock_conn([])
        result = sample_dfus(conn, {"default_mode": "stratified"}, **_BASELINE_KWARGS)
        assert result == []

    def test_limit_override_passed_as_sql_limit(self):
        """limit_override (CLI --limit-dfus) must bind to the SQL LIMIT, not
        the configured stratified.max_dfus default."""
        conn, cursor = _make_mock_conn([("A", "L1")])
        sample_dfus(conn, {"stratified": {"pct": 5, "max_dfus": 10000}},
                    limit_override=37, **_BASELINE_KWARGS)
        # SQL params: (baseline_model_id, win_start, win_end, pct, max_dfus).
        executed_args = cursor.execute.call_args
        params = executed_args[0][1]
        assert params[4] == 37, (
            f"limit_override=37 must reach SQL LIMIT bind; got params={params}"
        )

    def test_pct_passed_to_sql(self):
        """The pct from config must reach the SQL — drives bucket sampling."""
        conn, cursor = _make_mock_conn([])
        sample_dfus(conn, {"stratified": {"pct": 25, "max_dfus": 100}}, **_BASELINE_KWARGS)
        params = cursor.execute.call_args[0][1]
        assert params[3] == 25

    def test_default_max_dfus_used_when_no_override(self):
        conn, cursor = _make_mock_conn([])
        sample_dfus(conn, {"stratified": {"pct": 5, "max_dfus": 9999}}, **_BASELINE_KWARGS)
        params = cursor.execute.call_args[0][1]
        assert params[4] == 9999

    def test_pct_100_eligibility(self):
        """At pct=100, every bucket member is eligible (the SQL filter
        ``rn <= GREATEST(1, ceil(bucket_size * 100 / 100.0))`` resolves to
        ``rn <= bucket_size`` which is always true for the partition).

        We can't exercise the SQL directly without a DB, but we can assert
        pct=100 is forwarded as-is and the result respects the cap."""
        conn, cursor = _make_mock_conn([("A", "L1"), ("B", "L2")])
        result = sample_dfus(conn, {"stratified": {"pct": 100, "max_dfus": 500}}, **_BASELINE_KWARGS)
        params = cursor.execute.call_args[0][1]
        assert params[3] == 100
        assert result == [("A", "L1"), ("B", "L2")]

    def test_greatest_one_guard_in_sql(self):
        """The SQL must use GREATEST(1, ceil(bucket_size * pct / 100.0)) so a
        bucket of size 1 with low pct still admits at least one DFU per spec."""
        conn, cursor = _make_mock_conn([])
        sample_dfus(conn, {"stratified": {"pct": 1, "max_dfus": 10}}, **_BASELINE_KWARGS)
        sql_text = cursor.execute.call_args[0][0]
        assert "GREATEST(1" in sql_text, (
            "sample_dfus SQL must include GREATEST(1, ...) so buckets of size 1 "
            "always yield at least one DFU"
        )

    def test_baseline_coverage_filter_binds_window(self):
        """sample_dfus must inner-join on fact_external_forecast_monthly and
        bind the baseline window so we sample only DFUs with usable baselines."""
        conn, cursor = _make_mock_conn([])
        sample_dfus(conn, {"stratified": {"pct": 5, "max_dfus": 100}}, **_BASELINE_KWARGS)
        sql_text = cursor.execute.call_args[0][0]
        params = cursor.execute.call_args[0][1]
        assert "fact_external_forecast_monthly" in sql_text
        assert params[0] == _BASELINE_KWARGS["baseline_model_id"]
        assert params[1] == _BASELINE_KWARGS["baseline_window_start"]
        assert params[2] == _BASELINE_KWARGS["baseline_window_end"]

    def test_non_stratified_mode_falls_back_to_stratified(self):
        """Spec §4.4 lists full/targeted/cluster modes; runner currently warns
        and falls back to stratified. Confirm the call still completes."""
        conn, _ = _make_mock_conn([("A", "L1")])
        result = sample_dfus(conn, {"default_mode": "full"}, **_BASELINE_KWARGS)
        assert result == [("A", "L1")]


# ---------------------------------------------------------------------------
# 4. build_dfu_context — point-in-time correctness (PRD §4.3, #1 leakage risk)
# ---------------------------------------------------------------------------


def _make_context_conn(
    actuals_rows, baseline_rows, meta_row=("c1", "A"), customer_rows=(),
):
    """Return a mock psycopg conn whose four sequential execute() calls
    (actuals, baseline, meta, customers) feed the matching fetchall / fetchone returns.

    `meta_row` is a (cluster_assignment, abc_vol) tuple — matches the production
    SELECT in build_dfu_context after dropping the non-existent demand_pattern.

    `customer_rows` (default empty) feeds the v1.1.0 top-customer fetch. Each
    row is (customer_no, customer_name, month_str, sales_qty, total).
    """
    cursor = MagicMock()
    cursor.fetchall.side_effect = [
        list(actuals_rows), list(baseline_rows), list(customer_rows),
    ]
    cursor.fetchone.return_value = meta_row
    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cursor


class TestBuildDfuContextPointInTime:
    """Spec §4.3 — at month T, NO future data may appear in the AI's context.

    Key invariants the runner promises:
      * actuals filter is ``startdate < T`` (strict) — an actual exactly at T
        is the in-progress / not-yet-closed month and must be excluded.
      * actuals lookback is exactly 24 months — anything older is excluded.
      * baseline lookup is ``startdate >= T AND startdate < T + horizon``.
      * If baseline has fewer than ``horizon`` rows → return None.
    """

    def test_actuals_filter_uses_strict_less_than(self):
        """An actual whose startdate == T must NOT appear in the context.

        We don't have a live DB here — we assert the SQL parameter binding
        AND the SQL operator. This protects against a regression that
        changes ``startdate < %s`` to ``startdate <= %s``.
        """
        conn, cursor = _make_context_conn(
            actuals_rows=[("2026-04", 100.0), ("2026-03", 90.0)],
            baseline_rows=[("2026-05", 110.0), ("2026-06", 120.0), ("2026-07", 130.0)],
        )
        as_of = date(2026, 5, 1)
        ctx = build_dfu_context(conn, "A", "L1", as_of, horizon=3)
        assert ctx is not None

        # Inspect the first execute call (actuals query): SQL + params.
        actuals_call = cursor.execute.call_args_list[0]
        sql_text = actuals_call[0][0]
        params = actuals_call[0][1]

        # Operator must be strict <, not <= (point-in-time correctness).
        assert "startdate <  %s" in sql_text or "startdate < %s" in sql_text, (
            "Actuals SQL must use strict '<' on as_of (not '<=') to prevent "
            "leakage of the in-progress month into the AI context."
        )
        # The third bind parameter is the upper bound — must equal as_of.
        assert params[2] == as_of, (
            f"Actuals upper-bound bind must equal T={as_of}; got {params[2]}"
        )

    def test_actuals_lookback_is_24_months(self):
        """The lower bound of the actuals filter must be exactly T - 24 months
        (spec calls for ``actuals_last_24m``)."""
        conn, cursor = _make_context_conn(
            actuals_rows=[],
            baseline_rows=[("2026-05", 110.0)] * 3,
        )
        as_of = date(2026, 5, 1)
        build_dfu_context(conn, "A", "L1", as_of, horizon=3)

        params = cursor.execute.call_args_list[0][0][1]
        # params layout: (item_id, loc, as_of, horizon_start)
        expected_horizon_start = add_months(as_of, -24)
        assert params[3] == expected_horizon_start, (
            f"Actuals lookback lower bound must be T - 24 months = "
            f"{expected_horizon_start}; got {params[3]}"
        )

    def test_baseline_lookup_starts_at_or_after_t(self):
        """Baseline forecast must come from rows with ``startdate >= T``
        (the AI sees what was forecast FOR T+1 onwards)."""
        conn, cursor = _make_context_conn(
            actuals_rows=[],
            baseline_rows=[("2026-05", 110.0), ("2026-06", 120.0), ("2026-07", 130.0)],
        )
        as_of = date(2026, 5, 1)
        build_dfu_context(conn, "A", "L1", as_of, horizon=3)

        baseline_call = cursor.execute.call_args_list[1]
        sql_text = baseline_call[0][0]
        params = baseline_call[0][1]

        assert "startdate >= %s" in sql_text, (
            "Baseline SQL must use 'startdate >= %s' for the start bound."
        )
        # params: (item_id, loc, baseline_model_id, baseline_start, baseline_end, horizon)
        assert params[3] == as_of, (
            f"Baseline start bind must equal T={as_of}; got {params[3]}"
        )
        assert params[4] == add_months(as_of, 3), (
            f"Baseline end bind must equal T + horizon; got {params[4]}"
        )

    def test_returns_none_when_baseline_shorter_than_horizon(self):
        """Spec §10 risk #4: DFUs with no/short baseline snapshot must be skipped."""
        conn, _ = _make_context_conn(
            actuals_rows=[("2026-04", 100.0)],
            baseline_rows=[("2026-05", 110.0), ("2026-06", 120.0)],  # only 2, need 3
        )
        result = build_dfu_context(conn, "A", "L1", date(2026, 5, 1), horizon=3)
        assert result is None

    def test_returns_none_when_baseline_empty(self):
        conn, _ = _make_context_conn(
            actuals_rows=[],
            baseline_rows=[],
        )
        result = build_dfu_context(conn, "A", "L1", date(2026, 5, 1), horizon=3)
        assert result is None

    def test_actuals_quantities_floored_to_zero_when_null(self):
        """qty IS NULL in the DB row should not crash; the runner coerces to 0.0."""
        conn, _ = _make_context_conn(
            actuals_rows=[("2026-04", None), ("2026-03", 50.0)],
            baseline_rows=[("2026-05", 100.0)] * 3,
        )
        ctx = build_dfu_context(conn, "A", "L1", date(2026, 5, 1), horizon=3)
        assert ctx is not None
        assert ctx.actuals_last_24m[0] == ("2026-04", 0.0)
        assert ctx.actuals_last_24m[1] == ("2026-03", 50.0)

    def test_meta_missing_returns_none_metadata(self):
        """A DFU absent from dim_sku must produce a context with None metadata,
        not crash. The runner uses ``fetchone() or (None, None)`` for meta."""
        cursor = MagicMock()
        cursor.fetchall.side_effect = [
            [("2026-04", 100.0)],
            [("2026-05", 110.0), ("2026-06", 120.0), ("2026-07", 130.0)],
            [],                # customers fetchall (v1.1.0)
        ]
        cursor.fetchone.return_value = None
        conn = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        ctx = build_dfu_context(conn, "A", "L1", date(2026, 5, 1), horizon=3)
        assert ctx is not None
        assert ctx.cluster is None
        assert ctx.demand_pattern is None
        assert ctx.abc_vol is None
        assert ctx.top_customers is None

    def test_forecast_run_month_in_context_equals_as_of(self):
        conn, _ = _make_context_conn(
            actuals_rows=[],
            baseline_rows=[("2026-05", 110.0)] * 3,
        )
        as_of = date(2026, 5, 1)
        ctx = build_dfu_context(conn, "item-X", "loc-Y", as_of, horizon=3)
        assert ctx is not None
        assert ctx.forecast_run_month == as_of
        assert ctx.item_id == "item-X"
        assert ctx.loc == "loc-Y"


class TestBuildDfuContextLookbackBoundary:
    """Boundary tests for the 24-month lookback. Spec calls for exactly 24
    months — 25-month-old data must be excluded."""

    def test_24_month_lower_bound_value_for_as_of_2026_05(self):
        """T=2026-05-01 minus 24 months = 2024-05-01.
        A row at 2024-04-01 (25 months back) is below the lower bound."""
        as_of = date(2026, 5, 1)
        lower_bound = add_months(as_of, -24)
        assert lower_bound == date(2024, 5, 1)

        # Verify the SQL binds this exact lower bound (the WHERE clause uses
        # ``startdate >= %s`` so 2024-04-01 < 2024-05-01 is excluded).
        conn, cursor = _make_context_conn(
            actuals_rows=[],
            baseline_rows=[("2026-05", 100.0)] * 3,
        )
        build_dfu_context(conn, "A", "L1", as_of, horizon=3)

        params = cursor.execute.call_args_list[0][0][1]
        assert params[3] == date(2024, 5, 1)

    def test_24_month_boundary_crosses_year(self):
        """T=2026-02-01 minus 24 months = 2024-02-01 (clean year offset)."""
        as_of = date(2026, 2, 1)
        assert add_months(as_of, -24) == date(2024, 2, 1)


# ---------------------------------------------------------------------------
# 5. Top-customer fetch (v1.1.0) — point-in-time + grouping
# ---------------------------------------------------------------------------

class TestBuildDfuContextTopCustomers:
    def test_customer_rows_grouped_into_history(self):
        """Two customers × 3 months of rows -> 2 CustomerHistory entries."""
        customer_rows = [
            # (customer_no, customer_name, month, sales_qty, total)
            ("50592", "STMLS LLC",  "2025-08",  4.0, 18.0),
            ("50592", "STMLS LLC",  "2025-09",  6.0, 18.0),
            ("50592", "STMLS LLC",  "2025-10",  8.0, 18.0),
            ("68002", "WELCOME",    "2025-08",  3.0,  6.0),
            ("68002", "WELCOME",    "2025-09",  2.0,  6.0),
            ("68002", "WELCOME",    "2025-10",  1.0,  6.0),
        ]
        conn, _ = _make_context_conn(
            actuals_rows=[("2025-10", 14.0)],
            baseline_rows=[("2025-11", 28.0), ("2025-12", 66.0), ("2026-01", 18.0)],
            customer_rows=customer_rows,
        )
        ctx = build_dfu_context(conn, "916045", "1401-BULK", date(2025, 11, 1), horizon=3)
        assert ctx is not None
        assert ctx.top_customers is not None and len(ctx.top_customers) == 2
        # Order preserved from SQL (top-volume customer first).
        assert ctx.top_customers[0].customer_no == "50592"
        assert ctx.top_customers[0].customer_name == "STMLS LLC"
        assert ctx.top_customers[0].monthly == [
            ("2025-08", 4.0), ("2025-09", 6.0), ("2025-10", 8.0),
        ]
        assert ctx.top_customers[0].total() == 18.0
        assert ctx.top_customers[1].customer_no == "68002"
        assert ctx.top_customers[1].total() == 6.0

    def test_no_customer_rows_sets_top_customers_to_none(self):
        """Empty result -> top_customers stays None (skips prompt rendering)."""
        conn, _ = _make_context_conn(
            actuals_rows=[],
            baseline_rows=[("2025-11", 1.0)] * 3,
            customer_rows=[],
        )
        ctx = build_dfu_context(conn, "X", "L1", date(2025, 11, 1), horizon=3)
        assert ctx is not None
        assert ctx.top_customers is None

    def test_customer_query_window_is_strict_less_than_t(self):
        """Customer SQL must use ``startdate < T`` to prevent leakage at T."""
        conn, cursor = _make_context_conn(
            actuals_rows=[],
            baseline_rows=[("2025-11", 1.0)] * 3,
            customer_rows=[],
        )
        as_of = date(2025, 11, 1)
        build_dfu_context(conn, "X", "L1", as_of, horizon=3)
        # The customer SQL is the 4th execute (actuals=0, baseline=1, meta=2, customers=3).
        sql_text = cursor.execute.call_args_list[3][0][0]
        params = cursor.execute.call_args_list[3][0][1]
        assert "fact_customer_demand_monthly" in sql_text
        assert "startdate <  %s" in sql_text or "startdate < %s" in sql_text
        # T must appear as the strict upper bound for both top_n and main SELECT.
        # Params: (item_id, loc, win_start, T, top_k, item_id, loc, win_start, T)
        assert params[3] == as_of
        assert params[8] == as_of
