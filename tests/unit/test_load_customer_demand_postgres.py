"""Characterization tests for scripts/etl/load_customer_demand_postgres.py.

US1 (data-ingestion streamlining): pins partition naming, month-range, the
per-partition INSERT SQL, and partition create/drop BEFORE US4 (shared
partition manager) and US15 (change detection) touch this loader.
"""

import os
import sys
from datetime import date
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.etl import load_customer_demand_postgres as cd


class TestPartitionName:
    def test_january(self):
        assert cd._partition_name(date(2024, 1, 1)) == "fact_customer_demand_monthly_2024_01"

    def test_december(self):
        assert cd._partition_name(date(2024, 12, 1)) == "fact_customer_demand_monthly_2024_12"


class TestMonthRange:
    def test_mid_year_half_open_interval(self):
        assert cd._month_range(date(2024, 3, 1)) == ("2024-03-01", "2024-04-01")

    def test_december_wraps_year(self):
        assert cd._month_range(date(2024, 12, 1)) == ("2024-12-01", "2025-01-01")


class TestBuildPartitionInsertSql:
    def test_targets_partition_and_aggregates(self):
        sql = cd._build_partition_insert_sql("part_x")
        assert 'INSERT INTO "part_x"' in sql
        assert "GROUP BY" in sql
        # demand_ck is the concatenated business key
        assert "demand_ck" in sql
        assert "SUM(s.demand_qty" in sql


class TestEnsurePartitionExists:
    def test_creates_when_absent(self):
        cur = MagicMock()
        cur.fetchone.return_value = None  # partition does not exist
        name = cd._ensure_partition_exists(cur, date(2024, 5, 1))
        assert name == "fact_customer_demand_monthly_2024_05"
        executed = " ".join(str(c.args[0]) for c in cur.execute.call_args_list)
        assert "CREATE TABLE" in executed
        assert "FOR VALUES FROM ('2024-05-01') TO ('2024-06-01')" in executed

    def test_skips_create_when_present(self):
        cur = MagicMock()
        cur.fetchone.return_value = (1,)  # partition exists
        name = cd._ensure_partition_exists(cur, date(2024, 5, 1))
        assert name == "fact_customer_demand_monthly_2024_05"
        executed = " ".join(str(c.args[0]) for c in cur.execute.call_args_list)
        assert "CREATE TABLE" not in executed


class TestDropPartition:
    def test_drops_if_exists(self):
        cur = MagicMock()
        cd._drop_partition(cur, date(2024, 7, 1))
        executed = " ".join(str(c.args[0]) for c in cur.execute.call_args_list)
        assert 'DROP TABLE IF EXISTS "fact_customer_demand_monthly_2024_07"' in executed


def test_finalize_refreshes_every_dependent_view_before_completing_batch() -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    events: list[str] = []
    conn.commit.side_effect = lambda: events.append("commit")

    with (
        patch.object(
            cd,
            "mvs_for_tables",
            return_value=["mv_customer_demand_series_profile"],
        ),
        patch.object(
            cd,
            "refresh_for_tables",
            side_effect=lambda *_args, **_kwargs: (
                events.append("refresh")
                or {
                    "refreshed": ["mv_customer_demand_series_profile"],
                    "failed": [],
                    "missing": [],
                }
            ),
        ),
        patch.object(
            cd,
            "complete_batch",
            side_effect=lambda *_args: events.append("complete"),
        ),
    ):
        cd._finalize_customer_demand_load(
            conn,
            db_params={"dbname": "demand"},
            batch_id=91,
            rows_in=12,
            rows_out=10,
        )

    assert events == ["commit", "refresh", "complete", "commit"]
    marker_call = next(
        executed
        for executed in cur.execute.call_args_list
        if "customer_demand_profile_refresh_state" in executed.args[0]
    )
    assert "ON CONFLICT (singleton_id) DO UPDATE" in marker_call.args[0]
    assert marker_call.args[1] == (91,)


def test_finalize_does_not_complete_batch_when_profile_refresh_fails() -> None:
    conn = MagicMock()

    with (
        patch.object(
            cd,
            "mvs_for_tables",
            return_value=["mv_customer_demand_series_profile"],
        ),
        patch.object(
            cd,
            "refresh_for_tables",
            return_value={
                "refreshed": [],
                "failed": ["mv_customer_demand_series_profile"],
                "missing": [],
            },
        ),
        patch.object(cd, "complete_batch") as complete,
        pytest.raises(RuntimeError, match="materialized-view refresh failed"),
    ):
        cd._finalize_customer_demand_load(
            conn,
            db_params={"dbname": "demand"},
            batch_id=91,
            rows_in=12,
            rows_out=10,
        )

    complete.assert_not_called()
    assert conn.commit.call_args_list == [call()]


def test_post_write_failure_invalidates_profile_marker_with_failed_batch() -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur

    with patch.object(cd, "fail_batch") as fail:
        cd._fail_customer_demand_load(
            conn,
            batch_id=92,
            error=RuntimeError("profile refresh failed"),
            invalidate_profile=True,
        )

    fail.assert_called_once_with(cur, 92, "profile refresh failed")
    marker_delete = next(
        executed
        for executed in cur.execute.call_args_list
        if "DELETE FROM customer_demand_profile_refresh_state" in executed.args[0]
    )
    assert "singleton_id = 1" in marker_delete.args[0]
    conn.rollback.assert_called_once_with()
    conn.commit.assert_called_once_with()


def test_reconcile_abandoned_running_batches_invalidates_profile_marker() -> None:
    cur = MagicMock()
    cur.rowcount = 2

    reconciled = cd._reconcile_abandoned_customer_demand_loads(cur)

    assert reconciled == 2
    update_call, marker_delete = cur.execute.call_args_list
    assert "UPDATE audit_load_batch" in update_call.args[0]
    assert "domain = 'customer_demand'" in update_call.args[0]
    assert "status = 'running'" in update_call.args[0]
    assert "status = 'failed'" in update_call.args[0]
    assert "DELETE FROM customer_demand_profile_refresh_state" in marker_delete.args[0]
