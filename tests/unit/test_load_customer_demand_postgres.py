"""Characterization tests for scripts/etl/load_customer_demand_postgres.py.

US1 (data-ingestion streamlining): pins partition naming, month-range, the
per-partition INSERT SQL, and partition create/drop BEFORE US4 (shared
partition manager) and US15 (change detection) touch this loader.
"""

import os
import sys
from datetime import date
from unittest.mock import MagicMock

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
