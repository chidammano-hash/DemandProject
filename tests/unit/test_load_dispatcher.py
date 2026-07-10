"""Characterization + convergence tests for scripts/etl/load.py (US6).

load.py is the mode-aware ETL dispatcher (onetime | delta | file). US6 keeps
it as the canonical engine but converges its partition internals onto the
shared common/core/etl_helpers primitives. These tests pin the dispatcher's
key SQL-builder / validation behavior (previously untested) and verify the
convergence.
"""

import os
import sys
from datetime import date
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.etl import load as ld


def _executed_sql(cur) -> str:
    return " ; ".join(str(c.args[0]) for c in cur.execute.call_args_list)


class TestIsPgPartitioned:
    def test_true_when_relkind_p(self):
        cur = MagicMock()
        cur.fetchone.return_value = (True,)
        assert ld._is_pg_partitioned(cur, "fact_inventory_snapshot") is True
        assert "relkind" in _executed_sql(cur)

    def test_false_when_not_partitioned(self):
        cur = MagicMock()
        cur.fetchone.return_value = (False,)
        assert ld._is_pg_partitioned(cur, "fact_sales_monthly") is False

    def test_false_when_missing(self):
        cur = MagicMock()
        cur.fetchone.return_value = None
        assert ld._is_pg_partitioned(cur, "nope") is False


class TestEnsureMonthlyPartitions:
    def test_creates_one_partition_per_distinct_month(self):
        cur = MagicMock()
        # 1st execute: SELECT DISTINCT months -> two month rows (1-tuples)
        # then per month: SELECT pg_class (None -> create), CREATE
        cur.fetchall.return_value = [(date(2026, 3, 1),), (date(2026, 4, 1),)]
        cur.fetchone.side_effect = [None, None]  # neither partition exists yet
        created = ld._ensure_monthly_partitions(cur, "fact_inventory_snapshot", "snapshot_date")
        assert created == 2
        sql = _executed_sql(cur)
        assert "fact_inventory_snapshot_2026_03" in sql
        assert "fact_inventory_snapshot_2026_04" in sql
        assert "PARTITION OF" in sql

    def test_skips_existing_partitions(self):
        cur = MagicMock()
        cur.fetchall.return_value = [(date(2026, 3, 1),)]
        cur.fetchone.side_effect = [(1,)]  # already exists
        created = ld._ensure_monthly_partitions(cur, "fact_inventory_snapshot", "snapshot_date")
        assert created == 0
        assert "CREATE TABLE" not in _executed_sql(cur)


class TestResolveConflictTarget:
    def test_prefers_index_containing_ck_field(self):
        cur = MagicMock()
        cur.fetchall.return_value = [
            ("uq_a", ["x", "y", "z"]),
            ("uq_ck", ["sku_ck"]),
        ]
        assert ld._resolve_conflict_target(cur, "t", "sku_ck") == ["sku_ck"]

    def test_falls_back_to_shortest_unique(self):
        cur = MagicMock()
        cur.fetchall.return_value = [("uq_a", ["x", "y"]), ("uq_b", ["z"])]
        assert ld._resolve_conflict_target(cur, "t", None) == ["z"]

    def test_empty_when_no_unique(self):
        cur = MagicMock()
        cur.fetchall.return_value = []
        assert ld._resolve_conflict_target(cur, "t", "ck") == []


class TestFilterOrphanFks:
    def test_skips_fk_when_incoming_staging_omits_optional_local_column(self):
        cur = MagicMock()
        cur.fetchall.return_value = [
            ("champion_experiment", ["champion_experiment_id"], ["experiment_id"]),
        ]

        removed = ld._filter_orphan_fks(
            cur,
            "fact_external_forecast_monthly",
            available_columns={"forecast_ck", "model_id"},
        )

        assert removed == 0
        assert cur.execute.call_count == 1

    def test_filters_fk_when_incoming_staging_provides_local_column(self):
        cur = MagicMock()
        cur.fetchall.return_value = [
            ("dim_item", ["item_id"], ["item_id"]),
        ]
        cur.rowcount = 3

        removed = ld._filter_orphan_fks(
            cur,
            "fact_external_forecast_monthly",
            available_columns={"item_id"},
        )

        assert removed == 3
        assert cur.execute.call_count == 2


class TestOnetimeDelegatesToBulkLoader:
    """US7 verify: load.py's onetime path delegates the bulk insert to the
    canonical load_dataset_postgres.py engine (not a separate implementation)."""

    def test_generic_load_cmd_targets_bulk_loader(self):
        cmd = ld._generic_load_cmd("sales", replace=True)
        joined = " ".join(cmd)
        assert "load_dataset_postgres.py" in joined
        assert "--dataset" in cmd and "sales" in cmd
        assert "--replace" in cmd

    def test_generic_load_cmd_without_replace(self):
        cmd = ld._generic_load_cmd("item", replace=False)
        assert "--replace" not in cmd


class TestValidateArgs:
    def _args(self, **kw):
        from argparse import Namespace
        base = {"mode": "onetime", "domain": "sales", "slice_token": None, "file_arg": None}
        base.update(kw)
        return Namespace(**base)

    def test_slice_rejected_outside_file_mode(self):
        with pytest.raises(ValueError):
            ld._validate_args(self._args(mode="delta", slice_token="2026-03"))

    def test_file_rejected_outside_file_mode(self):
        with pytest.raises(ValueError):
            ld._validate_args(self._args(mode="onetime", file_arg="x.csv"))

    def test_onetime_ok(self):
        ld._validate_args(self._args(mode="onetime"))  # no raise
