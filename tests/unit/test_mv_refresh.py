"""Tests for the central MV-refresh service (common/core/mv_refresh.py).

The DDL-consistency tests are the mechanical enforcement of the rule
"every materialized view lives in MV_SOURCES": adding a CREATE MATERIALIZED
VIEW to sql/ without registering it here fails the suite.
"""
from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import psycopg
import pytest

from common.core.mv_refresh import (
    HEAVY_MVS,
    MV_SOURCES,
    all_mvs,
    mvs_for_tables,
    refresh_for_tables,
    refresh_materialized_views,
    refresh_tiers,
)

from common.core.paths import PROJECT_ROOT

SQL_DIR = PROJECT_ROOT / "sql"

_CREATE_RE = re.compile(r"CREATE MATERIALIZED VIEW (?:IF NOT EXISTS )?(\w+)", re.I)
_DROP_RE = re.compile(r"DROP MATERIALIZED VIEW (?:IF EXISTS )?(\w+)", re.I)


def _live_mvs_from_ddl() -> set[str]:
    """MVs whose last CREATE comes after their last DROP, in migration order."""
    last_event: dict[str, str] = {}  # mv -> "create" | "drop"
    for path in sorted(SQL_DIR.glob("*.sql")):
        if not path.is_file():
            continue
        text = path.read_text()
        events = [(m.start(), "create", m.group(1)) for m in _CREATE_RE.finditer(text)]
        events += [(m.start(), "drop", m.group(1)) for m in _DROP_RE.finditer(text)]
        for _, kind, name in sorted(events):
            last_event[name.lower()] = kind
    return {mv for mv, kind in last_event.items() if kind == "create"}


class TestMapMatchesDdl:
    def test_every_live_mv_is_registered(self):
        missing = _live_mvs_from_ddl() - set(MV_SOURCES)
        assert not missing, (
            f"Materialized views in sql/ DDL missing from MV_SOURCES: {sorted(missing)}. "
            "Register them in common/core/mv_refresh.py (in dependency order)."
        )

    def test_no_registered_mv_is_retired_or_unknown(self):
        unknown = set(MV_SOURCES) - _live_mvs_from_ddl()
        assert not unknown, (
            f"MV_SOURCES entries with no live CREATE MATERIALIZED VIEW in sql/: "
            f"{sorted(unknown)}"
        )

    def test_map_order_is_topological(self):
        seen: set[str] = set()
        for mv, sources in MV_SOURCES.items():
            upstream_mvs = sources & set(MV_SOURCES)
            not_yet = upstream_mvs - seen
            assert not not_yet, (
                f"{mv} depends on {sorted(not_yet)} which appear later in MV_SOURCES"
            )
            seen.add(mv)

    def test_declared_sources_appear_in_ddl_body(self):
        # Latest definition body per MV.
        bodies: dict[str, str] = {}
        for path in sorted(SQL_DIR.glob("*.sql")):
            if not path.is_file():
                continue
            text = re.sub(r"--.*", "", path.read_text())
            for m in re.finditer(
                r"CREATE MATERIALIZED VIEW (?:IF NOT EXISTS )?(\w+)\s+AS(.*?)(?=;)",
                text,
                re.S | re.I,
            ):
                bodies[m.group(1).lower()] = m.group(2)
        for mv, sources in MV_SOURCES.items():
            body = bodies.get(mv, "")
            assert body, f"No CREATE body found in sql/ for {mv}"
            for src in sources:
                assert re.search(rf"\b{src}\b", body), (
                    f"{mv} declares source {src!r} but its DDL body never references it"
                )


class TestClosure:
    def test_forecast_table_reaches_all_accuracy_mvs(self):
        mvs = mvs_for_tables(["fact_external_forecast_monthly"])
        for expected in (
            "agg_forecast_monthly",
            "agg_accuracy_by_dim",
            "agg_accuracy_by_dfu",
            "agg_dfu_coverage",
            "agg_dfu_naive_scale",
        ):
            assert expected in mvs
        # lag-archive MVs read backtest_lag_archive, not the main table
        assert "agg_accuracy_lag_archive" not in mvs

    def test_cluster_assignment_reaches_clustered_accuracy_mvs(self):
        mvs = mvs_for_tables(["sku_cluster_assignment"])
        for expected in (
            "agg_accuracy_by_dim",
            "agg_accuracy_by_dfu",
            "agg_dfu_coverage",
            "agg_accuracy_lag_archive",
            "agg_dfu_coverage_lag_archive",
        ):
            assert expected in mvs

    def test_lag_archive_reaches_archive_mvs_only(self):
        mvs = mvs_for_tables(["backtest_lag_archive"])
        assert "agg_accuracy_lag_archive" in mvs
        assert "agg_dfu_coverage_lag_archive" in mvs
        assert "agg_accuracy_by_dim" not in mvs

    def test_production_forecast_reaches_fairness_audit(self):
        assert "mv_fairness_audit" in mvs_for_tables(["fact_production_forecast"])

    def test_transitive_closure_through_mvs(self):
        mvs = mvs_for_tables(["fact_inventory_snapshot"])
        # agg_inventory_monthly -> mv_inventory_forecast_monthly ->
        # mv_inventory_health_score -> mv_control_tower_kpis
        for expected in (
            "agg_inventory_monthly",
            "mv_inventory_forecast_monthly",
            "mv_inventory_health_score",
            "mv_control_tower_kpis",
        ):
            assert expected in mvs

    def test_order_follows_map_order(self):
        mvs = mvs_for_tables(["fact_inventory_snapshot"])
        assert mvs.index("agg_inventory_monthly") < mvs.index("mv_inventory_forecast_monthly")
        assert mvs.index("mv_inventory_forecast_monthly") < mvs.index("mv_inventory_health_score")
        assert mvs.index("mv_inventory_health_score") < mvs.index("mv_control_tower_kpis")

    def test_include_heavy_false_skips_heavy_mvs(self):
        with_heavy = mvs_for_tables(["fact_inventory_snapshot"])
        without = mvs_for_tables(["fact_inventory_snapshot"], include_heavy=False)
        assert HEAVY_MVS & set(with_heavy)
        assert not (HEAVY_MVS & set(without))
        # Dependents of a skipped heavy MV are still refreshed.
        assert "mv_control_tower_kpis" in without

    def test_unknown_table_yields_nothing(self):
        assert mvs_for_tables(["no_such_table"]) == []

    def test_all_mvs_matches_map(self):
        assert all_mvs() == list(MV_SOURCES)


class TestRefreshTiers:
    def test_tiers_respect_dependencies(self):
        tiers = refresh_tiers(all_mvs())
        seen: set[str] = set()
        for tier in tiers:
            for mv in tier:
                upstream = MV_SOURCES[mv] & set(MV_SOURCES)
                assert upstream <= seen, f"{mv} scheduled before upstream {upstream - seen}"
            seen.update(tier)

    def test_subset_keeps_relative_order(self):
        tiers = refresh_tiers(["mv_control_tower_kpis", "agg_inventory_monthly"])
        flat = [mv for tier in tiers for mv in tier]
        assert flat.index("agg_inventory_monthly") < flat.index("mv_control_tower_kpis")


def _mock_conn(populated_rows):
    """Build a mocked psycopg connection whose cursor reports *populated_rows*."""
    cur = MagicMock()
    cur.fetchall.return_value = populated_rows
    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx, cur


class TestRefreshExecution:
    def test_refreshes_concurrently_when_populated(self):
        ctx, cur = _mock_conn([("agg_sales_monthly", True)])
        with patch("common.core.mv_refresh.psycopg.connect", return_value=ctx):
            result = refresh_materialized_views(
                ["agg_sales_monthly"], db_params={"host": "x"}
            )
        assert result["refreshed"] == ["agg_sales_monthly"]
        executed = " ".join(str(c.args[0]) for c in cur.execute.call_args_list)
        assert "CONCURRENTLY" in executed

    def test_plain_refresh_when_not_populated(self):
        ctx, cur = _mock_conn([("agg_sales_monthly", False)])
        with patch("common.core.mv_refresh.psycopg.connect", return_value=ctx):
            result = refresh_materialized_views(
                ["agg_sales_monthly"], db_params={"host": "x"}
            )
        assert result["refreshed"] == ["agg_sales_monthly"]
        refresh_calls = [
            str(c.args[0]) for c in cur.execute.call_args_list
            if "REFRESH" in str(c.args[0])
        ]
        assert refresh_calls and all("CONCURRENTLY" not in c for c in refresh_calls)

    def test_missing_mv_is_skipped(self):
        ctx, _ = _mock_conn([])
        with patch("common.core.mv_refresh.psycopg.connect", return_value=ctx):
            result = refresh_materialized_views(
                ["agg_sales_monthly"], db_params={"host": "x"}
            )
        assert result["missing"] == ["agg_sales_monthly"]
        assert result["refreshed"] == []

    def test_failure_does_not_block_remaining_mvs(self):
        ctx, cur = _mock_conn(
            [("agg_sales_monthly", True), ("agg_forecast_monthly", True)]
        )

        def _execute(stmt, *args):
            text = str(stmt)
            if "agg_sales_monthly" in text and "REFRESH" in text:
                raise psycopg.OperationalError("boom")

        cur.execute.side_effect = _execute
        with patch("common.core.mv_refresh.psycopg.connect", return_value=ctx):
            result = refresh_materialized_views(
                ["agg_sales_monthly", "agg_forecast_monthly"], db_params={"host": "x"}
            )
        # CONCURRENTLY fails then plain fails -> failed; the second MV proceeds.
        assert result["failed"] == ["agg_sales_monthly"]
        assert result["refreshed"] == ["agg_forecast_monthly"]

    def test_cancel_event_raises(self):
        from threading import Event

        cancel = Event()
        cancel.set()
        ctx, _ = _mock_conn([("agg_sales_monthly", True)])
        with patch("common.core.mv_refresh.psycopg.connect", return_value=ctx):
            with pytest.raises(RuntimeError, match="cancelled"):
                refresh_materialized_views(
                    ["agg_sales_monthly"], db_params={"host": "x"}, cancel_event=cancel
                )

    def test_refresh_for_tables_no_dependents_short_circuits(self):
        with patch("common.core.mv_refresh.psycopg.connect") as connect:
            result = refresh_for_tables(["no_such_table"], db_params={"host": "x"})
        assert result == {"refreshed": [], "failed": [], "missing": []}
        connect.assert_not_called()
