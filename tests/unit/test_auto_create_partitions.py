"""Unit tests for ``scripts.db.auto_create_partitions``.

Pure-function tests for the date math + DDL builders, plus a lightweight
mocked test of ``ensure_partitions`` to verify idempotency at the call layer.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from scripts.db.auto_create_partitions import (
    PARTITIONED_TABLES,
    PartitionedTable,
    add_weeks,
    build_monthly_partition_ddl,
    build_partition_ddl,
    build_weekly_partition_ddl,
    ensure_partitions,
    iso_week_partition_suffix,
    iso_week_start,
)


# ---------------------------------------------------------------------------
# ISO-week date math
# ---------------------------------------------------------------------------
class TestIsoWeekStart:
    """``iso_week_start`` collapses any date to the Monday of its ISO week."""

    @pytest.mark.parametrize(
        "given,expected",
        [
            # 2026-01-01 is a Thursday — Monday of that week is 2025-12-29.
            (date(2026, 1, 1), date(2025, 12, 29)),
            # 2026-01-05 is itself a Monday — should be unchanged.
            (date(2026, 1, 5), date(2026, 1, 5)),
            # 2026-01-11 is a Sunday — Monday of that week is 2026-01-05.
            (date(2026, 1, 11), date(2026, 1, 5)),
            # Year boundary: 2027-01-03 is a Sunday → Monday 2026-12-28.
            (date(2027, 1, 3), date(2026, 12, 28)),
        ],
    )
    def test_aligns_to_monday(self, given: date, expected: date) -> None:
        assert iso_week_start(given) == expected
        assert iso_week_start(given).isoweekday() == 1

    def test_idempotent_on_monday(self) -> None:
        monday = date(2026, 5, 4)  # known Monday
        assert iso_week_start(monday) == monday
        assert iso_week_start(iso_week_start(monday)) == monday


class TestIsoWeekSuffix:
    """``iso_week_partition_suffix`` follows ISO-8601 week numbering."""

    @pytest.mark.parametrize(
        "monday,expected",
        [
            # ISO week 1 of 2026 starts 2025-12-29 (because 2026-01-01 is Thu).
            (date(2025, 12, 29), "2026w01"),
            (date(2026, 1, 5), "2026w02"),
            # ISO week 53 of 2026 (long ISO year): 2026-12-28 → 2026w53.
            (date(2026, 12, 28), "2026w53"),
            # ISO week 1 of 2024 starts 2024-01-01 (Mon).
            (date(2024, 1, 1), "2024w01"),
        ],
    )
    def test_iso_week_suffix(self, monday: date, expected: str) -> None:
        assert iso_week_partition_suffix(monday) == expected


class TestAddWeeks:
    def test_simple(self) -> None:
        assert add_weeks(date(2026, 1, 5), 4) == date(2026, 2, 2)

    def test_crosses_year_boundary(self) -> None:
        assert add_weeks(date(2025, 12, 29), 1) == date(2026, 1, 5)


# ---------------------------------------------------------------------------
# DDL builders
# ---------------------------------------------------------------------------
class TestBuildWeeklyPartitionDdl:
    table = PartitionedTable(
        name="fact_inventory_snapshot",
        partition_prefix="fact_inventory_snapshot",
        column_type="date",
        interval="week",
    )

    def test_partition_name_iso_format(self) -> None:
        # ISO week 1 of 2026 starts 2025-12-29.
        name, _ = build_weekly_partition_ddl(self.table, date(2025, 12, 29))
        assert name == "fact_inventory_snapshot_2026w01"
        # All-lowercase, ASCII, valid Postgres identifier characters.
        assert name == name.lower()
        assert all(c.isalnum() or c == "_" for c in name)

    def test_ddl_uses_inclusive_exclusive_bounds(self) -> None:
        _, ddl = build_weekly_partition_ddl(self.table, date(2026, 1, 5))
        assert "CREATE TABLE IF NOT EXISTS" in ddl
        assert "PARTITION OF fact_inventory_snapshot" in ddl
        # Mon..next Mon (exclusive upper bound, matches Postgres FOR VALUES FROM ... TO ...).
        assert "FROM (DATE '2026-01-05') TO (DATE '2026-01-12')" in ddl

    def test_rejects_non_monday(self) -> None:
        # Tuesday — must raise so callers always align to Monday first.
        with pytest.raises(ValueError, match="Monday"):
            build_weekly_partition_ddl(self.table, date(2026, 1, 6))

    def test_timestamptz_column_renders_tz_literal(self) -> None:
        ts_table = PartitionedTable(
            name="fact_external_signal",
            partition_prefix="fact_external_signal",
            column_type="timestamptz",
            interval="week",
        )
        _, ddl = build_weekly_partition_ddl(ts_table, date(2026, 1, 5))
        assert "TIMESTAMPTZ '2026-01-05 00:00:00+00'" in ddl
        assert "TIMESTAMPTZ '2026-01-12 00:00:00+00'" in ddl


class TestBuildPartitionDdlDispatch:
    def test_monthly_dispatch(self) -> None:
        t = PartitionedTable(
            name="fact_x", partition_prefix="fact_x", column_type="date", interval="month"
        )
        name, ddl = build_partition_ddl(t, date(2026, 5, 1))
        assert name == "fact_x_2026_05"
        assert "FROM (DATE '2026-05-01') TO (DATE '2026-06-01')" in ddl

    def test_weekly_dispatch(self) -> None:
        t = PartitionedTable(
            name="fact_x", partition_prefix="fact_x", column_type="date", interval="week"
        )
        name, ddl = build_partition_ddl(t, date(2026, 5, 4))  # Monday
        assert name == "fact_x_2026w19"
        assert "FROM (DATE '2026-05-04') TO (DATE '2026-05-11')" in ddl

    def test_monthly_and_weekly_names_dont_collide(self) -> None:
        """Monthly = ``YYYY_MM``; weekly = ``YYYYwWW``. The ``w`` separator
        guarantees no name collision even if the same prefix is reused."""
        t_m = PartitionedTable("p", "p", "date", interval="month")
        t_w = PartitionedTable("p", "p", "date", interval="week")
        name_m, _ = build_monthly_partition_ddl(t_m, date(2026, 1, 1))
        # ISO week 1 2026 starts 2025-12-29.
        name_w, _ = build_weekly_partition_ddl(t_w, date(2025, 12, 29))
        assert name_m != name_w
        assert "_" in name_m and "w" in name_w


# ---------------------------------------------------------------------------
# ensure_partitions — idempotency at the call layer (DB mocked)
# ---------------------------------------------------------------------------
def _make_conn_with_table_present() -> MagicMock:
    """Return a ``psycopg.Connection``-shaped mock where ``_table_exists`` is True."""
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = (1,)
    conn.cursor.return_value.__enter__.return_value = cur
    return conn


class TestEnsurePartitionsWeekly:
    table = PartitionedTable(
        name="fact_inventory_snapshot",
        partition_prefix="fact_inventory_snapshot",
        column_type="date",
        interval="week",
    )

    def test_emits_horizon_statements(self) -> None:
        conn = _make_conn_with_table_present()
        # 2026-05-05 (Tue) → ISO week starts Mon 2026-05-04.
        stmts = ensure_partitions(
            conn, self.table, horizon=4, today=date(2026, 5, 5), dry_run=True
        )
        assert len(stmts) == 4
        # Names span four consecutive ISO weeks starting at week 19.
        suffixes = [f"2026w{19 + i:02d}" for i in range(4)]
        for stmt, suffix in zip(stmts, suffixes, strict=True):
            assert f"fact_inventory_snapshot_{suffix}" in stmt

    def test_idempotent_uses_create_if_not_exists(self) -> None:
        """Re-running yields the same DDL (which is itself a no-op on the DB)."""
        conn = _make_conn_with_table_present()
        first = ensure_partitions(
            conn, self.table, horizon=3, today=date(2026, 5, 5), dry_run=True
        )
        second = ensure_partitions(
            conn, self.table, horizon=3, today=date(2026, 5, 5), dry_run=True
        )
        assert first == second
        for stmt in first:
            assert "CREATE TABLE IF NOT EXISTS" in stmt

    def test_skips_when_parent_table_missing(self) -> None:
        conn = MagicMock()
        cur = MagicMock()
        cur.fetchone.return_value = None  # _table_exists -> False
        conn.cursor.return_value.__enter__.return_value = cur
        stmts = ensure_partitions(
            conn, self.table, horizon=4, today=date(2026, 5, 5), dry_run=True
        )
        assert stmts == []


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_every_entry_has_known_interval(self) -> None:
        for t in PARTITIONED_TABLES:
            assert t.interval in {"month", "week"}

    def test_no_duplicate_table_names(self) -> None:
        names = [t.name for t in PARTITIONED_TABLES]
        assert len(names) == len(set(names))
