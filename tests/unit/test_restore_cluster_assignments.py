"""Unit tests for scripts/ml/restore_cluster_assignments.py.

The repair re-applies the promoted cluster-label assignment into
sku_cluster_assignment when that durable table is empty or stale. These tests pin
the CSV validation + assignment-table write contract.
"""

import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ml.restore_cluster_assignments import (  # noqa: E402
    load_assignments,
    restore_cluster_assignments,
)


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "cluster_labels.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_load_assignments_valid(tmp_path):
    """A well-formed CSV loads with sku_ck + cluster_label as strings."""
    label = "very_high_volume_periodic"
    csv = _write_csv(tmp_path, [
        {"sku_ck": "84587_ALL_1401-BULK", "cluster_id": 5, "cluster_label": label},
        {"sku_ck": "11286_ALL_1401-BULK", "cluster_id": 5, "cluster_label": label},
    ])
    df = load_assignments(csv)
    assert len(df) == 2
    assert df["sku_ck"].dtype == object
    assert df["cluster_label"].dtype == object
    assert set(df["cluster_label"]) == {label}


def test_load_assignments_missing_file(tmp_path):
    """Absent CSV raises FileNotFoundError with a remediation hint."""
    with pytest.raises(FileNotFoundError):
        load_assignments(tmp_path / "does_not_exist.csv")


def test_load_assignments_missing_required_column(tmp_path):
    """A CSV without cluster_label is rejected (would silently no-op otherwise)."""
    csv = _write_csv(tmp_path, [{"sku_ck": "84587_ALL_1401-BULK", "cluster_id": 5}])
    with pytest.raises(ValueError, match="cluster_label"):
        load_assignments(csv)


def test_load_assignments_drops_null_rows(tmp_path):
    """Rows with a null sku_ck or cluster_label are dropped, not written as NULLs."""
    csv = _write_csv(tmp_path, [
        {"sku_ck": "84587_ALL_1401-BULK", "cluster_label": "very_high_volume_periodic"},
        {"sku_ck": "X_ALL_Y", "cluster_label": None},
        {"sku_ck": None, "cluster_label": "low_vol"},
    ])
    df = load_assignments(csv)
    assert len(df) == 1
    assert df.iloc[0]["sku_ck"] == "84587_ALL_1401-BULK"


# ---------------------------------------------------------------------------
# restore_cluster_assignments — DB contract (assignment table only)
# ---------------------------------------------------------------------------
# A small in-memory fake stands in for psycopg: dim_sku is a set of valid sku_ck
# rows, sku_cluster_assignment is a {(experiment_id, sku_ck): label} map, and
# the function's COPY + UPSERT are emulated so re-apply / idempotency /
# full-grain-join behaviour is pinned without a live DB.


class _FakeCursor:
    """Emulates the exact SQL restore_cluster_assignments issues."""

    def __init__(
        self,
        dim_sku_keys: set[str],
        assignments: dict[tuple[int, str], str],
        promoted_experiment_id: int | None = 7,
    ):
        self.dim_sku_keys = dim_sku_keys
        self.assignments = assignments
        self.promoted_experiment_id = promoted_experiment_id
        self._updates: dict[str, tuple[str | None, str]] = {}
        self._last_count = 0
        self._last_one = None
        self._has_last_one = False
        self.commits = 0
        self.rollbacks = 0
        self.rowcount = 0

    def execute(self, sql, params=None):
        s = " ".join(sql.split())
        if s.startswith("SELECT experiment_id FROM cluster_experiment"):
            self._last_one = (
                (self.promoted_experiment_id,)
                if self.promoted_experiment_id is not None
                else None
            )
            self._has_last_one = True
        elif "CREATE TEMP TABLE _cluster_updates" in s:
            self._updates = {}
        elif s.startswith("SELECT COUNT(*)") and "_cluster_updates" in s:
            # rows that match on the full sku_ck grain AND would change
            experiment_id = int(params[0])
            self._last_count = sum(
                1 for ck, (_, lbl) in self._updates.items()
                if ck in self.dim_sku_keys
                and self.assignments.get((experiment_id, ck)) != lbl
            )
        elif s.startswith("INSERT INTO sku_cluster_assignment"):
            experiment_id = int(params[0])
            changed = 0
            for ck, (_, lbl) in self._updates.items():
                if (
                    ck in self.dim_sku_keys
                    and self.assignments.get((experiment_id, ck)) != lbl
                ):
                    self.assignments[(experiment_id, ck)] = lbl
                    changed += 1
            self.rowcount = changed
        else:  # pragma: no cover — unexpected SQL fails loud
            raise AssertionError(f"unexpected SQL: {s}")

    def fetchone(self):
        if self._has_last_one:
            row = self._last_one
            self._last_one = None
            self._has_last_one = False
            return row
        return (self._last_count,)

    @contextmanager
    def copy(self, _sql):
        yield self  # COPY ctx: write_row stages into _updates

    def write_row(self, row):
        if len(row) == 2:
            sku_ck, cluster_label = row
            cluster_id = None
        else:
            sku_ck, cluster_id, cluster_label = row
        self._updates[sku_ck] = (cluster_id, cluster_label)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, dim_sku_keys, promoted_experiment_id: int | None = 7):
        self.assignments: dict[tuple[int, str], str] = {}
        self._cur = _FakeCursor(set(dim_sku_keys), self.assignments, promoted_experiment_id)

    def cursor(self):
        return self._cur

    def commit(self):
        self._cur.commits += 1

    def rollback(self):
        self._cur.rollbacks += 1


def _df(rows):
    return pd.DataFrame(rows)


def test_restore_reapplies_labels_to_assignment_table():
    """Every source sku_ck that exists in dim_sku gets written to the assignment table."""
    dim_sku_keys = {
        "84587_ALL_1401-BULK",
        "11286_ALL_1401-BULK",
        "99999_ALL_9999-BULK",  # not in source -> untouched
    }
    conn = _FakeConn(dim_sku_keys)
    df = _df([
        {"sku_ck": "84587_ALL_1401-BULK", "cluster_label": "very_high_volume_periodic"},
        {"sku_ck": "11286_ALL_1401-BULK", "cluster_label": "high_volume_periodic"},
    ])

    updated = restore_cluster_assignments(df, conn)

    assert updated == 2
    assert conn.assignments[(7, "84587_ALL_1401-BULK")] == "very_high_volume_periodic"
    assert conn.assignments[(7, "11286_ALL_1401-BULK")] == "high_volume_periodic"
    assert (7, "99999_ALL_9999-BULK") not in conn.assignments
    assert conn._cur.commits == 1


def test_restore_is_idempotent_second_run_noop():
    """Re-running with labels already present updates zero rows (idempotent)."""
    dim_sku_keys = {"84587_ALL_1401-BULK"}
    df = _df([{"sku_ck": "84587_ALL_1401-BULK", "cluster_label": "very_high_volume_periodic"}])

    conn1 = _FakeConn(dim_sku_keys)
    assert restore_cluster_assignments(df, conn1) == 1

    conn2 = _FakeConn(dim_sku_keys)
    conn2.assignments[(7, "84587_ALL_1401-BULK")] = "very_high_volume_periodic"
    assert restore_cluster_assignments(df, conn2) == 0
    assert conn2.assignments[(7, "84587_ALL_1401-BULK")] == "very_high_volume_periodic"


def test_restore_dry_run_does_not_write():
    """--dry-run counts the would-be changes but rolls back without mutating."""
    dim_sku_keys = {"84587_ALL_1401-BULK"}
    conn = _FakeConn(dim_sku_keys)
    df = _df([{"sku_ck": "84587_ALL_1401-BULK", "cluster_label": "very_high_volume_periodic"}])

    would_change = restore_cluster_assignments(df, conn, dry_run=True)

    assert would_change == 1
    assert conn.assignments == {}
    assert conn._cur.rollbacks == 1
    assert conn._cur.commits == 0


def test_restore_matches_full_sku_ck_grain_no_fanout():
    """Two SKUs sharing (item_id, loc) but differing in customer_group are
    distinct sku_ck keys — each gets exactly its own label (no fan-out)."""
    # Same item_id (84587) + loc (1401-BULK), different customer_group (ALL vs ON)
    dim_sku_keys = {
        "84587_ALL_1401-BULK",
        "84587_ON_1401-BULK",
    }
    conn = _FakeConn(dim_sku_keys)
    df = _df([
        {"sku_ck": "84587_ALL_1401-BULK", "cluster_label": "high_volume_periodic"},
        {"sku_ck": "84587_ON_1401-BULK", "cluster_label": "low_volume_intermittent"},
    ])

    updated = restore_cluster_assignments(df, conn)

    assert updated == 2  # each grain row touched once, not fanned out
    assert conn.assignments[(7, "84587_ALL_1401-BULK")] == "high_volume_periodic"
    assert conn.assignments[(7, "84587_ON_1401-BULK")] == "low_volume_intermittent"


def test_restore_requires_promoted_experiment():
    """Without a promoted experiment there is no valid assignment generation."""
    conn = _FakeConn({"84587_ALL_1401-BULK"}, promoted_experiment_id=None)
    df = _df([{"sku_ck": "84587_ALL_1401-BULK", "cluster_label": "very_high_volume_periodic"}])

    with pytest.raises(RuntimeError, match="No promoted cluster_experiment"):
        restore_cluster_assignments(df, conn)
