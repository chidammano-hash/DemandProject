"""Unit tests for scripts/ml/restore_cluster_assignments.py.

The repair re-applies the promoted cluster-label assignment onto
dim_sku.ml_cluster after a dim_sku reload wipes it (which otherwise collapses
every per-cluster tree forecast). These tests pin the CSV validation + load
contract; the DB UPDATE is exercised via the live pipeline.
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
    restore_ml_cluster,
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
# restore_ml_cluster — DB contract (assignment table + dim_sku cache)
# ---------------------------------------------------------------------------
# A small in-memory fake stands in for psycopg: dim_sku is a {sku_ck: ml_cluster}
# map, sku_cluster_assignment is a {(experiment_id, sku_ck): label} map, and
# the function's COPY + UPSERT + cache update are emulated so the re-apply /
# idempotency / full-grain-join behaviour is pinned without a live DB.


class _FakeCursor:
    """Emulates the exact SQL restore_ml_cluster issues against a dim_sku map."""

    def __init__(
        self,
        dim_sku: dict[str, str | None],
        assignments: dict[tuple[int, str], str],
    ):
        self.dim_sku = dim_sku          # sku_ck -> ml_cluster (full-grain key)
        self.assignments = assignments
        self._updates: dict[str, tuple[str | None, str]] = {}
        self._last_count = 0
        self._last_one = None
        self.commits = 0
        self.rollbacks = 0
        self.rowcount = 0

    def execute(self, sql, params=None):
        s = " ".join(sql.split())
        if s.startswith("SELECT experiment_id FROM cluster_experiment"):
            self._last_one = (7,)
        elif "CREATE TEMP TABLE _cluster_updates" in s:
            self._updates = {}
        elif s.startswith("SELECT COUNT(*)") and "_cluster_updates" in s:
            # rows that match on the full sku_ck grain AND would change
            self._last_count = sum(
                1 for ck, (_, lbl) in self._updates.items()
                if ck in self.dim_sku and self.dim_sku[ck] != lbl
            )
        elif s.startswith("INSERT INTO sku_cluster_assignment"):
            experiment_id = int(params[0])
            changed = 0
            for ck, (_, lbl) in self._updates.items():
                if ck in self.dim_sku:
                    self.assignments[(experiment_id, ck)] = lbl
                    changed += 1
            self.rowcount = changed
        elif s.startswith("UPDATE dim_sku"):
            changed = 0
            for ck, (_, lbl) in self._updates.items():
                # full-grain match: join is on d.sku_ck = u.sku_ck (no fan-out)
                if ck in self.dim_sku and self.dim_sku[ck] != lbl:
                    self.dim_sku[ck] = lbl
                    changed += 1
            self.rowcount = changed
        else:  # pragma: no cover — unexpected SQL fails loud
            raise AssertionError(f"unexpected SQL: {s}")

    def fetchone(self):
        if self._last_one is not None:
            row = self._last_one
            self._last_one = None
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
    def __init__(self, dim_sku):
        self.assignments: dict[tuple[int, str], str] = {}
        self._cur = _FakeCursor(dim_sku, self.assignments)

    def cursor(self):
        return self._cur

    def commit(self):
        self._cur.commits += 1

    def rollback(self):
        self._cur.rollbacks += 1


def _df(rows):
    return pd.DataFrame(rows)


def test_restore_reapplies_labels_to_null_rows():
    """Every sku_ck in the source gets its label written onto a NULL dim_sku row."""
    dim_sku = {
        "84587_ALL_1401-BULK": None,
        "11286_ALL_1401-BULK": None,
        "99999_ALL_9999-BULK": None,  # not in source -> untouched
    }
    conn = _FakeConn(dim_sku)
    df = _df([
        {"sku_ck": "84587_ALL_1401-BULK", "cluster_label": "very_high_volume_periodic"},
        {"sku_ck": "11286_ALL_1401-BULK", "cluster_label": "high_volume_periodic"},
    ])

    updated = restore_ml_cluster(df, conn)

    assert updated == 2
    assert dim_sku["84587_ALL_1401-BULK"] == "very_high_volume_periodic"
    assert dim_sku["11286_ALL_1401-BULK"] == "high_volume_periodic"
    assert dim_sku["99999_ALL_9999-BULK"] is None  # absent from source -> NULL
    assert conn.assignments[(7, "84587_ALL_1401-BULK")] == "very_high_volume_periodic"
    assert conn.assignments[(7, "11286_ALL_1401-BULK")] == "high_volume_periodic"
    assert conn._cur.commits == 1


def test_restore_is_idempotent_second_run_noop():
    """Re-running with labels already present updates zero rows (idempotent)."""
    dim_sku = {"84587_ALL_1401-BULK": None}
    df = _df([{"sku_ck": "84587_ALL_1401-BULK", "cluster_label": "very_high_volume_periodic"}])

    conn1 = _FakeConn(dim_sku)
    assert restore_ml_cluster(df, conn1) == 1

    conn2 = _FakeConn(dim_sku)  # labels now present
    assert restore_ml_cluster(df, conn2) == 0
    assert dim_sku["84587_ALL_1401-BULK"] == "very_high_volume_periodic"
    assert conn2.assignments[(7, "84587_ALL_1401-BULK")] == "very_high_volume_periodic"


def test_restore_dry_run_does_not_write():
    """--dry-run counts the would-be changes but rolls back without mutating."""
    dim_sku = {"84587_ALL_1401-BULK": None}
    conn = _FakeConn(dim_sku)
    df = _df([{"sku_ck": "84587_ALL_1401-BULK", "cluster_label": "very_high_volume_periodic"}])

    would_change = restore_ml_cluster(df, conn, dry_run=True)

    assert would_change == 1
    assert dim_sku["84587_ALL_1401-BULK"] is None  # unchanged
    assert conn._cur.rollbacks == 1
    assert conn._cur.commits == 0


def test_restore_matches_full_sku_ck_grain_no_fanout():
    """Two SKUs sharing (item_id, loc) but differing in customer_group are
    distinct sku_ck keys — each gets exactly its own label (no fan-out)."""
    # Same item_id (84587) + loc (1401-BULK), different customer_group (ALL vs ON)
    dim_sku = {
        "84587_ALL_1401-BULK": None,
        "84587_ON_1401-BULK": None,
    }
    conn = _FakeConn(dim_sku)
    df = _df([
        {"sku_ck": "84587_ALL_1401-BULK", "cluster_label": "high_volume_periodic"},
        {"sku_ck": "84587_ON_1401-BULK", "cluster_label": "low_volume_intermittent"},
    ])

    updated = restore_ml_cluster(df, conn)

    assert updated == 2  # each grain row touched once, not fanned out
    assert dim_sku["84587_ALL_1401-BULK"] == "high_volume_periodic"
    assert dim_sku["84587_ON_1401-BULK"] == "low_volume_intermittent"
