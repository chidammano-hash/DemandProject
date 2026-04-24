"""Tests for common.feature_store (Gen-4 Cross-cutting #5)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from common.feature_store import (
    FeatureView,
    get_point_in_time_features,
    register_feature_view,
)


class _FakeCursor:
    """Tiny cursor that records executes and plays back a queue of results."""

    def __init__(self, results: list | None = None):
        self.results: list = results or []
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, sql, params=()):
        self.executed.append((sql, tuple(params)))

    def fetchone(self):
        if not self.results:
            return None
        head = self.results.pop(0)
        # Allow callers to enqueue a single row or a list of rows (take first).
        if isinstance(head, list):
            return head[0] if head else None
        return head

    def fetchall(self):
        if not self.results:
            return []
        head = self.results.pop(0)
        return head if isinstance(head, list) else [head]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, cursor: _FakeCursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


# ---------------------------------------------------------------------------
# register_feature_view
# ---------------------------------------------------------------------------


def test_register_feature_view_creates_entity_and_view():
    results = [
        None,              # SELECT entity -> not present
        (5,),              # INSERT entity RETURNING id
        None,              # SELECT feature view -> not present
        (99,),             # INSERT feature view RETURNING id
    ]
    cur = _FakeCursor(results)
    conn = _FakeConn(cur)

    fv = FeatureView(
        name="sku_demand_features",
        entity_name="sku",
        entity_keys=["item_id", "loc"],
        features=["rolling_12m_mean", "zero_demand_pct"],
        source_table="dim_sku",
        event_ts_col="modified_ts",
    )
    fv_id = register_feature_view(conn, fv)

    assert fv_id == 99
    assert len(cur.executed) == 4
    assert "INSERT INTO feature_store_entity" in cur.executed[1][0]
    assert "INSERT INTO feature_store_feature_view" in cur.executed[3][0]


def test_register_feature_view_is_idempotent():
    # Entity already exists; feature view already exists -> UPDATE path.
    results = [
        (7, ["item_id", "loc"]),   # SELECT entity hit
        (42,),                     # SELECT feature view hit
    ]
    cur = _FakeCursor(results)
    conn = _FakeConn(cur)

    fv = FeatureView(
        name="sku_demand_features",
        entity_name="sku",
        entity_keys=["item_id", "loc"],
        features=["rolling_12m_mean"],
        source_table="dim_sku",
    )
    fv_id = register_feature_view(conn, fv)
    assert fv_id == 42
    # Last executed should be an UPDATE, not an INSERT
    last_sql = cur.executed[-1][0]
    assert "UPDATE feature_store_feature_view" in last_sql


def test_register_feature_view_validates_required_fields():
    cur = _FakeCursor([])
    conn = _FakeConn(cur)
    with pytest.raises(ValueError):
        register_feature_view(
            conn,
            FeatureView(
                name="",
                entity_name="sku",
                entity_keys=["item_id"],
                features=["x"],
                source_table="t",
            ),
        )


# ---------------------------------------------------------------------------
# get_point_in_time_features
# ---------------------------------------------------------------------------


def test_point_in_time_lookup_joins_history_and_orders_by_row_idx():
    # First call: _load_feature_view returns a single row
    view_meta = (
        "sku_demand_features",   # fv.name
        "sku",                   # entity.name
        ["item_id", "loc"],      # entity.entity_keys
        ["rolling_12m_mean"],    # fv.features
        "dim_sku",               # source_table
        "modified_ts",           # event_ts_col
        "dim_sku_history",       # history_table
        "owner",
        "desc",
    )
    # Second call: the PIT SELECT returns rows with (row_idx, item_id, loc, feature)
    pit_rows = [
        (0, "ITEM_A", "LOC1", 10.5),
        (1, "ITEM_B", "LOC2", None),
    ]
    cur = _FakeCursor([view_meta, pit_rows])
    conn = _FakeConn(cur)

    entities = [
        {"item_id": "ITEM_A", "loc": "LOC1"},
        {"item_id": "ITEM_B", "loc": "LOC2"},
    ]
    results = get_point_in_time_features(
        conn,
        entities,
        view_name="sku_demand_features",
        as_of_ts="2026-04-01",
    )
    assert results == [
        {"item_id": "ITEM_A", "loc": "LOC1", "rolling_12m_mean": 10.5},
        {"item_id": "ITEM_B", "loc": "LOC2", "rolling_12m_mean": None},
    ]
    # Verify the second query joined against the history table and ordered by row_idx
    pit_sql = cur.executed[1][0]
    assert "dim_sku_history" in pit_sql
    assert "LEFT JOIN LATERAL" in pit_sql
    assert "ORDER BY e.row_idx" in pit_sql
    # as_of_ts must be the last bind param after the VALUES rows.
    pit_params = cur.executed[1][1]
    assert pit_params[-1] == "2026-04-01"


def test_point_in_time_lookup_empty_entities_returns_empty_list():
    cur = _FakeCursor([])
    conn = _FakeConn(cur)
    assert get_point_in_time_features(conn, [], "any_view", "2026-04-01") == []


def test_point_in_time_lookup_rejects_missing_keys():
    view_meta = (
        "sku_demand_features",
        "sku",
        ["item_id", "loc"],
        ["rolling_12m_mean"],
        "dim_sku",
        "modified_ts",
        None,
        None,
        None,
    )
    cur = _FakeCursor([view_meta])
    conn = _FakeConn(cur)
    with pytest.raises(ValueError):
        get_point_in_time_features(
            conn,
            [{"item_id": "X"}],            # missing "loc"
            view_name="sku_demand_features",
            as_of_ts="2026-04-01",
        )


def test_point_in_time_lookup_raises_on_unknown_view():
    cur = _FakeCursor([None])
    conn = _FakeConn(cur)
    with pytest.raises(KeyError):
        get_point_in_time_features(
            conn, [{"item_id": "X", "loc": "L"}], "missing_view", "2026-04-01"
        )
