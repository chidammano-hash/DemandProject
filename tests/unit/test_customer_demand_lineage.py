from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from common.services.customer_demand_lineage import (
    CUSTOMER_DEMAND_LOAD_LOCK_KEY,
    customer_demand_snapshot_lock,
    customer_demand_snapshot_locked,
)


def test_snapshot_decorator_commits_lock_before_consumer_and_releases_after_commit() -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    events: list[str] = []
    cur.execute.side_effect = lambda sql, *_args: events.append(
        "unlock" if "unlock_shared" in sql else "lock"
    )
    conn.commit.side_effect = lambda: events.append("commit")
    conn.rollback.side_effect = lambda: events.append("rollback")

    @customer_demand_snapshot_locked
    def consume(_conn):
        events.append("consume")
        return "done"

    assert consume(conn) == "done"
    assert events == ["lock", "commit", "consume", "commit", "rollback", "unlock", "commit"]
    assert cur.execute.call_args_list[0].args[1] == (CUSTOMER_DEMAND_LOAD_LOCK_KEY,)


def test_snapshot_decorator_rolls_back_and_unlocks_after_consumer_failure() -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur

    @customer_demand_snapshot_locked
    def consume(_conn):
        raise ValueError("bad snapshot")

    with pytest.raises(ValueError, match="bad snapshot"):
        consume(conn)

    assert conn.rollback.call_count == 1
    unlock_sql = cur.execute.call_args_list[-1].args[0]
    assert "pg_advisory_unlock_shared" in unlock_sql


def test_snapshot_decorator_accepts_keyword_connection() -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur

    @customer_demand_snapshot_locked
    def consume(*, conn):
        return conn

    assert consume(conn=conn) is conn


def test_consumer_snapshot_starts_after_a_waiting_loader_publishes() -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    state = {"published_batch": 91, "lock_committed": False}

    def execute(sql, *_args):
        if "pg_advisory_lock_shared" in sql:
            # Model an exclusive loader completing while this shared lock waits.
            state["published_batch"] = 92

    def commit():
        state["lock_committed"] = True

    cur.execute.side_effect = execute
    conn.commit.side_effect = commit

    @customer_demand_snapshot_locked
    def consume(_conn):
        assert state["lock_committed"] is True
        return state["published_batch"]

    assert consume(conn) == 92


def test_snapshot_lock_context_holds_lock_for_caller_owned_transaction() -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    events: list[str] = []
    cur.execute.side_effect = lambda sql, *_args: events.append(
        "unlock" if "unlock_shared" in sql else "lock"
    )
    conn.commit.side_effect = lambda: events.append("commit")
    conn.rollback.side_effect = lambda: events.append("rollback")

    with customer_demand_snapshot_lock(conn):
        events.append("transaction")

    assert events == ["lock", "commit", "transaction", "rollback", "unlock", "commit"]
