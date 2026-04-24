"""Unit tests for common.ai.lineage."""
from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import pytest

from common.ai.lineage import VALID_KINDS, emit_event


def _make_cursor(returned_id: int = 42) -> MagicMock:
    cur = MagicMock()
    cur.fetchone.return_value = (returned_id,)
    return cur


def test_emit_event_inserts_row_and_returns_id():
    cur = _make_cursor(returned_id=7)
    new_id = emit_event(
        cur,
        kind="COMPLETE",
        job_id="promote_model",
        inputs=[{"namespace": "pg", "name": "staging"}],
        outputs=[{"namespace": "pg", "name": "production"}],
        facets={"model_id": "lgbm_cluster"},
    )
    assert new_id == 7
    cur.execute.assert_called_once()
    args = cur.execute.call_args.args
    sql, params = args[0], args[1]
    assert "INSERT INTO fact_lineage_event" in sql
    assert params[0] == "COMPLETE"
    assert params[1] == "promote_model"
    # params[2] is a UUID string auto-generated
    uuid.UUID(params[2])
    # inputs/outputs/facets serialized as JSON strings
    assert json.loads(params[3]) == [{"namespace": "pg", "name": "staging"}]
    assert json.loads(params[4]) == [{"namespace": "pg", "name": "production"}]
    assert json.loads(params[5]) == {"model_id": "lgbm_cluster"}


def test_emit_event_rejects_bad_kind():
    cur = _make_cursor()
    with pytest.raises(ValueError):
        emit_event(cur, kind="BOGUS", job_id="x")


def test_emit_event_rejects_empty_job_id():
    cur = _make_cursor()
    with pytest.raises(ValueError):
        emit_event(cur, kind="START", job_id="")


def test_emit_event_accepts_explicit_run_id():
    cur = _make_cursor(returned_id=1)
    my_run = str(uuid.uuid4())
    emit_event(cur, kind="START", job_id="backtest", run_id=my_run)
    params = cur.execute.call_args.args[1]
    assert params[2] == my_run


def test_valid_kinds_expected_set():
    assert VALID_KINDS == frozenset({"START", "COMPLETE", "FAIL", "ABORT"})
