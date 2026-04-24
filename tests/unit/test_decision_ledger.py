"""Tests for the hash-chained AI decision ledger."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from common.ai.decision_ledger import (
    GENESIS_HASH,
    DecisionRecord,
    append_decision,
    compute_row_hash,
    verify_chain,
)


def _make_record(**overrides):
    base = {
        "agent_id": "demand_agent",
        "action_type": "promote_model",
        "autonomy_tier": "suggestive",
        "subject_kind": "model_id",
        "subject_id": "lgbm_v5",
        "payload": {"wape_improvement": 0.04},
        "policy_id": "demand.promote_champion_model",
    }
    base.update(overrides)
    return DecisionRecord(**base)


def test_compute_row_hash_is_deterministic():
    rec = _make_record()
    h1 = compute_row_hash(rec, GENESIS_HASH)
    h2 = compute_row_hash(rec, GENESIS_HASH)
    assert h1 == h2
    assert len(h1) == 64


def test_compute_row_hash_changes_with_payload():
    rec1 = _make_record(payload={"x": 1})
    rec2 = _make_record(payload={"x": 2})
    assert compute_row_hash(rec1, GENESIS_HASH) != compute_row_hash(rec2, GENESIS_HASH)


def test_compute_row_hash_changes_with_prev_hash():
    rec = _make_record()
    assert compute_row_hash(rec, GENESIS_HASH) != compute_row_hash(rec, "a" * 64)


def test_append_decision_first_row_uses_genesis():
    cursor = MagicMock()
    # _fetch_latest_hash returns None → genesis
    cursor.fetchone.side_effect = [None, (42,)]

    new_id, row_hash = append_decision(cursor, _make_record())

    assert new_id == 42
    # The INSERT statement should have been called with prev_hash = genesis
    insert_call = cursor.execute.call_args_list[1]
    params = insert_call[0][1]
    assert params[7] == GENESIS_HASH  # prev_hash position
    assert params[8] == row_hash      # row_hash position


def test_append_decision_links_to_prior_row():
    cursor = MagicMock()
    prior_hash = "b" * 64
    cursor.fetchone.side_effect = [(prior_hash,), (43,)]

    new_id, row_hash = append_decision(cursor, _make_record())

    insert_call = cursor.execute.call_args_list[1]
    params = insert_call[0][1]
    assert params[7] == prior_hash
    assert row_hash == compute_row_hash(_make_record(), prior_hash)


def test_append_decision_rejects_invalid_tier():
    cursor = MagicMock()
    with pytest.raises(ValueError, match="Invalid autonomy_tier"):
        append_decision(cursor, _make_record(autonomy_tier="god_mode"))


def test_append_decision_rejects_missing_agent_id():
    cursor = MagicMock()
    with pytest.raises(ValueError, match="agent_id and action_type"):
        append_decision(cursor, _make_record(agent_id=""))


def test_verify_chain_empty_ledger_is_ok():
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    ok, errors = verify_chain(cursor)
    assert ok is True
    assert errors == []


def test_verify_chain_good_chain():
    rec1 = _make_record(subject_id="m1")
    rec2 = _make_record(subject_id="m2")
    h1 = compute_row_hash(rec1, GENESIS_HASH)
    h2 = compute_row_hash(rec2, h1)

    cursor = MagicMock()
    cursor.fetchall.return_value = [
        (
            1, rec1.agent_id, rec1.action_type, rec1.autonomy_tier,
            rec1.subject_kind, rec1.subject_id,
            json.dumps(rec1.payload, sort_keys=True, separators=(",", ":")),
            rec1.policy_id, GENESIS_HASH, h1,
        ),
        (
            2, rec2.agent_id, rec2.action_type, rec2.autonomy_tier,
            rec2.subject_kind, rec2.subject_id,
            json.dumps(rec2.payload, sort_keys=True, separators=(",", ":")),
            rec2.policy_id, h1, h2,
        ),
    ]
    ok, errors = verify_chain(cursor)
    assert ok is True
    assert errors == []


def test_verify_chain_detects_hash_tamper():
    rec = _make_record()
    h1 = compute_row_hash(rec, GENESIS_HASH)
    tampered = "c" * 64

    cursor = MagicMock()
    cursor.fetchall.return_value = [
        (
            1, rec.agent_id, rec.action_type, rec.autonomy_tier,
            rec.subject_kind, rec.subject_id,
            json.dumps(rec.payload, sort_keys=True, separators=(",", ":")),
            rec.policy_id, GENESIS_HASH, tampered,
        ),
    ]
    ok, errors = verify_chain(cursor)
    assert ok is False
    assert len(errors) == 1
    assert errors[0]["id"] == 1
    assert "row_hash mismatch" in errors[0]["reason"]
    # Sanity: the expected hash appears in the error message
    assert h1 in errors[0]["reason"]


def test_verify_chain_detects_prev_hash_break():
    rec1 = _make_record(subject_id="m1")
    rec2 = _make_record(subject_id="m2")
    h1 = compute_row_hash(rec1, GENESIS_HASH)
    wrong_prev = "d" * 64
    # Recompute h2 using the wrong prev_hash so row_hash is self-consistent
    # but the chain linkage is broken.
    h2 = compute_row_hash(rec2, wrong_prev)

    cursor = MagicMock()
    cursor.fetchall.return_value = [
        (
            1, rec1.agent_id, rec1.action_type, rec1.autonomy_tier,
            rec1.subject_kind, rec1.subject_id,
            json.dumps(rec1.payload, sort_keys=True, separators=(",", ":")),
            rec1.policy_id, GENESIS_HASH, h1,
        ),
        (
            2, rec2.agent_id, rec2.action_type, rec2.autonomy_tier,
            rec2.subject_kind, rec2.subject_id,
            json.dumps(rec2.payload, sort_keys=True, separators=(",", ":")),
            rec2.policy_id, wrong_prev, h2,
        ),
    ]
    ok, errors = verify_chain(cursor)
    assert ok is False
    # First error: prev_hash break on row 2
    prev_break = [e for e in errors if "prev_hash mismatch" in e["reason"]]
    assert prev_break and prev_break[0]["id"] == 2
