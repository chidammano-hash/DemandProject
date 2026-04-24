"""Tests for common.ai.envelope (multimodal response envelope)."""

from __future__ import annotations

import pytest

from common.ai.envelope import (
    VALID_BLOCK_KINDS,
    ResponseBlock,
    ResponseEnvelope,
)


def test_block_valid_kinds_accepted():
    for kind in VALID_BLOCK_KINDS:
        b = ResponseBlock(kind=kind, payload={"k": 1})
        assert b.kind == kind


def test_block_invalid_kind_raises():
    with pytest.raises(ValueError):
        ResponseBlock(kind="unknown", payload={})


def test_block_payload_must_be_dict():
    with pytest.raises(ValueError):
        ResponseBlock(kind="markdown", payload="not-a-dict")  # type: ignore[arg-type]


def test_block_round_trip():
    original = ResponseBlock(kind="chart", payload={"type": "line", "series": [1, 2, 3]})
    d = original.to_dict()
    assert d == {"kind": "chart", "payload": {"type": "line", "series": [1, 2, 3]}}
    restored = ResponseBlock.from_dict(d)
    assert restored.kind == original.kind
    assert restored.payload == original.payload


def test_block_from_dict_missing_kind_raises():
    with pytest.raises(ValueError):
        ResponseBlock.from_dict({"payload": {}})


def test_envelope_round_trip():
    env = ResponseEnvelope(
        narrative="Here are the top risks.",
        blocks=[
            ResponseBlock(kind="markdown", payload={"text": "**risk summary**"}),
            ResponseBlock(kind="table", payload={"columns": ["a"], "rows": [[1]]}),
        ],
        trace_id="trace_123",
    )
    d = env.to_dict()
    assert d["narrative"] == "Here are the top risks."
    assert d["trace_id"] == "trace_123"
    assert len(d["blocks"]) == 2

    restored = ResponseEnvelope.from_dict(d)
    assert restored.narrative == env.narrative
    assert restored.trace_id == env.trace_id
    assert [b.kind for b in restored.blocks] == ["markdown", "table"]


def test_envelope_empty_blocks_ok():
    env = ResponseEnvelope(narrative="just text")
    d = env.to_dict()
    assert d["blocks"] == []
    assert d["trace_id"] is None


def test_envelope_from_dict_requires_narrative():
    with pytest.raises(ValueError):
        ResponseEnvelope.from_dict({"blocks": []})


def test_envelope_from_dict_unknown_block_kind_raises():
    with pytest.raises(ValueError):
        ResponseEnvelope.from_dict({
            "narrative": "x",
            "blocks": [{"kind": "bogus", "payload": {}}],
        })


def test_envelope_from_dict_rejects_non_list_blocks():
    with pytest.raises(ValueError):
        ResponseEnvelope.from_dict({"narrative": "x", "blocks": "not-a-list"})


def test_envelope_from_dict_rejects_non_string_narrative():
    with pytest.raises(ValueError):
        ResponseEnvelope.from_dict({"narrative": 123, "blocks": []})


def test_envelope_trace_id_type_check():
    with pytest.raises(ValueError):
        ResponseEnvelope.from_dict({"narrative": "x", "blocks": [], "trace_id": 42})
