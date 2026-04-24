"""Tests for common.ai.rag (pgvector + BM25 hybrid retrieval)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from common.ai.rag import RagChunk, search, upsert_chunks, _embedding_to_pgvector


def test_embedding_to_pgvector_format():
    assert _embedding_to_pgvector([0.1, 0.2, 0.3]) == "[0.100000,0.200000,0.300000]"
    assert _embedding_to_pgvector(None) is None


def test_upsert_chunks_inserts_each_row():
    cursor = MagicMock()
    chunks = [
        RagChunk(doc_id="spec/a", chunk_index=0, source="spec", text="hello",
                 embedding=[0.1] * 4, metadata={"domain": "inv"}),
        RagChunk(doc_id="spec/a", chunk_index=1, source="spec", text="world",
                 embedding=[0.2] * 4),
    ]
    written = upsert_chunks(cursor, chunks)
    assert written == 2
    assert cursor.execute.call_count == 2
    # First call params include pgvector string.
    first_args = cursor.execute.call_args_list[0].args
    assert "INSERT INTO rag_chunk" in first_args[0]
    assert first_args[1][0] == "spec/a"            # doc_id
    assert first_args[1][1] == 0                    # chunk_index
    assert first_args[1][2] == "spec"               # source
    assert first_args[1][3] == "hello"              # text
    assert first_args[1][4].startswith("[0.10000")  # embedding
    assert json.loads(first_args[1][5]) == {"domain": "inv"}


def test_upsert_chunks_empty_is_noop():
    cursor = MagicMock()
    assert upsert_chunks(cursor, []) == 0
    cursor.execute.assert_not_called()


def test_search_k_zero_returns_empty():
    cursor = MagicMock()
    result = search(cursor, "anything", [0.1] * 4, k=0)
    assert result == []
    cursor.execute.assert_not_called()


def test_search_hybrid_fuses_vector_and_lexical():
    cursor = MagicMock()
    # Vector results, then lexical results (two fetchall calls).
    vec_rows = [
        (1, "docA", 0, "spec", "alpha", {"domain": "inv"}, 0.95),
        (2, "docB", 0, "spec", "beta",  {"domain": "inv"}, 0.85),
    ]
    lex_rows = [
        (2, "docB", 0, "spec", "beta",  {"domain": "inv"}, 0.70),
        (3, "docC", 0, "spec", "gamma", {"domain": "inv"}, 0.40),
    ]
    cursor.fetchall.side_effect = [vec_rows, lex_rows]
    out = search(cursor, "hello", [0.1] * 4, k=3)
    # doc 2 appears in both branches -> highest fused score.
    assert [c.id for c in out][0] == 2
    # All three distinct chunks returned.
    assert sorted(c.id for c in out) == [1, 2, 3]
    assert cursor.execute.call_count == 2
    for c in out:
        assert c.score is not None


def test_search_lexical_only_when_no_embedding():
    cursor = MagicMock()
    lex_rows = [
        (7, "docZ", 2, "runbook", "text", {}, 0.5),
    ]
    cursor.fetchall.side_effect = [lex_rows]
    out = search(cursor, "query", None, k=5)
    assert len(out) == 1
    assert out[0].id == 7
    # Only one execute call (lexical).
    assert cursor.execute.call_count == 1


def test_search_applies_filters():
    cursor = MagicMock()
    cursor.fetchall.side_effect = [[], []]
    _ = search(cursor, "q", [0.0] * 4, k=5, filters={"source": "spec", "metadata": {"domain": "inv"}})
    # Both calls must include "AND source = %s AND metadata @> %s" fragment.
    for call in cursor.execute.call_args_list:
        sql = call.args[0]
        assert "source = %s" in sql
        assert "metadata @> %s::jsonb" in sql


def test_search_handles_string_metadata_json():
    cursor = MagicMock()
    lex_rows = [(1, "d", 0, "spec", "t", json.dumps({"k": "v"}), 0.1)]
    cursor.fetchall.side_effect = [lex_rows]
    out = search(cursor, "q", None, k=3)
    assert out[0].metadata == {"k": "v"}


def test_upsert_chunks_raises_on_cursor_error():
    cursor = MagicMock()
    cursor.execute.side_effect = ValueError("bad embedding")
    chunks = [RagChunk(doc_id="a", chunk_index=0, source="spec", text="x")]
    with pytest.raises(ValueError):
        upsert_chunks(cursor, chunks)
