"""Tests for scripts.ai.ingest_docs — chunking + skip-empty cases."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.ai.ingest_docs import (
    build_chunks,
    chunk_text,
    ingest,
    run,
)


def test_chunk_text_splits_with_overlap():
    text = "a" * 1200
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    # Stride = 450; chunks at offsets 0, 450, 900 -> 3 chunks.
    assert len(chunks) == 3
    assert all(len(c) <= 500 for c in chunks)
    # First and second chunks share the overlap region.
    assert chunks[0][-50:] == chunks[1][:50]


def test_chunk_text_empty_returns_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_chunk_text_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=0, chunk_overlap=0)


def test_build_chunks_walks_directory(tmp_path: Path):
    (tmp_path / "a.md").write_text("hello " * 200)
    (tmp_path / "b.md").write_text("")          # empty -> skipped
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "c.md").write_text("another doc")

    chunks = build_chunks(tmp_path, source_label="sop")

    doc_ids = {c.doc_id for c in chunks}
    assert "a.md" in doc_ids
    assert "nested/c.md" in doc_ids
    assert "b.md" not in doc_ids
    # All chunks carry the source label and a zero-vector embedding placeholder.
    for c in chunks:
        assert c.source == "sop"
        assert c.embedding is not None
        assert all(v == 0.0 for v in c.embedding[:5])


def test_build_chunks_missing_root_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_chunks(tmp_path / "does_not_exist", source_label="sop")


def test_ingest_dry_run_writes_nothing():
    from common.ai.rag import RagChunk
    conn = MagicMock()
    chunks = [RagChunk(doc_id="x", chunk_index=0, source="sop", text="hi")]
    assert ingest(conn, chunks, dry_run=True) == 0
    conn.cursor.assert_not_called()


def test_ingest_empty_chunks_is_noop():
    conn = MagicMock()
    assert ingest(conn, [], dry_run=False) == 0
    conn.cursor.assert_not_called()


def test_run_dry_run_returns_zero(tmp_path: Path):
    (tmp_path / "runbook.md").write_text("some content")
    exit_code = run(["--root", str(tmp_path), "--dry-run"])
    assert exit_code == 0


def test_run_without_dry_run_returns_nonzero_stub(tmp_path: Path):
    # Without --dry-run the stub returns 1 because DB wiring is still TODO.
    (tmp_path / "runbook.md").write_text("some content")
    exit_code = run(["--root", str(tmp_path)])
    assert exit_code == 1
