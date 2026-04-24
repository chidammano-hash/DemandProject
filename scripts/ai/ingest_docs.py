"""Ingest SOPs / runbooks / post-mortems into the RAG corpus.

Gen-4 Roadmap AI-5. Walks a directory of markdown files, chunks each
file into overlapping windows, and upserts the chunks via
:func:`common.ai.rag.upsert_chunks`.

Embeddings are intentionally NOT generated here — each chunk is written
with a zero-vector placeholder. The production embedding provider will
plug in once an approved client lands; in the meantime chunks still get
their tsvector populated by the DB trigger, so lexical retrieval works
end-to-end.

Usage::

    python -m scripts.ai.ingest_docs --root docs/sop --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Iterator

from common.ai.rag import RagChunk, upsert_chunks

logger = logging.getLogger(__name__)

# Chunking params. Naive fixed-size chunks with a small overlap. Kept as
# named constants so tuning does not require poking string-manipulation
# code. Overlap preserves cross-chunk context so retrieval doesn't lose
# a sentence that spans a boundary.
DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 50

# Embedding dim placeholder — the real value matches the chosen provider.
# See `sql/139_create_rag_chunk.sql` for the on-DB dim.
_EMBEDDING_DIM_PLACEHOLDER: int = 1536


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split ``text`` into overlapping fixed-size chunks.

    Empty or whitespace-only inputs yield an empty list so callers can
    skip blank files cleanly. Chunks overlap by ``chunk_overlap`` chars
    to preserve cross-boundary context.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    stripped = text.strip()
    if not stripped:
        return []

    chunks: list[str] = []
    start = 0
    stride = chunk_size - chunk_overlap
    while start < len(stripped):
        chunks.append(stripped[start : start + chunk_size])
        start += stride
    return chunks


def _iter_markdown_files(root: Path) -> Iterator[Path]:
    yield from sorted(root.rglob("*.md"))


def build_chunks(
    root: Path,
    source_label: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[RagChunk]:
    """Walk ``root`` and return a list of :class:`RagChunk` entries.

    Each chunk carries a zero-vector embedding placeholder. The
    ``doc_id`` is the path relative to ``root`` so re-ingesting the same
    file updates the existing rows via the ``(doc_id, chunk_index)``
    unique key.
    """
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"ingest root not found: {root}")

    zero_embedding = [0.0] * _EMBEDDING_DIM_PLACEHOLDER
    chunks_out: list[RagChunk] = []

    for md_path in _iter_markdown_files(root):
        try:
            text = md_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("skipping %s: %s", md_path, exc)
            continue

        pieces = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not pieces:
            logger.info("skipping empty file %s", md_path)
            continue

        doc_id = str(md_path.relative_to(root))
        for idx, piece in enumerate(pieces):
            chunks_out.append(
                RagChunk(
                    doc_id=doc_id,
                    chunk_index=idx,
                    source=source_label,
                    text=piece,
                    # TODO(gen-4 AI-5): replace zero vector with real
                    # embedding from the approved provider when wired.
                    embedding=zero_embedding,
                    metadata={"path": str(md_path), "source": source_label},
                )
            )

    logger.info(
        "build_chunks root=%s files_scanned=%d chunks=%d",
        root, sum(1 for _ in _iter_markdown_files(root)), len(chunks_out),
    )
    return chunks_out


def ingest(
    conn: object,
    chunks: Iterable[RagChunk],
    *,
    dry_run: bool = False,
) -> int:
    """Upsert chunks via a connection cursor.

    Returns the number of rows written (0 when ``dry_run`` or no chunks).
    """
    chunk_list = list(chunks)
    if not chunk_list:
        return 0
    if dry_run:
        logger.info("dry-run: would write %d chunks", len(chunk_list))
        return 0

    with conn.cursor() as cur:  # type: ignore[attr-defined]
        written = upsert_chunks(cur, chunk_list)
    logger.info("ingest wrote %d chunks", written)
    return written


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest markdown docs into RAG corpus")
    parser.add_argument("--root", required=True, help="Root directory to walk (e.g. docs/sop)")
    parser.add_argument("--source", default="sop", help="Source label written on each chunk")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build chunks but do not write to the DB.",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.root).resolve()
    chunks = build_chunks(
        root,
        source_label=args.source,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    if args.dry_run:
        logger.info("--dry-run set; skipping DB write (%d chunks).", len(chunks))
        return 0

    # TODO(gen-4 AI-5): pull a connection from common.core.db when this
    # script is promoted from scaffold to production entrypoint. The
    # dry-run path above is the only supported mode for now.
    logger.info("connection wiring not implemented; rerun with --dry-run.")
    return 1


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(run())
