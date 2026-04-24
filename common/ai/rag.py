"""pgvector-backed RAG + knowledge-graph client.

Gen-4 Roadmap Cross-cutting #9. Implements hybrid retrieval over the
`rag_chunk` table (see `sql/139_create_rag_chunk.sql`):

  * dense cosine similarity via `embedding <=> %s::vector`
  * lexical BM25-style match via `ts_rank(ts_vector, plainto_tsquery(...))`
  * reciprocal-rank fusion of the two signals

No external LLM call happens inside this module — callers pass a
pre-computed `query_embedding` (list[float]) so this file stays
dependency-light and easy to unit-test with a mock cursor.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Cosine distance in pgvector is `<=>`; similarity = 1 - distance.
_VECTOR_SEARCH_SQL = """
SELECT id, doc_id, chunk_index, source, text, metadata,
       1 - (embedding <=> %s::vector) AS vec_score
FROM rag_chunk
WHERE embedding IS NOT NULL
{filters}
ORDER BY embedding <=> %s::vector
LIMIT %s
"""

_LEXICAL_SEARCH_SQL = """
SELECT id, doc_id, chunk_index, source, text, metadata,
       ts_rank(ts_vector, plainto_tsquery('english', %s)) AS lex_score
FROM rag_chunk
WHERE ts_vector @@ plainto_tsquery('english', %s)
{filters}
ORDER BY lex_score DESC
LIMIT %s
"""


@dataclass
class RagChunk:
    """One chunk of source content with optional dense embedding."""

    doc_id: str
    chunk_index: int
    source: str
    text: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Populated by search() on the retrieval side.
    id: int | None = None
    score: float | None = None


def _build_filter_clause(filters: dict[str, Any] | None) -> tuple[str, list[Any]]:
    """Translate a {column: value} dict into a `AND ...` SQL clause.

    Only supports equality against `source` and JSONB containment on
    `metadata` (filters['metadata'] = {...}). Unknown keys are ignored.
    """
    if not filters:
        return "", []
    fragments: list[str] = []
    params: list[Any] = []
    if "source" in filters:
        fragments.append("source = %s")
        params.append(filters["source"])
    if "metadata" in filters and isinstance(filters["metadata"], dict):
        fragments.append("metadata @> %s::jsonb")
        params.append(json.dumps(filters["metadata"]))
    if not fragments:
        return "", []
    return " AND " + " AND ".join(fragments), params


def _embedding_to_pgvector(embedding: list[float] | None) -> str | None:
    """pgvector text representation: '[0.1,0.2,...]'."""
    if embedding is None:
        return None
    return "[" + ",".join(f"{float(x):.6f}" for x in embedding) + "]"


def upsert_chunks(cursor: Any, chunks: list[RagChunk]) -> int:
    """Insert or replace chunks keyed by (doc_id, chunk_index).

    Returns the number of rows written. Trigger `rag_chunk_tsv_update`
    populates the tsvector automatically.
    """
    if not chunks:
        return 0
    rows_written = 0
    for chunk in chunks:
        emb = _embedding_to_pgvector(chunk.embedding)
        try:
            cursor.execute(
                """
                INSERT INTO rag_chunk
                    (doc_id, chunk_index, source, text, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s::vector, %s::jsonb)
                ON CONFLICT (doc_id, chunk_index) DO UPDATE SET
                    source = EXCLUDED.source,
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                """,
                (
                    chunk.doc_id,
                    chunk.chunk_index,
                    chunk.source,
                    chunk.text,
                    emb,
                    json.dumps(chunk.metadata or {}),
                ),
            )
            rows_written += 1
        except (ValueError, TypeError) as exc:
            logger.exception("upsert_chunks failed for %s#%s: %s", chunk.doc_id, chunk.chunk_index, exc)
            raise
    return rows_written


def search(
    cursor: Any,
    query_text: str,
    query_embedding: list[float] | None = None,
    k: int = 10,
    filters: dict[str, Any] | None = None,
    *,
    vec_weight: float = 0.6,
    lex_weight: float = 0.4,
    rrf_k: int = 60,
) -> list[RagChunk]:
    """Hybrid retrieval: vector cosine + BM25, fused via RRF.

    Reciprocal-rank fusion:
        score(chunk) = vec_weight * 1/(rrf_k + vec_rank)
                     + lex_weight * 1/(rrf_k + lex_rank)

    If `query_embedding` is None, only lexical search is used. If `k` is
    zero or the ts_vector query yields nothing AND embedding is None,
    returns [].
    """
    if k <= 0:
        return []

    filter_clause, filter_params = _build_filter_clause(filters)

    # Vector branch.
    vec_rows: list[tuple[Any, ...]] = []
    if query_embedding is not None:
        emb = _embedding_to_pgvector(query_embedding)
        vec_sql = _VECTOR_SEARCH_SQL.format(filters=filter_clause)
        cursor.execute(vec_sql, [emb, *filter_params, emb, k * 3])
        vec_rows = list(cursor.fetchall() or [])

    # Lexical branch.
    lex_sql = _LEXICAL_SEARCH_SQL.format(filters=filter_clause)
    cursor.execute(lex_sql, [query_text, query_text, *filter_params, k * 3])
    lex_rows = list(cursor.fetchall() or [])

    # Fuse via RRF, keyed by chunk id.
    fused: dict[int, dict[str, Any]] = {}
    for rank, row in enumerate(vec_rows, start=1):
        rid = row[0]
        fused.setdefault(rid, {"row": row, "vec_rank": None, "lex_rank": None})
        fused[rid]["vec_rank"] = rank
    for rank, row in enumerate(lex_rows, start=1):
        rid = row[0]
        fused.setdefault(rid, {"row": row, "vec_rank": None, "lex_rank": None})
        fused[rid]["lex_rank"] = rank

    scored: list[RagChunk] = []
    for rid, info in fused.items():
        vec_component = vec_weight / (rrf_k + info["vec_rank"]) if info["vec_rank"] else 0.0
        lex_component = lex_weight / (rrf_k + info["lex_rank"]) if info["lex_rank"] else 0.0
        total = vec_component + lex_component
        row = info["row"]
        metadata = row[5] if len(row) > 5 else {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except ValueError:
                metadata = {"_raw": metadata}
        scored.append(
            RagChunk(
                id=rid,
                doc_id=row[1],
                chunk_index=row[2],
                source=row[3],
                text=row[4],
                metadata=metadata or {},
                score=total,
            )
        )

    scored.sort(key=lambda c: c.score or 0.0, reverse=True)
    return scored[:k]


__all__ = [
    "RagChunk",
    "upsert_chunks",
    "search",
]
