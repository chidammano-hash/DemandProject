"""Three-tier agent memory: Working / Episodic / Semantic.

Gen-4 Roadmap Cross-cutting. Memory tiers are deliberately isolated:

  WorkingMemory   — in-process dict, TTL-bound. Per-request scratchpad.
  EpisodicMemory  — DB-backed (fact_decision). Outcome history tied
                    back to ai_decision_ledger.id. Used for learning
                    from past decisions.
  SemanticMemory  — pgvector via common/ai/rag.py. Long-term factual
                    knowledge (specs, runbooks, distilled insights).

Each tier is independently testable with a mock cursor.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from common.ai import rag as _rag

logger = logging.getLogger(__name__)


# ──────────────────────────── Working memory ─────────────────────────────

@dataclass
class _Entry:
    value: Any
    expires_at: float | None  # monotonic seconds; None = no expiry


class WorkingMemory:
    """TTL-bound in-process key-value store.

    Not thread-safe — callers that share an instance across threads
    must wrap access in their own lock. Intended lifetime: one
    agent-request.
    """

    def __init__(self, default_ttl: float | None = None) -> None:
        self._store: dict[str, _Entry] = {}
        self._default_ttl = default_ttl

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires = time.monotonic() + effective_ttl if effective_ttl else None
        self._store[key] = _Entry(value=value, expires_at=expires)

    def get(self, key: str, default: Any = None) -> Any:
        entry = self._store.get(key)
        if entry is None:
            return default
        if entry.expires_at is not None and time.monotonic() >= entry.expires_at:
            self._store.pop(key, None)
            return default
        return entry.value

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def __contains__(self, key: str) -> bool:
        return self.get(key, _SENTINEL) is not _SENTINEL

    def __len__(self) -> int:
        # Actively expire when measuring size so callers see truth.
        now = time.monotonic()
        self._store = {
            k: e for k, e in self._store.items()
            if e.expires_at is None or e.expires_at > now
        }
        return len(self._store)


_SENTINEL = object()


# ──────────────────────────── Episodic memory ────────────────────────────

@dataclass
class EpisodeRecord:
    """One outcome observation for a past decision."""
    decision_id: int
    succeeded: bool
    outcome: dict[str, Any] = field(default_factory=dict)
    reward: float | None = None
    notes: str | None = None


class EpisodicMemory:
    """DB-backed outcome log for AI decisions.

    Writes into `fact_decision` (sql/142). Each row FKs back to
    `ai_decision_ledger.id` so callers can JOIN to recover the full
    decision context.
    """

    def __init__(self, cursor_factory: Any) -> None:
        """`cursor_factory` returns a ready-to-use cursor when called.

        Example:
            pool = get_pool()
            def cf():
                with pool.connection() as c:
                    with c.cursor() as cur:
                        return cur
            mem = EpisodicMemory(cf)

        In tests, pass `lambda: mock_cursor`.
        """
        self._cursor_factory = cursor_factory

    def record(self, record: EpisodeRecord) -> int:
        """Insert one episode row. Returns the new fact_decision.id."""
        if not isinstance(record.decision_id, int) or record.decision_id <= 0:
            raise ValueError("decision_id must be a positive integer")
        cur = self._cursor_factory()
        cur.execute(
            """
            INSERT INTO fact_decision
                (decision_id, outcome_json, succeeded, reward, notes)
            VALUES (%s, %s::jsonb, %s, %s, %s)
            RETURNING id
            """,
            (
                record.decision_id,
                json.dumps(record.outcome or {}),
                bool(record.succeeded),
                record.reward,
                record.notes,
            ),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError("fact_decision INSERT returned no id")
        new_id = row[0]
        logger.info(
            "episodic.record decision_id=%s succeeded=%s id=%s",
            record.decision_id, record.succeeded, new_id,
        )
        return new_id

    def recent(self, limit: int = 50, *, only_successes: bool = False) -> list[dict[str, Any]]:
        """Return the most recent episodes for learning windows.

        Rows come back as dicts so callers don't need to know column order.
        """
        cur = self._cursor_factory()
        where = "WHERE succeeded = TRUE" if only_successes else ""
        cur.execute(
            f"""
            SELECT id, decision_id, outcome_ts, outcome_json, succeeded, reward, notes
            FROM fact_decision
            {where}
            ORDER BY outcome_ts DESC
            LIMIT %s
            """,
            (int(limit),),
        )
        out: list[dict[str, Any]] = []
        for row in cur.fetchall() or []:
            outcome_json = row[3]
            if isinstance(outcome_json, str):
                try:
                    outcome_json = json.loads(outcome_json)
                except ValueError:
                    outcome_json = {"_raw": outcome_json}
            out.append({
                "id": row[0],
                "decision_id": row[1],
                "outcome_ts": row[2],
                "outcome": outcome_json or {},
                "succeeded": row[4],
                "reward": row[5],
                "notes": row[6],
            })
        return out


# ──────────────────────────── Semantic memory ────────────────────────────

class SemanticMemory:
    """pgvector-backed long-term knowledge store.

    Thin wrapper around `common.ai.rag` so agents have one import site
    for "things I know about the world". All heavy lifting lives in
    rag.py; this class just forwards.
    """

    def __init__(self, cursor_factory: Any) -> None:
        self._cursor_factory = cursor_factory

    def remember(self, chunks: list[_rag.RagChunk]) -> int:
        cur = self._cursor_factory()
        return _rag.upsert_chunks(cur, chunks)

    def recall(
        self,
        query_text: str,
        query_embedding: list[float] | None = None,
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[_rag.RagChunk]:
        cur = self._cursor_factory()
        return _rag.search(cur, query_text, query_embedding, k=k, filters=filters)


__all__ = [
    "WorkingMemory",
    "EpisodicMemory",
    "EpisodeRecord",
    "SemanticMemory",
]
