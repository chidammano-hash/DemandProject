"""Immutable lineage for the sales source used by forecast artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_SHA256_HEX_LENGTH = 64


@dataclass(frozen=True)
class SalesSourceLineage:
    """One completed, content-addressed sales load."""

    batch_id: int
    source_hash: str


def load_completed_sales_lineage(conn: Any) -> SalesSourceLineage:
    """Load the latest synchronized full sales reload, failing closed.

    Historical ``safe_upsert`` batches changed only the current sales table and
    did not refresh ``fact_sales_monthly_original``. They therefore cannot be
    used as artifact lineage even when their audit row is newer and positive.
    The sales dispatcher now routes changes through the canonical dual-track
    loader; an older unsafe audit row requires one canonical reload first.
    """
    with conn.cursor() as cur:
        cur.execute(
            """SELECT batch_id, source_hash, source_file
               FROM audit_load_batch
               WHERE domain = 'sales'
                 AND status = 'completed'
                 AND row_count_out > 0
               ORDER BY completed_at DESC NULLS LAST, batch_id DESC
               LIMIT 1"""
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("A completed sales load is required for forecast artifacts")

    batch_id, raw_hash, raw_source_file = row
    source_file = str(raw_source_file or "").strip()
    if not source_file or source_file == "safe_upsert":
        raise RuntimeError(
            "The latest completed sales batch did not synchronize the immutable "
            "forecast source; run one canonical sales reload"
        )
    source_hash = str(raw_hash or "").strip().lower()
    if len(source_hash) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in source_hash
    ):
        raise RuntimeError(
            "The latest completed sales load must have a valid SHA-256 source hash"
        )
    return SalesSourceLineage(batch_id=int(batch_id), source_hash=source_hash)
