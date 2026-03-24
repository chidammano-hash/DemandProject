"""Minimal pipeline utilities: file hashing and batch tracking.

All medallion layers (bronze/silver/gold) have been removed.
Data loads directly from CSV into main tables via load_dataset_postgres.py.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HASH_CHUNK_SIZE = 1024 * 1024  # 1 MB


def file_hash(path: Path) -> str:
    """SHA-256 of first 10 MB for fast change detection."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for _ in range(10):
                chunk = f.read(HASH_CHUNK_SIZE)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except (OSError, IOError) as exc:
        logger.error("file_hash failed for %s: %s", path, exc)
        return ""


def create_batch(cur, domain: str, source_file: str | None = None,
                 source_hash: str | None = None) -> int:
    """Create a load batch record. Returns batch_id."""
    cur.execute(
        """INSERT INTO audit_load_batch
               (domain, layer, source_file, source_hash, status)
           VALUES (%s, 'direct', %s, %s, 'running')
           RETURNING batch_id""",
        [domain, source_file, source_hash],
    )
    return cur.fetchone()[0]


def complete_batch(cur, batch_id: int, row_count_in: int,
                   row_count_out: int) -> None:
    """Mark batch completed."""
    cur.execute(
        "UPDATE audit_load_batch SET status = 'completed', completed_at = now(), "
        "row_count_in = %s, row_count_out = %s WHERE batch_id = %s",
        [row_count_in, row_count_out, batch_id],
    )


def fail_batch(cur, batch_id: int, error: str) -> None:
    """Mark batch failed."""
    cur.execute(
        "UPDATE audit_load_batch SET status = 'failed', completed_at = now(), "
        "error_message = %s WHERE batch_id = %s",
        [error, batch_id],
    )
