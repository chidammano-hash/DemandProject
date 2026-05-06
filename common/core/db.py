"""Shared database connection helpers."""

import logging
import os
from typing import Any
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)


def get_db_params() -> dict[str, Any]:
    """Return psycopg connection parameters from environment variables."""
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5440")),
        "dbname": os.getenv("POSTGRES_DB", "demand_mvp"),
        "user": os.getenv("POSTGRES_USER", "demand"),
        "password": os.getenv("POSTGRES_PASSWORD", "demand"),
    }


def get_read_replica_params() -> dict[str, Any] | None:
    """Return psycopg connection parameters for the read replica, or None if not configured.

    Reads ``READ_REPLICA_URL`` (a standard ``postgres://user:pass@host:port/dbname`` URL).
    When unset, returns ``None`` and callers fall back to the primary pool. When set
    but unparseable, logs a warning and returns ``None`` (degrade safely to primary).

    Note: read replicas can lag behind the primary. Use only for queries that
    tolerate eventual consistency (analytics, dashboards). Do NOT use immediately
    after a write the user expects to see (e.g. read-after-write flows).
    """
    url = os.getenv("READ_REPLICA_URL")
    if not url:
        return None
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("postgres", "postgresql"):
            logger.warning(
                "READ_REPLICA_URL has unexpected scheme %r; expected 'postgres' or 'postgresql'. "
                "Falling back to primary.",
                parsed.scheme,
            )
            return None
        if not parsed.hostname:
            logger.warning("READ_REPLICA_URL missing hostname; falling back to primary.")
            return None
        # urlparse percent-decodes nothing — apply unquote on credential fields.
        return {
            "host": parsed.hostname,
            "port": int(parsed.port) if parsed.port else 5432,
            "dbname": (parsed.path or "/").lstrip("/") or "demand_mvp",
            "user": unquote(parsed.username) if parsed.username else "demand",
            "password": unquote(parsed.password) if parsed.password else "",
        }
    except (ValueError, TypeError) as exc:
        logger.warning("Failed to parse READ_REPLICA_URL: %s. Falling back to primary.", exc)
        return None
