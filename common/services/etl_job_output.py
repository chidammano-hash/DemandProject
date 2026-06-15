"""Parse the unified ETL dispatcher's final JSON stdout line.

``scripts/etl/load.py`` prints a final JSON object summarizing a load
(rows_loaded / rows_inserted / rows_updated / rows_deleted / error). The
JobManager ``load_domain`` handler (US17c) consumes it to populate the job
result. Pure stdout-string parsing — no DB, no network.
"""
from __future__ import annotations

import json
from typing import Any


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_final_json(stdout: str) -> dict[str, Any]:
    """Extract row metrics from the dispatcher's last non-empty JSON stdout line.

    Returns a dict with keys ``rows_loaded`` (int, defaults 0), ``rows_inserted``,
    ``rows_updated``, ``rows_deleted`` (int | None), and ``error`` (str | None).
    Any parse failure yields the empty/default shape rather than raising.
    """
    empty: dict[str, Any] = {
        "rows_loaded": 0, "rows_inserted": None,
        "rows_updated": None, "rows_deleted": None, "error": None,
    }
    if not stdout:
        return empty
    last_line = next(
        (s for s in (ln.strip() for ln in reversed(stdout.splitlines())) if s),
        "",
    )
    if not last_line:
        return empty
    try:
        payload = json.loads(last_line)
    except json.JSONDecodeError:
        return empty
    if not isinstance(payload, dict):
        return empty
    err = payload.get("error")
    return {
        "rows_loaded": _opt_int(payload.get("rows_loaded")) or 0,
        "rows_inserted": _opt_int(payload.get("rows_inserted")),
        "rows_updated": _opt_int(payload.get("rows_updated")),
        "rows_deleted": _opt_int(payload.get("rows_deleted")),
        "error": str(err) if err not in (None, "") else None,
    }
