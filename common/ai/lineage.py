"""Minimal OpenLineage emission.

Gen-4 Stream G / AI-10 (scaffold).

Writes events to ``fact_lineage_event`` using the OpenLineage-inspired
event shape: ``{kind, job_id, run_id, inputs, outputs, facets, ts}``.

This is intentionally local-only. A future integration with the
``openlineage-python`` client will add HTTP emission to a lineage
backend (Marquez, DataHub). Do NOT install that dep here — keep it
pure-Python.

TODO(gen-4): OpenLineage HTTP exporter with configurable URL + API key.
TODO(gen-4): Tap normalize/load scripts to emit START/COMPLETE events.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# OpenLineage standard event kinds.
VALID_KINDS: frozenset[str] = frozenset({"START", "COMPLETE", "FAIL", "ABORT"})


def emit_event(
    cursor: Any,
    *,
    kind: str,
    job_id: str,
    run_id: str | uuid.UUID | None = None,
    inputs: list[dict[str, Any]] | None = None,
    outputs: list[dict[str, Any]] | None = None,
    facets: dict[str, Any] | None = None,
) -> int:
    """Insert one lineage event row.

    Args:
        cursor: psycopg3 cursor owned by the caller (we do not commit).
        kind: one of ``START | COMPLETE | FAIL | ABORT``.
        job_id: logical job name (e.g. ``backtest_lgbm``, ``promote_model``).
        run_id: optional run UUID to group START/COMPLETE of the same
                run. Generated when absent.
        inputs, outputs: OpenLineage-shaped dataset refs.
        facets: arbitrary event facets.

    Returns:
        The primary key id of the inserted row.
    """
    if kind not in VALID_KINDS:
        raise ValueError(
            f"Invalid kind {kind!r}; expected one of {sorted(VALID_KINDS)}"
        )
    if not job_id:
        raise ValueError("job_id is required")

    resolved_run_id = str(run_id) if run_id else str(uuid.uuid4())
    inputs_json = json.dumps(inputs or [], sort_keys=True, separators=(",", ":"))
    outputs_json = json.dumps(outputs or [], sort_keys=True, separators=(",", ":"))
    facets_json = json.dumps(facets or {}, sort_keys=True, separators=(",", ":"))

    cursor.execute(
        """
        INSERT INTO fact_lineage_event
            (kind, job_id, run_id, inputs, outputs, facets)
        VALUES (%s, %s, %s::uuid, %s::jsonb, %s::jsonb, %s::jsonb)
        RETURNING id
        """,
        (kind, job_id, resolved_run_id, inputs_json, outputs_json, facets_json),
    )
    new_id = cursor.fetchone()[0]
    logger.info(
        "lineage emit id=%s kind=%s job=%s run=%s",
        new_id, kind, job_id, resolved_run_id,
    )
    return new_id


__all__ = ["emit_event", "VALID_KINDS"]
