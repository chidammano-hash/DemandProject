"""Durable lifecycle helpers for immutable forecast-generation runs."""

from __future__ import annotations

from datetime import date
from typing import Any
from uuid import UUID

from psycopg.types.json import Jsonb

GENERATOR_CONTRACT_METADATA_KEY = "generator_contract_version"
GENERATOR_CONTRACT_VERSION = "canonical-five-artifact-lineage-v2"


def build_generation_metadata(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Stamp immutable runs with the forecast implementation contract used."""
    stamped = dict(metadata or {})
    stamped[GENERATOR_CONTRACT_METADATA_KEY] = GENERATOR_CONTRACT_VERSION
    return stamped


def reserve_generation_run(
    cur: Any,
    *,
    run_id: UUID | str,
    generation_purpose: str,
    requested_model_id: str,
    record_month: date,
    horizon_months: int,
    created_by: str,
    metadata: dict[str, Any] | None = None,
    resume_invalid: bool = False,
) -> str:
    """Create a durable generating manifest or validate an existing reservation."""
    if generation_purpose not in {
        "release_candidate",
        "snapshot_contender",
        "shadow_candidate",
    }:
        raise ValueError("unsupported forecast generation purpose")
    if horizon_months <= 0:
        raise ValueError("forecast generation horizon must be positive")
    month = record_month.replace(day=1)
    cur.execute(
        """INSERT INTO forecast_generation_run
               (run_id, generation_purpose, run_status, promotion_eligible,
                requested_model_id, forecast_month_generated, horizon_months,
                created_by, metadata)
           VALUES (%s::uuid, %s, 'generating', FALSE, %s, %s, %s, %s, %s)
           ON CONFLICT (run_id) DO NOTHING""",
        (
            str(run_id),
            generation_purpose,
            requested_model_id,
            month,
            horizon_months,
            created_by,
            Jsonb(build_generation_metadata(metadata)),
        ),
    )
    cur.execute(
        """SELECT generation_purpose, requested_model_id,
                  forecast_month_generated, horizon_months, run_status,
                  row_count
           FROM forecast_generation_run
           WHERE run_id = %s::uuid
           FOR UPDATE""",
        (str(run_id),),
    )
    row = cur.fetchone()
    if row is None:
        raise ValueError("forecast generation reservation was not persisted")
    identity = (str(row[0]), str(row[1]), row[2], int(row[3] or 0))
    expected = (generation_purpose, requested_model_id, month, horizon_months)
    if identity != expected:
        raise ValueError("forecast generation run UUID has a different identity")

    status = str(row[4])
    if status == "generating":
        cur.execute(
            """SELECT EXISTS (
                   SELECT 1 FROM fact_production_forecast_staging
                   WHERE run_id = %s::uuid
               )""",
            (str(run_id),),
        )
        has_rows = bool(cur.fetchone()[0])
        if has_rows or int(row[5] or 0) > 0:
            raise ValueError(
                "generating forecast run already has staged rows and cannot be resumed"
            )
    if status == "invalid" and resume_invalid:
        cur.execute(
            """SELECT EXISTS (
                   SELECT 1 FROM fact_production_forecast_staging
                   WHERE run_id = %s::uuid
               )""",
            (str(run_id),),
        )
        has_rows = bool(cur.fetchone()[0])
        if has_rows or int(row[5] or 0) > 0:
            raise ValueError("invalid forecast generation run already has staged rows")
        cur.execute(
            """UPDATE forecast_generation_run
               SET run_status = 'generating', invalid_reason = NULL,
                   completed_at = NULL, created_by = %s
               WHERE run_id = %s::uuid AND run_status = 'invalid'""",
            (created_by, str(run_id)),
        )
        if cur.rowcount != 1:
            raise ValueError("invalid forecast generation run could not be resumed")
        return "generating"
    return status


def invalidate_generation_run(cur: Any, run_id: UUID | str, reason: str) -> bool:
    """Mark a not-yet-ready run invalid without mutating completed evidence."""
    safe_reason = reason.strip()[:500] or "forecast generation failed"
    cur.execute(
        """DELETE FROM fact_production_forecast_staging staging
           USING forecast_generation_run generation
           WHERE staging.run_id = %s::uuid
             AND generation.run_id = staging.run_id
             AND generation.generation_purpose IN (
                 'release_candidate', 'snapshot_contender', 'shadow_candidate'
             )
             AND generation.run_status IN ('generating', 'invalid')""",
        (str(run_id),),
    )
    cur.execute(
        """UPDATE forecast_generation_run
           SET run_status = 'invalid', promotion_eligible = FALSE,
               invalid_reason = %s, completed_at = NOW()
           WHERE run_id = %s::uuid
             AND generation_purpose IN ('release_candidate', 'snapshot_contender', 'shadow_candidate')
             AND run_status IN ('generating', 'invalid')""",
        (safe_reason, str(run_id)),
    )
    return cur.rowcount == 1
