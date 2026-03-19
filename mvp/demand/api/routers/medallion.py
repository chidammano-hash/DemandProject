"""Medallion pipeline observability endpoints — lineage, corrections, quarantine."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.core import get_conn

router = APIRouter(prefix="/data-quality", tags=["data-quality"])


# ---------------------------------------------------------------------------
# Load Batches
# ---------------------------------------------------------------------------

@router.get("/lineage/batches")
async def list_batches(
    domain: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, le=200),
):
    """List recent load batches."""
    with get_conn() as conn, conn.cursor() as cur:
        where, params = ["1=1"], []
        if domain:
            where.append("domain = %s")
            params.append(domain)
        if status:
            where.append("status = %s")
            params.append(status)
        params.append(limit)
        cur.execute(
            f"SELECT batch_id, domain, layer, source_file, source_hash, "
            f"row_count_in, row_count_out, row_count_quarantined, "
            f"status, started_at, completed_at, error_message "
            f"FROM audit_load_batch "
            f"WHERE {' AND '.join(where)} "
            f"ORDER BY started_at DESC LIMIT %s",
            params,
        )
        rows = cur.fetchall()

    return {
        "batches": [
            {
                "batch_id": r[0], "domain": r[1], "layer": r[2],
                "source_file": r[3], "source_hash": r[4],
                "row_count_in": r[5], "row_count_out": r[6],
                "row_count_quarantined": r[7], "status": r[8],
                "started_at": r[9].isoformat() if r[9] else None,
                "completed_at": r[10].isoformat() if r[10] else None,
                "error_message": r[11],
            }
            for r in rows
        ],
        "total": len(rows),
    }


@router.get("/lineage/batches/{batch_id}")
async def batch_detail(batch_id: int):
    """Batch detail with per-layer counts."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT batch_id, domain, layer, source_file, source_hash, "
            "row_count_in, row_count_out, row_count_quarantined, "
            "status, started_at, completed_at, error_message, metadata "
            "FROM audit_load_batch WHERE batch_id = %s",
            [batch_id],
        )
        r = cur.fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="batch not found")

        # Layer counts from lineage
        cur.execute(
            "SELECT layer_reached, count(*) FROM audit_row_lineage "
            "WHERE load_batch_id = %s GROUP BY layer_reached",
            [batch_id],
        )
        layer_counts = {row[0]: row[1] for row in cur.fetchall()}

    return {
        "batch_id": r[0], "domain": r[1], "layer": r[2],
        "source_file": r[3], "source_hash": r[4],
        "row_count_in": r[5], "row_count_out": r[6],
        "row_count_quarantined": r[7], "status": r[8],
        "started_at": r[9].isoformat() if r[9] else None,
        "completed_at": r[10].isoformat() if r[10] else None,
        "error_message": r[11],
        "layer_counts": layer_counts,
    }


# ---------------------------------------------------------------------------
# Row Lineage
# ---------------------------------------------------------------------------

@router.get("/lineage/row/{domain}/{business_key}")
async def row_lineage(domain: str, business_key: str):
    """Trace a row through bronze → silver → gold."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT lineage_id, load_batch_id, bronze_id, silver_id, gold_id, "
            "business_key, layer_reached, created_at "
            "FROM audit_row_lineage "
            "WHERE domain = %s AND business_key = %s "
            "ORDER BY created_at DESC LIMIT 10",
            [domain, business_key],
        )
        rows = cur.fetchall()

    return {
        "domain": domain,
        "business_key": business_key,
        "lineage": [
            {
                "lineage_id": r[0], "load_batch_id": r[1],
                "bronze_id": r[2], "silver_id": r[3], "gold_id": r[4],
                "layer_reached": r[5],
                "created_at": r[6].isoformat() if r[6] else None,
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# DQ Corrections
# ---------------------------------------------------------------------------

@router.get("/lineage/corrections")
async def list_corrections(
    domain: str | None = None,
    fix_type: str | None = None,
    batch_id: int | None = None,
    limit: int = Query(default=50, le=500),
):
    """List DQ corrections with audit trail."""
    with get_conn() as conn, conn.cursor() as cur:
        where, params = ["1=1"], []
        if domain:
            where.append("domain = %s")
            params.append(domain)
        if fix_type:
            where.append("fix_type = %s")
            params.append(fix_type)
        if batch_id:
            where.append("load_batch_id = %s")
            params.append(batch_id)
        params.append(limit)
        cur.execute(
            f"SELECT correction_id, domain, table_name, row_key, column_name, "
            f"old_value, new_value, fix_type, fix_strategy, applied_by, "
            f"applied_at, load_batch_id "
            f"FROM audit_dq_corrections "
            f"WHERE {' AND '.join(where)} "
            f"ORDER BY applied_at DESC LIMIT %s",
            params,
        )
        rows = cur.fetchall()

    return {
        "corrections": [
            {
                "correction_id": r[0], "domain": r[1], "table_name": r[2],
                "row_key": r[3], "column_name": r[4],
                "old_value": r[5], "new_value": r[6],
                "fix_type": r[7], "fix_strategy": r[8],
                "applied_by": r[9],
                "applied_at": r[10].isoformat() if r[10] else None,
                "load_batch_id": r[11],
            }
            for r in rows
        ],
        "total": len(rows),
    }


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------

@router.get("/quarantine")
async def list_quarantine(
    domain: str | None = None,
    resolved: bool | None = None,
    limit: int = Query(default=50, le=200),
):
    """List quarantined rows."""
    with get_conn() as conn, conn.cursor() as cur:
        where, params = ["1=1"], []
        if domain:
            where.append("domain = %s")
            params.append(domain)
        if resolved is not None:
            where.append("resolved = %s")
            params.append(resolved)
        params.append(limit)
        cur.execute(
            f"SELECT quarantine_id, domain, _bronze_id, _load_batch_id, "
            f"rejection_reason, rejection_details, raw_row, "
            f"resolved, resolved_by, quarantined_at "
            f"FROM silver_quarantine "
            f"WHERE {' AND '.join(where)} "
            f"ORDER BY quarantined_at DESC LIMIT %s",
            params,
        )
        rows = cur.fetchall()

    return {
        "quarantine": [
            {
                "quarantine_id": r[0], "domain": r[1],
                "bronze_id": r[2], "load_batch_id": r[3],
                "rejection_reason": r[4],
                "rejection_details": r[5],
                "raw_row": r[6],
                "resolved": r[7], "resolved_by": r[8],
                "quarantined_at": r[9].isoformat() if r[9] else None,
            }
            for r in rows
        ],
        "total": len(rows),
    }


@router.post("/quarantine/{quarantine_id}/resolve")
async def resolve_quarantine(quarantine_id: int):
    """Mark a quarantined row as resolved (dismissed)."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE silver_quarantine SET resolved = TRUE, resolved_by = 'user' "
            "WHERE quarantine_id = %s RETURNING quarantine_id",
            [quarantine_id],
        )
        row = cur.fetchone()
        conn.commit()
    if not row:
        raise HTTPException(status_code=404, detail="quarantine entry not found")
    return {"quarantine_id": row[0], "resolved": True}
