"""Data quality endpoints — load batch tracking."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.core import get_conn

router = APIRouter(prefix="/data-quality", tags=["Data Quality"])


@router.get("/batches")
async def list_batches(
    domain: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, le=200),
):
    """List recent load batches from audit_load_batch."""
    with get_conn() as conn, conn.cursor() as cur:
        where: list[str] = []
        params: list[object] = []
        if domain:
            where.append("domain = %s")
            params.append(domain)
        if status:
            where.append("status = %s")
            params.append(status)
        where_clause = (" WHERE " + " AND ".join(where)) if where else ""
        params.append(limit)
        cur.execute(
            "SELECT batch_id, domain, source_file, source_hash, status, "
            "row_count_in, row_count_out, started_at, completed_at, error_message "
            f"FROM audit_load_batch{where_clause} "
            "ORDER BY batch_id DESC LIMIT %s",
            params,
        )
        rows = cur.fetchall()

    return {
        "batches": [
            {
                "batch_id": r[0],
                "domain": r[1],
                "source_file": r[2],
                "source_hash": r[3],
                "status": r[4],
                "row_count_in": r[5],
                "row_count_out": r[6],
                "started_at": r[7].isoformat() if r[7] else None,
                "completed_at": r[8].isoformat() if r[8] else None,
                "error_message": r[9],
            }
            for r in rows
        ],
        "total": len(rows),
    }


@router.get("/batches/{batch_id}")
async def batch_detail(batch_id: int):
    """Get a single batch by ID."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT batch_id, domain, source_file, source_hash, status, "
            "row_count_in, row_count_out, started_at, completed_at, error_message "
            "FROM audit_load_batch WHERE batch_id = %s",
            [batch_id],
        )
        r = cur.fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="batch not found")

    return {
        "batch_id": r[0],
        "domain": r[1],
        "source_file": r[2],
        "source_hash": r[3],
        "status": r[4],
        "row_count_in": r[5],
        "row_count_out": r[6],
        "started_at": r[7].isoformat() if r[7] else None,
        "completed_at": r[8].isoformat() if r[8] else None,
        "error_message": r[9],
    }
