"""Inventory Planning — IPfeature5: Replenishment Policy Management endpoints."""
from __future__ import annotations

from typing import Any

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel

from api.auth import require_api_key
from api.core import get_conn, set_cache
from common.core.planning_date import get_planning_date

router = APIRouter(tags=["inv-planning"])


class PolicyCreateBody(BaseModel):
    policy_id: str
    policy_name: str
    policy_type: str
    segment: str | None = None
    review_cycle_days: int | None = None
    service_level: float | None = None
    use_eoq: bool = True
    use_safety_stock: bool = True
    notes: str | None = None


class PolicyUpdateBody(BaseModel):
    policy_name: str | None = None
    policy_type: str | None = None
    segment: str | None = None
    review_cycle_days: int | None = None
    service_level: float | None = None
    use_eoq: bool | None = None
    use_safety_stock: bool | None = None
    active: bool | None = None
    notes: str | None = None


class PolicyAssignBody(BaseModel):
    # Individual assignment
    item_id: str | None = None
    loc: str | None = None
    policy_id: str | None = None
    override_reason: str | None = None
    # Bulk by segment
    segment: str | None = None


_VALID_POLICY_TYPES = {"continuous_rop", "periodic_review", "min_max", "manual"}


@router.get("/inv-planning/policies")
def get_policies(response: FastAPIResponse) -> dict:
    """List all active policies with DFU assignment counts.

    Cache: 300s.
    """
    set_cache(response, max_age=300)

    sql = """
        SELECT
            p.policy_id,
            p.policy_name,
            p.policy_type,
            p.segment,
            p.review_cycle_days,
            p.service_level,
            p.use_eoq,
            p.use_safety_stock,
            p.active,
            COUNT(a.item_id) AS dfu_count
        FROM dim_replenishment_policy p
        LEFT JOIN fact_dfu_policy_assignment a ON a.policy_id = p.policy_id
        GROUP BY p.policy_sk, p.policy_id, p.policy_name, p.policy_type,
                 p.segment, p.review_cycle_days, p.service_level,
                 p.use_eoq, p.use_safety_stock, p.active
        ORDER BY p.policy_id
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

    # Column order matches SELECT: policy_id(0), policy_name(1), policy_type(2),
    # segment(3), review_cycle_days(4), service_level(5), use_eoq(6),
    # use_safety_stock(7), active(8), dfu_count(9)
    def _row(r: tuple) -> dict:
        return {
            "policy_id":         r[0],
            "policy_name":       r[1],
            "policy_type":       r[2],
            "segment":           r[3],
            "review_cycle_days": r[4],
            "service_level":     float(r[5]) if r[5] is not None else None,
            "use_eoq":           bool(r[6]),
            "use_safety_stock":  bool(r[7]),
            "active":            bool(r[8]),
            "dfu_count":         int(r[9]),
        }

    return {"policies": [_row(r) for r in rows]}


@router.post("/inv-planning/policies", status_code=201, dependencies=[Depends(require_api_key)])
def create_policy(body: PolicyCreateBody) -> dict:
    """Create a new replenishment policy. Auth required."""
    if body.policy_type not in _VALID_POLICY_TYPES:
        raise HTTPException(status_code=422, detail=f"policy_type must be one of {sorted(_VALID_POLICY_TYPES)}")

    sql = """
        INSERT INTO dim_replenishment_policy
            (policy_id, policy_name, policy_type, segment, review_cycle_days,
             service_level, use_eoq, use_safety_stock, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING policy_id, policy_name, policy_type, segment,
                  review_cycle_days, service_level, use_eoq, use_safety_stock, active, notes
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(sql, (
                    body.policy_id, body.policy_name, body.policy_type,
                    body.segment, body.review_cycle_days, body.service_level,
                    body.use_eoq, body.use_safety_stock, body.notes,
                ))
                conn.commit()
                row = cur.fetchone()
                cols = [d[0] for d in cur.description]
            except psycopg.Error as exc:
                conn.rollback()
                raise HTTPException(status_code=409, detail="Policy with this ID already exists.") from exc

    d = dict(zip(cols, row))
    return {
        "policy_id":         d["policy_id"],
        "policy_name":       d["policy_name"],
        "policy_type":       d["policy_type"],
        "segment":           d["segment"],
        "review_cycle_days": d["review_cycle_days"],
        "service_level":     float(d["service_level"]) if d["service_level"] is not None else None,
        "use_eoq":           bool(d["use_eoq"]),
        "use_safety_stock":  bool(d["use_safety_stock"]),
        "active":            bool(d["active"]),
        "notes":             d["notes"],
        "dfu_count":         0,
    }


@router.put("/inv-planning/policies/{policy_id}", dependencies=[Depends(require_api_key)])
def update_policy(policy_id: str, body: PolicyUpdateBody) -> dict:
    """Update an existing policy by policy_id. Auth required."""
    updates: list[str] = ["modified_ts = NOW()"]
    params: list[Any] = []

    if body.policy_name is not None:
        updates.append("policy_name = %s")
        params.append(body.policy_name)
    if body.policy_type is not None:
        if body.policy_type not in _VALID_POLICY_TYPES:
            raise HTTPException(status_code=422, detail=f"policy_type must be one of {sorted(_VALID_POLICY_TYPES)}")
        updates.append("policy_type = %s")
        params.append(body.policy_type)
    if body.segment is not None:
        updates.append("segment = %s")
        params.append(body.segment)
    if body.review_cycle_days is not None:
        updates.append("review_cycle_days = %s")
        params.append(body.review_cycle_days)
    if body.service_level is not None:
        updates.append("service_level = %s")
        params.append(body.service_level)
    if body.use_eoq is not None:
        updates.append("use_eoq = %s")
        params.append(body.use_eoq)
    if body.use_safety_stock is not None:
        updates.append("use_safety_stock = %s")
        params.append(body.use_safety_stock)
    if body.active is not None:
        updates.append("active = %s")
        params.append(body.active)
    if body.notes is not None:
        updates.append("notes = %s")
        params.append(body.notes)

    params.append(policy_id)
    sql = f"""
        UPDATE dim_replenishment_policy
        SET {', '.join(updates)}
        WHERE policy_id = %s
        RETURNING policy_id, policy_name, policy_type, segment,
                  review_cycle_days, service_level, use_eoq, use_safety_stock, active, notes
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found")
            cols = [d[0] for d in cur.description]

    # Count DFUs assigned
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM fact_dfu_policy_assignment WHERE policy_id = %s", [policy_id])
            dfu_count = cur.fetchone()[0] or 0

    d = dict(zip(cols, row))
    return {
        "policy_id":         d["policy_id"],
        "policy_name":       d["policy_name"],
        "policy_type":       d["policy_type"],
        "segment":           d["segment"],
        "review_cycle_days": d["review_cycle_days"],
        "service_level":     float(d["service_level"]) if d["service_level"] is not None else None,
        "use_eoq":           bool(d["use_eoq"]),
        "use_safety_stock":  bool(d["use_safety_stock"]),
        "active":            bool(d["active"]),
        "notes":             d["notes"],
        "dfu_count":         int(dfu_count),
    }


@router.get("/inv-planning/policy-assignments")
def get_policy_assignments(
    response: FastAPIResponse,
    item: str = Query(default="", max_length=120),
    location: str = Query(default="", max_length=120),
    policy_id: str = Query(default="", max_length=80),
    assigned_by: str = Query(default="", max_length=20),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """Paginated DFU policy assignments.

    Cache: 120s.
    """
    set_cache(response, max_age=120)

    where_parts: list[str] = []
    params: list[Any] = []

    if item.strip():
        where_parts.append("a.item_id ILIKE %s")
        params.append(f"%{item.strip()}%")
    if location.strip():
        where_parts.append("a.loc ILIKE %s")
        params.append(f"%{location.strip()}%")
    if policy_id.strip():
        where_parts.append("a.policy_id = %s")
        params.append(policy_id.strip())
    if assigned_by.strip():
        where_parts.append("a.assigned_by = %s")
        params.append(assigned_by.strip())

    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    count_sql = f"SELECT COUNT(*) FROM fact_dfu_policy_assignment a {where_clause}"
    data_sql = f"""
        SELECT
            a.item_id,
            a.loc,
            a.policy_id,
            p.policy_name,
            p.policy_type,
            a.override_reason,
            a.assigned_by,
            a.effective_date
        FROM fact_dfu_policy_assignment a
        JOIN dim_replenishment_policy p ON p.policy_id = a.policy_id
        {where_clause}
        ORDER BY a.item_id, a.loc
        LIMIT %s OFFSET %s
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, params)
            total = cur.fetchone()[0] or 0

            cur.execute(data_sql, [*params, limit, offset])
            rows = cur.fetchall()
            [d[0] for d in cur.description]

    return {
        "total": int(total),
        "rows": [
            {
                "item_id":         r[0],
                "loc":             r[1],
                "policy_id":       r[2],
                "policy_name":     r[3],
                "policy_type":     r[4],
                "override_reason": r[5],
                "assigned_by":     r[6],
                "effective_date":  r[7].isoformat() if r[7] else None,
            }
            for r in rows
        ],
    }


@router.post("/inv-planning/policy-assignments/assign", dependencies=[Depends(require_api_key)])
def assign_policy(body: PolicyAssignBody) -> dict:
    """Assign a policy to one DFU (individual) or all DFUs in a segment (bulk).

    Individual: { item_id, loc, policy_id, override_reason }
    Bulk:       { segment, policy_id }
    Auth required.
    """
    effective_date = get_planning_date()
    assigned_count = 0
    failed_count = 0
    already_assigned_count = 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Verify policy exists
            if body.policy_id:
                cur.execute("SELECT 1 FROM dim_replenishment_policy WHERE policy_id = %s", [body.policy_id])
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail=f"Policy '{body.policy_id}' not found")

            if body.item_id and body.loc and body.policy_id:
                # Individual assignment
                upsert_sql = """
                    INSERT INTO fact_dfu_policy_assignment
                        (item_id, loc, policy_id, override_reason, assigned_by, effective_date)
                    VALUES (%s, %s, %s, %s, 'manual', %s)
                    ON CONFLICT (item_id, loc) DO UPDATE SET
                        policy_id      = EXCLUDED.policy_id,
                        override_reason= EXCLUDED.override_reason,
                        assigned_by    = 'manual',
                        effective_date = EXCLUDED.effective_date,
                        modified_ts    = NOW()
                """
                try:
                    cur.execute(upsert_sql, (
                        body.item_id, body.loc, body.policy_id,
                        body.override_reason, effective_date,
                    ))
                    assigned_count = 1
                except psycopg.Error:
                    failed_count = 1

            elif body.segment and body.policy_id:
                # Bulk assignment by segment — assign all DFUs with matching abc_vol
                # or variability_class matching the segment
                bulk_sql = """
                    INSERT INTO fact_dfu_policy_assignment
                        (item_id, loc, policy_id, assigned_by, effective_date)
                    SELECT
                        d.item_id,
                        d.loc,
                        %s,
                        'system',
                        %s
                    FROM dim_sku d
                    WHERE d.abc_vol = %s OR d.variability_class = %s
                    ON CONFLICT (item_id, loc) DO UPDATE SET
                        policy_id      = EXCLUDED.policy_id,
                        assigned_by    = 'system',
                        effective_date = EXCLUDED.effective_date,
                        modified_ts    = NOW()
                    WHERE fact_dfu_policy_assignment.assigned_by = 'system'
                """
                try:
                    cur.execute(bulk_sql, (
                        body.policy_id, effective_date,
                        body.segment.upper(), body.segment.lower(),
                    ))
                    assigned_count = cur.rowcount
                except psycopg.Error:
                    failed_count = 1
            else:
                raise HTTPException(
                    status_code=422,
                    detail="Provide either (item_id + loc + policy_id) or (segment + policy_id)",
                )

        conn.commit()

    return {
        "assigned_count":         assigned_count,
        "failed_count":           failed_count,
        "already_assigned_count": already_assigned_count,
    }


@router.get("/inv-planning/policy-assignments/compliance")
def get_policy_compliance(response: FastAPIResponse) -> dict:
    """Portfolio-level policy compliance metrics.

    Returns: total_dfus, assigned_count, unassigned_count, assignment_pct,
    and per-policy breakdown with avg_dos where available.

    Cache: 300s.
    """
    set_cache(response, max_age=300)

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Total DFUs
            cur.execute("SELECT COUNT(*) FROM dim_sku")
            total_dfus = cur.fetchone()[0] or 0

            # Assigned DFUs
            cur.execute("SELECT COUNT(DISTINCT item_id || '|' || loc) FROM fact_dfu_policy_assignment")
            assigned_count = cur.fetchone()[0] or 0

            # Per-policy breakdown
            by_policy_sql = """
                SELECT
                    p.policy_id,
                    p.policy_name,
                    p.policy_type,
                    COUNT(a.item_id)       AS dfu_count,
                    AVG(inv.dos)           AS avg_dos
                FROM dim_replenishment_policy p
                LEFT JOIN fact_dfu_policy_assignment a ON a.policy_id = p.policy_id
                LEFT JOIN (
                    SELECT item_id, loc,
                           CASE WHEN AVG(daily_sales) > 0
                                THEN AVG(eom_on_hand) / AVG(daily_sales)
                                ELSE NULL
                           END AS dos
                    FROM agg_inventory_monthly
                    GROUP BY item_id, loc
                ) inv ON inv.item_id = a.item_id AND inv.loc = a.loc
                GROUP BY p.policy_id, p.policy_name, p.policy_type
                ORDER BY p.policy_id
            """
            try:
                cur.execute(by_policy_sql)
                policy_rows = cur.fetchall()
                policy_cols = [d[0] for d in cur.description]
            except Exception:
                # If agg_inventory_monthly doesn't exist, fall back to simpler query
                cur.execute("""
                    SELECT
                        p.policy_id,
                        p.policy_name,
                        p.policy_type,
                        COUNT(a.item_id) AS dfu_count,
                        NULL             AS avg_dos
                    FROM dim_replenishment_policy p
                    LEFT JOIN fact_dfu_policy_assignment a ON a.policy_id = p.policy_id
                    GROUP BY p.policy_id, p.policy_name, p.policy_type
                    ORDER BY p.policy_id
                """)
                policy_rows = cur.fetchall()
                policy_cols = [d[0] for d in cur.description]

    unassigned_count = int(total_dfus) - int(assigned_count)
    assignment_pct = (float(assigned_count) / float(total_dfus) * 100.0) if total_dfus > 0 else 0.0

    by_policy: dict[str, dict] = {}
    for row in policy_rows:
        d = dict(zip(policy_cols, row))
        by_policy[d["policy_id"]] = {
            "policy_name":    d["policy_name"],
            "policy_type":    d["policy_type"],
            "dfu_count":      int(d["dfu_count"]),
            "below_ss_pct":   None,   # IPfeature3 not yet implemented
            "avg_ss_coverage": None,  # IPfeature3 not yet implemented
            "avg_dos":        float(d["avg_dos"]) if d["avg_dos"] is not None else None,
        }

    return {
        "total_dfus":       int(total_dfus),
        "assigned_count":   int(assigned_count),
        "unassigned_count": int(unassigned_count),
        "assignment_pct":   round(assignment_pct, 2),
        "by_policy":        by_policy,
    }
