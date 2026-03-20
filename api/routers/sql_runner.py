"""SQL Runner API — execute read-only SQL queries from the UI.

Provides endpoints to run ad-hoc SELECT queries against the database,
browse schema metadata, and review query history.  All queries are
executed inside a read-only transaction with a configurable timeout.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.core import get_conn
from common.utils import load_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sql-runner", tags=["sql-runner"])

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_DEFAULT_CFG = {"max_rows": 1000, "statement_timeout_ms": 30000, "enabled": True}


def _cfg() -> dict:
    try:
        return load_config("sql_runner_config")
    except Exception:
        return _DEFAULT_CFG


# ---------------------------------------------------------------------------
# Dangerous statement detector
# ---------------------------------------------------------------------------
_WRITE_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|COPY|VACUUM|REINDEX|CLUSTER|COMMENT|LOCK)\b",
    re.IGNORECASE,
)


def _is_write_statement(sql: str) -> bool:
    """Return True if *sql* contains any non-SELECT statement keyword."""
    # Strip comments and string literals before checking
    cleaned = re.sub(r"--[^\n]*", "", sql)  # line comments
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)  # block comments
    cleaned = re.sub(r"'[^']*'", "''", cleaned)  # string literals
    return bool(_WRITE_PATTERN.search(cleaned))


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class SqlRequest(BaseModel):
    sql: str = Field(..., min_length=1, max_length=10_000)
    max_rows: int | None = Field(None, ge=1, le=10_000)


class SqlResponse(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    elapsed_ms: float


class TableColumn(BaseModel):
    name: str
    data_type: str
    is_nullable: bool


class TableInfo(BaseModel):
    schema_name: str
    table_name: str
    table_type: str
    columns: list[TableColumn]


class QueryHistoryEntry(BaseModel):
    sql: str
    executed_at: str
    elapsed_ms: float
    row_count: int
    status: str


# In-memory history (last 50 queries, per-process)
_history: list[dict] = []
_MAX_HISTORY = 50


# ---------------------------------------------------------------------------
# POST /execute
# ---------------------------------------------------------------------------
@router.post("/execute", response_model=SqlResponse)
async def execute_query(body: SqlRequest):
    """Execute a read-only SQL query and return results."""
    cfg = _cfg()
    if not cfg.get("enabled", True):
        raise HTTPException(status_code=403, detail="SQL Runner is disabled")

    sql = body.sql.strip()
    if not sql:
        raise HTTPException(status_code=400, detail="SQL query is empty")

    if _is_write_statement(sql):
        raise HTTPException(
            status_code=400,
            detail="Only SELECT / WITH queries are allowed. Write operations are blocked.",
        )

    max_rows = body.max_rows or cfg.get("max_rows", 1000)
    timeout_ms = cfg.get("statement_timeout_ms", 30000)

    start = time.perf_counter()
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SET LOCAL statement_timeout = '{int(timeout_ms)}'")  # safe: int from config
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute(sql)
                if cur.description is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Query did not return results. Only SELECT queries are supported.",
                    )
                columns = [desc[0] for desc in cur.description]
                rows = [list(row) for row in cur.fetchmany(max_rows + 1)]
                truncated = len(rows) > max_rows
                if truncated:
                    rows = rows[:max_rows]
    except HTTPException:
        raise
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        _record_history(sql, elapsed, 0, "error")
        logger.warning("SQL Runner error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    elapsed = (time.perf_counter() - start) * 1000
    _record_history(sql, elapsed, len(rows), "ok")

    # Serialize values that are not JSON-native
    serialized_rows = []
    for row in rows:
        serialized_rows.append([_serialize(v) for v in row])

    return SqlResponse(
        columns=columns,
        rows=serialized_rows,
        row_count=len(serialized_rows),
        truncated=truncated,
        elapsed_ms=round(elapsed, 1),
    )


def _serialize(v: Any) -> Any:
    """Convert non-JSON-native types to strings."""
    if v is None:
        return None
    if isinstance(v, (int, float, bool, str)):
        return v
    return str(v)


def _record_history(sql: str, elapsed_ms: float, row_count: int, status: str):
    _history.append({
        "sql": sql[:500],
        "executed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_ms": round(elapsed_ms, 1),
        "row_count": row_count,
        "status": status,
    })
    while len(_history) > _MAX_HISTORY:
        _history.pop(0)


# ---------------------------------------------------------------------------
# GET /schema
# ---------------------------------------------------------------------------
@router.get("/schema")
async def get_schema():
    """Return all user tables and views with their columns."""
    sql = """
        SELECT
            t.table_schema,
            t.table_name,
            t.table_type,
            c.column_name,
            c.data_type,
            c.is_nullable
        FROM information_schema.tables t
        JOIN information_schema.columns c
          ON c.table_schema = t.table_schema
         AND c.table_name = t.table_name
        WHERE t.table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY t.table_schema, t.table_name, c.ordinal_position
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(sql)
            raw = cur.fetchall()

    tables: dict[str, dict] = {}
    for schema, table, ttype, col_name, dtype, nullable in raw:
        key = f"{schema}.{table}"
        if key not in tables:
            tables[key] = {
                "schema_name": schema,
                "table_name": table,
                "table_type": ttype,
                "columns": [],
            }
        tables[key]["columns"].append({
            "name": col_name,
            "data_type": dtype,
            "is_nullable": nullable == "YES",
        })

    return {"tables": list(tables.values())}


# ---------------------------------------------------------------------------
# GET /history
# ---------------------------------------------------------------------------
@router.get("/history")
async def get_history():
    """Return recent query history (in-memory, last 50)."""
    return {"history": list(reversed(_history))}
