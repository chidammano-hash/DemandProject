"""Chat / Natural Language Queries endpoint (feature 12)."""
from __future__ import annotations

from typing import Any
import json
import logging
import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from common.domain_specs import DOMAIN_SPECS
from api.core import get_conn, get_openai
from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    domain: str = ""


def _build_schema_summary() -> str:
    """Compact schema summary for the system prompt."""
    lines: list[str] = []
    for spec in DOMAIN_SPECS.values():
        cols = []
        for c in spec.columns:
            if c in spec.int_fields:
                cols.append(f"{c} (int)")
            elif c in spec.float_fields:
                cols.append(f"{c} (numeric)")
            elif c in spec.date_fields:
                cols.append(f"{c} (date)")
            else:
                cols.append(f"{c} (text)")
        lines.append(f"Table: {spec.table}  PK: {spec.ck_field}  Key: {', '.join(spec.key_fields)}")
        lines.append(f"  Columns: {', '.join(cols)}")
    return "\n".join(lines)


def _vector_search(question_embedding: list[float], top_k: int = 10) -> list[str]:
    """Retrieve most relevant schema context via pgvector cosine similarity."""
    sql = """
        SELECT source_text
        FROM chat_embeddings
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, (str(question_embedding), top_k))
            return [r[0] for r in cur.fetchall()]
    except Exception:
        logger.warning("pgvector embedding search failed", exc_info=True)
        return []


CHAT_SYSTEM_PROMPT = """You are a SQL expert for a demand forecasting PostgreSQL database called Demand Studio.

## Schema
{schema}

## Retrieved Context
{context}

## Business Rules
- Forecast accuracy formula: 100 - (100 * SUM(ABS(basefcst_pref - tothist_dmd)) / NULLIF(ABS(SUM(tothist_dmd)), 0))
- Bias formula: (SUM(basefcst_pref) / NULLIF(SUM(tothist_dmd), 0)) - 1
- WAPE formula: 100 * SUM(ABS(basefcst_pref - tothist_dmd)) / NULLIF(ABS(SUM(tothist_dmd)), 0)
- Only sales rows with type=1 exist in fact_sales_monthly
- All startdate/fcstdate values are month-start dates (first day of month)
- Forecast lag is 0-4 months (startdate - fcstdate in months)
- model_id on forecasts defaults to 'external' for source-system forecasts

## Instructions
1. Answer the user's question about the demand data.
2. If you need to query data, generate a PostgreSQL SELECT statement.
3. ONLY generate SELECT statements. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or any DDL/DML.
4. Always include LIMIT 500 at the end of your queries.
5. Use proper column names exactly as listed in the schema.
6. For date filtering, use ISO format: '2024-01-01'.

Respond in JSON format:
{{"answer": "your natural language answer", "sql": "SELECT ... LIMIT 500"}}

If no SQL is needed (e.g., the question is about schema or definitions), set sql to null:
{{"answer": "your explanation", "sql": null}}"""


def _is_safe_sql(sql: str) -> bool:
    """Check that SQL is a SELECT statement only."""
    cleaned = re.sub(r'--[^\n]*', '', sql)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip().upper()
    if not cleaned.startswith("SELECT"):
        return False
    forbidden = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "GRANT", "REVOKE", "COPY"}
    tokens = set(re.findall(r'\b[A-Z]+\b', cleaned))
    if tokens & forbidden:
        return False
    return True


def _execute_readonly_sql(sql: str) -> tuple[list[str], list[list[Any]]]:
    """Execute SQL in a read-only transaction with timeout. Returns (columns, rows)."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL statement_timeout = '5000'")
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute(sql)
        if cur.description is None:
            return [], []
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchmany(500)
        return columns, [list(r) for r in rows]


@router.post("/chat", dependencies=[Depends(require_api_key)])
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    client = get_openai()

    # 1. Embed the question
    embed_resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=question,
    )
    question_embedding = embed_resp.data[0].embedding

    # 2. Vector search for relevant context
    context_chunks = _vector_search(question_embedding, top_k=10)
    context_text = "\n".join(f"- {chunk}" for chunk in context_chunks) if context_chunks else "(no embeddings available)"

    # 3. Build prompt
    schema_summary = _build_schema_summary()
    system_prompt = CHAT_SYSTEM_PROMPT.format(schema=schema_summary, context=context_text)

    user_msg = question
    if req.domain.strip():
        user_msg = f"[Current domain context: {req.domain.strip()}] {question}"

    # 4. Call GPT-4o
    chat_resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )
    raw_content = chat_resp.choices[0].message.content or "{}"

    # 5. Parse response
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        parsed = {"answer": raw_content, "sql": None}

    answer = parsed.get("answer", "I couldn't generate an answer.")
    sql = parsed.get("sql")
    data = None
    columns: list[str] = []
    row_count = None
    error_msg = None

    # 6. Execute SQL if present and safe
    if sql and isinstance(sql, str) and sql.strip():
        sql = sql.strip().rstrip(";")
        if not _is_safe_sql(sql):
            error_msg = "Generated SQL was blocked for safety reasons (only SELECT allowed)."
            sql = None
        else:
            try:
                columns, raw_rows = _execute_readonly_sql(sql)
                row_count = len(raw_rows)
                data = [
                    {col: (str(val) if val is not None else None) for col, val in zip(columns, row)}
                    for row in raw_rows
                ]
            except Exception as exc:
                error_msg = f"SQL execution error: {exc}"

    result: dict[str, Any] = {
        "answer": answer,
        "sql": sql,
        "data": data,
        "columns": columns,
        "row_count": row_count,
    }
    if error_msg:
        result["error"] = error_msg
    return result
