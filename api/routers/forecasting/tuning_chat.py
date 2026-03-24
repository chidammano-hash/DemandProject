"""Tuning Chat endpoints — AI-powered LGBM hyperparameter tuning sessions."""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from api.core import get_conn

logger = logging.getLogger(__name__)

router = APIRouter(tags=["lgbm-tuning"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CreateSessionBody(BaseModel):
    title: str = Field(default="New Tuning Session", min_length=1, max_length=200)


class SendMessageBody(BaseModel):
    content: str = Field(min_length=1, max_length=5000)


class ConfirmRunBody(BaseModel):
    recommendation_message_id: int
    override_params: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_message(
    conn: Any,
    session_id: str,
    role: str,
    content: str,
    message_type: str = "text",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Insert a chat message and return it as a dict."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tuning_chat_message
                (session_id, role, content, message_type, metadata)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING message_id, session_id, role, content, message_type, metadata, created_at
            """,
            (session_id, role, content, message_type,
             json.dumps(metadata) if metadata else None),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=500, detail="Failed to insert message")
        cols = ["message_id", "session_id", "role", "content",
                "message_type", "metadata", "created_at"]
        conn.commit()
        return dict(zip(cols, row))


def _fetch_messages(conn: Any, session_id: str) -> list[dict[str, Any]]:
    """Fetch all messages for a session ordered by creation time."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT message_id, session_id, role, content,
                   message_type, metadata, created_at
            FROM tuning_chat_message
            WHERE session_id = %s::uuid
            ORDER BY created_at
            """,
            (session_id,),
        )
        cols = ["message_id", "session_id", "role", "content",
                "message_type", "metadata", "created_at"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]




# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/lgbm-tuning/chat/sessions", status_code=201)
def create_session(body: CreateSessionBody) -> dict:
    """Create a new chat session, seeded with current run summary context."""
    with get_conn() as conn:
        # Seed context with recent run summary
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_id, run_label, accuracy_pct, wape, bias, status
                FROM lgbm_tuning_run
                ORDER BY started_at DESC LIMIT 10
                """,
            )
            cols = ["run_id", "run_label", "accuracy_pct", "wape", "bias", "status"]
            recent_runs = [dict(zip(cols, r)) for r in cur.fetchall()]

            context = {"recent_runs": recent_runs, "total_runs": len(recent_runs)}

            cur.execute(
                """
                INSERT INTO tuning_chat_session (title, context)
                VALUES (%s, %s)
                RETURNING session_id, title, status, created_at, updated_at
                """,
                (body.title, json.dumps(context, default=str)),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=500, detail="Failed to create session")
            session = dict(zip(
                ["session_id", "title", "status", "created_at", "updated_at"], row,
            ))
        conn.commit()
        return {"session": session}


@router.get("/lgbm-tuning/chat/sessions")
def list_sessions(
    status: str = Query(default="active", max_length=20),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict:
    """List chat sessions."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.session_id, s.title, s.status, s.created_at, s.updated_at,
                       COUNT(m.message_id) AS message_count
                FROM tuning_chat_session s
                LEFT JOIN tuning_chat_message m ON m.session_id = s.session_id
                WHERE s.status = %s
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
                LIMIT %s
                """,
                (status, limit),
            )
            cols = ["session_id", "title", "status", "created_at", "updated_at", "message_count"]
            sessions = [dict(zip(cols, r)) for r in cur.fetchall()]
        return {"sessions": sessions}


@router.get("/lgbm-tuning/chat/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    """Get session with full message history."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT session_id, title, status, context, created_at, updated_at
                FROM tuning_chat_session
                WHERE session_id = %s::uuid
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Session not found")
            session = dict(zip(
                ["session_id", "title", "status", "context", "created_at", "updated_at"], row,
            ))

        messages = _fetch_messages(conn, session_id)
        return {"session": session, "messages": messages}


@router.post("/lgbm-tuning/chat/sessions/{session_id}/messages")
def send_message(session_id: str, body: SendMessageBody) -> dict:
    """Send a user message and get AI response.

    Flow:
    1. Validate session exists
    2. Insert user message
    3. Load message history
    4. Call TuningAdvisorAgent.run_turn()
    5. Insert AI response message(s)
    6. Return new messages
    """
    with get_conn() as conn:
        # Validate session
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status FROM tuning_chat_session WHERE session_id = %s::uuid",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Session not found")
            if row[0] != "active":
                raise HTTPException(status_code=400, detail="Session is archived")

        # Insert user message
        user_msg = _insert_message(conn, session_id, "user", body.content)

        # Load full history for AI context
        all_messages = _fetch_messages(conn, session_id)

        # Build conversation for the agent (only user/assistant text messages)
        conversation: list[dict[str, Any]] = []
        for m in all_messages:
            if m["role"] in ("user", "assistant") and m["message_type"] == "text":
                conversation.append({"role": m["role"], "content": m["content"]})
            elif m["message_type"] == "recommendation" and m["role"] == "assistant":
                conversation.append({"role": "assistant", "content": m["content"]})
            elif m["message_type"] in ("run_completed", "run_failed") and m["role"] == "system":
                conversation.append({"role": "user", "content": f"[System: {m['content']}]"})

        # Apply sliding window
        max_history = 40
        if len(conversation) > max_history:
            conversation = conversation[:3] + conversation[-(max_history - 3):]

        # Run AI turn
        from common.ai.tuning_advisor import TuningAdvisorAgent
        advisor = TuningAdvisorAgent()
        ai_text, tool_calls = advisor.run_turn(session_id, conversation)

        # Insert AI response messages
        new_messages = [user_msg]

        # Check if any tool call was a recommendation
        recommendation = None
        for tc in tool_calls:
            if tc["tool_name"] == "recommend_params" and isinstance(tc["tool_result"], dict):
                if "error" not in tc["tool_result"]:
                    recommendation = tc["tool_result"]

        if recommendation:
            # Insert recommendation as a separate message
            rec_msg = _insert_message(
                conn, session_id, "assistant", ai_text or "Here is my recommendation:",
                message_type="text",
            )
            new_messages.append(rec_msg)

            rec_card = _insert_message(
                conn, session_id, "assistant",
                json.dumps(recommendation),
                message_type="recommendation",
                metadata=recommendation,
            )
            new_messages.append(rec_card)
        elif ai_text:
            ai_msg = _insert_message(conn, session_id, "assistant", ai_text)
            new_messages.append(ai_msg)

        # Update session timestamp
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE tuning_chat_session SET updated_at = now() WHERE session_id = %s::uuid",
                (session_id,),
            )
        conn.commit()

        return {"messages": new_messages}


@router.post("/lgbm-tuning/chat/sessions/{session_id}/confirm-run")
def confirm_run(session_id: str, body: ConfirmRunBody) -> dict:
    """Confirm an AI recommendation and trigger async backtest."""
    with get_conn() as conn:
        # Validate session
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status FROM tuning_chat_session WHERE session_id = %s::uuid",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Session not found")

        # Fetch recommendation message
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT metadata FROM tuning_chat_message
                WHERE message_id = %s AND session_id = %s::uuid
                  AND message_type = 'recommendation'
                """,
                (body.recommendation_message_id, session_id),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=400, detail="Recommendation message not found")
            recommendation = row[0] if isinstance(row[0], dict) else json.loads(row[0] or "{}")

        # Check for active runs
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM lgbm_tuning_run WHERE status = 'running'",
            )
            active_count = cur.fetchone()[0]  # type: ignore[index]
            if active_count > 0:
                raise HTTPException(status_code=409, detail="A tuning run is already active")

        # Extract and optionally override params
        overrides = recommendation.get("overrides", {})
        if body.override_params:
            overrides.update(body.override_params)

        strategy_label = recommendation.get("strategy_label", "chat_experiment")

        # Register run
        from common.ml.tuning_tracker import register_run
        run_id = register_run(
            run_label=strategy_label,
            model_id="lgbm_cluster",
            params=overrides,
            notes=f"Chat session {session_id}: {recommendation.get('description', '')}",
        )

        # Insert run_started message
        _insert_message(
            conn, session_id, "system",
            f"Run #{run_id} started with strategy '{strategy_label}'",
            message_type="run_started",
            metadata={"run_id": run_id, "params": overrides, "strategy_label": strategy_label},
        )
        conn.commit()

    # Submit via JobManager — gets PID tracking, cancel, log streaming, restart recovery
    from common.job_registry import JobManager
    mgr = JobManager()
    mgr.submit_job(
        job_type="tuning_backtest",
        params={
            "run_id": run_id,
            "session_id": session_id,
            "overrides": overrides,
            "strategy_label": strategy_label,
        },
        label=f"AI Tuning: {strategy_label}",
        triggered_by="ai_tuning_chat",
    )

    return {"run_id": run_id, "status": "started", "strategy_label": strategy_label}


@router.get("/lgbm-tuning/chat/sessions/{session_id}/run-status/{run_id}")
def get_run_status(session_id: str, run_id: int) -> dict:
    """Poll for run completion status."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT status, started_at, completed_at,
                   accuracy_pct, wape, bias, n_predictions, n_dfus
            FROM lgbm_tuning_run WHERE run_id = %s
            """,
            (run_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")

        result: dict[str, Any] = {
            "run_id": run_id,
            "status": row[0],
            "started_at": row[1],
            "completed_at": row[2],
        }

        if row[1] and row[2]:
            result["elapsed_seconds"] = int((row[2] - row[1]).total_seconds())
        elif row[1]:
            import datetime
            elapsed = datetime.datetime.now(datetime.timezone.utc) - row[1]
            result["elapsed_seconds"] = int(elapsed.total_seconds())

        if row[0] == "completed" and row[3] is not None:
            result["results"] = {
                "accuracy_pct": row[3],
                "wape": row[4],
                "bias": row[5],
                "n_predictions": row[6],
                "n_dfus": row[7],
            }

    return result
