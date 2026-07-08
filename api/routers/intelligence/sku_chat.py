"""SKU Chatbot API router — spec docs/specs/06-ai-platform/07-sku-chatbot.md.

Endpoints under ``/sku-chat``:
  GET  /sku-chat/config            Active runtime/model tiers, auth mode, guardrails (no secrets)
  POST /sku-chat/session           Create a session id bound to a SKU (key-guarded)
  GET  /sku-chat/session/{id}      Session + ordered message history (Phase 3)
  POST /sku-chat/stream            Stream one chat turn as Server-Sent Events (key-guarded)

The agent (and the Claude Agent SDK it needs) is imported lazily inside the
stream handler so the API boots without the ``agent`` extra installed.
Persistence (Phase 3) is best-effort — see common/ai/sku_chat/store.py.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.auth import require_api_key
from common.ai.sku_chat import store
from common.ai.sku_chat.config import get_sku_chat_config

log = logging.getLogger(__name__)
router = APIRouter(prefix="/sku-chat", tags=["sku-chat"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ChatHistoryMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class SessionRequest(BaseModel):
    item_id: str
    loc: str
    customer_group: str = ""


class StreamRequest(BaseModel):
    question: str
    item_id: str
    loc: str
    customer_group: str = ""
    session_id: str | None = None
    history: list[ChatHistoryMessage] = []
    model_tier: str | None = None  # force "fast" | "standard" | "deep"
    model: str | None = None  # force a specific model id
    page_focus: str | None = None  # what UI page the planner is on (per-page customization)


def _persistence_enabled(cfg: dict) -> bool:
    return bool((cfg.get("persistence") or {}).get("enabled", True))


# ---------------------------------------------------------------------------
# GET /sku-chat/config
# ---------------------------------------------------------------------------
@router.get("/config")
async def get_config() -> dict[str, Any]:
    """Return the active routing config for the UI (never leaks credentials)."""
    cfg = get_sku_chat_config()
    runtime_provider = str((cfg.get("runtime") or {}).get("provider", "claude")).lower()
    return {
        "runtime_provider": runtime_provider,
        "auth_mode": (cfg.get("auth") or {}).get("mode", "auto"),
        "models": cfg.get("models") or {},
        "codex_models": cfg.get("codex_models") or {},
        "routing": {
            "default_tier": (cfg.get("routing") or {}).get("default_tier", "standard"),
            "allow_user_override": (cfg.get("routing") or {}).get(
                "allow_user_override", True
            ),
        },
        "guardrails": cfg.get("guardrails") or {},
        "tools": (cfg.get("tools") or {}).get("allowed") or [],
        "persistence": _persistence_enabled(cfg),
    }


# ---------------------------------------------------------------------------
# POST /sku-chat/session
# ---------------------------------------------------------------------------
@router.post("/session", dependencies=[Depends(require_api_key)])
async def create_session(body: SessionRequest, request: Request) -> dict[str, Any]:
    """Create a session id the client uses to correlate a conversation."""
    from api.core import _get_pool

    session_id = str(uuid.uuid4())
    cfg = get_sku_chat_config()
    if _persistence_enabled(cfg):
        store.ensure_session(
            _get_pool(),
            session_id,
            body.item_id,
            body.customer_group,
            body.loc,
            request.headers.get("X-User"),
        )
    return {
        "session_id": session_id,
        "item_id": body.item_id,
        "customer_group": body.customer_group,
        "loc": body.loc,
    }


# ---------------------------------------------------------------------------
# GET /sku-chat/session/{session_id}
# ---------------------------------------------------------------------------
@router.get("/session/{session_id}", dependencies=[Depends(require_api_key)])
async def get_session_history(session_id: str) -> dict[str, Any]:
    """Return a session and its ordered message history (Phase 3).

    Key-guarded like its write siblings: the history contains conversation
    content, so it must not be readable by an unauthenticated caller who guesses
    or enumerates session ids.
    """
    from api.core import _get_pool

    session = store.get_session(_get_pool(), session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


# ---------------------------------------------------------------------------
# POST /sku-chat/stream
# ---------------------------------------------------------------------------
def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


@router.post("/stream", dependencies=[Depends(require_api_key)])
async def stream_turn(body: StreamRequest, request: Request) -> StreamingResponse:
    """Stream one SKU-scoped chat turn as Server-Sent Events."""
    from api.core import _get_pool
    from common.ai.sku_chat import champion_adjust
    from common.ai.sku_chat.agent import SkuChatAgent, SkuChatContext

    pool = _get_pool()
    cfg = get_sku_chat_config()
    agent = SkuChatAgent(pool, cfg)
    ctx = SkuChatContext(body.item_id, body.customer_group, body.loc)
    history = [{"role": m.role, "content": m.content} for m in body.history]
    session_id = body.session_id or str(uuid.uuid4())
    created_by = request.headers.get("X-User")
    persist = _persistence_enabled(cfg)

    async def generate():
        started = time.monotonic()
        tier: str | None = None
        model: str | None = None
        full_text = ""
        usage: dict[str, Any] | None = None
        cost: float | None = None
        tool_calls = 0
        truncated = False

        if persist:
            await asyncio.to_thread(
                store.ensure_session,
                pool, session_id, body.item_id, body.customer_group, body.loc, created_by,
            )
            await asyncio.to_thread(
                store.save_message, pool, session_id, "user", body.question,
            )

        adjust_proposed = False
        try:
            async for event in agent.stream_turn(
                body.question,
                ctx,
                history=history,
                model_tier=body.model_tier,
                model=body.model,
                session_id=session_id,
                page_focus=body.page_focus,
            ):
                etype = event.get("type")
                if etype == "meta":
                    tier = event.get("tier")
                    model = event.get("model")
                    event = {**event, "session_id": session_id}
                elif etype == "text":
                    full_text += event.get("chunk", "")
                elif etype == "tool":
                    tool_calls += 1
                    if "apply_champion_adjustment" in (event.get("name") or ""):
                        adjust_proposed = True
                elif etype == "result":
                    full_text = event.get("text") or full_text
                    usage = event.get("usage")
                    cost = event.get("cost_usd")
                elif etype == "error" and event.get("truncated"):
                    truncated = True
                yield _sse(event)
            # Surface any adjustments the agent staged this turn for planner approval.
            if adjust_proposed:
                for pend in champion_adjust.list_pending(pool, session_id):
                    yield _sse(
                        {
                            "type": "approval_request",
                            "approval_id": pend.get("approval_id"),
                            "item_id": pend.get("item_id"),
                            "loc": pend.get("loc"),
                            "preview": pend.get("preview"),
                        }
                    )
        except (ValueError, RuntimeError):
            log.exception("sku-chat stream failed for %s@%s", body.item_id, body.loc)
            yield _sse(
                {"type": "error", "message": "Chat failed. Check server logs for details."}
            )
        finally:
            if persist:
                latency_ms = int((time.monotonic() - started) * 1000)
                message_id = await asyncio.to_thread(
                    store.save_message, pool, session_id, "assistant", full_text, model, tier,
                )
                await asyncio.to_thread(
                    store.log_call,
                    pool,
                    session_id,
                    message_id,
                    model=model,
                    tier=tier,
                    usage=usage,
                    cost_usd=cost,
                    tool_calls=tool_calls,
                    latency_ms=latency_ms,
                    truncated=truncated,
                )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# POST /sku-chat/adjustment/{approval_id} — approve (apply) or reject a staged
# champion-forecast adjustment. Approve calls the guarded save_adjustment.
# ---------------------------------------------------------------------------
class AdjustmentDecision(BaseModel):
    decision: str  # "approve" | "reject"


@router.post("/adjustment/{approval_id}", dependencies=[Depends(require_api_key)])
async def decide_adjustment(approval_id: str, body: AdjustmentDecision) -> dict[str, Any]:
    """Approve (apply) or reject a champion-forecast adjustment the agent staged."""
    from api.core import _get_pool
    from common.ai.sku_chat import champion_adjust

    pool = _get_pool()
    decision = body.decision.lower()
    try:
        if decision == "reject":
            return champion_adjust.reject_adjustment(pool, approval_id)
        if decision == "approve":
            return await asyncio.to_thread(
                champion_adjust.apply_adjustment, pool, approval_id
            )
    except champion_adjust.AdjustmentError:
        log.exception("sku-chat: adjustment %s could not be processed", approval_id)
        raise HTTPException(status_code=400, detail="Could not process adjustment.") from None
    raise HTTPException(status_code=400, detail="decision must be 'approve' or 'reject'")
