"""API tests for the SKU Chatbot router (/sku-chat/*)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport


async def fake_stream(self, question, ctx, *, history=None, model_tier=None, model=None, session_id=None, page_focus=None):
    """Stand-in for SkuChatAgent.stream_turn — no Agent SDK, no model calls."""
    yield {"type": "meta", "tier": "standard", "model": "claude-sonnet-4-6"}
    yield {"type": "text", "chunk": "Lead time is 30 days."}
    yield {"type": "result", "text": "Lead time is 30 days.", "cost_usd": 0.01, "usage": {}}


async def fake_stream_with_adjust(self, question, ctx, *, history=None, model_tier=None, model=None, session_id=None, page_focus=None):
    """Stand-in that proposes a champion adjustment (fires the apply tool)."""
    yield {"type": "meta", "tier": "deep", "model": "claude-opus-4-8"}
    yield {
        "type": "tool",
        "name": "mcp__sku__apply_champion_adjustment",
        "input": {"item_id": "100320", "loc": "DC1", "rationale": "reduce Q3"},
    }
    yield {"type": "result", "text": "Staged an adjustment; approve to apply.", "cost_usd": 0.02, "usage": {}}


@pytest.mark.asyncio
async def test_config_endpoint(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    from api.main import app

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/sku-chat/config")

    assert resp.status_code == 200
    data = resp.json()
    assert data["runtime_provider"] == "codex"
    assert data["auth_mode"] == "auto"
    assert data["models"]["deep"] == "claude-opus-4-8"
    assert data["codex_models"]["standard"] == "gpt-5.5"
    assert "mcp__sku__get_sku_profile" in data["tools"]
    assert data["persistence"] is True
    assert "ANTHROPIC_API_KEY" not in resp.text  # never leak secrets
    assert "CODEX_API_KEY" not in resp.text


@pytest.mark.asyncio
async def test_create_session(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with patch("api.core._get_pool", return_value=MagicMock()), patch(
        "common.ai.sku_chat.store.ensure_session"
    ) as ensure:
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/sku-chat/session",
                json={"item_id": "100320", "loc": "DC1", "customer_group": "RETAIL"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["item_id"] == "100320"
    assert data["session_id"]
    ensure.assert_called_once()  # session persisted


@pytest.mark.asyncio
async def test_session_requires_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    from api.main import app

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/sku-chat/session",
            json={"item_id": "100320", "loc": "DC1"},
        )
    assert resp.status_code == 401

    with patch("api.core._get_pool", return_value=MagicMock()), patch(
        "common.ai.sku_chat.store.ensure_session"
    ):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            ok = await client.post(
                "/sku-chat/session",
                json={"item_id": "100320", "loc": "DC1"},
                headers={"X-API-Key": "secret"},
            )
    assert ok.status_code == 200


@pytest.mark.asyncio
async def test_stream_emits_sse_events_and_persists(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with patch("api.core._get_pool", return_value=MagicMock()), patch(
        "common.ai.sku_chat.agent.SkuChatAgent.stream_turn", fake_stream
    ), patch("common.ai.sku_chat.store.ensure_session") as ensure, patch(
        "common.ai.sku_chat.store.save_message", return_value=42
    ) as save, patch("common.ai.sku_chat.store.log_call") as log_call:
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/sku-chat/stream",
                json={
                    "question": "What is the lead time?",
                    "item_id": "100320",
                    "loc": "DC1",
                    "customer_group": "RETAIL",
                    "session_id": "sess-1",
                },
            )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert '"type": "meta"' in body
    assert '"session_id": "sess-1"' in body  # session id injected into meta
    assert "Lead time is 30 days." in body
    assert '"type": "result"' in body

    # Persistence: user + assistant messages saved, call logged.
    ensure.assert_called_once()
    assert save.call_count == 2
    log_call.assert_called_once()


@pytest.mark.asyncio
async def test_get_session_history_found(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    session = {
        "session_id": "sess-1",
        "item_id": "100320",
        "loc": "DC1",
        "messages": [{"id": 1, "role": "user", "content": "hi"}],
    }
    with patch("api.core._get_pool", return_value=MagicMock()), patch(
        "common.ai.sku_chat.store.get_session", return_value=session
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sku-chat/session/sess-1")

    assert resp.status_code == 200
    assert resp.json()["messages"][0]["content"] == "hi"


@pytest.mark.asyncio
async def test_get_session_history_requires_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    from api.main import app

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/sku-chat/session/sess-1")
    assert resp.status_code == 401  # history must not be readable without the key


@pytest.mark.asyncio
async def test_get_session_history_404(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with patch("api.core._get_pool", return_value=MagicMock()), patch(
        "common.ai.sku_chat.store.get_session", return_value=None
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sku-chat/session/missing")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Champion-forecast adjustment (agentic write tool + approval gate)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_emits_approval_request_when_adjustment_staged(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    pending = [
        {
            "approval_id": "appr-1",
            "item_id": "100320",
            "loc": "DC1",
            "preview": {"recommendation_code": "SCALE_DOWN", "rec_pct_change": -15.0},
        }
    ]
    with patch("api.core._get_pool", return_value=MagicMock()), patch(
        "common.ai.sku_chat.agent.SkuChatAgent.stream_turn", fake_stream_with_adjust
    ), patch("common.ai.sku_chat.store.ensure_session"), patch(
        "common.ai.sku_chat.store.save_message", return_value=1
    ), patch("common.ai.sku_chat.store.log_call"), patch(
        "common.ai.sku_chat.champion_adjust.list_pending", return_value=pending
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/sku-chat/stream",
                json={
                    "question": "Adjust this SKU's forecast down for Q3",
                    "item_id": "100320",
                    "loc": "DC1",
                    "session_id": "s1",
                },
            )

    assert resp.status_code == 200
    body = resp.text
    assert '"type": "approval_request"' in body
    assert '"approval_id": "appr-1"' in body


@pytest.mark.asyncio
async def test_decide_adjustment_approve(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with patch("api.core._get_pool", return_value=MagicMock()), patch(
        "common.ai.sku_chat.champion_adjust.apply_adjustment",
        return_value={"approval_id": "appr-1", "status": "approved", "result": {"saved": True}},
    ) as apply:
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/sku-chat/adjustment/appr-1", json={"decision": "approve"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "approved"
    apply.assert_called_once()


@pytest.mark.asyncio
async def test_decide_adjustment_reject(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with patch("api.core._get_pool", return_value=MagicMock()), patch(
        "common.ai.sku_chat.champion_adjust.reject_adjustment",
        return_value={"approval_id": "appr-1", "status": "rejected"},
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/sku-chat/adjustment/appr-1", json={"decision": "reject"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "rejected"


@pytest.mark.asyncio
async def test_decide_adjustment_invalid_decision(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with patch("api.core._get_pool", return_value=MagicMock()):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/sku-chat/adjustment/appr-1", json={"decision": "maybe"})

    assert resp.status_code == 400
