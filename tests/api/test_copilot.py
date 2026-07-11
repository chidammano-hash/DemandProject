"""API contract tests for owner-scoped grounded Copilot sessions."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from httpx import ASGITransport

from common.auth import CurrentUser, get_current_user


class _FakeCopilotService:
    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, object]] = {}

    async def create_session(
        self,
        *,
        owner: CurrentUser,
        context: dict[str, object],
        idempotency_key: str,
    ) -> dict[str, object]:
        session = {
            "session_id": "session-1",
            "owner_id": owner.user_id,
            "context": context,
            "status": "active",
        }
        self.sessions[owner.user_id] = session
        return session

    async def get_session(self, session_id: str, *, owner: CurrentUser) -> dict[str, object] | None:
        session = self.sessions.get(owner.user_id)
        return session if session and session["session_id"] == session_id else None

    async def run_turn(
        self,
        session_id: str,
        *,
        owner: CurrentUser,
        prompt: str,
        idempotency_key: str,
    ) -> dict[str, object] | None:
        if await self.get_session(session_id, owner=owner) is None:
            return None
        return {
            "turn_id": "turn-1",
            "answer": "The current forecast is 42 units.",
            "citations": [
                {
                    "evidence_id": "evidence-1",
                    "claim": "The current forecast is 42 units",
                    "source": "copilot.dfu_evidence",
                    "business_key": "ITEM-1|LOC-1",
                    "freshness": "active_release",
                    "content_hash": "a" * 64,
                    "values": {"forecast_qty": 42},
                }
            ],
            "action_request": None,
        }


async def _planner() -> CurrentUser:
    return CurrentUser(user_id="planner-1", email="planner@example.com", role="planner")


async def _other_planner() -> CurrentUser:
    return CurrentUser(user_id="planner-2", email="other@example.com", role="planner")


@pytest.fixture
async def copilot_client() -> AsyncIterator[tuple[httpx.AsyncClient, _FakeCopilotService]]:
    from api.main import app
    from api.routers.intelligence.copilot import _get_copilot_service

    service = _FakeCopilotService()
    app.dependency_overrides[get_current_user] = _planner
    app.dependency_overrides[_get_copilot_service] = lambda: service
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, service
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(_get_copilot_service, None)


@pytest.mark.asyncio
async def test_create_session_resolves_owner_and_rejects_client_lineage(
    copilot_client: tuple[httpx.AsyncClient, _FakeCopilotService],
) -> None:
    client, _ = copilot_client

    response = await client.post(
        "/ai-copilot/sessions",
        headers={"Idempotency-Key": "create-1"},
        json={
            "page": "itemAnalysis",
            "item_id": "ITEM-1",
            "customer_group": "",
            "loc": "LOC-1",
        },
    )

    assert response.status_code == 201
    assert response.json()["owner_id"] == "planner-1"

    rejected = await client.post(
        "/ai-copilot/sessions",
        headers={"Idempotency-Key": "create-2"},
        json={
            "page": "itemAnalysis",
            "item_id": "ITEM-1",
            "loc": "LOC-1",
            "promotion_id": 999,
        },
    )
    assert rejected.status_code == 422


@pytest.mark.asyncio
async def test_session_history_is_owner_scoped(
    copilot_client: tuple[httpx.AsyncClient, _FakeCopilotService],
) -> None:
    client, _ = copilot_client
    await client.post(
        "/ai-copilot/sessions",
        headers={"Idempotency-Key": "create-1"},
        json={"page": "overview"},
    )

    from api.main import app

    app.dependency_overrides[get_current_user] = _other_planner
    response = await client.get("/ai-copilot/sessions/session-1")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_turn_returns_only_final_validated_answer_and_citations(
    copilot_client: tuple[httpx.AsyncClient, _FakeCopilotService],
) -> None:
    client, _ = copilot_client
    await client.post(
        "/ai-copilot/sessions",
        headers={"Idempotency-Key": "create-1"},
        json={"page": "itemAnalysis", "item_id": "ITEM-1", "loc": "LOC-1"},
    )

    response = await client.post(
        "/ai-copilot/sessions/session-1/turns",
        headers={"Idempotency-Key": "turn-1"},
        json={"prompt": "Explain the forecast"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "The current forecast is 42 units."
    assert payload["citations"][0]["content_hash"] == "a" * 64
    assert payload["action_request"] is None
