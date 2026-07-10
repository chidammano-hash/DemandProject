"""API tests for grounded Customer Analytics questions."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from httpx import ASGITransport

from common.ai.customer_analytics_assistant import CustomerAnalyticsAnswer
from tests.api.conftest import make_pool


@pytest.mark.asyncio
async def test_customer_analytics_ask_uses_filtered_database_context():
    pool, _, _ = make_pool()
    context = {
        "filters": {"state": "TX"},
        "kpis": [{"key": "fill_rate", "value": 91.2, "delta": -1.4}],
        "top_customers": [],
        "service_risks": [],
    }
    answer = CustomerAnalyticsAnswer(
        answer="Texas service softened by 1.4 points.",
        provider="codex",
        model="gpt-5.5",
        tier="deep",
    )
    build_context = AsyncMock(return_value=context)
    ask = AsyncMock(return_value=answer)

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "api.routers.intelligence.customer_analytics.assistant._build_dashboard_context",
            build_context,
        ),
        patch(
            "api.routers.intelligence.customer_analytics.assistant.answer_customer_question",
            ask,
        ),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/customer-analytics/ask",
                json={
                    "question": "Why is service slipping?",
                    "filters": {"state": "TX"},
                    "active_view": "service",
                    "history": [],
                },
            )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Texas service softened by 1.4 points.",
        "provider": "codex",
        "model": "gpt-5.5",
        "tier": "deep",
        "evidence": ["KPIs", "Top demand customers", "Lowest fill-rate customers"],
    }
    build_context.assert_awaited_once()
    ask.assert_awaited_once_with("Why is service slipping?", context, history=[])
