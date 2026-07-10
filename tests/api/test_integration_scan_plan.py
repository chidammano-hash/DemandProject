"""API tests for the AI-assisted integration scan planner."""
from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
from httpx import ASGITransport

from tests.api.conftest import make_pool


@pytest.mark.asyncio
async def test_scan_plan_endpoint_returns_planner_result():
    pool, _, _ = make_pool(fetchall_return=[], fetchone_return=None, description=[])
    payload = {
        "plan_id": "plan-1",
        "provider": "ollama",
        "model": "llama3.1:8b",
        "status": "planned",
        "confidence": 0.9,
        "explanation": "Run sales first.",
        "risk_flags": [],
        "questions": [],
        "recommended_chain": [{"step": 1, "domain": "sales", "mode": "delta", "slice": None}],
        "evidence": [{"kind": "scan", "label": "sales", "value": "hash mismatch"}],
        "scanned_at": "2026-07-10T10:00:00Z",
        "changes": [{"domain": "sales", "kind": "fact", "changed": True, "reason": "hash mismatch"}],
        "proposed_chain": [{"step": 1, "domain": "sales", "mode": "delta", "slice": None}],
    }
    with (
        patch("api.core._get_pool", return_value=pool),
        patch("api.routers.platform.integration_chain.run_scan_planner", return_value=payload) as mock_plan,
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/integration/scan/plan",
                json={"answers": [{"question_id": "q1", "answer": "yes"}]},
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["plan_id"] == "plan-1"
    assert body["provider"] == "ollama"
    assert body["recommended_chain"][0]["domain"] == "sales"
    assert body["status"] == "planned"
    mock_plan.assert_called_once()
    assert mock_plan.call_args.kwargs["answers"] == [{"question_id": "q1", "answer": "yes"}]
