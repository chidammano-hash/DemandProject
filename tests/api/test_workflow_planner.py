"""API contract tests for the Operations Workbench planner."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
from httpx import ASGITransport

from tests.api.conftest import make_pool


@pytest.mark.asyncio
async def test_workflow_plan_returns_ai_verified_recommendation():
    pool, _, _ = make_pool()
    payload = {
        "plan_id": "workflow-plan-1",
        "provider": "codex",
        "model": "gpt-5.5",
        "ai_verified": True,
        "status": "planned",
        "confidence": 0.94,
        "explanation": "Refresh source data before rebuilding models.",
        "risk_flags": [],
        "questions": [],
        "recommendations": [
            {
                "pipeline_name": "data-refresh",
                "title": "Data Refresh",
                "description": "Refresh changed inputs.",
                "priority": "critical",
                "reason": "Sales changed.",
                "blockers": [],
                "steps": [
                    {"position": 1, "job_type": "etl_pipeline", "params": {}, "label": None},
                ],
            }
        ],
        "evidence": {"changed_domains": ["sales"]},
        "scanned_at": "2026-07-10T10:00:00Z",
    }
    with (
        patch("api.core._get_pool", return_value=pool),
        patch("api.routers.core.jobs.run_workflow_planner", return_value=payload) as planner,
    ):
        from api.main import app

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/jobs/workflow-plan",
                json={"answers": [{"question_id": "active_job_handling", "answer": "wait"}]},
            )

    assert response.status_code == 200
    assert response.json()["recommendations"][0]["pipeline_name"] == "data-refresh"
    planner.assert_called_once_with(
        pool,
        answers=[{"question_id": "active_job_handling", "answer": "wait"}],
    )
