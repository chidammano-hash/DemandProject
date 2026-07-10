"""Unit tests for the AI Integration Scan Orchestrator."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from common.ai.llm_client import LLMClientError


def _scan_payload() -> dict[str, object]:
    return {
        "scanned_at": "2026-07-10T10:00:00Z",
        "changes": [
            {
                "domain": "sales",
                "kind": "fact",
                "changed": True,
                "reason": "hash mismatch",
                "proposed_mode": "delta",
                "proposed_slice": None,
                "source_files": [{"path": "data/input/sales.csv", "changed": True}],
            }
        ],
        "proposed_chain": [{"step": 1, "domain": "sales", "mode": "delta", "slice": None}],
    }


def test_run_scan_planner_returns_model_decision():
    pool = MagicMock()
    mock_client = MagicMock()
    mock_client.chat.return_value = SimpleNamespace(
        parsed={
            "status": "planned",
            "confidence": 0.92,
            "explanation": "Sales can run immediately.",
            "risk_flags": ["none"],
            "questions": [],
            "recommended_chain": [{"step": 1, "domain": "sales", "mode": "delta", "slice": None}],
        }
    )

    with (
        patch("common.ai.integration_scan.planner.scan_input_dir", return_value=_scan_payload()),
        patch("common.ai.integration_scan.planner._fetch_job_context", return_value=[]),
        patch("common.ai.integration_scan.planner._fetch_batch_context", return_value=[]),
        patch("common.ai.integration_scan.planner.build_from_config", return_value=mock_client),
    ):
        from common.ai.integration_scan.planner import run_scan_planner

        out = run_scan_planner(pool)

    assert out["status"] == "planned"
    assert out["confidence"] == 0.92
    assert out["recommended_chain"][0]["domain"] == "sales"
    assert out["provider"] == "ollama"
    assert out["questions"] == []
    mock_client.chat.assert_called_once()


def test_run_scan_planner_falls_back_when_model_unavailable():
    pool = MagicMock()
    fallback_scan = _scan_payload()

    with (
        patch("common.ai.integration_scan.planner.scan_input_dir", return_value=fallback_scan),
        patch("common.ai.integration_scan.planner._fetch_job_context", return_value=[]),
        patch("common.ai.integration_scan.planner._fetch_batch_context", return_value=[]),
        patch("common.ai.integration_scan.planner.build_from_config", side_effect=LLMClientError("boom")),
    ):
        from common.ai.integration_scan.planner import run_scan_planner

        out = run_scan_planner(pool)

    assert out["status"] == "fallback"
    assert out["recommended_chain"] == fallback_scan["proposed_chain"]
    assert out["risk_flags"] == ["llm_unavailable"]
    assert out["questions"] == []
