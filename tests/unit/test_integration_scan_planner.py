"""Unit tests for the AI Integration Scan Orchestrator."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from common.ai.sku_chat.agent import CodexRuntimeError


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
    codex_answer = json.dumps(
        {
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
        patch(
            "common.ai.integration_scan.planner._run_codex_exec",
            new=AsyncMock(return_value=codex_answer),
        ) as codex_exec,
    ):
        from common.ai.integration_scan.planner import run_scan_planner

        out = run_scan_planner(pool)

    assert out["status"] == "planned"
    assert out["confidence"] == 0.92
    assert out["recommended_chain"][0]["domain"] == "sales"
    assert out["provider"] == "codex"
    assert out["model"] == "gpt-5.5"
    assert out["questions"] == []
    codex_exec.assert_awaited_once()


def test_run_scan_planner_falls_back_when_model_unavailable():
    pool = MagicMock()
    fallback_scan = _scan_payload()

    with (
        patch("common.ai.integration_scan.planner.scan_input_dir", return_value=fallback_scan),
        patch("common.ai.integration_scan.planner._fetch_job_context", return_value=[]),
        patch("common.ai.integration_scan.planner._fetch_batch_context", return_value=[]),
        patch(
            "common.ai.integration_scan.planner._run_codex_exec",
            new=AsyncMock(side_effect=CodexRuntimeError("boom")),
        ),
    ):
        from common.ai.integration_scan.planner import run_scan_planner

        out = run_scan_planner(pool)

    assert out["status"] == "fallback"
    assert out["recommended_chain"] == fallback_scan["proposed_chain"]
    assert out["risk_flags"] == ["llm_unavailable"]
    assert out["questions"] == []


def test_integration_scan_runtime_can_switch_to_openai_for_production(monkeypatch):
    monkeypatch.setenv("INTEGRATION_SCAN_AI_RUNTIME", "openai")
    from common.ai.integration_scan.planner import _runtime_provider

    assert _runtime_provider({"runtime": {"provider": "codex"}}) == "openai"


def test_codex_decision_normalizes_safe_terminal_status_alias():
    from common.ai.integration_scan.planner import _normalize_decision_payload

    payload = _normalize_decision_payload(
        {
            "status": "no_changes_detected",
            "confidence": 0.99,
            "explanation": "No inputs changed.",
        }
    )

    assert payload["status"] == "planned"
    assert payload["risk_flags"] == []
    assert payload["questions"] == []
    assert payload["recommended_chain"] == []
