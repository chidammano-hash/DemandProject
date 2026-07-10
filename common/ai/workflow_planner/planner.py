"""System-first, AI-verified planning across operational workflows."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from common.ai.llm_client import (
    LLMClientError,
    LLMJSONParseError,
    build_from_config,
)
from common.ai.sku_chat.agent import CodexRuntimeError, _run_codex_exec
from common.ai.sku_chat.auth import SkuChatAuthError, resolve_auth_env
from common.core.planning_date import get_planning_date
from common.core.utils import load_config
from common.services.integration_scanner import scan_input_dir
from common.services.pipeline_presets import load_pipeline_presets, preset_steps

logger = logging.getLogger(__name__)

_CFG_NAME = "integration_scan_config"
_DEFAULT_CONFIDENCE = 0.8
_VALID_RUNTIMES = {"codex", "openai"}


@dataclass(frozen=True, slots=True)
class WorkflowState:
    """Deterministic operational evidence used to select safe workflows."""

    planning_month: date
    active_jobs: list[dict[str, Any]]
    clustered_skus: int
    latest_feature_refresh: datetime | None
    latest_cluster_promotion: datetime | None
    stale_tuning_profiles: int
    latest_sales_load: datetime | None
    latest_champion_promotion: datetime | None
    latest_inventory_refresh: datetime | None
    active_production_month: date | None
    active_production_promoted_at: datetime | None
    planning_month_production_rows: int
    planning_month_roster_models: int
    planning_month_snapshot_rows: int


@dataclass(frozen=True, slots=True)
class SystemRecommendation:
    """One pipeline selected by deterministic readiness rules."""

    pipeline_name: str
    priority: Literal["critical", "high", "medium", "low"]
    reason: str
    blockers: tuple[str, ...] = ()


class WorkflowAnswer(BaseModel):
    question_id: str
    answer: str


class WorkflowQuestion(BaseModel):
    id: str
    prompt: str
    answer_type: Literal["text", "choice", "boolean"] = "choice"
    options: list[str] = Field(default_factory=list)
    required: bool = True
    reason: str | None = None


class WorkflowDecision(BaseModel):
    status: Literal["questions", "planned"]
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    risk_flags: list[str] = Field(default_factory=list)
    questions: list[WorkflowQuestion] = Field(default_factory=list)
    recommended_pipeline_names: list[str] = Field(default_factory=list)


def _planning_month() -> date:
    planning_date = get_planning_date()
    return planning_date.replace(day=1)


def _runtime_provider(cfg: dict[str, Any]) -> str:
    configured = str((cfg.get("runtime") or {}).get("provider", "codex"))
    provider = os.getenv("INTEGRATION_SCAN_AI_RUNTIME", configured).strip().lower()
    if provider not in _VALID_RUNTIMES:
        raise ValueError("Workflow planner runtime must be 'codex' or 'openai'.")
    return provider


def _planner_model(cfg: dict[str, Any], provider: str) -> str:
    model = str((cfg.get("models") or {}).get(provider, "")).strip()
    if not model:
        raise ValueError(f"Workflow planner has no model configured for {provider}.")
    return model


def _verification_failure(exc: Exception) -> tuple[str, str]:
    message = str(exc).lower()
    if "usage limit" in message or "rate limit" in message:
        return (
            "ai_usage_limit",
            "AI verification reached its current usage limit; the system-safe plan is still available.",
        )
    if "auth" in message or "api key" in message or "sign-in" in message:
        return (
            "ai_auth_required",
            "AI verification needs valid runtime authentication; the system-safe plan is still available.",
        )
    if "timed out" in message or "timeout" in message:
        return (
            "ai_timeout",
            "AI verification timed out; the system-safe plan is still available.",
        )
    return (
        "ai_verification_unavailable",
        "AI verification is unavailable; the system-safe plan is still available.",
    )


def _fetch_state(pool: Any) -> WorkflowState:
    planning_month = _planning_month()
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT job_id, job_type, status, pipeline_id, submitted_at
               FROM job_history
               WHERE status IN ('queued', 'running')
               ORDER BY submitted_at DESC
               LIMIT 20"""
        )
        active_jobs = [
            {
                "job_id": row[0],
                "job_type": row[1],
                "status": row[2],
                "pipeline_id": row[3],
                "submitted_at": row[4],
            }
            for row in cur.fetchall()
        ]
        cur.execute(
            """SELECT
                 (SELECT COUNT(*) FROM current_sku_cluster_assignment
                  WHERE ml_cluster IS NOT NULL),
                 (SELECT MAX(completed_at) FROM job_history
                  WHERE status = 'completed' AND job_type = 'compute_sku_features'),
                 (SELECT MAX(promoted_at) FROM cluster_experiment
                  WHERE is_promoted = TRUE),
                 (SELECT COUNT(*) FROM cluster_tuning_profile_state WHERE stale = TRUE),
                 (SELECT MAX(completed_at) FROM audit_load_batch
                  WHERE domain = 'sales' AND status = 'completed'),
                 (SELECT MAX(promoted_at) FROM champion_experiment
                  WHERE is_promoted = TRUE),
                 (SELECT MAX(completed_at) FROM job_history
                  WHERE status = 'completed'
                    AND job_type IN (
                      'inventory_planning_pipeline', 'compute_safety_stock',
                      'compute_eoq', 'compute_replenishment_plan'
                    )),
                 (SELECT CASE
                           WHEN plan_version ~ '^[0-9]{4}-[0-9]{2}$'
                           THEN TO_DATE(plan_version, 'YYYY-MM')
                         END
                 FROM model_promotion_log
                  WHERE is_active = TRUE ORDER BY promoted_at DESC, id DESC LIMIT 1),
                 (SELECT promoted_at FROM model_promotion_log
                  WHERE is_active = TRUE ORDER BY promoted_at DESC, id DESC LIMIT 1),
                 (SELECT COUNT(*) FROM fact_production_forecast
                  WHERE model_id = 'champion'
                    AND plan_version = TO_CHAR(%s::date, 'YYYY-MM')),
                 (SELECT COUNT(DISTINCT model_id) FROM forecast_snapshot_roster
                  WHERE record_month = %s),
                 (SELECT COUNT(*) FROM fact_forecast_snapshot
                  WHERE record_month = %s)""",
            (planning_month, planning_month, planning_month),
        )
        row = cur.fetchone() or (
            0,
            None,
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            0,
            0,
            0,
        )
    return WorkflowState(
        planning_month=planning_month,
        active_jobs=active_jobs,
        clustered_skus=int(row[0] or 0),
        latest_feature_refresh=row[1],
        latest_cluster_promotion=row[2],
        stale_tuning_profiles=int(row[3] or 0),
        latest_sales_load=row[4],
        latest_champion_promotion=row[5],
        latest_inventory_refresh=row[6],
        active_production_month=row[7],
        active_production_promoted_at=row[8],
        planning_month_production_rows=int(row[9] or 0),
        planning_month_roster_models=int(row[10] or 0),
        planning_month_snapshot_rows=int(row[11] or 0),
    )


def system_recommendations(
    scan: dict[str, Any],
    state: WorkflowState,
) -> list[SystemRecommendation]:
    """Derive an ordered remediation plan without relying on an LLM."""
    recommendations: list[SystemRecommendation] = []
    changed_domains = [
        str(change.get("domain")) for change in scan.get("changes", []) if change.get("changed")
    ]
    if changed_domains:
        recommendations.append(
            SystemRecommendation(
                "data-refresh",
                "critical",
                f"Changed source data detected for {', '.join(changed_domains)}.",
            )
        )

    clusters_are_stale = bool(
        state.latest_feature_refresh
        and (
            state.latest_cluster_promotion is None
            or state.latest_feature_refresh > state.latest_cluster_promotion
        )
    )
    if state.clustered_skus == 0 or clusters_are_stale:
        reason = (
            "No promoted SKU cluster assignments are available."
            if state.clustered_skus == 0
            else "SKU features are newer than the promoted cluster assignments."
        )
        recommendations.append(
            SystemRecommendation(
                "clustering-refresh",
                "high",
                reason,
                ("Complete data refresh first.",) if changed_domains else (),
            )
        )
        return recommendations

    champion_is_stale = bool(
        state.latest_sales_load
        and (
            state.latest_champion_promotion is None
            or state.latest_sales_load > state.latest_champion_promotion
        )
    )
    if state.stale_tuning_profiles or champion_is_stale:
        reasons = []
        if state.stale_tuning_profiles:
            reasons.append(f"{state.stale_tuning_profiles} tuning profile(s) are stale")
        if champion_is_stale:
            reasons.append("sales data is newer than the promoted champion")
        recommendations.append(
            SystemRecommendation(
                "model-refresh",
                "high",
                "; ".join(reasons).capitalize() + ".",
                ("Complete data refresh first.",) if changed_domains else (),
            )
        )

    production_is_current = (
        state.active_production_month == state.planning_month
        and state.planning_month_production_rows > 0
    )
    if not production_is_current:
        recommendations.append(
            SystemRecommendation(
                "forecast-publish",
                "high",
                f"No promoted champion production release exists for {state.planning_month:%Y-%m}.",
                ("Refresh models first.",) if champion_is_stale else (),
            )
        )
    elif state.planning_month_roster_models == 4 and state.planning_month_snapshot_rows == 0:
        recommendations.append(
            SystemRecommendation(
                "forecast-snapshot-bundle",
                "medium",
                "The champion-plus-three roster is ready but has not been archived.",
            )
        )

    inventory_is_stale = bool(
        production_is_current
        and state.active_production_promoted_at
        and (
            state.latest_inventory_refresh is None
            or state.active_production_promoted_at > state.latest_inventory_refresh
        )
    )
    if inventory_is_stale:
        recommendations.append(
            SystemRecommendation(
                "inventory-refresh",
                "medium",
                "Inventory planning outputs predate the current champion forecast.",
            )
        )

    return recommendations


def _heuristic_questions(state: WorkflowState) -> list[WorkflowQuestion]:
    if not state.active_jobs:
        return []
    return [
        WorkflowQuestion(
            id="active_job_handling",
            prompt=(
                f"{len(state.active_jobs)} workflow job(s) are already queued or running. "
                "Should the recommendation wait or be queued after them?"
            ),
            options=["Wait for active jobs", "Queue after active jobs", "Inspect active jobs"],
            reason="Concurrent workflow runs can duplicate expensive work or race promotions.",
        )
    ]


def _apply_answer_guardrails(
    recommendations: list[SystemRecommendation],
    answers: list[WorkflowAnswer],
) -> list[SystemRecommendation]:
    active_job_answer = next(
        (
            answer.answer.strip().lower()
            for answer in answers
            if answer.question_id == "active_job_handling"
        ),
        "",
    )
    if not active_job_answer or "queue after" in active_job_answer:
        return recommendations
    blocker = (
        "Inspect active jobs in Workflow Library before execution."
        if "inspect" in active_job_answer
        else "Wait for active jobs to finish, then analyze workflows again."
    )
    return [replace(item, blockers=(*item.blockers, blocker)) for item in recommendations]


def _normalize_decision(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    status = str(normalized.get("status", "")).lower()
    normalized["status"] = "questions" if "question" in status or "clarif" in status else "planned"
    raw_questions = normalized.get("questions") or []
    normalized["questions"] = [
        {
            "id": f"workflow_question_{index}",
            "prompt": question,
            "answer_type": "text",
            "options": [],
            "required": True,
            "reason": "The answer may change the safe workflow sequence.",
        }
        if isinstance(question, str)
        else question
        for index, question in enumerate(raw_questions, start=1)
    ]
    normalized.setdefault("risk_flags", [])
    normalized.setdefault("recommended_pipeline_names", [])
    normalized.setdefault("explanation", "")
    return normalized


def _ai_decision(
    scan: dict[str, Any],
    state: WorkflowState,
    system_items: list[SystemRecommendation],
    answers: list[WorkflowAnswer],
) -> WorkflowDecision:
    cfg = load_config(_CFG_NAME) or {}
    provider = _runtime_provider(cfg)
    model = _planner_model(cfg, provider)
    allowed = [item.pipeline_name for item in system_items]
    prompt = (
        "You are the Operations Workflow Verifier. Review deterministic system evidence and "
        "return JSON only with status, confidence, explanation, risk_flags, questions, and "
        "recommended_pipeline_names. Preserve the supplied system sequence; names must "
        f"come only from this allowed list: {allowed}. Never invent jobs or parameters. Ask a "
        "question only when its answer changes safety or ordering.\n\n"
        + json.dumps(
            {
                "scan": scan,
                "workflow_state": asdict(state),
                "system_recommendations": [asdict(item) for item in system_items],
                "answers": [answer.model_dump() for answer in answers],
            },
            default=str,
            indent=2,
        )
    )
    if provider == "codex":
        timeout = float((cfg.get("cost_controls") or {}).get("per_call_timeout_seconds", 120))
        raw = asyncio.run(
            _run_codex_exec(
                prompt,
                model_id=model,
                cfg=cfg,
                env=resolve_auth_env(cfg, provider="codex"),
                timeout_s=timeout,
            )
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```")
            cleaned = cleaned.removesuffix("```").strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMJSONParseError("Could not parse workflow planner JSON") from exc
    else:
        client = build_from_config({**cfg, "provider": "openai"})
        response = client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You verify operational workflow plans. Return only the requested "
                        "JSON and never invent executable jobs or parameters."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            json_mode=True,
            temperature=float(cfg.get("temperature", 0.0)),
            max_tokens=int(cfg.get("max_tokens", 1200)),
        )
        payload = response.parsed or {}
    decision = WorkflowDecision.model_validate(_normalize_decision(payload))
    decision.recommended_pipeline_names = [
        name for name in decision.recommended_pipeline_names if name in allowed
    ]
    if decision.status == "questions" and not decision.questions:
        decision.status = "planned"
        decision.risk_flags.append("ai_question_missing")
    return decision


def _recommendation_payload(
    item: SystemRecommendation,
    presets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    preset = presets[item.pipeline_name]
    steps = preset_steps(preset)
    return {
        "pipeline_name": item.pipeline_name,
        "title": item.pipeline_name.replace("-", " ").title(),
        "description": str(preset.get("description") or ""),
        "priority": item.priority,
        "reason": item.reason,
        "blockers": list(item.blockers),
        "steps": [
            {
                "position": index,
                "job_type": step["job_type"],
                "params": step["params"],
                "label": step["label"],
            }
            for index, step in enumerate(steps, start=1)
        ],
    }


def run_workflow_planner(
    pool: Any,
    *,
    answers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Scan operational state and return an AI-verified, executable plan."""
    scan = scan_input_dir(pool) or {}
    state = _fetch_state(pool)
    presets = load_pipeline_presets()
    system_items = [
        item for item in system_recommendations(scan, state) if item.pipeline_name in presets
    ]
    answer_models = [WorkflowAnswer.model_validate(answer) for answer in (answers or [])]
    system_items = _apply_answer_guardrails(system_items, answer_models)
    heuristic_questions = _heuristic_questions(state) if not answer_models else []
    cfg = load_config(_CFG_NAME) or {}
    provider = _runtime_provider(cfg)
    model = _planner_model(cfg, provider)
    ai_verified = False
    try:
        decision = _ai_decision(scan, state, system_items, answer_models)
        ai_verified = True
    except (
        CodexRuntimeError,
        LLMClientError,
        LLMJSONParseError,
        SkuChatAuthError,
        ValidationError,
        ValueError,
    ) as exc:
        logger.warning("workflow planner used deterministic fallback: %s", exc)
        risk_flag, explanation = _verification_failure(exc)
        decision = WorkflowDecision(
            status="questions" if heuristic_questions else "planned",
            confidence=_DEFAULT_CONFIDENCE,
            explanation=(
                explanation
                if system_items
                else "System readiness checks found no workflow that needs to run."
            ),
            risk_flags=[risk_flag],
            questions=heuristic_questions,
            recommended_pipeline_names=[item.pipeline_name for item in system_items],
        )

    if heuristic_questions and not answer_models:
        decision.status = "questions"
        decision.questions = decision.questions or heuristic_questions

    # Execution order is a safety property derived from deterministic dependencies.
    # AI verifies and explains the plan but cannot reorder or silently drop a step.
    ordered_names = [item.pipeline_name for item in system_items]
    item_by_name = {item.pipeline_name: item for item in system_items}
    recommendations = [
        _recommendation_payload(item_by_name[name], presets)
        for name in ordered_names
        if name in item_by_name
    ]
    changed_domains = [
        str(change.get("domain")) for change in scan.get("changes", []) if change.get("changed")
    ]
    return {
        "plan_id": str(uuid.uuid4()),
        "provider": provider,
        "model": model,
        "ai_verified": ai_verified,
        "status": decision.status,
        "confidence": decision.confidence,
        "explanation": decision.explanation,
        "risk_flags": decision.risk_flags,
        "questions": [question.model_dump() for question in decision.questions],
        "recommendations": recommendations,
        "evidence": {
            "planning_month": state.planning_month.isoformat(),
            "changed_domains": changed_domains,
            "active_job_count": len(state.active_jobs),
            "clustered_skus": state.clustered_skus,
            "latest_feature_refresh": (
                state.latest_feature_refresh.isoformat() if state.latest_feature_refresh else None
            ),
            "latest_cluster_promotion": (
                state.latest_cluster_promotion.isoformat()
                if state.latest_cluster_promotion
                else None
            ),
            "stale_tuning_profiles": state.stale_tuning_profiles,
            "active_production_month": (
                state.active_production_month.isoformat() if state.active_production_month else None
            ),
            "planning_month_production_rows": state.planning_month_production_rows,
            "planning_month_roster_models": state.planning_month_roster_models,
            "planning_month_snapshot_rows": state.planning_month_snapshot_rows,
        },
        "scanned_at": str(scan.get("scanned_at", "")),
    }
