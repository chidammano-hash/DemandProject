"""AI planning layer for the Integration tab's Scan Now flow."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from common.ai.llm_client import LLMClientError, LLMJSONParseError, build_from_config
from common.ai.sku_chat.agent import CodexRuntimeError, _run_codex_exec
from common.ai.sku_chat.auth import SkuChatAuthError, resolve_auth_env
from common.core.utils import load_config
from common.services.integration_scanner import scan_input_dir

logger = logging.getLogger(__name__)

_CFG_NAME = "integration_scan_config"
_DEFAULT_THRESHOLD = 0.8
_DEFAULT_MAX_QUESTIONS = 3
_VALID_RUNTIMES = {"codex", "openai"}


class PlannerAnswer(BaseModel):
    """One answer submitted back to the planner."""

    question_id: str = Field(..., description="Stable question identifier from the prior planner turn.")
    answer: str = Field(..., description="User response text.")


class PlannerQuestion(BaseModel):
    """One question the planner wants the user to answer."""

    id: str = Field(..., description="Stable question identifier.")
    prompt: str = Field(..., description="Question text shown to the user.")
    answer_type: Literal["text", "choice", "boolean"] = Field(
        default="text",
        description="How the UI should collect the answer.",
    )
    options: list[str] = Field(default_factory=list, description="Choice options for select-style questions.")
    required: bool = Field(default=True, description="Whether the question must be answered.")
    reason: str | None = Field(default=None, description="Why the planner is asking.")


class PlannerEvidence(BaseModel):
    """One evidence line exposed to the UI."""

    kind: Literal["scan", "job", "batch"] = Field(..., description="Evidence source bucket.")
    label: str = Field(..., description="Short label for the evidence.")
    value: str = Field(..., description="Human-readable evidence value.")


class PlannerDecision(BaseModel):
    """Validated LLM output for a scan-planning turn."""

    status: Literal["questions", "planned"] = Field(..., description="Whether the planner needs more input.")
    confidence: float = Field(ge=0.0, le=1.0, description="Planner confidence in the returned sequence.")
    explanation: str = Field(default="", description="Short rationale for the recommendation.")
    risk_flags: list[str] = Field(default_factory=list, description="Ambiguities or cautions the UI should show.")
    questions: list[PlannerQuestion] = Field(default_factory=list, description="Optional clarifying questions.")
    recommended_chain: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered chain of jobs the UI may execute if the plan is final.",
    )


@dataclass(slots=True)
class PlannerRun:
    """Full planner response returned by the API layer."""

    plan_id: str
    provider: str
    model: str
    status: str
    confidence: float
    explanation: str
    risk_flags: list[str]
    questions: list[dict[str, Any]]
    recommended_chain: list[dict[str, Any]]
    evidence: list[dict[str, str]]
    scanned_at: str
    changes: list[dict[str, Any]]
    proposed_chain: list[dict[str, Any]]


def _load_cfg() -> dict[str, Any]:
    cfg = load_config(_CFG_NAME)
    if not cfg:
        cfg = {
            "runtime": {"provider": "codex"},
            "models": {"codex": "gpt-5.5", "openai": "gpt-5.5"},
            "cost_controls": {"per_call_timeout_seconds": 45},
        }
    return cfg


def _runtime_provider(cfg: dict[str, Any]) -> str:
    configured = str((cfg.get("runtime") or {}).get("provider", "codex"))
    provider = os.getenv("INTEGRATION_SCAN_AI_RUNTIME", configured).strip().lower()
    if provider not in _VALID_RUNTIMES:
        raise ValueError("Integration Scan AI runtime must be 'codex' or 'openai'.")
    return provider


def _planner_model(cfg: dict[str, Any], provider: str) -> str:
    model = str((cfg.get("models") or {}).get(provider, "")).strip()
    if not model:
        raise ValueError(f"Integration Scan AI has no model configured for {provider}.")
    return model


def _normalize_decision_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize harmless Codex status aliases before strict Pydantic validation."""
    normalized = dict(payload)
    status = str(normalized.get("status", "")).strip().lower()
    if status.startswith("no_change") or status in {"ready", "safe", "final", "complete"}:
        normalized["status"] = "planned"
    elif status.startswith("question") or status.startswith("need") or "clarif" in status:
        normalized["status"] = "questions"

    raw_questions = normalized.get("questions") or []
    normalized["questions"] = [
        {
            "id": f"planner_question_{index}",
            "prompt": question,
            "answer_type": "text",
            "options": [],
            "required": True,
            "reason": "The answer may change the safe execution sequence.",
        }
        if isinstance(question, str)
        else question
        for index, question in enumerate(raw_questions, start=1)
    ]
    normalized.setdefault("risk_flags", [])
    normalized.setdefault("recommended_chain", [])
    normalized.setdefault("explanation", "")
    return normalized


def _ground_recommended_chain(
    recommended_chain: list[dict[str, Any]], scan: dict[str, Any],
) -> list[dict[str, Any]]:
    """Keep AI-selected ordering, but take executable step details from the scan.

    The model can choose to omit a safe step or change the order. It must never
    invent a load mode, slice, or file: those values are determined by the
    scanner and enforced at the submission endpoint.
    """
    proposed = list(scan.get("proposed_chain") or [])
    by_domain = {
        str(step.get("domain")): step
        for step in proposed
        if isinstance(step, dict) and step.get("domain")
    }
    grounded: list[dict[str, Any]] = []
    included: set[str] = set()
    for recommendation in recommended_chain:
        if not isinstance(recommendation, dict):
            continue
        domain = str(recommendation.get("domain") or "")
        if domain not in by_domain or domain in included:
            continue
        included.add(domain)
        source = by_domain[domain]
        grounded.append({
            "step": len(grounded) + 1,
            "domain": domain,
            "mode": source.get("mode"),
            "slice": source.get("slice"),
            "file": source.get("file"),
        })
    return grounded


def _row_dicts(cur) -> list[dict[str, Any]]:
    cols = [d[0] for d in (cur.description or [])]
    return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


def _fetch_job_context(pool) -> list[dict[str, Any]]:
    sql = (
        "SELECT job_id, job_type, job_label, status, params, submitted_at, started_at, "
        "completed_at, pipeline_id, pipeline_step, triggered_by "
        "FROM job_history "
        "WHERE job_type = 'load_domain' AND status IN ('queued', 'running') "
        "ORDER BY submitted_at DESC LIMIT 12"
    )
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql)
        return _row_dicts(cur)


def _fetch_batch_context(pool, domains: list[str]) -> list[dict[str, Any]]:
    if not domains:
        return []
    sql = (
        "SELECT domain, source_file, source_hash, row_count_in, row_count_out, "
        "status, started_at, completed_at "
        "FROM audit_load_batch "
        "WHERE status = 'completed' AND domain = ANY(%s) "
        "ORDER BY completed_at DESC NULLS LAST, started_at DESC LIMIT 24"
    )
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (domains,))
        return _row_dicts(cur)


def _scan_summary(scan: dict[str, Any]) -> dict[str, Any]:
    changes = scan.get("changes") or []
    proposed_chain = scan.get("proposed_chain") or []
    changed = [c for c in changes if c.get("changed")]
    changed_domains = [str(c.get("domain")) for c in changed]
    inventory_changes = [c for c in changed if c.get("domain") == "inventory"]
    return {
        "changed_domains": changed_domains,
        "changed_count": len(changed),
        "inventory_changed_count": len(inventory_changes),
        "proposed_steps": len(proposed_chain),
        "inventory_reasons": [c.get("reason") for c in inventory_changes if c.get("reason")],
    }


def _heuristic_questions(
    scan_summary: dict[str, Any],
    jobs: list[dict[str, Any]],
) -> list[PlannerQuestion]:
    questions: list[PlannerQuestion] = []
    changed_domains = scan_summary["changed_domains"]
    inventory_changed_count = scan_summary["inventory_changed_count"]
    active_domains = [str(j.get("domain") or "") for j in jobs if j.get("domain")]

    if inventory_changed_count > 1:
        questions.append(PlannerQuestion(
            id="inventory_slice_scope",
            prompt="I found multiple inventory snapshots changed. Do you want the newest changed slice only, or should I replay every changed slice?",
            answer_type="choice",
            options=["Newest changed slice only", "Replay every changed slice"],
            reason="More than one inventory snapshot changed.",
        ))

    overlapping = sorted({d for d in changed_domains if d in active_domains})
    if overlapping:
        questions.append(PlannerQuestion(
            id="active_job_overlap",
            prompt=f"There is already an active job for {', '.join(overlapping)}. Should I wait, queue behind it, or stop and inspect first?",
            answer_type="choice",
            options=["Wait for active jobs", "Queue behind them", "Stop and inspect"],
            reason="A changed domain is already in flight.",
        ))

    if scan_summary["changed_count"] >= 4 and not questions:
        questions.append(PlannerQuestion(
            id="broad_change_scope",
            prompt="Several domains changed at once. Should I keep the conservative full chain, or only run the changed domains?",
            answer_type="choice",
            options=["Conservative full chain", "Only changed domains"],
            reason="A broad change set usually deserves a quick confirmation.",
        ))

    return questions[:_DEFAULT_MAX_QUESTIONS]


def _evidence_for_ui(
    scan: dict[str, Any],
    jobs: list[dict[str, Any]],
    batches: list[dict[str, Any]],
) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    for change in (scan.get("changes") or []):
        if not change.get("changed"):
            continue
        evidence.append({
            "kind": "scan",
            "label": str(change.get("domain") or "domain"),
            "value": str(change.get("reason") or "changed"),
        })
    for job in jobs[:6]:
        evidence.append({
            "kind": "job",
            "label": f"{job.get('domain') or 'job'} / {job.get('status') or 'queued'}",
            "value": f"pipeline {job.get('pipeline_id') or '-'} step {job.get('pipeline_step') or '-'}",
        })
    for batch in batches[:6]:
        evidence.append({
            "kind": "batch",
            "label": str(batch.get("domain") or "batch"),
            "value": f"{batch.get('source_file') or 'n/a'} · {batch.get('completed_at') or batch.get('started_at') or 'no time'}",
        })
    return evidence


def _build_prompt(
    scan: dict[str, Any],
    jobs: list[dict[str, Any]],
    batches: list[dict[str, Any]],
    answers: list[PlannerAnswer],
    cfg: dict[str, Any],
) -> list[dict[str, str]]:
    threshold = float(cfg.get("confidence_threshold", _DEFAULT_THRESHOLD))
    prompt = {
        "planning_rules": {
            "confidence_threshold": threshold,
            "max_questions": int(cfg.get("max_questions_per_scan", _DEFAULT_MAX_QUESTIONS)),
            "must_ask_when_active_jobs_overlap": True,
            "must_ask_when_inventory_changes_span_multiple_snapshots": True,
            "prefer_dimension_first_then_fact_order": True,
            "never_execute_writes": True,
        },
        "scan": scan,
        "active_jobs": jobs,
        "recent_batches": batches,
        "answers": [a.model_dump() for a in answers],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are the Integration Scan Orchestrator. "
                "Inspect the scan snapshot, live job state, and batch history. "
                "Return JSON only with keys: status, confidence, explanation, risk_flags, questions, recommended_chain. "
                "If certainty is low or an answer would change the safe sequence, ask concise questions first. "
                "When finalizing a plan, keep the existing scan order unless evidence clearly supports a safer order. "
                "Use only domains, modes, slices, and files supplied in scan.proposed_chain. "
                "Do not write to the database."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(prompt, default=str, indent=2),
        },
    ]


def _fallback_decision(
    scan: dict[str, Any],
    jobs: list[dict[str, Any]],
    questions: list[PlannerQuestion],
    *,
    threshold: float,
) -> PlannerDecision:
    status = "questions" if questions else "planned"
    explanation = (
        "Used deterministic scan order because the local model was unavailable."
        if status == "planned"
        else "More context is needed before choosing a safe execution sequence."
    )
    return PlannerDecision(
        status=status,
        confidence=threshold if status == "planned" else max(0.5, threshold - 0.2),
        explanation=explanation,
        risk_flags=["llm_unavailable"],
        questions=questions,
        recommended_chain=list(scan.get("proposed_chain") or []),
    )


def _build_decision(
    scan: dict[str, Any],
    jobs: list[dict[str, Any]],
    batches: list[dict[str, Any]],
    answers: list[PlannerAnswer],
    cfg: dict[str, Any],
) -> PlannerDecision:
    messages = _build_prompt(scan, jobs, batches, answers, cfg)
    provider = _runtime_provider(cfg)
    if provider == "codex":
        model = _planner_model(cfg, provider)
        prompt = "\n\n".join(
            f"{message['role'].upper()}:\n{message['content']}" for message in messages
        )
        timeout = float((cfg.get("cost_controls") or {}).get("per_call_timeout_seconds", 90))
        answer = asyncio.run(
            _run_codex_exec(
                prompt,
                model_id=model,
                cfg=cfg,
                env=resolve_auth_env(cfg, provider="codex"),
                timeout_s=timeout,
            )
        )
        cleaned = answer.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```")
            cleaned = cleaned.removesuffix("```").strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMJSONParseError("Could not parse JSON from codex") from exc
        decision = PlannerDecision.model_validate(_normalize_decision_payload(parsed))
        decision.recommended_chain = _ground_recommended_chain(decision.recommended_chain, scan)
        return decision

    openai_cfg = {**cfg, "provider": "openai"}
    client = build_from_config(openai_cfg)
    response = client.chat(
        messages,
        json_mode=True,
        temperature=float(cfg.get("temperature", 0.0)),
        max_tokens=int(cfg.get("max_tokens", 1024)),
    )
    parsed = response.parsed or {}
    return PlannerDecision.model_validate(parsed)


def run_scan_planner(pool, *, answers: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run the deterministic scan and return an AI-reviewed execution plan."""
    cfg = _load_cfg()
    threshold = float(cfg.get("confidence_threshold", _DEFAULT_THRESHOLD))
    scan = scan_input_dir(pool) or {}
    job_rows = _fetch_job_context(pool)
    changed_domains = [str(c.get("domain")) for c in scan.get("changes", []) if c.get("changed")]
    batch_rows = _fetch_batch_context(pool, changed_domains)
    scan_summary = _scan_summary(scan)
    heuristic_questions = _heuristic_questions(scan_summary, job_rows)
    evidence = _evidence_for_ui(scan, job_rows, batch_rows)

    answer_models = [PlannerAnswer.model_validate(a) for a in (answers or [])]
    plan_id = str(uuid.uuid4())
    used_fallback = False

    try:
        decision = _build_decision(scan, job_rows, batch_rows, answer_models, cfg)
        if decision.status == "planned" and decision.confidence < threshold and heuristic_questions and not answer_models:
            decision = PlannerDecision(
                status="questions",
                confidence=decision.confidence,
                explanation=decision.explanation,
                risk_flags=sorted({*decision.risk_flags, "confidence_below_threshold"}),
                questions=decision.questions or heuristic_questions,
                recommended_chain=decision.recommended_chain or list(scan.get("proposed_chain") or []),
            )
        elif decision.status == "questions" and not decision.questions:
            decision = PlannerDecision(
                status="questions",
                confidence=decision.confidence,
                explanation=decision.explanation,
                risk_flags=sorted({*decision.risk_flags, "planner_returned_no_questions"}),
                questions=heuristic_questions,
                recommended_chain=decision.recommended_chain or list(scan.get("proposed_chain") or []),
            )
    except (
        CodexRuntimeError,
        LLMClientError,
        LLMJSONParseError,
        SkuChatAuthError,
        ValidationError,
        ValueError,
    ) as exc:
        logger.warning("integration scan planner fell back to deterministic plan: %s", exc)
        decision = _fallback_decision(scan, job_rows, heuristic_questions, threshold=threshold)
        used_fallback = True

    response_status = decision.status
    if used_fallback and not decision.questions:
        response_status = "fallback"

    provider = _runtime_provider(cfg)
    result = PlannerRun(
        plan_id=plan_id,
        provider=provider,
        model=_planner_model(cfg, provider),
        status=response_status,
        confidence=decision.confidence,
        explanation=decision.explanation,
        risk_flags=decision.risk_flags,
        questions=[q.model_dump() for q in decision.questions],
        recommended_chain=decision.recommended_chain or list(scan.get("proposed_chain") or []),
        evidence=evidence,
        scanned_at=str(scan.get("scanned_at", "")),
        changes=list(scan.get("changes") or []),
        proposed_chain=list(scan.get("proposed_chain") or []),
    )
    return {
        "plan_id": result.plan_id,
        "provider": result.provider,
        "model": result.model,
        "status": result.status,
        "confidence": result.confidence,
        "explanation": result.explanation,
        "risk_flags": result.risk_flags,
        "questions": result.questions,
        "recommended_chain": result.recommended_chain,
        "evidence": result.evidence,
        "scanned_at": result.scanned_at,
        "changes": result.changes,
        "proposed_chain": result.proposed_chain,
    }
