"""Bounded evidence loop and same-turn citation validation for the Copilot."""

from __future__ import annotations

from dataclasses import dataclass

from common.ai.copilot.contracts import (
    ConversationTurn,
    CopilotAnswer,
    CopilotContext,
    CopilotProvider,
    EvidenceBundle,
    ProviderRequest,
    ToolCall,
    ToolResult,
)
from common.ai.copilot.tools import ToolRegistry


@dataclass(frozen=True, slots=True)
class RuntimeLimits:
    max_turns: int = 6
    max_prompt_chars: int = 4_000
    max_tool_calls: int = 8
    max_conversation_chars: int = 4_000


@dataclass(frozen=True, slots=True)
class CopilotRunResult:
    answer: CopilotAnswer
    evidence: tuple[EvidenceBundle, ...]


class CopilotRuntime:
    def __init__(
        self,
        *,
        provider: CopilotProvider,
        tools: ToolRegistry,
        limits: RuntimeLimits | None = None,
        prefetch_context_evidence: bool = False,
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._limits = limits or RuntimeLimits()
        self._prefetch = prefetch_context_evidence

    async def run_grounded(
        self,
        *,
        prompt: str,
        context: CopilotContext,
        conversation: tuple[ConversationTurn, ...] = (),
    ) -> CopilotRunResult:
        normalized = prompt.strip()
        if not normalized or len(normalized) > self._limits.max_prompt_chars:
            raise ValueError("prompt is empty or exceeds the configured limit")
        results: list[ToolResult] = []
        evidence_rows: list[EvidenceBundle] = []
        evidence_ids: set[str] = set()
        calls = 0

        if self._prefetch:
            call = self._anchored_call(context)
            evidence = await self._execute(call, context)
            results.append(ToolResult(call_id=call.call_id, evidence=evidence))
            evidence_rows.append(evidence)
            evidence_ids.add(evidence.evidence_id)
            calls = 1

        for _ in range(self._limits.max_turns):
            turn = await self._provider.complete(
                ProviderRequest(
                    prompt=normalized,
                    context=context,
                    tool_names=self._tools.names,
                    prior_results=tuple(results),
                    conversation=conversation,
                )
            )
            if turn.answer is not None:
                self._validate_answer(turn.answer, tuple(row.evidence_id for row in evidence_rows))
                return CopilotRunResult(turn.answer, tuple(evidence_rows))
            for call in turn.tool_calls:
                calls += 1
                if calls > self._limits.max_tool_calls:
                    raise RuntimeError("Copilot tool-call limit reached")
                evidence = await self._execute(call, context)
                if evidence.evidence_id in evidence_ids:
                    raise ValueError("evidence IDs must be unique within a turn")
                evidence_ids.add(evidence.evidence_id)
                evidence_rows.append(evidence)
                results.append(ToolResult(call_id=call.call_id, evidence=evidence))
        raise RuntimeError("Copilot turn limit reached")

    async def _execute(self, call: ToolCall, context: CopilotContext) -> EvidenceBundle:
        self._validate_call_context(call, context)
        evidence = await self._tools.execute(call.name, call.arguments)
        if evidence.tool_name != call.name:
            raise ValueError("evidence provenance does not match the tool call")
        if evidence.content_hash != evidence.expected_content_hash():
            raise ValueError("evidence content integrity check failed")
        self._validate_evidence_context(evidence, context)
        return evidence

    @staticmethod
    def _validate_call_context(call: ToolCall, context: CopilotContext) -> None:
        dfu_tools = {
            "get_dfu_evidence",
            "get_forecast_evidence",
            "get_customer_evidence",
        }
        if call.name in dfu_tools and context.item_id is not None:
            expected = (context.item_id, context.customer_group, context.loc)
            actual = (
                call.arguments.get("item_id"),
                call.arguments.get("customer_group"),
                call.arguments.get("loc"),
            )
            if actual != expected:
                raise ValueError("tool call is outside the anchored DFU")
        if call.name == "get_inventory_plan_evidence" and context.item_id is not None:
            if (call.arguments.get("item_id"), call.arguments.get("loc")) != (
                context.item_id,
                context.loc,
            ):
                raise ValueError("tool call is outside the anchored DFU")
        if call.name == "get_inventory_opportunity_evidence" and context.opportunity_id:
            if call.arguments.get("opportunity_id") != context.opportunity_id:
                raise ValueError("tool call is outside the anchored opportunity")

    @staticmethod
    def _validate_evidence_context(evidence: EvidenceBundle, context: CopilotContext) -> None:
        pairs = (
            (evidence.item_id, context.item_id),
            (evidence.loc, context.loc),
            (evidence.opportunity_id, context.opportunity_id),
            (evidence.promotion_id, context.promotion_id),
            (evidence.production_run_id, context.production_run_id),
            (evidence.inventory_run_id, context.inventory_run_id),
        )
        if any(
            expected is not None and actual not in {None, expected} for actual, expected in pairs
        ):
            raise ValueError("current evidence no longer matches the session context")

    @staticmethod
    def _anchored_call(context: CopilotContext) -> ToolCall:
        if context.opportunity_id:
            return ToolCall(
                "server-context-evidence",
                "get_inventory_opportunity_evidence",
                {"opportunity_id": context.opportunity_id},
            )
        if context.workflow_run_id:
            return ToolCall(
                "server-context-evidence",
                "get_workflow_run_evidence",
                {"workflow_run_id": context.workflow_run_id},
            )
        if context.item_id and context.loc:
            return ToolCall(
                "server-context-evidence",
                "get_dfu_evidence",
                {
                    "item_id": context.item_id,
                    "customer_group": context.customer_group,
                    "loc": context.loc,
                },
            )
        return ToolCall("server-context-evidence", "get_portfolio_evidence", {"limit": 10})

    @staticmethod
    def _validate_answer(answer: CopilotAnswer, evidence_order: tuple[str, ...]) -> None:
        if answer.action_request is not None:
            raise ValueError("general Copilot answers cannot request actions")
        if not answer.answer.strip() or not evidence_order or not answer.citations:
            raise ValueError("Copilot answer must be non-empty and cite evidence")
        citation_ids = tuple(citation.evidence_id for citation in answer.citations)
        unknown = set(citation_ids) - set(evidence_order)
        if unknown:
            raise ValueError(f"citation was not produced by evidence tools: {min(unknown)}")
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("each evidence record may be cited only once")
        expected_order = tuple(row for row in evidence_order if row in citation_ids)
        if citation_ids != expected_order:
            raise ValueError("Copilot citations must follow evidence retrieval order")
        if any(not citation.claim.strip() for citation in answer.citations):
            raise ValueError("Copilot citation claims cannot be empty")
