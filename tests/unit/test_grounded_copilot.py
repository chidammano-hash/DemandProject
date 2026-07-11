"""Grounding, evidence-integrity, and context-boundary tests for the Copilot core."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from common.ai.copilot.contracts import (
    Citation,
    CopilotAnswer,
    CopilotContext,
    EvidenceBundle,
    ProviderRequest,
    ProviderTurn,
    ToolCall,
)
from common.ai.copilot.evidence_tools import build_evidence_tools
from common.ai.copilot.runtime import CopilotRuntime


class _Reader:
    def __init__(self, evidence: EvidenceBundle) -> None:
        self.evidence = evidence
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def read_evidence(self, tool_name: str, arguments: dict[str, object]) -> EvidenceBundle:
        self.calls.append((tool_name, arguments))
        return self.evidence


class _Provider:
    def __init__(self, turns: list[ProviderTurn]) -> None:
        self.turns = turns
        self.requests: list[ProviderRequest] = []

    async def complete(self, request: ProviderRequest) -> ProviderTurn:
        self.requests.append(request)
        return self.turns.pop(0)


def _evidence(**overrides: object) -> EvidenceBundle:
    values: dict[str, object] = {"item_id": "ITEM-1", "forecast_qty": 42.0}
    values.update(overrides.pop("values", {}))
    return EvidenceBundle.create(
        evidence_id=str(overrides.pop("evidence_id", "evidence-1")),
        tool_name=str(overrides.pop("tool_name", "get_dfu_evidence")),
        source=str(overrides.pop("source", "copilot.dfu_evidence")),
        business_key=str(overrides.pop("business_key", "ITEM-1|LOC-1")),
        as_of=overrides.pop("as_of", datetime(2026, 7, 1, tzinfo=UTC)),
        freshness=str(overrides.pop("freshness", "active_release")),
        values=values,
        item_id=str(overrides.pop("item_id", "ITEM-1")),
        loc=str(overrides.pop("loc", "LOC-1")),
        customer_group=str(overrides.pop("customer_group", "")),
        promotion_id=int(overrides.pop("promotion_id", 77)),
        production_run_id=str(overrides.pop("production_run_id", "run-1")),
    )


def test_evidence_hash_covers_values_and_lineage() -> None:
    original = _evidence()
    changed_value = _evidence(values={"item_id": "ITEM-1", "forecast_qty": 43.0})
    changed_release = _evidence(promotion_id=78)

    assert original.content_hash == original.expected_content_hash()
    assert changed_value.content_hash != original.content_hash
    assert changed_release.content_hash != original.content_hash


@pytest.mark.asyncio
async def test_context_anchored_runtime_prefetches_evidence_before_generation() -> None:
    evidence = _evidence()
    reader = _Reader(evidence)
    provider = _Provider(
        [
            ProviderTurn(
                answer=CopilotAnswer(
                    answer="The active forecast is 42 units.",
                    citations=(Citation(evidence_id="evidence-1", claim="Forecast is 42"),),
                )
            )
        ]
    )
    runtime = CopilotRuntime(
        provider=provider,
        tools=build_evidence_tools(reader),
        prefetch_context_evidence=True,
    )

    result = await runtime.run_grounded(
        prompt="What is the forecast?",
        context=CopilotContext(
            page="itemAnalysis",
            item_id="ITEM-1",
            loc="LOC-1",
            promotion_id=77,
            production_run_id="run-1",
        ),
    )

    assert reader.calls == [
        (
            "get_dfu_evidence",
            {"item_id": "ITEM-1", "customer_group": "", "loc": "LOC-1"},
        )
    ]
    assert provider.requests[0].prior_results[0].evidence == evidence
    assert result.answer.answer == "The active forecast is 42 units."


@pytest.mark.asyncio
async def test_runtime_rejects_fabricated_citation() -> None:
    reader = _Reader(_evidence())
    provider = _Provider(
        [
            ProviderTurn(
                answer=CopilotAnswer(
                    answer="Unsupported answer",
                    citations=(Citation(evidence_id="fabricated", claim="Unsupported"),),
                )
            )
        ]
    )
    runtime = CopilotRuntime(
        provider=provider,
        tools=build_evidence_tools(reader),
        prefetch_context_evidence=True,
    )

    with pytest.raises(ValueError, match="not produced"):
        await runtime.run_grounded(
            prompt="What is the forecast?",
            context=CopilotContext(page="itemAnalysis", item_id="ITEM-1", loc="LOC-1"),
        )


@pytest.mark.asyncio
async def test_evidence_registry_rejects_extra_arguments_before_read() -> None:
    reader = _Reader(_evidence())
    tools = build_evidence_tools(reader)

    with pytest.raises(ValueError, match="unexpected arguments"):
        await tools.execute(
            "get_dfu_evidence",
            {
                "item_id": "ITEM-1",
                "customer_group": "",
                "loc": "LOC-1",
                "arbitrary_sql": "drop table anything",
            },
        )

    assert reader.calls == []


@pytest.mark.asyncio
async def test_runtime_rejects_provider_tool_call_outside_context() -> None:
    evidence = _evidence()
    reader = _Reader(evidence)
    provider = _Provider(
        [
            ProviderTurn(
                tool_calls=(
                    ToolCall(
                        call_id="call-2",
                        name="get_dfu_evidence",
                        arguments={
                            "item_id": "ITEM-2",
                            "customer_group": "",
                            "loc": "LOC-1",
                        },
                    ),
                )
            )
        ]
    )
    runtime = CopilotRuntime(provider=provider, tools=build_evidence_tools(reader))

    with pytest.raises(ValueError, match="outside the anchored DFU"):
        await runtime.run_grounded(
            prompt="Compare another item",
            context=CopilotContext(page="itemAnalysis", item_id="ITEM-1", loc="LOC-1"),
        )

    assert reader.calls == []
