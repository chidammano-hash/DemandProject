"""Typed, provider-neutral contracts for evidence-grounded Copilot turns."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol


def _content_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class CopilotContext:
    """Server-resolved page and business-object scope for one conversation."""

    page: str
    item_id: str | None = None
    customer_group: str = ""
    loc: str | None = None
    opportunity_id: str | None = None
    exception_id: str | None = None
    workflow_run_id: str | None = None
    promotion_id: int | None = None
    production_run_id: str | None = None
    inventory_run_id: str | None = None

    def __post_init__(self) -> None:
        if not self.page.strip():
            raise ValueError("page is required")
        if bool(self.item_id) != bool(self.loc):
            raise ValueError("item_id and loc must be supplied together")


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """One bounded authoritative snapshot and its reproducibility metadata."""

    evidence_id: str
    tool_name: str
    source: str
    business_key: str
    as_of: datetime
    freshness: str
    values: dict[str, Any]
    content_hash: str
    item_id: str | None = None
    customer_group: str = ""
    loc: str | None = None
    opportunity_id: str | None = None
    exception_id: str | None = None
    workflow_run_id: str | None = None
    promotion_id: int | None = None
    production_run_id: str | None = None
    inventory_run_id: str | None = None

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "source": self.source,
            "business_key": self.business_key,
            "as_of": self.as_of.isoformat(),
            "freshness": self.freshness,
            "values": self.values,
            "item_id": self.item_id,
            "customer_group": self.customer_group,
            "loc": self.loc,
            "opportunity_id": self.opportunity_id,
            "exception_id": self.exception_id,
            "workflow_run_id": self.workflow_run_id,
            "promotion_id": self.promotion_id,
            "production_run_id": self.production_run_id,
            "inventory_run_id": self.inventory_run_id,
        }

    def expected_content_hash(self) -> str:
        return _content_hash(self._hash_payload())

    @classmethod
    def create(
        cls,
        *,
        evidence_id: str,
        tool_name: str,
        source: str,
        business_key: str,
        as_of: datetime,
        freshness: str,
        values: dict[str, Any],
        item_id: str | None = None,
        customer_group: str = "",
        loc: str | None = None,
        opportunity_id: str | None = None,
        exception_id: str | None = None,
        workflow_run_id: str | None = None,
        promotion_id: int | None = None,
        production_run_id: str | None = None,
        inventory_run_id: str | None = None,
    ) -> EvidenceBundle:
        if bool(item_id) != bool(loc):
            raise ValueError("evidence item_id and loc must be supplied together")
        row = cls(
            evidence_id=evidence_id,
            tool_name=tool_name,
            source=source,
            business_key=business_key,
            as_of=as_of,
            freshness=freshness,
            values=dict(values),
            content_hash="",
            item_id=item_id,
            customer_group=customer_group,
            loc=loc,
            opportunity_id=opportunity_id,
            exception_id=exception_id,
            workflow_run_id=workflow_run_id,
            promotion_id=promotion_id,
            production_run_id=production_run_id,
            inventory_run_id=inventory_run_id,
        )
        return cls(
            **{
                field_name: getattr(row, field_name)
                for field_name in row.__dataclass_fields__
                if field_name != "content_hash"
            },
            content_hash=row.expected_content_hash(),
        )


@dataclass(frozen=True, slots=True)
class Citation:
    evidence_id: str
    claim: str


@dataclass(frozen=True, slots=True)
class CopilotAnswer:
    answer: str
    citations: tuple[Citation, ...]
    action_request: None = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProviderTurn:
    tool_calls: tuple[ToolCall, ...] = ()
    answer: CopilotAnswer | None = None

    def __post_init__(self) -> None:
        if bool(self.tool_calls) == bool(self.answer):
            raise ValueError("provider turn must contain tool calls or one answer")


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    evidence: EvidenceBundle


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    question: str
    answer: str


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    prompt: str
    context: CopilotContext
    tool_names: tuple[str, ...]
    prior_results: tuple[ToolResult, ...] = field(default_factory=tuple)
    conversation: tuple[ConversationTurn, ...] = field(default_factory=tuple)


class CopilotProvider(Protocol):
    async def complete(self, request: ProviderRequest) -> ProviderTurn: ...
