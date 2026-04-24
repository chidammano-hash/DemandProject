"""Multimodal response envelope for agent output.

Gen-4 Roadmap Cross-cutting. Agents return structured envelopes rather
than free-form text so the UI can render richly (charts, tables, action
cards) without the model needing to know client-side component APIs.

Round-trip rules:
  * `to_dict()` produces a JSON-serializable dict.
  * `from_dict()` validates `kind` against VALID_BLOCK_KINDS and raises
    ValueError for unknown kinds. Missing optional fields default
    sensibly so older payloads stay loadable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

BlockKind = Literal["chart", "table", "action_card", "markdown"]

VALID_BLOCK_KINDS: frozenset[str] = frozenset({
    "chart",        # payload: {type: 'line'|'bar'|..., series: [...], axes: {...}}
    "table",        # payload: {columns: [...], rows: [[...]]}
    "action_card",  # payload: {title, body, actions: [{id, label, kind}]}
    "markdown",     # payload: {text: '...'}
})


@dataclass
class ResponseBlock:
    """One renderable fragment of an agent reply."""
    kind: BlockKind
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if self.kind not in VALID_BLOCK_KINDS:
            raise ValueError(
                f"Invalid block kind '{self.kind}'. "
                f"Must be one of {sorted(VALID_BLOCK_KINDS)}."
            )
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "payload": dict(self.payload)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResponseBlock":
        if not isinstance(data, dict):
            raise ValueError("ResponseBlock.from_dict expects a dict")
        kind = data.get("kind")
        payload = data.get("payload", {})
        if kind is None:
            raise ValueError("ResponseBlock payload missing 'kind'")
        return cls(kind=kind, payload=payload or {})


@dataclass
class ResponseEnvelope:
    """Top-level structured agent reply.

    `narrative` is always present (plain-text summary); `blocks` can be
    empty. `trace_id` points into logs/ai_decision_ledger for audit.
    """
    narrative: str
    blocks: list[ResponseBlock] = field(default_factory=list)
    trace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "narrative": self.narrative,
            "blocks": [b.to_dict() for b in self.blocks],
            "trace_id": self.trace_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResponseEnvelope":
        if not isinstance(data, dict):
            raise ValueError("ResponseEnvelope.from_dict expects a dict")
        narrative = data.get("narrative")
        if not isinstance(narrative, str):
            raise ValueError("narrative must be a string")
        raw_blocks = data.get("blocks", []) or []
        if not isinstance(raw_blocks, list):
            raise ValueError("blocks must be a list")
        blocks = [ResponseBlock.from_dict(b) for b in raw_blocks]
        trace_id = data.get("trace_id")
        if trace_id is not None and not isinstance(trace_id, str):
            raise ValueError("trace_id must be a string or None")
        return cls(narrative=narrative, blocks=blocks, trace_id=trace_id)


__all__ = [
    "BlockKind",
    "VALID_BLOCK_KINDS",
    "ResponseBlock",
    "ResponseEnvelope",
]
