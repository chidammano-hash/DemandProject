"""Fixed typed evidence surface for the DemandProject planning Copilot."""

from __future__ import annotations

from typing import Any, Protocol

from common.ai.copilot.contracts import EvidenceBundle
from common.ai.copilot.tools import ToolRegistry


class EvidenceReader(Protocol):
    async def read_evidence(self, tool_name: str, arguments: dict[str, Any]) -> EvidenceBundle: ...


_EXPECTED_KEYS: dict[str, frozenset[str]] = {
    "get_portfolio_evidence": frozenset({"limit"}),
    "get_dfu_evidence": frozenset({"item_id", "customer_group", "loc"}),
    "get_forecast_evidence": frozenset({"item_id", "customer_group", "loc"}),
    "get_inventory_plan_evidence": frozenset({"item_id", "loc"}),
    "get_inventory_opportunity_evidence": frozenset({"opportunity_id"}),
    "get_exception_evidence": frozenset({"exception_id"}),
    "get_customer_evidence": frozenset({"item_id", "customer_group", "loc"}),
    "get_workflow_run_evidence": frozenset({"workflow_run_id"}),
}


def _required_text(arguments: dict[str, Any], key: str, *, allow_blank: bool = False) -> None:
    value = arguments.get(key)
    valid = isinstance(value, str) and len(value) <= 200
    if not allow_blank:
        valid = valid and bool(value.strip())
    if not valid:
        raise ValueError(f"{key} must be at most 200 characters")


def _validate_arguments(tool_name: str, arguments: dict[str, Any]) -> None:
    expected = _EXPECTED_KEYS[tool_name]
    supplied = frozenset(arguments)
    if supplied != expected:
        raise ValueError(
            f"unexpected arguments={sorted(supplied - expected)}; "
            f"missing arguments={sorted(expected - supplied)}"
        )
    if tool_name == "get_portfolio_evidence":
        limit = arguments["limit"]
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise ValueError("limit must be an integer between 1 and 20")
        return
    for key in expected:
        _required_text(arguments, key, allow_blank=key == "customer_group")


def build_evidence_tools(reader: EvidenceReader) -> ToolRegistry:
    registry = ToolRegistry()
    for tool_name in _EXPECTED_KEYS:

        async def execute(
            arguments: dict[str, Any], *, _tool_name: str = tool_name
        ) -> EvidenceBundle:
            _validate_arguments(_tool_name, arguments)
            return await reader.read_evidence(_tool_name, arguments)

        registry.register_read_tool(tool_name, execute)
    return registry
