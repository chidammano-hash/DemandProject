"""Closed read-only tool registry for grounded Copilot evidence."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from common.ai.copilot.contracts import EvidenceBundle

ReadTool = Callable[[dict[str, Any]], Awaitable[EvidenceBundle]]


class ToolRegistry:
    def __init__(self) -> None:
        self._read_tools: dict[str, ReadTool] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._read_tools)

    def register_read_tool(self, name: str, tool: ReadTool) -> None:
        if name in self._read_tools:
            raise ValueError(f"duplicate evidence tool: {name}")
        self._read_tools[name] = tool

    async def execute(self, name: str, arguments: dict[str, Any]) -> EvidenceBundle:
        try:
            tool = self._read_tools[name]
        except KeyError as exc:
            raise ValueError(f"unknown or write-capable evidence tool: {name}") from exc
        return await tool(arguments)
