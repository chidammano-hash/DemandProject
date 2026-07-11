"""Application service boundary for persistent grounded Copilot sessions."""

from __future__ import annotations

from typing import Protocol

from common.auth import CurrentUser


class CopilotUnavailableError(RuntimeError):
    """The configured Copilot runtime or persistence layer is unavailable."""


class CopilotGroundingError(ValueError):
    """The requested context, evidence, or provider output is invalid."""


class CopilotConflictError(RuntimeError):
    """The request conflicts with an active or idempotent turn."""


class CopilotService(Protocol):
    async def create_session(
        self,
        *,
        owner: CurrentUser,
        context: dict[str, object],
        idempotency_key: str,
    ) -> dict[str, object]: ...

    async def get_session(
        self, session_id: str, *, owner: CurrentUser
    ) -> dict[str, object] | None: ...

    async def run_turn(
        self,
        session_id: str,
        *,
        owner: CurrentUser,
        prompt: str,
        idempotency_key: str,
    ) -> dict[str, object] | None: ...


class UnavailableCopilotService:
    def _raise(self) -> None:
        raise CopilotUnavailableError("grounded Copilot is not configured")

    async def create_session(
        self,
        *,
        owner: CurrentUser,
        context: dict[str, object],
        idempotency_key: str,
    ) -> dict[str, object]:
        self._raise()
        return {}

    async def get_session(self, session_id: str, *, owner: CurrentUser) -> dict[str, object] | None:
        self._raise()
        return None

    async def run_turn(
        self,
        session_id: str,
        *,
        owner: CurrentUser,
        prompt: str,
        idempotency_key: str,
    ) -> dict[str, object] | None:
        self._raise()
        return None
