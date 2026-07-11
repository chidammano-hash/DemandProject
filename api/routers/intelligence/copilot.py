"""Owner-scoped API for the evidence-grounded planning Copilot."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from api.auth import require_api_key
from common.ai.copilot.service import (
    CopilotConflictError,
    CopilotGroundingError,
    CopilotService,
    CopilotUnavailableError,
    UnavailableCopilotService,
)
from common.auth import CurrentUser, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai-copilot", tags=["ai-copilot"])

_service: CopilotService = UnavailableCopilotService()


def configure_copilot_service(service: CopilotService) -> None:
    global _service
    _service = service


def _get_copilot_service() -> CopilotService:
    return _service


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateSessionRequest(_StrictModel):
    page: str = Field(min_length=1, max_length=80)
    item_id: str | None = Field(default=None, max_length=200)
    customer_group: str = Field(default="", max_length=200)
    loc: str | None = Field(default=None, max_length=200)
    opportunity_id: str | None = Field(default=None, max_length=200)
    exception_id: str | None = Field(default=None, max_length=200)
    workflow_run_id: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def validate_dfu(self) -> CreateSessionRequest:
        if bool(self.item_id) != bool(self.loc):
            raise ValueError("item_id and loc must be supplied together")
        return self


class TurnRequest(_StrictModel):
    prompt: str = Field(min_length=1, max_length=4_000)


def _context(body: CreateSessionRequest) -> dict[str, object]:
    return body.model_dump(exclude_none=True)


@router.post(
    "/sessions",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
)
async def create_session(
    body: CreateSessionRequest,
    owner: CurrentUser = Depends(get_current_user),
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=200),
    service: CopilotService = Depends(_get_copilot_service),
) -> dict[str, object]:
    try:
        return await service.create_session(
            owner=owner,
            context=_context(body),
            idempotency_key=idempotency_key,
        )
    except CopilotGroundingError as exc:
        raise HTTPException(status_code=422, detail="Copilot context was rejected") from exc
    except CopilotConflictError as exc:
        raise HTTPException(status_code=409, detail="Copilot session conflict") from exc
    except CopilotUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Copilot is unavailable") from exc


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    owner: CurrentUser = Depends(get_current_user),
    service: CopilotService = Depends(_get_copilot_service),
) -> dict[str, object]:
    try:
        result = await service.get_session(session_id, owner=owner)
    except CopilotUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Copilot is unavailable") from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Copilot session not found")
    return result


@router.post(
    "/sessions/{session_id}/turns",
    dependencies=[Depends(require_api_key)],
)
async def run_turn(
    session_id: str,
    body: TurnRequest,
    owner: CurrentUser = Depends(get_current_user),
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=200),
    service: CopilotService = Depends(_get_copilot_service),
) -> dict[str, Any]:
    try:
        result = await service.run_turn(
            session_id,
            owner=owner,
            prompt=body.prompt,
            idempotency_key=idempotency_key,
        )
    except CopilotGroundingError as exc:
        raise HTTPException(status_code=422, detail="Copilot output was rejected") from exc
    except CopilotConflictError as exc:
        raise HTTPException(status_code=409, detail="Copilot turn conflict") from exc
    except CopilotUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Copilot is unavailable") from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Copilot session not found")
    return result
