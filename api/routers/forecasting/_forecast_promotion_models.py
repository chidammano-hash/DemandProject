"""Typed API models for run-scoped forecast promotion."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ForecastPromotionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    promotion_type: str
    plan_version: str
    source_run_id: UUID
    production_run_id: UUID
    candidate_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    outgoing_archive_checksum: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    rows_promoted: int = Field(ge=1)
    dfu_count: int = Field(ge=1)


class ForecastGenerationSubmittedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    model_id: str
    source_run_id: UUID


class ForecastStagingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    source_run_id: UUID
    status: str
    rows_staged: int = Field(ge=1)
    dfu_count: int = Field(ge=1)
    candidate_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
