"""Grounded AI Q&A for the Customer Analytics workspace."""
from __future__ import annotations

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel, ConfigDict, Field

from api.auth import require_api_key
from common.ai.customer_analytics_assistant import (
    CustomerAnalyticsAssistantError,
    answer_customer_question,
)

from .kpis import customer_analytics_kpis
from .ranking import customer_analytics_ranking

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/customer-analytics", tags=["customer-analytics"])


class AssistantFilters(BaseModel):
    """Current dashboard filters used to ground the answer."""

    model_config = ConfigDict(extra="forbid")

    item_id: str | None = Field(default=None, max_length=120)
    date_from: str | None = Field(default=None, max_length=10)
    date_to: str | None = Field(default=None, max_length=10)
    channel: str | None = Field(default=None, max_length=160)
    store_type: str | None = Field(default=None, max_length=160)
    state: str | None = Field(default=None, max_length=8)


class AssistantHistoryMessage(BaseModel):
    """One bounded prior chat turn."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class CustomerAnalyticsAskRequest(BaseModel):
    """Question plus its visible dashboard scope."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=2, max_length=500)
    filters: AssistantFilters = Field(default_factory=AssistantFilters)
    active_view: Literal["overview", "customers", "segments", "service", "behavior"] = "overview"
    history: list[AssistantHistoryMessage] = Field(default_factory=list, max_length=6)


class CustomerAnalyticsAskResponse(BaseModel):
    """Grounded answer and runtime disclosure for the UI."""

    answer: str
    provider: str
    model: str
    tier: str
    evidence: list[str]


async def _build_dashboard_context(body: CustomerAnalyticsAskRequest) -> dict:
    filters = body.filters.model_dump()
    kpis_result, top_result, risk_result = await asyncio.gather(
        customer_analytics_kpis(response=FastAPIResponse(), **filters),
        customer_analytics_ranking(
            response=FastAPIResponse(),
            sort="demand_desc",
            top_n=5,
            min_demand=0,
            **filters,
        ),
        customer_analytics_ranking(
            response=FastAPIResponse(),
            sort="fill_rate_asc",
            top_n=5,
            min_demand=0,
            **filters,
        ),
    )
    return {
        "active_view": body.active_view,
        "filters": filters,
        "metric_definitions": {
            "total_demand": "value is cases; delta is month-over-month percent change",
            "fill_rate": "value is percent; delta is month-over-month percentage-point change",
            "oos_volume": "value is lost cases; delta is month-over-month percent change",
            "active_customers": "value is customer count; delta is month-over-month percent change",
            "concentration_top10": "value is percent of demand from the top 10 customers; delta is unavailable",
            "order_demand_ratio": "value is sales divided by demand; delta is unavailable",
        },
        "kpis": kpis_result.get("kpis", []),
        "top_customers": top_result.get("customers", []),
        "service_risks": risk_result.get("customers", []),
    }


@router.post(
    "/ask",
    response_model=CustomerAnalyticsAskResponse,
    dependencies=[Depends(require_api_key)],
)
async def ask_customer_analytics(body: CustomerAnalyticsAskRequest) -> CustomerAnalyticsAskResponse:
    """Answer a question using KPIs and customer rankings for the active filters."""
    try:
        context = await _build_dashboard_context(body)
        answer = await answer_customer_question(
            body.question.strip(),
            context,
            history=[message.model_dump() for message in body.history],
        )
    except CustomerAnalyticsAssistantError as exc:
        logger.warning("customer analytics assistant unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Customer Analytics assistant is temporarily unavailable",
        ) from exc

    return CustomerAnalyticsAskResponse(
        answer=answer.answer,
        provider=answer.provider,
        model=answer.model,
        tier=answer.tier,
        evidence=["KPIs", "Top demand customers", "Lowest fill-rate customers"],
    )
