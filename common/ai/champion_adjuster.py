"""AI Champion adjuster — recommendation prompt, parsing, and apply rules.

Spec: docs/specs/02-forecasting/27-ai-champion-forecast.md

Repurposed from the (removed) AI FVA backtest recommender into a forward-only
champion adjuster. Three responsibilities:
  1. Build the per-DFU prompt from current context (recent actuals + the champion
     forward forecast + metadata + optional event notes).
  2. Validate the LLM's structured response against the recommendation schema.
  3. Apply the recommendation deterministically to produce the AI-adjusted
     ("ai_champion") forward forecast.

This module is *pure* — no DB calls. The service
(common/ai/champion_adjust_service.py) fetches context and persists the
adjusted forecast.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.ai.llm_client import ChatResponse, LLMClient, LLMJSONParseError

log = logging.getLogger(__name__)

PROMPT_VERSION = "v1.1.0"  # v1.1.0 adds top-customer history

# ---------------------------------------------------------------------------
# Recommendation schema
# ---------------------------------------------------------------------------

RecommendationCode = Literal[
    "KEEP", "SCALE_UP", "SCALE_DOWN", "REPLACE", "SHIFT_TIMING", "OVERRIDE_TO_BASELINE"
]


class Recommendation(BaseModel):
    """Validated AI Champion forecast recommendation."""
    model_config = ConfigDict(extra="forbid")

    recommendation_code: RecommendationCode
    pct_change: float | None = Field(default=None, ge=-100, le=500)  # for SCALE_*
    proposed_qty: list[float] | None = None                          # for REPLACE / SHIFT_TIMING
    apply_horizon_months: int = Field(default=3, ge=1, le=6)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=10, max_length=1000)
    evidence_keys: list[str] = Field(default_factory=list)

    @field_validator("recommendation_code")
    @classmethod
    def _validate_required_fields(cls, v):
        return v


# ---------------------------------------------------------------------------
# Per-DFU context built by the generator (current planning date)
# ---------------------------------------------------------------------------

@dataclass
class CustomerHistory:
    """Per-customer monthly sales history for one (item_id, location_id).
    All months are actuals up to the planning date."""
    customer_no: str
    customer_name: str | None
    monthly: list[tuple[str, float]]   # [(YYYY-MM, qty), ...] chronological

    def total(self) -> float:
        return sum(q for _, q in self.monthly)


@dataclass
class DfuContext:
    """Context for one DFU at the current planning date (T)."""
    item_id: str
    loc: str
    forecast_run_month: date            # T = planning date / champion plan month
    actuals_last_24m: list[tuple[str, float]]   # [(YYYY-MM, qty), ...] ending at T
    baseline_forecast: list[tuple[str, float]]  # champion forward forecast for T+1..T+H
    cluster: str | None = None
    demand_pattern: str | None = None
    abc_vol: str | None = None
    notes: str | None = None             # any anomaly / customer-event signals
    top_customers: list[CustomerHistory] | None = None  # top-K customers buying this item@loc
    item_attrs: dict[str, str] | None = None      # dim_sku attributes (brand, category, size, …)
    location_attrs: dict[str, str] | None = None  # dim_location attributes (site, state, …)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "loc": self.loc,
            "forecast_run_month": self.forecast_run_month.isoformat(),
            "actuals_last_24m": [{"month": m, "qty": q} for m, q in self.actuals_last_24m],
            "baseline_forecast": [{"month": m, "qty": q} for m, q in self.baseline_forecast],
            "cluster": self.cluster,
            "demand_pattern": self.demand_pattern,
            "abc_vol": self.abc_vol,
            "item_attrs": self.item_attrs,
            "location_attrs": self.location_attrs,
            "notes": self.notes,
            "top_customers": [
                {
                    "customer_no": c.customer_no,
                    "customer_name": c.customer_name,
                    "monthly": [{"month": m, "qty": q} for m, q in c.monthly],
                }
                for c in (self.top_customers or [])
            ],
        }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert demand-planning reviewer assisting a supply-chain forecasting system.

For each DFU (item x location), you are given:
  * the recent actuals history (up to 24 months ending at the planning month T)
  * the champion model's forward forecast for the next H months (T+1 through T+H)
  * DFU metadata (cluster, demand pattern, ABC class)
  * item attributes (brand, category, size, supplier, …) and location attributes
    (site, state, region) — use these to reason about product/market context
  * top customers buying this item at this location, with their recent per-month sales
    (use this to spot customer concentration, single-customer ramps/drops,
    and churn driving the DFU-level pattern)
  * any anomaly or customer-event notes

Your job is to decide whether the champion forward forecast should be ADJUSTED by a
human-grade override to produce the "AI Champion" forecast. You must respond with a
JSON object matching this schema:

{
  "recommendation_code": "KEEP" | "SCALE_UP" | "SCALE_DOWN" | "REPLACE" | "SHIFT_TIMING" | "OVERRIDE_TO_BASELINE",
  "pct_change":           number | null,            // required for SCALE_UP / SCALE_DOWN; range -50..+50 typical
  "proposed_qty":         [number, ...] | null,     // required for REPLACE / SHIFT_TIMING; one entry per future month
  "apply_horizon_months": integer 1..6,             // how many months ahead the recommendation applies
  "confidence":           number 0..1,              // your calibrated confidence
  "rationale":            string,                   // 1-3 sentences citing specific evidence
  "evidence_keys":        [string, ...]             // short tags like "trend_break", "promo_pull_forward",
                                                     // "customer_concentration", "seasonality_shift",
                                                     // "outlier_recent_month"
}

Decision principles:
  - Default to KEEP unless you have specific, citable evidence that the baseline is biased.
  - SCALE_UP / SCALE_DOWN by 5-30% when the recent trend clearly diverges from baseline.
  - REPLACE only when you can specify per-month quantities with high confidence.
  - SHIFT_TIMING when demand was pulled forward / pushed back (typically promotions).
  - OVERRIDE_TO_BASELINE if context is too sparse to justify a confident judgment.
  - Penalize confidence below 0.6 when evidence is weak — that triggers KEEP downstream.
  - Base your judgment on the data shown plus any event notes provided.
  - When the customer mix shows high concentration (one customer is a large share of
    volume) and that customer's recent monthly trend diverges from the baseline,
    weight that signal heavily. Use evidence_keys like "customer_concentration",
    "customer_ramp", "customer_churn", "customer_loss".

Respond ONLY with the JSON. No prose, no markdown, no code fences."""


def build_user_prompt(ctx: DfuContext) -> str:
    """Format a per-DFU user message with the current context."""
    actuals_str = ", ".join(f"{m}={q:.0f}" for m, q in ctx.actuals_last_24m)
    baseline_str = ", ".join(f"{m}={q:.0f}" for m, q in ctx.baseline_forecast)
    meta = []
    if ctx.cluster:
        meta.append(f"cluster={ctx.cluster}")
    if ctx.demand_pattern:
        meta.append(f"pattern={ctx.demand_pattern}")
    if ctx.abc_vol:
        meta.append(f"abc={ctx.abc_vol}")
    meta_str = ", ".join(meta) if meta else "none"

    parts = [
        f"DFU: item_id={ctx.item_id}, location={ctx.loc}",
        f"Planning month T = {ctx.forecast_run_month.isoformat()}",
        f"Metadata: {meta_str}",
    ]
    if ctx.item_attrs:
        parts.append(f"Item attributes: {_format_attrs(ctx.item_attrs)}")
    if ctx.location_attrs:
        parts.append(f"Location attributes: {_format_attrs(ctx.location_attrs)}")
    parts += [
        f"Actuals (last {len(ctx.actuals_last_24m)} months ending at T): {actuals_str}",
        f"Champion forward forecast (T+1..T+{len(ctx.baseline_forecast)}): {baseline_str}",
    ]
    if ctx.top_customers:
        parts.append(_format_top_customers(ctx.top_customers))
    if ctx.notes:
        parts.append(f"Anomaly/event notes: {ctx.notes}")
    parts.append("\nReturn the JSON recommendation now.")
    return "\n".join(parts)


def _format_attrs(attrs: dict[str, str]) -> str:
    """Render a compact ``key=value`` list, skipping empty values."""
    return ", ".join(f"{k}={v}" for k, v in attrs.items() if v not in (None, "", "None"))


def _format_top_customers(customers: list[CustomerHistory], *, recent_months: int = 6) -> str:
    """Render the top-K customers + their last ``recent_months`` of sales.

    Keeps the prompt compact: only the last ``recent_months`` per customer are
    shown, and each customer's total over the full lookback window is listed
    once at the end of the line so the LLM can judge concentration cheaply.
    """
    lines = [f"Top customers for this item@loc (sales last 12 months, showing last {recent_months}):"]
    grand_total = sum(c.total() for c in customers) or 1.0
    for c in customers:
        recent = c.monthly[-recent_months:]
        per_month = ", ".join(f"{m}={q:.0f}" for m, q in recent)
        share = 100.0 * c.total() / grand_total
        name = c.customer_name or "?"
        lines.append(
            f"  - {c.customer_no} {name}: {per_month} "
            f"(12mo total: {c.total():.0f}, {share:.0f}% of top set)"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Call the LLM and parse
# ---------------------------------------------------------------------------

def recommend(client: LLMClient, ctx: DfuContext, *, max_tokens: int = 1024) -> tuple[Recommendation, ChatResponse]:
    """Call the LLM for one DFU and return a validated Recommendation + raw ChatResponse.

    Raises:
        LLMJSONParseError: response was not valid JSON.
        pydantic.ValidationError: response JSON didn't match the schema.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_prompt(ctx)},
    ]
    resp = client.chat(messages, json_mode=True, max_tokens=max_tokens)
    if resp.parsed is None:
        raise LLMJSONParseError("Recommendation parse: no parsed JSON on response")
    rec = Recommendation.model_validate(resp.parsed)
    return rec, resp


# ---------------------------------------------------------------------------
# Apply guardrails
# ---------------------------------------------------------------------------

def apply_guardrails(rec: Recommendation, guardrails: dict[str, Any], horizon_months: int) -> Recommendation:
    """Clip / downgrade the recommendation per config guardrails.

    Returns a (possibly modified) Recommendation. Never raises — out-of-bounds
    recommendations are downgraded to OVERRIDE_TO_BASELINE rather than rejected.
    """
    max_pct = guardrails.get("max_abs_pct_change", 50)
    min_conf = guardrails.get("min_confidence", 0.60)
    require_evidence = guardrails.get("require_evidence", True)
    reject_on_overflow = guardrails.get("reject_on_horizon_overflow", True)

    # Low confidence → downgrade to baseline
    if rec.confidence < min_conf:
        return rec.model_copy(update={
            "recommendation_code": "OVERRIDE_TO_BASELINE",
            "pct_change": None,
            "proposed_qty": None,
            "rationale": f"[downgraded: confidence {rec.confidence:.2f} < {min_conf}] {rec.rationale}",
        })

    # Missing evidence → downgrade to baseline
    if require_evidence and not rec.evidence_keys and rec.recommendation_code != "KEEP":
        return rec.model_copy(update={
            "recommendation_code": "OVERRIDE_TO_BASELINE",
            "pct_change": None,
            "proposed_qty": None,
            "rationale": f"[downgraded: no evidence_keys] {rec.rationale}",
        })

    # Horizon overflow → clip to allowed horizon
    if reject_on_overflow and rec.apply_horizon_months > horizon_months:
        rec = rec.model_copy(update={"apply_horizon_months": horizon_months})

    # Magnitude clip on SCALE_*
    if rec.recommendation_code in ("SCALE_UP", "SCALE_DOWN") and rec.pct_change is not None:
        if abs(rec.pct_change) > max_pct:
            clipped = max_pct if rec.pct_change > 0 else -max_pct
            rec = rec.model_copy(update={
                "pct_change": clipped,
                "rationale": f"[clipped: |{rec.pct_change}%| > {max_pct}% guardrail] {rec.rationale}",
            })

    return rec


# ---------------------------------------------------------------------------
# Apply the recommendation to compute the AI-adjusted (ai_champion) forecast
# ---------------------------------------------------------------------------

def apply_recommendation(
    rec: Recommendation,
    baseline_forecast: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Deterministically translate a Recommendation into per-month AI-adjusted qty.

    Input baseline_forecast is the per-month champion forward forecast for
    T+1..T+H (list of (YYYY-MM, qty)). Returns the same shape with AI-adjusted
    quantities (the ai_champion forecast).
    """
    horizon = rec.apply_horizon_months
    code = rec.recommendation_code

    if code in ("KEEP", "OVERRIDE_TO_BASELINE"):
        return list(baseline_forecast)

    if code == "SCALE_UP":
        factor = 1.0 + (rec.pct_change or 0) / 100.0
        return [
            (m, q * factor) if i < horizon else (m, q)
            for i, (m, q) in enumerate(baseline_forecast)
        ]

    if code == "SCALE_DOWN":
        factor = 1.0 - abs(rec.pct_change or 0) / 100.0
        factor = max(factor, 0.0)
        return [
            (m, q * factor) if i < horizon else (m, q)
            for i, (m, q) in enumerate(baseline_forecast)
        ]

    if code == "REPLACE":
        proposed = rec.proposed_qty or []
        return [
            (m, float(proposed[i])) if i < min(horizon, len(proposed)) else (m, q)
            for i, (m, q) in enumerate(baseline_forecast)
        ]

    if code == "SHIFT_TIMING":
        # proposed_qty[i] interpreted as the new per-month total (same as REPLACE for
        # the affected horizon). Distinct code for bookkeeping & analysis.
        proposed = rec.proposed_qty or []
        return [
            (m, float(proposed[i])) if i < min(horizon, len(proposed)) else (m, q)
            for i, (m, q) in enumerate(baseline_forecast)
        ]

    return list(baseline_forecast)
