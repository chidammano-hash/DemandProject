"""On-demand, single-DFU AI Champion adjustment (interactive).

Spec: docs/specs/02-forecasting/27-ai-champion-forecast.md

DB-aware orchestration around the pure ``champion_adjuster`` brain. Powers the
interactive "AI Adjust" button on the Item Analysis tab — there is no batch
pipeline. Two operations:

  * :func:`adjust_dfu`   — build the DFU context (champion forward forecast +
    actuals + item/location attributes + top customers), call the LLM once, and
    return a *preview* (no DB write).
  * :func:`save_adjustment` — re-derive the adjustment deterministically from the
    (guard-railed) recommendation against the authoritative champion baseline and
    persist it to ``fact_ai_champion_forecast`` under an interactive run.

Quantities are always recomputed server-side from the champion baseline in the
DB — the client only echoes back the recommendation, never the quantities.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import psycopg

from common.ai.champion_adjuster import (
    PROMPT_VERSION,
    CustomerHistory,
    DfuContext,
    Recommendation,
    apply_guardrails,
    apply_recommendation,
    recommend,
)
from common.ai.llm_client import LLMClientError, build_from_config
from common.core.db import get_db_params
from common.core.planning_date import get_planning_date
from common.core.utils import _ts, load_config

logger = logging.getLogger(__name__)

CONFIG_NAME = "ai_champion_config"
TARGET_MODEL_ID = "ai_champion"


class NoChampionForecast(ValueError):  # noqa: N818 — domain exception, matches UnknownAlgorithm convention
    """The DFU has no promoted champion forward forecast to adjust."""


class UnknownProvider(ValueError):  # noqa: N818 — domain exception, matches UnknownAlgorithm convention
    """Requested provider is not configured in ai_champion_config.models."""


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------

@dataclass
class MonthRow:
    forecast_month: date
    horizon_months: int
    champion_qty: float
    ai_qty: float
    pct_change: float | None   # per-month derived (ai vs champion)


@dataclass
class AdjustPreview:
    item_id: str
    loc: str
    plan_version: str
    provider: str
    model: str
    prompt_version: str
    recommendation_code: str
    rec_pct_change: float | None
    proposed_qty: list[float] | None
    apply_horizon_months: int
    confidence: float | None
    rationale: str
    evidence_keys: list[str]
    months: list[MonthRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "loc": self.loc,
            "plan_version": self.plan_version,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "recommendation_code": self.recommendation_code,
            "rec_pct_change": self.rec_pct_change,
            "proposed_qty": self.proposed_qty,
            "apply_horizon_months": self.apply_horizon_months,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "evidence_keys": self.evidence_keys,
            "months": [
                {
                    "forecast_month": m.forecast_month.isoformat(),
                    "horizon_months": m.horizon_months,
                    "champion_qty": round(m.champion_qty, 2),
                    "ai_qty": round(m.ai_qty, 2),
                    "pct_change": round(m.pct_change, 2) if m.pct_change is not None else None,
                }
                for m in self.months
            ],
        }


# ---------------------------------------------------------------------------
# Single-DFU context loaders
# ---------------------------------------------------------------------------

def _month_start(d: date) -> date:
    return d.replace(day=1)


def _resolve_plan_version(cur: Any, source_model_id: str) -> str | None:
    cur.execute(
        "SELECT plan_version FROM fact_production_forecast WHERE model_id = %s "
        "GROUP BY plan_version ORDER BY MAX(generated_at) DESC NULLS LAST LIMIT 1",
        (source_model_id,),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _load_champion_forward(cur: Any, source_model_id: str, plan_version: str, plan_month: date,
                           window: int, item_id: str, loc: str) -> list[tuple[str, float, date, int]]:
    cur.execute(
        """SELECT forecast_month, forecast_qty, horizon_months
           FROM fact_production_forecast
           WHERE model_id = %s AND plan_version = %s AND forecast_month >= %s
             AND item_id = %s AND loc = %s
           ORDER BY forecast_month
           LIMIT %s""",
        (source_model_id, plan_version, plan_month, item_id, loc, window),
    )
    out: list[tuple[str, float, date, int]] = []
    for i, (fmonth, qty, horizon) in enumerate(cur.fetchall()):
        out.append((fmonth.strftime("%Y-%m"), float(qty or 0.0), fmonth, int(horizon or i + 1)))
    return out


def _load_actuals(cur: Any, plan_month: date, lookback: int, item_id: str, loc: str) -> list[tuple[str, float]]:
    cur.execute(
        """SELECT to_char(startdate, 'YYYY-MM') AS ym, SUM(qty)
           FROM fact_sales_monthly
           WHERE type = 1 AND item_id = %s AND loc = %s
             AND startdate < %s AND startdate >= %s - (%s * interval '1 month')
           GROUP BY ym ORDER BY ym""",
        (item_id, loc, plan_month, plan_month, lookback),
    )
    return [(ym, float(qty or 0.0)) for ym, qty in cur.fetchall()]


def _load_item_attrs(cur: Any, item_id: str, loc: str) -> tuple[dict[str, str], str | None, str | None]:
    """Return (display attrs, cluster_assignment, abc_vol) for one DFU from dim_sku."""
    cur.execute(
        """SELECT brand_desc, prod_cat_desc, prod_subgrp_desc, size, premise,
                  region, supplier_desc, abc_vol, cluster_assignment
           FROM dim_sku WHERE item_id = %s AND loc = %s LIMIT 1""",
        (item_id, loc),
    )
    row = cur.fetchone()
    if not row:
        return {}, None, None
    brand, cat, subgrp, size, premise, region, supplier, abc, cluster = row
    attrs = {
        "brand": brand, "category": cat, "subgroup": subgrp, "size": size,
        "premise": premise, "region": region, "supplier": supplier,
    }
    return {k: v for k, v in attrs.items() if v}, cluster, abc


def _load_location_attrs(cur: Any, loc: str) -> dict[str, str]:
    cur.execute(
        "SELECT site_desc, state_id, primary_demand_location FROM dim_location "
        "WHERE location_id = %s LIMIT 1",
        (loc,),
    )
    row = cur.fetchone()
    if not row:
        return {}
    site, state, primary = row
    attrs = {"site": site, "state": state, "primary_demand_location": primary}
    return {k: v for k, v in attrs.items() if v}


def _load_top_customers(cur: Any, plan_month: date, top_k: int, item_id: str, loc: str) -> list[CustomerHistory]:
    if top_k <= 0:
        return []
    cur.execute(
        """SELECT customer_group, to_char(startdate, 'YYYY-MM') AS ym, SUM(qty)
           FROM fact_sales_monthly
           WHERE type = 1 AND item_id = %s AND loc = %s
             AND startdate < %s AND startdate >= %s - (12 * interval '1 month')
           GROUP BY customer_group, ym ORDER BY customer_group, ym""",
        (item_id, loc, plan_month, plan_month),
    )
    by_cust: dict[str, list[tuple[str, float]]] = {}
    for cg, ym, qty in cur.fetchall():
        by_cust.setdefault(cg, []).append((ym, float(qty or 0.0)))
    ranked = sorted(by_cust.items(), key=lambda kv: sum(q for _, q in kv[1]), reverse=True)
    return [CustomerHistory(customer_no=cg, customer_name=None, monthly=monthly)
            for cg, monthly in ranked[:top_k]]


@dataclass
class _ContextBundle:
    ctx: DfuContext
    forward: list[tuple[str, float, date, int]]
    plan_version: str
    plan_month: date


def _build_context(cur: Any, cfg: dict, item_id: str, loc: str) -> _ContextBundle:
    """Load all per-DFU context. Raises NoChampionForecast if there is no baseline."""
    defaults = cfg.get("defaults", {})
    source_model_id = defaults.get("source_model_id", "champion")
    window = int(defaults.get("forecast_window_months", 12))
    lookback = int(defaults.get("actuals_lookback_months", 24))
    top_k = int(defaults.get("top_customers", 5))
    plan_month = _month_start(get_planning_date())

    plan_version = _resolve_plan_version(cur, source_model_id)
    if not plan_version:
        raise NoChampionForecast(f"No champion forecast (model_id={source_model_id!r}) in any plan version")

    forward = _load_champion_forward(cur, source_model_id, plan_version, plan_month, window, item_id, loc)
    if not forward:
        raise NoChampionForecast(f"No champion forward forecast for {item_id}/{loc} in plan {plan_version}")

    actuals = _load_actuals(cur, plan_month, lookback, item_id, loc)
    item_attrs, cluster, abc = _load_item_attrs(cur, item_id, loc)
    location_attrs = _load_location_attrs(cur, loc)
    customers = _load_top_customers(cur, plan_month, top_k, item_id, loc)

    ctx = DfuContext(
        item_id=item_id, loc=loc, forecast_run_month=plan_month,
        actuals_last_24m=actuals,
        baseline_forecast=[(ym, q) for ym, q, _m, _h in forward],
        cluster=cluster, abc_vol=abc,
        item_attrs=item_attrs or None, location_attrs=location_attrs or None,
        top_customers=customers or None,
    )
    return _ContextBundle(ctx=ctx, forward=forward, plan_version=plan_version, plan_month=plan_month)


# ---------------------------------------------------------------------------
# Recommendation → per-month rows (deterministic, derived from champion baseline)
# ---------------------------------------------------------------------------

def _months_from_rec(rec: Recommendation, forward: list[tuple[str, float, date, int]]) -> list[MonthRow]:
    baseline = [(ym, q) for ym, q, _m, _h in forward]
    adjusted = apply_recommendation(rec, baseline)
    months: list[MonthRow] = []
    for (_ym, champ, fmonth, horizon), (_ym2, ai) in zip(forward, adjusted, strict=False):
        pct = ((ai / champ) - 1.0) * 100.0 if champ else None
        months.append(MonthRow(forecast_month=fmonth, horizon_months=horizon,
                               champion_qty=champ, ai_qty=ai, pct_change=pct))
    return months


def _resolve_provider_model(cfg: dict, provider: str | None) -> tuple[str, str]:
    effective = provider or cfg.get("provider", "ollama")
    models = cfg.get("models", {})
    if effective not in models:
        raise UnknownProvider(f"Provider {effective!r} not in ai_champion_config.models ({sorted(models)})")
    return effective, models[effective]


# ---------------------------------------------------------------------------
# Public operations
# ---------------------------------------------------------------------------

def adjust_dfu(item_id: str, loc: str, *, provider: str | None = None) -> AdjustPreview:
    """Call the LLM once for one DFU and return a preview (no DB write).

    Raises NoChampionForecast / UnknownProvider. LLM/parse failures degrade to a
    KEEP recommendation so the caller always gets a complete preview.
    """
    cfg = load_config(CONFIG_NAME)
    guardrails = cfg.get("apply_guardrails", {})
    horizon = int(cfg.get("defaults", {}).get("horizon_months", 3))
    effective_provider, model = _resolve_provider_model(cfg, provider)

    with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
        bundle = _build_context(cur, cfg, item_id, loc)

    client = build_from_config(cfg, override_provider=effective_provider)
    try:
        rec, _resp = recommend(client, bundle.ctx)
        rec = apply_guardrails(rec, guardrails, horizon)
    except (LLMClientError, ValueError) as exc:
        logger.warning("%s AI adjust %s/%s failed (%s) — keeping champion", _ts(), item_id, loc, type(exc).__name__)
        rec = Recommendation(recommendation_code="KEEP", confidence=0.0,
                             rationale="LLM unavailable — kept champion baseline.")

    months = _months_from_rec(rec, bundle.forward)
    return AdjustPreview(
        item_id=item_id, loc=loc, plan_version=bundle.plan_version,
        provider=effective_provider, model=model, prompt_version=PROMPT_VERSION,
        recommendation_code=rec.recommendation_code, rec_pct_change=rec.pct_change,
        proposed_qty=rec.proposed_qty, apply_horizon_months=rec.apply_horizon_months,
        confidence=rec.confidence, rationale=rec.rationale, evidence_keys=rec.evidence_keys,
        months=months,
    )


def _get_or_create_interactive_run(cur: Any, plan_version: str, provider: str, model: str) -> Any:
    """One open 'interactive' run per (plan_version, provider); created on first save."""
    cur.execute(
        "SELECT run_id FROM ai_champion_run "
        "WHERE plan_version = %s AND provider = %s AND status = 'interactive' "
        "ORDER BY started_at DESC LIMIT 1",
        (plan_version, provider),
    )
    row = cur.fetchone()
    if row:
        cur.execute("UPDATE ai_champion_run SET ai_model = %s WHERE run_id = %s", (model, row[0]))
        return row[0]
    run_id = uuid.uuid4()
    cur.execute(
        """INSERT INTO ai_champion_run
               (run_id, plan_version, provider, ai_model, prompt_version, status, est_cost_usd)
           VALUES (%s, %s, %s, %s, %s, 'interactive', 0)""",
        (run_id, plan_version, provider, model, PROMPT_VERSION),
    )
    return run_id


_UPSERT_SQL = """
    INSERT INTO fact_ai_champion_forecast
        (run_id, plan_version, item_id, loc, forecast_month, horizon_months,
         champion_qty, ai_qty, model_id, recommendation_code, pct_change,
         confidence, rationale, evidence_keys)
    VALUES (%(run_id)s, %(plan_version)s, %(item_id)s, %(loc)s, %(forecast_month)s,
            %(horizon_months)s, %(champion_qty)s, %(ai_qty)s, %(model_id)s,
            %(recommendation_code)s, %(pct_change)s, %(confidence)s,
            %(rationale)s, %(evidence_keys)s)
    ON CONFLICT (plan_version, item_id, loc, forecast_month) DO UPDATE SET
        run_id = EXCLUDED.run_id, horizon_months = EXCLUDED.horizon_months,
        champion_qty = EXCLUDED.champion_qty, ai_qty = EXCLUDED.ai_qty,
        recommendation_code = EXCLUDED.recommendation_code, pct_change = EXCLUDED.pct_change,
        confidence = EXCLUDED.confidence, rationale = EXCLUDED.rationale,
        evidence_keys = EXCLUDED.evidence_keys, generated_at = now()
"""


def save_adjustment(item_id: str, loc: str, *, provider: str | None,
                    recommendation: dict[str, Any]) -> dict[str, Any]:
    """Persist an adjustment, re-deriving quantities from the champion baseline.

    The client passes back the recommendation it previewed; quantities are
    recomputed here from the authoritative DB baseline (and guardrails re-applied),
    so the saved ai_qty never trusts client-supplied numbers.
    """
    cfg = load_config(CONFIG_NAME)
    guardrails = cfg.get("apply_guardrails", {})
    horizon = int(cfg.get("defaults", {}).get("horizon_months", 3))
    effective_provider, model = _resolve_provider_model(cfg, provider)

    rec = apply_guardrails(Recommendation.model_validate(recommendation), guardrails, horizon)

    with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
        bundle = _build_context(cur, cfg, item_id, loc)
        months = _months_from_rec(rec, bundle.forward)
        run_id = _get_or_create_interactive_run(cur, bundle.plan_version, effective_provider, model)
        cur.executemany(_UPSERT_SQL, [
            {
                "run_id": run_id, "plan_version": bundle.plan_version,
                "item_id": item_id, "loc": loc, "forecast_month": m.forecast_month,
                "horizon_months": m.horizon_months, "champion_qty": m.champion_qty,
                "ai_qty": m.ai_qty, "model_id": TARGET_MODEL_ID,
                "recommendation_code": rec.recommendation_code, "pct_change": m.pct_change,
                "confidence": rec.confidence, "rationale": rec.rationale,
                "evidence_keys": rec.evidence_keys,
            }
            for m in months
        ])
        cur.execute(
            """UPDATE ai_champion_run SET
                   n_dfus = (SELECT COUNT(DISTINCT (item_id, loc))
                             FROM fact_ai_champion_forecast WHERE run_id = %s),
                   n_adjusted = (SELECT COUNT(DISTINCT (item_id, loc))
                                 FROM fact_ai_champion_forecast
                                 WHERE run_id = %s
                                   AND recommendation_code NOT IN ('KEEP', 'OVERRIDE_TO_BASELINE')),
                   completed_at = now()
               WHERE run_id = %s""",
            (run_id, run_id, run_id),
        )
        conn.commit()

    logger.info("%s AI Champion saved %s/%s rec=%s (%d months)", _ts(), item_id, loc,
                rec.recommendation_code, len(months))
    return {
        "item_id": item_id, "loc": loc, "plan_version": bundle.plan_version,
        "run_id": str(run_id), "recommendation_code": rec.recommendation_code,
        "saved_months": len(months),
    }
