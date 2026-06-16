"""AI Champion forward adjuster — generate the ai_champion forecast.

Spec: docs/specs/02-forecasting/27-ai-champion-forecast.md

Forward-only repurpose of the (removed) AI FVA backtest. Reads the promoted
champion production forecast (fact_production_forecast, model_id='champion'),
asks an LLM to adjust each DFU's near-term forward forecast, applies the
recommendation deterministically, and writes the result with model_id='ai_champion'
into fact_ai_champion_forecast. No backtest, no historical actuals, no grading.

Providers (config/ai/ai_champion_config.yaml): ollama (default, local, $0) or
anthropic (Opus 4.7, metered API). Run:

    python -m scripts.forecasting.generate_ai_champion_forecast                 # Ollama, full plan
    python -m scripts.forecasting.generate_ai_champion_forecast --limit-dfus 50 # smoke
    python -m scripts.forecasting.generate_ai_champion_forecast --provider anthropic
    python -m scripts.forecasting.generate_ai_champion_forecast --dry-run
"""
from __future__ import annotations

import argparse
import logging
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import psycopg

from common.ai.champion_adjuster import (
    DfuContext,
    CustomerHistory,
    PROMPT_VERSION,
    apply_guardrails,
    apply_recommendation,
    recommend,
)
from common.ai.llm_client import LLMClient, LLMClientError, build_from_config
from common.core.db import get_db_params
from common.core.planning_date import get_planning_date
from common.core.utils import _ts, load_config

logger = logging.getLogger(__name__)

CONFIG_NAME = "ai_champion_config"
TARGET_MODEL_ID = "ai_champion"


class CostCapExceeded(RuntimeError):
    """Pre-flight estimate exceeds the configured per-run cost cap (paid providers)."""


# ---------------------------------------------------------------------------
# Context loading (current planning date — forward only, no point-in-time)
# ---------------------------------------------------------------------------

def _month_start(d: date) -> date:
    return d.replace(day=1)


def _resolve_plan_version(cur, source_model_id: str, plan_version: str | None) -> str:
    if plan_version:
        return plan_version
    cur.execute(
        "SELECT plan_version FROM fact_production_forecast WHERE model_id = %s "
        "GROUP BY plan_version ORDER BY MAX(generated_at) DESC NULLS LAST LIMIT 1",
        (source_model_id,),
    )
    row = cur.fetchone()
    if not row:
        raise RuntimeError(f"No fact_production_forecast rows for model_id={source_model_id!r}")
    return row[0]


def load_champion_forward(cur, source_model_id: str, plan_version: str, plan_month: date,
                          window: int) -> dict[tuple[str, str], list[tuple[str, float, date, int]]]:
    """Per-DFU forward champion forecast: {(item,loc): [(ym, qty, month, horizon), ...]}.

    Only forward months (>= plan_month), capped to the first ``window`` per DFU.
    """
    cur.execute(
        """SELECT item_id, loc, forecast_month, forecast_qty, horizon_months
           FROM fact_production_forecast
           WHERE model_id = %s AND plan_version = %s AND forecast_month >= %s
           ORDER BY item_id, loc, forecast_month""",
        (source_model_id, plan_version, plan_month),
    )
    out: dict[tuple[str, str], list[tuple[str, float, date, int]]] = defaultdict(list)
    for item_id, loc, fmonth, qty, horizon in cur.fetchall():
        rows = out[(item_id, loc)]
        if len(rows) < window:
            rows.append((fmonth.strftime("%Y-%m"), float(qty or 0.0), fmonth, int(horizon or len(rows) + 1)))
    return out


def load_actuals(cur, plan_month: date, lookback: int) -> dict[tuple[str, str], list[tuple[str, float]]]:
    """Per-DFU monthly actuals for the trailing ``lookback`` months ending before plan_month."""
    cur.execute(
        """SELECT item_id, loc, to_char(startdate, 'YYYY-MM') AS ym, SUM(qty)
           FROM fact_sales_monthly
           WHERE type = 1 AND startdate < %s
             AND startdate >= %s - (%s * interval '1 month')
           GROUP BY item_id, loc, ym
           ORDER BY item_id, loc, ym""",
        (plan_month, plan_month, lookback),
    )
    out: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for item_id, loc, ym, qty in cur.fetchall():
        out[(item_id, loc)].append((ym, float(qty or 0.0)))
    return out


def load_metadata(cur) -> dict[tuple[str, str], tuple[str | None, str | None]]:
    """Per-DFU (cluster, abc_vol) metadata from dim_sku."""
    cur.execute("SELECT item_id, loc, cluster_assignment, abc_vol FROM dim_sku")
    return {(item_id, loc): (cluster, abc) for item_id, loc, cluster, abc in cur.fetchall()}


def load_top_customers(cur, plan_month: date, top_k: int) -> dict[tuple[str, str], list[CustomerHistory]]:
    """Per-DFU top-``top_k`` customers (by trailing-12-month sales) with monthly history."""
    if top_k <= 0:
        return {}
    cur.execute(
        """SELECT item_id, loc, customer_group, to_char(startdate, 'YYYY-MM') AS ym, SUM(qty)
           FROM fact_sales_monthly
           WHERE type = 1 AND startdate < %s
             AND startdate >= %s - (12 * interval '1 month')
           GROUP BY item_id, loc, customer_group, ym
           ORDER BY item_id, loc, customer_group, ym""",
        (plan_month, plan_month),
    )
    # {(item,loc): {customer_group: [(ym, qty), ...]}}
    grouped: dict[tuple[str, str], dict[str, list[tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for item_id, loc, cg, ym, qty in cur.fetchall():
        grouped[(item_id, loc)][cg].append((ym, float(qty or 0.0)))

    out: dict[tuple[str, str], list[CustomerHistory]] = {}
    for dfu, by_cust in grouped.items():
        ranked = sorted(by_cust.items(), key=lambda kv: sum(q for _, q in kv[1]), reverse=True)
        out[dfu] = [
            CustomerHistory(customer_no=cg, customer_name=None, monthly=monthly)
            for cg, monthly in ranked[:top_k]
        ]
    return out


# ---------------------------------------------------------------------------
# Per-DFU adjustment (pure compute — no DB)
# ---------------------------------------------------------------------------

def _adjust_one(
    client: LLMClient,
    dfu: tuple[str, str],
    forward: list[tuple[str, float, date, int]],
    actuals: list[tuple[str, float]],
    meta: tuple[str | None, str | None],
    customers: list[CustomerHistory] | None,
    plan_month: date,
    guardrails: dict,
    horizon_months: int,
) -> tuple[list[dict], bool]:
    """Adjust one DFU's forward forecast. Returns (per-month row dicts, changed).

    On any LLM/parse error the DFU falls back to KEEP (= champion), so it still
    yields a complete ai_champion forecast.
    """
    item_id, loc = dfu
    baseline = [(ym, qty) for ym, qty, _m, _h in forward]
    cluster, abc = meta

    ctx = DfuContext(
        item_id=item_id, loc=loc, forecast_run_month=plan_month,
        actuals_last_24m=actuals, baseline_forecast=baseline,
        cluster=cluster, abc_vol=abc, top_customers=customers or None,
    )

    try:
        rec, _resp = recommend(client, ctx)
        rec = apply_guardrails(rec, guardrails, horizon_months)
    except (LLMClientError, ValueError) as exc:  # parse/validation/transport → keep champion
        logger.warning("%s DFU %s/%s adjustment failed (%s) — keeping champion",
                       _ts(), item_id, loc, type(exc).__name__)
        adjusted = baseline
        rec = None
    else:
        adjusted = apply_recommendation(rec, baseline)

    rows = []
    for (ym, champ_qty, fmonth, horizon), (_ym2, ai_qty) in zip(forward, adjusted):
        rows.append({
            "item_id": item_id, "loc": loc, "forecast_month": fmonth,
            "horizon_months": horizon, "champion_qty": champ_qty, "ai_qty": ai_qty,
            "recommendation_code": rec.recommendation_code if rec else "KEEP",
            "pct_change": rec.pct_change if rec else None,
            "confidence": rec.confidence if rec else None,
            "rationale": rec.rationale if rec else "LLM unavailable — kept champion baseline.",
            "evidence_keys": rec.evidence_keys if rec else [],
        })
    changed = bool(rec and rec.recommendation_code not in ("KEEP", "OVERRIDE_TO_BASELINE"))
    return rows, changed


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

_INSERT_SQL = """
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


def _persist(cur, run_id: str, plan_version: str, rows: list[dict]) -> None:
    cur.executemany(_INSERT_SQL, [
        {**r, "run_id": run_id, "plan_version": plan_version, "model_id": TARGET_MODEL_ID}
        for r in rows
    ])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_pipeline(
    *,
    provider: str | None = None,
    plan_version: str | None = None,
    limit_dfus: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Generate the ai_champion forecast. Returns a summary dict."""
    cfg = load_config(CONFIG_NAME)
    defaults = cfg.get("defaults", {})
    source_model_id = defaults.get("source_model_id", "champion")
    horizon_months = int(defaults.get("horizon_months", 3))
    window = int(defaults.get("forecast_window_months", 12))
    lookback = int(defaults.get("actuals_lookback_months", 24))
    top_k = int(defaults.get("top_customers", 5))
    guardrails = cfg.get("apply_guardrails", {})
    cost = cfg.get("cost_controls", {})
    workers = int(cost.get("max_concurrent_ai_calls", 4))

    effective_provider = provider or cfg.get("provider", "ollama")
    model = cfg["models"][effective_provider]
    plan_month = _month_start(get_planning_date())
    db_params = get_db_params()

    # ── Load context ───────────────────────────────────────────────────────
    with psycopg.connect(**db_params) as conn, conn.cursor() as cur:
        plan_version = _resolve_plan_version(cur, source_model_id, plan_version)
        forward = load_champion_forward(cur, source_model_id, plan_version, plan_month, window)
        dfus = sorted(forward.keys())
        if limit_dfus is not None:
            dfus = dfus[:limit_dfus]
            forward = {d: forward[d] for d in dfus}
        actuals = load_actuals(cur, plan_month, lookback)
        meta = load_metadata(cur)
        customers = load_top_customers(cur, plan_month, top_k)

    n_dfus = len(dfus)
    per_call = float(cost.get("per_call_estimated_cost_usd", {}).get(effective_provider, 0.0))
    est_cost = round(per_call * n_dfus, 4)
    max_cost = float(cost.get("per_run_max_cost_usd", 0) or 0)

    logger.info(
        "%s AI Champion plan=%s provider=%s model=%s dfus=%d window=%d est_cost=$%.2f",
        _ts(), plan_version, effective_provider, model, n_dfus, window, est_cost,
    )

    if dry_run:
        return {"dry_run": True, "plan_version": plan_version, "provider": effective_provider,
                "model": model, "n_dfus": n_dfus, "window_months": window, "est_cost_usd": est_cost}

    if effective_provider != "ollama" and max_cost and est_cost > max_cost:
        raise CostCapExceeded(
            f"Estimated ${est_cost:.2f} exceeds per_run_max_cost_usd ${max_cost:.2f} "
            f"for provider {effective_provider} ({n_dfus} DFUs). Lower --limit-dfus or raise the cap."
        )

    client = build_from_config(cfg, override_provider=provider)
    run_id = str(uuid.uuid4())

    # ── Insert run header ────────────────────────────────────────────────────
    with psycopg.connect(**db_params, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO ai_champion_run
                   (run_id, plan_version, provider, ai_model, prompt_version, status, n_dfus, est_cost_usd)
               VALUES (%s, %s, %s, %s, %s, 'running', %s, %s)""",
            (run_id, plan_version, effective_provider, model, PROMPT_VERSION, n_dfus, est_cost),
        )

    # ── Fan out LLM calls; persist on the main thread ────────────────────────
    # try/finally (no bare except): any propagating error marks the run failed
    # and re-raises naturally for the caller/job runner.
    n_adjusted = 0
    succeeded = False
    try:
        with psycopg.connect(**db_params) as conn:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _adjust_one, client, dfu, forward[dfu],
                        actuals.get(dfu, []), meta.get(dfu, (None, None)),
                        customers.get(dfu), plan_month, guardrails, horizon_months,
                    ): dfu
                    for dfu in dfus
                }
                for fut in as_completed(futures):
                    rows, changed = fut.result()
                    n_adjusted += int(changed)
                    with conn.cursor() as cur:
                        _persist(cur, run_id, plan_version, rows)
                    conn.commit()
        succeeded = True
    finally:
        if not succeeded:
            with psycopg.connect(**db_params, autocommit=True) as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE ai_champion_run SET status='failed', completed_at=now(), error=%s WHERE run_id=%s",
                    ("pipeline error — see logs", run_id),
                )

    with psycopg.connect(**db_params, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE ai_champion_run SET status='succeeded', n_adjusted=%s, completed_at=now() WHERE run_id=%s",
            (n_adjusted, run_id),
        )

    logger.info("%s AI Champion run %s done — %d/%d DFUs adjusted", _ts(), run_id, n_adjusted, n_dfus)
    return {"run_id": run_id, "plan_version": plan_version, "provider": effective_provider,
            "model": model, "n_dfus": n_dfus, "n_adjusted": n_adjusted, "est_cost_usd": est_cost}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Generate the forward-only AI Champion forecast.")
    ap.add_argument("--provider", default=None, help="Override config provider (ollama|anthropic|openai|openai_compat)")
    ap.add_argument("--plan-version", default=None, help="Champion plan_version to adjust (default: latest)")
    ap.add_argument("--limit-dfus", type=int, default=None, help="Adjust only the first N DFUs (smoke runs)")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan + cost estimate; no LLM/DB writes")
    args = ap.parse_args()

    summary = run_pipeline(
        provider=args.provider,
        plan_version=args.plan_version,
        limit_dfus=args.limit_dfus,
        dry_run=args.dry_run,
    )
    logger.info("%s Summary: %s", _ts(), summary)


if __name__ == "__main__":
    main()
