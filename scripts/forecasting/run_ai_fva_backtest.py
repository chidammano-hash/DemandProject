"""AI Planner FVA Backtest — walk-forward runner.

Spec: docs/specs/PRD/PRD-ai-planner-fva-backtest.md

Walks `window_months` months back from `as_of_date`, and for each month T:
  1. Samples DFUs per the configured strategy (default: stratified, capped at 10K).
  2. Builds point-in-time context for each DFU (no future leakage).
  3. Calls the AI Planner via LLMClient for a forecast-adjustment recommendation.
  4. Applies guardrails and computes the AI-adjusted forecast.
  5. Persists recommendation + per-month baseline/AI quantities to fact tables.

After all months complete, backfills actuals to fact_ai_adjusted_forecast and
refreshes the FVA materialized views.

Usage:
    uv run python -m scripts.forecasting.run_ai_fva_backtest \\
        --window-months 10 --as-of-date 2026-04-01 --limit-dfus 100 [--dry-run]

Defaults to provider=ollama (free, local). Override via --provider anthropic etc.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import psycopg
from psycopg import sql as psycopg_sql

from common.ai.fva_recommender import (
    PROMPT_VERSION,
    CustomerHistory,
    DfuContext,
    apply_guardrails,
    apply_recommendation,
    recommend,
)
from common.ai.llm_client import LLMClientError, LLMJSONParseError, build_from_config
from common.core.db import get_db_params
from common.core.planning_date import get_planning_date
from common.core.utils import load_config

log = logging.getLogger(__name__)

# Failures a backtest run can realistically hit: DB errors, LLM transport/parse
# errors, and config/data shape errors. Caught to mark the run "failed" before
# re-raising; an unexpected type propagates loudly (a programming bug).
_RUN_ERRORS = (psycopg.Error, LLMClientError, LLMJSONParseError, ValueError, KeyError)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def month_floor(d: date) -> date:
    return d.replace(day=1)


def add_months(d: date, n: int) -> date:
    """Add n months to d (clamped to first-of-month)."""
    d = month_floor(d)
    total = d.year * 12 + (d.month - 1) + n
    return date(total // 12, total % 12 + 1, 1)


def walk_back_months(as_of: date, window: int) -> list[date]:
    """[as_of - window + 1, ..., as_of] in chronological order, all month-floor."""
    return [add_months(month_floor(as_of), -i) for i in range(window - 1, -1, -1)]


# ---------------------------------------------------------------------------
# DFU sampling (PRD §4.4)
# ---------------------------------------------------------------------------

def sample_dfus(
    conn,
    sample_cfg: dict,
    *,
    baseline_model_id: str,
    baseline_window_start: date,
    baseline_window_end: date,
    limit_override: int | None = None,
) -> list[tuple[str, str]]:
    """Return [(item_id, loc), ...] sampled per the configured strategy.

    Only DFUs with at least one baseline forecast row inside
    [baseline_window_start, baseline_window_end) are eligible — otherwise the
    walk-forward loop would skip them at every T. This narrows the population
    from all of dim_sku to the DFUs that actually have usable baseline coverage.

    Stratifies on (cluster_assignment, abc_vol) — the two persisted segment
    columns on dim_sku. `demand_pattern` exists only as an in-memory clustering
    label and is not currently stored on the dim, so it cannot be a stratum.
    """
    mode = sample_cfg.get("default_mode", "stratified")
    strat = sample_cfg.get("stratified", {})
    pct = strat.get("pct", 5)
    min_dfus = strat.get("min_dfus", 500)
    max_dfus = limit_override or strat.get("max_dfus", 10000)

    if mode != "stratified":
        log.warning("Sample mode %r not implemented — falling back to stratified", mode)

    sql = """
        WITH eligible AS (
            SELECT DISTINCT
                s.item_id, s.loc,
                COALESCE(s.cluster_assignment, '_none') AS cluster,
                COALESCE(s.abc_vol,            '_none') AS abc
            FROM dim_sku s
            JOIN fact_external_forecast_monthly f
              ON f.item_id  = s.item_id
             AND f.loc      = s.loc
             AND f.model_id = %s
             AND f.startdate >= %s
             AND f.startdate <  %s
        ), ranked AS (
            SELECT
                item_id, loc, cluster, abc,
                row_number() OVER (
                    PARTITION BY cluster, abc ORDER BY random()
                ) AS rn,
                count(*) OVER (
                    PARTITION BY cluster, abc
                ) AS bucket_size
            FROM eligible
        )
        SELECT item_id, loc
        FROM ranked
        WHERE rn <= GREATEST(1, ceil(bucket_size * %s / 100.0))
        ORDER BY random()
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (
            baseline_model_id, baseline_window_start, baseline_window_end,
            pct, max_dfus,
        ))
        rows = cur.fetchall()

    if len(rows) < min_dfus:
        log.warning("Sample size %d below floor %d — using as-is", len(rows), min_dfus)

    log.info("Sampled %d DFUs (mode=stratified, pct=%s, cap=%d)", len(rows), pct, max_dfus)
    return [(r[0], r[1]) for r in rows]


# ---------------------------------------------------------------------------
# Point-in-time context builder (PRD §4.3 — no leakage)
# ---------------------------------------------------------------------------

CUSTOMER_LOOKBACK_MONTHS = 12          # how far back we read per-customer history
CUSTOMER_TOP_K = 5                     # how many top customers to include in the prompt


def build_dfu_context(
    conn,
    item_id: str,
    loc: str,
    as_of: date,
    horizon: int,
    *,
    baseline_model_id: str = "external",
) -> DfuContext | None:
    """Fetch point-in-time context for one DFU at month T = as_of.

    Returns None if no baseline forecast snapshot exists for this DFU at T
    (the run logs and skips these per PRD §10 risk #4).
    """
    actuals_sql = """
        SELECT to_char(startdate, 'YYYY-MM') AS m, qty
        FROM fact_sales_monthly
        WHERE item_id = %s AND loc = %s
          AND startdate <  %s
          AND startdate >= %s
        ORDER BY startdate
    """
    horizon_start = add_months(as_of, -24)

    baseline_sql = """
        SELECT to_char(startdate, 'YYYY-MM') AS m, basefcst_pref
        FROM fact_external_forecast_monthly
        WHERE item_id = %s AND loc = %s
          AND model_id = %s
          AND startdate >= %s AND startdate < %s
        ORDER BY startdate
        LIMIT %s
    """
    baseline_start = as_of
    baseline_end = add_months(as_of, horizon)

    meta_sql = """
        SELECT cluster_assignment, abc_vol
        FROM dim_sku
        WHERE item_id = %s AND loc = %s
        LIMIT 1
    """
    # Top-K customers by SUM(sales_qty) over [T - CUSTOMER_LOOKBACK_MONTHS, T),
    # returned with per-month rows ordered by total DESC then startdate.
    # Strictly point-in-time: WHERE startdate < as_of.
    customers_sql = """
        WITH top_n AS (
            SELECT customer_no, SUM(sales_qty) AS total
            FROM fact_customer_demand_monthly
            WHERE item_id = %s AND location_id = %s
              AND startdate >= %s AND startdate < %s
            GROUP BY customer_no
            HAVING SUM(sales_qty) > 0
            ORDER BY total DESC NULLS LAST
            LIMIT %s
        ),
        names AS (
            SELECT customer_no, MIN(customer_name) AS customer_name
            FROM dim_customer
            GROUP BY customer_no
        )
        SELECT
            cd.customer_no,
            COALESCE(n.customer_name, '') AS customer_name,
            to_char(cd.startdate, 'YYYY-MM') AS m,
            cd.sales_qty,
            t.total
        FROM fact_customer_demand_monthly cd
        JOIN top_n t USING (customer_no)
        LEFT JOIN names n USING (customer_no)
        WHERE cd.item_id = %s AND cd.location_id = %s
          AND cd.startdate >= %s AND cd.startdate < %s
        ORDER BY t.total DESC, cd.startdate
    """
    customer_window_start = add_months(as_of, -CUSTOMER_LOOKBACK_MONTHS)

    with conn.cursor() as cur:
        cur.execute(actuals_sql, (item_id, loc, as_of, horizon_start))
        actuals = [(m, float(q or 0)) for m, q in cur.fetchall()]

        cur.execute(baseline_sql, (item_id, loc, baseline_model_id, baseline_start, baseline_end, horizon))
        baseline = [(m, float(q or 0)) for m, q in cur.fetchall()]

        if len(baseline) < horizon:
            return None  # no baseline snapshot at T

        cur.execute(meta_sql, (item_id, loc))
        meta_row = cur.fetchone() or (None, None)

        cur.execute(customers_sql, (
            item_id, loc, customer_window_start, as_of, CUSTOMER_TOP_K,
            item_id, loc, customer_window_start, as_of,
        ))
        customer_rows = cur.fetchall()

    top_customers = _group_customer_rows(customer_rows)

    return DfuContext(
        item_id=item_id,
        loc=loc,
        forecast_run_month=as_of,
        actuals_last_24m=actuals,
        baseline_forecast=baseline,
        cluster=meta_row[0],
        demand_pattern=None,
        abc_vol=meta_row[1],
        top_customers=top_customers or None,
    )


def _group_customer_rows(rows) -> list[CustomerHistory]:
    """Collapse SQL output (one row per customer-month) into one CustomerHistory
    per customer, preserving total-desc order from the SQL."""
    grouped: dict[str, CustomerHistory] = {}
    order: list[str] = []
    for cust_no, cust_name, month, qty, _total in rows:
        if cust_no not in grouped:
            grouped[cust_no] = CustomerHistory(
                customer_no=cust_no,
                customer_name=cust_name or None,
                monthly=[],
            )
            order.append(cust_no)
        grouped[cust_no].monthly.append((month, float(qty or 0)))
    return [grouped[c] for c in order]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def insert_run(conn, run: dict) -> None:
    sql = """
        INSERT INTO ai_fva_backtest_run (
            run_id, status, window_months, as_of_date, horizon_months,
            sample_strategy, provider, ai_model, prompt_version,
            apply_guardrails, estimated_cost_usd, created_by, notes
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (
            run["run_id"], "running", run["window_months"], run["as_of_date"],
            run["horizon_months"], json.dumps(run["sample_strategy"]),
            run["provider"], run["ai_model"], run["prompt_version"],
            json.dumps(run["apply_guardrails"]), run.get("estimated_cost_usd"),
            run.get("created_by"), run.get("notes"),
        ))
    conn.commit()


def finalize_run(conn, run_id: uuid.UUID, status: str, *,
                 n_dfus: int | None = None, n_recs: int | None = None,
                 actual_cost: float | None = None, error: str | None = None) -> None:
    sql = """
        UPDATE ai_fva_backtest_run
        SET status=%s, completed_at=now(),
            n_dfus_sampled=%s, n_recommendations=%s,
            actual_cost_usd=%s, error_message=%s
        WHERE run_id=%s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (status, n_dfus, n_recs, actual_cost, error, str(run_id)))
    conn.commit()


def insert_recommendation(conn, run_id: uuid.UUID, ctx: DfuContext, rec, resp) -> None:
    sql = """
        INSERT INTO fact_ai_forecast_recommendation (
            run_id, item_id, loc, forecast_run_month,
            recommendation_code, pct_change, proposed_qty,
            apply_horizon_months, confidence, rationale, evidence_keys,
            ai_call_ms, ai_tokens_in, ai_tokens_out
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT DO NOTHING
    """
    with conn.cursor() as cur:
        cur.execute(sql, (
            str(run_id), ctx.item_id, ctx.loc, ctx.forecast_run_month,
            rec.recommendation_code, rec.pct_change,
            json.dumps(rec.proposed_qty) if rec.proposed_qty else None,
            rec.apply_horizon_months, rec.confidence, rec.rationale,
            json.dumps(rec.evidence_keys),
            resp.elapsed_ms, resp.tokens_in, resp.tokens_out,
        ))


def insert_adjusted_forecasts(
    conn, run_id: uuid.UUID, ctx: DfuContext,
    baseline: list[tuple[str, float]], ai: list[tuple[str, float]],
) -> None:
    sql = """
        INSERT INTO fact_ai_adjusted_forecast (
            run_id, item_id, loc, forecast_run_month,
            target_month, lag, baseline_qty, ai_qty
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT DO NOTHING
    """
    rows = []
    for lag_idx, ((m_b, q_b), (_m_ai, q_ai)) in enumerate(zip(baseline, ai, strict=False), start=1):
        target_month = date.fromisoformat(f"{m_b}-01")
        rows.append((
            str(run_id), ctx.item_id, ctx.loc, ctx.forecast_run_month,
            target_month, lag_idx, q_b, q_ai,
        ))
    with conn.cursor() as cur:
        cur.executemany(sql, rows)


def insert_audit(conn, run_id: uuid.UUID, ctx: DfuContext, response_raw: dict) -> None:
    sql = """
        INSERT INTO ai_planner_backtest_audit (
            run_id, item_id, loc, forecast_run_month,
            context_payload, ai_response_raw
        ) VALUES (%s,%s,%s,%s,%s,%s)
        ON CONFLICT DO NOTHING
    """
    with conn.cursor() as cur:
        cur.execute(sql, (
            str(run_id), ctx.item_id, ctx.loc, ctx.forecast_run_month,
            json.dumps(ctx.to_dict()), json.dumps(response_raw),
        ))


def backfill_actuals(conn, run_id: uuid.UUID) -> int:
    """Update fact_ai_adjusted_forecast.actual_qty from fact_sales_monthly."""
    sql = """
        UPDATE fact_ai_adjusted_forecast f
        SET actual_qty = s.qty
        FROM fact_sales_monthly s
        WHERE f.run_id = %s
          AND f.item_id = s.item_id
          AND f.loc     = s.loc
          AND f.target_month = s.startdate
    """
    with conn.cursor() as cur:
        cur.execute(sql, (str(run_id),))
        n = cur.rowcount
    conn.commit()
    log.info("Backfilled %d actuals into fact_ai_adjusted_forecast", n)
    return n


def _process_dfu(client, ctx, guardrails, horizon):
    """LLM call + guardrails + apply recommendation. NO DB ops (thread-safe).

    Returns (ctx, rec, resp, ai_forecast) on success, or None on parse/LLM error.
    Runs in worker threads — psycopg connections must NOT be touched here.
    """
    try:
        rec, resp = recommend(client, ctx)
    except (LLMJSONParseError, LLMClientError) as exc:
        log.warning("AI call failed for %s@%s @ %s: %s",
                    ctx.item_id, ctx.loc, ctx.forecast_run_month, exc)
        return None
    rec = apply_guardrails(rec, guardrails, horizon)
    ai_forecast = apply_recommendation(rec, ctx.baseline_forecast)
    return (ctx, rec, resp, ai_forecast)


def refresh_mvs(conn) -> None:
    for mv in ("mv_ai_fva_overall", "mv_ai_fva_by_recommendation",
               "mv_ai_fva_by_month", "mv_ai_fva_by_dfu"):
        with conn.cursor() as cur:
            cur.execute(
                psycopg_sql.SQL("REFRESH MATERIALIZED VIEW {}").format(
                    psycopg_sql.Identifier(mv)
                )
            )
        log.info("Refreshed %s", mv)
    conn.commit()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_backtest(args, config: dict) -> uuid.UUID:
    run_id = uuid.uuid4()
    provider = args.provider or config.get("provider", "ollama")
    model = config["models"][provider]
    horizon = args.horizon_months or config["defaults"]["horizon_months"]
    window = args.window_months or config["defaults"]["window_months"]
    as_of = args.as_of_date or add_months(get_planning_date(),
                                           -config["defaults"]["as_of_offset_months"])
    as_of = month_floor(as_of)
    baseline_model_id = args.baseline_model_id or config["defaults"].get("baseline_model_id", "external")

    estimated_cost = None
    if not args.dry_run:
        per_call = config["cost_controls"]["per_call_estimated_cost_usd"].get(provider, 0.0)

    log.info("Starting AI FVA backtest: run_id=%s provider=%s model=%s window=%d as_of=%s horizon=%d baseline=%s",
             run_id, provider, model, window, as_of, horizon, baseline_model_id)

    client = build_from_config(config, override_provider=provider)
    guardrails = config.get("apply_guardrails", {})

    # Baseline coverage window for the whole run: from the earliest T to the
    # latest T + horizon (exclusive). Used to filter sample_dfus to DFUs with
    # at least one baseline row inside this window.
    months = walk_back_months(as_of, window)
    baseline_window_start = months[0]
    baseline_window_end = add_months(months[-1], horizon)

    with psycopg.connect(**get_db_params()) as conn:
        # Sample DFUs once for the whole run (stable across months).
        dfus = sample_dfus(
            conn, config.get("sampling", {}),
            baseline_model_id=baseline_model_id,
            baseline_window_start=baseline_window_start,
            baseline_window_end=baseline_window_end,
            limit_override=args.limit_dfus,
        )
        if not dfus:
            raise RuntimeError("No DFUs sampled — check dim_sku is populated.")

        per_call = config["cost_controls"]["per_call_estimated_cost_usd"].get(provider, 0.0)
        estimated_cost = round(per_call * len(dfus) * window, 2)
        log.info("Estimated cost: $%.2f (%d DFUs x %d months @ $%.4f/call, provider=%s)",
                 estimated_cost, len(dfus), window, per_call, provider)
        cap = config["cost_controls"].get("per_run_max_cost_usd", 1500)
        if provider != "ollama" and estimated_cost > cap:
            raise RuntimeError(f"Estimated cost ${estimated_cost} exceeds cap ${cap}; aborting.")

        if args.dry_run:
            log.info("--dry-run: skipping LLM calls and DB writes. Exiting.")
            return run_id

        insert_run(conn, {
            "run_id": str(run_id), "window_months": window, "as_of_date": as_of,
            "horizon_months": horizon, "sample_strategy": config.get("sampling", {}),
            "provider": provider, "ai_model": model, "prompt_version": PROMPT_VERSION,
            "apply_guardrails": guardrails, "estimated_cost_usd": estimated_cost,
            "created_by": "cli", "notes": args.notes,
        })

        max_workers = config["cost_controls"].get("max_concurrent_ai_calls", 4)
        log.info("Concurrency: %d worker threads for LLM calls", max_workers)

        n_recs = 0
        try:
            for month_t in walk_back_months(as_of, window):
                log.info("--- Walk-forward month T = %s ---", month_t)

                # Phase 1: build all DfuContexts on the main thread (DB-bound, fast).
                contexts = []
                for item_id, loc in dfus:
                    ctx = build_dfu_context(
                        conn, item_id, loc, month_t, horizon,
                        baseline_model_id=baseline_model_id,
                    )
                    if ctx is not None:
                        contexts.append(ctx)
                log.info("Built %d DFU contexts (skipped %d with no baseline snapshot)",
                         len(contexts), len(dfus) - len(contexts))

                # Phase 2: parallelize LLM calls; persist results on the main thread
                # as each completes (psycopg connection is NOT thread-safe).
                with ThreadPoolExecutor(max_workers=max_workers,
                                         thread_name_prefix="ai-fva") as ex:
                    futures = [
                        ex.submit(_process_dfu, client, c, guardrails, horizon)
                        for c in contexts
                    ]
                    for fut in as_completed(futures):
                        result = fut.result()
                        if result is None:
                            continue
                        ctx, rec, resp, ai_forecast = result
                        insert_recommendation(conn, run_id, ctx, rec, resp)
                        insert_adjusted_forecasts(conn, run_id, ctx,
                                                   ctx.baseline_forecast, ai_forecast)
                        insert_audit(conn, run_id, ctx, resp.raw)
                        n_recs += 1

                conn.commit()
                log.info("Month %s complete; cumulative recs=%d", month_t, n_recs)

            backfill_actuals(conn, run_id)
            if not args.skip_mvs:
                refresh_mvs(conn)
            finalize_run(conn, run_id, "succeeded",
                         n_dfus=len(dfus), n_recs=n_recs,
                         actual_cost=0.0 if provider == "ollama" else estimated_cost)
            log.info("Backtest complete: run_id=%s recs=%d", run_id, n_recs)

        except _RUN_ERRORS as exc:
            log.exception("Backtest failed mid-run")
            finalize_run(conn, run_id, "failed", error=str(exc))
            raise

    return run_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--window-months", type=int, default=None,
                   help="Walk-forward window (default from config)")
    p.add_argument("--as-of-date", type=date.fromisoformat, default=None,
                   help="End-of-window month (YYYY-MM-DD; defaults to last completed month)")
    p.add_argument("--horizon-months", type=int, default=None,
                   help="Forecast horizon to evaluate (default from config)")
    p.add_argument("--provider", type=str, default=None,
                   choices=["ollama", "anthropic", "openai", "openai_compat"],
                   help="Override LLM provider (default from config)")
    p.add_argument("--limit-dfus", type=int, default=None,
                   help="Hard DFU cap for smoke tests (overrides sample max_dfus)")
    p.add_argument("--baseline-model-id", type=str, default=None,
                   help="model_id in fact_external_forecast_monthly used as baseline (default 'external')")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan + cost estimate; skip LLM and DB writes")
    p.add_argument("--skip-mvs", action="store_true",
                   help="Skip MV refresh at end (faster smoke test)")
    p.add_argument("--notes", type=str, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    config = load_config("ai_planner_fva_backtest_config")
    try:
        run_id = run_backtest(args, config)
        log.info("Run ID: %s", run_id)
        return 0
    except _RUN_ERRORS:
        log.exception("Backtest failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
