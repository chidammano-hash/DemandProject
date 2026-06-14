"""AI Planner FVA Backtest API router.

Spec: docs/specs/PRD/PRD-ai-planner-fva-backtest.md (§6 API Surface)

Endpoints under /ai-planner/fva-backtest/* — minimal MVP set:
  POST   /runs                        kick off a new backtest (background thread)
  GET    /runs                        list runs
  GET    /runs/{run_id}               run metadata + status
  GET    /runs/{run_id}/summary       overall lift, baseline vs AI WAPE, win rate
  GET    /runs/{run_id}/by-recommendation   per-recommendation-code rollup
  GET    /runs/{run_id}/by-month      month-by-month FVA over the window
  GET    /runs/{run_id}/dfus          per-DFU drill-down (paginated)

Per CLAUDE.md: get_conn() (NOT Depends(_get_pool)), %s placeholders, write
endpoints guarded by Depends(require_api_key), Pydantic v2.
"""
from __future__ import annotations

import logging
import threading
import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from api.auth import require_api_key
from api.core import get_conn

log = logging.getLogger(__name__)
router = APIRouter(prefix="/ai-planner/fva-backtest", tags=["ai-planner", "fva-backtest"])


# ---------------------------------------------------------------------------
# Pydantic v2 models
# ---------------------------------------------------------------------------

class StartRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_months: int | None = Field(default=None, ge=1, le=36)
    as_of_date: date | None = None
    horizon_months: int | None = Field(default=None, ge=1, le=12)
    provider: str | None = Field(default=None, pattern="^(ollama|anthropic|openai|openai_compat)$")
    limit_dfus: int | None = Field(default=None, ge=1, le=50_000)
    notes: str | None = Field(default=None, max_length=1000)


class RunSummary(BaseModel):
    run_id: str
    status: str
    started_at: str | None
    completed_at: str | None
    window_months: int
    as_of_date: str
    horizon_months: int
    provider: str
    ai_model: str
    n_dfus_sampled: int | None
    n_recommendations: int | None
    estimated_cost_usd: float | None
    actual_cost_usd: float | None
    error_message: str | None


class DfuDetailLag(BaseModel):
    forecast_run_month: str
    target_month: str
    lag: int
    baseline_qty: float | None
    ai_qty: float | None
    actual_qty: float | None


class DfuDetailRecommendation(BaseModel):
    forecast_run_month: str
    recommendation_code: str
    pct_change: float | None
    confidence: float | None
    rationale: str | None
    evidence_keys: list[str] | None


class DfuDetailSummary(BaseModel):
    n_obs: int
    baseline_wape_pct: float | None
    ai_wape_pct: float | None
    lift_pp: float | None


class DfuDetailResponse(BaseModel):
    run_id: str
    item_id: str
    loc: str
    summary: DfuDetailSummary
    lags: list[DfuDetailLag]
    recommendations: list[DfuDetailRecommendation]


# ---------------------------------------------------------------------------
# Background launcher
# ---------------------------------------------------------------------------

def _launch_backtest_thread(req: StartRunRequest) -> None:
    """Fire the backtest runner in a daemon thread.

    For long jobs this should move to pg-queue (per CLAUDE.md memory note).
    For MVP, in-process thread keeps the dependency surface minimal.
    """

    from scripts.forecasting.run_ai_fva_backtest import main as runner_main

    argv: list[str] = []
    if req.window_months is not None:
        argv += ["--window-months", str(req.window_months)]
    if req.as_of_date is not None:
        argv += ["--as-of-date", req.as_of_date.isoformat()]
    if req.horizon_months is not None:
        argv += ["--horizon-months", str(req.horizon_months)]
    if req.provider is not None:
        argv += ["--provider", req.provider]
    if req.limit_dfus is not None:
        argv += ["--limit-dfus", str(req.limit_dfus)]
    if req.notes is not None:
        argv += ["--notes", req.notes]

    def _runner():
        # runner_main handles its own exceptions and returns an exit code;
        # surface a non-zero status so a failed run is visible in the API logs.
        if runner_main(argv) != 0:
            log.error("AI FVA backtest background thread exited with a non-zero status")

    t = threading.Thread(target=_runner, name="ai-fva-backtest", daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# POST /runs — start a new run
# ---------------------------------------------------------------------------

@router.post("/runs", status_code=202, dependencies=[Depends(require_api_key)])
async def start_run(req: StartRunRequest):
    """Start a new AI FVA backtest. Returns immediately; runs in background."""
    _launch_backtest_thread(req)
    return {
        "status": "accepted",
        "message": "Backtest started in background. Poll GET /ai-planner/fva-backtest/runs to find run_id.",
    }


# ---------------------------------------------------------------------------
# GET /runs — list runs
# ---------------------------------------------------------------------------

@router.get("/runs")
async def list_runs(
    limit: int = Query(50, ge=1, le=500),
    status: str | None = Query(None),
):
    # Build query conditionally — psycopg3 can't infer the type of a bare
    # `%s IS NULL` placeholder without an explicit ::text cast, so we just
    # omit the filter when status is None.
    base_sql = """
        SELECT run_id::text, status,
               to_char(started_at, 'YYYY-MM-DD"T"HH24:MI:SSOF') AS started_at,
               to_char(completed_at, 'YYYY-MM-DD"T"HH24:MI:SSOF') AS completed_at,
               window_months, as_of_date::text, horizon_months,
               provider, ai_model,
               n_dfus_sampled, n_recommendations,
               estimated_cost_usd, actual_cost_usd, error_message
        FROM ai_fva_backtest_run
    """
    if status is None:
        sql = base_sql + " ORDER BY started_at DESC LIMIT %s"
        params: tuple = (limit,)
    else:
        sql = base_sql + " WHERE status = %s ORDER BY started_at DESC LIMIT %s"
        params = (status, limit)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]
    return {"runs": rows, "count": len(rows)}


# ---------------------------------------------------------------------------
# GET /runs/{run_id}
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}", response_model=RunSummary)
async def get_run(run_id: str):
    sql = """
        SELECT run_id::text, status,
               to_char(started_at, 'YYYY-MM-DD"T"HH24:MI:SSOF'),
               to_char(completed_at, 'YYYY-MM-DD"T"HH24:MI:SSOF'),
               window_months, as_of_date::text, horizon_months,
               provider, ai_model,
               n_dfus_sampled, n_recommendations,
               estimated_cost_usd, actual_cost_usd, error_message
        FROM ai_fva_backtest_run WHERE run_id = %s
    """
    try:
        uuid.UUID(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid run_id (must be a UUID).") from exc
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (run_id,))
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return RunSummary(
        run_id=row[0], status=row[1], started_at=row[2], completed_at=row[3],
        window_months=row[4], as_of_date=row[5], horizon_months=row[6],
        provider=row[7], ai_model=row[8],
        n_dfus_sampled=row[9], n_recommendations=row[10],
        estimated_cost_usd=float(row[11]) if row[11] is not None else None,
        actual_cost_usd=float(row[12]) if row[12] is not None else None,
        error_message=row[13],
    )


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/summary — the headline FVA number
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}/summary")
async def run_summary(run_id: str):
    """Headline FVA: baseline WAPE, AI WAPE, lift, win rate."""
    sql = """
        SELECT baseline_wape_pct, ai_wape_pct, lift_pct,
               n_dfus, n_winners, n_losers, n_ties, win_rate_pct
        FROM mv_ai_fva_overall WHERE run_id = %s
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (run_id,))
        row = cur.fetchone()
    if row is None:
        return {"run_id": run_id, "summary": None,
                "message": "No FVA summary yet. Backfill actuals + refresh MVs."}
    return {
        "run_id": run_id,
        "baseline_wape_pct": float(row[0]) if row[0] is not None else None,
        "ai_wape_pct":       float(row[1]) if row[1] is not None else None,
        "lift_pct":          float(row[2]) if row[2] is not None else None,
        "n_dfus":            row[3],
        "n_winners":         row[4],
        "n_losers":          row[5],
        "n_ties":            row[6],
        "win_rate_pct":      float(row[7]) if row[7] is not None else None,
    }


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/by-recommendation
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}/by-recommendation")
async def by_recommendation(run_id: str):
    sql = """
        SELECT recommendation_code, baseline_wape_pct, ai_wape_pct, lift_pct,
               n_obs, avg_confidence, avg_pct_change
        FROM mv_ai_fva_by_recommendation
        WHERE run_id = %s
        ORDER BY n_obs DESC
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (run_id,))
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]
    return {"run_id": run_id, "rows": rows}


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/by-month
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}/by-month")
async def by_month(run_id: str):
    sql = """
        SELECT forecast_run_month::text, baseline_wape_pct, ai_wape_pct, n_dfus
        FROM mv_ai_fva_by_month
        WHERE run_id = %s
        ORDER BY forecast_run_month
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (run_id,))
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]
    return {"run_id": run_id, "rows": rows}


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/dfus — per-DFU drill-down
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}/dfus")
async def dfus_for_run(
    run_id: str,
    limit: int = Query(100, ge=1, le=2000),
    sort: str = Query("error_reduction", pattern="^(error_reduction|item_id)$"),
):
    order_by = "abs_error_reduction DESC" if sort == "error_reduction" else "item_id ASC, loc ASC"
    sql = f"""
        SELECT item_id, loc, sae_baseline, sae_ai,
               abs_error_reduction, n_obs
        FROM mv_ai_fva_by_dfu
        WHERE run_id = %s
        ORDER BY {order_by}
        LIMIT %s
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (run_id, limit))
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]
    return {"run_id": run_id, "rows": rows, "count": len(rows)}


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/dfu-detail — per-DFU walk-forward detail + recommendations
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}/dfu-detail", response_model=DfuDetailResponse)
async def dfu_detail(
    run_id: str,
    item_id: str = Query(..., min_length=1),
    loc: str = Query(..., min_length=1),
):
    """Walk-forward detail for one DFU: each (forecast_run_month, target_month,
    lag) with baseline vs AI vs actual, plus the AI recommendation at each T.

    Powers the UI drill-in that lets a planner inspect *what* the AI did for a
    specific item/location and *how much* it helped vs. the baseline.
    """
    try:
        uuid.UUID(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid run_id (must be a UUID).") from exc

    lags_sql = """
        SELECT
            forecast_run_month::text, target_month::text, lag,
            baseline_qty, ai_qty, actual_qty
        FROM fact_ai_adjusted_forecast
        WHERE run_id = %s AND item_id = %s AND loc = %s
        ORDER BY forecast_run_month, lag
    """
    recs_sql = """
        SELECT
            forecast_run_month::text, recommendation_code, pct_change,
            confidence, rationale, evidence_keys
        FROM fact_ai_forecast_recommendation
        WHERE run_id = %s AND item_id = %s AND loc = %s
        ORDER BY forecast_run_month
    """
    # WAPE in the accuracy form (higher = better) — matches sql/186 MVs.
    summary_sql = """
        SELECT
            COUNT(*) FILTER (WHERE actual_qty IS NOT NULL)                                  AS n_obs,
            COALESCE(SUM(ABS(COALESCE(actual_qty,0)))     FILTER (WHERE actual_qty IS NOT NULL), 0) AS abs_actual,
            COALESCE(SUM(ABS(baseline_qty - actual_qty))  FILTER (WHERE actual_qty IS NOT NULL), 0) AS sae_baseline,
            COALESCE(SUM(ABS(ai_qty       - actual_qty))  FILTER (WHERE actual_qty IS NOT NULL), 0) AS sae_ai
        FROM fact_ai_adjusted_forecast
        WHERE run_id = %s AND item_id = %s AND loc = %s
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(lags_sql, (run_id, item_id, loc))
        lag_rows = cur.fetchall()
        cur.execute(recs_sql, (run_id, item_id, loc))
        rec_rows = cur.fetchall()
        cur.execute(summary_sql, (run_id, item_id, loc))
        srow = cur.fetchone()

    if not lag_rows and not rec_rows:
        raise HTTPException(status_code=404, detail="DFU not found in run.")

    n_obs = int(srow[0]) if srow and srow[0] is not None else 0
    abs_actual = float(srow[1]) if srow and srow[1] is not None else 0.0
    sae_base = float(srow[2]) if srow and srow[2] is not None else 0.0
    sae_ai = float(srow[3]) if srow and srow[3] is not None else 0.0
    base_wape = 100.0 - 100.0 * sae_base / abs_actual if abs_actual > 0 else None
    ai_wape = 100.0 - 100.0 * sae_ai / abs_actual if abs_actual > 0 else None
    lift = (ai_wape - base_wape) if (base_wape is not None and ai_wape is not None) else None

    return DfuDetailResponse(
        run_id=run_id, item_id=item_id, loc=loc,
        summary=DfuDetailSummary(
            n_obs=n_obs,
            baseline_wape_pct=base_wape,
            ai_wape_pct=ai_wape,
            lift_pp=lift,
        ),
        lags=[
            DfuDetailLag(
                forecast_run_month=r[0], target_month=r[1], lag=int(r[2]),
                baseline_qty=float(r[3]) if r[3] is not None else None,
                ai_qty=float(r[4]) if r[4] is not None else None,
                actual_qty=float(r[5]) if r[5] is not None else None,
            ) for r in lag_rows
        ],
        recommendations=[
            DfuDetailRecommendation(
                forecast_run_month=r[0],
                recommendation_code=r[1],
                pct_change=float(r[2]) if r[2] is not None else None,
                confidence=float(r[3]) if r[3] is not None else None,
                rationale=r[4],
                evidence_keys=r[5] if isinstance(r[5], list) else None,
            ) for r in rec_rows
        ],
    )


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/report.html — printable FVA report (browser print-to-PDF)
# ---------------------------------------------------------------------------

def _fmt(n, suffix: str = "", places: int = 2) -> str:
    if n is None:
        return "—"
    try:
        return f"{float(n):.{places}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


@router.get("/runs/{run_id}/report.html", response_class=HTMLResponse)
async def run_report_html(run_id: str):
    """Printable HTML report — browsers can print this to PDF.

    Avoids a server-side PDF dependency (reportlab/weasyprint). Frontend can
    call window.print() or open in a new tab and print from there.
    """
    try:
        uuid.UUID(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid run_id (must be a UUID).") from exc

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT run_id::text, status, started_at::text, completed_at::text,
                   window_months, as_of_date::text, horizon_months,
                   provider, ai_model, n_dfus_sampled, n_recommendations,
                   actual_cost_usd, notes
            FROM ai_fva_backtest_run WHERE run_id = %s
        """, (run_id,))
        meta = cur.fetchone()
        if meta is None:
            raise HTTPException(status_code=404, detail="Run not found.")

        cur.execute("""
            SELECT baseline_wape_pct, ai_wape_pct, lift_pct,
                   n_dfus, n_winners, n_losers, n_ties, win_rate_pct
            FROM mv_ai_fva_overall WHERE run_id = %s
        """, (run_id,))
        overall = cur.fetchone()

        cur.execute("""
            SELECT recommendation_code, baseline_wape_pct, ai_wape_pct, lift_pct,
                   n_obs, avg_confidence, avg_pct_change
            FROM mv_ai_fva_by_recommendation
            WHERE run_id = %s ORDER BY n_obs DESC
        """, (run_id,))
        by_rec = cur.fetchall()

        cur.execute("""
            SELECT forecast_run_month::text, baseline_wape_pct, ai_wape_pct, n_dfus
            FROM mv_ai_fva_by_month WHERE run_id = %s
            ORDER BY forecast_run_month
        """, (run_id,))
        by_month = cur.fetchall()

    overall_html = (
        f"<p>Baseline WAPE: <strong>{_fmt(overall[0], '%')}</strong> · "
        f"AI WAPE: <strong>{_fmt(overall[1], '%')}</strong> · "
        f"<span class='lift {'positive' if (overall[2] or 0) > 0 else 'negative'}'>"
        f"Lift: <strong>{_fmt(overall[2], 'pp')}</strong></span></p>"
        f"<p>{overall[3] or 0} DFUs evaluated · "
        f"{overall[4] or 0} winners / {overall[5] or 0} losers / {overall[6] or 0} ties · "
        f"Win rate: <strong>{_fmt(overall[7], '%')}</strong></p>"
        if overall else "<p>No FVA summary yet — backfill actuals + refresh MVs.</p>"
    )

    by_rec_rows = "".join(
        f"<tr><td>{r[0]}</td><td>{_fmt(r[1], '%')}</td><td>{_fmt(r[2], '%')}</td>"
        f"<td>{_fmt(r[3], 'pp')}</td><td>{r[4]}</td><td>{_fmt(r[5], '', 3)}</td>"
        f"<td>{_fmt(r[6], '%')}</td></tr>"
        for r in by_rec
    )
    by_month_rows = "".join(
        f"<tr><td>{r[0]}</td><td>{_fmt(r[1], '%')}</td><td>{_fmt(r[2], '%')}</td><td>{r[3]}</td></tr>"
        for r in by_month
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI Planner FVA Backtest Report</title>
<style>
  @page {{ size: letter; margin: 0.6in; }}
  body {{ font-family: -apple-system,Helvetica,Arial,sans-serif; color:#1a1a1a; max-width:7.5in; margin:0 auto; line-height:1.4; }}
  h1 {{ font-size:18pt; border-bottom:2px solid #1a3a5c; padding-bottom:6px; margin-bottom:4px; }}
  h2 {{ font-size:13pt; color:#1a3a5c; margin-top:20px; border-bottom:1px solid #ccc; padding-bottom:2px; }}
  table {{ width:100%; border-collapse:collapse; font-size:10pt; margin-top:6px; }}
  th, td {{ border:1px solid #ddd; padding:4px 8px; text-align:left; }}
  th {{ background:#f4f6f9; }}
  .meta {{ font-size:9pt; color:#555; }}
  .lift.positive {{ color:#15803d; }}
  .lift.negative {{ color:#b91c1c; }}
  .print-btn {{ background:#1a3a5c; color:#fff; padding:6px 12px; border:none; border-radius:4px; cursor:pointer; }}
  @media print {{ .no-print {{ display:none; }} }}
</style>
</head>
<body>
  <div class="no-print" style="text-align:right; margin-bottom:10px;">
    <button class="print-btn" onclick="window.print()">Print / Save as PDF</button>
  </div>
  <h1>AI Planner FVA Backtest Report</h1>
  <p class="meta">
    Run ID: <code>{meta[0]}</code> · Status: {meta[1]} ·
    Started: {meta[2] or '—'} · Completed: {meta[3] or '—'}
  </p>
  <p class="meta">
    Window: {meta[4]} months ending {meta[5]} · Horizon: {meta[6]} months ·
    Provider: {meta[7]}/{meta[8]} ·
    DFUs: {meta[9] or '—'} · Recommendations: {meta[10] or '—'} ·
    Spend: ${_fmt(meta[11], '', 2)}
  </p>

  <h2>Headline Result</h2>
  {overall_html}

  <h2>FVA by Recommendation Type</h2>
  <table>
    <thead><tr><th>Code</th><th>Baseline WAPE</th><th>AI WAPE</th><th>Lift</th><th>Obs</th><th>Avg Conf.</th><th>Avg %Δ</th></tr></thead>
    <tbody>{by_rec_rows or "<tr><td colspan='7'>No data yet.</td></tr>"}</tbody>
  </table>

  <h2>FVA by Month (Walk-Forward)</h2>
  <table>
    <thead><tr><th>Month T</th><th>Baseline WAPE</th><th>AI WAPE</th><th>DFUs</th></tr></thead>
    <tbody>{by_month_rows or "<tr><td colspan='4'>No data yet.</td></tr>"}</tbody>
  </table>

  <p class="meta" style="margin-top:24px; color:#888;">
    WAPE = 100 - 100 * sum(|F-A|) / |sum(A)| (accuracy form: higher = better). Lift = AI WAPE - Baseline WAPE (positive = AI improved).
    Spec: docs/specs/PRD/PRD-ai-planner-fva-backtest.md
  </p>
</body>
</html>"""
    return HTMLResponse(content=html)
