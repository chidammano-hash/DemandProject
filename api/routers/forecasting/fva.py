"""Forecast Value Added (FVA) & ROI tracking endpoints (Spec 08-07)."""

from __future__ import annotations

import logging
from datetime import date

import psycopg
from fastapi import APIRouter, HTTPException, Query

from api.core import compute_kpis, get_conn, get_read_only_conn
from common.core.planning_date import get_planning_date
from common.core.utils import get_algorithm_roster
from common.services.cache import cached_sync

router = APIRouter(prefix="/fva", tags=["fva"])
logger = logging.getLogger(__name__)


# Each tuple: (stage_id, label, description, default_state, missing_state).
# ``default_state`` is "planned" for stages that are always reserved (no source
# data yet). ``missing_state`` is the state a normally-measured ("actual") stage
# degrades to when its source yields no rows:
#   - external degrades to "missing" (genuinely no forecast → honest blank).
#   - champion is measured from the promoted champion-selection experiment's
#     backtest accuracy (``champion_experiment_month``, a stored 100 - WAPE over
#     per-DFU champion-selected forecasts at execution lag, windowed by months) —
#     NOT from fact_external_forecast_monthly (which only ever holds
#     model_id='external') nor fact_production_forecast (forward-only, no
#     measurable actual overlap). It degrades to "planned" (not "missing") when
#     no promoted experiment exists, so it presents consistently with the
#     AI/Planner reserved stages on a fresh DB.
STAGE_DEFS = [
    (
        "external",
        "External",
        "Current ERP or external forecast before model selection.",
        "actual",
        "missing",
    ),
    (
        "champion",
        "Champion",
        "Best measured statistical or ML model once champion-vs-actual outcomes are available.",
        "actual",
        "planned",
    ),
    (
        "ai_adjusted",
        "AI Adjusted",
        "Reserved for AI-assisted forecast interventions once they are measured.",
        "planned",
        "planned",
    ),
    (
        "planner_adjusted",
        "Planner Adjusted",
        "Reserved for human overrides once measured outcomes are available.",
        "planned",
        "planned",
    ),
]


def _round_or_none(value: float | None, digits: int = 2) -> float | None:
    return round(float(value), digits) if value is not None else None


def _build_stage(
    stage_id: str,
    label: str,
    description: str,
    default_state: str,
    missing_state: str,
    model: dict | None,
) -> dict:
    if default_state == "planned":
        state, accuracy, n_rows = "planned", None, 0
    elif model is None:
        state, accuracy, n_rows = missing_state, None, 0
    else:
        state, accuracy, n_rows = "actual", model["accuracy_pct"], model["n_rows"]
    return {
        "stage_id": stage_id,
        "label": label,
        "description": description,
        "state": state,
        "accuracy_pct": accuracy,
        "delta_vs_prev": None,
        "n_rows": n_rows,
    }


@router.get("/waterfall")
async def fva_waterfall(
    months: int = Query(12, ge=1, le=36),
):
    """FVA ladder data: external -> champion -> future adjustment stages.

    ``months`` windows every measured stage to the trailing horizon. Accuracy is
    always measured at each DFU's execution lag (the horizon it is operationally
    forecast at).
    """
    # Shared DFU-month filter: execution-lag rows within the requested horizon.
    # See docs/specs/01-foundation/06-execution-lag.md for the canonical pattern.
    # F9.1: anchor the window to the planning date, NOT the DB wall-clock
    # ``current_date`` — the demo forecast horizon trails the system clock, so a
    # ``current_date``-anchored 3-month window matches zero rows and blanks the
    # ladder. The planning date is bound as a parameter (``%s::date``), in line
    # with sibling forecasting routers (production_forecast.py, consensus_plan.py).
    planning_dt = get_planning_date()
    # dim_sku is keyed by (item_id, customer_group, loc); customer_group is
    # non-unique per (item_id, loc), so omitting it from the join fans out every
    # forecast row across customer_groups and corrupts the volume-weighted
    # WAPE/FVA numbers. Join on the full DFU grain (matches accuracy.py /
    # accuracy_budget.py); fact_external_forecast_monthly.customer_group is NOT NULL.
    dfu_filter = (
        "FROM fact_external_forecast_monthly f "
        "JOIN dim_sku d ON d.item_id = f.item_id "
        "AND d.customer_group = f.customer_group AND d.loc = f.loc "
        "WHERE f.startdate >= %s::date - (%s * interval '1 month') "
        "AND f.lag = COALESCE(d.execution_lag, 0) "
        "AND f.tothist_dmd IS NOT NULL"
    )
    # Each query embedding ``dfu_filter`` must supply (planning_dt, months) in order.
    dfu_params = (planning_dt, months)
    with get_conn() as conn, conn.cursor() as cur:
        # Accuracy per model_id at each DFU's execution lag (planning-relevant horizon).
        cur.execute(
            f"""SELECT f.model_id,
                       100.0 - (100.0 * sum(abs(f.basefcst_pref - f.tothist_dmd)) / NULLIF(abs(sum(f.tothist_dmd)), 0)) AS accuracy_pct,
                       count(*) AS n_rows
                {dfu_filter}
                GROUP BY f.model_id
                ORDER BY accuracy_pct DESC NULLS LAST""",
            dfu_params,
        )
        rows = cur.fetchall()

    models = {
        r[0]: {"model_id": r[0], "accuracy_pct": _round_or_none(r[1]), "n_rows": r[2]} for r in rows
    }

    # `ai_adjusted` and `planner_adjusted` remain reserved ("planned") stages in
    # the ladder — there is no measured AI accuracy to source. The forward-only
    # AI Champion adjuster (docs/specs/02-forecasting/27-ai-champion-forecast.md)
    # writes a forward forecast (model_id='ai_champion') with no historical
    # actual overlap, so it cannot feed this accuracy waterfall.

    # Promote `champion` and `ceiling` from the promoted champion-selection
    # experiment, windowed by `months` over the per-month rollup
    # (champion_experiment_month), volume-weighted by n_champions so champion and
    # ceiling track the selector like external/naive. champion_accuracy /
    # ceiling_accuracy are stored as 100 - WAPE over the per-DFU champion-selected
    # (and oracle-best) backtest forecasts at execution lag — uniform with the
    # rest of this endpoint. See scripts/ml/run_champion_experiment.py and
    # docs/specs/02-forecasting/07-champion-selection.md. Returns a
    # (champion_accuracy, ceiling_accuracy, n) triple.
    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute(
                """SELECT
                      SUM(m.champion_accuracy * m.n_champions) / NULLIF(SUM(m.n_champions), 0),
                      SUM(m.ceiling_accuracy  * m.n_champions) / NULLIF(SUM(m.n_champions), 0),
                      SUM(m.n_champions)
                   FROM champion_experiment_month m
                   JOIN champion_experiment e ON e.experiment_id = m.experiment_id
                   WHERE e.is_promoted = TRUE
                     AND m.month_start >= %s::date - (%s * interval '1 month')
                     AND m.month_start <  %s::date""",
                (planning_dt, months, planning_dt),
            )
            champ_row = cur.fetchone()
        except psycopg.Error:
            # Tables may not exist yet on a fresh DB. Silently fall back.
            conn.rollback()
            champ_row = None
    if champ_row is not None and len(champ_row) >= 3:
        champ_n_rows = int(champ_row[2] or 0)
        if champ_row[0] is not None:
            models["champion"] = {
                "model_id": "champion",
                "accuracy_pct": _round_or_none(float(champ_row[0])),
                "n_rows": champ_n_rows,
            }
        if champ_row[1] is not None:
            models["ceiling"] = {
                "model_id": "ceiling",
                "accuracy_pct": _round_or_none(float(champ_row[1])),
                "n_rows": champ_n_rows,
            }

    stages = [
        _build_stage(
            stage_id, label, description, default_state, missing_state, models.get(stage_id)
        )
        for stage_id, label, description, default_state, missing_state in STAGE_DEFS
    ]
    for idx in range(1, len(stages)):
        prev = stages[idx - 1]
        curr = stages[idx]
        if prev["state"] == "actual" and curr["state"] == "actual":
            curr["delta_vs_prev"] = _round_or_none(curr["accuracy_pct"] - prev["accuracy_pct"], 1)

    ceiling_model = models.get("ceiling")
    benchmark = {
        "stage_id": "ceiling",
        "label": "Ceiling Benchmark",
        "description": "Reference best-case benchmark rather than a production handoff stage.",
        "state": "actual" if ceiling_model else "missing",
        "accuracy_pct": ceiling_model["accuracy_pct"] if ceiling_model else None,
        "delta_vs_prev": None,
        "n_rows": ceiling_model["n_rows"] if ceiling_model else 0,
    }

    return {
        "months": months,
        "waterfall": {
            "stages": stages,
            "benchmark": benchmark,
            "external": models.get("external"),
            "champion": models.get("champion"),
            "ceiling": models.get("ceiling"),
            "models": list(models.values()),
        },
    }


@router.get("/snapshot-accuracy")
@cached_sync(ttl=300, group="fva_snapshot")
def snapshot_accuracy(
    record_month: date = Query(..., description="Archived planning month"),
    lag: int | None = Query(None, ge=0, le=5, description="Optional snapshot lag"),
):
    """Live accuracy for the frozen champion-plus-three archive roster."""
    sql = """WITH roster AS (
                   SELECT record_month, model_id, snapshot_role, contender_rank
                   FROM forecast_snapshot_roster
                   WHERE record_month = %s
               ),
               lag_grid AS (
                   SELECT r.record_month, r.model_id, r.snapshot_role, r.contender_rank,
                          l.lag::smallint AS lag,
                          (r.record_month + (l.lag * INTERVAL '1 month'))::date AS forecast_month
                   FROM roster r
                   CROSS JOIN generate_series(0, 5) AS l(lag)
                   WHERE (%s::smallint IS NULL OR l.lag = %s)
               ),
               filtered AS (
                   SELECT a.record_month, a.model_id, r.snapshot_role, r.contender_rank,
                          a.lag, a.forecast_month, a.item_id, a.loc,
                          a.forecast_qty, a.actual_qty, a.abs_error
                   FROM agg_accuracy_snapshot a
                   JOIN forecast_snapshot_roster r
                     ON r.record_month = a.record_month
                    AND r.model_id = a.model_id
                   WHERE a.record_month = %s
                     AND (%s::smallint IS NULL OR a.lag = %s)
               ),
               own AS (
                   SELECT model_id, snapshot_role, contender_rank, lag, forecast_month,
                          SUM(forecast_qty) AS sum_forecast,
                          SUM(actual_qty) AS sum_actual,
                          SUM(abs_error) AS sum_abs_error,
                          COUNT(DISTINCT (item_id, loc)) AS n_dfus
                   FROM filtered
                   GROUP BY model_id, snapshot_role, contender_rank, lag, forecast_month
               ),
               common_pairs AS (
                   SELECT contender.model_id, contender.lag, contender.forecast_month,
                          SUM(contender.forecast_qty) AS contender_forecast,
                          SUM(contender.actual_qty) AS common_actual,
                          SUM(contender.abs_error) AS contender_abs_error,
                          SUM(champion.forecast_qty) AS champion_forecast,
                          SUM(champion.abs_error) AS champion_abs_error,
                          COUNT(*) AS n_dfus_common
                   FROM filtered contender
                   JOIN filtered champion
                     ON champion.model_id = 'champion'
                    AND champion.lag = contender.lag
                    AND champion.forecast_month = contender.forecast_month
                    AND champion.item_id = contender.item_id
                    AND champion.loc = contender.loc
                   WHERE contender.model_id <> 'champion'
                   GROUP BY contender.model_id, contender.lag, contender.forecast_month
               )
               SELECT grid.model_id, grid.snapshot_role, grid.contender_rank, grid.lag,
                      grid.forecast_month, own.sum_forecast, own.sum_actual,
                      own.sum_abs_error, own.n_dfus, common_pairs.contender_forecast,
                      common_pairs.common_actual, common_pairs.contender_abs_error,
                      common_pairs.champion_forecast, common_pairs.champion_abs_error,
                      common_pairs.n_dfus_common
               FROM lag_grid grid
               LEFT JOIN own
                 ON own.model_id = grid.model_id
                AND own.lag = grid.lag
                AND own.forecast_month = grid.forecast_month
               LEFT JOIN common_pairs
                 ON common_pairs.model_id = grid.model_id
                AND common_pairs.lag = grid.lag
                AND common_pairs.forecast_month = grid.forecast_month
               ORDER BY CASE WHEN grid.snapshot_role = 'champion' THEN 0 ELSE 1 END,
                        grid.contender_rank NULLS FIRST, grid.lag, grid.forecast_month"""
    try:
        with get_read_only_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, (record_month, lag, lag, record_month, lag, lag))
            db_rows = cur.fetchall()
    except psycopg.Error:
        logger.exception("Failed to load forecast snapshot accuracy")
        raise HTTPException(
            status_code=500, detail="Failed to load forecast snapshot accuracy"
        ) from None

    rows = []
    for row in db_rows:
        own = compute_kpis(
            float(row[5] or 0), float(row[6] or 0), float(row[7] or 0), int(row[8] or 0)
        )
        if row[0] == "champion":
            delta = 0.0
            common_count = own["dfu_count"]
        elif row[14] is None:
            delta = None
            common_count = 0
        else:
            contender_common = compute_kpis(
                float(row[9] or 0), float(row[10] or 0), float(row[11] or 0), int(row[14] or 0)
            )
            champion_common = compute_kpis(
                float(row[12] or 0), float(row[10] or 0), float(row[13] or 0), int(row[14] or 0)
            )
            contender_accuracy = contender_common["accuracy_pct"]
            champion_accuracy = champion_common["accuracy_pct"]
            delta = (
                round(contender_accuracy - champion_accuracy, 4)
                if contender_accuracy is not None and champion_accuracy is not None
                else None
            )
            common_count = int(row[14] or 0)
        rows.append(
            {
                "model_id": row[0],
                "snapshot_role": row[1],
                "contender_rank": row[2],
                "lag": int(row[3]),
                "forecast_month": row[4].isoformat(),
                "n_dfus": own["dfu_count"],
                "accuracy_pct": own["accuracy_pct"],
                "wape": own["wape"],
                "bias": own["bias"],
                "fva_vs_champion_pts": delta,
                "n_dfus_common": common_count,
            }
        )
    return {"record_month": record_month.isoformat(), "rows": rows}


@router.get("/snapshot-months")
@cached_sync(ttl=300, group="fva_snapshot")
def snapshot_months():
    """Available snapshot months and their closed-lag coverage."""
    sql = """SELECT r.record_month, COUNT(DISTINCT a.lag), MAX(a.forecast_month),
                      MAX(a.last_refresh_at)
             FROM forecast_snapshot_roster r
             LEFT JOIN agg_accuracy_snapshot a ON a.record_month = r.record_month
             GROUP BY r.record_month
             ORDER BY r.record_month DESC"""
    try:
        with get_read_only_conn() as conn, conn.cursor() as cur:
            cur.execute(sql)
            db_rows = cur.fetchall()
    except psycopg.Error:
        logger.exception("Failed to load forecast snapshot months")
        raise HTTPException(
            status_code=500, detail="Failed to load forecast snapshot months"
        ) from None
    return {
        "months": [
            {
                "record_month": row[0].isoformat(),
                "closed_lag_count": int(row[1] or 0),
                "latest_closed_forecast_month": row[2].isoformat() if row[2] else None,
                "last_refresh_at": row[3].isoformat() if row[3] else None,
            }
            for row in db_rows
        ]
    }


@router.get("/historical-backtest-months")
@cached_sync(ttl=300, group="fva_historical")
def historical_backtest_months():
    """Latest three pre-snapshot months with legacy backtest evidence.

    These months are intentionally separate from the immutable live-forward
    snapshot archive. A month with a frozen snapshot roster can never appear
    in this legacy selector.
    """
    planning_month = get_planning_date().replace(day=1)
    sql = """WITH first_live AS (
                 SELECT MIN(record_month) AS record_month
                 FROM forecast_snapshot_roster
             ),
             historical AS (
                 SELECT DISTINCT a.startdate AS month_start
                 FROM backtest_lag_archive a
                 CROSS JOIN first_live
                 WHERE a.startdate < COALESCE(first_live.record_month, %s::date)
                   AND NOT EXISTS (
                       SELECT 1
                       FROM forecast_snapshot_roster r
                       WHERE r.record_month = a.startdate
                   )
             )
             SELECT month_start
             FROM historical
             ORDER BY month_start DESC
             LIMIT 3"""
    try:
        with get_read_only_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, (planning_month,))
            db_rows = cur.fetchall()
    except psycopg.Error:
        logger.exception("Failed to load historical backtest months")
        raise HTTPException(
            status_code=500, detail="Failed to load historical backtest months"
        ) from None
    return {
        "months": [row[0].isoformat() for row in db_rows],
        "evidence_type": "historical_backtest",
    }


@router.get("/historical-backtest-accuracy")
@cached_sync(ttl=300, group="fva_historical")
def historical_backtest_accuracy(
    month: date = Query(..., description="Historical target month"),
):
    """Five-model lag accuracy from legacy backtests, never live snapshots.

    The legacy backtest contract collected lags 0 through 4. Lag 5 is emitted
    as ``not_collected`` so the UI can keep a six-column comparison without
    implying that missing historical evidence is pending or reconstructable.
    """
    model_ids = list(get_algorithm_roster(stage="compete"))
    sql = """SELECT a.model_id, a.lag,
                    COUNT(DISTINCT (a.item_id, a.customer_group, a.loc))::bigint,
                    SUM(a.basefcst_pref), SUM(a.tothist_dmd),
                    SUM(ABS(a.basefcst_pref - a.tothist_dmd))
             FROM backtest_lag_archive a
             WHERE a.startdate = %s
               AND a.model_id = ANY(%s)
               AND a.basefcst_pref IS NOT NULL
               AND a.tothist_dmd IS NOT NULL
               AND NOT EXISTS (
                   SELECT 1
                   FROM forecast_snapshot_roster r
                   WHERE r.record_month = %s
               )
             GROUP BY a.model_id, a.lag
             ORDER BY a.model_id, a.lag"""
    try:
        with get_read_only_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, (month, model_ids, month))
            db_rows = cur.fetchall()
    except psycopg.Error:
        logger.exception("Failed to load historical backtest accuracy")
        raise HTTPException(
            status_code=500, detail="Failed to load historical backtest accuracy"
        ) from None

    measured: dict[tuple[str, int], dict] = {}
    for model_id, lag, n_dfus, sum_forecast, sum_actual, sum_abs_error in db_rows:
        measured[(str(model_id), int(lag))] = compute_kpis(
            float(sum_forecast or 0),
            float(sum_actual or 0),
            float(sum_abs_error or 0),
            int(n_dfus or 0),
        )

    rows = []
    for model_id in model_ids:
        for lag in range(6):
            kpis = measured.get((model_id, lag))
            if lag == 5:
                evidence_state = "not_collected"
            elif kpis is None:
                evidence_state = "missing"
            else:
                evidence_state = "measured"
            rows.append(
                {
                    "model_id": model_id,
                    "lag": lag,
                    "n_dfus": kpis["dfu_count"] if kpis else 0,
                    "accuracy_pct": kpis["accuracy_pct"] if kpis else None,
                    "wape": kpis["wape"] if kpis else None,
                    "bias": kpis["bias"] if kpis else None,
                    "evidence_state": evidence_state,
                }
            )
    return {
        "month": month.isoformat(),
        "evidence_type": "historical_backtest",
        "source": "backtest_lag_archive",
        "supported_lags": list(range(5)),
        "unsupported_lags": [5],
        "rows": rows,
    }


@router.get("/interventions")
async def list_interventions(
    intervention_type: str = Query("", description="Filter by type"),
    status: str = Query("", description="Filter by status"),
    user_id: str = Query("", description="Filter by user"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List tracked interventions with before/after metrics."""
    where = []
    params: list = []
    if intervention_type:
        where.append("intervention_type = %s")
        params.append(intervention_type)
    if status:
        where.append("status = %s")
        params.append(status)
    if user_id:
        where.append("user_id = %s::uuid")
        params.append(user_id)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM fact_intervention_metrics {where_sql}", params)
        total = cur.fetchone()[0]

        cur.execute(
            f"""SELECT intervention_id, user_id, intervention_type, resource_type, resource_id,
                       metric_before, metric_after, financial_impact_estimate,
                       actual_financial_impact, measurement_window_start,
                       measurement_window_end, status, created_at
                FROM fact_intervention_metrics
                {where_sql}
                ORDER BY created_at DESC LIMIT %s OFFSET %s""",
            [*params, limit, offset],
        )
        rows = cur.fetchall()

    return {
        "total": total,
        "interventions": [
            {
                "intervention_id": r[0],
                "user_id": str(r[1]) if r[1] else None,
                "intervention_type": r[2],
                "resource_type": r[3],
                "resource_id": r[4],
                "metric_before": r[5],
                "metric_after": r[6],
                "financial_impact_estimate": float(r[7]) if r[7] else None,
                "actual_financial_impact": float(r[8]) if r[8] else None,
                "measurement_window_start": r[9].isoformat() if r[9] else None,
                "measurement_window_end": r[10].isoformat() if r[10] else None,
                "status": r[11],
                "created_at": r[12].isoformat() if r[12] else None,
            }
            for r in rows
        ],
    }


@router.get("/roi-summary")
async def roi_summary(
    months: int = Query(12, ge=1, le=36),
):
    """Aggregate ROI metrics: total interventions, measured outcomes, net financial impact."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT
                 count(*) AS total_interventions,
                 count(*) FILTER (WHERE status = 'measured') AS measured,
                 count(*) FILTER (WHERE status = 'pending') AS pending,
                 coalesce(sum(financial_impact_estimate), 0) AS total_estimated_impact,
                 coalesce(sum(actual_financial_impact) FILTER (WHERE status = 'measured'), 0) AS total_actual_impact
               FROM fact_intervention_metrics
               WHERE created_at >= %s::date - (%s * interval '1 month')""",
            (get_planning_date(), months),
        )
        row = cur.fetchone()

    return {
        "months": months,
        "total_interventions": row[0],
        "measured": row[1],
        "pending": row[2],
        "total_estimated_impact": float(row[3]) if row[3] else 0,
        "total_actual_impact": float(row[4]) if row[4] else 0,
    }
