"""Shared invariants for the bounded live-forecast snapshot archive."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any
from uuid import UUID

from common.services.forecast_lineage import (
    compute_production_payload_stats,
    compute_snapshot_champion_payload_stats,
)

SNAPSHOT_LAGS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)


def select_top_contenders(
    runs: Iterable[Mapping[str, Any]],
    *,
    contender_count: int = 3,
) -> list[dict[str, Any]]:
    """Return the fixed top contenders, ranked by completed backtest WAPE.

    Callers supply one latest completed, database-loaded run per forecastable
    model. ``None`` WAPE runs are intentionally ineligible: ranking an unknown
    score would make the archive roster non-deterministic.
    """
    eligible = [dict(run) for run in runs if run.get("wape") is not None]
    eligible.sort(
        key=lambda run: (
            float(run["wape"]),
            -float(run.get("accuracy_pct") or float("-inf")),
            _descending_datetime_key(run.get("completed_at")),
            str(run["model_id"]),
        )
    )
    if len(eligible) < contender_count:
        raise ValueError(
            f"Expected three eligible contender models (configured count={contender_count}); "
            f"found {len(eligible)}"
        )

    selected = eligible[:contender_count]
    for rank, run in enumerate(selected, start=1):
        run["contender_rank"] = rank
    return selected


def _descending_datetime_key(value: Any) -> float:
    """Sort newer completed runs first without requiring a datetime subtype."""
    if value is None:
        return float("inf")
    timestamp = getattr(value, "timestamp", None)
    if callable(timestamp):
        return -float(timestamp())
    return float("inf")


def missing_required_lags(
    counts_by_model: Mapping[str, Mapping[int, int]],
) -> dict[str, list[int]]:
    """Return required archive lags that have no rows for each selected model."""
    missing: dict[str, list[int]] = {}
    for model_id, counts in counts_by_model.items():
        absent = [lag for lag in SNAPSHOT_LAGS if int(counts.get(lag, 0)) <= 0]
        if absent:
            missing[model_id] = absent
    return missing


def cleanup_reconciliation_issues(
    expected_contender_counts: Mapping[tuple[str, str], int],
    archived_contender_counts: Mapping[tuple[str, str], int],
    *,
    champion_archive_count: int,
) -> list[str]:
    """List selected-roster archive shortfalls that block staging cleanup."""
    issues = [
        (
            f"{model_id}/{run_id}: expected {expected}, "
            f"archived {int(archived_contender_counts.get((model_id, run_id), 0))}"
        )
        for (model_id, run_id), expected in expected_contender_counts.items()
        if int(archived_contender_counts.get((model_id, run_id), 0)) != int(expected)
    ]
    if champion_archive_count <= 0:
        issues.append("champion archive is missing")
    return issues


def _transaction_roster(cur: Any, record_month: date) -> list[tuple[Any, ...]]:
    cur.execute(
        """SELECT model_id, snapshot_role, contender_rank, generation_run_id
           FROM forecast_snapshot_roster
           WHERE record_month = %s
           ORDER BY CASE WHEN snapshot_role = 'champion' THEN 0 ELSE 1 END,
                    contender_rank NULLS FIRST""",
        (record_month,),
    )
    rows = list(cur.fetchall())
    champion_count = sum(1 for row in rows if row[1] == "champion")
    contender_ranks = [int(row[2]) for row in rows if row[1] == "contender"]
    if champion_count != 1 or contender_ranks != [1, 2, 3]:
        raise ValueError(
            "snapshot roster must contain one champion and contender ranks 1, 2, and 3"
        )
    return rows


def archive_snapshot_in_transaction(
    cur: Any,
    *,
    record_month: date,
    production_run_id: UUID,
    source_promotion_id: int,
    overwrite: bool = False,
) -> str:
    """Archive and reconcile one outgoing release using the caller's transaction."""
    roster = _transaction_roster(cur, record_month)
    conflict_clause = (
        """ON CONFLICT (record_month, model_id, item_id, loc, forecast_month)
           DO UPDATE SET
               horizon_months = EXCLUDED.horizon_months,
               forecast_qty = EXCLUDED.forecast_qty,
               forecast_qty_lower = EXCLUDED.forecast_qty_lower,
               forecast_qty_upper = EXCLUDED.forecast_qty_upper,
               source_model_id = EXCLUDED.source_model_id,
               cluster_id = EXCLUDED.cluster_id,
               plan_version = EXCLUDED.plan_version,
               run_id = EXCLUDED.run_id,
               generated_at = EXCLUDED.generated_at,
               is_recursive = EXCLUDED.is_recursive,
               lag_source = EXCLUDED.lag_source,
               source_promotion_id = EXCLUDED.source_promotion_id,
               archived_at = NOW()"""
        if overwrite
        else """ON CONFLICT (record_month, model_id, item_id, loc, forecast_month)
                DO NOTHING"""
    )
    cur.execute(
        """INSERT INTO fact_forecast_snapshot
               (record_month, model_id, item_id, loc, forecast_month,
                horizon_months, forecast_qty, forecast_qty_lower,
                forecast_qty_upper, source_model_id, cluster_id, plan_version,
                run_id, generated_at, is_recursive, lag_source)
           SELECT r.record_month, s.model_id, s.item_id, s.loc,
                  s.forecast_month, s.horizon_months, s.forecast_qty,
                  s.forecast_qty_lower, s.forecast_qty_upper, NULL,
                  s.cluster_id, NULL, s.run_id, s.generated_at,
                  s.is_recursive, s.lag_source
           FROM forecast_snapshot_roster r
           JOIN fact_production_forecast_staging s
             ON s.run_id = r.generation_run_id
            AND s.generation_purpose = 'snapshot_contender'
            AND s.candidate_model_id = r.model_id
            AND s.model_id = r.model_id
            AND s.forecast_month_generated = r.record_month
           WHERE r.record_month = %s
             AND r.snapshot_role = 'contender'
             AND s.forecast_month >= r.record_month
             AND s.forecast_month < r.record_month + INTERVAL '6 months'
           """
        + conflict_clause,
        (record_month,),
    )
    cur.execute(
        """INSERT INTO fact_forecast_snapshot
               (record_month, model_id, item_id, loc, forecast_month,
                horizon_months, forecast_qty, forecast_qty_lower,
                forecast_qty_upper, source_model_id, cluster_id, plan_version,
                run_id, generated_at, is_recursive, lag_source,
                source_promotion_id)
           SELECT r.record_month, 'champion', p.item_id, p.loc,
                  p.forecast_month, p.horizon_months, p.forecast_qty,
                  p.forecast_qty_lower, p.forecast_qty_upper,
                  p.source_model_id, p.cluster_id, p.plan_version,
                  p.run_id, p.generated_at, p.is_recursive, p.lag_source, %s
           FROM forecast_snapshot_roster r
           JOIN fact_production_forecast p
             ON p.model_id = 'champion'
            AND p.plan_version = TO_CHAR(r.record_month, 'YYYY-MM')
            AND p.run_id = %s::uuid
           WHERE r.record_month = %s
             AND r.snapshot_role = 'champion'
             AND p.forecast_month >= r.record_month
             AND p.forecast_month < r.record_month + INTERVAL '6 months'
           """
        + conflict_clause,
        (source_promotion_id, str(production_run_id), record_month),
    )
    cur.execute(
        """SELECT model_id, lag, COUNT(*)
           FROM fact_forecast_snapshot
           WHERE record_month = %s
           GROUP BY model_id, lag""",
        (record_month,),
    )
    counts = {str(row[0]): {} for row in roster}
    for model_id, lag, count in cur.fetchall():
        if str(model_id) in counts:
            counts[str(model_id)][int(lag)] = int(count)
    missing = missing_required_lags(counts)
    if missing:
        raise ValueError(f"outgoing archive is missing required lags: {missing}")

    end_month_index = record_month.year * 12 + record_month.month - 1 + 6
    end_month = date(end_month_index // 12, end_month_index % 12 + 1, 1)
    production_stats = compute_production_payload_stats(
        cur,
        production_run_id,
        start_month=record_month,
        end_month=end_month,
    )
    snapshot_stats = compute_snapshot_champion_payload_stats(
        cur,
        record_month=record_month,
        production_run_id=production_run_id,
    )
    if (
        production_stats.row_count <= 0
        or snapshot_stats.row_count != production_stats.row_count
        or snapshot_stats.dfu_count != production_stats.dfu_count
        or snapshot_stats.checksum != production_stats.checksum
    ):
        raise ValueError("outgoing champion archive checksum does not match production")
    return snapshot_stats.checksum
