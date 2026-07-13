"""Private helpers for production-model and snapshot readiness APIs."""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import psycopg
from fastapi import HTTPException
from psycopg import sql

from common.core.paths import PROJECT_ROOT
from common.core.utils import load_forecast_pipeline_config
from common.ml.neural_forecast import NeuralCohortIdentity
from common.services.champion_lineage import (
    GovernedChampionLineageError,
    load_active_governed_champion_lineage,
)
from common.services.cluster_lineage import (
    PromotedClusterPopulation,
    load_promoted_cluster_population,
)
from common.services.forecast_lineage import sha256_file
from common.services.forecast_snapshot_validation import (
    SnapshotContenderIntegrityError,
    SnapshotContenderStaleError,
)
from common.services.sales_lineage import SalesSourceLineage, load_completed_sales_lineage

logger = logging.getLogger(__name__)

DIRECT_INFERENCE_MODEL_IDS = frozenset({"mstl", "chronos2_enriched"})
PUBLISH_PIPELINE_NAME = "forecast-publish"
CHAMPION_REFRESH_PIPELINE_NAME = "champion-refresh"


@dataclass(frozen=True, slots=True)
class CurrentTrainingLineage:
    """One repeatable-read snapshot used by every persisted-model check."""

    sales: SalesSourceLineage
    history_end: date
    clustering_enabled: bool
    clusters: PromotedClusterPopulation | None
    cluster_stale_reason: str | None
    neural_cohorts: dict[int, NeuralCohortIdentity]
    neural_cohort_stale_reason: str | None


def load_latest_closed_sales_month(
    conn: Any,
    *,
    sales_table: str,
    expected_history_end: date,
) -> date:
    """Prove that the strict forecast mirror contains the latest closed month."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT MAX(startdate) FROM {} WHERE type = 1 AND startdate <= %s"
            ).format(sql.Identifier(sales_table)),
            (expected_history_end,),
        )
        row = cur.fetchone()
    latest = row[0] if row else None
    if latest is None:
        raise RuntimeError("The canonical forecast sales mirror has no closed history")
    latest_month = latest.date() if hasattr(latest, "date") else latest
    normalized = latest_month.replace(day=1)
    if normalized != expected_history_end:
        raise RuntimeError("The canonical forecast sales mirror is missing the latest closed month")
    return normalized


def production_model_base_dir(config: Mapping[str, Any] | None = None) -> Path:
    """Resolve the configured production artifact root."""
    resolved_config = config if config is not None else load_forecast_pipeline_config()
    raw_path = (
        resolved_config.get("production_forecast", {})
        .get("model_registry", {})
        .get("base_path", "data/models")
    )
    path = Path(str(raw_path))
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_current_training_lineage(
    config: Mapping[str, Any],
    *,
    neural_min_history_values: Collection[int],
    get_conn: Callable[[], Any],
    get_planning_date: Callable[[], date],
    load_completed_sales_lineage: Callable[..., SalesSourceLineage],
    load_promoted_cluster_population: Callable[..., PromotedClusterPopulation],
    resolve_forecast_sales_table: Callable[..., str],
    load_neural_training_cohort_identity: Callable[..., NeuralCohortIdentity],
    require_direct_history: bool = False,
) -> CurrentTrainingLineage:
    """Read one repeatable snapshot of current sales, clusters, and neural cohorts."""
    clustering = config.get("clustering")
    if not isinstance(clustering, Mapping):
        raise ValueError("Forecast configuration is missing clustering settings")
    clustering_enabled = clustering.get("enabled")
    if not isinstance(clustering_enabled, bool):
        raise ValueError("clustering.enabled must be explicitly true or false")

    cluster_stale_reason: str | None = None
    neural_cohort_stale_reason: str | None = None
    neural_cohorts: dict[int, NeuralCohortIdentity] = {}
    planning_month = get_planning_date().replace(day=1)
    history_end = (planning_month - timedelta(days=1)).replace(day=1)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        sales = load_completed_sales_lineage(conn)
        try:
            clusters = load_promoted_cluster_population(conn) if clustering_enabled else None
        except RuntimeError:
            logger.warning("Current promoted clustering is not ready", exc_info=True)
            clusters = None
            cluster_stale_reason = (
                "Current promoted clustering is unavailable or invalid. Run the clustering "
                "refresh, then Forecast Publish to rebuild LightGBM."
            )
        if neural_min_history_values or require_direct_history:
            try:
                with conn.cursor() as cur:
                    sales_table = resolve_forecast_sales_table(cur)
                if require_direct_history:
                    load_latest_closed_sales_month(
                        conn,
                        sales_table=sales_table,
                        expected_history_end=history_end,
                    )
                for min_history in sorted(set(neural_min_history_values)):
                    neural_cohorts[min_history] = load_neural_training_cohort_identity(
                        conn,
                        sales_table=sales_table,
                        history_end=history_end,
                        min_history=min_history,
                    )
            except (RuntimeError, ValueError):
                if require_direct_history:
                    raise
                logger.warning("Current neural training cohort is not ready", exc_info=True)
                neural_cohorts.clear()
                neural_cohort_stale_reason = (
                    "The current neural training cohort cannot be proven. Run Forecast "
                    "Publish after the canonical sales source is ready."
                )
    return CurrentTrainingLineage(
        sales=sales,
        history_end=history_end,
        clustering_enabled=clustering_enabled,
        clusters=clusters,
        cluster_stale_reason=cluster_stale_reason,
        neural_cohorts=neural_cohorts,
        neural_cohort_stale_reason=neural_cohort_stale_reason,
    )


def mark_not_trained(entry: dict[str, Any], *, stale_reason: str | None = None) -> None:
    """Set the common fail-closed readiness response fields."""
    entry.update(
        trained=False,
        ready=False,
        trained_at=None,
        training_mode=None,
        n_dfus=None,
        planning_date=None,
    )
    if stale_reason is not None:
        entry["stale_reason"] = stale_reason


def retrain_reason(model_id: str) -> str:
    stale_inputs = (
        "sales, history, configuration, runtime, or eligible DFU cohort"
        if model_id in {"nhits", "nbeats"}
        else "sales, history, configuration, or promoted clustering"
    )
    return (
        f"The active {model_id} production artifact is stale for the current {stale_inputs}. "
        f"Run Forecast Publish to retrain {model_id} before generating a production forecast."
    )


def missing_artifact_reason(model_id: str) -> str:
    return (
        f"The active {model_id} production artifact is missing. Run Forecast Publish "
        "to build current production artifacts."
    )


def invalid_artifact_reason(model_id: str) -> str:
    return (
        f"The active {model_id} production artifact is invalid. Run Forecast Publish "
        "to rebuild current production artifacts."
    )


def mark_direct_model_ready(
    entry: dict[str, Any],
    *,
    history_end: date,
    source_sales_batch_id: int,
    config_checksum: str,
    runtime_contract: Mapping[str, str],
) -> None:
    """Mark a direct model ready only after current-data/config/runtime preflight."""
    entry.update(
        trained=False,
        ready=True,
        trained_at=None,
        training_mode="direct_inference",
        n_dfus=None,
        planning_date=history_end.isoformat(),
        source_sales_batch_id=source_sales_batch_id,
        config_checksum=config_checksum,
        runtime_contract=dict(runtime_contract),
    )


def validate_active_champion_readiness(conn: Any, cur: Any) -> dict[str, Any]:
    """Validate the sole governed champion against current inputs and routing bytes."""
    lineage = load_active_governed_champion_lineage(cur)
    sales = load_completed_sales_lineage(conn)
    clusters = load_promoted_cluster_population(conn)
    if (
        lineage["source_sales_batch_id"] != sales.batch_id
        or lineage["data_checksum"] != sales.source_hash
        or lineage["cluster_experiment_id"] != clusters.experiment_id
        or lineage["cluster_assignment_count"] != clusters.assignment_count
        or lineage["cluster_assignment_checksum"] != clusters.assignment_checksum
    ):
        raise GovernedChampionLineageError(
            "The active champion does not match current sales and clustering"
        )
    experiment_id = int(lineage["experiment_id"])
    cur.execute(
        """SELECT results_artifact_checksum
           FROM champion_experiment
           WHERE experiment_id = %s
             AND is_promoted = TRUE
             AND is_results_promoted = TRUE""",
        (experiment_id,),
    )
    row = cur.fetchone()
    expected_checksum = str(row[0]).strip().lower() if row and row[0] else None
    winners_path = PROJECT_ROOT / "data" / "champion" / f"experiment_{experiment_id}_winners.csv"
    if expected_checksum is None or not winners_path.is_file():
        raise GovernedChampionLineageError(
            "The active champion routing artifact is unavailable"
        )
    if sha256_file(winners_path) != expected_checksum:
        raise GovernedChampionLineageError(
            "The active champion routing artifact checksum changed"
        )
    return lineage


def evaluate_snapshot_roster_readiness(
    *,
    get_conn: Callable[[], Any],
    get_planning_date: Callable[[], date],
    validate_ready_snapshot_contender: Callable[..., Any],
    validate_active_champion: Callable[[Any, Any], dict[str, Any]],
) -> dict[str, Any]:
    """Validate the current champion plus exact top-three snapshot contenders."""
    planning_month = get_planning_date().replace(day=1)
    active_champion_ready = True
    active_champion_stale_reason: str | None = None
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            try:
                validate_active_champion(conn, cur)
            except (FileNotFoundError, GovernedChampionLineageError, OSError, RuntimeError):
                logger.warning("Active governed champion is not ready", exc_info=True)
                active_champion_ready = False
                active_champion_stale_reason = (
                    "There is no sole current governed champion results experiment with an "
                    "intact routing artifact. Run the named Champion Refresh pipeline."
                )
            cur.execute(
                """SELECT model_id, snapshot_role, contender_rank,
                          source_backtest_run_id, generation_run_id
                   FROM forecast_snapshot_roster
                   WHERE record_month = %s
                   ORDER BY CASE WHEN snapshot_role = 'champion' THEN 0 ELSE 1 END,
                            contender_rank NULLS FIRST, model_id""",
                (planning_month,),
            )
            rows = cur.fetchall()

            champion_rows = [
                row
                for row in rows
                if row[1] == "champion"
                and row[0] == "champion"
                and row[2] is None
                and row[3] is None
                and row[4] is None
            ]
            contender_rows = [row for row in rows if row[1] == "contender"]
            exact_ranks = [row[2] for row in contender_rows] == [1, 2, 3]
            distinct_models = len({str(row[0]) for row in contender_rows}) == 3
            complete_structure = (
                len(rows) == 4
                and len(champion_rows) == 1
                and len(contender_rows) == 3
                and exact_ranks
                and distinct_models
                and len({row[4] for row in contender_rows}) == 3
                and all(
                    row[0] != "champion" and row[3] is not None and row[4] is not None
                    for row in contender_rows
                )
            )

            contenders: list[dict[str, Any]] = []
            has_integrity_failure = False
            if complete_structure:
                for model_id, _role, rank, backtest_run_id, run_id in contender_rows:
                    stale_reason: str | None = None
                    try:
                        validate_ready_snapshot_contender(
                            cur,
                            run_id=run_id,
                            model_id=str(model_id),
                            record_month=planning_month,
                            source_backtest_run_id=int(backtest_run_id),
                        )
                    except SnapshotContenderIntegrityError as exc:
                        has_integrity_failure = True
                        stale_reason = str(exc)
                    except SnapshotContenderStaleError as exc:
                        stale_reason = str(exc)
                    contenders.append(
                        {
                            "model_id": str(model_id),
                            "rank": int(rank),
                            "ready": stale_reason is None,
                            "stale_reason": stale_reason,
                        }
                    )
    except psycopg.Error as exc:
        logger.exception("Snapshot roster readiness query failed")
        raise HTTPException(
            status_code=500,
            detail="snapshot roster readiness check failed",
        ) from exc

    ready_contender_count = sum(1 for contender in contenders if contender["ready"])
    ready = active_champion_ready and complete_structure and ready_contender_count == 3
    if ready:
        stale_reason = None
        action_pipeline = None
    elif not active_champion_ready:
        stale_reason = active_champion_stale_reason
        action_pipeline = CHAMPION_REFRESH_PIPELINE_NAME
    elif not complete_structure:
        stale_reason = (
            "The current champion plus exact top-three snapshot roster is incomplete. "
            "Run Forecast Publish to prepare the release candidate and contender evidence."
        )
        action_pipeline = PUBLISH_PIPELINE_NAME
    elif has_integrity_failure:
        stale_reason = (
            "Snapshot contender evidence failed an integrity check. Review the failed evidence "
            "in Jobs before rebuilding; automatic replacement is intentionally blocked."
        )
        action_pipeline = None
    else:
        stale_reason = (
            "One or more top-three snapshot contenders are stale or invalid. Run Forecast "
            "Publish to rebuild the release candidate and contender evidence."
        )
        action_pipeline = PUBLISH_PIPELINE_NAME
    return {
        "planning_month": planning_month.isoformat(),
        "ready": ready,
        "champion_ready": active_champion_ready and len(champion_rows) == 1,
        "roster_model_count": len(rows),
        "ready_contender_count": ready_contender_count,
        "required_contender_count": 3,
        "contenders": contenders,
        "stale_reason": stale_reason,
        "action_pipeline": action_pipeline,
    }
