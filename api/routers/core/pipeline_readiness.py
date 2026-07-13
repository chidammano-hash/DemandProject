"""Fail-closed readiness for the governed forecasting lifecycle."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

import psycopg
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response as FastAPIResponse

from api.core import get_read_only_conn, set_cache
from common.services.cache import cached_sync

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dashboard"])

_CANONICAL_MODELS = {
    "chronos2_enriched",
    "lgbm_cluster",
    "mstl",
    "nbeats",
    "nhits",
}


def _check(
    *,
    stage: str,
    severity: str,
    title: str,
    detail: str,
    action: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "stale",
        "severity": severity,
        "title": title,
        "detail": detail,
        "action": action,
    }


def _decode_mapping(value: object) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return dict(decoded) if isinstance(decoded, Mapping) else None
    return None


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _positive_int(value: object) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return None
    return value


def _governed_roster_is_exact(spec: Mapping[str, Any]) -> bool:
    models = spec.get("models")
    raw_run_ids = spec.get("backtest_run_ids")
    if (
        not isinstance(models, list)
        or len(models) != len(_CANONICAL_MODELS)
        or any(not isinstance(model_id, str) for model_id in models)
        or set(models) != _CANONICAL_MODELS
    ):
        return False
    if not isinstance(raw_run_ids, list) or len(raw_run_ids) != len(_CANONICAL_MODELS):
        return False
    run_ids: dict[str, int] = {}
    for pair in raw_run_ids:
        if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str):
            return False
        run_id = _positive_int(pair[1])
        if run_id is None or pair[0] in run_ids:
            return False
        run_ids[pair[0]] = run_id
    return (
        set(run_ids) == _CANONICAL_MODELS
        and len(run_ids) == len(raw_run_ids)
    )


def _append_governed_checks(
    checks: list[dict[str, Any]],
    row: tuple[Any, ...] | None,
) -> None:
    if row is None or len(row) != 10:
        checks.append(
            _check(
                stage="champion",
                severity="high",
                title="Champion lineage could not be verified",
                detail=(
                    "The governed champion state is incomplete. Run named pipeline "
                    "'model-refresh' before generating or publishing forecasts."
                ),
            )
        )
        return

    (
        champion_count,
        experiment_id,
        is_promoted,
        champion_cluster_id,
        cluster_count,
        current_cluster_id,
        raw_job_params,
        current_sales_batch_id,
        current_source_hash,
        current_source_file,
    ) = row
    if (
        int(champion_count or 0) != 1
        or experiment_id is None
        or not bool(is_promoted)
    ):
        checks.append(
            _check(
                stage="champion",
                severity="high",
                title="No sole governed champion is promoted",
                detail=(
                    "Production forecasting requires exactly one results-promoted governed "
                    "champion. Run named pipeline 'model-refresh'."
                ),
            )
        )
        return

    params = _decode_mapping(raw_job_params)
    spec = _decode_mapping(params.get("governed_spec")) if params is not None else None
    if spec is None or not _governed_roster_is_exact(spec):
        checks.append(
            _check(
                stage="champion",
                severity="high",
                title="Promoted champion lacks governed five-model evidence",
                detail=(
                    f"Champion experiment {experiment_id} is not bound to the exact five-model "
                    "governed backtest roster. Run named pipeline 'model-refresh'."
                ),
            )
        )
        return

    spec_cluster_id = _positive_int(spec.get("cluster_experiment_id"))
    if (
        int(cluster_count or 0) != 1
        or champion_cluster_id is None
        or current_cluster_id is None
        or spec_cluster_id is None
        or int(champion_cluster_id) != int(current_cluster_id)
        or spec_cluster_id != int(current_cluster_id)
    ):
        checks.append(
            _check(
                stage="champion",
                severity="high",
                title="Promoted champion predates the current clusters",
                detail=(
                    f"Champion experiment {experiment_id} does not carry the sole current "
                    "promoted cluster lineage. Run named pipeline 'model-refresh'."
                ),
            )
        )

    source_file = str(current_source_file or "").strip()
    if (
        current_sales_batch_id is None
        or not _is_sha256(current_source_hash)
        or not source_file
        or source_file == "safe_upsert"
    ):
        checks.append(
            _check(
                stage="forecast",
                severity="high",
                title="Current sales lineage could not be verified",
                detail=(
                    "Run named pipeline 'data-refresh' to synchronize the canonical sales "
                    "source before model-refresh and forecast-publish."
                ),
            )
        )
        return

    champion_sales_batch_id = spec.get("source_sales_batch_id")
    champion_source_hash = spec.get("data_checksum")
    if (
        not isinstance(champion_sales_batch_id, int)
        or isinstance(champion_sales_batch_id, bool)
        or champion_sales_batch_id != int(current_sales_batch_id)
        or champion_source_hash != current_source_hash
    ):
        checks.append(
            _check(
                stage="forecast",
                severity="medium",
                title="Sales data is newer than the promoted champion",
                detail=(
                    f"Champion experiment {experiment_id} uses sales batch "
                    f"{champion_sales_batch_id}, while batch {current_sales_batch_id} is "
                    "current. Run 'model-refresh', then 'forecast-publish'."
                ),
            )
        )


def _load_governed_readiness(cur: Any) -> tuple[Any, ...] | None:
    cur.execute(
        """WITH selected_champion AS (
                 SELECT experiment_id, is_promoted, cluster_experiment_id, job_id
                 FROM champion_experiment
                 WHERE is_results_promoted = TRUE
                 ORDER BY results_promoted_at DESC NULLS LAST, experiment_id DESC
                 LIMIT 1
             ), selected_cluster AS (
                 SELECT experiment_id
                 FROM cluster_experiment
                 WHERE is_promoted = TRUE
                 ORDER BY promoted_at DESC NULLS LAST, experiment_id DESC
                 LIMIT 1
             ), selected_sales AS (
                 SELECT batch_id, source_hash, source_file
                 FROM audit_load_batch
                 WHERE domain = 'sales'
                   AND status = 'completed'
                   AND row_count_out > 0
                 ORDER BY completed_at DESC NULLS LAST, batch_id DESC
                 LIMIT 1
             )
             SELECT
                 (SELECT COUNT(*) FROM champion_experiment
                  WHERE is_results_promoted = TRUE),
                 champion.experiment_id,
                 champion.is_promoted,
                 champion.cluster_experiment_id,
                 (SELECT COUNT(*) FROM cluster_experiment WHERE is_promoted = TRUE),
                 cluster.experiment_id,
                 job.params,
                 sales.batch_id,
                 sales.source_hash,
                 sales.source_file
             FROM (SELECT 1) anchor
             LEFT JOIN selected_champion champion ON TRUE
             LEFT JOIN selected_cluster cluster ON TRUE
             LEFT JOIN job_history job ON job.job_id = champion.job_id
             LEFT JOIN selected_sales sales ON TRUE"""
    )
    return cur.fetchone()


@router.get("/dashboard/pipeline-readiness")
@cached_sync(ttl=60, group="dashboard")
def get_pipeline_readiness(response: FastAPIResponse) -> dict[str, Any]:
    """Return fail-closed, lineage-backed forecasting lifecycle readiness."""
    checks: list[dict[str, Any]] = []
    with get_read_only_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute(
                "SELECT "
                "(SELECT COUNT(*) FROM dim_sku) AS total, "
                "(SELECT COUNT(*) FROM current_sku_cluster_assignment "
                " WHERE ml_cluster IS NOT NULL) AS clustered"
            )
            row = cur.fetchone()
            total, clustered = row if row is not None else (0, 0)
        except psycopg.Error:
            logger.exception("pipeline-readiness data/cluster check failed")
            raise HTTPException(
                status_code=500,
                detail="Failed to read pipeline readiness",
            ) from None

        if int(total or 0) == 0:
            checks.append(
                _check(
                    stage="data",
                    severity="high",
                    title="Forecast source data is not ready",
                    detail=(
                        "No SKU population is available. Run named pipeline 'data-refresh', "
                        "then 'clustering-refresh'."
                    ),
                )
            )
        elif int(clustered or 0) == 0:
            checks.append(
                _check(
                    stage="clustering",
                    severity="high",
                    title="Clustering needs to be re-run",
                    detail=(
                        "No SKUs have a promoted ML cluster assignment. Run named pipeline "
                        "'clustering-refresh' before model-refresh."
                    ),
                    action={
                        "kind": "navigate",
                        "target": "clusters",
                        "label": "Open Clustering",
                    },
                )
            )

        try:
            cur.execute(
                """SELECT COUNT(*) FROM cluster_tuning_profile_state tuning
                   WHERE tuning.stale = TRUE
                     AND EXISTS (
                         SELECT 1 FROM current_sku_cluster_assignment assignment
                         WHERE assignment.ml_cluster = tuning.cluster_name
                     )"""
            )
            tuning_row = cur.fetchone()
            stale_profiles = int(tuning_row[0]) if tuning_row else 0
        except psycopg.Error:
            conn.rollback()
            checks.append(
                _check(
                    stage="tuning",
                    severity="high",
                    title="Tuning readiness could not be verified",
                    detail=(
                        "Current-cluster tuning state is unavailable. Repair the schema or "
                        "database query, then run named pipeline 'model-refresh'."
                    ),
                )
            )
        else:
            if stale_profiles:
                checks.append(
                    _check(
                        stage="tuning",
                        severity="medium",
                        title=(
                            f"{stale_profiles} tuning profile(s) predate the current clusters"
                        ),
                        detail=(
                            "Run named pipeline 'model-refresh' to retune the exact current "
                            "cluster labels before governed backtests."
                        ),
                    )
                )

        try:
            governance_row = _load_governed_readiness(cur)
        except psycopg.Error:
            conn.rollback()
            checks.append(
                _check(
                    stage="champion",
                    severity="high",
                    title="Champion readiness could not be verified",
                    detail=(
                        "Governed champion lineage is unavailable. Repair the schema or query, "
                        "then run named pipeline 'model-refresh'."
                    ),
                )
            )
        else:
            _append_governed_checks(checks, governance_row)

    set_cache(response, max_age=60)
    return {"ready": len(checks) == 0, "checks": checks}
