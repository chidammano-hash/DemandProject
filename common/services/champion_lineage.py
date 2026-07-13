"""Immutable source lineage for governed champion promotions."""

from __future__ import annotations

import json
from typing import Any

CANONICAL_CHAMPION_MODELS = (
    "lgbm_cluster",
    "nhits",
    "nbeats",
    "mstl",
    "chronos2_enriched",
)
GOVERNED_CHAMPION_LINEAGE_METADATA_KEY = "governed_champion_lineage"
GOVERNED_PROMOTION_MODE = "governed_atomic_refresh"


class GovernedChampionLineageError(ValueError):
    """The active champion lacks complete governed source evidence."""


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _decode_snapshot(value: object) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise GovernedChampionLineageError(
                "Champion promotion audit contains invalid JSON"
            ) from exc
    if not isinstance(value, dict):
        raise GovernedChampionLineageError(
            "Champion promotion audit has no governed configuration snapshot"
        )
    return value


def load_governed_champion_lineage(
    cur: Any,
    *,
    experiment_id: int,
) -> dict[str, Any]:
    """Load and normalize the latest governed audit for one champion experiment."""
    if not _positive_integer(experiment_id):
        raise GovernedChampionLineageError(
            "Champion experiment identifier must be positive"
        )
    cur.execute(
        """SELECT config_snapshot
           FROM champion_promotion_log
           WHERE experiment_id = %s
           ORDER BY promoted_at DESC, id DESC
           LIMIT 1""",
        (experiment_id,),
    )
    row = cur.fetchone()
    if row is None:
        raise GovernedChampionLineageError(
            "The active champion has no governed promotion audit"
        )
    snapshot = _decode_snapshot(row[0])
    models = snapshot.get("models")
    raw_run_ids = snapshot.get("backtest_run_ids")
    if (
        snapshot.get("_promotion_mode") != GOVERNED_PROMOTION_MODE
        or not isinstance(models, list)
        or tuple(models) != CANONICAL_CHAMPION_MODELS
        or not isinstance(raw_run_ids, list)
        or len(raw_run_ids) != len(CANONICAL_CHAMPION_MODELS)
    ):
        raise GovernedChampionLineageError(
            "The active champion was not promoted by the canonical five-model workflow"
        )

    run_ids: dict[str, int] = {}
    for pair in raw_run_ids:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not isinstance(pair[0], str)
            or not _positive_integer(pair[1])
            or pair[0] in run_ids
        ):
            raise GovernedChampionLineageError(
                "Champion promotion audit has invalid backtest run lineage"
            )
        run_ids[pair[0]] = int(pair[1])
    if tuple(run_ids) != CANONICAL_CHAMPION_MODELS:
        raise GovernedChampionLineageError(
            "Champion promotion audit does not contain the exact five backtest runs"
        )

    source_sales_batch_id = snapshot.get("source_sales_batch_id")
    data_checksum = snapshot.get("data_checksum")
    cluster_experiment_id = snapshot.get("cluster_experiment_id")
    cluster_assignment_count = snapshot.get("cluster_assignment_count")
    cluster_assignment_checksum = snapshot.get("cluster_assignment_checksum")
    if (
        not _positive_integer(source_sales_batch_id)
        or not _sha256(data_checksum)
        or not _positive_integer(cluster_experiment_id)
        or not _positive_integer(cluster_assignment_count)
        or not _sha256(cluster_assignment_checksum)
    ):
        raise GovernedChampionLineageError(
            "Champion promotion audit has incomplete sales or cluster lineage"
        )
    return {
        "experiment_id": int(experiment_id),
        "models": list(CANONICAL_CHAMPION_MODELS),
        "backtest_run_ids": run_ids,
        "source_sales_batch_id": int(source_sales_batch_id),
        "data_checksum": str(data_checksum),
        "cluster_experiment_id": int(cluster_experiment_id),
        "cluster_assignment_count": int(cluster_assignment_count),
        "cluster_assignment_checksum": str(cluster_assignment_checksum),
    }


def load_active_governed_champion_lineage(cur: Any) -> dict[str, Any]:
    """Load governed lineage for the one active config-and-results experiment."""
    cur.execute(
        """SELECT experiment_id
           FROM champion_experiment
           WHERE is_promoted = TRUE
             AND is_results_promoted = TRUE
           ORDER BY promoted_at DESC, experiment_id DESC"""
    )
    rows = cur.fetchall()
    if len(rows) != 1 or not _positive_integer(rows[0][0]):
        raise GovernedChampionLineageError(
            "Exactly one governed champion experiment must be active"
        )
    return load_governed_champion_lineage(
        cur,
        experiment_id=int(rows[0][0]),
    )
