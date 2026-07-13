"""Current-lineage validation for immutable snapshot contender runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

from psycopg import sql

from common.core.paths import PROJECT_ROOT
from common.core.utils import load_forecast_pipeline_config
from common.ml.direct_model_lineage import (
    DIRECT_MODEL_CONFIG_METADATA_KEY,
    SOURCE_MODEL_ROSTER_METADATA_KEY,
    DirectModelLineageError,
    validate_direct_model_config_lineage,
)
from common.ml.generation_config_lineage import (
    GENERATION_CONFIG_METADATA_KEY,
    GenerationConfigLineageError,
    validate_generation_config_lineage,
)
from common.ml.neural_artifacts import (
    load_neural_training_cohort_identity,
    read_active_neural_artifact_ref,
)
from common.ml.neural_forecast import SUPPORTED_NEURAL_MODELS
from common.ml.tree_artifact_lineage import ProductionTreeArtifactLineage
from common.ml.tree_artifacts import (
    build_production_tree_model_config_payload,
    build_tree_artifact_spec,
    read_active_tree_artifact_ref,
)
from common.services.champion_lineage import (
    GOVERNED_CHAMPION_LINEAGE_METADATA_KEY,
    GovernedChampionLineageError,
    load_active_governed_champion_lineage,
)
from common.services.cluster_lineage import load_promoted_cluster_population
from common.services.forecast_generation import (
    GENERATOR_CONTRACT_METADATA_KEY,
    GENERATOR_CONTRACT_VERSION,
)
from common.services.forecast_lineage import (
    ForecastPayloadStats,
    compute_staging_payload_stats,
)
from common.services.forecast_population import resolve_forecast_sales_table

_SOURCE_SALES_METADATA_KEY = "source_sales"
_NEURAL_ARTIFACTS_METADATA_KEY = "neural_artifacts"
_TREE_ARTIFACTS_METADATA_KEY = "tree_artifacts"
_REQUIRED_LAGS = frozenset(range(6))
_SHA256_LENGTH = 64


class SnapshotContenderIntegrityError(ValueError):
    """A frozen contender no longer matches its immutable database evidence."""


class SnapshotContenderStaleError(ValueError):
    """A valid frozen contender no longer matches current inputs or artifacts."""


@dataclass(frozen=True, slots=True)
class CurrentSalesSource:
    """Current synchronized sales identity and latest closed source month."""

    batch_id: int
    source_hash: str
    history_end: date
    sales_table: str


def _previous_month(value: date) -> date:
    index = value.year * 12 + value.month - 2
    return date(index // 12, index % 12 + 1, 1)


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_current_sales_source(cur: Any, record_month: date) -> CurrentSalesSource:
    """Load the synchronized source identity used by a current contender."""
    cur.execute(
        """SELECT batch_id, source_hash, source_file
           FROM audit_load_batch
           WHERE domain = 'sales'
             AND status = 'completed'
             AND row_count_out > 0
           ORDER BY completed_at DESC NULLS LAST, batch_id DESC
           LIMIT 1"""
    )
    row = cur.fetchone()
    if row is None:
        raise SnapshotContenderStaleError(
            "Current completed sales lineage is unavailable; reload sales first"
        )
    batch_id, source_hash, source_file = row
    normalized_hash = str(source_hash or "").strip().lower()
    if not source_file or str(source_file).strip() == "safe_upsert":
        raise SnapshotContenderStaleError(
            "The latest sales load did not synchronize the immutable forecast source"
        )
    if not _valid_sha256(normalized_hash):
        raise SnapshotContenderStaleError(
            "The latest sales load has no valid source payload checksum"
        )

    sales_table = resolve_forecast_sales_table(cur)
    cur.execute(
        sql.SQL(
            """SELECT MAX(startdate)
               FROM {}
               WHERE type = 1
                 AND qty IS NOT NULL
                 AND startdate < %s"""
        ).format(sql.Identifier(sales_table)),
        (record_month,),
    )
    history_row = cur.fetchone()
    history_end = history_row[0] if history_row else None
    expected_history_end = _previous_month(record_month)
    if history_end != expected_history_end:
        actual = history_end.isoformat() if isinstance(history_end, date) else "unavailable"
        raise SnapshotContenderStaleError(
            "Sales history is not current through the latest closed month "
            f"{expected_history_end.isoformat()} (found {actual})"
        )
    return CurrentSalesSource(
        batch_id=int(batch_id),
        source_hash=normalized_hash,
        history_end=history_end,
        sales_table=sales_table,
    )


def _model_registry_path(config: dict[str, Any], project_root: Path) -> Path:
    production = config.get("production_forecast")
    if not isinstance(production, dict):
        raise SnapshotContenderStaleError(
            "Current production forecast configuration is unavailable"
        )
    registry = production.get("model_registry")
    raw_path = registry.get("base_path") if isinstance(registry, dict) else None
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise SnapshotContenderStaleError("Current production model registry path is unavailable")
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def _expected_neural_artifact_metadata(
    ref: Any,
    recorded: object,
) -> dict[str, Any]:
    required = {
        "artifact_id",
        "config_checksum",
        "data_checksum",
        "source_sales_batch_id",
        "history_end",
        "training_cohort_checksum",
        "training_data_checksum",
        "training_contract_version",
        "runtime_contract_checksum",
    }
    if not isinstance(recorded, dict) or not required.issubset(recorded):
        raise SnapshotContenderStaleError(
            "Snapshot contender has incomplete neural artifact lineage"
        )
    try:
        return {
            "artifact_id": ref.artifact_id,
            **{key: ref.metadata[key] for key in recorded if key != "artifact_id"},
        }
    except (AttributeError, KeyError, TypeError) as exc:
        raise SnapshotContenderStaleError(
            "The active neural artifact has incomplete lineage metadata"
        ) from exc


def _validate_current_neural_artifact(
    cur: Any,
    *,
    model_id: str,
    metadata: dict[str, Any],
    algorithms: dict[str, Any],
    current_source: CurrentSalesSource,
    base_dir: Path,
) -> None:
    raw_artifacts = metadata.get(_NEURAL_ARTIFACTS_METADATA_KEY)
    if not isinstance(raw_artifacts, dict) or set(raw_artifacts) != {model_id}:
        raise SnapshotContenderStaleError(
            f"{model_id} snapshot contender is missing exact neural artifact lineage"
        )
    algorithm = algorithms.get(model_id)
    params = algorithm.get("params") if isinstance(algorithm, dict) else None
    if not isinstance(params, dict):
        raise SnapshotContenderStaleError(
            f"Current {model_id} production parameters are unavailable"
        )
    try:
        min_history = int(params["min_history"])
        cohort = load_neural_training_cohort_identity(
            cur.connection,
            sales_table=current_source.sales_table,
            history_end=current_source.history_end,
            min_history=min_history,
        )
        ref = read_active_neural_artifact_ref(
            model_id=model_id,
            base_dir=base_dir,
            expected_params=params,
            expected_source_sales_batch_id=current_source.batch_id,
            expected_data_checksum=current_source.source_hash,
            expected_history_end=current_source.history_end,
            expected_training_cohort_checksum=cohort.checksum,
            expected_training_dfu_count=cohort.dfu_count,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SnapshotContenderStaleError(
            f"Current {model_id} neural artifact is unavailable or stale"
        ) from exc
    recorded = raw_artifacts[model_id]
    expected = _expected_neural_artifact_metadata(ref, recorded)
    if recorded != expected:
        raise SnapshotContenderStaleError(
            f"Current {model_id} neural artifact differs from the snapshot contender"
        )


def _validate_current_tree_artifact(
    cur: Any,
    *,
    model_id: str,
    metadata: dict[str, Any],
    config: dict[str, Any],
    current_source: CurrentSalesSource,
    base_dir: Path,
    project_root: Path,
) -> None:
    raw_artifacts = metadata.get(_TREE_ARTIFACTS_METADATA_KEY)
    if not isinstance(raw_artifacts, dict) or set(raw_artifacts) != {model_id}:
        raise SnapshotContenderStaleError(
            "LightGBM snapshot contender is missing exact tree artifact lineage"
        )
    clustering = config.get("clustering")
    clustering_enabled = clustering.get("enabled") if isinstance(clustering, dict) else None
    if not isinstance(clustering_enabled, bool):
        raise SnapshotContenderStaleError("Current production clustering mode is unavailable")
    if clustering_enabled:
        try:
            population = load_promoted_cluster_population(cur.connection)
        except (AttributeError, RuntimeError, ValueError) as exc:
            raise SnapshotContenderStaleError(
                "Current promoted cluster assignment lineage is unavailable"
            ) from exc
        cluster_experiment_id = population.experiment_id
        assignment_count = population.assignment_count
        assignment_checksum = population.assignment_checksum
        cluster_labels = population.cluster_labels
        cluster_strategy = "per_cluster"
    else:
        cluster_experiment_id = None
        assignment_count = None
        assignment_checksum = None
        cluster_labels = {"global"}
        cluster_strategy = "global"

    try:
        lineage = ProductionTreeArtifactLineage(
            source_sales_batch_id=current_source.batch_id,
            data_checksum=current_source.source_hash,
            history_end=current_source.history_end,
            cluster_experiment_id=cluster_experiment_id,
            cluster_assignment_count=assignment_count,
            cluster_assignment_checksum=assignment_checksum,
        )
        spec = build_tree_artifact_spec(
            model_id=model_id,
            model_config=build_production_tree_model_config_payload(
                config,
                model_id=model_id,
                project_root=project_root,
            ),
            lineage=lineage,
            cluster_strategy=cluster_strategy,
            cluster_labels=cluster_labels,
        )
        ref = read_active_tree_artifact_ref(
            model_id=model_id,
            base_dir=base_dir,
            expected_spec=spec,
        )
        recorded = raw_artifacts[model_id]
        required = {
            "artifact_set_id",
            "config_checksum",
            "cluster_strategy",
            "cluster_labels",
            "lineage",
        }
        if not isinstance(recorded, dict) or not required.issubset(recorded):
            raise ValueError("snapshot contender tree artifact lineage is incomplete")
        expected = {
            "artifact_set_id": ref.artifact_set_id,
            **{key: ref.metadata[key] for key in recorded if key != "artifact_set_id"},
        }
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SnapshotContenderStaleError(
            "Current LightGBM artifact set is unavailable or stale"
        ) from exc
    if recorded != expected:
        raise SnapshotContenderStaleError(
            "Current LightGBM artifact set differs from the snapshot contender"
        )


def _validate_current_model_lineage(
    cur: Any,
    *,
    model_id: str,
    metadata: dict[str, Any],
    current_source: CurrentSalesSource,
    project_root: Path,
) -> None:
    """Validate config plus the active artifact family relevant to one model."""
    raw_roster = metadata.get(SOURCE_MODEL_ROSTER_METADATA_KEY)
    if raw_roster != [model_id]:
        raise SnapshotContenderStaleError(
            "Snapshot contender source-model roster does not match its selected model"
        )
    config = load_forecast_pipeline_config()
    algorithms = config.get("algorithms")
    if not isinstance(algorithms, dict) or model_id not in algorithms:
        raise SnapshotContenderStaleError(
            f"Snapshot contender model {model_id!r} is not in the current roster"
        )
    try:
        validate_direct_model_config_lineage(
            metadata.get(DIRECT_MODEL_CONFIG_METADATA_KEY, {}),
            algorithms=algorithms,
            required_model_ids={model_id},
        )
        validate_generation_config_lineage(
            metadata.get(GENERATION_CONFIG_METADATA_KEY),
            pipeline_config=config,
            source_model_ids={model_id},
        )
    except (DirectModelLineageError, GenerationConfigLineageError) as exc:
        raise SnapshotContenderStaleError(
            f"Current generation configuration differs for {model_id}"
        ) from exc

    base_dir = _model_registry_path(config, project_root)
    if model_id in SUPPORTED_NEURAL_MODELS:
        if metadata.get(_TREE_ARTIFACTS_METADATA_KEY) not in (None, {}):
            raise SnapshotContenderStaleError(
                f"{model_id} contender has unrelated tree artifact lineage"
            )
        _validate_current_neural_artifact(
            cur,
            model_id=model_id,
            metadata=metadata,
            algorithms=algorithms,
            current_source=current_source,
            base_dir=base_dir,
        )
    elif model_id == "lgbm_cluster":
        if metadata.get(_NEURAL_ARTIFACTS_METADATA_KEY) not in (None, {}):
            raise SnapshotContenderStaleError(
                "LightGBM contender has unrelated neural artifact lineage"
            )
        _validate_current_tree_artifact(
            cur,
            model_id=model_id,
            metadata=metadata,
            config=config,
            current_source=current_source,
            base_dir=base_dir,
            project_root=project_root,
        )
    elif model_id in {"mstl", "chronos2_enriched"}:
        if metadata.get(_NEURAL_ARTIFACTS_METADATA_KEY) not in (None, {}) or metadata.get(
            _TREE_ARTIFACTS_METADATA_KEY
        ) not in (None, {}):
            raise SnapshotContenderStaleError(
                f"{model_id} contender has unrelated fitted artifact lineage"
            )
    else:
        raise SnapshotContenderStaleError(f"Unsupported snapshot contender model {model_id!r}")


def _validate_source_metadata(
    metadata: dict[str, Any],
    *,
    current: CurrentSalesSource,
) -> None:
    source = metadata.get(_SOURCE_SALES_METADATA_KEY)
    expected = {
        "source_sales_batch_id": current.batch_id,
        "data_checksum": current.source_hash,
        "history_end": current.history_end.isoformat(),
    }
    if source != expected:
        raise SnapshotContenderStaleError(
            "Snapshot contender sales batch, payload hash, or history month is stale"
        )


def validate_ready_snapshot_contender(
    cur: Any,
    *,
    run_id: UUID | str,
    model_id: str,
    record_month: date,
    source_backtest_run_id: int | None = None,
    project_root: Path = PROJECT_ROOT,
) -> ForecastPayloadStats:
    """Validate one ready contender against immutable and current evidence."""
    cur.execute(
        """SELECT generation_purpose, requested_model_id,
                  forecast_month_generated, horizon_months, run_status,
                  promotion_eligible, row_count, dfu_count,
                  candidate_model_count, source_sales_batch_id,
                  artifact_checksum, metadata
           FROM forecast_generation_run
           WHERE run_id = %s::uuid""",
        (str(run_id),),
    )
    row = cur.fetchone()
    if row is None:
        raise SnapshotContenderIntegrityError(f"{model_id} snapshot generation manifest is missing")
    identity = (str(row[0]), str(row[1]), row[2], int(row[3] or 0))
    if identity != ("snapshot_contender", model_id, record_month, 6):
        raise SnapshotContenderIntegrityError(
            f"{model_id} snapshot generation manifest has a different identity"
        )
    if str(row[4]) != "ready" or bool(row[5]):
        raise SnapshotContenderIntegrityError(
            f"{model_id} snapshot generation manifest is not ready non-release evidence"
        )
    metadata = dict(row[11]) if isinstance(row[11], dict) else {}
    if metadata.get(GENERATOR_CONTRACT_METADATA_KEY) != GENERATOR_CONTRACT_VERSION:
        raise SnapshotContenderStaleError(
            f"{model_id} snapshot generation manifest uses an outdated generator contract"
        )
    stats = compute_staging_payload_stats(cur, run_id)
    expected_stats = (
        int(row[6] or 0),
        int(row[7] or 0),
        int(row[8] or 0),
        str(row[10] or ""),
    )
    actual_stats = (
        stats.row_count,
        stats.dfu_count,
        stats.source_model_count,
        stats.checksum,
    )
    if actual_stats != expected_stats or stats.row_count <= 0:
        raise SnapshotContenderIntegrityError(
            f"{model_id} staged payload no longer matches its generation manifest"
        )

    cur.execute(
        """SELECT
                  ((EXTRACT(YEAR FROM forecast_month)
                    - EXTRACT(YEAR FROM %s::date)) * 12
                   + (EXTRACT(MONTH FROM forecast_month)
                      - EXTRACT(MONTH FROM %s::date)))::integer AS lag,
                  COUNT(*)::integer
           FROM fact_production_forecast_staging
           WHERE run_id = %s::uuid
             AND generation_purpose = 'snapshot_contender'
             AND candidate_model_id = %s
             AND model_id = %s
             AND forecast_month_generated = %s
           GROUP BY 1
           ORDER BY 1""",
        (
            record_month,
            record_month,
            str(run_id),
            model_id,
            model_id,
            record_month,
        ),
    )
    lag_counts = {int(lag): int(count) for lag, count in cur.fetchall()}
    if set(lag_counts) != _REQUIRED_LAGS or sum(lag_counts.values()) != stats.row_count:
        raise SnapshotContenderIntegrityError(
            f"{model_id} snapshot payload must contain exactly lags 0..5"
        )

    try:
        current_governed_lineage = load_active_governed_champion_lineage(cur)
    except GovernedChampionLineageError as exc:
        raise SnapshotContenderStaleError(
            "The active governed champion lineage is unavailable"
        ) from exc
    expected_backtest_run_id = current_governed_lineage["backtest_run_ids"].get(
        model_id
    )
    recorded_backtest_run_id = metadata.get("source_backtest_run_id")
    if (
        metadata.get(GOVERNED_CHAMPION_LINEAGE_METADATA_KEY)
        != current_governed_lineage
        or not isinstance(recorded_backtest_run_id, int)
        or recorded_backtest_run_id != expected_backtest_run_id
        or (
            source_backtest_run_id is not None
            and recorded_backtest_run_id != source_backtest_run_id
        )
    ):
        raise SnapshotContenderStaleError(
            f"{model_id} snapshot contender does not match the governed backtest run"
        )

    current_source = _load_current_sales_source(cur, record_month)
    if row[9] is None or int(row[9]) != current_source.batch_id:
        raise SnapshotContenderStaleError(
            f"{model_id} snapshot manifest references a stale sales batch"
        )
    current_clusters = load_promoted_cluster_population(cur.connection)
    if (
        current_governed_lineage["source_sales_batch_id"] != current_source.batch_id
        or current_governed_lineage["data_checksum"] != current_source.source_hash
        or current_governed_lineage["cluster_experiment_id"]
        != current_clusters.experiment_id
        or current_governed_lineage["cluster_assignment_count"]
        != current_clusters.assignment_count
        or current_governed_lineage["cluster_assignment_checksum"]
        != current_clusters.assignment_checksum
    ):
        raise SnapshotContenderStaleError(
            "The governed champion no longer matches current sales or clustering"
        )
    _validate_source_metadata(metadata, current=current_source)
    _validate_current_model_lineage(
        cur,
        model_id=model_id,
        metadata=metadata,
        current_source=current_source,
        project_root=project_root,
    )
    return stats
