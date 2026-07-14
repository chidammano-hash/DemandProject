"""Run-scoped, all-or-nothing forecast release promotion."""

from __future__ import annotations

import logging
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

from psycopg.types.json import Jsonb

from common.core.constants import CHAMPION_MODEL_ID, CUSTOMER_BOTTOM_UP_BLEND_MODEL_ID
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
from common.ml.tree_artifact_lineage import (
    ProductionTreeArtifactLineage,
    TreeArtifactLineageError,
)
from common.ml.tree_artifacts import read_active_tree_artifact_ref
from common.services.champion_lineage import (
    GOVERNED_CHAMPION_LINEAGE_METADATA_KEY,
    GovernedChampionLineageError,
    load_governed_champion_lineage,
)
from common.services.cluster_lineage import load_promoted_cluster_population
from common.services.customer_demand_lineage import customer_demand_snapshot_lock
from common.services.customer_forecast_blend import (
    compute_customer_blend_component_stats,
    load_customer_blend_component_lineage,
)
from common.services.customer_forecast_blend_contract import (
    CUSTOMER_BLEND_CONTRACT_VERSION,
    CUSTOMER_BLEND_LINEAGE_METADATA_KEY,
    customer_blend_config_checksum,
    validate_blend_settings,
)
from common.services.customer_forecast_blend_evidence import (
    compute_customer_forecast_output_stats,
)
from common.services.forecast_generation import (
    GENERATOR_CONTRACT_METADATA_KEY,
    GENERATOR_CONTRACT_VERSION,
)
from common.services.forecast_lineage import (
    ForecastPayloadStats,
    compute_champion_results_stats,
    compute_production_payload_stats,
    compute_staging_payload_stats,
    sha256_file,
)
from common.services.forecast_population import (
    build_forecast_eligibility_ctes,
    resolve_forecast_sales_table,
)
from common.services.forecast_release import (
    ReleaseQualityMetrics,
    ReleaseReadinessThresholds,
    evaluate_quality_checks,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForecastGenerationManifest:
    run_id: UUID
    generation_purpose: str
    run_status: str
    promotion_eligible: bool
    requested_model_id: str
    forecast_month_generated: date
    horizon_months: int
    row_count: int
    dfu_count: int
    source_model_count: int
    champion_experiment_id: int | None
    cluster_experiment_id: int | None
    source_sales_batch_id: int | None
    routing_artifact_checksum: str | None
    champion_results_checksum: str | None
    artifact_checksum: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PromotionResult:
    model_id: str
    promotion_type: str
    plan_version: str
    source_run_id: UUID
    production_run_id: UUID
    candidate_checksum: str
    outgoing_archive_checksum: str | None
    rows_promoted: int
    dfu_count: int


@dataclass(frozen=True)
class ForecastStagingResult:
    model_id: str
    source_run_id: UUID
    status: str
    rows_staged: int
    dfu_count: int
    candidate_checksum: str


class PromotionConflictError(ValueError):
    """Expected fail-closed release rejection with a stable public code."""

    def __init__(self, code: str, message: str, *, status_code: int = 409):
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.status_code = status_code


def _add_months(value: date, months: int) -> date:
    index = value.year * 12 + (value.month - 1) + months
    return date(index // 12, index % 12 + 1, 1)


def validate_generation_manifest(
    manifest: ForecastGenerationManifest,
    *,
    model_id: str,
    planning_month: date,
    required_months: int,
    require_staged: bool = True,
) -> None:
    """Reject a source manifest that cannot identify one safe release candidate."""
    if manifest.generation_purpose != "release_candidate" or manifest.run_status != "ready":
        raise PromotionConflictError(
            "candidate_run_not_promotable",
            "The selected generation run is not eligible for release.",
        )
    if require_staged and not manifest.promotion_eligible:
        raise PromotionConflictError(
            "candidate_not_staged",
            "Promote the selected generated candidate to staging first.",
        )
    if manifest.metadata.get(GENERATOR_CONTRACT_METADATA_KEY) != GENERATOR_CONTRACT_VERSION:
        raise PromotionConflictError(
            "candidate_lineage_mismatch",
            "The selected generation run was produced by an outdated forecast generator; "
            "generate a new release candidate.",
        )
    if manifest.requested_model_id != model_id:
        raise PromotionConflictError(
            "candidate_lineage_mismatch",
            "The selected generation run belongs to a different release model.",
        )
    if manifest.forecast_month_generated != planning_month:
        raise PromotionConflictError(
            "stale_candidate_evidence",
            "The selected generation run is not from the current planning month.",
        )
    if manifest.horizon_months < required_months:
        raise PromotionConflictError(
            "candidate_gate_failed",
            f"The release candidate must cover at least {required_months} months.",
        )
    if manifest.row_count <= 0 or manifest.dfu_count <= 0:
        raise PromotionConflictError(
            "candidate_gate_failed",
            "The release candidate contains no usable forecast population.",
        )
    if not manifest.artifact_checksum:
        raise PromotionConflictError(
            "candidate_lineage_mismatch",
            "The release candidate is missing its immutable payload checksum.",
        )


def _load_manifest(cur: Any, source_run_id: UUID) -> ForecastGenerationManifest:
    cur.execute(
        """SELECT run_id, generation_purpose, run_status, promotion_eligible,
                  requested_model_id, forecast_month_generated, horizon_months,
                  row_count, dfu_count, candidate_model_count,
                  champion_experiment_id, cluster_experiment_id,
                  source_sales_batch_id, routing_artifact_checksum,
                  champion_results_checksum, artifact_checksum, metadata
           FROM forecast_generation_run
           WHERE run_id = %s::uuid
           FOR UPDATE""",
        (str(source_run_id),),
    )
    row = cur.fetchone()
    if row is None:
        raise PromotionConflictError(
            "candidate_run_not_found",
            "The selected forecast generation run does not exist.",
            status_code=404,
        )
    return ForecastGenerationManifest(
        run_id=UUID(str(row[0])),
        generation_purpose=str(row[1]),
        run_status=str(row[2]),
        promotion_eligible=bool(row[3]),
        requested_model_id=str(row[4]),
        forecast_month_generated=row[5],
        horizon_months=int(row[6] or 0),
        row_count=int(row[7] or 0),
        dfu_count=int(row[8] or 0),
        source_model_count=int(row[9] or 0),
        champion_experiment_id=int(row[10]) if row[10] is not None else None,
        cluster_experiment_id=int(row[11]) if row[11] is not None else None,
        source_sales_batch_id=int(row[12]) if row[12] is not None else None,
        routing_artifact_checksum=str(row[13]) if row[13] else None,
        champion_results_checksum=str(row[14]) if row[14] else None,
        artifact_checksum=str(row[15]) if row[15] else None,
        metadata=dict(row[16]) if isinstance(row[16], dict) else {},
    )


def _validate_manifest_payload(
    manifest: ForecastGenerationManifest,
    stats: ForecastPayloadStats,
) -> None:
    if (
        stats.row_count != manifest.row_count
        or stats.dfu_count != manifest.dfu_count
        or stats.source_model_count != manifest.source_model_count
        or stats.checksum != manifest.artifact_checksum
    ):
        raise PromotionConflictError(
            "candidate_lineage_mismatch",
            "The staged forecast payload no longer matches its generation manifest.",
        )


def stage_forecast_run(
    conn: Any,
    *,
    model_id: str,
    source_run_id: UUID,
    planning_month: date,
) -> ForecastStagingResult:
    """Approve one immutable generated candidate for later production promotion."""
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext('forecast_release_staging'))")
            manifest = _load_manifest(cur, source_run_id)
            validate_generation_manifest(
                manifest,
                model_id=model_id,
                planning_month=planning_month,
                required_months=1,
                require_staged=False,
            )
            stats = compute_staging_payload_stats(cur, source_run_id)
            _validate_manifest_payload(manifest, stats)
            is_customer_blend = (
                CUSTOMER_BLEND_LINEAGE_METADATA_KEY in manifest.metadata
                or _is_customer_blend_payload(cur, manifest.run_id)
            )
            if is_customer_blend:
                _validate_customer_bottom_up_blend(
                    cur,
                    manifest=manifest,
                    pipeline_config=load_forecast_pipeline_config(),
                    require_lineage=True,
                )
            status = "already_staged" if manifest.promotion_eligible else "staged"
            if not manifest.promotion_eligible:
                cur.execute(
                    """UPDATE forecast_generation_run
                       SET promotion_eligible = TRUE
                       WHERE run_id = %s::uuid
                         AND run_status = 'ready'
                         AND promotion_eligible = FALSE""",
                    (str(source_run_id),),
                )
                if cur.rowcount != 1:
                    raise PromotionConflictError(
                        "concurrent_staging_conflict",
                        "The generated candidate changed while it was being staged.",
                    )
            return ForecastStagingResult(
                model_id=model_id,
                source_run_id=source_run_id,
                status=status,
                rows_staged=stats.row_count,
                dfu_count=stats.dfu_count,
                candidate_checksum=stats.checksum,
            )


def _validate_direct_model_lineage(
    cur: Any,
    *,
    manifest: ForecastGenerationManifest,
    model_id: str,
    pipeline_config: dict[str, Any] | None = None,
) -> None:
    """Reject candidates whose direct-adapter config changed after generation."""
    del cur  # The pre-aggregation roster is immutable manifest evidence.
    raw_roster = manifest.metadata.get(SOURCE_MODEL_ROSTER_METADATA_KEY)
    if (
        not isinstance(raw_roster, list)
        or not raw_roster
        or any(not isinstance(value, str) or not value.strip() for value in raw_roster)
        or raw_roster != sorted(set(raw_roster))
    ):
        raise PromotionConflictError(
            "candidate_lineage_mismatch",
            "The forecast candidate is missing its canonical source-model roster.",
        )
    source_model_ids = set(raw_roster)
    current = pipeline_config if pipeline_config is not None else load_forecast_pipeline_config()
    algorithms = current.get("algorithms")
    if not isinstance(algorithms, dict):
        raise PromotionConflictError(
            "candidate_lineage_mismatch",
            "Current direct-model configuration is unavailable.",
        )
    unknown_models = sorted(source_model_ids - set(algorithms))
    if unknown_models:
        raise PromotionConflictError(
            "candidate_lineage_mismatch",
            "The forecast candidate contains a model outside the current canonical roster.",
        )
    if model_id != CHAMPION_MODEL_ID and source_model_ids != {model_id}:
        raise PromotionConflictError(
            "candidate_lineage_mismatch",
            "The single-model candidate source roster does not match its requested model.",
        )
    try:
        validate_direct_model_config_lineage(
            manifest.metadata.get(DIRECT_MODEL_CONFIG_METADATA_KEY, {}),
            algorithms=algorithms,
            required_model_ids=source_model_ids,
        )
        validate_generation_config_lineage(
            manifest.metadata.get(GENERATION_CONFIG_METADATA_KEY),
            pipeline_config=current,
            source_model_ids=source_model_ids,
        )
    except (DirectModelLineageError, GenerationConfigLineageError) as exc:
        raise PromotionConflictError(
            "candidate_lineage_mismatch",
            "A direct forecast model changed after this candidate was generated; "
            "generate a new release candidate.",
        ) from exc


def _validate_governed_champion_source(
    cur: Any,
    *,
    manifest: ForecastGenerationManifest,
) -> None:
    """Bind a champion candidate to the exact governed champion-refresh inputs."""
    recorded = manifest.metadata.get(GOVERNED_CHAMPION_LINEAGE_METADATA_KEY)
    source = manifest.metadata.get("source_sales")
    if manifest.champion_experiment_id is None:
        raise PromotionConflictError(
            "candidate_lineage_mismatch",
            "The champion candidate has no governed experiment lineage.",
        )
    try:
        current = load_governed_champion_lineage(
            cur,
            experiment_id=int(manifest.champion_experiment_id),
        )
    except GovernedChampionLineageError as exc:
        raise PromotionConflictError(
            "candidate_lineage_mismatch",
            "The active champion lacks governed source evidence; run champion-refresh.",
        ) from exc
    if (
        not isinstance(recorded, dict)
        or not isinstance(source, dict)
        or recorded != current
        or manifest.source_sales_batch_id != current["source_sales_batch_id"]
        or manifest.cluster_experiment_id != current["cluster_experiment_id"]
        or source.get("source_sales_batch_id") != current["source_sales_batch_id"]
        or source.get("data_checksum") != current["data_checksum"]
    ):
        raise PromotionConflictError(
            "candidate_lineage_mismatch",
            "The champion candidate and governed champion-refresh use different sales, "
            "cluster, or backtest lineage; run model-refresh, champion-refresh, then "
            "prepare a new release.",
        )


def _validate_active_model_artifacts(
    *,
    conn: Any,
    cur: Any,
    manifest: ForecastGenerationManifest,
    pipeline_config: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> None:
    """Require the exact persisted artifact versions used during generation."""
    raw_roster = manifest.metadata.get(SOURCE_MODEL_ROSTER_METADATA_KEY)
    if not isinstance(raw_roster, list):
        raise PromotionConflictError(
            "candidate_lineage_mismatch",
            "The forecast candidate is missing its canonical source-model roster.",
        )
    production = pipeline_config.get("production_forecast")
    registry = production.get("model_registry") if isinstance(production, dict) else None
    raw_base_path = registry.get("base_path") if isinstance(registry, dict) else None
    if not isinstance(raw_base_path, str) or not raw_base_path.strip():
        raise PromotionConflictError(
            "candidate_lineage_mismatch",
            "The production model registry path is unavailable.",
        )
    base_dir = Path(raw_base_path)
    if not base_dir.is_absolute():
        base_dir = project_root / base_dir

    try:
        if "lgbm_cluster" in raw_roster:
            tree_artifacts = manifest.metadata.get("tree_artifacts")
            tree_entry = (
                tree_artifacts.get("lgbm_cluster") if isinstance(tree_artifacts, dict) else None
            )
            generated_id = (
                tree_entry.get("artifact_set_id") if isinstance(tree_entry, dict) else None
            )
            if not isinstance(generated_id, str) or not generated_id:
                raise ValueError("candidate tree artifact ID is missing")
            active_tree = read_active_tree_artifact_ref(
                model_id="lgbm_cluster",
                base_dir=base_dir,
            )
            if active_tree.artifact_set_id != generated_id:
                raise ValueError("active tree artifact changed after generation")

        neural_artifacts = manifest.metadata.get("neural_artifacts")
        algorithms = pipeline_config.get("algorithms")
        sales_table: str | None = None
        for neural_model_id in sorted(set(raw_roster) & SUPPORTED_NEURAL_MODELS):
            neural_entry = (
                neural_artifacts.get(neural_model_id)
                if isinstance(neural_artifacts, dict)
                else None
            )
            generated_id = (
                neural_entry.get("artifact_id") if isinstance(neural_entry, dict) else None
            )
            algorithm_entry = (
                algorithms.get(neural_model_id) if isinstance(algorithms, dict) else None
            )
            expected_params = (
                algorithm_entry.get("params") if isinstance(algorithm_entry, dict) else None
            )
            history_end = (
                neural_entry.get("history_end") if isinstance(neural_entry, dict) else None
            )
            generated_cohort_checksum = (
                neural_entry.get("training_cohort_checksum")
                if isinstance(neural_entry, dict)
                else None
            )
            if (
                not isinstance(generated_id, str)
                or not generated_id
                or not isinstance(expected_params, dict)
                or not isinstance(history_end, str)
                or not history_end
                or not isinstance(generated_cohort_checksum, str)
                or not generated_cohort_checksum
            ):
                raise ValueError("candidate neural artifact lineage is missing")
            min_history = int(expected_params["min_history"])
            if sales_table is None:
                sales_table = resolve_forecast_sales_table(cur)
            current_cohort = load_neural_training_cohort_identity(
                conn,
                sales_table=sales_table,
                history_end=history_end,
                min_history=min_history,
            )
            if current_cohort.checksum != generated_cohort_checksum:
                raise ValueError("neural training cohort changed after generation")
            active_neural = read_active_neural_artifact_ref(
                model_id=neural_model_id,
                base_dir=base_dir,
                expected_params=expected_params,
                expected_source_sales_batch_id=int(neural_entry["source_sales_batch_id"]),
                expected_data_checksum=str(neural_entry["data_checksum"]),
                expected_history_end=history_end,
                expected_training_cohort_checksum=current_cohort.checksum,
                expected_training_dfu_count=current_cohort.dfu_count,
            )
            if active_neural.artifact_id != generated_id:
                raise ValueError("active neural artifact changed after generation")
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise PromotionConflictError(
            "candidate_lineage_mismatch",
            "A production model artifact changed after this candidate was generated; "
            "generate a new release candidate.",
        ) from exc


def _validate_tree_model_lineage(
    conn: Any,
    *,
    manifest: ForecastGenerationManifest,
) -> None:
    """Require current assignment identity for every candidate that used LightGBM."""
    raw_roster = manifest.metadata.get(SOURCE_MODEL_ROSTER_METADATA_KEY)
    if not isinstance(raw_roster, list) or "lgbm_cluster" not in raw_roster:
        return
    tree_artifacts = manifest.metadata.get("tree_artifacts")
    if not isinstance(tree_artifacts, dict):
        raise PromotionConflictError(
            "candidate_lineage_mismatch",
            "The LightGBM candidate is missing immutable tree artifact lineage.",
        )
    tree_metadata = tree_artifacts.get("lgbm_cluster")
    if not isinstance(tree_metadata, dict):
        raise PromotionConflictError(
            "candidate_lineage_mismatch",
            "The LightGBM candidate is missing its artifact-set identity.",
        )
    try:
        lineage = ProductionTreeArtifactLineage.from_metadata(tree_metadata["lineage"])
    except (KeyError, TypeError, TreeArtifactLineageError) as exc:
        raise PromotionConflictError(
            "candidate_lineage_mismatch",
            "The LightGBM candidate has invalid clustering lineage.",
        ) from exc
    if lineage.cluster_experiment_id is None:
        return
    current = load_promoted_cluster_population(conn)
    if (
        manifest.cluster_experiment_id != current.experiment_id
        or lineage.cluster_experiment_id != current.experiment_id
        or lineage.cluster_assignment_count != current.assignment_count
        or lineage.cluster_assignment_checksum != current.assignment_checksum
    ):
        raise PromotionConflictError(
            "candidate_lineage_mismatch",
            "Promoted cluster assignments changed after this forecast candidate was generated.",
        )


def _validate_customer_bottom_up_blend(
    cur: Any,
    *,
    manifest: ForecastGenerationManifest,
    pipeline_config: dict[str, Any],
    require_lineage: bool = False,
) -> dict[str, Any] | None:
    """Require current Croston backtest evidence for a derived customer blend."""
    recorded = manifest.metadata.get(CUSTOMER_BLEND_LINEAGE_METADATA_KEY)
    if recorded is None:
        if require_lineage:
            raise PromotionConflictError(
                "customer_blend_lineage_mismatch",
                "The customer bottom-up blend is missing required lineage metadata.",
            )
        return None
    if not isinstance(recorded, dict):
        raise PromotionConflictError(
            "customer_blend_lineage_mismatch",
            "The customer bottom-up blend has invalid lineage metadata.",
        )
    try:
        settings = validate_blend_settings(pipeline_config["customer_forecast"])
        from common.services.customer_forecast import (
            customer_forecast_config_checksum,
            get_customer_forecast_settings,
        )
        from common.services.customer_forecast_backtest import (
            compute_customer_backtest_component_stats,
            customer_backtest_config_checksum,
            get_customer_backtest_settings,
        )

        current_customer_settings = get_customer_forecast_settings()
        current_backtest_settings = get_customer_backtest_settings()
    except (KeyError, TypeError, ValueError) as exc:
        raise PromotionConflictError(
            "customer_blend_lineage_mismatch",
            "The current customer bottom-up blend policy is invalid.",
        ) from exc
    if not settings.promotion_enabled:
        raise PromotionConflictError(
            "customer_blend_promotion_disabled",
            f"Customer blend promotion is disabled: {settings.promotion_reason}.",
        )
    required_strings = (
        "customer_run_id",
        "customer_config_checksum",
        "customer_source_checksum",
        "customer_output_checksum",
        "source_run_id",
        "source_production_run_id",
        "source_production_checksum",
        "backtest_run_id",
        "backtest_config_checksum",
        "backtest_component_checksum",
        "component_checksum",
    )
    required_positive_integers = (
        "source_customer_demand_batch_id",
        "customer_output_row_count",
        "customer_output_series_count",
        "backtest_component_rows",
        "row_count",
        "dfu_count",
    )
    required_nonnegative_integers = (
        "blended_row_count",
        "fallback_row_count",
    )
    if (
        recorded.get("contract_version") != CUSTOMER_BLEND_CONTRACT_VERSION
        or recorded.get("model_id") != settings.model_id
        or recorded.get("config_checksum") != customer_blend_config_checksum(settings)
        or recorded.get("customer_config_checksum")
        != customer_forecast_config_checksum(current_customer_settings)
        or recorded.get("backtest_config_checksum")
        != customer_backtest_config_checksum(
            current_backtest_settings,
            settings,
            current_customer_settings,
        )
        or any(
            not isinstance(recorded.get(key), str) or not recorded[key] for key in required_strings
        )
        or not isinstance(recorded.get("source_promotion_id"), int)
        or any(
            type(recorded.get(key)) is not int or recorded[key] <= 0
            for key in required_positive_integers
        )
        or any(
            type(recorded.get(key)) is not int or recorded[key] < 0
            for key in required_nonnegative_integers
        )
        or not isinstance(recorded.get("backtest_gate"), dict)
        or recorded["backtest_gate"].get("passed") is not True
    ):
        raise PromotionConflictError(
            "customer_blend_lineage_mismatch",
            "The customer bottom-up blend is missing current configuration or backtest lineage.",
        )

    cur.execute(
        """SELECT promotion.source_run_id, promotion.production_run_id,
                  promotion.production_checksum,
                  customer.config_checksum, customer.source_checksum,
                  customer.row_count, customer.eligible_series,
                  backtest.config_checksum, backtest.component_checksum,
                  backtest.component_rows,
                  accuracy.gate_passed, accuracy.gate_reason,
                  customer.source_customer_demand_batch_id,
                  (
                      SELECT batch_id
                      FROM audit_load_batch
                      WHERE domain = 'customer_demand'
                        AND status = 'completed'
                      ORDER BY completed_at DESC NULLS LAST, batch_id DESC
                      LIMIT 1
                  ),
                  (
                      SELECT source_batch_id
                      FROM customer_demand_profile_refresh_state
                      WHERE singleton_id = 1
                  ),
                  (
                      SELECT COUNT(*)
                      FROM audit_load_batch
                      WHERE domain = 'customer_demand'
                        AND status = 'running'
                  )
           FROM model_promotion_log promotion
           JOIN customer_forecast_run customer
             ON customer.run_id = %s::uuid
            AND customer.run_status = 'completed'
           JOIN customer_forecast_backtest_run backtest
             ON backtest.run_id = %s::uuid
            AND backtest.run_status = 'completed'
            AND backtest.customer_run_id = customer.run_id
            AND backtest.source_promotion_id = promotion.id
            AND backtest.source_production_run_id = promotion.production_run_id
           JOIN customer_bottom_up_backtest_accuracy accuracy
             ON accuracy.backtest_run_id = backtest.run_id
           WHERE promotion.id = %s
             AND promotion.is_active = TRUE
             AND promotion.model_id = 'champion'
           FOR SHARE OF promotion""",
        (
            recorded["customer_run_id"],
            recorded["backtest_run_id"],
            int(recorded["source_promotion_id"]),
        ),
    )
    row = cur.fetchone()
    if row is None or (
        str(row[0]) != recorded["source_run_id"]
        or str(row[1]) != recorded["source_production_run_id"]
        or row[2] != recorded["source_production_checksum"]
        or row[3] != recorded["customer_config_checksum"]
        or row[4] != recorded["customer_source_checksum"]
        or int(row[5]) != recorded["customer_output_row_count"]
        or int(row[6]) != recorded["customer_output_series_count"]
        or row[7] != recorded["backtest_config_checksum"]
        or row[8] != recorded["backtest_component_checksum"]
        or int(row[9]) != recorded["backtest_component_rows"]
        or not bool(row[10])
        or row[12] is None
        or int(row[12]) != recorded["source_customer_demand_batch_id"]
        or row[13] is None
        or int(row[13]) != recorded["source_customer_demand_batch_id"]
        or row[14] is None
        or int(row[14]) != recorded["source_customer_demand_batch_id"]
        or int(row[15] or 0) > 0
    ):
        raise PromotionConflictError(
            "customer_blend_lineage_mismatch",
            "The customer forecast, source champion, or accuracy backtest changed after blending.",
        )
    try:
        customer_stats = compute_customer_forecast_output_stats(
            cur,
            recorded["customer_run_id"],
        )
        backtest_stats = compute_customer_backtest_component_stats(
            cur,
            recorded["backtest_run_id"],
        )
        source_stats = compute_production_payload_stats(
            cur,
            recorded["source_production_run_id"],
        )
        component_lineage = load_customer_blend_component_lineage(cur, manifest.run_id)
    except ValueError as exc:
        raise PromotionConflictError(
            "customer_blend_lineage_mismatch",
            "The customer blend source evidence could not be verified.",
        ) from exc
    if (
        customer_stats.checksum != recorded["customer_output_checksum"]
        or customer_stats.row_count != recorded["customer_output_row_count"]
        or customer_stats.series_count != recorded["customer_output_series_count"]
        or backtest_stats
        != (
            recorded["backtest_component_checksum"],
            recorded["backtest_component_rows"],
        )
        or source_stats.checksum != recorded["source_production_checksum"]
        or str(component_lineage.customer_run_id) != recorded["customer_run_id"]
        or str(component_lineage.backtest_run_id) != recorded["backtest_run_id"]
        or component_lineage.source_promotion_id != recorded["source_promotion_id"]
        or str(component_lineage.source_production_run_id) != recorded["source_production_run_id"]
    ):
        raise PromotionConflictError(
            "customer_blend_lineage_mismatch",
            "The customer forecast, source champion, or backtest payload no longer matches its lineage.",
        )
    component_stats = compute_customer_blend_component_stats(cur, manifest.run_id)
    if component_stats != (
        recorded["component_checksum"],
        recorded["row_count"],
        recorded["dfu_count"],
        recorded["blended_row_count"],
        recorded["fallback_row_count"],
    ):
        raise PromotionConflictError(
            "customer_blend_lineage_mismatch",
            "The customer bottom-up blend components no longer match their manifest.",
        )
    return {
        "customer_run_id": recorded["customer_run_id"],
        "backtest_run_id": recorded["backtest_run_id"],
        "source_promotion_id": recorded["source_promotion_id"],
        "source_customer_demand_batch_id": recorded["source_customer_demand_batch_id"],
        "customer_output_checksum": customer_stats.checksum,
        "backtest_component_checksum": backtest_stats[0],
        "component_checksum": component_stats[0],
        "backtest_gate": recorded["backtest_gate"],
    }


def _is_customer_blend_payload(cur: Any, run_id: UUID) -> bool:
    """Identify blend staging rows even when their manifest metadata is missing."""
    cur.execute(
        """SELECT EXISTS (
                   SELECT 1
                   FROM fact_production_forecast_staging
                   WHERE run_id = %s::uuid
                     AND model_id = %s
               )""",
        (str(run_id), CUSTOMER_BOTTOM_UP_BLEND_MODEL_ID),
    )
    row = cur.fetchone()
    return row is not None and row[0] is True


def _current_release_evidence(cur: Any) -> tuple[Any, ...]:
    cur.execute(
        """SELECT
               (SELECT batch_id
                FROM audit_load_batch
                WHERE domain = 'sales' AND status = 'completed'
                ORDER BY completed_at DESC NULLS LAST, batch_id DESC
                LIMIT 1),
               (SELECT COUNT(*) FROM champion_experiment
                WHERE is_results_promoted = TRUE),
               (SELECT experiment_id FROM champion_experiment
                WHERE is_results_promoted = TRUE
                ORDER BY results_promoted_at DESC NULLS LAST, experiment_id DESC
                LIMIT 1),
               (SELECT cluster_experiment_id FROM champion_experiment
                WHERE is_results_promoted = TRUE
                ORDER BY results_promoted_at DESC NULLS LAST, experiment_id DESC
                LIMIT 1),
               (SELECT results_artifact_checksum FROM champion_experiment
                WHERE is_results_promoted = TRUE
                ORDER BY results_promoted_at DESC NULLS LAST, experiment_id DESC
                LIMIT 1),
               (SELECT results_forecast_checksum FROM champion_experiment
                WHERE is_results_promoted = TRUE
                ORDER BY results_promoted_at DESC NULLS LAST, experiment_id DESC
                LIMIT 1),
               (SELECT results_forecast_row_count FROM champion_experiment
                WHERE is_results_promoted = TRUE
                ORDER BY results_promoted_at DESC NULLS LAST, experiment_id DESC
                LIMIT 1),
               (SELECT COUNT(*) FROM cluster_experiment WHERE is_promoted = TRUE),
               (SELECT experiment_id FROM cluster_experiment
                WHERE is_promoted = TRUE
                ORDER BY promoted_at DESC NULLS LAST, experiment_id DESC
                LIMIT 1),
               (SELECT COUNT(*) FROM current_sku_cluster_assignment),
               (SELECT COUNT(*) FROM cluster_tuning_profile_state tuning
                WHERE tuning.stale = TRUE
                  AND EXISTS (
                      SELECT 1 FROM current_sku_cluster_assignment assignment
                      WHERE assignment.ml_cluster = tuning.cluster_name
                  ))"""
    )
    row = cur.fetchone()
    if row is None:
        raise PromotionConflictError(
            "candidate_gate_failed",
            "Release-control evidence is unavailable.",
        )
    return row


def _candidate_coverage(
    cur: Any,
    *,
    source_run_id: UUID,
    model_id: str,
    planning_month: date,
    required_months: int,
    min_history_months: int,
    active_window_months: int,
) -> tuple[int, int, int, int, int, int]:
    end_month = _add_months(planning_month, required_months)
    sales_table = resolve_forecast_sales_table(cur)
    eligibility = build_forecast_eligibility_ctes(
        planning_month=planning_month,
        min_history_months=min_history_months,
        active_window_months=active_window_months,
        sales_table=sales_table,
    )
    query = (
        "WITH "
        + eligibility.sql
        + """, candidate_rows AS (
                 SELECT item_id, loc, forecast_month,
                        forecast_qty, forecast_qty_lower, forecast_qty_upper,
                        model_id
                 FROM fact_production_forecast_staging
                 WHERE run_id = %s::uuid
                   AND generation_purpose = 'release_candidate'
                   AND candidate_model_id = %s
                   AND forecast_month >= %s
                   AND forecast_month < %s
             ), served AS (
                 SELECT item_id, loc,
                        COUNT(*) AS row_count,
                        COUNT(DISTINCT forecast_month) AS month_count
                 FROM candidate_rows
                 GROUP BY item_id, loc
             )
             SELECT
                 (SELECT COUNT(*) FROM eligible_item_locations)::integer,
                 COUNT(*) FILTER (WHERE served.month_count = %s)::integer,
                 COUNT(*) FILTER (
                     WHERE served.month_count > 0 AND served.month_count < %s
                 )::integer,
                 (SELECT COUNT(*) FROM candidate_rows)::integer,
                 (SELECT COUNT(*) FROM candidate_rows
                  WHERE forecast_qty_lower IS NOT NULL
                    AND forecast_qty_upper IS NOT NULL)::integer,
                 (SELECT COUNT(*) FROM candidate_rows
                  WHERE forecast_qty < 0
                     OR forecast_qty_lower < 0
                     OR forecast_qty_upper < 0
                     OR forecast_qty_lower > forecast_qty
                     OR forecast_qty_upper < forecast_qty)::integer
             FROM served
             JOIN eligible_item_locations USING (item_id, loc)"""
    )
    cur.execute(
        query,
        (
            *eligibility.params,
            str(source_run_id),
            model_id,
            planning_month,
            end_month,
            required_months,
            required_months,
        ),
    )
    row = cur.fetchone()
    return tuple(int(value or 0) for value in (row or (0, 0, 0, 0, 0, 0)))


def _candidate_quality_report(
    cur: Any,
    *,
    champion_experiment_id: int,
    planning_month: date,
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate exact experiment-stamped champion quality on one common cohort."""
    require_external_benchmark = bool(policy["require_external_benchmark"])
    required_model_count = 3 if require_external_benchmark else 2
    cur.execute(
        """WITH champion_keys AS (
                 SELECT f.item_id, f.customer_group, f.loc, f.startdate, f.lag,
                        f.tothist_dmd::numeric AS actual_qty
                 FROM fact_external_forecast_monthly f
                 JOIN dim_sku d
                   ON d.item_id = f.item_id
                  AND d.customer_group = f.customer_group
                  AND d.loc = f.loc
                 WHERE f.model_id = 'champion'
                   AND f.champion_experiment_id = %s
                   AND f.lag = COALESCE(d.execution_lag, 0)
                   AND f.lag BETWEEN 0 AND 4
                   AND f.basefcst_pref IS NOT NULL
                   AND f.tothist_dmd IS NOT NULL
                   AND f.startdate >= %s::date - (%s * INTERVAL '1 month')
                   AND f.startdate < %s::date
             ), required_prior_months AS (
                 SELECT DISTINCT startdate - INTERVAL '12 months' AS startdate
                 FROM champion_keys
             ), sales_by_dfu AS (
                 SELECT sales.item_id, sales.customer_group, sales.loc,
                        sales.startdate, SUM(sales.qty)::numeric AS qty
                 FROM fact_sales_monthly sales
                 JOIN required_prior_months required
                   ON required.startdate = sales.startdate
                 WHERE sales.type = 1
                 GROUP BY 1, 2, 3, 4
             ), scored AS (
                 SELECT f.item_id, f.customer_group, f.loc, f.startdate,
                        f.lag, f.model_id,
                        f.basefcst_pref::numeric AS forecast_qty,
                        f.tothist_dmd::numeric AS actual_qty
                 FROM fact_external_forecast_monthly f
                 JOIN dim_sku d
                   ON d.item_id = f.item_id
                  AND d.customer_group = f.customer_group
                  AND d.loc = f.loc
                 WHERE f.model_id IN ('champion', 'external')
                   AND (%s OR f.model_id <> 'external')
                   AND (
                       f.model_id <> 'champion'
                       OR f.champion_experiment_id = %s
                   )
                   AND f.lag = COALESCE(d.execution_lag, 0)
                   AND f.lag BETWEEN 0 AND 4
                   AND f.basefcst_pref IS NOT NULL
                   AND f.tothist_dmd IS NOT NULL
                   AND f.startdate >= %s::date - (%s * INTERVAL '1 month')
                   AND f.startdate < %s::date
                 UNION ALL
                 SELECT keys.item_id, keys.customer_group, keys.loc, keys.startdate,
                        keys.lag, 'seasonal_naive' AS model_id,
                        COALESCE(prior.qty, 0)::numeric AS forecast_qty,
                        keys.actual_qty
                 FROM champion_keys keys
                 LEFT JOIN sales_by_dfu prior
                   ON prior.item_id = keys.item_id
                  AND prior.customer_group = keys.customer_group
                  AND prior.loc = keys.loc
                  AND prior.startdate = keys.startdate - INTERVAL '12 months'
             ), key_quality AS (
                 SELECT item_id, customer_group, loc, startdate, lag,
                        COUNT(DISTINCT model_id) AS model_count,
                        MIN(actual_qty) AS min_actual,
                        MAX(actual_qty) AS max_actual
                 FROM scored
                 GROUP BY item_id, customer_group, loc, startdate, lag
                 HAVING COUNT(*) = %s AND COUNT(DISTINCT model_id) = %s
             ), common_keys AS (
                 SELECT item_id, customer_group, loc, startdate, lag
                 FROM key_quality
                 WHERE min_actual = max_actual
             ), champion_population AS (
                 SELECT COUNT(*)::bigint AS observations,
                        COUNT(DISTINCT (item_id, customer_group, loc))::bigint AS dfus
                 FROM scored
                 WHERE model_id = 'champion'
             ), model_metrics AS (
                 SELECT scored.model_id,
                        COUNT(*)::bigint AS observations,
                        COUNT(DISTINCT (
                            scored.item_id, scored.customer_group, scored.loc
                        ))::bigint AS dfus,
                        COUNT(DISTINCT scored.startdate)::integer AS closed_months,
                        SUM(scored.actual_qty)::numeric AS actual_volume,
                        100.0 * SUM(ABS(scored.forecast_qty - scored.actual_qty))
                            / NULLIF(ABS(SUM(scored.actual_qty)), 0) AS wape_pct,
                        100.0 * (
                            SUM(scored.forecast_qty)
                            / NULLIF(SUM(scored.actual_qty), 0) - 1.0
                        ) AS bias_pct
                 FROM scored
                 JOIN common_keys
                   USING (item_id, customer_group, loc, startdate, lag)
                 GROUP BY scored.model_id
             )
             SELECT
                 MAX(observations) FILTER (WHERE model_id = 'champion'),
                 MAX(dfus) FILTER (WHERE model_id = 'champion'),
                 MAX(wape_pct) FILTER (WHERE model_id = 'champion'),
                 MAX(bias_pct) FILTER (WHERE model_id = 'champion'),
                 MAX(wape_pct) FILTER (WHERE model_id = 'seasonal_naive'),
                 MAX(wape_pct) FILTER (WHERE model_id = 'external'),
                 (SELECT COUNT(*) FROM key_quality
                  WHERE min_actual IS DISTINCT FROM max_actual),
                 (SELECT observations FROM champion_population),
                 (SELECT dfus FROM champion_population),
                 MAX(closed_months) FILTER (WHERE model_id = 'champion'),
                 MAX(actual_volume) FILTER (WHERE model_id = 'champion')
             FROM model_metrics""",
        (
            champion_experiment_id,
            planning_month,
            int(policy["quality_lookback_months"]),
            planning_month,
            require_external_benchmark,
            champion_experiment_id,
            planning_month,
            int(policy["quality_lookback_months"]),
            planning_month,
            required_model_count,
            required_model_count,
        ),
    )
    row = cur.fetchone()
    if row is None:
        raise PromotionConflictError(
            "candidate_quality_failed",
            "Candidate quality evidence is unavailable.",
        )
    metrics = ReleaseQualityMetrics(
        dfu_months=int(row[0] or 0),
        dfus=int(row[1] or 0),
        champion_observations=int(row[7] or 0),
        champion_dfus=int(row[8] or 0),
        closed_months=int(row[9] or 0),
        actual_volume=float(row[10] or 0),
        champion_wape_pct=float(row[2]) if row[2] is not None else None,
        champion_bias_pct=float(row[3]) if row[3] is not None else None,
        naive_wape_pct=float(row[4]) if row[4] is not None else None,
        external_wape_pct=float(row[5]) if row[5] is not None else None,
    )
    thresholds = ReleaseReadinessThresholds(
        require_external_benchmark=require_external_benchmark,
        min_relative_wape_lift_vs_naive_pct=float(policy["min_relative_wape_lift_vs_naive_pct"]),
        min_accuracy_delta_vs_external_pct_points=float(
            policy["min_accuracy_delta_vs_external_pct_points"]
        ),
        max_abs_bias_pct=float(policy["max_abs_bias_pct"]),
        min_current_plan_coverage_frac=float(policy["min_coverage_frac"]),
        min_common_cohort_coverage_frac=float(policy["min_common_cohort_coverage_frac"]),
        min_common_cohort_closed_months=int(policy["min_common_cohort_closed_months"]),
        min_common_cohort_dfus=int(policy["min_common_cohort_dfus"]),
        min_common_cohort_actual_volume=float(policy["min_common_cohort_actual_volume"]),
    )
    quality_checks = evaluate_quality_checks(metrics, thresholds)
    actuals_aligned = int(row[6] or 0) == 0 and metrics.dfu_months > 0
    quality_checks.append(
        {
            "id": "actual_alignment",
            "status": "pass" if actuals_aligned else "block",
            "value": int(row[6] or 0),
            "threshold": 0,
            "message": "Actual demand must match across the common cohort.",
        }
    )
    if any(check["status"] == "block" for check in quality_checks):
        raise PromotionConflictError(
            "candidate_quality_failed",
            "The experiment-stamped champion does not meet release quality policy.",
        )
    return quality_checks


def _validate_candidate_evidence(
    cur: Any,
    *,
    conn: Any,
    manifest: ForecastGenerationManifest,
    model_id: str,
    planning_month: date,
    policy: dict[str, Any],
    release_stats: ForecastPayloadStats,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    required_months = int(policy["required_months"])
    evidence = _current_release_evidence(cur)
    (
        latest_sales_batch_id,
        results_promoted_count,
        results_experiment_id,
        results_cluster_id,
        results_artifact_checksum,
        results_forecast_checksum,
        results_forecast_row_count,
        promoted_cluster_count,
        promoted_cluster_id,
        assignment_count,
        stale_tuning_count,
    ) = evidence
    checks: dict[str, Any] = {
        "source_run_id": str(manifest.run_id),
        "planning_month": planning_month.isoformat(),
        "required_months": required_months,
        "latest_sales_batch_id": latest_sales_batch_id,
        "candidate_sales_batch_id": manifest.source_sales_batch_id,
    }
    if latest_sales_batch_id is None or manifest.source_sales_batch_id != int(
        latest_sales_batch_id
    ):
        raise PromotionConflictError(
            "stale_candidate_evidence",
            "A newer sales load exists than the selected forecast generation run.",
        )

    pipeline_config = load_forecast_pipeline_config()
    is_customer_blend = (
        CUSTOMER_BLEND_LINEAGE_METADATA_KEY in manifest.metadata
        or _is_customer_blend_payload(cur, manifest.run_id)
    )
    customer_blend_checks = _validate_customer_bottom_up_blend(
        cur,
        manifest=manifest,
        pipeline_config=pipeline_config,
        require_lineage=is_customer_blend,
    )
    if customer_blend_checks is not None:
        checks["customer_bottom_up_blend"] = customer_blend_checks
    _validate_direct_model_lineage(
        cur,
        manifest=manifest,
        model_id=model_id,
        pipeline_config=pipeline_config,
    )
    _validate_active_model_artifacts(
        conn=conn,
        cur=cur,
        manifest=manifest,
        pipeline_config=pipeline_config,
        project_root=project_root,
    )
    _validate_tree_model_lineage(conn, manifest=manifest)

    if model_id == CHAMPION_MODEL_ID:
        _validate_governed_champion_source(cur, manifest=manifest)
        lineage_matches = (
            results_promoted_count == 1
            and manifest.champion_experiment_id == results_experiment_id
            and manifest.cluster_experiment_id == results_cluster_id
            and manifest.routing_artifact_checksum == results_artifact_checksum
            and manifest.champion_results_checksum == results_forecast_checksum
            and promoted_cluster_count == 1
            and manifest.cluster_experiment_id == promoted_cluster_id
            and int(assignment_count or 0) > 0
            and int(stale_tuning_count or 0) == 0
        )
        if not lineage_matches:
            raise PromotionConflictError(
                "candidate_lineage_mismatch",
                "Champion results, cluster lineage, assignments, or tuning are not release-current.",
            )
        routing_path = (
            project_root
            / "data"
            / "champion"
            / f"experiment_{manifest.champion_experiment_id}_winners.csv"
        )
        if (
            not routing_path.exists()
            or sha256_file(routing_path) != manifest.routing_artifact_checksum
        ):
            raise PromotionConflictError(
                "candidate_lineage_mismatch",
                "The champion routing artifact no longer matches the generated release candidate.",
            )
        current_results = compute_champion_results_stats(
            cur,
            int(manifest.champion_experiment_id),
        )
        if (
            current_results.checksum != manifest.champion_results_checksum
            or current_results.row_count != int(results_forecast_row_count or 0)
        ):
            raise PromotionConflictError(
                "candidate_lineage_mismatch",
                "The historical champion evidence changed after results promotion.",
            )
        checks["quality_checks"] = _candidate_quality_report(
            cur,
            champion_experiment_id=int(manifest.champion_experiment_id),
            planning_month=planning_month,
            policy=policy,
        )
    elif release_stats.source_model_count != 1:
        raise PromotionConflictError(
            "candidate_lineage_mismatch",
            "A single-model release candidate contains multiple source models.",
        )

    coverage = _candidate_coverage(
        cur,
        source_run_id=manifest.run_id,
        model_id=model_id,
        planning_month=planning_month,
        required_months=required_months,
        min_history_months=int(policy.get("min_history_months", 3)),
        active_window_months=int(policy.get("active_window_months", 12)),
    )
    eligible_dfus, complete_dfus, partial_dfus, window_rows, ci_rows, invalid_rows = coverage
    coverage_frac = complete_dfus / eligible_dfus if eligible_dfus else 0.0
    ci_coverage_frac = ci_rows / window_rows if window_rows else 0.0
    checks.update(
        {
            "eligible_dfus": eligible_dfus,
            "complete_dfus": complete_dfus,
            "coverage_frac": coverage_frac,
            "partial_dfus": partial_dfus,
            "window_rows": window_rows,
            "ci_coverage_frac": ci_coverage_frac,
            "invalid_rows": invalid_rows,
            "candidate_checksum": release_stats.checksum,
        }
    )
    if eligible_dfus <= 0 or window_rows <= 0:
        raise PromotionConflictError(
            "candidate_gate_failed",
            "The release candidate has no eligible six-month planning population.",
        )
    if partial_dfus > 0:
        raise PromotionConflictError(
            "candidate_gate_failed",
            "The release candidate contains DFUs with route gaps inside the required window.",
        )
    if coverage_frac < float(policy["min_coverage_frac"]):
        raise PromotionConflictError(
            "candidate_gate_failed",
            "The release candidate does not meet the required DFU coverage.",
        )
    if invalid_rows > 0 or ci_coverage_frac < float(policy["min_ci_coverage_frac"]):
        raise PromotionConflictError(
            "candidate_gate_failed",
            "The release candidate has invalid quantities or insufficient confidence intervals.",
        )
    return checks


def _lock_active_release_for_replacement(cur: Any) -> tuple[int | None, dict[str, Any] | None]:
    """Lock and describe the active release without coupling replacement to Period Roll."""
    cur.execute(
        """SELECT id, plan_version, model_id, source_run_id, production_run_id
           FROM model_promotion_log
           WHERE is_active = TRUE
           ORDER BY promoted_at DESC, id DESC
           LIMIT 1
           FOR UPDATE"""
    )
    row = cur.fetchone()
    if row is None:
        return None, None
    return int(row[0]), {
        "promotion_id": int(row[0]),
        "plan_version": str(row[1]),
        "model_id": str(row[2]),
        "source_run_id": str(row[3]) if row[3] is not None else None,
        "production_run_id": str(row[4]) if row[4] is not None else None,
        "status": "replaced",
    }


def _promotion_requires_customer_demand_lock(conn: Any, source_run_id: UUID) -> bool:
    """Identify immutable customer-blend lineage before opening the release snapshot."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT EXISTS (
                   SELECT 1
                   FROM forecast_generation_run generation
                   WHERE generation.run_id = %s::uuid
                     AND (
                         generation.metadata ? %s
                         OR EXISTS (
                             SELECT 1
                             FROM fact_production_forecast_staging staging
                             WHERE staging.run_id = generation.run_id
                               AND staging.model_id = %s
                         )
                     )
               )""",
            (
                str(source_run_id),
                CUSTOMER_BLEND_LINEAGE_METADATA_KEY,
                CUSTOMER_BOTTOM_UP_BLEND_MODEL_ID,
            ),
        )
        row = cur.fetchone()
    # End the preflight transaction so the shared lock is committed before the
    # subsequent SERIALIZABLE release snapshot begins.
    conn.commit()
    return row is not None and row[0] is True


def promote_forecast_run(
    conn: Any,
    *,
    model_id: str,
    source_run_id: UUID,
    planning_month: date,
    promoted_by: str,
    notes: str | None,
    policy: dict[str, Any],
) -> PromotionResult:
    """Atomically validate and publish one source run as the sole active release."""
    required_months = int(policy["required_months"])
    production_run_id = UUID(str(uuid.uuid4()))
    plan_version = planning_month.strftime("%Y-%m")
    requires_customer_demand_lock = _promotion_requires_customer_demand_lock(
        conn,
        source_run_id,
    )
    lineage_lock = (
        customer_demand_snapshot_lock(conn)
        if requires_customer_demand_lock
        else nullcontext()
    )
    with lineage_lock, conn.transaction():
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            cur.execute("SELECT pg_advisory_xact_lock(hashtext('forecast_release_promotion'))")
            manifest = _load_manifest(cur, source_run_id)
            validate_generation_manifest(
                manifest,
                model_id=model_id,
                planning_month=planning_month,
                required_months=required_months,
            )
            full_stats = compute_staging_payload_stats(cur, source_run_id)
            _validate_manifest_payload(manifest, full_stats)
            release_end_month = _add_months(planning_month, manifest.horizon_months)
            release_stats = compute_staging_payload_stats(
                cur,
                source_run_id,
                start_month=planning_month,
                end_month=release_end_month,
            )
            if release_stats.row_count <= 0:
                raise PromotionConflictError(
                    "candidate_gate_failed",
                    "The selected run has no rows in the release window.",
                )
            gate_report = _validate_candidate_evidence(
                cur,
                conn=conn,
                manifest=manifest,
                model_id=model_id,
                planning_month=planning_month,
                policy=policy,
                release_stats=release_stats,
            )
            outgoing_id, outgoing_report = _lock_active_release_for_replacement(cur)
            if outgoing_report is not None:
                gate_report["outgoing_release"] = outgoing_report

            if outgoing_id is not None:
                cur.execute(
                    """UPDATE model_promotion_log
                       SET is_active = FALSE, demoted_at = NOW()
                       WHERE id = %s AND is_active = TRUE""",
                    (outgoing_id,),
                )
                if cur.rowcount != 1:
                    raise PromotionConflictError(
                        "concurrent_release_conflict",
                        "The outgoing release changed during promotion.",
                    )

            promotion_type = "champion" if model_id == CHAMPION_MODEL_ID else "single"
            cur.execute(
                """INSERT INTO model_promotion_log
                       (model_id, promotion_type, champion_experiment_id,
                        plan_version, is_active, dfu_count, total_rows,
                        promoted_by, notes, source_run_id, production_run_id,
                        gate_report, candidate_checksum, production_checksum,
                        replaces_promotion_id)
                   VALUES (%s, %s, %s, %s, TRUE, %s, %s, %s, %s,
                           %s::uuid, %s::uuid, %s, %s, %s, %s)
                   RETURNING id""",
                (
                    model_id,
                    promotion_type,
                    manifest.champion_experiment_id,
                    plan_version,
                    release_stats.dfu_count,
                    release_stats.row_count,
                    promoted_by,
                    notes,
                    str(source_run_id),
                    str(production_run_id),
                    Jsonb(gate_report),
                    release_stats.checksum,
                    release_stats.checksum,
                    outgoing_id,
                ),
            )
            promotion_row = cur.fetchone()
            if promotion_row is None:
                raise RuntimeError("promotion audit insert returned no identifier")
            promotion_id = int(promotion_row[0])

            cur.execute("DELETE FROM fact_production_forecast")
            cur.execute(
                """INSERT INTO fact_production_forecast
                       (plan_version, item_id, loc, forecast_month, forecast_qty,
                        forecast_qty_lower, forecast_qty_upper, model_id,
                        source_model_id, cluster_id, horizon_months, is_recursive,
                        lag_source, generated_at, run_id, source_run_id,
                        promotion_log_id, lineage_status)
                   SELECT %s, s.item_id, s.loc, s.forecast_month, s.forecast_qty,
                          s.forecast_qty_lower, s.forecast_qty_upper, %s,
                          s.model_id, s.cluster_id, s.horizon_months,
                          s.is_recursive, s.lag_source, s.generated_at,
                          %s::uuid, %s::uuid, %s, 'verified'
                   FROM fact_production_forecast_staging s
                   WHERE s.run_id = %s::uuid
                     AND s.generation_purpose = 'release_candidate'
                     AND s.candidate_model_id = %s
                     AND s.forecast_month >= %s
                     AND s.forecast_month < %s""",
                (
                    plan_version,
                    model_id,
                    str(production_run_id),
                    str(source_run_id),
                    promotion_id,
                    str(source_run_id),
                    model_id,
                    planning_month,
                    release_end_month,
                ),
            )
            rows_promoted = int(cur.rowcount)
            production_stats = compute_production_payload_stats(
                cur,
                production_run_id,
                start_month=planning_month,
                end_month=release_end_month,
            )
            if (
                rows_promoted != release_stats.row_count
                or production_stats.row_count != release_stats.row_count
                or production_stats.dfu_count != release_stats.dfu_count
                or production_stats.checksum != release_stats.checksum
            ):
                raise PromotionConflictError(
                    "production_checksum_mismatch",
                    "Published forecast values do not match the selected release candidate.",
                )
            cur.execute(
                """UPDATE forecast_generation_run
                   SET run_status = 'promoted', promotion_eligible = FALSE
                   WHERE run_id = %s::uuid
                     AND run_status = 'ready'
                     AND promotion_eligible = TRUE""",
                (str(source_run_id),),
            )
            if cur.rowcount != 1:
                raise PromotionConflictError(
                    "concurrent_release_conflict",
                    "The source generation run changed during promotion.",
                )

    return PromotionResult(
        model_id=model_id,
        promotion_type=promotion_type,
        plan_version=plan_version,
        source_run_id=source_run_id,
        production_run_id=production_run_id,
        candidate_checksum=release_stats.checksum,
        outgoing_archive_checksum=None,
        rows_promoted=rows_promoted,
        dfu_count=release_stats.dfu_count,
    )
