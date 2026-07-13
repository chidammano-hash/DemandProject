"""Governed champion refresh used by the forecasting lifecycle pipelines.

The refresh creates a new immutable champion experiment, evaluates it without
touching the incumbent, and then swaps historical champion facts plus both
promotion flags in one database transaction.  A failed evaluation or load
therefore cannot erase the currently promoted champion evidence.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from threading import Event
from typing import Any

from common.core.mv_refresh import refresh_for_tables
from common.core.paths import DATA_DIR, SCRIPTS_DIR
from common.core.utils import get_competing_model_ids, load_forecast_pipeline_config
from common.services.champion_lineage import (
    CANONICAL_CHAMPION_MODELS,
    GOVERNED_PROMOTION_MODE,
)
from common.services.cluster_lineage import load_promoted_cluster_population
from common.services.forecast_lineage import (
    compute_champion_results_stats,
    sha256_file,
)
from common.services.job_state import _UV, _get_conn, _run_subprocess
from common.services.sales_lineage import load_completed_sales_lineage
from scripts.ml.run_champion_selection import (
    _load_cached_winners,
    assert_competing_models_covered,
    compute_ceiling_winners,
    insert_ceiling_forecasts,
    insert_champion_forecasts,
    insert_ensemble_forecasts,
    insert_fallback_champions,
)

logger = logging.getLogger(__name__)

_PROMOTION_LOCK = "governed_champion_refresh"


@dataclass(frozen=True)
class ChampionRefreshSpec:
    """Exact production champion contract captured for one refresh."""

    strategy: str
    strategy_params: dict[str, Any]
    meta_learner_params: dict[str, Any]
    models: tuple[str, ...]
    metric: str
    lag_mode: str
    min_dfu_rows: int
    champion_model_id: str
    fallback_model_id: str
    cluster_experiment_id: int
    source_sales_batch_id: int | None = None
    data_checksum: str | None = None
    cluster_assignment_count: int | None = None
    cluster_assignment_checksum: str | None = None
    backtest_run_ids: tuple[tuple[str, int], ...] = ()
    source_experiment_id: int | None = None


@dataclass(frozen=True)
class ChampionAssignmentCandidate:
    """Completed analysis experiment selected as the assignment strategy source."""

    experiment_id: int
    label: str
    strategy: str
    strategy_params: dict[str, Any]
    meta_learner_params: dict[str, Any]
    models: tuple[str, ...]
    metric: str
    lag_mode: str
    min_dfu_rows: int


def _json_object(value: object, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"champion.{field_name} must be an object")
    return dict(value)


def build_refresh_spec(
    pipeline_config: dict[str, Any],
    *,
    competing_models: list[str],
    cluster_experiment_id: int,
) -> ChampionRefreshSpec:
    """Validate and freeze the current production champion configuration."""
    champion = pipeline_config.get("champion")
    if not isinstance(champion, dict):
        raise ValueError("forecast pipeline config is missing the champion section")

    configured_models = champion.get("models")
    canonical_set = set(CANONICAL_CHAMPION_MODELS)
    if (
        not isinstance(configured_models, list)
        or len(configured_models) != len(CANONICAL_CHAMPION_MODELS)
        or set(configured_models) != canonical_set
        or len(competing_models) != len(CANONICAL_CHAMPION_MODELS)
        or set(competing_models) != canonical_set
    ):
        raise ValueError(
            "Champion refresh requires the exact canonical five-model roster: "
            f"{list(CANONICAL_CHAMPION_MODELS)}"
        )

    strategy = champion.get("strategy")
    metric = champion.get("metric")
    lag_mode = champion.get("lag")
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("champion.strategy must be configured")
    if not isinstance(metric, str) or not metric:
        raise ValueError("champion.metric must be configured")
    if not isinstance(lag_mode, (str, int)):
        raise ValueError("champion.lag must be configured")

    min_dfu_rows = champion.get("min_dfu_rows", champion.get("min_sku_rows"))
    if not isinstance(min_dfu_rows, int) or min_dfu_rows < 1:
        raise ValueError("champion.min_dfu_rows must be a positive integer")

    champion_model_id = champion.get("champion_model_id")
    fallback_model_id = champion.get("fallback_model_id")
    if champion_model_id != "champion":
        raise ValueError("champion.champion_model_id must remain 'champion'")
    if fallback_model_id not in canonical_set:
        raise ValueError("champion.fallback_model_id must be one of the canonical five models")

    return ChampionRefreshSpec(
        strategy=strategy,
        strategy_params=_json_object(
            champion.get("strategy_params"),
            field_name="strategy_params",
        ),
        meta_learner_params=_json_object(
            champion.get("meta_learner"),
            field_name="meta_learner",
        ),
        models=CANONICAL_CHAMPION_MODELS,
        metric=metric,
        lag_mode=str(lag_mode),
        min_dfu_rows=min_dfu_rows,
        champion_model_id=champion_model_id,
        fallback_model_id=str(fallback_model_id),
        cluster_experiment_id=int(cluster_experiment_id),
    )


def load_refresh_spec() -> ChampionRefreshSpec:
    """Load the governed spec and prove all five candidate models are present."""
    competing_models = get_competing_model_ids()
    with _get_conn() as conn, conn.transaction():
        sales = load_completed_sales_lineage(conn)
        clusters = load_promoted_cluster_population(conn)
        with conn.cursor() as cur:
            cur.execute(
                """SELECT DISTINCT ON (model_id)
                          model_id, id, metadata
                   FROM backtest_run
                   WHERE model_id = ANY(%s)
                     AND status = 'completed'
                     AND is_loaded_to_db = TRUE
                   ORDER BY model_id, completed_at DESC NULLS LAST, id DESC""",
                (competing_models,),
            )
            backtest_rows = cur.fetchall()
            backtest_run_ids = validate_backtest_lineage_rows(
                backtest_rows,
                models=CANONICAL_CHAMPION_MODELS,
                source_sales_batch_id=sales.batch_id,
                data_checksum=sales.source_hash,
                cluster_experiment_id=clusters.experiment_id,
                cluster_assignment_count=clusters.assignment_count,
                cluster_assignment_checksum=clusters.assignment_checksum,
            )
            assert_competing_models_covered(cur, competing_models)

    base_spec = build_refresh_spec(
        load_forecast_pipeline_config(),
        competing_models=competing_models,
        cluster_experiment_id=clusters.experiment_id,
    )
    return replace(
        base_spec,
        source_sales_batch_id=sales.batch_id,
        data_checksum=sales.source_hash,
        cluster_assignment_count=clusters.assignment_count,
        cluster_assignment_checksum=clusters.assignment_checksum,
        backtest_run_ids=backtest_run_ids,
    )


def load_champion_assignment_candidate(
    experiment_id: int,
) -> ChampionAssignmentCandidate:
    """Load a completed experiment that can safely source an assignment strategy."""
    if not _positive_integer(experiment_id):
        raise ValueError("source_experiment_id must be a positive integer")
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT label, status, strategy, strategy_params,
                      meta_learner_params, models, metric, lag_mode, min_sku_rows
               FROM champion_experiment
               WHERE experiment_id = %s""",
            (experiment_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise LookupError(f"Champion experiment {experiment_id} not found")

    (
        label,
        status,
        strategy,
        raw_strategy_params,
        raw_meta_params,
        raw_models,
        metric,
        lag_mode,
        min_sku_rows,
    ) = row
    if status != "completed":
        raise ValueError(
            f"Champion experiment {experiment_id} is {status!r}; "
            "only completed experiments can be assigned"
        )
    models = _decode_json(raw_models, field_name="models")
    candidate = ChampionAssignmentCandidate(
        experiment_id=experiment_id,
        label=str(label),
        strategy=str(strategy),
        strategy_params=_json_object(
            _decode_json(raw_strategy_params, field_name="strategy_params"),
            field_name="strategy_params",
        ),
        meta_learner_params=_json_object(
            _decode_json(raw_meta_params, field_name="meta_learner_params"),
            field_name="meta_learner_params",
        ),
        models=tuple(models) if isinstance(models, list) else (),
        metric=str(metric),
        lag_mode=str(lag_mode),
        min_dfu_rows=int(min_sku_rows),
    )
    _validate_assignment_candidate(candidate)
    return candidate


def _validate_assignment_candidate(
    candidate: ChampionAssignmentCandidate,
) -> None:
    if len(candidate.models) != len(CANONICAL_CHAMPION_MODELS) or set(candidate.models) != set(
        CANONICAL_CHAMPION_MODELS
    ):
        raise ValueError(
            "Champion assignment requires the exact canonical five-model roster: "
            f"{list(CANONICAL_CHAMPION_MODELS)}"
        )
    if not candidate.strategy or not candidate.metric or not candidate.lag_mode:
        raise ValueError("Selected champion experiment has an incomplete strategy contract")
    if candidate.min_dfu_rows < 1:
        raise ValueError("Selected champion experiment min_sku_rows must be positive")


def build_selected_refresh_spec(
    current_spec: ChampionRefreshSpec,
    candidate: ChampionAssignmentCandidate,
) -> ChampionRefreshSpec:
    """Bind a selected strategy to the current sales/cluster/backtest lineage."""
    _validate_assignment_candidate(candidate)
    return replace(
        current_spec,
        source_experiment_id=candidate.experiment_id,
        strategy=candidate.strategy,
        strategy_params=candidate.strategy_params,
        meta_learner_params=candidate.meta_learner_params,
        models=CANONICAL_CHAMPION_MODELS,
        metric=candidate.metric,
        lag_mode=candidate.lag_mode,
        min_dfu_rows=candidate.min_dfu_rows,
    )


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_backtest_lineage_rows(
    rows: list[tuple[Any, ...]],
    *,
    models: tuple[str, ...],
    source_sales_batch_id: int,
    data_checksum: str,
    cluster_experiment_id: int,
    cluster_assignment_count: int,
    cluster_assignment_checksum: str,
) -> tuple[tuple[str, int], ...]:
    """Require one latest loaded run per model on one exact current lineage."""
    expected_models = set(models)
    expected_fields = {
        "source_sales_batch_id",
        "data_checksum",
        "cluster_experiment_id",
        "cluster_assignment_count",
        "cluster_assignment_checksum",
    }
    expected_lineage = {
        "source_sales_batch_id": source_sales_batch_id,
        "data_checksum": data_checksum,
        "cluster_experiment_id": cluster_experiment_id,
        "cluster_assignment_count": cluster_assignment_count,
        "cluster_assignment_checksum": cluster_assignment_checksum,
    }
    observed: dict[str, int] = {}
    stale_models: list[str] = []
    for raw_model_id, raw_run_id, raw_metadata in rows:
        model_id = str(raw_model_id)
        if model_id not in expected_models or model_id in observed:
            stale_models.append(model_id)
            continue
        metadata = _decode_json(raw_metadata, field_name="backtest metadata")
        lineage = metadata.get("governed_lineage") if isinstance(metadata, dict) else None
        if not isinstance(lineage, dict) or set(lineage) != expected_fields:
            stale_models.append(model_id)
            continue
        if (
            not _positive_integer(lineage.get("source_sales_batch_id"))
            or not _sha256(lineage.get("data_checksum"))
            or not _positive_integer(lineage.get("cluster_experiment_id"))
            or not _positive_integer(lineage.get("cluster_assignment_count"))
            or not _sha256(lineage.get("cluster_assignment_checksum"))
            or lineage != expected_lineage
            or not _positive_integer(raw_run_id)
        ):
            stale_models.append(model_id)
            continue
        observed[model_id] = int(raw_run_id)

    missing_models = sorted(expected_models - set(observed))
    if stale_models or missing_models or set(observed) != expected_models:
        details: list[str] = []
        if missing_models:
            details.append(f"missing/stale models: {', '.join(missing_models)}")
        unexpected = sorted(set(stale_models) - set(missing_models))
        if unexpected:
            details.append(f"invalid rows: {', '.join(unexpected)}")
        raise RuntimeError(
            "Champion refresh requires the latest completed and loaded backtest for "
            "all five models to carry the current governed lineage ("
            + "; ".join(details)
            + "). Run model-refresh before retrying champion-refresh; the incumbent "
            "champion was not touched."
        )
    return tuple((model_id, observed[model_id]) for model_id in models)


def create_governed_experiment(
    spec: ChampionRefreshSpec,
    *,
    job_id: str | None,
    source_candidate: ChampionAssignmentCandidate | None = None,
) -> int:
    """Create a queued experiment without changing either promoted incumbent."""
    if source_candidate is None:
        label = "Governed champion refresh"
        template_id = "governed-champion-refresh"
        source_note = "using the configured production strategy"
    else:
        label = f"Assigned from #{source_candidate.experiment_id}: {source_candidate.label}"
        template_id = "selected-champion-assignment"
        source_note = (
            f"from selected analysis experiment #{source_candidate.experiment_id} "
            f"({source_candidate.label})"
        )
    notes = (
        f"Created by the governed champion assignment lifecycle {source_note}. "
        "The selected strategy is re-evaluated on current five-model backtests; "
        "the incumbent remains active until historical results are complete."
    )
    with _get_conn() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute(
            """INSERT INTO champion_experiment
                   (label, notes, template_id, job_id, strategy, strategy_params,
                    meta_learner_params, models, metric, lag_mode, min_sku_rows,
                    cluster_experiment_id)
               SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                      cluster.experiment_id
               FROM cluster_experiment cluster
               WHERE cluster.experiment_id = %s
                 AND cluster.is_promoted = TRUE
               RETURNING experiment_id""",
            (
                label,
                notes,
                template_id,
                job_id,
                spec.strategy,
                json.dumps(spec.strategy_params),
                json.dumps(spec.meta_learner_params),
                json.dumps(list(spec.models)),
                spec.metric,
                spec.lag_mode,
                spec.min_dfu_rows,
                spec.cluster_experiment_id,
            ),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(
                "The promoted clustering experiment changed before champion refresh "
                "could be created; restart champion-refresh"
            )
    return int(row[0])


def _spec_payload(spec: ChampionRefreshSpec) -> dict[str, Any]:
    payload = asdict(spec)
    payload["models"] = list(spec.models)
    payload["backtest_run_ids"] = [list(pair) for pair in spec.backtest_run_ids]
    return payload


def refresh_spec_from_payload(payload: object) -> ChampionRefreshSpec:
    """Parse the exact refresh context persisted with a durable job."""
    if not isinstance(payload, dict):
        raise RuntimeError("Persisted governed champion spec is not an object")
    expected_keys = set(ChampionRefreshSpec.__dataclass_fields__)
    if set(payload) != expected_keys:
        raise RuntimeError("Persisted governed champion spec has an invalid schema")
    models = payload.get("models")
    run_ids = payload.get("backtest_run_ids")
    if not isinstance(models, list) or tuple(models) != CANONICAL_CHAMPION_MODELS:
        raise RuntimeError("Persisted governed champion spec has an invalid model roster")
    if not isinstance(run_ids, list) or not all(
        isinstance(pair, list)
        and len(pair) == 2
        and isinstance(pair[0], str)
        and _positive_integer(pair[1])
        for pair in run_ids
    ):
        raise RuntimeError("Persisted governed champion spec has invalid backtest run IDs")
    try:
        return ChampionRefreshSpec(
            strategy=str(payload["strategy"]),
            strategy_params=_json_object(
                payload["strategy_params"],
                field_name="strategy_params",
            ),
            meta_learner_params=_json_object(
                payload["meta_learner_params"],
                field_name="meta_learner_params",
            ),
            models=tuple(models),
            metric=str(payload["metric"]),
            lag_mode=str(payload["lag_mode"]),
            min_dfu_rows=int(payload["min_dfu_rows"]),
            champion_model_id=str(payload["champion_model_id"]),
            fallback_model_id=str(payload["fallback_model_id"]),
            cluster_experiment_id=int(payload["cluster_experiment_id"]),
            source_sales_batch_id=int(payload["source_sales_batch_id"]),
            data_checksum=str(payload["data_checksum"]),
            cluster_assignment_count=int(payload["cluster_assignment_count"]),
            cluster_assignment_checksum=str(payload["cluster_assignment_checksum"]),
            backtest_run_ids=tuple((pair[0], int(pair[1])) for pair in run_ids),
            source_experiment_id=(
                int(payload["source_experiment_id"])
                if payload["source_experiment_id"] is not None
                else None
            ),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Persisted governed champion spec contains invalid values") from exc


def persist_job_experiment_id(
    job_id: str | None,
    experiment_id: int,
    spec: ChampionRefreshSpec,
) -> None:
    """Bind exact experiment + lineage context before managed subprocess work."""
    if job_id is None:
        return
    with _get_conn() as conn:
        result = conn.execute(
            """UPDATE job_history
               SET params = COALESCE(params, '{}'::jsonb)
                            || jsonb_build_object(
                                'experiment_id', %s::bigint,
                                'governed_spec', %s::jsonb
                            )
               WHERE job_id = %s""",
            (experiment_id, json.dumps(_spec_payload(spec)), job_id),
        )
    if int(result.rowcount or 0) != 1:
        raise RuntimeError(f"Could not persist champion experiment {experiment_id} on job {job_id}")


def champion_winners_path(experiment_id: int) -> Path:
    """Return the sole routing artifact path for an experiment."""
    return DATA_DIR / "champion" / f"experiment_{experiment_id}_winners.csv"


def run_champion_experiment_job(
    experiment_id: int,
    *,
    progress_cb: Callable | None,
    cancel_event: Event | None,
    job_id: str | None,
) -> str:
    """Run the experiment through the managed durable subprocess boundary."""
    command = [
        _UV,
        "run",
        "python",
        str(SCRIPTS_DIR / "ml" / "run_champion_experiment.py"),
        "--experiment-id",
        str(experiment_id),
    ]
    return _run_subprocess(
        command,
        progress_cb,
        f"Evaluating governed champion experiment #{experiment_id}",
        cancel_event=cancel_event,
        job_id=job_id,
    )


def _decode_json(value: object, *, field_name: str) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Champion experiment has invalid {field_name} JSON") from exc
    return value


def _validated_experiment_spec(
    row: tuple[Any, ...],
    *,
    expected: ChampionRefreshSpec,
) -> tuple[bool, str | None, str | None, int | None]:
    (
        status,
        raw_models,
        strategy,
        raw_strategy_params,
        raw_meta_params,
        metric,
        lag_mode,
        min_sku_rows,
        cluster_experiment_id,
        is_promoted,
        is_results_promoted,
        results_artifact_checksum,
        results_forecast_checksum,
        results_forecast_row_count,
    ) = row
    models = _decode_json(raw_models, field_name="models")
    strategy_params = _decode_json(raw_strategy_params, field_name="strategy_params") or {}
    meta_params = _decode_json(raw_meta_params, field_name="meta_learner_params") or {}
    actual = (
        tuple(models) if isinstance(models, list) else (),
        strategy,
        strategy_params,
        meta_params,
        metric,
        str(lag_mode),
        int(min_sku_rows),
        int(cluster_experiment_id) if cluster_experiment_id is not None else None,
    )
    wanted = (
        expected.models,
        expected.strategy,
        expected.strategy_params,
        expected.meta_learner_params,
        expected.metric,
        expected.lag_mode,
        expected.min_dfu_rows,
        expected.cluster_experiment_id,
    )
    if status != "completed":
        raise RuntimeError(
            f"Champion experiment is {status!r}; only completed experiments can be promoted"
        )
    if actual != wanted:
        raise RuntimeError(
            "Champion experiment no longer matches the current governed five-model contract; "
            "restart champion-refresh"
        )
    if bool(is_promoted) != bool(is_results_promoted):
        raise RuntimeError(
            "Champion experiment has a partial promotion state; resolve it before retrying"
        )
    return (
        bool(is_promoted),
        str(results_artifact_checksum) if results_artifact_checksum else None,
        str(results_forecast_checksum) if results_forecast_checksum else None,
        int(results_forecast_row_count) if results_forecast_row_count is not None else None,
    )


def _current_refresh_context_matches(
    current: ChampionRefreshSpec,
    expected: ChampionRefreshSpec,
) -> bool:
    """Keep selected strategy frozen while requiring current governed lineage."""
    if expected.source_experiment_id is None:
        return current == expected
    return (
        replace(
            expected,
            source_experiment_id=None,
            strategy=current.strategy,
            strategy_params=current.strategy_params,
            meta_learner_params=current.meta_learner_params,
            metric=current.metric,
            lag_mode=current.lag_mode,
            min_dfu_rows=current.min_dfu_rows,
        )
        == current
    )


def _promotion_snapshot(spec: ChampionRefreshSpec) -> dict[str, Any]:
    snapshot = _spec_payload(spec)
    snapshot["_promotion_mode"] = GOVERNED_PROMOTION_MODE
    return snapshot


def finalize_governed_champion_refresh(
    experiment_id: int,
    *,
    job_id: str | None,
    winners_csv: Path | None = None,
    expected_spec: ChampionRefreshSpec | None = None,
    refresh_views: bool = True,
) -> dict[str, Any]:
    """Atomically load, audit, and promote one completed experiment.

    This function is idempotent for restart recovery.  The first execution
    rewrites champion facts and both promotion flags under one transaction;
    subsequent execution verifies the stored checksums instead of rewriting.
    """
    current_spec = load_refresh_spec()
    if expected_spec is not None and not _current_refresh_context_matches(
        current_spec,
        expected_spec,
    ):
        raise RuntimeError(
            "Sales, clustering, backtest lineage, or governed model roster changed "
            "while the experiment was running; the incumbent champion was not touched"
        )
    spec = expected_spec or current_spec
    artifact_path = winners_csv or champion_winners_path(experiment_id)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Champion experiment {experiment_id} completed without winners artifact "
            f"{artifact_path}"
        )
    routing_checksum = sha256_file(artifact_path)
    winners_df, winners, is_ensemble = _load_cached_winners(
        artifact_path,
        list(spec.models),
    )
    if winners_df.empty or not winners:
        raise RuntimeError(
            f"Champion experiment {experiment_id} produced an empty winners artifact"
        )

    required_columns = {"item_id", "customer_group", "loc", "startdate", "model_id"}
    missing_columns = required_columns - set(winners_df.columns)
    if missing_columns:
        raise RuntimeError(
            f"Champion winners artifact is missing required columns: {sorted(missing_columns)}"
        )

    previous_experiment_id: int | None = None
    champion_rows = 0
    ceiling_rows = 0
    already_promoted = False
    with _get_conn() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (_PROMOTION_LOCK,))
        cur.execute(
            """SELECT status, models, strategy, strategy_params,
                      meta_learner_params, metric, lag_mode, min_sku_rows,
                      cluster_experiment_id, is_promoted, is_results_promoted,
                      results_artifact_checksum, results_forecast_checksum,
                      results_forecast_row_count
               FROM champion_experiment
               WHERE experiment_id = %s
               FOR UPDATE""",
            (experiment_id,),
        )
        experiment_row = cur.fetchone()
        if experiment_row is None:
            raise RuntimeError(f"Champion experiment {experiment_id} does not exist")
        (
            already_promoted,
            stored_routing_checksum,
            stored_results_checksum,
            stored_row_count,
        ) = _validated_experiment_spec(experiment_row, expected=spec)

        cur.execute(
            """SELECT experiment_id
               FROM cluster_experiment
               WHERE is_promoted = TRUE
               ORDER BY promoted_at DESC NULLS LAST, experiment_id DESC
               LIMIT 1
               FOR SHARE"""
        )
        cluster_row = cur.fetchone()
        if cluster_row is None or int(cluster_row[0]) != spec.cluster_experiment_id:
            raise RuntimeError(
                "The promoted clustering experiment changed during champion-refresh; "
                "the incumbent champion was left unchanged"
            )

        if already_promoted:
            results_stats = compute_champion_results_stats(cur, experiment_id)
            if (
                stored_routing_checksum != routing_checksum
                or stored_results_checksum != results_stats.checksum
                or stored_row_count != results_stats.row_count
                or results_stats.row_count <= 0
            ):
                raise RuntimeError(
                    "Promoted champion evidence no longer matches its stored checksums"
                )
            champion_rows = results_stats.row_count
        else:
            cur.execute(
                """SELECT experiment_id
                   FROM champion_experiment
                   WHERE is_promoted = TRUE
                     AND experiment_id != %s
                   ORDER BY promoted_at DESC NULLS LAST, experiment_id DESC
                   LIMIT 1""",
                (experiment_id,),
            )
            previous_row = cur.fetchone()
            previous_experiment_id = int(previous_row[0]) if previous_row else None

            if is_ensemble:
                inserted = insert_ensemble_forecasts(
                    cur,
                    winners_df,
                    spec.champion_model_id,
                    experiment_id,
                )
            else:
                inserted = insert_champion_forecasts(
                    cur,
                    winners,
                    spec.champion_model_id,
                    experiment_id,
                )
            fallback_inserted = insert_fallback_champions(
                cur,
                spec.lag_mode,
                spec.champion_model_id,
                spec.fallback_model_id,
                experiment_id,
            )
            champion_rows = int(inserted or 0) + int(fallback_inserted or 0)

            ceiling_winners = compute_ceiling_winners(
                cur,
                list(spec.models),
                spec.lag_mode,
            )
            ceiling_rows = insert_ceiling_forecasts(cur, ceiling_winners, "ceiling")

            results_stats = compute_champion_results_stats(cur, experiment_id)
            if results_stats.row_count <= 0:
                raise RuntimeError(
                    "Governed champion load produced no experiment-stamped rows; "
                    "the incumbent was left unchanged"
                )
            if champion_rows != results_stats.row_count:
                raise RuntimeError(
                    "Governed champion row-count audit failed; the incumbent was left unchanged"
                )

            cur.execute(
                """UPDATE champion_experiment
                   SET is_promoted = FALSE,
                       is_results_promoted = FALSE
                   WHERE experiment_id != %s
                     AND (is_promoted = TRUE OR is_results_promoted = TRUE)""",
                (experiment_id,),
            )
            cur.execute(
                """UPDATE champion_experiment
                   SET is_promoted = TRUE,
                       promoted_at = NOW(),
                       is_results_promoted = TRUE,
                       results_promoted_at = NOW(),
                       results_promote_job_id = %s,
                       results_artifact_checksum = %s,
                       results_forecast_checksum = %s,
                       results_forecast_row_count = %s
                   WHERE experiment_id = %s
                     AND status = 'completed'
                     AND cluster_experiment_id = %s""",
                (
                    job_id,
                    routing_checksum,
                    results_stats.checksum,
                    results_stats.row_count,
                    experiment_id,
                    spec.cluster_experiment_id,
                ),
            )
            if int(cur.rowcount or 0) != 1:
                raise RuntimeError(
                    "Champion promotion compare-and-swap failed; the incumbent was left unchanged"
                )
            cur.execute(
                """INSERT INTO champion_promotion_log
                       (experiment_id, promoted_by, previous_experiment_id,
                        strategy, champion_accuracy, config_snapshot)
                   SELECT experiment_id, %s, %s, strategy,
                          champion_accuracy, %s
                   FROM champion_experiment
                   WHERE experiment_id = %s""",
                (
                    "champion-refresh",
                    previous_experiment_id,
                    json.dumps(_promotion_snapshot(spec)),
                    experiment_id,
                ),
            )
            if int(cur.rowcount or 0) != 1:
                raise RuntimeError(
                    "Champion promotion audit insert failed; the incumbent was left unchanged"
                )

    view_refresh: dict[str, list[str]] | None = None
    if refresh_views:
        view_refresh = refresh_for_tables(["fact_external_forecast_monthly"])

    result = {
        "experiment_id": experiment_id,
        "previous_experiment_id": previous_experiment_id,
        "backtest_run_ids": dict(spec.backtest_run_ids),
        "source_sales_batch_id": spec.source_sales_batch_id,
        "data_checksum": spec.data_checksum,
        "cluster_experiment_id": spec.cluster_experiment_id,
        "cluster_assignment_count": spec.cluster_assignment_count,
        "cluster_assignment_checksum": spec.cluster_assignment_checksum,
        "routing_artifact_checksum": routing_checksum,
        "results_forecast_checksum": results_stats.checksum,
        "results_forecast_row_count": results_stats.row_count,
        "champion_rows": champion_rows,
        "ceiling_rows": int(ceiling_rows or 0),
        "already_promoted": already_promoted,
        "view_refresh": view_refresh,
    }
    if spec.source_experiment_id is not None:
        result["source_experiment_id"] = spec.source_experiment_id
    return result


def run_governed_champion_refresh(
    params: dict[str, Any],
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Create, run, and atomically promote a governed champion experiment."""
    if params.get("experiment_id") is not None:
        raise ValueError(
            "governed_champion_refresh creates its own experiment; experiment_id is not accepted"
        )
    raw_source_experiment_id = params.get("source_experiment_id")
    if raw_source_experiment_id is not None and not _positive_integer(raw_source_experiment_id):
        raise ValueError("source_experiment_id must be a positive integer")
    source_candidate = None
    if progress_cb:
        progress_cb(pct=5, msg="Validating canonical five-model champion inputs")
    spec = load_refresh_spec()
    if raw_source_experiment_id is not None:
        source_candidate = load_champion_assignment_candidate(raw_source_experiment_id)
        spec = build_selected_refresh_spec(spec, source_candidate)

    if progress_cb:
        progress_cb(pct=10, msg="Creating governed champion experiment")
    if source_candidate is None:
        experiment_id = create_governed_experiment(spec, job_id=job_id)
    else:
        experiment_id = create_governed_experiment(
            spec,
            job_id=job_id,
            source_candidate=source_candidate,
        )
    persist_job_experiment_id(job_id, experiment_id, spec)

    if progress_cb:
        progress_cb(pct=15, msg=f"Evaluating champion experiment #{experiment_id}")
    run_champion_experiment_job(
        experiment_id,
        progress_cb=progress_cb,
        cancel_event=cancel_event,
        job_id=job_id,
    )

    artifact_path = champion_winners_path(experiment_id)
    if progress_cb:
        progress_cb(
            pct=75,
            msg=(
                f"Atomically promoting champion experiment #{experiment_id}; "
                "the incumbent remains active until commit"
            ),
        )
    result = finalize_governed_champion_refresh(
        experiment_id,
        job_id=job_id,
        winners_csv=artifact_path,
        expected_spec=spec,
    )
    if source_candidate is not None:
        result["source_experiment_id"] = source_candidate.experiment_id
    if progress_cb:
        progress_cb(pct=100, msg=f"Champion experiment #{experiment_id} is active")
    return result
