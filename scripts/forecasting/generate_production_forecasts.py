"""
F1.1 — Production Forecast Generation Pipeline

Runs champion ML models over a forward horizon and writes predictions to
fact_production_forecast_staging. This is the bridge between the backtesting engine
(which evaluates historical accuracy) and operational planning (which needs
future-period demand signals).

The key difference from backtesting:
  - Backtest: predicts historical months where actuals exist (for model evaluation)
  - Production: predicts future months where no actuals exist yet (for planning)

Usage:
    uv run python scripts/generate_production_forecasts.py
    uv run python scripts/generate_production_forecasts.py --horizon 6
    uv run python scripts/generate_production_forecasts.py --dfu 100320 1401-BULK
    uv run python scripts/generate_production_forecasts.py --dry-run

Algorithm:
    For each DFU in champion assignments:
        1. Load the cluster's .pkl model artifact
        2. Build an inference grid (feature matrix for T+1 through T+horizon)
        3. Generate predictions recursively: lag_1 for T+2 = model's T+1 prediction
        4. Write to fact_production_forecast_staging with upsert on (model_id, item_id, loc, forecast_month)
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
import time
import uuid
import warnings
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from lightgbm.basic import LightGBMError

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psycopg
from psycopg.types.json import Jsonb

from common.core.constants import (
    CAT_FEATURES,
    CHAMPION_MODEL_ID,
    CROSTON_FEATURES,
    DERIVED_FEATURES,
    LAG_RANGE,
    ROLLING_WINDOWS,
)
from common.core.db import get_db_params
from common.core.planning_date import get_planning_date
from common.core.sql_helpers import read_sql_chunked
from common.core.utils import get_forecastable_model_ids, load_forecast_pipeline_config
from common.ml.direct_model_lineage import (
    DIRECT_MODEL_CONFIG_METADATA_KEY,
    SOURCE_MODEL_ROSTER_METADATA_KEY,
    build_direct_model_config_lineage,
)
from common.ml.feature_engineering import compute_ts_profile_from_values
from common.ml.forecast_ci import build_sigma_lookup, compute_ci_bounds
from common.ml.generation_config_lineage import (
    GENERATION_CONFIG_METADATA_KEY,
    build_generation_config_lineage,
)
from common.ml.neural_artifacts import (
    LoadedNeuralArtifact,
    load_active_neural_artifact,
    load_neural_training_cohort_identity,
)
from common.ml.neural_forecast import (
    SUPPORTED_NEURAL_MODELS,
    FittedNeuralModel,
    NeuralCohortIdentity,
)
from common.ml.production_non_tree import run_canonical_non_tree_forecast
from common.ml.tree_artifact_lineage import ProductionTreeArtifactLineage
from common.ml.tree_artifacts import (
    LoadedTreeArtifactSet,
    TreeArtifactSpec,
    build_production_tree_model_config_payload,
    build_tree_artifact_spec,
    load_active_tree_artifact_set,
)
from common.services.champion_lineage import (
    GOVERNED_CHAMPION_LINEAGE_METADATA_KEY,
    GovernedChampionLineageError,
    load_active_governed_champion_lineage,
    load_governed_champion_lineage,
)
from common.services.cluster_lineage import load_promoted_cluster_population
from common.services.forecast_generation import (
    build_generation_metadata,
    invalidate_generation_run,
    reserve_generation_run,
)
from common.services.forecast_lineage import (
    compute_staging_payload_stats,
    sha256_file,
)
from common.services.forecast_population import (
    build_forecast_eligibility_ctes,
    resolve_forecast_sales_table,
)
from common.services.perf_profiler import profiled_section
from common.services.sales_lineage import load_completed_sales_lineage

PIPELINE_CONFIG_PATH = ROOT / "config" / "forecasting" / "forecast_pipeline_config.yaml"
CHAMPION_WINNERS_DIR = ROOT / "data" / "champion"

logger = logging.getLogger(__name__)


def _to_cluster_id(cluster_id) -> int | str | None:
    """Normalize cluster_id for DB storage.

    ml_cluster can be an integer (e.g. 0, 3) or a string label
    (e.g. 'high_volume_declining'). Try int conversion; keep as str if not numeric.
    """
    if cluster_id is None or bool(pd.isna(cluster_id)):
        return None
    try:
        return int(cluster_id)
    except (ValueError, TypeError):
        return str(cluster_id)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config() -> dict:
    """Load production forecast config.

    Reads from the consolidated forecast_pipeline_config.yaml (production_forecast section).
    """
    if not PIPELINE_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Pipeline config not found: {PIPELINE_CONFIG_PATH}")
    with open(PIPELINE_CONFIG_PATH) as f:
        pipeline = yaml.safe_load(f) or {}
    pf = pipeline.get("production_forecast", {})
    # Map to the structure the rest of the script expects
    return {
        "inference": {
            "horizon_months": pf.get("horizon_months", 24),
            "recursive": pf.get("recursive", True),
        },
        "confidence_interval": pf.get("confidence_interval", {}),
        "model_selection": {
            "strategy": "champion",
            "fallback_model_id": pf.get("fallback_model_id", "lgbm_cluster"),
        },
        "plan_version": {
            "format": pf.get("plan_version_format", "%Y-%m"),
            "keep_last_n_versions": pf.get("keep_last_n_versions", 3),
        },
        "model_registry": pf.get("model_registry", {"base_path": "data/models"}),
        "algorithms": pipeline.get("algorithms", {}),
        "_full_pipeline": pipeline,
        # New cold-start fields (used in main loop)
        "_pipeline": {
            "lookback_months": pf["lookback_months"],
            "min_history_months": pf.get("min_history_months", 12),
            "cold_start_model_id": pf.get("cold_start_model_id", "lgbm_cluster"),
            "cold_start_min_months": pf.get("cold_start_min_months", 3),
            "active_window_months": pipeline["forecast_snapshot"]["active_window_months"],
            "clustering_enabled": pipeline["clustering"]["enabled"],
        },
    }


def _validate_forecastable_model_ids(model_ids: set[str], *, source: str) -> None:
    """Reject retired model IDs before loading artifacts or writing staging rows."""
    valid_models = get_forecastable_model_ids()
    unsupported = sorted(model_ids - set(valid_models))
    if unsupported:
        raise ValueError(
            f"{source} contains unsupported production forecast model(s) {unsupported}; "
            f"valid configured models are {valid_models}"
        )


def load_non_tree_model_ids() -> set[str]:
    """Champion source models that have NO persisted ``.pkl`` and are generated
    by their canonical production adapters.

    These are every algorithm whose config ``type`` is not ``tree`` — the
    MSTL, the foundation model, and the deep-learning models. Routing a champion whose
    ``source_model_id`` is one of these through its true adapter (rather
    than the tree-batch path) keeps the SHIPPED model equal to the DISPLAYED
    champion: otherwise the tree path substitutes ``fallback_model_id``.
    Tree champions keep the tree-batch path.
    """
    with open(PIPELINE_CONFIG_PATH) as f:
        pipeline = yaml.safe_load(f) or {}
    algos = pipeline.get("algorithms", {})
    forecastable = set(get_forecastable_model_ids())
    return {
        mid
        for mid, spec in algos.items()
        if mid in forecastable and (spec or {}).get("type") != "tree"
    }


def _algorithm_params(config: dict[str, Any], model_id: str) -> dict[str, Any]:
    """Return the configured parameters for one canonical model."""
    entry = config.get("algorithms", {}).get(model_id)
    if not isinstance(entry, dict) or not isinstance(entry.get("params"), dict):
        raise ValueError(f"Production forecast parameters are missing for {model_id}")
    return dict(entry["params"])


def _population_min_history(
    model_id: str | None,
    config: dict[str, Any],
    *,
    cold_start_min_months: int,
) -> int:
    """Return the observation floor for one direct-model population."""
    if model_id in {"chronos2_enriched", "mstl", "nhits", "nbeats"}:
        return max(
            cold_start_min_months,
            int(_algorithm_params(config, model_id)["min_history"]),
        )
    return cold_start_min_months


def _generate_canonical_non_tree_rows(
    *,
    config: dict[str, Any],
    model_id: str,
    sales_df: pd.DataFrame,
    dfu_attrs: pd.DataFrame,
    item_attrs: pd.DataFrame,
    target_dfus: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    forecast_month_generated,
    run_id: str,
    fitted_neural_model: FittedNeuralModel | None = None,
) -> list[dict[str, Any]]:
    """Run one non-tree model with its exact algorithm YAML parameters.

    Confidence intervals are attached only after customer-group aggregation,
    so one item/location residual is not duplicated for each group.
    """
    return run_canonical_non_tree_forecast(
        model_id=model_id,
        sales_df=sales_df,
        dfu_attrs=dfu_attrs,
        item_attrs=item_attrs,
        target_dfus=target_dfus,
        predict_months=predict_months,
        params=_algorithm_params(config, model_id),
        forecast_month_generated=forecast_month_generated,
        run_id=run_id,
        sigma_lookup={},
        ci_cfg=None,
        fitted_neural_model=fitted_neural_model,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_active_models(
    model_id: str,
    config: dict,
    *,
    expected_spec: TreeArtifactSpec | None = None,
) -> LoadedTreeArtifactSet:
    """Load one checksummed active all-cluster version before inference."""
    if expected_spec is None:
        raise ValueError("Production tree loading requires an expected artifact spec")
    base_path = config.get("model_registry", {}).get("base_path", "data/models")
    base_dir = Path(base_path)
    if not base_dir.is_absolute():
        base_dir = ROOT / base_dir
    loaded = load_active_tree_artifact_set(
        model_id=model_id,
        expected_spec=expected_spec,
        base_dir=base_dir,
    )
    logger.info(
        "Loaded %d %s cluster models from immutable set %s",
        len(loaded.artifacts),
        model_id,
        loaded.ref.artifact_set_id,
    )
    return loaded


class _LoadedTreeModelMap(dict[str, dict[str, dict[str, Any]]]):
    """Runtime artifact map paired with immutable set references."""

    def __init__(self) -> None:
        super().__init__()
        self.artifact_sets: dict[str, LoadedTreeArtifactSet] = {}


def _load_tree_models(
    model_ids_needed: set[str],
    non_tree_models: set[str],
    config: dict[str, Any],
    *,
    expected_specs: Mapping[str, TreeArtifactSpec] | None = None,
) -> _LoadedTreeModelMap:
    """Load artifacts only for trees; canonical adapters ignore stale pickles."""
    loaded = _LoadedTreeModelMap()
    for model_id in sorted(model_ids_needed - non_tree_models):
        try:
            if expected_specs is None:
                # Kept only as a narrow test seam; the real loader refuses an
                # unvalidated call when it has not been monkeypatched.
                raw: Any = load_active_models(model_id, config)
                loaded[model_id] = raw
                continue
            spec = expected_specs.get(model_id)
            if spec is None:
                raise RuntimeError(f"Missing expected tree artifact spec for {model_id}")
            artifact_set = load_active_models(
                model_id,
                config,
                expected_spec=spec,
            )
            loaded[model_id] = artifact_set.artifacts
            loaded.artifact_sets[model_id] = artifact_set
        except FileNotFoundError as exc:
            logger.warning("%s", exc)
    return loaded


def _load_completed_sales_lineage(conn) -> tuple[int, str]:
    """Return the immutable lineage expected by production neural artifacts."""
    lineage = load_completed_sales_lineage(conn)
    return lineage.batch_id, lineage.source_hash


def _load_neural_models(
    model_ids_needed: set[str],
    config: dict[str, Any],
    *,
    source_sales_batch_id: int,
    data_checksum: str,
    history_end,
    expected_cohorts: Mapping[str, NeuralCohortIdentity],
) -> dict[str, LoadedNeuralArtifact]:
    """Load each required neural artifact once under exact data/config lineage."""
    base_path = Path(config.get("model_registry", {}).get("base_path", "data/models"))
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    loaded: dict[str, LoadedNeuralArtifact] = {}
    for model_id in sorted(model_ids_needed & set(SUPPORTED_NEURAL_MODELS)):
        expected_cohort = expected_cohorts.get(model_id)
        if expected_cohort is None:
            raise RuntimeError(
                f"Required {model_id} artifact has no current training-cohort identity"
            )
        loaded[model_id] = load_active_neural_artifact(
            model_id=model_id,
            params=_algorithm_params(config, model_id),
            source_sales_batch_id=source_sales_batch_id,
            data_checksum=data_checksum,
            history_end=history_end,
            base_dir=base_path,
            expected_training_cohort_checksum=expected_cohort.checksum,
            expected_training_dfu_count=expected_cohort.dfu_count,
        )
    return loaded


def _neural_generation_metadata(
    loaded: dict[str, LoadedNeuralArtifact],
) -> dict[str, Any]:
    """Return immutable neural artifact lineage for the generation manifest."""
    if not loaded:
        return {}
    keys = (
        "config_checksum",
        "data_checksum",
        "source_sales_batch_id",
        "history_end",
        "training_cohort_checksum",
        "training_data_checksum",
        "training_contract_version",
        "runtime_contract_checksum",
    )
    return {
        "neural_artifacts": {
            model_id: {
                "artifact_id": artifact.ref.artifact_id,
                **{key: artifact.ref.metadata[key] for key in keys},
            }
            for model_id, artifact in sorted(loaded.items())
        }
    }


def _tree_generation_metadata(loaded: _LoadedTreeModelMap) -> dict[str, Any]:
    """Return the exact immutable LightGBM set used by this generation."""
    if not loaded.artifact_sets:
        return {}
    keys = (
        "config_checksum",
        "cluster_strategy",
        "cluster_labels",
        "lineage",
    )
    return {
        "tree_artifacts": {
            model_id: {
                "artifact_set_id": artifact_set.ref.artifact_set_id,
                **{key: artifact_set.ref.metadata[key] for key in keys},
            }
            for model_id, artifact_set in sorted(loaded.artifact_sets.items())
        }
    }


def _source_generation_metadata(
    *,
    source_sales_batch_id: int,
    data_checksum: str,
    history_end,
) -> dict[str, Any]:
    """Stamp source lineage even when a contender model has no artifact."""
    return {
        "source_sales": {
            "source_sales_batch_id": source_sales_batch_id,
            "data_checksum": data_checksum,
            "history_end": pd.Timestamp(history_end).date().isoformat(),
        }
    }


def _direct_generation_metadata(
    config: dict[str, Any],
    model_ids: set[str],
) -> dict[str, Any]:
    """Stamp the exact pre-aggregation source roster and direct-model configs."""
    algorithms = config.get("algorithms")
    if not isinstance(algorithms, dict):
        raise ValueError("Production forecast algorithm configuration is unavailable")
    roster = sorted(model_ids)
    if not roster:
        raise ValueError("Production forecast source-model roster must not be empty")
    return {
        SOURCE_MODEL_ROSTER_METADATA_KEY: roster,
        DIRECT_MODEL_CONFIG_METADATA_KEY: build_direct_model_config_lineage(
            algorithms,
            model_ids,
        ),
    }


def _generation_config_metadata(
    config: dict[str, Any],
    model_ids: set[str],
) -> dict[str, Any]:
    """Stamp the current global/model config used to create staged output."""
    full_pipeline = config.get("_full_pipeline")
    if not isinstance(full_pipeline, dict):
        raise ValueError("Full forecast pipeline configuration is unavailable")
    return {
        GENERATION_CONFIG_METADATA_KEY: build_generation_config_lineage(
            full_pipeline,
            model_ids,
        )
    }


def _begin_generation_snapshot(conn) -> None:
    """Start one stable snapshot that can persist the generated candidate."""
    conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")


def _resolve_tree_artifact(
    loaded_models: dict[str, dict],
    model_id: str,
    cluster_id: object,
) -> dict | None:
    """Resolve the exact current-cluster artifact; never substitute another cluster."""
    cluster_models = loaded_models.get(model_id)
    if cluster_models is None:
        return None

    def _validate(artifact: dict, expected_cluster: str) -> dict:
        if artifact.get("model_id") != model_id:
            raise RuntimeError(f"{model_id} production artifact has mismatched model_id")
        if artifact.get("training_mode") != "production":
            raise RuntimeError(f"{model_id} artifact is not a production final fit")
        if str(artifact.get("cluster_label")) != expected_cluster:
            raise RuntimeError(f"{model_id} production artifact has mismatched cluster_label")
        return artifact

    if set(cluster_models) == {"global"}:
        global_artifact = cluster_models["global"]
        if global_artifact.get("cluster_strategy") != "global":
            raise RuntimeError(f"{model_id} global artifact does not declare global training")
        return _validate(global_artifact, "global")

    normalized = _to_cluster_id(cluster_id)
    candidates = (cluster_id, normalized, str(normalized) if normalized is not None else None)
    for candidate in candidates:
        if candidate in cluster_models:
            return _validate(cluster_models[candidate], str(normalized))
    raise RuntimeError(
        f"Required {model_id} production artifact for current cluster {normalized!s} is missing; "
        "retrain production models after the promoted clustering change"
    )


# ---------------------------------------------------------------------------
# Champion assignment query
# ---------------------------------------------------------------------------


def _get_promoted_champion_experiment_id(conn) -> int | None:
    """Return the active promoted champion experiment, if one exists."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT experiment_id
            FROM champion_experiment
            WHERE is_promoted = TRUE
            ORDER BY promoted_at DESC
            LIMIT 1
        """)
        row = cur.fetchone()
    return int(row[0]) if row else None


def check_champion_cluster_lineage(conn, allow_mismatch: bool = False) -> None:
    """Refuse to generate when the promoted champion predates a re-clustering.

    The champion's winners CSV and the per-cluster ``.pkl`` artifacts were
    built under the cluster experiment recorded on ``champion_experiment``
    (sql/198). If clustering was re-promoted since, the current
    promoted ``current_sku_cluster_assignment`` no longer matches that membership and per-cluster
    routing silently degrades. ``--allow-cluster-mismatch`` downgrades the
    failure to a warning.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'champion_experiment'
                  AND column_name = 'cluster_experiment_id'
                LIMIT 1
            """)
            if cur.fetchone() is None:
                logger.warning(
                    "champion_experiment.cluster_experiment_id missing (apply sql/198) — "
                    "cluster-generation preflight skipped."
                )
                return
            cur.execute("""
                SELECT experiment_id, cluster_experiment_id
                FROM champion_experiment
                WHERE is_promoted = TRUE
                ORDER BY promoted_at DESC
                LIMIT 1
            """)
            champ_row = cur.fetchone()
            cur.execute("""
                SELECT experiment_id FROM cluster_experiment
                WHERE is_promoted ORDER BY promoted_at DESC LIMIT 1
            """)
            cluster_row = cur.fetchone()
    except psycopg.Error as exc:
        logger.exception("Champion/cluster lineage preflight failed")
        raise RuntimeError("Champion/cluster lineage could not be verified") from exc

    if champ_row is None:
        return  # no promoted champion — downstream handles that as before
    champ_id, champ_cluster_id = champ_row
    current_cluster_id = cluster_row[0] if cluster_row else None
    if champ_cluster_id is None:
        logger.warning(
            "Champion experiment %s has no cluster lineage (pre-sql/198 row); "
            "generation match cannot be verified.",
            champ_id,
        )
        return
    if current_cluster_id is not None and int(champ_cluster_id) != int(current_cluster_id):
        msg = (
            f"Champion experiment {champ_id} was computed under cluster experiment "
            f"{champ_cluster_id}, but cluster experiment {current_cluster_id} is now "
            "promoted — winners routing and per-cluster models likely mismatch the "
            "current promoted SKU cluster assignments. Re-run backtests + champion selection under "
            "the new clusters, or pass --allow-cluster-mismatch to override."
        )
        if not allow_mismatch:
            raise RuntimeError(msg)
        logger.warning("%s (continuing: --allow-cluster-mismatch)", msg)


def _load_promoted_winners_assignments(
    conn,
    item_id: str | None = None,
    loc: str | None = None,
) -> pd.DataFrame | None:
    """Load champion routing from the promoted experiment winners CSV.

    The promote endpoint treats ``data/champion/experiment_<id>_winners.csv`` as
    source of truth. Production generation must use the same source; otherwise it
    can stage forecasts from stale ``fact_external_forecast_monthly`` champion
    rows and the product shows a promoted champion that was not actually used.
    Returns ``None`` only when no promoted champion experiment exists, allowing
    legacy installs to fall back to the historical DB-based route.
    """
    experiment_id = _get_promoted_champion_experiment_id(conn)
    if experiment_id is None:
        return None

    winners_path = CHAMPION_WINNERS_DIR / f"experiment_{experiment_id}_winners.csv"
    if not winners_path.exists():
        raise FileNotFoundError(
            f"Promoted champion experiment {experiment_id} has no winners file: "
            f"{winners_path}. Re-run/promote a champion experiment before generating "
            "production champion forecasts."
        )

    winners = pd.read_csv(
        winners_path,
        dtype={"item_id": str, "customer_group": str, "loc": str, "model_id": str},
    )
    required = {"item_id", "customer_group", "loc", "model_id", "startdate"}
    missing = required - set(winners.columns)
    if missing:
        raise ValueError(
            f"Winners file {winners_path.name} is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )
    invalid_customer_group = winners["customer_group"].isna() | (
        winners["customer_group"].astype(str).str.strip() == ""
    )
    if invalid_customer_group.any():
        raise ValueError(f"Winners file {winners_path.name} contains a blank customer_group")
    if item_id:
        winners = winners[winners["item_id"] == str(item_id)]
    if loc:
        winners = winners[winners["loc"] == str(loc)]
    if winners.empty:
        return pd.DataFrame(
            columns=[
                "item_id",
                "loc",
                "startdate",
                "source_model_id",
                "source_mix",
                "cluster_id",
                "customer_group",
            ]
        )

    winners = winners.copy()
    winners["startdate"] = pd.to_datetime(winners["startdate"]).dt.to_period("M").dt.to_timestamp()
    winners["source_model_id"] = winners["model_id"].astype(str)
    if "source_mix" in winners.columns:
        winners["source_mix"] = winners["source_mix"].apply(_parse_ensemble_mix)
    else:
        winners["source_mix"] = None
    sort_cols = [
        "item_id",
        "customer_group",
        "loc",
        "startdate",
        "source_model_id",
    ]
    winners = winners.sort_values(sort_cols).drop_duplicates(
        ["item_id", "customer_group", "loc", "startdate"], keep="first"
    )

    attrs = load_dfu_attrs(conn, item_id, loc)
    if attrs.empty:
        winners["cluster_id"] = pd.NA
        winners["sku_ck"] = pd.NA
        return winners[
            [
                "sku_ck",
                "item_id",
                "loc",
                "startdate",
                "source_model_id",
                "source_mix",
                "cluster_id",
                "customer_group",
            ]
        ]

    attrs = attrs.copy()
    attrs["item_id"] = attrs["item_id"].astype(str)
    attrs["loc"] = attrs["loc"].astype(str)
    attrs["customer_group"] = attrs["customer_group"].astype(str)
    attrs = attrs.rename(columns={"ml_cluster": "cluster_id"})
    merged = winners.merge(
        attrs[["sku_ck", "item_id", "customer_group", "loc", "cluster_id"]],
        on=["item_id", "customer_group", "loc"],
        how="left",
        validate="many_to_one",
    )

    df = merged[
        [
            "sku_ck",
            "item_id",
            "loc",
            "startdate",
            "source_model_id",
            "source_mix",
            "cluster_id",
            "customer_group",
        ]
    ]
    logger.info(
        "Promoted champion assignments loaded from %s: %s DFU-months across %s DFUs",
        winners_path.name,
        f"{len(df):,}",
        f"{df.drop_duplicates(['item_id', 'customer_group', 'loc']).shape[0]:,}",
    )
    return df.reset_index(drop=True)


def _parse_ensemble_mix(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, (list, str)):
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError) as exc:
                raise ValueError("ensemble source_mix is not valid JSON") from exc
    else:
        parsed = value
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("ensemble source_mix must contain at least one member")

    allowed_models = set(get_forecastable_model_ids())
    normalized: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    for entry in parsed:
        if not isinstance(entry, dict):
            raise ValueError("ensemble source_mix members must be objects")
        model_id = entry.get("model")
        weight = entry.get("weight")
        if not isinstance(model_id, str) or model_id not in allowed_models:
            raise ValueError("ensemble source_mix contains an unsupported model")
        if model_id in seen_models:
            raise ValueError("ensemble source_mix contains a duplicate model")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("ensemble source_mix weights must be numeric")
        numeric_weight = float(weight)
        if not np.isfinite(numeric_weight) or numeric_weight < 0:
            raise ValueError("ensemble source_mix weights must be finite and nonnegative")
        seen_models.add(model_id)
        normalized.append({"model": model_id, "weight": numeric_weight})

    total_weight = sum(entry["weight"] for entry in normalized)
    if total_weight <= 0 or abs(total_weight - 1.0) > 1e-3:
        raise ValueError("ensemble source_mix weights must sum to one")
    for entry in normalized:
        entry["weight"] /= total_weight
    return normalized


def build_ensemble_routing(
    champion_df: pd.DataFrame,
) -> dict[tuple[str, str, str], dict[pd.Timestamp, list[dict[str, Any]]]]:
    routing: dict[tuple[str, str, str], dict[pd.Timestamp, list[dict[str, Any]]]] = defaultdict(
        dict
    )
    if "source_mix" not in champion_df.columns:
        return routing
    for row in champion_df.itertuples(index=False):
        raw_mix = getattr(row, "source_mix", None)
        source_model_id = getattr(row, "source_model_id", None)
        if source_model_id != "ensemble" and not raw_mix:
            continue
        mix = _parse_ensemble_mix(raw_mix)
        if source_model_id == "ensemble" and mix is None:
            raise ValueError("ensemble source_mix is required for an ensemble route")
        if mix is None:
            continue
        # A top-K strategy can legitimately have only one eligible model for
        # an early/sparse DFU-month. Route that row as the underlying model;
        # it is not an ensemble and should not create a synthetic sixth model.
        if len(mix) == 1:
            continue
        month = pd.Timestamp(row.startdate).normalize().replace(day=1)
        routing[(row.item_id, row.customer_group, row.loc)][month] = mix
    return routing


def get_champion_assignments(
    conn, item_id: str | None = None, loc: str | None = None
) -> pd.DataFrame:
    """Return the champion model assignment per DFU-MONTH.

    Returns `source_model_id` — the underlying algorithm (e.g. lgbm_cluster) whose
    artifacts are used for production inference.  Populated by champion selection
    via the source_model_id column (added in F1.1 via sql/007_create_fact_external_forecast_monthly.sql).
    Every champion row written by run_champion_selection now carries a source —
    single-model winners (the winning model), ensemble winners (the literal
    "ensemble"), and warm-up/uncovered-month fallback rows (the champion-config
    fallback model). NULL therefore means only a legacy row written before that
    column existed; the caller falls back to `fallback_model_id` from config.

    Champion routing is genuinely (item_id, customer_group, loc,
    forecast_month)-grained: a DFU
    can win a different model for each month. This returns ONE row per full
    DFU-month so the month dimension is preserved downstream
    (collapsing to one row per DFU shipped the latest month's model
    across the whole horizon — see issue promote-per-month-collapse).

    Returns DataFrame with columns:
        item_id, loc, startdate, source_model_id, cluster_id, customer_group.
    """
    promoted_df = _load_promoted_winners_assignments(conn, item_id, loc)
    if promoted_df is not None:
        return promoted_df

    where_clauses = ["f.model_id = 'champion'"]
    params: list = []

    if item_id:
        where_clauses.append("f.item_id = %s")
        params.append(item_id)
    if loc:
        where_clauses.append("f.loc = %s")
        params.append(loc)

    where_sql = " AND ".join(where_clauses)

    # Preserve customer_group because tree/statistical inference runs at that
    # training grain. Group forecasts are summed to item/location only after
    # each group has been generated under its own winner.
    sql = f"""
        SELECT DISTINCT ON (f.item_id, f.customer_group, f.loc, f.startdate)
            d.sku_ck,
            f.item_id                   AS item_id,
            f.customer_group,
            f.loc,
            f.startdate,
            f.source_model_id,
            f.source_mix,
            ca.ml_cluster               AS cluster_id
        FROM fact_external_forecast_monthly f
        JOIN dim_sku d ON d.item_id = f.item_id
                      AND d.customer_group = f.customer_group
                      AND d.loc = f.loc
        LEFT JOIN current_sku_cluster_assignment ca
               ON ca.sku_ck = d.sku_ck
        WHERE {where_sql}
        ORDER BY f.item_id, f.customer_group, f.loc, f.startdate,
                 f.source_model_id ASC NULLS LAST
    """

    # Streamed: scans fact_external_forecast_monthly. DISTINCT ON shrinks the
    # output to one row per full DFU-month, but the underlying scan
    # is still fact-table-sized.
    df = read_sql_chunked(conn, sql, params=params)
    with_src = int(df["source_model_id"].notna().sum())
    n_dfus = df.drop_duplicates(["item_id", "customer_group", "loc"]).shape[0] if len(df) else 0
    logger.info(
        "Champion assignments loaded: %s DFU-months across %s DFUs (%s with source_model_id)",
        f"{len(df):,}",
        f"{n_dfus:,}",
        f"{with_src:,}",
    )
    return df


_DFU_KEY_COLUMNS = ["item_id", "customer_group", "loc"]


def load_forecast_population(
    conn,
    *,
    planning_month,
    min_history_months: int,
    active_window_months: int,
    item_id: str | None = None,
    loc: str | None = None,
    sales_table: str | None = None,
) -> pd.DataFrame:
    """Load every active customer group for release-eligible item/locations.

    The release gate measures coverage at the production table's
    ``(item_id, loc)`` grain.  Generation still happens at customer-group
    grain, so selecting groups independently could silently omit demand from a
    short-history group while the aggregate item/location appeared complete.
    The shared eligibility contract excludes the whole item/location unless
    every active group meets the requested history floor.
    """
    if sales_table is None:
        with conn.cursor() as cur:
            sales_table = resolve_forecast_sales_table(cur)
    eligibility = build_forecast_eligibility_ctes(
        planning_month=planning_month,
        min_history_months=min_history_months,
        active_window_months=active_window_months,
        item_id=item_id,
        loc=loc,
        sales_table=sales_table,
    )
    query = f"""
        WITH {eligibility.sql}
        SELECT d.sku_ck, d.item_id, d.customer_group, d.loc,
               ca.ml_cluster AS cluster_id
        FROM dim_sku d
        JOIN active_customer_groups active
          USING (item_id, customer_group, loc)
        JOIN eligible_item_locations eligible USING (item_id, loc)
        LEFT JOIN current_sku_cluster_assignment ca
          ON ca.sku_ck = d.sku_ck
        ORDER BY d.item_id, d.customer_group, d.loc
    """
    population = read_sql_chunked(conn, query, params=list(eligibility.params))
    logger.info(
        "Forecastable population loaded: %s active DFUs",
        f"{len(population):,}",
    )
    return population


def _align_routes_to_population(
    routes: pd.DataFrame,
    population: pd.DataFrame,
    *,
    fallback_model_id: str,
    planning_month,
) -> pd.DataFrame:
    """Drop stale routes and add explicit fallback rows for uncovered DFUs."""
    required_population = {"sku_ck", "cluster_id", *_DFU_KEY_COLUMNS}
    missing = required_population - set(population.columns)
    if missing:
        raise ValueError(f"Forecast population is missing columns: {sorted(missing)}")
    if population.empty:
        raise RuntimeError("No active forecast population meets release eligibility")

    population_columns = ["sku_ck", *_DFU_KEY_COLUMNS, "cluster_id"]
    normalized_population = population[population_columns].copy()
    identity_columns = ["sku_ck", *_DFU_KEY_COLUMNS]
    if normalized_population[identity_columns].isna().any(axis=1).any():
        raise ValueError("Forecast population contains a null DFU identity")
    for column in ["sku_ck", *_DFU_KEY_COLUMNS]:
        normalized_population[column] = normalized_population[column].astype(str)
    if normalized_population.duplicated(_DFU_KEY_COLUMNS).any():
        raise ValueError("Forecast population contains duplicate DFU identities")

    route_columns = [
        "startdate",
        "source_model_id",
        "source_mix",
        *_DFU_KEY_COLUMNS,
    ]
    normalized_routes = routes.copy()
    for column in route_columns:
        if column not in normalized_routes.columns:
            normalized_routes[column] = None
    for column in _DFU_KEY_COLUMNS:
        normalized_routes[column] = normalized_routes[column].astype(str)
    normalized_routes = normalized_routes[route_columns]
    normalized_routes["startdate"] = (
        pd.to_datetime(
            normalized_routes["startdate"],
            errors="coerce",
        )
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    planning_route_cutoff = pd.Timestamp(planning_month).to_period("M").to_timestamp()
    normalized_routes = normalized_routes[
        normalized_routes["source_model_id"].notna()
        & normalized_routes["startdate"].notna()
        & (normalized_routes["startdate"] <= planning_route_cutoff)
    ]
    routed = normalized_routes.merge(
        normalized_population,
        on=_DFU_KEY_COLUMNS,
        how="inner",
        validate="many_to_one",
    )

    routed_keys = routed[_DFU_KEY_COLUMNS].drop_duplicates()
    uncovered = normalized_population.merge(
        routed_keys,
        on=_DFU_KEY_COLUMNS,
        how="left",
        indicator=True,
    )
    uncovered = uncovered[uncovered["_merge"] == "left_only"].drop(columns="_merge")
    fallback = uncovered.assign(
        startdate=pd.Timestamp(planning_month).to_period("M").to_timestamp(),
        source_model_id=fallback_model_id,
        source_mix=None,
    )
    fallback = fallback[routed.columns]
    if routed.empty:
        aligned = fallback.reset_index(drop=True)
    elif fallback.empty:
        aligned = routed.reset_index(drop=True)
    else:
        aligned = pd.concat([routed, fallback], ignore_index=True)
    logger.info(
        "Champion routing aligned to population: %s routed, %s fallback DFUs",
        f"{routed_keys.shape[0]:,}",
        f"{len(uncovered):,}",
    )
    return aligned.sort_values(
        [*_DFU_KEY_COLUMNS, "startdate", "source_model_id"],
        na_position="last",
    ).reset_index(drop=True)


def build_month_routing(
    champion_df: pd.DataFrame,
) -> dict[tuple[str, str, str], dict[pd.Timestamp, str]]:
    """Map each DFU to its per-month champion source_model_id.

    Champion routing is (item_id, customer_group, loc, forecast_month)-grained.
    This returns
    ``{(item_id, customer_group, loc): {forecast_month: source_model_id}}`` so the generation
    loop can stage each month under its TRUE champion model rather than
    collapsing a DFU to a single model across the whole horizon.

    ``startdate`` (the champion row's month label) is normalized to the first
    of the month, matching ``forecast_month`` on the produced rows. Rows whose
    ``source_model_id`` is NULL are skipped — the caller applies its config
    fallback for those months.
    """
    routing: dict[tuple[str, str, str], dict[pd.Timestamp, str]] = defaultdict(dict)
    if champion_df.empty or "startdate" not in champion_df.columns:
        return routing
    for row in champion_df.itertuples(index=False):
        src = getattr(row, "source_model_id", None)
        if src == "ensemble":
            mix = _parse_ensemble_mix(getattr(row, "source_mix", None))
            if mix is None or len(mix) != 1:
                continue
            src = mix[0]["model"]
        if src is None or (isinstance(src, float) and pd.isna(src)):
            continue
        month = pd.Timestamp(row.startdate).normalize().replace(day=1)
        routing[(row.item_id, row.customer_group, row.loc)][month] = src
    return routing


def _resolve_champion_route(
    dfu: tuple[str, str, str],
    month: pd.Timestamp,
    month_routing: dict[tuple[str, str, str], dict[pd.Timestamp, str]],
    ensemble_routing: dict[tuple[str, str, str], dict[pd.Timestamp, list[dict[str, Any]]]],
) -> tuple[str, Any] | None:
    """Resolve the latest route known on or before one production month.

    Future-only routing evidence is never used.  ``None`` lets the caller use
    the configured LightGBM fallback without leaking a future winner.
    """
    candidates: list[tuple[pd.Timestamp, str, Any]] = []
    for route_month, model_id in month_routing.get(dfu, {}).items():
        candidates.append((pd.Timestamp(route_month), "model", model_id))
    for route_month, mix in ensemble_routing.get(dfu, {}).items():
        candidates.append((pd.Timestamp(route_month), "ensemble", mix))
    if not candidates:
        return None
    normalized_month = pd.Timestamp(month).to_period("M").to_timestamp()
    historical = [candidate for candidate in candidates if candidate[0] <= normalized_month]
    if not historical:
        return None
    selected = max(historical, key=lambda candidate: candidate[0])
    return selected[1], selected[2]


def collapse_to_dfu(champion_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-month champion frame to one row per full DFU.

    The generation loops build a per-DFU inference grid (cluster_id and
    customer_group are DFU-level, not month-level), so they iterate once per
    DFU. Per-month routing is carried separately via :func:`build_month_routing`
    and applied as a post-generation filter. The retained ``source_model_id``
    is the deterministic earliest-month champion (rows arrive ordered by
    startdate ASC from :func:`get_champion_assignments`); months that disagree
    are regenerated under their own model and filtered in.
    """
    if champion_df.empty:
        return champion_df
    deduped = champion_df.drop_duplicates(subset=["item_id", "customer_group", "loc"], keep="first")
    return deduped.reset_index(drop=True)


def filter_rows_to_champion_months(
    rows: list[dict],
    month_routing: dict[tuple[str, str, str], dict[pd.Timestamp, str]],
    ensemble_routing: dict[tuple[str, str, str], dict[pd.Timestamp, list[dict[str, Any]]]]
    | None = None,
) -> list[dict]:
    """Keep only generated rows whose model_id wins that (DFU, forecast_month).

    A DFU is generated under EACH distinct model it wins across the horizon;
    this drops the months where a given model is NOT the champion, so the
    surviving rows are per-month correct (recursive coherence is preserved —
    each kept month came from a full-horizon recursion of its own model).

    A DFU with no per-month routing entry (cold-start, --model-id override, or
    a fully NULL-source champion) is passed through unchanged: those
    intentionally use a single model across all months.

    Production months normally follow the last winner observed on or before the
    month. This as-of rule carries the newest validated route into the forward
    horizon without choosing a model lexically or looking into future rows.
    """
    ensemble_routing = ensemble_routing or {}
    if not month_routing and not ensemble_routing:
        return rows
    kept: list[dict] = []
    ensemble_rows: dict[tuple[str, str, str, pd.Timestamp], list[dict]] = defaultdict(list)
    ensemble_mixes: dict[tuple[str, str, str, pd.Timestamp], list[dict[str, Any]]] = {}
    for row in rows:
        dfu = (row["item_id"], row["customer_group"], row["loc"])
        month = pd.Timestamp(row["forecast_month"]).normalize().replace(day=1)
        resolved = _resolve_champion_route(
            dfu,
            month,
            month_routing,
            ensemble_routing,
        )
        if resolved is None:
            kept.append(row)
            continue
        route_type, route_value = resolved
        if route_type == "ensemble":
            mix = route_value
            if row["model_id"] in {entry.get("model") for entry in mix}:
                key = (dfu[0], dfu[1], dfu[2], month)
                ensemble_rows[key].append(row)
                ensemble_mixes[key] = mix
            continue
        winner = route_value
        # champion; the other enqueued models lost the month.
        if winner == row["model_id"]:
            kept.append(row)
    for (item_id, customer_group, loc, month), member_rows in ensemble_rows.items():
        mix = ensemble_mixes[(item_id, customer_group, loc, month)]
        by_model = {row["model_id"]: row for row in member_rows}
        if any(entry.get("model") not in by_model for entry in mix):
            continue
        blended = dict(member_rows[0])
        blended["model_id"] = "ensemble"
        for field in ("forecast_qty", "forecast_qty_lower", "forecast_qty_upper"):
            values = [by_model[entry["model"]].get(field) for entry in mix]
            blended[field] = (
                sum(
                    float(value) * float(entry.get("weight", 0))
                    for value, entry in zip(values, mix, strict=True)
                )
                if all(value is not None for value in values)
                else None
            )
        kept.append(blended)
    return kept


def aggregate_customer_group_forecasts(
    rows: list[dict],
) -> list[dict]:
    """Sum full-grain DFU forecasts to the production item/location grain.

    Champion routing and model inputs stay at
    ``(item_id, customer_group, loc)``. The production staging table is
    intentionally ``(item_id, loc, forecast_month)``; aggregation therefore
    happens only after each customer group has been forecast under its own
    winner. Mixed group winners are represented as ``ensemble`` because no
    single source algorithm owns the aggregate; the requested release model is
    tracked separately as ``candidate_model_id`` in staging.
    """
    if not rows:
        return []

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    seen_group_months: set[tuple] = set()
    for row in rows:
        customer_group = row.get("customer_group")
        if customer_group is None:
            raise ValueError("customer_group is required before production aggregation")
        source_key = (
            row["item_id"],
            customer_group,
            row["loc"],
            row["forecast_month"],
        )
        if source_key in seen_group_months:
            raise ValueError("multiple forecasts survived for one customer-group month")
        seen_group_months.add(source_key)
        aggregate_key = (
            row["forecast_month_generated"],
            row["item_id"],
            row["loc"],
            row["forecast_month"],
            row["horizon_months"],
            row["run_id"],
        )
        grouped[aggregate_key].append(row)

    aggregated: list[dict] = []
    for member_rows in grouped.values():
        result = {key: value for key, value in member_rows[0].items() if key != "customer_group"}
        result["forecast_qty"] = round(sum(float(row["forecast_qty"]) for row in member_rows), 2)
        result["forecast_qty_lower"] = None
        result["forecast_qty_upper"] = None
        cluster_ids = [row.get("cluster_id") for row in member_rows]
        result["cluster_id"] = (
            cluster_ids[0]
            if cluster_ids[0] is not None
            and all(cluster_id == cluster_ids[0] for cluster_id in cluster_ids)
            else None
        )
        source_model_ids = {str(row["model_id"]) for row in member_rows}
        sole_source = next(iter(source_model_ids)) if len(source_model_ids) == 1 else None
        result["model_id"] = sole_source if sole_source is not None else "ensemble"
        recursive_values = {bool(row["is_recursive"]) for row in member_rows}
        lag_sources = {str(row["lag_source"]) for row in member_rows}
        result["is_recursive"] = any(recursive_values)
        result["lag_source"] = next(iter(lag_sources)) if len(lag_sources) == 1 else "mixed"
        result["generated_at"] = max(row["generated_at"] for row in member_rows)
        aggregated.append(result)

    return sorted(
        aggregated,
        key=lambda row: (str(row["item_id"]), str(row["loc"]), row["forecast_month"]),
    )


def validate_customer_group_forecast_coverage(
    rows: list[dict],
    champion_df: pd.DataFrame,
    *,
    horizon: int,
) -> None:
    """Fail before aggregation if any routed customer group is incomplete."""
    expected_dfus = {
        (str(row.item_id), str(row.customer_group), str(row.loc))
        for row in collapse_to_dfu(champion_df).itertuples(index=False)
    }
    actual_months: dict[tuple[str, str, str], set] = defaultdict(set)
    actual_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        key = (
            str(row["item_id"]),
            str(row["customer_group"]),
            str(row["loc"]),
        )
        actual_months[key].add(row["forecast_month"])
        actual_counts[key] += 1

    incomplete = sorted(
        key
        for key in expected_dfus
        if len(actual_months.get(key, set())) != horizon or actual_counts.get(key, 0) != horizon
    )
    if incomplete:
        sample = ", ".join("/".join(key) for key in incomplete[:5])
        raise RuntimeError(
            f"Forecast generation is incomplete for {len(incomplete)} customer-group "
            f"DFU(s); expected exactly {horizon} month(s) each. Sample: {sample}"
        )


def validate_customer_group_history(
    sales_index: dict,
    champion_df: pd.DataFrame,
    *,
    minimum_months: int,
) -> None:
    """Fail before model execution when any target group lacks viable history."""
    insufficient: list[tuple[str, str, str, int]] = []
    for row in collapse_to_dfu(champion_df).itertuples(index=False):
        key = (str(row.item_id), str(row.customer_group), str(row.loc))
        history = sales_index.get(key)
        months = int(history[2]) if history is not None else 0
        if months < minimum_months:
            insufficient.append((*key, months))
    if insufficient:
        sample = ", ".join(
            f"{item}/{group}/{loc} ({months}m)" for item, group, loc, months in insufficient[:5]
        )
        raise RuntimeError(
            f"Insufficient production history for {len(insufficient)} customer-group "
            f"DFU(s); minimum is {minimum_months} months. Exclude or seed the full "
            f"item/location cohort before release generation. Sample: {sample}"
        )


def validate_route_history_requirements(
    sales_index: dict,
    champion_df: pd.DataFrame,
    *,
    production_months: list[pd.Timestamp],
    month_routing: dict[tuple[str, str, str], dict[pd.Timestamp, str]],
    ensemble_routing: dict[tuple[str, str, str], dict[pd.Timestamp, list[dict[str, Any]]]],
    min_history_months: int,
    mstl_min_history: int,
) -> None:
    """Reject a stale champion route that its canonical adapter cannot serve."""
    invalid_mstl: list[tuple[str, str, str, int]] = []
    for row in collapse_to_dfu(champion_df).itertuples(index=False):
        key = (str(row.item_id), str(row.customer_group), str(row.loc))
        history = sales_index.get(key)
        history_months = int(history[2]) if history is not None else 0
        if history_months < min_history_months:
            continue
        resolved_models: set[str] = set()
        for month in production_months:
            resolved = _resolve_champion_route(
                key,
                month,
                month_routing,
                ensemble_routing,
            )
            if resolved is None:
                continue
            route_type, route_value = resolved
            if route_type == "model":
                resolved_models.add(str(route_value))
            else:
                resolved_models.update(
                    str(entry["model"]) for entry in route_value if entry.get("model")
                )
        if "mstl" in resolved_models and history_months < mstl_min_history:
            invalid_mstl.append((*key, history_months))
    if invalid_mstl:
        sample = ", ".join(
            f"{item}/{group}/{loc} ({months}m)" for item, group, loc, months in invalid_mstl[:5]
        )
        raise RuntimeError(
            f"MSTL requires {mstl_min_history} months, but the promoted champion "
            f"routes {len(invalid_mstl)} shorter DFU(s). Re-run the MSTL backtest "
            f"and champion selection with the current adapter. Sample: {sample}"
        )


def attach_aggregate_confidence_intervals(
    rows: list[dict],
    *,
    sigma_lookup: dict,
    ci_cfg: dict,
) -> list[dict]:
    """Attach one uncertainty interval to each item/location aggregate."""
    enriched: list[dict] = []
    for row in rows:
        result = dict(row)
        sigma = sigma_lookup.get((row["item_id"], row["loc"]))
        if sigma is not None:
            lower, upper = compute_ci_bounds(
                point_forecast=float(row["forecast_qty"]),
                sigma=sigma,
                horizon=int(row["horizon_months"]),
                z_lower=ci_cfg["z_lower"],
                z_upper=ci_cfg["z_upper"],
                scaling=ci_cfg["horizon_scaling"],
            )
            result["forecast_qty_lower"] = lower
            result["forecast_qty_upper"] = upper
        enriched.append(result)
    return enriched


# ---------------------------------------------------------------------------
# Historical sales loading
# ---------------------------------------------------------------------------


def load_recent_sales(
    conn,
    target_dfus: pd.DataFrame,
    *,
    lookback_months: int,
    target_batch_size: int = 10_000,
    sales_table: str | None = None,
) -> pd.DataFrame:
    """Load observed history at the training DFU grain in target batches.

    Target serialization and each database result are bounded by
    ``target_batch_size``. Calendar completion happens once in
    :func:`build_sales_index`; expanding every target to every month in SQL
    would multiply sparse histories and exhaust memory before inference.

    Returns columns: sku_ck, item_id, customer_group, loc, startdate, qty,
    first_sale_month.
    """
    if lookback_months <= 0:
        raise ValueError("lookback_months must be positive")
    if target_batch_size <= 0:
        raise ValueError("target_batch_size must be positive")
    target_columns = ("sku_ck", "item_id", "customer_group", "loc")
    missing_targets = set(target_columns) - set(target_dfus.columns)
    if missing_targets:
        raise ValueError(f"Target DFUs are missing columns: {sorted(missing_targets)}")
    targets = target_dfus[list(target_columns)].drop_duplicates().copy()
    if targets.empty:
        raise ValueError("At least one target DFU is required")
    for column in target_columns:
        invalid = targets[column].isna() | (targets[column].astype(str).str.strip() == "")
        if invalid.any():
            raise ValueError(f"Target DFUs contain a blank {column}")
        targets[column] = targets[column].astype(str)
    if targets["sku_ck"].duplicated().any():
        raise ValueError("Target DFUs contain duplicate sku_ck mappings")
    if targets.duplicated(["item_id", "customer_group", "loc"]).any():
        raise ValueError("Target DFUs contain duplicate item/customer-group/location mappings")

    planning_month = pd.Timestamp(get_planning_date()).normalize().replace(day=1)
    history_end = planning_month - pd.DateOffset(months=1)
    history_start = history_end - pd.DateOffset(months=lookback_months - 1)

    with conn.cursor() as cur:
        if sales_table is None:
            sales_table = resolve_forecast_sales_table(cur)

        cur.execute(
            psycopg.sql.SQL(
                """SELECT MAX(startdate)
                   FROM {}
                   WHERE type = 1
                     AND qty IS NOT NULL
                     AND startdate <= %s"""
            ).format(psycopg.sql.Identifier(sales_table)),
            (history_end.date(),),
        )
        latest_source_row = cur.fetchone()
        latest_source_month = latest_source_row[0] if latest_source_row else None
    normalized_source_month = (
        pd.Timestamp(latest_source_month).normalize().replace(day=1)
        if latest_source_month is not None
        else None
    )
    if normalized_source_month != history_end:
        available = (
            normalized_source_month.strftime("%Y-%m")
            if normalized_source_month is not None
            else "none"
        )
        raise RuntimeError(
            "Sales history is not ready for the latest closed month "
            f"{history_end:%Y-%m}; {sales_table} is current through {available}. "
            "Complete the sales load before production forecast generation."
        )

    # Preserve the exact training DFU grain. Aggregating customer groups here
    # makes both lag features and static demand profiles differ from training.
    table_identifier = psycopg.sql.Identifier(sales_table).as_string(conn)
    sql = f"""
        WITH targets AS (
            SELECT *
            FROM jsonb_to_recordset(%s::jsonb) AS target(
                sku_ck TEXT,
                item_id TEXT,
                customer_group TEXT,
                loc TEXT
            )
        ),
        first_sales AS (
            SELECT target.sku_ck, MIN(sales.startdate) AS first_sale_month
            FROM targets target
            LEFT JOIN {table_identifier} sales
                   ON sales.item_id = target.item_id
                  AND sales.customer_group = target.customer_group
                  AND sales.loc = target.loc
                  AND sales.type = 1
                  AND sales.qty IS NOT NULL
                  AND sales.startdate <= %s
            GROUP BY target.sku_ck
        ),
        window_sales AS (
            SELECT target.sku_ck, sales.startdate,
                   SUM(COALESCE(sales.qty, 0)) AS qty
            FROM targets target
            JOIN {table_identifier} sales
              ON sales.item_id = target.item_id
             AND sales.customer_group = target.customer_group
             AND sales.loc = target.loc
             AND sales.type = 1
             AND sales.qty IS NOT NULL
             AND sales.startdate >= %s
             AND sales.startdate <= %s
            GROUP BY target.sku_ck, sales.startdate
        )
        SELECT target.sku_ck, target.item_id, target.customer_group, target.loc,
               sales.startdate, sales.qty,
               first_sales.first_sale_month
        FROM targets target
        LEFT JOIN window_sales sales
               ON sales.sku_ck = target.sku_ck
        LEFT JOIN first_sales ON first_sales.sku_ck = target.sku_ck
        ORDER BY target.item_id, target.customer_group, target.loc, sales.startdate
    """

    frames: list[pd.DataFrame] = []
    for offset in range(0, len(targets), target_batch_size):
        target_batch = targets.iloc[offset : offset + target_batch_size]
        params = [
            json.dumps(target_batch.to_dict("records")),
            history_end.date(),
            history_start.date(),
            history_end.date(),
        ]
        frames.append(read_sql_chunked(conn, sql, params=params))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df["startdate"] = pd.to_datetime(df["startdate"])
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0)
    df.attrs["history_start"] = history_start
    df.attrs["history_end"] = history_end
    logger.info(
        "Recent sales loaded: %s rows, %s DFUs",
        f"{len(df):,}",
        f"{df.groupby(['item_id', 'customer_group', 'loc']).ngroups:,}",
    )
    return df


# ---------------------------------------------------------------------------
# DFU attributes loading
# ---------------------------------------------------------------------------


def load_dfu_attrs(conn, item_id: str | None = None, loc: str | None = None) -> pd.DataFrame:
    """Load DFU attributes needed for feature construction."""
    where_clauses = []
    params: list = []

    if item_id:
        where_clauses.append("d.item_id = %s")
        params.append(item_id)
    if loc:
        where_clauses.append("d.loc = %s")
        params.append(loc)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sql = f"""
        SELECT d.sku_ck, d.item_id AS item_id, d.customer_group, d.loc,
               ca.ml_cluster,
               d.execution_lag, d.total_lt, d.brand, d.region, d.abc_vol
        FROM dim_sku d
        LEFT JOIN current_sku_cluster_assignment ca
               ON ca.sku_ck = d.sku_ck
        {where_sql}
    """

    # dim_sku is dim-shaped but grows to ~10M rows at 40x; defensive stream.
    df = read_sql_chunked(conn, sql, params=params)
    logger.info("DFU attributes loaded: %s rows", f"{len(df):,}")
    return df


def load_item_attrs(conn, item_id: str | None = None) -> pd.DataFrame:
    """Load item-level attributes (bpc, item_proof, case_weight) from dim_item."""
    where_sql = "WHERE item_id = %s" if item_id else ""
    params = [item_id] if item_id else []
    sql = f"""
        SELECT item_id, bpc, item_proof, case_weight
        FROM dim_item
        {where_sql}
    """
    # dim_item is dim-shaped; streaming kept for consistency.
    df = read_sql_chunked(conn, sql, params=params)
    df["bpc"] = pd.to_numeric(df["bpc"], errors="coerce").fillna(0)
    df["item_proof"] = pd.to_numeric(df["item_proof"], errors="coerce").fillna(0)
    df["case_weight"] = pd.to_numeric(df["case_weight"], errors="coerce").fillna(0)
    logger.info("Item attributes loaded: %s rows", f"{len(df):,}")
    return df


def build_item_index(item_attrs: pd.DataFrame) -> dict[str, dict]:
    """Pre-index item attributes by item_id → {bpc, item_proof, case_weight}."""
    return item_attrs.set_index("item_id")[["bpc", "item_proof", "case_weight"]].to_dict("index")


# ---------------------------------------------------------------------------
# Pre-indexing helpers — O(1) lookup replacements for per-DFU DataFrame scans
# ---------------------------------------------------------------------------


def build_sales_index(sales_df: pd.DataFrame) -> dict[tuple, tuple]:
    """Pre-index calendar-complete sales histories by training DFU grain.

    Replaces O(N) pandas boolean-filter scan in build_inference_grid with an
    O(1) dict lookup. Values are ``(dates, qty, active_length)``. ``qty`` is
    padded to the shared training calendar for all fitted feature semantics;
    ``active_length`` measures post-introduction tenure for eligibility only.
    """
    index: dict[tuple, tuple] = {}
    if sales_df.empty:
        return index
    required_columns = {
        "item_id",
        "customer_group",
        "loc",
        "startdate",
        "qty",
        "first_sale_month",
    }
    missing_columns = required_columns - set(sales_df.columns)
    if missing_columns:
        raise ValueError(f"Sales history is missing columns: {sorted(missing_columns)}")

    group_columns = ["item_id", "customer_group", "loc"]
    history_start = (
        pd.Timestamp(sales_df.attrs.get("history_start", sales_df["startdate"].min()))
        .normalize()
        .replace(day=1)
    )
    history_end = (
        pd.Timestamp(sales_df.attrs.get("history_end", sales_df["startdate"].max()))
        .normalize()
        .replace(day=1)
    )
    calendar = pd.date_range(history_start, history_end, freq="MS")

    grouped = sales_df.sort_values("startdate").groupby(group_columns, sort=False)
    for raw_key, grp in grouped:
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        observations = grp.dropna(subset=["startdate"])
        monthly = observations.groupby("startdate", sort=True)["qty"].sum()
        monthly.index = pd.DatetimeIndex(monthly.index).normalize()
        completed = monthly.reindex(calendar, fill_value=0.0).astype(float)
        first_sale = pd.to_datetime(grp["first_sale_month"], errors="coerce").min()
        if pd.isna(first_sale):
            active_length = 0
        else:
            active_start = max(
                history_start,
                pd.Timestamp(first_sale).normalize().replace(day=1),
            )
            active_length = len(pd.date_range(active_start, history_end, freq="MS"))
        index[key] = (list(calendar.values), list(completed.values), active_length)
    return index


def build_attrs_index(dfu_attrs: pd.DataFrame) -> dict[tuple, dict]:
    """Pre-index DFU attributes by their full training grain.

    Replaces O(N) pandas boolean-filter scan with O(1) dict lookup.
    """
    records = dfu_attrs.set_index(["item_id", "customer_group", "loc"]).to_dict("index")
    return records


def build_item_location_cluster_map(dfu_attrs: pd.DataFrame) -> dict[tuple, str]:
    """Map an item/location to one unambiguous customer-group cluster label."""
    result: dict[tuple, str] = {}
    for key, group in dfu_attrs.groupby(["item_id", "loc"], sort=False):
        normalized = [_to_cluster_id(value) for value in group["ml_cluster"]]
        unique = set(normalized)
        result[key] = (
            str(normalized[0]) if len(unique) == 1 and normalized[0] is not None else "unknown"
        )
    return result


def _lookup_dfu(index: dict, item_id: str, customer_group: str | None, loc: str):
    """Look up a DFU using the training grain."""
    if customer_group is None:
        raise ValueError("customer_group is required for production DFU lookup")
    return index.get((item_id, customer_group, loc))


# ---------------------------------------------------------------------------
# Inference grid construction
# ---------------------------------------------------------------------------


def build_inference_grid(
    item_id: str,
    loc: str,
    cluster_id: int | str,
    sales_history: pd.DataFrame | None = None,
    dfu_attrs: pd.DataFrame | None = None,
    horizon: int = 24,
    min_months: int = 3,
    *,
    sales_index: dict | None = None,
    attrs_index: dict | None = None,
    item_index: dict | None = None,
    customer_group: str | None = None,
) -> pd.DataFrame | None:
    """Build a feature matrix for recursive inference over the next `horizon` months.

    For T+1: lag_1 = last known actual
    For T+2: lag_1 = model's T+1 prediction (written back after each step)
    For T+N: lag_1 = model's T+(N-1) prediction

    Returns DataFrame with `horizon` rows ready for model.predict(), or None if
    insufficient history.
    """
    if sales_index is not None:
        # Fast O(1) path — used by production main loop
        entry = _lookup_dfu(sales_index, item_id, customer_group, loc)
        if entry is None:
            return None
        dates_arr, profile_qty_arr, active_length = entry
        if active_length < min_months:
            return None
        # Training computes lags, rolling statistics, and Croston features on
        # the full shared calendar, including pre-introduction zeros. Tenure is
        # an eligibility signal only; slicing here changes the fitted feature
        # semantics for every young DFU.
        qty_series = list(profile_qty_arr)
        profile_qty_series = list(profile_qty_arr)
        last_month = pd.Timestamp(dates_arr[-1])
    else:
        # Legacy DataFrame path — used by tests
        sales_mask = (sales_history["item_id"] == item_id) & (sales_history["loc"] == loc)
        if customer_group is not None and "customer_group" in sales_history.columns:
            sales_mask &= sales_history["customer_group"] == customer_group
        dfu_sales = sales_history[sales_mask].sort_values("startdate").copy()
        if len(dfu_sales) < min_months:
            return None
        last_month = dfu_sales["startdate"].max()
        qty_series = list(dfu_sales["qty"].values)
        profile_qty_series = qty_series.copy()

    # Build a series of future months T+1 ... T+horizon
    future_months = [last_month + pd.DateOffset(months=h) for h in range(1, horizon + 1)]

    # Build rows for each future month
    rows = []

    # DFU attributes for categorical features
    if attrs_index is not None:
        attrs = _lookup_dfu(attrs_index, item_id, customer_group, loc) or {}
    else:
        attrs_mask = (dfu_attrs["item_id"] == item_id) & (dfu_attrs["loc"] == loc)
        if customer_group is not None and "customer_group" in dfu_attrs.columns:
            attrs_mask &= dfu_attrs["customer_group"] == customer_group
        dfu_row = dfu_attrs[attrs_mask]
        attrs = dfu_row.iloc[0].to_dict() if len(dfu_row) > 0 else {}

    # Item-level attributes (bpc, item_proof, case_weight from dim_item)
    item_attrs = (item_index or {}).get(item_id, {})

    # TS profile features (static per-DFU, computed from full history)
    _ts_profile = compute_ts_profile_from_values(profile_qty_series)

    def _numeric_or_zero(value: object) -> float:
        """Match training's pd.to_numeric(...).fillna(0) attribute contract."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return numeric if np.isfinite(numeric) else 0.0

    for h, fmonth in enumerate(future_months):
        # At step h=0 (T+1), qty_series contains all actuals
        # At step h=1+ (T+2+), qty_series[-1] is the model's T+h prediction
        n = len(qty_series)
        row: dict = {}

        # Lag features: lag_k = qty at (current position - k)
        for lag_n in LAG_RANGE:
            idx = n - lag_n
            row[f"qty_lag_{lag_n}"] = float(qty_series[idx]) if idx >= 0 else 0.0

        # Rolling features
        for w in ROLLING_WINDOWS:
            window_vals = qty_series[max(0, n - w) : n]
            if window_vals:
                row[f"rolling_mean_{w}m"] = float(np.mean(window_vals))
                row[f"rolling_std_{w}m"] = (
                    float(np.std(window_vals, ddof=1)) if len(window_vals) > 1 else 0.0
                )
            else:
                row[f"rolling_mean_{w}m"] = 0.0
                row[f"rolling_std_{w}m"] = 0.0

        # Calendar features
        month_num = fmonth.month
        row["month"] = month_num
        row["quarter"] = (month_num - 1) // 3 + 1
        # Fourier seasonal terms (replaces legacy month_sin/cos)
        for period in [12, 6, 4, 3]:
            angle = 2.0 * np.pi * month_num / period
            row[f"fourier_sin_{period}"] = float(np.sin(angle))
            row[f"fourier_cos_{period}"] = float(np.cos(angle))
        row["is_quarter_end"] = 1 if month_num in (3, 6, 9, 12) else 0
        row["is_year_end"] = 1 if month_num == 12 else 0
        row["days_in_month"] = float(
            fmonth.days_in_month
            if hasattr(fmonth, "days_in_month")
            else pd.Timestamp(fmonth).days_in_month
        )

        # DFU attributes (categorical and numeric)
        for col in CAT_FEATURES:
            row[col] = attrs.get(col, "__unknown__")
        row["execution_lag"] = _numeric_or_zero(attrs.get("execution_lag"))
        row["total_lt"] = _numeric_or_zero(attrs.get("total_lt"))

        # Item-level attributes (from dim_item)
        row["bpc"] = _numeric_or_zero(item_attrs.get("bpc"))
        row["item_proof"] = _numeric_or_zero(item_attrs.get("item_proof"))
        row["case_weight"] = _numeric_or_zero(item_attrs.get("case_weight"))

        # Derived demand features (same as feature_engineering._recompute_derived_features)
        lag1 = row.get("qty_lag_1", 0.0)
        lag2 = row.get("qty_lag_2", 0.0)
        lag12 = row.get("qty_lag_12", 0.0)
        row["mom_growth"] = max(-2.0, min(2.0, (lag1 - lag2) / (abs(lag2) + 1.0)))
        rm3 = row.get("rolling_mean_3m", 0.0)
        rm6 = row.get("rolling_mean_6m", 0.0)
        rm12 = row.get("rolling_mean_12m", 0.0)
        row["demand_accel"] = rm3 - rm6
        rs3 = row.get("rolling_std_3m", 0.0)
        row["volatility_ratio"] = rs3 / (abs(rm3) + 1.0)

        # Lag ratio features
        row["lag_ratio_yoy"] = max(-10.0, min(10.0, lag1 / (abs(lag12) + 1.0)))
        row["lag_ratio_mom"] = max(-10.0, min(10.0, lag1 / (abs(lag2) + 1.0)))
        row["lag_ratio_3v12"] = max(-10.0, min(10.0, rm3 / (abs(rm12) + 1.0)))

        # Zero-demand count (last 6 lags)
        row["n_zero_last_6m"] = sum(1.0 for i in range(1, 7) if row.get(f"qty_lag_{i}", 0.0) == 0.0)

        # TS profile features (static per-DFU)
        row.update(_ts_profile)

        row["_forecast_month"] = fmonth
        row["_horizon"] = h + 1
        row["_lag_source"] = "actual" if h == 0 else "predicted"
        row["_history_length"] = min(len(qty_series), max(LAG_RANGE))
        rows.append(row)

        # Placeholder — will be filled during recursive inference
        qty_series.append(0.0)

    if not rows:
        return None

    grid_df = pd.DataFrame(rows)

    return grid_df


# ---------------------------------------------------------------------------
# Recursive inference
# ---------------------------------------------------------------------------


def generate_forecast_recursive(
    model,
    feature_cols: list[str],
    grid: pd.DataFrame,
    horizon: int,
    item_id: str,
    customer_group: str,
    loc: str,
    forecast_month_generated,
    run_id: str,
    model_id: str,
    cluster_id: int | str,
) -> list[dict]:
    """Run recursive inference for one DFU over the horizon.

    Predicts month-by-month, writing each prediction back into the grid as
    lag_1 for the next month (recursive write-back).

    Returns list of row dicts ready for DB insert.
    """
    rows = []
    predicted_values: list[float] = []

    # Identify feature columns available in grid (exclude metadata cols)
    meta_cols = {"_forecast_month", "_horizon", "_lag_source"}
    available_features = [c for c in feature_cols if c in grid.columns and c not in meta_cols]

    for h in range(horizon):
        row_data = grid.iloc[[h]].copy()

        # Write-back: update lag_1 with previous prediction for h >= 1
        if h > 0 and predicted_values:
            row_data["qty_lag_1"] = predicted_values[-1]
            # Update lag_2, lag_3 etc. from prior predictions
            for k in range(2, min(h + 2, max(LAG_RANGE) + 1)):
                lag_col = f"qty_lag_{k}"
                prev_idx = h - k
                if prev_idx >= 0 and lag_col in row_data.columns:
                    row_data[lag_col] = predicted_values[prev_idx]
            # Recompute rolling features from updated lag values
            for w in ROLLING_WINDOWS:
                lag_vals = []
                for k in range(1, w + 1):
                    lag_col = f"qty_lag_{k}"
                    if lag_col in row_data.columns:
                        lag_vals.append(float(row_data[lag_col].iloc[0]))
                if lag_vals:
                    row_data[f"rolling_mean_{w}m"] = np.mean(lag_vals)
                    row_data[f"rolling_std_{w}m"] = (
                        np.std(lag_vals, ddof=1) if len(lag_vals) > 1 else 0.0
                    )
            # Recompute derived features after lag/rolling update
            l1 = float(row_data["qty_lag_1"].iloc[0]) if "qty_lag_1" in row_data.columns else 0.0
            l2 = float(row_data["qty_lag_2"].iloc[0]) if "qty_lag_2" in row_data.columns else 0.0
            row_data["mom_growth"] = max(-2.0, min(2.0, (l1 - l2) / (abs(l2) + 1.0)))
            rm3 = (
                float(row_data["rolling_mean_3m"].iloc[0])
                if "rolling_mean_3m" in row_data.columns
                else 0.0
            )
            rm6 = (
                float(row_data["rolling_mean_6m"].iloc[0])
                if "rolling_mean_6m" in row_data.columns
                else 0.0
            )
            row_data["demand_accel"] = rm3 - rm6
            rs3 = (
                float(row_data["rolling_std_3m"].iloc[0])
                if "rolling_std_3m" in row_data.columns
                else 0.0
            )
            row_data["volatility_ratio"] = rs3 / (abs(rm3) + 1.0)

        # Predict using only the features the model was trained on
        X = row_data[available_features].fillna(0)
        pred = float(model.predict(X)[0])
        pred = max(0.0, round(pred, 2))  # no negative forecasts

        predicted_values.append(pred)

        forecast_month = grid.iloc[h]["_forecast_month"]
        if hasattr(forecast_month, "date"):
            forecast_month_date = forecast_month.date().replace(day=1)
        else:
            forecast_month_date = forecast_month

        rows.append(
            {
                "forecast_month_generated": forecast_month_generated,
                "item_id": item_id,
                "customer_group": customer_group,
                "loc": loc,
                "forecast_month": forecast_month_date,
                "forecast_qty": pred,
                "forecast_qty_lower": None,
                "forecast_qty_upper": None,
                "model_id": model_id,
                "cluster_id": _to_cluster_id(cluster_id),
                "horizon_months": h + 1,
                "is_recursive": h > 0,
                "lag_source": "actual" if h == 0 else "predicted",
                "run_id": run_id,
                "generated_at": datetime.now(UTC),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Batched inference — cluster-level batch predictions
# ---------------------------------------------------------------------------


def _assign_batch_feature(
    feature_matrix: np.ndarray,
    column_indices: dict[str, int],
    feature_name: str,
    values: np.ndarray,
) -> None:
    """Assign one recursive feature when the trained artifact selected it."""
    feature_idx = column_indices.get(feature_name)
    if feature_idx is not None:
        feature_matrix[:, feature_idx] = values


def _refresh_batch_recursive_features(
    feature_matrix: np.ndarray,
    column_indices: dict[str, int],
    lag_history: np.ndarray,
    valid_history_lengths: np.ndarray,
) -> None:
    """Project full recursive demand state into the artifact's selected features.

    ``lag_history`` is zero-padded to twelve columns for stable vectorization,
    while ``valid_history_lengths`` distinguishes padding from genuine zero
    demand for short-history DFUs.
    """
    for lag in LAG_RANGE:
        _assign_batch_feature(
            feature_matrix,
            column_indices,
            f"qty_lag_{lag}",
            lag_history[:, lag - 1],
        )

    rolling_means: dict[int, np.ndarray] = {}
    rolling_stds: dict[int, np.ndarray] = {}
    for window in ROLLING_WINDOWS:
        values = lag_history[:, :window]
        valid_counts = np.minimum(valid_history_lengths, window)
        valid_mask = np.arange(window)[None, :] < valid_counts[:, None]
        sums = np.where(valid_mask, values, 0.0).sum(axis=1)
        means = np.divide(
            sums,
            valid_counts,
            out=np.zeros(len(values), dtype=float),
            where=valid_counts > 0,
        )
        centered = np.where(valid_mask, values - means[:, None], 0.0)
        sample_variance = np.divide(
            (centered * centered).sum(axis=1),
            valid_counts - 1,
            out=np.zeros(len(values), dtype=float),
            where=valid_counts > 1,
        )
        rolling_means[window] = means
        rolling_stds[window] = np.sqrt(np.maximum(sample_variance, 0.0))
        _assign_batch_feature(
            feature_matrix,
            column_indices,
            f"rolling_mean_{window}m",
            rolling_means[window],
        )
        _assign_batch_feature(
            feature_matrix,
            column_indices,
            f"rolling_std_{window}m",
            rolling_stds[window],
        )

    lag1 = lag_history[:, 0]
    lag2 = lag_history[:, 1]
    lag12 = lag_history[:, 11]
    mean3 = rolling_means[3]
    mean6 = rolling_means[6]
    mean12 = rolling_means[12]
    std3 = rolling_stds[3]
    derived_values = {
        "mom_growth": np.clip((lag1 - lag2) / (np.abs(lag2) + 1.0), -2.0, 2.0),
        "demand_accel": mean3 - mean6,
        "volatility_ratio": std3 / (np.abs(mean3) + 1.0),
        "lag_ratio_yoy": np.clip(lag1 / (np.abs(lag12) + 1.0), -10.0, 10.0),
        "lag_ratio_mom": np.clip(lag1 / (np.abs(lag2) + 1.0), -10.0, 10.0),
        "lag_ratio_3v12": np.clip(mean3 / (np.abs(mean12) + 1.0), -10.0, 10.0),
        "n_zero_last_6m": (
            (lag_history[:, :6] == 0.0)
            & (np.arange(6)[None, :] < np.minimum(valid_history_lengths, 6)[:, None])
        ).sum(axis=1),
    }
    for feature_name, values in derived_values.items():
        _assign_batch_feature(feature_matrix, column_indices, feature_name, values)

    valid_lag_mask = (
        np.arange(lag_history.shape[1])[None, :]
        < np.minimum(valid_history_lengths, lag_history.shape[1])[:, None]
    )
    positive = (lag_history > 0.0) & valid_lag_mask
    positive_count = positive.sum(axis=1)
    valid_count = valid_lag_mask.sum(axis=1)
    row_count = len(feature_matrix)
    demand_size = np.divide(
        np.where(positive, lag_history, 0.0).sum(axis=1),
        positive_count,
        out=np.zeros(row_count, dtype=float),
        where=positive_count > 0,
    )
    demand_interval = np.divide(
        valid_count,
        positive_count,
        out=np.where(valid_count > 0, valid_count, lag_history.shape[1]).astype(float),
        where=positive_count > 0,
    )
    demand_interval = np.maximum(demand_interval, 1.0)
    values_by_feature = {
        "croston_demand_size": demand_size,
        "croston_demand_interval": demand_interval,
        "croston_probability": 1.0 / demand_interval,
    }
    for feature_name, values in values_by_feature.items():
        _assign_batch_feature(feature_matrix, column_indices, feature_name, values)


def _artifact_recursive_lag_smooth(artifact: Mapping[str, object]) -> float:
    """Return the immutable recursive smoothing factor carried by the tree fit."""
    contract = artifact.get("recursive_training")
    if contract is None:
        if artifact.get("config_checksum") is not None:
            raise ValueError(
                "Checksummed tree artifact is missing its recursive_training contract"
            )
        # Narrow compatibility for hand-built unit fixtures and non-production callers.
        return 0.0
    if not isinstance(contract, Mapping):
        raise ValueError("Tree artifact recursive_training contract must be a mapping")
    value = contract.get("lag_smooth")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Tree artifact recursive_training.lag_smooth must be numeric")
    lag_smooth = float(value)
    if not np.isfinite(lag_smooth) or not 0 <= lag_smooth <= 1:
        raise ValueError(
            "Tree artifact recursive_training.lag_smooth must be between 0 and 1"
        )
    return lag_smooth


def generate_forecasts_batch(
    artifact: dict,
    dfu_list: list[tuple],  # list of (champ_dict, grid)
    horizon: int,
    forecast_month_generated,
    run_id: str,
    model_id: str,
) -> list[dict]:
    """Vectorised batch inference for all DFUs in a single cluster group.

    Builds a single (n_dfus, n_features) numpy array upfront, then for each
    horizon step:
      1. Calls model.predict() once on the full matrix (already fast)
      2. Updates lag/rolling columns with numpy array operations (vectorised
         across all DFUs at once, no Python-level per-DFU loop)

    This is 5-20x faster than the old approach which did pd.concat() + a
    Python loop per DFU at every horizon step.
    """
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    recursive_lag_smooth = _artifact_recursive_lag_smooth(artifact)
    meta_cols = {"_forecast_month", "_horizon", "_lag_source", "_history_length"}
    available_features = [c for c in feature_cols if c not in meta_cols]

    # Filter valid DFUs (non-None grid)
    valid_pairs = [(i, champ, grid) for i, (champ, grid) in enumerate(dfu_list) if grid is not None]
    if not valid_pairs:
        return []

    valid_champs = [t[1] for t in valid_pairs]
    valid_grids = [t[2] for t in valid_pairs]
    if any(len(grid) < horizon for grid in valid_grids):
        raise ValueError("Inference grid is shorter than requested horizon")

    # -----------------------------------------------------------------------
    # Build initial feature matrix X_np[j, :] from row 0 of each DFU's grid
    # -----------------------------------------------------------------------
    init_frames = [g.iloc[[0]] for g in valid_grids]
    init_df = pd.concat(init_frames, ignore_index=True)
    recursive_feature_names = {
        *{f"qty_lag_{lag}" for lag in LAG_RANGE},
        *{f"rolling_mean_{window}m" for window in ROLLING_WINDOWS},
        *{f"rolling_std_{window}m" for window in ROLLING_WINDOWS},
        *DERIVED_FEATURES,
        *CROSTON_FEATURES,
    }
    requires_recursive_state = bool(
        recursive_feature_names.intersection(available_features)
    )
    full_lag_columns = [f"qty_lag_{lag}" for lag in LAG_RANGE]
    if requires_recursive_state:
        missing_lags = [
            column for column in full_lag_columns if column not in init_df.columns
        ]
        if missing_lags:
            raise RuntimeError(
                "Production inference grid is missing required lag features: "
                + ", ".join(missing_lags)
            )
        lag_history = (
            init_df[full_lag_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .to_numpy(dtype=float)
        )
    else:
        lag_history = np.zeros((len(init_df), len(full_lag_columns)), dtype=float)
    history_length_values = (
        init_df["_history_length"]
        if "_history_length" in init_df.columns
        else pd.Series(max(LAG_RANGE), index=init_df.index)
    )
    valid_history_lengths = (
        pd.to_numeric(history_length_values, errors="coerce").fillna(0).to_numpy(dtype=int)
    )
    valid_history_lengths = np.clip(valid_history_lengths, 0, max(LAG_RANGE))

    missing_static_features = [
        column
        for column in available_features
        if column not in init_df.columns and column not in recursive_feature_names
    ]
    if missing_static_features:
        raise RuntimeError(
            "Production inference grid cannot supply artifact feature(s): "
            + ", ".join(missing_static_features)
        )
    # Recursive features are immediately projected from lag_history below; the
    # placeholder allocates their matrix columns but is never served as input.
    for column in recursive_feature_names & set(available_features):
        if column not in init_df.columns:
            init_df[column] = 0.0
    avail = available_features

    X_df = init_df[avail].copy()

    raw_encoders = artifact.get("categorical_encoders")
    categorical_features = [column for column in avail if column in CAT_FEATURES]
    if categorical_features and not isinstance(raw_encoders, dict):
        raise RuntimeError(
            "Production tree artifact is missing categorical_encoders for "
            f"{', '.join(categorical_features)}"
        )
    artifact_encoders = raw_encoders or {}

    def _encode_categories(frame: pd.DataFrame) -> pd.DataFrame:
        encoded = frame.copy()
        for column in categorical_features:
            mapping = artifact_encoders.get(column)
            if not isinstance(mapping, dict):
                raise RuntimeError(
                    f"Production tree artifact is missing categorical_encoders for {column}"
                )
            values = encoded[column].fillna("__unknown__").astype(str)
            unknown = sorted(set(values) - set(mapping))
            if unknown:
                sample = ", ".join(unknown[:3])
                raise ValueError(
                    f"Production inference has unknown category values for {column}: {sample}"
                )
            encoded[column] = values.map(mapping).astype(int)
        return encoded

    X_df = _encode_categories(X_df).fillna(0)
    X_np = X_df.to_numpy(dtype=float)  # (n_valid, n_avail)
    col_idx = {c: i for i, c in enumerate(avail)}
    _refresh_batch_recursive_features(
        X_np,
        col_idx,
        lag_history,
        valid_history_lengths,
    )

    horizon_feature_names = [name for name in avail if name not in recursive_feature_names]

    # -----------------------------------------------------------------------
    # Forecast months for each valid DFU (from their grids)
    # -----------------------------------------------------------------------
    forecast_months: list[list] = []
    for g in valid_grids:
        fms = []
        for h in range(min(horizon, len(g))):
            fm = g.iloc[h]["_forecast_month"]
            fms.append(fm.date().replace(day=1) if hasattr(fm, "date") else fm)
        forecast_months.append(fms)

    # -----------------------------------------------------------------------
    # Recursive inference loop — one predict() call per horizon step
    # -----------------------------------------------------------------------
    all_preds = np.zeros((len(valid_pairs), horizon))

    for h in range(horizon):
        if h > 0 and horizon_feature_names:
            horizon_df = pd.concat(
                [grid.iloc[[h]] for grid in valid_grids],
                ignore_index=True,
            )
            missing_horizon_features = [
                column for column in horizon_feature_names if column not in horizon_df.columns
            ]
            if missing_horizon_features:
                raise RuntimeError(
                    "Production inference horizon cannot supply artifact feature(s): "
                    + ", ".join(missing_horizon_features)
                )
            horizon_df = _encode_categories(horizon_df)
            for column in horizon_feature_names:
                X_np[:, col_idx[column]] = (
                    pd.to_numeric(horizon_df[column], errors="coerce")
                    .fillna(0)
                    .to_numpy(dtype=float)
                )

        # Batch predict on current feature matrix
        try:
            # Preserve the feature names recorded by sklearn/LightGBM during
            # fitting. Passing the backing ndarray floods managed-job logs with
            # "X does not have valid feature names" for every horizon step.
            booster = getattr(model, "booster_", None)
            if booster is not None:
                # The matrix is already ordered and categoricals are encoded.
                # Native Booster inference avoids sklearn's pandas categorical
                # metadata check while retaining the trained feature order.
                raw_preds = booster.predict(X_np)
            else:
                prediction_frame = pd.DataFrame(X_np.copy(), columns=avail)
                raw_preds = model.predict(prediction_frame)
            step_preds = np.asarray(raw_preds, dtype=float)
            if step_preds.ndim != 1 or step_preds.shape[0] != len(valid_pairs):
                raise ValueError("Prediction output must contain exactly one value per DFU")
            if not np.isfinite(step_preds).all():
                raise ValueError("Prediction output contains non-finite values")
            step_preds = np.maximum(0.0, step_preds)
        except (ValueError, TypeError, ArithmeticError, LightGBMError):
            # Previously this substituted a full column of zeros, which silently
            # corrupts the forecast (an all-zero forecast reads downstream as
            # "no demand"). Fail loud instead so a systematic feature/dtype/model
            # bug surfaces rather than writing zeros to staging.
            logger.exception("Prediction failed for cluster group at horizon step %d", h)
            raise

        all_preds[:, h] = step_preds

        if h < horizon - 1:
            recursive_lag = step_preds
            # Match the evaluated backtest contract: step 2 and step 3 consume
            # raw prediction-derived lag-1. The prediction from step 3 onward is
            # blended with the target horizon row's prior lag-1 before write-back.
            if h >= 2 and recursive_lag_smooth > 0:
                prior_lag = pd.to_numeric(
                    pd.Series(
                        [grid.iloc[h + 1].get("qty_lag_1") for grid in valid_grids]
                    ),
                    errors="coerce",
                ).to_numpy(dtype=float)
                finite_prior = np.isfinite(prior_lag)
                smoothed = (
                    step_preds * (1 - recursive_lag_smooth)
                    + prior_lag * recursive_lag_smooth
                )
                recursive_lag = np.where(finite_prior, smoothed, step_preds)
            lag_history[:, 1:] = lag_history[:, :-1].copy()
            lag_history[:, 0] = recursive_lag
            valid_history_lengths = np.minimum(valid_history_lengths + 1, max(LAG_RANGE))
            _refresh_batch_recursive_features(
                X_np,
                col_idx,
                lag_history,
                valid_history_lengths,
            )

    # -----------------------------------------------------------------------
    # Build output row dicts
    # -----------------------------------------------------------------------
    ts_now = datetime.now(UTC)
    all_rows: list[dict] = []
    for j, champ in enumerate(valid_champs):
        item_id = champ["item_id"]
        customer_group = champ["customer_group"]
        loc = champ["loc"]
        cluster_id = _to_cluster_id(champ["cluster_id"])
        months = forecast_months[j]

        for h in range(min(len(months), horizon)):
            pred = round(float(all_preds[j, h]), 2)

            all_rows.append(
                {
                    "forecast_month_generated": forecast_month_generated,
                    "item_id": item_id,
                    "customer_group": customer_group,
                    "loc": loc,
                    "forecast_month": months[h],
                    "forecast_qty": pred,
                    "forecast_qty_lower": None,
                    "forecast_qty_upper": None,
                    "model_id": model_id,
                    "cluster_id": cluster_id,
                    "horizon_months": h + 1,
                    "is_recursive": h > 0,
                    "lag_source": "actual" if h == 0 else "predicted",
                    "run_id": run_id,
                    "generated_at": ts_now,
                }
            )

    return all_rows


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------


def _collect_generation_evidence(
    conn,
    *,
    candidate_model_id: str,
    generation_purpose: str,
) -> dict[str, object | None]:
    """Capture release-control evidence before an immutable run is persisted."""
    sales_lineage = load_completed_sales_lineage(conn)
    if generation_purpose == "snapshot_contender":
        try:
            with conn.cursor() as cur:
                governed_lineage = load_active_governed_champion_lineage(cur)
        except GovernedChampionLineageError as exc:
            raise ValueError(
                "snapshot contenders require a governed champion; run model-refresh"
            ) from exc
        promoted_clusters = load_promoted_cluster_population(conn)
        source_backtest_run_id = governed_lineage["backtest_run_ids"].get(
            candidate_model_id
        )
        if (
            governed_lineage["source_sales_batch_id"] != sales_lineage.batch_id
            or governed_lineage["data_checksum"] != sales_lineage.source_hash
            or governed_lineage["cluster_experiment_id"]
            != promoted_clusters.experiment_id
            or governed_lineage["cluster_assignment_count"]
            != promoted_clusters.assignment_count
            or governed_lineage["cluster_assignment_checksum"]
            != promoted_clusters.assignment_checksum
            or not isinstance(source_backtest_run_id, int)
            or source_backtest_run_id <= 0
        ):
            raise ValueError(
                "snapshot contender inputs differ from the governed model-refresh; "
                "run model-refresh before forecast-publish"
            )
        return {
            "champion_experiment_id": governed_lineage["experiment_id"],
            "cluster_experiment_id": governed_lineage["cluster_experiment_id"],
            "source_sales_batch_id": sales_lineage.batch_id,
            "routing_artifact_checksum": None,
            "champion_results_checksum": None,
            GOVERNED_CHAMPION_LINEAGE_METADATA_KEY: governed_lineage,
            "source_backtest_run_id": source_backtest_run_id,
        }

    with conn.cursor() as cur:
        cur.execute(
            """SELECT experiment_id
               FROM cluster_experiment
               WHERE is_promoted = TRUE
               ORDER BY promoted_at DESC, experiment_id DESC
               LIMIT 1"""
        )
        cluster_row = cur.fetchone()

        champion_experiment_id: int | None = None
        governed_champion_lineage: dict[str, Any] | None = None
        routing_checksum: str | None = None
        cluster_experiment_id = int(cluster_row[0]) if cluster_row else None
        if candidate_model_id == CHAMPION_MODEL_ID:
            cur.execute(
                """SELECT experiment_id, cluster_experiment_id,
                          COALESCE(is_results_promoted, FALSE),
                          results_artifact_checksum,
                          results_forecast_checksum
                   FROM champion_experiment
                   WHERE is_promoted = TRUE
                   ORDER BY promoted_at DESC, experiment_id DESC
                   LIMIT 1"""
            )
            champion_row = cur.fetchone()
            if champion_row is None:
                raise ValueError(
                    "a promoted champion experiment is required for champion generation"
                )
            champion_experiment_id = int(champion_row[0])
            champion_cluster_id = champion_row[1]
            if not bool(champion_row[2]):
                raise ValueError(
                    "promote the champion experiment results before release generation"
                )
            if champion_cluster_id is None or cluster_experiment_id is None:
                raise ValueError("champion and promoted cluster lineage are required")
            if int(champion_cluster_id) != cluster_experiment_id:
                raise ValueError("champion and promoted cluster experiment lineage do not match")
            stored_routing_checksum = str(champion_row[3]) if champion_row[3] else None
            champion_results_checksum = str(champion_row[4]) if champion_row[4] else None
            if stored_routing_checksum is None or champion_results_checksum is None:
                raise ValueError("promoted champion results are missing checksum evidence")
            winners_path = CHAMPION_WINNERS_DIR / f"experiment_{champion_experiment_id}_winners.csv"
            if not winners_path.exists():
                raise ValueError("the promoted champion routing artifact is missing")
            routing_checksum = sha256_file(winners_path)
            if routing_checksum != stored_routing_checksum:
                raise ValueError("promoted champion routing artifact checksum has changed")
            try:
                governed_champion_lineage = load_governed_champion_lineage(
                    cur,
                    experiment_id=champion_experiment_id,
                )
            except GovernedChampionLineageError as exc:
                raise ValueError(
                    "the promoted champion lacks governed source lineage; run model-refresh"
                ) from exc
        else:
            champion_results_checksum = None

    if governed_champion_lineage is not None:
        promoted_clusters = load_promoted_cluster_population(conn)
        if (
            governed_champion_lineage["source_sales_batch_id"] != sales_lineage.batch_id
            or governed_champion_lineage["data_checksum"] != sales_lineage.source_hash
            or governed_champion_lineage["cluster_experiment_id"]
            != promoted_clusters.experiment_id
            or governed_champion_lineage["cluster_assignment_count"]
            != promoted_clusters.assignment_count
            or governed_champion_lineage["cluster_assignment_checksum"]
            != promoted_clusters.assignment_checksum
        ):
            raise ValueError(
                "the promoted champion was selected on stale sales or clustering; "
                "run model-refresh before forecast-publish"
            )

    return {
        "champion_experiment_id": champion_experiment_id,
        "cluster_experiment_id": cluster_experiment_id,
        "source_sales_batch_id": sales_lineage.batch_id,
        "routing_artifact_checksum": routing_checksum,
        "champion_results_checksum": champion_results_checksum,
        GOVERNED_CHAMPION_LINEAGE_METADATA_KEY: governed_champion_lineage,
    }


def write_forecast_staging(
    rows: list[dict],
    conn,
    model_id: str,
    dry_run: bool = False,
    *,
    generation_purpose: str = "release_candidate",
    generation_evidence: dict[str, object | None] | None = None,
    generation_metadata: dict[str, Any] | None = None,
    promotion_eligible: bool = False,
    requested_horizon: int | None = None,
) -> int:
    """Persist one immutable generation run and its exact staging payload."""
    if dry_run:
        logger.info("[DRY RUN] Would insert %s rows for %s", f"{len(rows):,}", model_id)
        return len(rows)
    if not rows:
        raise ValueError("no forecast rows were generated for the requested run")
    if generation_purpose not in {"release_candidate", "snapshot_contender"}:
        raise ValueError("generation purpose must be release_candidate or snapshot_contender")

    run_ids = {str(row["run_id"]) for row in rows}
    record_months = {row["forecast_month_generated"] for row in rows}
    if len(run_ids) != 1 or len(record_months) != 1:
        raise ValueError("one staging write must contain exactly one run and record month")
    run_id = next(iter(run_ids))
    record_month = next(iter(record_months))
    horizon_months = requested_horizon or max(int(row["horizon_months"]) for row in rows)
    evidence = (
        generation_evidence
        if generation_evidence is not None
        else _collect_generation_evidence(
            conn,
            candidate_model_id=model_id,
            generation_purpose=generation_purpose,
        )
    )
    eligible = bool(promotion_eligible and generation_purpose == "release_candidate")

    manifest_metadata = build_generation_metadata(generation_metadata)
    governed_champion_lineage = evidence.get(
        GOVERNED_CHAMPION_LINEAGE_METADATA_KEY
    )
    if generation_purpose == "release_candidate" and model_id == CHAMPION_MODEL_ID:
        if not isinstance(governed_champion_lineage, dict):
            raise ValueError(
                "champion release generation requires governed model-refresh lineage"
            )
        manifest_metadata[GOVERNED_CHAMPION_LINEAGE_METADATA_KEY] = (
            governed_champion_lineage
        )
    if generation_purpose == "snapshot_contender":
        source_backtest_run_id = evidence.get("source_backtest_run_id")
        if (
            not isinstance(governed_champion_lineage, dict)
            or not isinstance(source_backtest_run_id, int)
            or source_backtest_run_id <= 0
        ):
            raise ValueError(
                "snapshot contender generation requires governed backtest lineage"
            )
        manifest_metadata[GOVERNED_CHAMPION_LINEAGE_METADATA_KEY] = (
            governed_champion_lineage
        )
        manifest_metadata["source_backtest_run_id"] = source_backtest_run_id
    evidence_params = (
        evidence.get("champion_experiment_id"),
        evidence.get("cluster_experiment_id"),
        evidence.get("source_sales_batch_id"),
        evidence.get("routing_artifact_checksum"),
        evidence.get("champion_results_checksum"),
        Jsonb(manifest_metadata),
        run_id,
    )
    staged_rows = [
        {
            **row,
            "generation_purpose": generation_purpose,
            "candidate_model_id": model_id,
        }
        for row in rows
    ]

    sql = """
        INSERT INTO fact_production_forecast_staging
            (model_id, candidate_model_id, generation_purpose,
             item_id, loc, forecast_month, forecast_month_generated,
             forecast_qty, forecast_qty_lower, forecast_qty_upper, cluster_id,
             horizon_months, is_recursive, lag_source, generated_at, run_id)
        VALUES
            (%(model_id)s, %(candidate_model_id)s, %(generation_purpose)s,
             %(item_id)s, %(loc)s, %(forecast_month)s, %(forecast_month_generated)s,
             %(forecast_qty)s, %(forecast_qty_lower)s, %(forecast_qty_upper)s, %(cluster_id)s,
             %(horizon_months)s, %(is_recursive)s, %(lag_source)s, %(generated_at)s, %(run_id)s)
        ON CONFLICT (run_id, generation_purpose, candidate_model_id,
                     item_id, loc, forecast_month)
        DO NOTHING
    """
    try:
        with conn.cursor() as cur:
            status = reserve_generation_run(
                cur,
                run_id=run_id,
                generation_purpose=generation_purpose,
                requested_model_id=model_id,
                record_month=record_month,
                horizon_months=horizon_months,
                created_by="forecast-generator",
                metadata=manifest_metadata,
                resume_invalid=True,
            )
            if status != "generating":
                raise ValueError(
                    f"forecast generation run is already {status} and cannot be overwritten"
                )
            cur.execute(
                """UPDATE forecast_generation_run
                   SET champion_experiment_id = %s,
                       cluster_experiment_id = %s,
                       source_sales_batch_id = %s,
                       routing_artifact_checksum = %s,
                       champion_results_checksum = %s,
                       metadata = %s
                   WHERE run_id = %s::uuid
                     AND run_status = 'generating'""",
                evidence_params,
            )
            if cur.rowcount != 1:
                raise ValueError("generation reservation could not accept current lineage")
            cur.executemany(sql, staged_rows)
            stats = compute_staging_payload_stats(cur, run_id)
            if stats.row_count != len(staged_rows):
                raise ValueError(
                    "staging row count does not match the immutable generation payload"
                )
            cur.execute(
                """UPDATE forecast_generation_run
                   SET run_status = 'ready',
                       promotion_eligible = %s,
                       row_count = %s,
                       dfu_count = %s,
                       candidate_model_count = %s,
                       artifact_checksum = %s,
                       completed_at = NOW()
                   WHERE run_id = %s::uuid
                     AND run_status = 'generating'""",
                (
                    eligible,
                    stats.row_count,
                    stats.dfu_count,
                    stats.source_model_count,
                    stats.checksum,
                    run_id,
                ),
            )
            if cur.rowcount != 1:
                raise ValueError("generation manifest did not transition to ready")
        conn.commit()
    except (psycopg.Error, ValueError):
        conn.rollback()
        raise
    return len(rows)


@contextmanager
def _invalidate_generation_reservation_on_failure(
    db: dict[str, Any],
    *,
    run_id: str,
    dry_run: bool,
):
    """Invalidate a reserved run after its write transaction has rolled back."""
    try:
        yield
    except BaseException as exc:
        if not dry_run:
            try:
                with psycopg.connect(**db) as failure_conn, failure_conn.cursor() as cur:
                    invalidate_generation_run(
                        cur,
                        run_id,
                        f"forecast generator failed with {type(exc).__name__}",
                    )
            except psycopg.Error:
                logger.exception(
                    "Could not invalidate failed forecast generation reservation %s",
                    run_id,
                )
        raise


# ---------------------------------------------------------------------------
def _missing_required_tree_model_ids(
    required_tree_model_ids: set[str],
    loaded_models: dict[str, dict],
) -> list[str]:
    """Return required tree model ids without loaded cluster artifacts."""
    return sorted(
        model_id for model_id in required_tree_model_ids if not loaded_models.get(model_id)
    )


def _cluster_assignments_wiped(
    champion_df: pd.DataFrame,
    loaded_models: dict[str, dict],
    non_tree_models: set[str],
) -> bool:
    """Detect the promoted-cluster wipe failure mode (fail loud, don't ship garbage).

    Per-cluster LightGBM artifacts partition on promoted SKU cluster assignments (read
    into ``champion_df.cluster_id``). A missing current assignment table/view makes
    ``cluster_id`` NULL for EVERY DFU. The tree path then silently routes every
    DFU to a single arbitrary cluster model (``_resolve_artifact``'s ``min()``
    fallback), so high-volume forecasts collapse to a tiny near-constant value
    while models that do not use the LightGBM cluster artifact
    stay correct.

    Returns True only for the catastrophic signature: a MULTI-cluster tree model
    is loaded (per-cluster partitioning genuinely in effect — not a single global
    model under ``clustering.enabled = false``) AND ``cluster_id`` is NULL for
    every champion DFU. Partial NULLs are left to the per-DFU
    ``cluster_fallback_count`` warning; this guards the all-NULL catastrophe.
    """
    if champion_df.empty or "cluster_id" not in champion_df.columns:
        return False
    multi_cluster_tree = any(
        mid not in non_tree_models and isinstance(cmodels, dict) and len(cmodels) > 1
        for mid, cmodels in loaded_models.items()
    )
    if not multi_cluster_tree:
        return False
    return bool(champion_df["cluster_id"].isna().all())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate production forecasts for future months (F1.1)"
    )
    parser.add_argument(
        "--horizon", type=int, default=None, help="Months ahead to forecast (default from config)"
    )
    parser.add_argument(
        "--dfu",
        nargs=2,
        metavar=("ITEM", "LOC"),
        help="Run for a single DFU only: --dfu 100320 1401-BULK",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to DB")
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override model_id (default: champion assignment per DFU)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Use this UUID for staging lineage (default: generate a new UUID)",
    )
    parser.add_argument(
        "--generation-purpose",
        choices=("release_candidate", "snapshot_contender"),
        default="release_candidate",
        help="Immutable run purpose; snapshot contenders are never promotable",
    )
    parser.add_argument(
        "--plan-version",
        type=str,
        default=None,
        help="Override plan version label (e.g. '2026-02'). Defaults to current month.",
    )
    parser.add_argument(
        "--max-dfus",
        type=int,
        default=None,
        help="Limit to first N DFUs (for testing/sampling). Default: all DFUs.",
    )
    parser.add_argument(
        "--confidence-intervals",
        dest="confidence_intervals",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force CI (P10/P90) bands on/off. Default: use config "
        "confidence_interval.enabled. Use --no-confidence-intervals to force off.",
    )
    args = parser.parse_args()

    if args.model_id is not None:
        try:
            _validate_forecastable_model_ids({args.model_id}, source="--model-id")
        except ValueError as exc:
            parser.error(str(exc))

    config = load_config()
    pipeline_cfg = config.get("_pipeline", {})
    horizon = args.horizon or config["inference"]["horizon_months"]
    fallback_model_id = config["model_selection"]["fallback_model_id"]
    lookback_months = pipeline_cfg["lookback_months"]
    min_history_months = pipeline_cfg.get("min_history_months", 12)
    cold_start_model_id = pipeline_cfg.get("cold_start_model_id", "lgbm_cluster")
    cold_start_min_months = pipeline_cfg.get("cold_start_min_months", 3)
    active_window_months = pipeline_cfg["active_window_months"]
    clustering_enabled = pipeline_cfg["clustering_enabled"]
    if not isinstance(clustering_enabled, bool):
        raise ValueError("clustering.enabled must be explicitly true or false")
    _validate_forecastable_model_ids(
        {fallback_model_id, cold_start_model_id},
        source="Production forecast configuration",
    )
    # Champion sources with no .pkl (statistical / foundation / DL) — generated from
    # history with their TRUE model_id instead of the lgbm fallback (F-11).
    non_tree_models = load_non_tree_model_ids()

    plan_version = args.plan_version or get_planning_date().strftime(
        config["plan_version"]["format"]
    )
    forecast_month_generated = get_planning_date().replace(day=1)
    if args.run_id:
        try:
            run_id = str(uuid.UUID(args.run_id))
        except ValueError as exc:
            raise SystemExit(f"--run-id must be a UUID: {args.run_id}") from exc
    else:
        run_id = str(uuid.uuid4())
    production_months = list(
        pd.date_range(
            pd.Timestamp(forecast_month_generated),
            periods=horizon,
            freq="MS",
        )
    )

    item_filter = args.dfu[0] if args.dfu else None
    loc_filter = args.dfu[1] if args.dfu else None

    logger.info("Production Forecast Generation -- F1.1")
    logger.info("plan_version=%s, horizon=%d, run_id=%s...", plan_version, horizon, run_id[:8])
    logger.info(
        "Cold-start routing: DFUs with < %d months → %s, < %d months → skip",
        min_history_months,
        cold_start_model_id,
        cold_start_min_months,
    )
    if args.dry_run:
        logger.info("DRY RUN -- no data will be written")

    t_start = time.time()
    db = get_db_params()

    with _invalidate_generation_reservation_on_failure(
        db,
        run_id=run_id,
        dry_run=args.dry_run,
    ), psycopg.connect(**db) as conn:
        _begin_generation_snapshot(conn)
        sales_lineage_before = _load_completed_sales_lineage(conn)
        promoted_clusters = load_promoted_cluster_population(conn) if clustering_enabled else None
        # Preflight: champion ↔ cluster generation lineage (sql/198)
        if args.generation_purpose == "release_candidate" and args.model_id is None:
            check_champion_cluster_lineage(conn, allow_mismatch=False)

        # Load data
        logger.info("Step 1: Loading data...")
        with profiled_section("load_data"):
            with conn.cursor() as cur:
                sales_table = resolve_forecast_sales_table(cur)
            # Per-month (item_id, loc, startdate) champion assignments. A DFU
            # can win a different model each month; this frame preserves that.
            population = load_forecast_population(
                conn,
                planning_month=forecast_month_generated,
                min_history_months=_population_min_history(
                    args.model_id,
                    config,
                    cold_start_min_months=cold_start_min_months,
                ),
                active_window_months=active_window_months,
                item_id=item_filter,
                loc=loc_filter,
                sales_table=sales_table,
            )
            if args.model_id:
                raw_routes = pd.DataFrame()
                route_fallback_model_id = args.model_id
            else:
                raw_routes = get_champion_assignments(
                    conn,
                    item_filter,
                    loc_filter,
                )
                route_fallback_model_id = fallback_model_id
            if raw_routes.empty and args.model_id is None and item_filter is None:
                raise RuntimeError(
                    "No champion assignments found. Run champion selection before generation."
                )
            champion_month_df = _align_routes_to_population(
                raw_routes,
                population,
                fallback_model_id=route_fallback_model_id,
                planning_month=forecast_month_generated,
            )

            if args.max_dfus:
                # Sample the FIRST N distinct DFUs (not the first N month-rows,
                # which would truncate a DFU's horizon mid-stream).
                distinct_dfus = champion_month_df[_DFU_KEY_COLUMNS].drop_duplicates()
                keep_dfus = distinct_dfus.head(args.max_dfus)
                if len(keep_dfus) < len(distinct_dfus):
                    champion_month_df = champion_month_df.merge(
                        keep_dfus,
                        on=_DFU_KEY_COLUMNS,
                        how="inner",
                    )
                    logger.info("Sampling limited to %s DFUs (--max-dfus)", f"{args.max_dfus:,}")

            # Per-DFU month → champion model_id routing (applied as a
            # post-generation filter so each month ships its true champion).
            month_routing = build_month_routing(champion_month_df)
            ensemble_routing = build_ensemble_routing(champion_month_df)
            # One row per DFU for the per-DFU generation loops (cluster_id /
            # customer_group are DFU-level). The kept source_model_id is the
            # earliest month's champion; other months are regenerated under
            # their own model and filtered in.
            champion_df = collapse_to_dfu(champion_month_df)

            sales_df = load_recent_sales(
                conn,
                champion_df,
                lookback_months=lookback_months,
                sales_table=sales_table,
            )
            dfu_attrs = load_dfu_attrs(conn, item_filter, loc_filter)
            item_attrs_df = load_item_attrs(conn, item_filter)
            sales_lineage_after = _load_completed_sales_lineage(conn)
            if sales_lineage_after != sales_lineage_before:
                raise RuntimeError(
                    "The completed sales batch changed while production forecast "
                    "inputs were loaded; retry against one stable source batch"
                )

        # Determine which model_ids we need to load.
        # source_model_id = the underlying algorithm that won per DFU (e.g. lgbm_cluster).
        # Populated by champion selection after sql/007_create_fact_external_forecast_monthly.sql is applied.
        # Always include fallback_model_id so every DFU has at least one model to use.
        # Use the PER-MONTH frame so every model that wins ANY month is loaded.
        if args.model_id:
            model_ids_needed = {args.model_id}
        else:
            model_ids_needed: set[str] = {
                fallback_model_id,
                cold_start_model_id,
            }
            for row in champion_df.itertuples(index=False):
                dfu_key = (str(row.item_id), str(row.customer_group), str(row.loc))
                for month in production_months:
                    resolved = _resolve_champion_route(
                        dfu_key,
                        month,
                        month_routing,
                        ensemble_routing,
                    )
                    if resolved is None:
                        continue
                    route_type, route_value = resolved
                    if route_type == "model":
                        model_ids_needed.add(str(route_value))
                    else:
                        model_ids_needed.update(
                            str(entry["model"]) for entry in route_value if entry.get("model")
                        )

        _validate_forecastable_model_ids(
            model_ids_needed,
            source="Champion routing",
        )

        required_tree_model_ids = {
            model_id for model_id in model_ids_needed if model_id not in non_tree_models
        }
        expected_tree_specs: dict[str, TreeArtifactSpec] = {}
        if "lgbm_cluster" in required_tree_model_ids:
            expected_tree_lineage = ProductionTreeArtifactLineage(
                source_sales_batch_id=sales_lineage_after[0],
                data_checksum=sales_lineage_after[1],
                history_end=pd.Timestamp(sales_df.attrs["history_end"]).date(),
                cluster_experiment_id=(
                    promoted_clusters.experiment_id if promoted_clusters is not None else None
                ),
                cluster_assignment_count=(
                    promoted_clusters.assignment_count
                    if promoted_clusters is not None
                    else None
                ),
                cluster_assignment_checksum=(
                    promoted_clusters.assignment_checksum
                    if promoted_clusters is not None
                    else None
                ),
            )
            expected_tree_specs["lgbm_cluster"] = build_tree_artifact_spec(
                model_id="lgbm_cluster",
                model_config=build_production_tree_model_config_payload(
                    load_forecast_pipeline_config(),
                    model_id="lgbm_cluster",
                    project_root=ROOT,
                ),
                lineage=expected_tree_lineage,
                cluster_strategy=("per_cluster" if clustering_enabled else "global"),
                cluster_labels=(
                    promoted_clusters.cluster_labels
                    if promoted_clusters is not None
                    else {"global"}
                ),
            )

        # Load model artifacts
        logger.info("Step 2: Loading model artifacts for: %s", model_ids_needed)
        with profiled_section("load_models"):
            neural_cohorts_by_min_history: dict[int, NeuralCohortIdentity] = {}
            expected_neural_cohorts: dict[str, NeuralCohortIdentity] = {}
            for neural_model_id in sorted(model_ids_needed & set(SUPPORTED_NEURAL_MODELS)):
                min_history = int(
                    _algorithm_params(config, neural_model_id)["min_history"]
                )
                if min_history not in neural_cohorts_by_min_history:
                    neural_cohorts_by_min_history[min_history] = (
                        load_neural_training_cohort_identity(
                            conn,
                            sales_table=sales_table,
                            history_end=pd.Timestamp(sales_df.attrs["history_end"]).date(),
                            min_history=min_history,
                        )
                    )
                expected_neural_cohorts[neural_model_id] = (
                    neural_cohorts_by_min_history[min_history]
                )
            loaded_models = _load_tree_models(
                model_ids_needed,
                non_tree_models,
                config,
                expected_specs=expected_tree_specs,
            )
            neural_models = _load_neural_models(
                model_ids_needed,
                config,
                source_sales_batch_id=sales_lineage_after[0],
                data_checksum=sales_lineage_after[1],
                history_end=pd.Timestamp(sales_df.attrs["history_end"]).date(),
                expected_cohorts=expected_neural_cohorts,
            )

        missing_tree_models = _missing_required_tree_model_ids(
            required_tree_model_ids,
            loaded_models,
        )
        if missing_tree_models:
            raise RuntimeError(
                "Required tree model artifact(s) missing for production generation: "
                f"{', '.join(missing_tree_models)}. Run production final-fit training "
                "(`uv run python scripts/ml/train_production_models.py --model "
                "lgbm_cluster`) before generating forecasts."
            )

        if not loaded_models:
            if args.model_id:
                # Non-tree model: run its canonical direct adapter.
                logger.info(
                    "No .pkl artifacts for %s — using direct inference from history", args.model_id
                )
            else:
                logger.info(
                    "No tree artifacts are needed; champion routing will use direct "
                    "non-tree inference for: %s",
                    model_ids_needed,
                )

        # Data-integrity guard: abort if promoted cluster assignments were wiped (every DFU
        # NULL while per-cluster trees are loaded). Otherwise tree inference would
        # silently collapse high-volume forecasts to a near-constant. Fail loud.
        if _cluster_assignments_wiped(champion_df, loaded_models, non_tree_models):
            raise RuntimeError(
                "current_sku_cluster_assignment is empty for every champion DFU while multi-cluster "
                "LightGBM artifacts are loaded — per-cluster inference would collapse to a "
                "single arbitrary cluster model (high-volume forecasts crash to a "
                "near-constant). Restore the promoted cluster assignment first:\n"
                "  uv run python scripts/ml/restore_cluster_assignments.py"
            )

        # Pre-index data structures for O(1) per-DFU lookups
        logger.info("Step 2b: Pre-indexing data structures...")
        with profiled_section("pre_index"):
            sales_index = build_sales_index(sales_df)
            attrs_index = build_attrs_index(dfu_attrs)
            item_index = build_item_index(item_attrs_df)
        logger.info(
            "Indexed %s DFUs (sales), %s DFUs (attrs), %s items",
            f"{len(sales_index):,}",
            f"{len(attrs_index):,}",
            f"{len(item_index):,}",
        )
        validate_customer_group_history(
            sales_index,
            champion_df,
            minimum_months=cold_start_min_months,
        )
        if args.model_id is None:
            validate_route_history_requirements(
                sales_index,
                champion_df,
                production_months=production_months,
                month_routing=month_routing,
                ensemble_routing=ensemble_routing,
                min_history_months=min_history_months,
                mstl_min_history=int(_algorithm_params(config, "mstl")["min_history"]),
            )

        # Build forecast CI sigma lookup (per-DFU uncertainty from backtest residuals).
        # CLI --confidence-intervals/--no-confidence-intervals overrides the config
        # default so the UI's "Include Confidence Intervals" toggle takes effect.
        ci_cfg = config.get("confidence_interval", {})
        ci_enabled = ci_cfg.get("enabled", False)
        if args.confidence_intervals is not None:
            ci_enabled = args.confidence_intervals
        sigma_lookup: dict = {}
        if ci_enabled:
            logger.info("Step 2c: Building forecast uncertainty (CI bands)...")
            with profiled_section("build_ci_sigma"):
                cluster_map = build_item_location_cluster_map(dfu_attrs)
                sigma_lookup = build_sigma_lookup(
                    conn,
                    config,
                    cluster_map,
                    requested_model_id=args.model_id or CHAMPION_MODEL_ID,
                )
            logger.info("CI sigma lookup: %s DFUs mapped", f"{len(sigma_lookup):,}")

        staging_model_id = args.model_id or CHAMPION_MODEL_ID
        generation_evidence = _collect_generation_evidence(
            conn,
            candidate_model_id=staging_model_id,
            generation_purpose=args.generation_purpose,
        )
        evidence_sales_batch = generation_evidence.get("source_sales_batch_id")
        if evidence_sales_batch is not None and int(evidence_sales_batch) != sales_lineage_after[0]:
            raise RuntimeError(
                "Forecast generation evidence and sales history do not share one source batch"
            )
        evidence_cluster_id = generation_evidence.get("cluster_experiment_id")
        if (
            evidence_cluster_id is not None
            and clustering_enabled
            and promoted_clusters is not None
            and int(evidence_cluster_id) != promoted_clusters.experiment_id
        ):
            raise RuntimeError(
                "Forecast generation evidence and tree artifacts do not share one "
                "promoted clustering experiment"
            )
        generation_metadata = _source_generation_metadata(
            source_sales_batch_id=sales_lineage_after[0],
            data_checksum=sales_lineage_after[1],
            history_end=sales_df.attrs["history_end"],
        )
        generation_metadata.update(
            _direct_generation_metadata(config, model_ids_needed)
        )
        generation_metadata.update(
            _generation_config_metadata(config, model_ids_needed)
        )
        generation_metadata.update(_neural_generation_metadata(neural_models))
        generation_metadata.update(_tree_generation_metadata(loaded_models))
        # End the stable read snapshot before long-running inference. The next
        # transaction on this connection is read-write for the immutable write.
        conn.commit()

        all_rows: list[dict] = []
        skipped = 0

        if args.model_id in non_tree_models:
            # Canonical direct-inference models run their backtest-matched adapter.
            with profiled_section("direct_inference"):
                all_rows = _generate_canonical_non_tree_rows(
                    config=config,
                    model_id=args.model_id,
                    sales_df=sales_df,
                    dfu_attrs=dfu_attrs,
                    item_attrs=item_attrs_df,
                    target_dfus=champion_df,
                    predict_months=production_months,
                    forecast_month_generated=forecast_month_generated,
                    run_id=run_id,
                    fitted_neural_model=(
                        neural_models[args.model_id].fitted_model
                        if args.model_id in neural_models
                        else None
                    ),
                )
            skipped = len(champion_df) - len(all_rows) // max(horizon, 1)
            logger.info(
                "Step 3 complete (direct inference): %s rows, %s skipped",
                f"{len(all_rows):,}",
                f"{skipped:,}",
            )
        else:
            # ── Tree model batch inference ───────────────────────────────
            # Group DFUs by (model_id, cluster_id) for batched inference
            logger.info("Step 3: Building cluster groups for %s DFUs...", f"{len(champion_df):,}")
            with profiled_section("build_cluster_groups"):
                cluster_groups: dict[tuple, list] = defaultdict(list)
                # Non-tree champions are collected here and generated by their
                # canonical adapters AFTER this loop —
                # they have no .pkl and must keep their TRUE model_id, never the lgbm
                # fallback (F-11: displayed champion == shipped model).
                non_tree_routed: dict[str, list] = defaultdict(list)

                def _enqueue_dfu_model(
                    item_id,
                    customer_group,
                    loc,
                    cluster_id,
                    model_id,
                    champ,
                ) -> bool:
                    """Route ONE (DFU, model) into the tree or statistical path.

                    Returns True if the DFU-model was enqueued (or routed to the
                    canonical non-tree adapter), False if it was skipped (no
                    artifact / no grid). A DFU may be enqueued under several
                    models when its per-month champion varies; the per-month
                    filter later keeps only the winning months.
                    """
                    nonlocal skipped
                    # Non-tree champion source (MSTL, foundation, or DL): no .pkl exists, so the tree path
                    # would substitute the lgbm fallback and ship LightGBM under the
                    # champion's label. Route it to the canonical adapter instead so
                    # the shipped forecast actually IS its champion model (F-11).
                    if model_id in non_tree_models:
                        non_tree_routed[model_id].append(champ)
                        return True

                    artifact = _resolve_tree_artifact(
                        loaded_models,
                        model_id,
                        cluster_id,
                    )
                    if artifact is None:
                        return False

                    grid = build_inference_grid(
                        item_id=item_id,
                        loc=loc,
                        cluster_id=cluster_id,
                        horizon=horizon,
                        min_months=cold_start_min_months,
                        sales_index=sales_index,
                        attrs_index=attrs_index,
                        item_index=item_index,
                        customer_group=customer_group,
                    )
                    if grid is None:
                        return False

                    cluster_groups[(model_id, cluster_id)].append(
                        (
                            {
                                "item_id": item_id,
                                "customer_group": customer_group,
                                "loc": loc,
                                "cluster_id": cluster_id,
                            },
                            grid,
                            artifact,
                        )
                    )
                    return True

                cold_start_count = 0
                for _, champ in champion_df.iterrows():
                    item_id = champ["item_id"]
                    loc = champ["loc"]
                    raw_customer_group = champ.get("customer_group")
                    customer_group = (
                        None
                        if raw_customer_group is None or pd.isna(raw_customer_group)
                        else str(raw_customer_group)
                    )
                    cluster_id = champ["cluster_id"]

                    # Determine history length for cold-start routing
                    sales_entry = _lookup_dfu(sales_index, item_id, customer_group, loc)
                    n_months = int(sales_entry[2]) if sales_entry else 0

                    if args.model_id:
                        # Single-model override: one model across the whole horizon.
                        if not _enqueue_dfu_model(
                            item_id,
                            customer_group,
                            loc,
                            cluster_id,
                            args.model_id,
                            champ,
                        ):
                            skipped += 1
                        continue
                    if n_months < min_history_months:
                        # Cold-start DFU: insufficient history for LightGBM —
                        # one cold-start model across the whole horizon. This
                        # OVERRIDES any per-month champion routing, so drop the
                        # DFU from month_routing too — otherwise the post-filter
                        # would discard its cold-start rows (their model_id does
                        # not match the routed winners).
                        dfu_key = (item_id, customer_group, loc)
                        month_routing.pop(dfu_key, None)
                        ensemble_routing.pop(dfu_key, None)
                        if _enqueue_dfu_model(
                            item_id,
                            customer_group,
                            loc,
                            cluster_id,
                            cold_start_model_id,
                            champ,
                        ):
                            cold_start_count += 1
                        else:
                            skipped += 1
                        continue

                    # Mature DFU: enqueue only the models resolved for the
                    # requested forward horizon. The post-filter retains the
                    # exact as-of route for each month.
                    dfu_key = (item_id, customer_group, loc)
                    dfu_models: set[str] = set()
                    for month in production_months:
                        resolved = _resolve_champion_route(
                            dfu_key,
                            month,
                            month_routing,
                            ensemble_routing,
                        )
                        if resolved is None:
                            continue
                        route_type, route_value = resolved
                        if route_type == "model":
                            dfu_models.add(str(route_value))
                        else:
                            dfu_models.update(
                                str(entry["model"]) for entry in route_value if entry.get("model")
                            )
                    if not dfu_models:
                        # No route was valid as of any requested production
                        # month.  Never resurrect a future-only winner from the
                        # collapsed frame; use the explicit production fallback.
                        dfu_models.add(fallback_model_id)
                    enqueued_any = False
                    for model_id in sorted(dfu_models):
                        if _enqueue_dfu_model(
                            item_id,
                            customer_group,
                            loc,
                            cluster_id,
                            model_id,
                            champ,
                        ):
                            enqueued_any = True
                    if not enqueued_any:
                        skipped += 1

            logger.info(
                "%s DFUs in %d cluster groups, %s skipped, %s cold-start (→ %s)",
                f"{sum(len(v) for v in cluster_groups.values()):,}",
                len(cluster_groups),
                f"{skipped:,}",
                f"{cold_start_count:,}",
                cold_start_model_id,
            )
            n_non_tree = sum(len(v) for v in non_tree_routed.values())
            if n_non_tree:
                logger.info(
                    "%s DFU(s) routed to canonical non-tree adapters: %s",
                    f"{n_non_tree:,}",
                    ", ".join(sorted(non_tree_routed)),
                )
            # Batch-predict per cluster group — parallelise across independent groups.
            # Guarded: when every champion routes to the statistical path (no tree
            # groups), ThreadPoolExecutor(max_workers=0) would raise — skip it.
            if cluster_groups:
                n_workers = min(len(cluster_groups), min(os.cpu_count() or 4, 4))
                logger.info(
                    "Step 3b: Running batched inference (%d groups, %d workers)...",
                    len(cluster_groups),
                    n_workers,
                )

                def _run_group(key_entries):
                    (mid, cid), entries = key_entries
                    art = entries[0][2]
                    dfu_lst = [(e[0], e[1]) for e in entries]
                    rows = generate_forecasts_batch(
                        artifact=art,
                        dfu_list=dfu_lst,
                        horizon=horizon,
                        forecast_month_generated=forecast_month_generated,
                        run_id=run_id,
                        model_id=mid,
                    )
                    return (mid, cid, len(dfu_lst), rows)

                with profiled_section("batched_inference"):
                    with ThreadPoolExecutor(max_workers=n_workers) as executor:
                        futures = {
                            executor.submit(_run_group, item): item
                            for item in cluster_groups.items()
                        }
                        for future in as_completed(futures):
                            mid, cid, n_dfus, batch_rows = future.result()
                            all_rows.extend(batch_rows)
                            logger.info(
                                "(%s, cluster %s): %s DFUs -> %s rows",
                                mid,
                                cid,
                                f"{n_dfus:,}",
                                f"{len(batch_rows):,}",
                            )

            # Non-tree champions use the same implementations as their
            # backtests, preserving displayed/evaluated/shipped identity.
            if non_tree_routed:
                with profiled_section("canonical_non_tree_inference"):
                    for mid, champ_rows in sorted(non_tree_routed.items()):
                        model_rows = _generate_canonical_non_tree_rows(
                            config=config,
                            model_id=mid,
                            sales_df=sales_df,
                            dfu_attrs=dfu_attrs,
                            item_attrs=item_attrs_df,
                            target_dfus=pd.DataFrame(champ_rows),
                            predict_months=production_months,
                            forecast_month_generated=forecast_month_generated,
                            run_id=run_id,
                            fitted_neural_model=(
                                neural_models[mid].fitted_model if mid in neural_models else None
                            ),
                        )
                        all_rows.extend(model_rows)
                        logger.info(
                            "(%s, canonical adapter): %s DFUs -> %s rows",
                            mid,
                            f"{len(champ_rows):,}",
                            f"{len(model_rows):,}",
                        )

            logger.info(
                "Step 3 complete: %s rows, %s skipped", f"{len(all_rows):,}", f"{skipped:,}"
            )

        # Per-month champion routing: each mature DFU was generated under every
        # model it wins across the horizon; drop the months where a given model
        # is NOT the champion so each (DFU, month) ships its true champion. A
        # no-op for --model-id / cold-start (no per-month routing) runs.
        if not args.model_id and (month_routing or ensemble_routing):
            n_before = len(all_rows)
            all_rows = filter_rows_to_champion_months(all_rows, month_routing, ensemble_routing)
            logger.info(
                "Per-month champion filter: kept %s of %s rows",
                f"{len(all_rows):,}",
                f"{n_before:,}",
            )

        # Production tables operate at item/location grain, while all model
        # training and champion routing above operate at customer-group DFU
        # grain. Aggregate only after every group has followed its own route.
        validate_customer_group_forecast_coverage(
            all_rows,
            champion_df,
            horizon=horizon,
        )
        group_row_count = len(all_rows)
        all_rows = aggregate_customer_group_forecasts(all_rows)
        if ci_enabled and sigma_lookup:
            all_rows = attach_aggregate_confidence_intervals(
                all_rows,
                sigma_lookup=sigma_lookup,
                ci_cfg=ci_cfg,
            )
        logger.info(
            "Customer-group aggregation: %s DFU rows -> %s item/location rows",
            f"{group_row_count:,}",
            f"{len(all_rows):,}",
        )

        # Write to staging table
        logger.info(
            "Step 4: Writing to fact_production_forecast_staging (model_id=%s)...", staging_model_id
        )
        with profiled_section("write_forecast_staging"):
            written = write_forecast_staging(
                all_rows,
                conn,
                staging_model_id,
                dry_run=args.dry_run,
                generation_purpose=args.generation_purpose,
                generation_evidence=generation_evidence,
                generation_metadata=generation_metadata,
                promotion_eligible=False,
                requested_horizon=horizon,
            )
        logger.info("Written: %s rows", f"{written:,}")

    elapsed = time.time() - t_start
    logger.info("Production forecast complete in %.0fs (%.1fm)", elapsed, elapsed / 60)
    logger.info(
        "plan_version=%s, rows=%s, skipped=%s", plan_version, f"{written:,}", f"{skipped:,}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
