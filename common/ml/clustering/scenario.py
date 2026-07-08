"""Reusable clustering scenario infrastructure.

Contains the functions that are imported by routers, job_state, and pipeline
scripts for scenario ID generation, promotion, result retrieval, and config
defaults.  Orchestration (``run_scenario``, ``_run_full_pipeline``) stays in
``scripts/run_clustering_scenario.py``.
"""

import gzip
import io
import json
import logging
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from common.core.db import get_db_params
from common.core.mv_refresh import refresh_for_tables
from common.services.perf_profiler import profiled_section

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Root of the project — two levels up from common/ml/clustering/
from common.core.paths import PROJECT_ROOT as ROOT
# Directory to store scenario temp data
SCENARIO_BASE = Path("/tmp/clustering_scenarios")

# Scenario IDs must match this pattern (same regex as enforced in clusters.py router)
_SCENARIO_ID_RE = re.compile(r"^sc_\d{8}_\d{6}_[0-9a-f]{4}$")

# Hardcoded fallback defaults used when no promoted experiment is found in the DB.
_FALLBACK_DEFAULTS: dict[str, Any] = {
    "time_window_months": 36,
    "min_months_history": 12,
    "k_range": [9, 18],
    "min_cluster_size_pct": 2.0,
    "use_pca": False,
    "pca_components": None,
    "labeling": {
        "volume_thresholds": {"very_high": 0.90, "high": 0.75, "low": 0.25, "very_low": 0.10},
        "cv_thresholds": {"steady": 0.4, "volatile": 0.8},
        "seasonality_threshold": 0.3,
        "zero_demand_threshold": 0.15,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_scenario_dir(scenario_id: str) -> Path:
    """Return the scenario directory path after validating scenario_id.

    Raises ValueError if the ID format is invalid or if the resolved path
    would escape SCENARIO_BASE (path traversal guard).
    """
    if not _SCENARIO_ID_RE.match(scenario_id):
        raise ValueError(
            f"Invalid scenario_id '{scenario_id}'. "
            "Expected format: sc_YYYYMMDD_HHMMSS_<4hex>"
        )
    resolved = (SCENARIO_BASE / scenario_id).resolve()
    base_resolved = SCENARIO_BASE.resolve()
    if not resolved.is_relative_to(base_resolved):
        raise ValueError(f"scenario_id '{scenario_id}' resolves outside SCENARIO_BASE")
    return resolved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_scenario_id() -> str:
    """Generate a unique scenario ID."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:4]
    return f"sc_{ts}_{short_uuid}"


def _load_config_defaults() -> dict[str, Any]:
    """Load defaults from promoted experiment in DB, falling back to hardcoded defaults."""
    try:
        import psycopg

        db = get_db_params()
        with psycopg.connect(**db) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT feature_params, model_params, label_params "
                "FROM cluster_experiment "
                "WHERE is_promoted = TRUE "
                "ORDER BY promoted_at DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row:
                fp = row[0] if isinstance(row[0], dict) else (json.loads(row[0]) if row[0] else {})
                mp = row[1] if isinstance(row[1], dict) else (json.loads(row[1]) if row[1] else {})
                lp = row[2] if isinstance(row[2], dict) else (json.loads(row[2]) if row[2] else {})
                return {
                    "time_window_months": fp.get("time_window_months", 36),
                    "min_months_history": fp.get("min_months_history", 12),
                    "k_range": mp.get("k_range", [9, 18]),
                    "min_cluster_size_pct": mp.get("min_cluster_size_pct", 2.0),
                    "use_pca": mp.get("use_pca", False),
                    "pca_components": mp.get("pca_components"),
                    "labeling": {
                        "volume_thresholds": {
                            "high": lp.get("volume_high", 0.75),
                            "low": lp.get("volume_low", 0.25),
                        },
                        "cv_thresholds": {
                            "steady": lp.get("cv_steady", 0.4),
                            "volatile": lp.get("cv_volatile", 0.8),
                        },
                        "seasonality_threshold": lp.get("seasonality_threshold", 0.3),
                        "zero_demand_threshold": lp.get("zero_demand_threshold", 0.15),
                    },
                }
    except Exception:
        logging.getLogger(__name__).warning(
            "Could not load config defaults from DB — using hardcoded defaults",
            exc_info=True,
        )
    return {**_FALLBACK_DEFAULTS, "labeling": {**_FALLBACK_DEFAULTS["labeling"]}}


def get_scenario_result(scenario_id: str) -> dict[str, Any] | None:
    """Retrieve a previously run scenario result."""
    try:
        scenario_dir = _safe_scenario_dir(scenario_id)
    except ValueError:
        return None
    result_path = scenario_dir / "scenario_result.json"
    if not result_path.exists():
        return None
    with open(result_path) as f:
        return json.load(f)


def store_durable_labels(experiment_id: int, labels_path: Path) -> None:
    """Persist a scenario's per-SKU labels onto its experiment row (gzip-compressed).

    Makes the experiment re-promotable from the database alone, even after the
    working ``/tmp`` artifacts are cleared. Best-effort: a missing file or a DB
    error is logged, never raised — the run itself already succeeded.
    """
    import psycopg

    if not labels_path.exists():
        return
    try:
        gz = gzip.compress(labels_path.read_bytes())
        with psycopg.connect(**get_db_params()) as conn:
            conn.execute(
                "UPDATE cluster_experiment SET cluster_labels_gz = %s WHERE experiment_id = %s",
                (gz, experiment_id),
            )
    except (OSError, psycopg.Error):
        logger.warning("Failed to store durable labels for experiment %d", experiment_id)


def _load_label_bytes(scenario_id: str, labels_path: Path) -> bytes | None:
    """Return the cluster_labels.csv content for a scenario.

    Prefers the working file; falls back to the durable copy stored on the
    experiment row (``cluster_labels_gz``). Returns None when neither exists.
    """
    import psycopg

    if labels_path.exists():
        return labels_path.read_bytes()
    try:
        with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT cluster_labels_gz FROM cluster_experiment WHERE scenario_id = %s",
                (scenario_id,),
            )
            row = cur.fetchone()
    except psycopg.Error:
        logger.exception("Failed to read durable labels for scenario %s", scenario_id)
        return None
    if row and row[0]:
        return gzip.decompress(bytes(row[0]))
    return None


def promote_scenario(scenario_id: str) -> dict[str, Any]:
    """Promote a scenario to production by updating dim_sku.ml_cluster.

    After updating the SKU dimension table, this also refreshes accuracy
    materialized views so that Accuracy Comparison reflects the new clusters.

    Labels are loaded from the working scenario dir when present, else from the
    durable copy on the experiment row — so any completed experiment stays
    promotable even after its /tmp artifacts are gone.
    """
    import psycopg

    scenario_dir = _safe_scenario_dir(scenario_id)
    labels_path = scenario_dir / "cluster_labels.csv"

    labels_bytes = _load_label_bytes(scenario_id, labels_path)
    if labels_bytes is None:
        raise FileNotFoundError(
            f"Cluster labels unavailable for scenario {scenario_id} "
            "(no working file and no durable copy). Re-run the experiment."
        )

    # Copy labels to production location (reconstructs the file from the durable
    # copy if the working file was gone).
    with profiled_section("copy_artifacts_to_production"):
        prod_dir = ROOT / "data" / "clustering"
        prod_dir.mkdir(parents=True, exist_ok=True)
        (prod_dir / "cluster_labels.csv").write_bytes(labels_bytes)

        # Also copy centroids and profiles
        for fname in ["cluster_centroids.csv", "scenario_result.json"]:
            src = scenario_dir / fname
            if src.exists():
                shutil.copy2(src, prod_dir / fname)

        # Update cluster_metadata.json so the overview panel shows correct metrics
        result_path = scenario_dir / "scenario_result.json"
        if result_path.exists():
            with open(result_path) as f:
                scenario_data = json.load(f)
            scenario_result = scenario_data.get("result", {})
            metadata = {
                "optimal_k": scenario_result.get("optimal_k"),
                "silhouette_score": scenario_result.get("silhouette_score"),
                "inertia": scenario_result.get("inertia"),
                "total_dfus": scenario_result.get("total_dfus"),
                "k_selection_results": scenario_result.get("k_selection_results"),
            }
            with open(prod_dir / "cluster_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

    # Update database
    with profiled_section("update_database"):
        df = pd.read_csv(io.BytesIO(labels_bytes))
        db = get_db_params()

        with psycopg.connect(**db) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TEMP TABLE _cluster_updates (
                        sku_ck TEXT PRIMARY KEY,
                        cluster_label TEXT NOT NULL
                    ) ON COMMIT DROP
                """)

                valid = df.dropna(subset=["cluster_label"])
                with cur.copy("COPY _cluster_updates (sku_ck, cluster_label) FROM STDIN") as copy:
                    for _, r in valid.iterrows():
                        copy.write_row((str(r["sku_ck"]), str(r["cluster_label"])))

                cur.execute("""
                    UPDATE dim_sku d
                    SET ml_cluster = u.cluster_label,
                        modified_ts = NOW()
                    FROM _cluster_updates u
                    WHERE d.sku_ck = u.sku_ck
                """)
                updated_count = cur.rowcount

                # Gen-4 SC-9: mark per-cluster tuning profiles stale so the
                # next tuning run re-trains them against the new partitions.
                # Best-effort — the table is created by sql/148; missing table
                # is fine (feature flag disabled).
                try:
                    cur.execute("SAVEPOINT invalidate_profiles")
                    stale_reason = f"cluster_promotion:{scenario_id}"
                    cluster_names = (
                        df["cluster_label"].dropna().astype(str).unique().tolist()
                        if "cluster_label" in df.columns else []
                    )
                    for cname in cluster_names:
                        cur.execute("""
                            INSERT INTO cluster_tuning_profile_state
                                (cluster_name, stale, stale_reason, stale_since, modified_ts)
                            VALUES (%s, TRUE, %s, NOW(), NOW())
                            ON CONFLICT (cluster_name) DO UPDATE
                            SET stale = TRUE,
                                stale_reason = EXCLUDED.stale_reason,
                                stale_since = EXCLUDED.stale_since,
                                modified_ts = NOW()
                        """, (cname, stale_reason))
                    logger.info(
                        "Marked %d cluster_tuning_profile_state rows stale",
                        len(cluster_names),
                    )
                except psycopg.Error as exc:
                    cur.execute("ROLLBACK TO SAVEPOINT invalidate_profiles")
                    logger.warning(
                        "cluster_tuning_profile_state update skipped "
                        "(apply sql/148 if persistence needed): %s", exc,
                    )

                conn.commit()

        # Refresh every MV that embeds dim_sku attributes (accuracy slices,
        # coverage, fill rate, inventory bridge, ...) so dashboards reflect the
        # new clusters. Runs on its own autocommit connection (CONCURRENTLY is
        # illegal inside a transaction block). include_heavy=False: a label
        # change does not justify the 10-30 min mv_intramonth_stockout rebuild
        # — the scheduled refresh_all_mvs safety net covers it.
        logger.info("Refreshing dim_sku-dependent materialized views after cluster promotion ...")
        refresh_for_tables(["dim_sku"], db_params=db, include_heavy=False)

    # Build distribution
    distribution: dict[str, int] = {}
    if "cluster_label" in df.columns:
        for label, count in df["cluster_label"].value_counts().items():
            distribution[str(label)] = int(count)

    return {
        "status": "promoted",
        "scenario_id": scenario_id,
        "dfus_updated": updated_count,
        "cluster_distribution": distribution,
    }
