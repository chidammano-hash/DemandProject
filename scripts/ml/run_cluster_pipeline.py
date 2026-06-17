"""Unified clustering pipeline — single entry point for both CLI and UI.

Creates a cluster_experiment row, runs the scenario via run_clustering_scenario,
and auto-promotes the result. Reads defaults from the currently promoted
experiment in the DB, falling back to hardcoded defaults.

Usage:
    python scripts/ml/run_cluster_pipeline.py
    python scripts/ml/run_cluster_pipeline.py --label "My Run"
    python scripts/ml/run_cluster_pipeline.py --no-promote
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.db import get_db_params  # noqa: E402
from scripts.ml.run_clustering_scenario import (  # noqa: E402
    generate_scenario_id,
    promote_scenario,
    run_scenario,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded defaults (formerly in clustering_config.yaml)
# ---------------------------------------------------------------------------
DEFAULT_FEATURE_PARAMS: dict[str, Any] = {
    "time_window_months": 36,
    "min_months_history": 1,
}

DEFAULT_MODEL_PARAMS: dict[str, Any] = {
    "k_range": [9, 18],
    "min_cluster_size_pct": 2.0,
    "use_pca": False,
    "pca_components": None,
    "all_features": False,
}

DEFAULT_LABEL_PARAMS: dict[str, Any] = {
    "volume_high": 0.75,
    "volume_low": 0.25,
    "cv_steady": 0.4,
    "cv_volatile": 0.8,
    "seasonality_threshold": 0.3,
    "zero_demand_threshold": 0.15,
}


def get_effective_params() -> dict[str, Any]:
    """Return params from the promoted experiment, or hardcoded defaults."""
    import psycopg

    try:
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
                fp = row[0] if isinstance(row[0], dict) else json.loads(row[0]) if row[0] else {}
                mp = row[1] if isinstance(row[1], dict) else json.loads(row[1]) if row[1] else {}
                lp = row[2] if isinstance(row[2], dict) else json.loads(row[2]) if row[2] else {}
                return {
                    "feature_params": {**DEFAULT_FEATURE_PARAMS, **fp},
                    "model_params": {**DEFAULT_MODEL_PARAMS, **mp},
                    "label_params": {**DEFAULT_LABEL_PARAMS, **lp},
                }
    except Exception:
        logger.warning("Could not read promoted experiment from DB — using hardcoded defaults", exc_info=True)

    return {
        "feature_params": {**DEFAULT_FEATURE_PARAMS},
        "model_params": {**DEFAULT_MODEL_PARAMS},
        "label_params": {**DEFAULT_LABEL_PARAMS},
    }


def run_unified_pipeline(
    feature_params: dict[str, Any] | None = None,
    model_params: dict[str, Any] | None = None,
    label_params: dict[str, Any] | None = None,
    label: str = "Pipeline Run",
    auto_promote: bool = True,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run the full clustering pipeline through the experiment system.

    1. Resolve effective params (promoted experiment or defaults)
    2. Create a cluster_experiment row (linked to job_id, started_at stamped)
    3. Run via run_scenario() — which writes all metrics to the row
    4. Auto-promote if requested, then finalize the row to 'completed'

    The experiment row stays ``running`` until the whole pipeline (clustering +
    promote) is done, so it never reads ``completed`` while the job is still
    working. run_scenario is the single writer of metrics — this function does
    NOT re-write them (a past bug clobbered the scalar metrics with NULLs).
    """
    import psycopg

    # Resolve params
    effective = get_effective_params()
    fp = feature_params or effective["feature_params"]
    mp = model_params or effective["model_params"]
    lp = label_params or effective["label_params"]

    scenario_id = generate_scenario_id()
    db = get_db_params()

    # Create experiment row
    experiment_id: int | None = None
    try:
        with psycopg.connect(**db) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO cluster_experiment
                    (scenario_id, label, status, job_id, started_at,
                     feature_params, model_params, label_params)
                VALUES (%s, %s, 'running', %s, NOW(), %s, %s, %s)
                RETURNING experiment_id
                """,
                [scenario_id, label, job_id, json.dumps(fp), json.dumps(mp), json.dumps(lp)],
            )
            experiment_id = cur.fetchone()[0]
            conn.commit()
        logger.info("Created experiment #%d (scenario %s)", experiment_id, scenario_id)
    except Exception:
        logger.exception("Failed to create experiment row — running without tracking")

    # Run scenario. run_scenario writes ALL metrics to the experiment row itself
    # (via _update_experiment_completed). We pass final_status='running' when we
    # will auto-promote, so the row only flips to 'completed' after promote.
    t0 = time.time()
    try:
        result = run_scenario(
            scenario_id=scenario_id,
            experiment_id=experiment_id,
            feature_params=fp,
            model_params=mp,
            label_params=lp,
            final_status=("running" if auto_promote else "completed"),
        )
        runtime = time.time() - t0
        logger.info("Scenario completed in %.1fs", runtime)
    except Exception:
        runtime = time.time() - t0
        if experiment_id:
            try:
                with psycopg.connect(**db) as conn, conn.cursor() as cur:
                    cur.execute(
                        "UPDATE cluster_experiment SET status = 'failed', "
                        "completed_at = NOW(), runtime_seconds = %s "
                        "WHERE experiment_id = %s",
                        [runtime, experiment_id],
                    )
                    conn.commit()
            except psycopg.Error:
                logger.exception("Failed to update experiment to 'failed' status")
        raise

    # Auto-promote, then finalize the row to 'completed'. A promote failure must
    # not crash the job or leave the row stuck 'running' — clustering succeeded,
    # so we still mark it completed (just not promoted).
    if auto_promote and result.get("status") == "completed":
        logger.info("Auto-promoting scenario %s ...", scenario_id)
        promoted = False
        try:
            promote_result = promote_scenario(scenario_id)
            logger.info("Promoted: %d DFUs updated", promote_result.get("dfus_updated", 0))
            promoted = True
        except (FileNotFoundError, OSError, ValueError, KeyError, psycopg.Error):
            logger.exception("Auto-promote failed for scenario %s", scenario_id)
        if experiment_id:
            _finalize_pipeline_experiment(db, experiment_id, promoted=promoted)

    return result


def _finalize_pipeline_experiment(db: dict, experiment_id: int, promoted: bool) -> None:
    """Mark a pipeline experiment 'completed' once clustering + promote are done.

    On a successful promote, demote any previously-promoted row and flag this one.
    """
    import psycopg

    try:
        with psycopg.connect(**db) as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE cluster_experiment SET status = 'completed', completed_at = NOW() "
                "WHERE experiment_id = %s",
                [experiment_id],
            )
            if promoted:
                cur.execute(
                    "UPDATE cluster_experiment SET is_promoted = FALSE, promoted_at = NULL "
                    "WHERE is_promoted = TRUE AND experiment_id != %s",
                    [experiment_id],
                )
                cur.execute(
                    "UPDATE cluster_experiment SET is_promoted = TRUE, promoted_at = NOW() "
                    "WHERE experiment_id = %s",
                    [experiment_id],
                )
            conn.commit()
    except psycopg.Error:
        logger.exception("Failed to finalize pipeline experiment %d", experiment_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run unified clustering pipeline")
    parser.add_argument("--label", default="Pipeline Run", help="Experiment label")
    parser.add_argument(
        "--no-promote", action="store_true", help="Skip auto-promotion",
    )
    args = parser.parse_args()

    result = run_unified_pipeline(
        label=args.label,
        auto_promote=not args.no_promote,
    )
    logger.info("Result: %s", result.get("status", "unknown"))
