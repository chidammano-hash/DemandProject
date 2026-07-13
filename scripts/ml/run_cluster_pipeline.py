"""Unified clustering pipeline — single entry point for both CLI and UI.

Creates a cluster_experiment row, runs the scenario via run_clustering_scenario,
and auto-promotes the result. Reads defaults from the currently promoted
experiment in the DB, falling back to hardcoded defaults.

Usage:
    python scripts/ml/run_cluster_pipeline.py
    python scripts/ml/run_cluster_pipeline.py --label "My Run"
    python scripts/ml/run_cluster_pipeline.py --no-promote
    python scripts/ml/run_cluster_pipeline.py --job-id JOB --attempt-token TOKEN \
        --params-json '{"auto_promote":true}'
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
from common.core.mv_refresh import refresh_for_tables  # noqa: E402
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

_PIPELINE_PARAM_KEYS = frozenset(
    {
        "feature_params",
        "model_params",
        "label_params",
        "label",
        "auto_promote",
    }
)


def parse_pipeline_params(raw_params: str) -> dict[str, Any]:
    """Parse the exact JSON payload accepted by the managed CLI boundary."""
    try:
        params = json.loads(raw_params)
    except json.JSONDecodeError as exc:
        raise ValueError("Cluster pipeline params must be valid JSON") from exc
    if not isinstance(params, dict):
        raise ValueError("Cluster pipeline params must be a JSON object")

    unsupported = sorted(set(params) - _PIPELINE_PARAM_KEYS)
    if unsupported:
        raise ValueError("Unsupported cluster pipeline parameter(s): " + ", ".join(unsupported))
    for key in ("feature_params", "model_params", "label_params"):
        value = params.get(key)
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"{key} must be an object")
    label = params.get("label")
    if label is not None and (not isinstance(label, str) or not label.strip()):
        raise ValueError("label must be a non-empty string")
    auto_promote = params.get("auto_promote")
    if auto_promote is not None and not isinstance(auto_promote, bool):
        raise ValueError("auto_promote must be a boolean")
    return params


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
        logger.warning(
            "Could not read promoted experiment from DB — using hardcoded defaults", exc_info=True
        )

    return {
        "feature_params": {**DEFAULT_FEATURE_PARAMS},
        "model_params": {**DEFAULT_MODEL_PARAMS},
        "label_params": {**DEFAULT_LABEL_PARAMS},
    }


def _create_pipeline_experiment(
    db: dict[str, Any],
    *,
    scenario_id: str,
    label: str,
    job_id: str | None,
    attempt_token: str | None,
    feature_params: dict[str, Any],
    model_params: dict[str, Any],
    label_params: dict[str, Any],
) -> int:
    """Create the experiment and bind it to one exact managed attempt."""
    import psycopg

    if bool(job_id) != bool(attempt_token):
        raise ValueError("job_id and attempt_token must be provided together")

    with psycopg.connect(**db) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO cluster_experiment
                (scenario_id, label, status, job_id, started_at,
                 feature_params, model_params, label_params)
            VALUES (%s, %s, 'running', %s, NOW(), %s, %s, %s)
            RETURNING experiment_id
            """,
            (
                scenario_id,
                label,
                job_id,
                json.dumps(feature_params),
                json.dumps(model_params),
                json.dumps(label_params),
            ),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("Cluster experiment insert returned no identity")
        experiment_id = int(row[0])

        if job_id and attempt_token:
            lineage = json.dumps(
                {
                    "cluster_experiment_id": experiment_id,
                    "cluster_scenario_id": scenario_id,
                }
            )
            cur.execute(
                """
                UPDATE job_history AS job
                SET params = COALESCE(job.params, '{}'::jsonb) || %s::jsonb
                WHERE job.job_id = %s
                  AND job.attempt_token = %s
                  AND job.status = %s
                """,
                (lineage, job_id, attempt_token, "running"),
            )
            if int(cur.rowcount or 0) != 1:
                raise RuntimeError(
                    "Cluster experiment could not bind to the exact running job attempt"
                )
        conn.commit()
    return experiment_id


def run_unified_pipeline(
    feature_params: dict[str, Any] | None = None,
    model_params: dict[str, Any] | None = None,
    label_params: dict[str, Any] | None = None,
    label: str = "Pipeline Run",
    auto_promote: bool = True,
    job_id: str | None = None,
    attempt_token: str | None = None,
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
    # Resolve params
    effective = get_effective_params()
    fp = feature_params or effective["feature_params"]
    mp = model_params or effective["model_params"]
    lp = label_params or effective["label_params"]

    scenario_id = generate_scenario_id()
    db = get_db_params()

    experiment_id = _create_pipeline_experiment(
        db,
        scenario_id=scenario_id,
        label=label,
        job_id=job_id,
        attempt_token=attempt_token,
        feature_params=fp,
        model_params=mp,
        label_params=lp,
    )
    logger.info("Created experiment #%d (scenario %s)", experiment_id, scenario_id)

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
        if result.get("status") != "completed":
            error = result.get("error") or "unknown clustering error"
            raise RuntimeError(f"Cluster scenario failed: {error}")

        if auto_promote:
            logger.info("Auto-promoting scenario %s ...", scenario_id)
            promote_result = promote_scenario(scenario_id)
            logger.info(
                "Promoted: %d DFUs updated",
                promote_result.get("dfus_updated", 0),
            )
            _finalize_pipeline_experiment(db, experiment_id, promoted=True)
    except Exception:  # noqa: BLE001,RUF100 — lifecycle boundary persists failure.
        runtime = time.time() - t0
        logger.exception("Unified cluster pipeline failed for experiment %d", experiment_id)
        _mark_pipeline_experiment_failed(db, experiment_id, runtime)
        raise

    return {
        **result,
        "experiment_id": experiment_id,
        "promoted": auto_promote,
    }


def _mark_pipeline_experiment_failed(
    db: dict[str, Any],
    experiment_id: int,
    runtime_seconds: float,
) -> None:
    """Best-effort failure reconciliation for the child process boundary."""
    import psycopg

    try:
        with psycopg.connect(**db) as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE cluster_experiment SET status = 'failed', "
                "completed_at = NOW(), runtime_seconds = %s "
                "WHERE experiment_id = %s AND status IN ('queued', 'running')",
                (runtime_seconds, experiment_id),
            )
            conn.commit()
    except psycopg.Error:
        logger.exception(
            "Failed to update cluster experiment %d to failed",
            experiment_id,
        )


def _finalize_pipeline_experiment(
    db: dict[str, Any],
    experiment_id: int,
    promoted: bool,
) -> None:
    """Mark a pipeline experiment 'completed' once clustering + promote are done.

    On a successful promote, demote any previously-promoted row and flag this one.
    """
    import psycopg

    with psycopg.connect(**db) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE cluster_experiment SET status = 'completed', completed_at = NOW() "
            "WHERE experiment_id = %s AND status = 'running'",
            (experiment_id,),
        )
        if int(cur.rowcount or 0) != 1:
            raise RuntimeError(
                f"Cluster experiment {experiment_id} was not running at finalization"
            )
        if promoted:
            cur.execute(
                "UPDATE cluster_experiment SET is_promoted = FALSE, promoted_at = NULL "
                "WHERE is_promoted = TRUE AND experiment_id != %s",
                (experiment_id,),
            )
            cur.execute(
                "UPDATE cluster_experiment SET is_promoted = TRUE, promoted_at = NOW() "
                "WHERE experiment_id = %s",
                (experiment_id,),
            )
            if int(cur.rowcount or 0) != 1:
                raise RuntimeError(f"Cluster experiment {experiment_id} could not be promoted")
        conn.commit()
    if promoted:
        logger.info("Refreshing cluster-assignment-dependent materialized views ...")
        refresh_for_tables(
            ["sku_cluster_assignment"],
            db_params=db,
            include_heavy=False,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run unified clustering pipeline")
    parser.add_argument("--label", default="Pipeline Run", help="Experiment label")
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Skip auto-promotion",
    )
    parser.add_argument("--job-id", help="Managed job_history identity")
    parser.add_argument("--attempt-token", help="Exact managed job attempt token")
    parser.add_argument(
        "--params-json",
        default="{}",
        help="Canonical managed pipeline parameter object",
    )
    args = parser.parse_args(argv)
    if bool(args.job_id) != bool(args.attempt_token):
        parser.error("--job-id and --attempt-token must be provided together")

    try:
        params = parse_pipeline_params(args.params_json)
    except ValueError as exc:
        parser.error(str(exc))

    result = run_unified_pipeline(
        feature_params=params.get("feature_params"),
        model_params=params.get("model_params"),
        label_params=params.get("label_params"),
        label=str(params.get("label") or args.label),
        auto_promote=bool(params.get("auto_promote", not args.no_promote)),
        job_id=args.job_id,
        attempt_token=args.attempt_token,
    )
    logger.info("Result: %s", result.get("status", "unknown"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
