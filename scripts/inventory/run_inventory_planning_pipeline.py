"""End-to-End Inventory Planning Pipeline.

Orchestrates the full inventory planning chain in sequence:
  1. Safety Stock → fact_safety_stock_targets
  2. EOQ → fact_eoq_targets
  3. Replenishment Plan → fact_replenishment_plan
  4. Planned Orders → fact_planned_orders
  5. Exceptions → fact_replenishment_exceptions

Each step calls the corresponding script via subprocess. The pipeline
reads demand inputs from the production forecast (promoted or staging).

Usage:
    uv run python scripts/run_inventory_planning_pipeline.py
    uv run python scripts/run_inventory_planning_pipeline.py --steps safety_stock,eoq
    uv run python scripts/run_inventory_planning_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.services.perf_profiler import profiled_section

logger = logging.getLogger(__name__)

_UV = shutil.which("uv") or "uv"


# ---------------------------------------------------------------------------
# Pipeline step definitions
# ---------------------------------------------------------------------------

STEPS = [
    {
        "id": "safety_stock",
        "label": "Safety Stock",
        "script": "scripts/compute_safety_stock.py",
        "args": [],
        "output_table": "fact_safety_stock_targets",
    },
    {
        "id": "eoq",
        "label": "EOQ & Cycle Stock",
        "script": "scripts/compute_eoq.py",
        "args": [],
        "output_table": "fact_eoq_targets",
    },
    {
        "id": "replenishment_plan",
        "label": "Replenishment Plan",
        "script": "scripts/compute_replenishment_plan.py",
        "args": [],
        "output_table": "fact_replenishment_plan",
    },
    {
        "id": "planned_orders",
        "label": "Planned Orders",
        "script": "scripts/generate_planned_orders.py",
        "args": [],
        "output_table": "fact_planned_orders",
    },
    {
        "id": "exceptions",
        "label": "Exceptions",
        "script": "scripts/generate_replenishment_exceptions.py",
        "args": [],
        "output_table": "fact_replenishment_exceptions",
    },
]

ALL_STEP_IDS = [s["id"] for s in STEPS]


# ---------------------------------------------------------------------------
# Run a single step
# ---------------------------------------------------------------------------


def run_step(
    step: dict,
    dry_run: bool = False,
    step_number: int = 1,
    total_steps: int = 1,
) -> dict:
    """Run a single pipeline step via subprocess.

    Returns dict with step_id, success, duration_s, output.
    """
    script_path = ROOT / step["script"]
    if not script_path.exists():
        logger.warning("Script not found: %s — skipping %s", script_path, step["id"])
        return {"step_id": step["id"], "success": False, "duration_s": 0, "output": "Script not found"}

    cmd = [_UV, "run", "python", str(script_path)] + step["args"]
    if dry_run:
        cmd.append("--dry-run")

    logger.info("Running step [%d/%d]: %s", step_number, total_steps, step["label"])
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=600,  # 10 min per step max
        )
        duration = time.time() - t0

        if result.returncode != 0:
            logger.error(
                "Step %s FAILED (exit %d, %.0fs):\n%s",
                step["id"], result.returncode, duration,
                (result.stderr or result.stdout or "(no output)")[-2000:],
            )
            return {
                "step_id": step["id"],
                "success": False,
                "duration_s": round(duration, 1),
                "output": (result.stderr or result.stdout or "")[-2000:],
            }

        logger.info("Step %s completed in %.0fs", step["id"], duration)
        return {
            "step_id": step["id"],
            "success": True,
            "duration_s": round(duration, 1),
            "output": result.stdout[-200:] if result.stdout else "",
        }

    except subprocess.TimeoutExpired:
        duration = time.time() - t0
        logger.error("Step %s TIMED OUT after %.0fs", step["id"], duration)
        return {
            "step_id": step["id"],
            "success": False,
            "duration_s": round(duration, 1),
            "output": "Timed out after 600s",
        }
    except Exception as exc:
        duration = time.time() - t0
        logger.exception("Step %s raised exception", step["id"])
        return {
            "step_id": step["id"],
            "success": False,
            "duration_s": round(duration, 1),
            "output": str(exc),
        }


# ---------------------------------------------------------------------------
# Post-pipeline MV refresh
# ---------------------------------------------------------------------------


def _refresh_integrated_mv() -> None:
    """Refresh the integrated targets MV after SS + EOQ are computed."""
    import psycopg

    from common.db import get_db_params

    try:
        with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
            cur.execute(
                "REFRESH MATERIALIZED VIEW CONCURRENTLY mv_integrated_planning_targets"
            )
            conn.commit()
        logger.info("Refreshed mv_integrated_planning_targets")
    except psycopg.Error as exc:
        logger.warning("Failed to refresh integrated targets MV: %s", exc)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_pipeline(
    step_ids: list[str] | None = None,
    dry_run: bool = False,
    progress_cb=None,
    halt_on_failure: bool = False,
) -> dict:
    """Run the full inventory planning pipeline.

    Args:
        step_ids: Subset of steps to run (None = all).
        dry_run: Preview without writing to DB.
        progress_cb: Optional callback for progress reporting.
        halt_on_failure: Stop pipeline on first step failure.

    Returns:
        Summary dict with per-step results.
    """
    steps_to_run = [s for s in STEPS if step_ids is None or s["id"] in step_ids]

    if not steps_to_run:
        logger.warning("No valid steps to run. Valid IDs: %s", ALL_STEP_IDS)
        return {"success": False, "steps": [], "error": "No valid steps"}

    # Validate all step scripts exist before starting
    for step in steps_to_run:
        script_path = ROOT / step["script"]
        if not script_path.exists():
            logger.error("Script not found: %s — aborting pipeline", script_path)
            return {
                "success": False,
                "steps": [],
                "error": f"Missing script: {step['script']}",
            }

    logger.info(
        "Inventory Planning Pipeline — %d steps: %s",
        len(steps_to_run),
        ", ".join(s["id"] for s in steps_to_run),
    )
    if dry_run:
        logger.info("DRY RUN — no data will be written")

    t_start = time.time()
    results = []
    completed = []
    failed = []

    for i, step in enumerate(steps_to_run):
        pct = int((i / len(steps_to_run)) * 100)
        if progress_cb:
            progress_cb(pct=pct, msg=f"Running {step['label']}...")

        with profiled_section(f"pipeline_{step['id']}"):
            result = run_step(step, dry_run=dry_run, step_number=i + 1, total_steps=len(steps_to_run))

        results.append(result)
        if result["success"]:
            completed.append(step["id"])
        else:
            failed.append(step["id"])
            if halt_on_failure:
                logger.error("Step %s failed — halting pipeline (--halt-on-failure)", step["id"])
                break
            logger.error("Step %s failed — continuing with remaining steps", step["id"])

    # Refresh integrated targets MV after SS + EOQ steps complete
    if not dry_run and ("safety_stock" in completed or "eoq" in completed):
        _refresh_integrated_mv()

    total_duration = time.time() - t_start

    if progress_cb:
        progress_cb(pct=100, msg=f"Pipeline complete in {total_duration:.0f}s")

    summary = {
        "success": len(failed) == 0,
        "total_duration_s": round(total_duration, 1),
        "steps_completed": completed,
        "steps_failed": failed,
        "steps": results,
    }

    logger.info(
        "Pipeline complete in %.0fs — %d/%d steps succeeded%s",
        total_duration,
        len(completed),
        len(steps_to_run),
        f" (failed: {', '.join(failed)})" if failed else "",
    )

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end inventory planning pipeline"
    )
    parser.add_argument(
        "--steps",
        default=None,
        help=f"Comma-separated step IDs to run (default: all). Valid: {','.join(ALL_STEP_IDS)}",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without DB writes")
    parser.add_argument(
        "--halt-on-failure", action="store_true",
        help="Stop pipeline on first step failure (default: continue)",
    )
    args = parser.parse_args()

    step_ids = args.steps.split(",") if args.steps else None

    if step_ids:
        invalid = set(step_ids) - set(ALL_STEP_IDS)
        if invalid:
            parser.error(f"Invalid step IDs: {invalid}. Valid: {ALL_STEP_IDS}")

    summary = run_pipeline(
        step_ids=step_ids,
        dry_run=args.dry_run,
        halt_on_failure=args.halt_on_failure,
    )

    if not summary["success"]:
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()
