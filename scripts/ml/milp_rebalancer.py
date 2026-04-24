"""CLI entry point for the MILP transfer/replenishment rebalancer.

Gen-4 Roadmap AI-8 (Stream H Phase 2). Reads open exceptions from
``fact_replenishment_exceptions`` and spare-stock candidates from
``fact_inventory_snapshot`` (or a YAML-driven subset), runs the MILP
solver from :mod:`common.ml.milp`, and prints a human-readable plan.

The real solver will use PuLP + highspy once those packages are on the
allow-list; for now the command falls back to the greedy solver in
``common.ml.milp.solve_rebalance``.

Usage::

    python -m scripts.ml.milp_rebalancer --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from common.ml.milp import (
    ExceptionProblem,
    TransferCandidate,
    solve_rebalance,
)

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MILP transfer/replenishment rebalancer")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the plan but do not write recommendations back to DB.",
    )
    parser.add_argument(
        "--emergency-po-unit-cost",
        type=float,
        default=1.0,
        help="Per-unit cost for emergency-PO fallback (default 1.0).",
    )
    return parser.parse_args(argv)


def _load_inputs(_conn: Any) -> tuple[list[ExceptionProblem], list[TransferCandidate]]:
    """Stub loader. TODO(gen-4 AI-8): wire to fact_replenishment_exceptions
    and fact_inventory_snapshot to populate exception + transfer-candidate
    lists. For now returns empty lists so the CLI is callable end-to-end.
    """
    return [], []


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logger.info("milp_rebalancer start dry_run=%s", args.dry_run)

    # TODO(gen-4 AI-8): replace None with real connection once _load_inputs
    # hits Postgres.
    exceptions, transfer_pool = _load_inputs(None)
    solution = solve_rebalance(
        exceptions,
        transfer_pool,
        emergency_po_unit_cost=args.emergency_po_unit_cost,
    )

    logger.info(
        "milp_rebalancer solution solver=%s transfers=%d pos=%d total_cost=%.2f",
        solution.solver,
        len(solution.transfers),
        len(solution.emergency_pos),
        solution.total_cost,
    )
    if args.dry_run:
        logger.info("--dry-run set; not writing recommendations.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(run())
