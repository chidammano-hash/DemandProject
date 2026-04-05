#!/usr/bin/env python
"""CLI batch script — AI Planning Agent insight generation.

IPAIfeature1: Run portfolio scan or single-DFU analysis.

Usage:
    # Full portfolio scan
    uv run python scripts/generate_ai_insights.py --portfolio

    # Single DFU
    uv run python scripts/generate_ai_insights.py --item 100320 --loc 1401-BULK

    # Dry run (logs but does not write to DB)
    uv run python scripts/generate_ai_insights.py --portfolio --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml

from common.services.perf_profiler import profiled_section
from common.utils import load_config as _load_config_shared

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("generate_ai_insights")


def _load_config(path: str = "config/ai_planner_config.yaml") -> dict:
    """Load AI planner config via load_config() for _includes support."""
    try:
        cfg_name = os.path.splitext(os.path.basename(path))[0]
        return _load_config_shared(cfg_name)
    except FileNotFoundError:
        log.warning("Config file not found at %s — using defaults", path)
        return {
            "model": "claude-opus-4-6",
            "max_tokens": 4096,
            "portfolio_scan_limit": 100,
        }


def _build_pool():
    """Build a psycopg connection pool using shared db config."""
    import psycopg_pool
    from common.db import get_db_params

    p = get_db_params()
    dsn = (
        f"host={p['host']} port={p['port']} "
        f"dbname={p['dbname']} user={p['user']} "
        f"password={p['password']}"
    )
    pool = psycopg_pool.ConnectionPool(dsn, min_size=1, max_size=4, open=True)
    return pool


def run_portfolio(config: dict, dry_run: bool) -> None:
    """Run a full portfolio scan."""
    scan_run_id = str(uuid.uuid4())
    log.info("Portfolio scan starting  scan_run_id=%s  dry_run=%s", scan_run_id, dry_run)

    if dry_run:
        log.info("[DRY RUN] Would call AIPlannerAgent.run_portfolio_scan()")
        log.info("[DRY RUN] Config: %s", json.dumps(config, indent=2))
        return

    with profiled_section("build_pool"):
        pool = _build_pool()
    from common.ai_planner import AIPlannerAgent

    agent = AIPlannerAgent(pool, config)
    with profiled_section("portfolio_scan"):
        result = agent.run_portfolio_scan(scan_run_id)

    log.info(
        "Portfolio scan complete  scan_run_id=%s  total_insights=%d",
        result.get("scan_run_id", scan_run_id),
        result.get("total_insights", 0),
    )
    log.info("Summary: %s", result.get("summary", ""))
    pool.close()


def run_dfu(item_id: str, loc: str, config: dict, dry_run: bool) -> None:
    """Run single-DFU analysis."""
    scan_run_id = str(uuid.uuid4())
    log.info("DFU analysis: %s @ %s  scan_run_id=%s  dry_run=%s", item_id, loc, scan_run_id, dry_run)

    if dry_run:
        log.info("[DRY RUN] Would call AIPlannerAgent.run_sku_analysis('%s', '%s')", item_id, loc)
        return

    with profiled_section("build_pool"):
        pool = _build_pool()
    from common.ai_planner import AIPlannerAgent

    agent = AIPlannerAgent(pool, config)
    with profiled_section("sku_analysis"):
        insights = agent.run_sku_analysis(item_id, loc, scan_run_id)

    log.info("DFU analysis complete  insights_created=%d", len(insights))
    for ins in insights:
        log.info(
            "  [%s] %s — %s",
            ins.get("severity", "?").upper(),
            ins.get("insight_type", "?"),
            ins.get("summary", ""),
        )
    pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AI planning insights")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--portfolio", action="store_true", help="Run portfolio scan")
    group.add_argument("--item", metavar="ITEM_NO", help="Single DFU item number")
    parser.add_argument("--loc",    metavar="LOC",     help="Single DFU location (required with --item)")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without writing to DB")
    parser.add_argument("--config", default="config/ai_planner_config.yaml", help="Config file path")
    args = parser.parse_args()

    config = _load_config(args.config)

    if args.portfolio:
        run_portfolio(config, dry_run=args.dry_run)
    else:
        if not args.loc:
            parser.error("--loc is required when --item is specified")
        run_dfu(args.item, args.loc, config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
