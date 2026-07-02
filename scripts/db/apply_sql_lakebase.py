#!/usr/bin/env python
"""Apply ``sql/*.sql`` DDL migrations to Lakebase (or any networked Postgres) in order.

Host-side equivalent of ``make db-apply-sql``. That Make target runs ``psql`` *inside*
the Docker Postgres container (``docker compose exec``), so it cannot reach a managed
database like Databricks **Lakebase**. This script connects over the network instead and
applies every ``sql/*.sql`` file in sorted (numeric-prefix) order — the same ordering the
Make target relies on, since the migrations are not individually idempotent across each
other but are individually ``IF NOT EXISTS``-guarded.

Connection
----------
Resolved from the standard ``POSTGRES_*`` env (via ``common.core.db.get_db_params``) or,
if set, ``DATABASE_URL`` (a ``postgres://user:pass@host:port/dbname`` URL — takes
precedence). For **Lakebase token auth**, set the password to a *fresh* OAuth token before
running (a one-shot schema apply finishes well inside the token TTL), e.g.::

    export POSTGRES_PASSWORD="$(databricks auth token --host "$DATABRICKS_HOST" | jq -r .access_token)"
    export POSTGRES_HOST=<instance>.database.cloud.databricks.com POSTGRES_PORT=5432
    export POSTGRES_DB=databricks_postgres POSTGRES_USER=<your-identity> PGSSLMODE=require
    uv run python -m scripts.db.apply_sql_lakebase

Managed-platform notes
----------------------
* ``sql/170_enable_pg_stat_statements.sql`` and the ``CREATE EXTENSION`` lines may be
  role-restricted on Lakebase. Such files are listed in ``TOLERATE_FAIL`` and logged as a
  WARNING instead of aborting the run. Use ``--strict`` to make every failure fatal, or
  ``--continue-on-error`` to tolerate *all* failures (bootstrap/dev only).
* The source dim/fact tables (``dim_item``, ``fact_sales_monthly``, …) are created here as
  *native* tables. When you wire Delta→Lakebase sync, drop and replace those specific
  tables with synced (read-only) tables afterwards — see
  ``docs/specs/01-foundation/09-databricks-lakebase-migration.md`` §1.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import psycopg

from common.core.db import get_db_params
from common.core.paths import SQL_DIR

logger = logging.getLogger(__name__)

# Files whose failure is tolerated by default (extension/role-restricted on managed PG).
# Failing these logs a WARNING and continues rather than aborting the whole schema apply.
TOLERATE_FAIL: frozenset[str] = frozenset({"170_enable_pg_stat_statements.sql"})


def select_files(
    files: list[Path], *, only: str | None = None, start_from: str | None = None
) -> list[Path]:
    """Return ``files`` (already sorted) filtered by an ``only`` substring and/or a
    ``start_from`` numeric prefix (inclusive). Pure — no I/O — so it is unit-testable."""
    selected = files
    if start_from is not None:
        selected = [f for f in selected if f.name >= start_from]
    if only is not None:
        selected = [f for f in selected if only in f.name]
    return selected


def apply_files(
    conn: psycopg.Connection,
    files: list[Path],
    *,
    continue_on_error: bool,
    tolerate: frozenset[str],
    dry_run: bool,
) -> tuple[int, list[str]]:
    """Apply each file in order. Returns ``(applied_count, warnings)``.

    Raises ``psycopg.Error`` on the first fatal failure (a file that is neither in
    ``tolerate`` nor covered by ``continue_on_error``)."""
    applied = 0
    warnings: list[str] = []
    for path in files:
        if dry_run:
            logger.info("would apply %s", path.name)
            applied += 1
            continue
        sql_text = path.read_text(encoding="utf-8")
        try:
            with conn.cursor() as cur:
                cur.execute(sql_text)  # type: ignore[arg-type]
            conn.commit()
            logger.info("applied  %s", path.name)
            applied += 1
        except psycopg.Error as exc:
            conn.rollback()
            tolerated = path.name in tolerate or continue_on_error
            if not tolerated:
                logger.exception("FAILED   %s — aborting", path.name)
                raise
            msg = f"{path.name}: {exc.__class__.__name__} (tolerated)"
            logger.warning("skipped  %s", msg)
            warnings.append(msg)
    return applied, warnings


def _connect(sslmode: str | None) -> psycopg.Connection:
    """Open a connection from ``DATABASE_URL`` or ``POSTGRES_*`` env. Never logs the password."""
    url = os.getenv("DATABASE_URL")
    if url:
        logger.info("connecting via DATABASE_URL")
        return psycopg.connect(url)
    params = get_db_params()
    if sslmode and "sslmode" not in params:
        params["sslmode"] = sslmode
    logger.info(
        "connecting to %s:%s/%s as %s",
        params.get("host"),
        params.get("port"),
        params.get("dbname"),
        params.get("user"),
    )
    return psycopg.connect(**params)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply sql/*.sql DDL to Lakebase in order.")
    parser.add_argument("--only", help="apply only files whose name contains this substring")
    parser.add_argument(
        "--start-from", help="apply files with name >= this (e.g. 100_) — inclusive resume"
    )
    parser.add_argument("--dry-run", action="store_true", help="list what would run; no DB writes")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat every failure as fatal (do not tolerate restricted extensions)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="tolerate ALL file failures (bootstrap/dev only) — implies non-strict",
    )
    parser.add_argument(
        "--sslmode",
        default=os.getenv("PGSSLMODE", "require"),
        help="sslmode when connecting via POSTGRES_* env (default: require; Lakebase needs SSL)",
    )
    args = parser.parse_args(argv)

    files = sorted(SQL_DIR.glob("*.sql"))
    if not files:
        logger.error("no sql/*.sql files found under %s", SQL_DIR)
        return 1
    selected = select_files(files, only=args.only, start_from=args.start_from)
    if not selected:
        logger.error("no files matched the given filters")
        return 1
    tolerate: frozenset[str] = frozenset() if args.strict else TOLERATE_FAIL

    logger.info("applying %d of %d sql files (sorted order)", len(selected), len(files))
    if args.dry_run:
        apply_files(
            conn=None,  # type: ignore[arg-type]
            files=selected,
            continue_on_error=args.continue_on_error,
            tolerate=tolerate,
            dry_run=True,
        )
        logger.info("dry-run complete — %d files would be applied", len(selected))
        return 0

    conn = _connect(args.sslmode)
    try:
        applied, warnings = apply_files(
            conn,
            selected,
            continue_on_error=args.continue_on_error,
            tolerate=tolerate,
            dry_run=False,
        )
    except psycopg.Error:
        return 1
    finally:
        conn.close()

    logger.info("applied %d sql files; %d skipped/tolerated", applied, len(warnings))
    for w in warnings:
        logger.warning("  tolerated: %s", w)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    sys.exit(main())
