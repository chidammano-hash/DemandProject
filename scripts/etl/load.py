"""Unified ETL load dispatcher — one entry point, three modes.

Subsumes 3 existing load patterns:
  * Generic domain loader  -> load_dataset_postgres.py
  * Partitioned (customer_demand --month) -> load_customer_demand_postgres.py
  * Partitioned file-glob (inventory) -> load_dataset_postgres.py with --replace

Modes:
  onetime  Full TRUNCATE + INSERT (subprocess loader with --replace).
  delta    Hash-detect via audit_load_batch; skip if unchanged, else load
           without --replace (default ON CONFLICT upsert).
  file     Per-slice or per-file reload. Requires --slice for partitioned
           domains. For non-customer-demand partitioned domains, deletes
           the slice's row range first, then loads.

Usage:
    python -m scripts.etl.load --domain sales --mode onetime
    python -m scripts.etl.load --domain sales --mode delta
    python -m scripts.etl.load --domain inventory --mode file --slice 2026_03
    python -m scripts.etl.load --domain customer_demand --mode file --slice 2026-03
    python -m scripts.etl.load --domain all --mode delta

Output (one JSON line per processed domain):
    {"domain":"sales","mode":"delta","slice":null,"rows_loaded":12345,
     "duration_s":3.2,"status":"success","error":null,
     "started_at":"...","completed_at":"..."}

Exit codes:
    0  success (all domains succeeded or skipped)
    1  failure (one or more domains failed)
    2  delta no-change (single-domain runs only)
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.db import get_db_params
from common.core.domain_partition import (
    get_partition,
    is_partitioned,
    slice_to_date_range,
)
from common.core.etl_helpers import (
    create_monthly_partition,
    delete_partition_range,
    is_pg_partitioned,
    monthly_partition_name,
)
from common.engines.medallion import file_hash

logger = logging.getLogger(__name__)

ALLOWED_MODES = ("onetime", "delta", "file")

# Slice-delete table/column are derived from the domain's DomainSpec.table and
# its DomainPartition.field (common/core/domain_partition.py) — no separate map.
# customer_demand is handled by its own loader's --month flag (partition drop+recreate).

CUSTOMER_DEMAND = "customer_demand"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_iso() -> str:
    """ISO-8601 UTC timestamp, no microseconds, with trailing 'Z'."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _emit(payload: dict[str, Any]) -> None:
    """Write one JSON record to stdout."""
    print(json.dumps(payload, default=str))


def _load_etl_config() -> dict | None:
    """Try config/etl_config.yaml then config/etl/etl_config.yaml. Return None if missing."""
    try:
        from common.core.utils import load_config
    except ImportError:
        logger.warning("common.core.utils.load_config unavailable")
        return None
    for name in ("etl_config.yaml", "etl/etl_config.yaml"):
        try:
            cfg = load_config(name)
            if cfg:
                return cfg
        except (FileNotFoundError, OSError, ValueError) as exc:
            logger.debug("config %s not loadable: %s", name, exc)
    logger.warning("etl_config.yaml not found in either location")
    return None


def _domain_order(cfg: dict | None) -> list[str]:
    """Return configured domain_order list, or a default 11-domain order."""
    if cfg and isinstance(cfg.get("domain_order"), list):
        return list(cfg["domain_order"])
    return [
        "item", "location", "customer", "time", "sku",
        "sales", "forecast", "inventory", "customer_demand",
        "sourcing", "purchase_order",
    ]


def _resolve_source_csv(domain: str) -> Path | None:
    """Locate the normalized clean CSV path for a domain.

    Customer_demand and the standard domains use data/<clean_file>; for inventory
    the loader still consumes data/staged/inventory_clean.csv. Returns None if unknown.
    """
    try:
        from common.core.domain_specs import get_spec
    except ImportError:
        logger.error("common.core.domain_specs.get_spec unavailable")
        return None
    try:
        spec = get_spec(domain)
    except (KeyError, ValueError):
        # customer_demand may not be in DOMAIN_SPECS for older versions; fall back.
        if domain == CUSTOMER_DEMAND:
            return ROOT / "data" / "staged" / "customer_demand_clean.csv"
        return None
    return ROOT / "data" / spec.clean_file


def _fetch_last_hash(domain: str) -> str | None:
    """Look up the most recent completed batch's source_hash for the domain."""
    try:
        with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT source_hash FROM audit_load_batch "
                "WHERE domain = %s AND status = 'completed' "
                "ORDER BY completed_at DESC LIMIT 1",
                [domain],
            )
            row = cur.fetchone()
            return row[0] if row else None
    except psycopg.Error as exc:
        logger.warning("audit_load_batch lookup failed for %s: %s", domain, exc)
        return None


def _delete_slice_rows(domain: str, slice_str: str) -> int:
    """For partitioned non-customer-demand domains, DELETE rows in the slice range."""
    from common.core.domain_specs import get_spec

    part = get_partition(domain)
    if part is None:
        raise ValueError(f"{domain} is not partitioned")
    table = get_spec(domain).table
    start_d, end_d = slice_to_date_range(slice_str, part.format)
    with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
        deleted = delete_partition_range(cur, table, part.field, start_d, end_d)
        conn.commit()
    logger.info("Deleted %d rows from %s for slice %s", deleted, table, slice_str)
    return deleted


def _refresh_mvs(domain: str) -> None:
    """Refresh every MV depending on the tables the domain load wrote.

    The dependent set comes from the central dependency map
    (common/core/mv_refresh.py) keyed on the domain's written tables —
    including post-load hook tables — instead of a per-domain config list.
    """
    from common.core.etl_helpers import tables_written_for_domain
    from common.core.mv_refresh import refresh_for_tables

    try:
        tables = tables_written_for_domain(domain)
    except ValueError:
        logger.warning("Skipping MV refresh — unknown domain %r", domain)
        return
    try:
        refresh_for_tables(tables)
    except psycopg.Error as exc:
        logger.warning("MV refresh connection failed: %s", exc)


# ---------------------------------------------------------------------------
# Subprocess wrappers around existing loaders
# ---------------------------------------------------------------------------

def _run_loader(cmd: list[str], dry_run: bool) -> None:
    """Invoke an existing loader as a subprocess. Raises CalledProcessError on failure."""
    if dry_run:
        logger.info("DRY-RUN cmd: %s", " ".join(cmd))
        return
    logger.info("EXEC: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _generic_load_cmd(domain: str, replace: bool) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "etl" / "load_dataset_postgres.py"),
        "--dataset", domain,
    ] + (["--replace"] if replace else [])


def _normalize_needed(domain: str) -> bool:
    """True if the raw input is newer than the cleaned output (or clean missing).

    Skips re-normalize when nothing has changed — avoids rewriting the 100 MB
    cleaned CSV on every UI submit. Conservative: returns True on any uncertainty
    (missing raw spec, time domain which has no input file, etc.) so a
    re-normalize still happens when in doubt.
    """
    try:
        from common.core.domain_specs import get_spec
        spec = get_spec(domain)
    except (ImportError, KeyError, ValueError):
        return True
    raw = ROOT / "data" / "input" / spec.source_file
    clean = ROOT / "data" / spec.clean_file
    if not raw.exists() or not clean.exists():
        return True
    return raw.stat().st_mtime > clean.stat().st_mtime


def _normalize_domain(domain: str, dry_run: bool) -> None:
    """Re-run normalize so the cleaned CSV matches the current raw input.

    Skips when the raw input mtime is older than the cleaned output (nothing
    to do). Mirrors run_pipeline.normalize_domain / normalize_inventory for the
    cases where work is actually required.
    """
    if domain not in ("inventory", CUSTOMER_DEMAND) and not _normalize_needed(domain):
        logger.info("normalize: %s up-to-date — skipping", domain)
        return
    # Invoke as -m so 'common' resolves via the package path (cwd=ROOT).
    if domain == "inventory":
        cmd = [sys.executable, "-m", "scripts.etl.normalize_inventory_csv"]
    elif domain == CUSTOMER_DEMAND:
        cmd = [sys.executable, "-m", "scripts.etl.normalize_customer_demand_csv"]
    else:
        cmd = [sys.executable, "-m", "scripts.etl.normalize_dataset_csv",
               "--dataset", domain]
    logger.info("normalize: %s -> %s", domain, " ".join(cmd))
    if dry_run:
        return
    result = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr
        )


def _customer_demand_cmd(
    *, replace: bool = False, month: str | None = None,
    file: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "etl" / "load_customer_demand_postgres.py"),
    ]
    if file:
        cmd.extend(["--file", file])
    if month:
        cmd.extend(["--month", month])
    if replace:
        cmd.append("--replace")
    return cmd


# ---------------------------------------------------------------------------
# Mode dispatchers — one per (mode, customer_demand?) combination
# ---------------------------------------------------------------------------

def _do_onetime(domain: str, dry_run: bool) -> None:
    if domain == CUSTOMER_DEMAND:
        _run_loader(_customer_demand_cmd(replace=True), dry_run)
    else:
        _run_loader(_generic_load_cmd(domain, replace=True), dry_run)


def _do_delta(domain: str, dry_run: bool) -> str | dict[str, Any]:
    """Hash-detect; if unchanged, return 'skipped'. Otherwise upsert (no truncate).

    For non-partitioned dim tables we use _safe_upsert() (COPY + ON CONFLICT),
    which never TRUNCATEs the target and never CASCADEs to fact tables. For
    partitioned/customer_demand domains we fall back to the existing loader
    path (which is partition-aware and non-destructive at the parent level).

    Returns 'skipped' when the source is unchanged, or a dict with
    {'status':'success', 'inserted':N, 'updated':N, 'deleted':N} when a real
    upsert happened (counts come straight from RETURNING, not a snapshot diff).
    """
    if domain == CUSTOMER_DEMAND:
        changed_files = _multi_file_changed_files(domain)
        if not changed_files:
            logger.info("delta: customer_demand unchanged — skipping")
            return "skipped"
        logger.info("delta: customer_demand has %d changed file(s)", len(changed_files))
        _normalize_domain(domain, dry_run)
        if dry_run:
            return "success"
        _run_loader(_customer_demand_cmd(), dry_run=False)
        _record_multi_file_hashes(domain, changed_files)
        return "success"

    # Multi-file domains — detection + audit
    # recording is per-file so the scanner correctly reports which files
    # changed and stops perma-flagging the domain after a successful load.
    if _multi_file_glob_pattern(domain):
        changed_files = _multi_file_changed_files(domain)
        if not changed_files:
            logger.info("delta: %s unchanged (all source files match) — skipping", domain)
            return "skipped"
        logger.info("delta: %s has %d changed file(s)", domain, len(changed_files))
        if dry_run:
            return "success"
        _normalize_domain(domain, dry_run)
        try:
            counts = _safe_upsert(domain)
            logger.info(
                "delta: %s upserted (inserted=%d, updated=%d, deleted=%d)",
                domain, counts["inserted"], counts["updated"], counts["deleted"],
            )
            _record_multi_file_hashes(domain, changed_files)
            return {"status": "success", **counts}
        except (psycopg.Error, ValueError, FileNotFoundError, ImportError):
            logger.exception("safe upsert failed for %s", domain)
            raise

    # Hash the RAW input file (data/input/<source_file>) — not the cleaned
    # CSV. The scanner hashes the raw file too; comparing apples to apples
    # makes the "Detect Changes" UI honest. Avoids re-normalizing when the
    # raw input hasn't changed at all.
    raw_path = _resolve_raw_input(domain)
    if raw_path is None or not raw_path.exists():
        logger.warning("Raw input not found for %s (path=%s) — proceeding to load",
                       domain, raw_path)
        current_hash = ""
    else:
        current_hash = file_hash(raw_path)

    last_hash = _fetch_last_hash(domain)
    if current_hash and last_hash and current_hash == last_hash:
        if domain != "sales" or _sales_source_is_synchronized():
            logger.info("delta: %s unchanged (hash=%s) — skipping", domain, current_hash[:12])
            return "skipped"

    # Source changed — re-normalize so the cleaned CSV reflects current raw,
    # then run the safe upsert.
    _normalize_domain(domain, dry_run)

    if dry_run:
        logger.info("DRY-RUN: would delta-upsert %s", domain)
        return "success"

    if domain == "sales":
        # Sales is a strict dual-track source: the canonical loader replaces
        # current and immutable-original facts in one transaction and records
        # the same batch hash. A generic upsert could advance audit lineage
        # while leaving the raw model-training mirror stale.
        from common.core.domain_specs import get_spec
        from scripts.etl.load_dataset_postgres import load_domain

        spec = get_spec(domain)
        result = load_domain(
            spec,
            ROOT / "data" / spec.clean_file,
            source_hash_override=current_hash or None,
        )
        loaded = int(result.get("rows_loaded", 0))
        return {
            "status": "success",
            "inserted": loaded,
            "updated": 0,
            "deleted": 0,
        }

    # Remaining domains route through the safe non-destructive upsert path.
    # customer_demand previously used its dedicated loader, but that loader
    # raises CardinalityViolation when source has duplicate (demand_ck,
    # startdate) rows (~0.3% of feeds). _safe_upsert dedupes via DISTINCT ON.
    try:
        counts = _safe_upsert(domain)
        logger.info(
            "delta: %s upserted (inserted=%d, updated=%d, deleted=%d)",
            domain, counts["inserted"], counts["updated"], counts["deleted"],
        )
        _record_audit_hash(
            domain, current_hash,
            counts["inserted"] + counts["updated"],
        )
        return {"status": "success", **counts}
    except (psycopg.Error, ValueError, FileNotFoundError, ImportError):
        logger.exception(
            "safe upsert failed for %s; NOT falling back to destructive loader",
            domain,
        )
        raise


def _multi_file_glob_pattern(domain: str) -> str | None:
    """Return the glob pattern for a multi-file domain (None if single-file)."""
    try:
        from common.core.domain_specs import get_spec
        spec = get_spec(domain)
    except (ImportError, KeyError, ValueError):
        return None
    src = getattr(spec, "source_file", None) or ""
    return src if "*" in src else None


def _multi_file_source_files(domain: str) -> list:
    """Return list of source files for a glob-driven domain (sorted)."""
    import glob as _glob
    pattern = _multi_file_glob_pattern(domain)
    if not pattern:
        return []
    full = str(ROOT / "data" / "input" / pattern)
    return sorted(__import__("pathlib").Path(p) for p in _glob.glob(full))


def _multi_file_changed_files(domain: str) -> list[tuple]:
    """Return [(Path, current_raw_hash)] for glob-domain files differing from audit.

    Empty list = everything up-to-date. On DB error, returns ALL files as
    "changed" so the load runs (safe default).
    """
    files = _multi_file_source_files(domain)
    if not files:
        return []
    changed = []
    try:
        with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
            if domain == CUSTOMER_DEMAND:
                cur.execute(
                    """SELECT batch.metadata -> 'source_file_hashes'
                       FROM customer_demand_profile_refresh_state state
                       JOIN audit_load_batch batch
                         ON batch.batch_id = state.source_batch_id
                       WHERE state.singleton_id = 1
                         AND batch.domain = 'customer_demand'
                         AND batch.status = 'completed'"""
                )
                row = cur.fetchone()
                stored_hashes = dict(row[0]) if row and isinstance(row[0], dict) else {}
                for fp in files:
                    current = file_hash(fp)
                    if stored_hashes.get(fp.name) != current:
                        changed.append((fp, current))
                return changed
            for fp in files:
                current = file_hash(fp)
                cur.execute(
                    "SELECT source_hash FROM audit_load_batch "
                    "WHERE domain=%s AND source_file=%s "
                    "AND status='completed' ORDER BY completed_at DESC LIMIT 1",
                    [domain, fp.name],
                )
                row = cur.fetchone()
                if (row[0] if row else None) != current:
                    changed.append((fp, current))
    except psycopg.Error as exc:
        logger.warning("%s change detection failed: %s — assuming all changed", domain, exc)
        return [(fp, file_hash(fp)) for fp in files]
    return changed


def _record_multi_file_hashes(domain: str, files_with_hashes: list[tuple]) -> None:
    """Write one audit_load_batch row per changed file in a glob domain.

    Matches the per-file lookup the scanner does, so subsequent scans see
    matching hashes and stop perma-flagging the domain as changed.
    """
    if not files_with_hashes:
        return
    try:
        with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
            if domain == CUSTOMER_DEMAND:
                source_hashes = {fp.name: value for fp, value in files_with_hashes}
                cur.execute(
                    """UPDATE audit_load_batch batch
                       SET metadata = jsonb_set(
                           COALESCE(batch.metadata, '{}'::jsonb),
                           '{source_file_hashes}',
                           COALESCE(batch.metadata -> 'source_file_hashes', '{}'::jsonb)
                               || %s::jsonb,
                           TRUE
                       )
                       FROM customer_demand_profile_refresh_state state
                       WHERE state.singleton_id = 1
                         AND state.source_batch_id = batch.batch_id
                         AND batch.domain = 'customer_demand'
                         AND batch.status = 'completed'""",
                    (json.dumps(source_hashes, sort_keys=True),),
                )
                conn.commit()
                return
            for fp, h in files_with_hashes:
                cur.execute(
                    "INSERT INTO audit_load_batch "
                    "(domain, layer, source_file, source_hash, status, "
                    " row_count_in, row_count_out, started_at, completed_at) "
                    "VALUES (%s, 'direct', %s, %s, 'completed', "
                    " 0, 0, NOW(), NOW())",
                    [domain, fp.name, h],
                )
            conn.commit()
    except psycopg.Error as exc:
        logger.warning("failed to record %s per-file hashes: %s", domain, exc)


# Backward-compat aliases (keep external callers working).
_inventory_source_files = lambda: _multi_file_source_files("inventory")  # noqa: E731
_inventory_changed_files = lambda: _multi_file_changed_files("inventory")  # noqa: E731
def _record_inventory_per_file_hashes(files_with_hashes: list[tuple]) -> None:
    _record_multi_file_hashes("inventory", files_with_hashes)


def _resolve_raw_input(domain: str):
    """Return the RAW input path for the domain (data/input/<source_file>).

    Returns None for domains without a single canonical raw input (inventory's
    multi-file glob, time's auto-generated stub).
    """
    try:
        from common.core.domain_specs import get_spec
        spec = get_spec(domain)
    except (ImportError, KeyError, ValueError):
        return None
    src = getattr(spec, "source_file", None)
    if not src or "*" in src or src.startswith("_generated"):
        return None
    return ROOT / "data" / "input" / src


def _sales_source_is_synchronized() -> bool:
    """Return whether the latest sales audit and immutable mirror are coherent."""
    from common.services.forecast_population import resolve_forecast_sales_table

    try:
        with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
            resolve_forecast_sales_table(cur)
    except (psycopg.Error, RuntimeError):
        logger.warning(
            "Sales hash is unchanged but forecast lineage is not synchronized; "
            "forcing a canonical dual-track reload",
            exc_info=True,
        )
        return False
    return True


def _record_audit_hash(domain: str, source_hash: str, row_count: int) -> None:
    """Record a synthetic audit_load_batch entry so future delta runs see the
    hash. Mirrors what the standard loader writes via medallion.complete_batch.
    """
    if not source_hash:
        return
    try:
        with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO audit_load_batch "
                "(domain, layer, source_file, source_hash, status, "
                " row_count_in, row_count_out, started_at, completed_at) "
                "VALUES (%s, 'direct', %s, %s, 'completed', %s, %s, NOW(), NOW())",
                (domain, "safe_upsert", source_hash, row_count, row_count),
            )
            conn.commit()
    except psycopg.Error as exc:
        logger.warning("failed to record audit hash for %s: %s", domain, exc)


def _do_file(
    domain: str, slice_str: str | None, file_arg: str | None, dry_run: bool,
) -> str | dict[str, Any]:
    """Per-slice or per-file reload.

    Slice path (the only useful one — the API blocks file mode for non-partitioned
    domains): stage the cleaned CSV, filter to the slice's date range, DELETE the
    existing slice rows, then INSERT the staged subset. Returns insert/delete
    counts straight from RETURNING.
    """
    partitioned = is_partitioned(domain)
    if partitioned and not slice_str and not file_arg:
        raise ValueError(
            f"--slice is required for mode=file when domain={domain!r} is partitioned"
        )

    if slice_str:
        spec = get_partition(domain)
        if spec is None:
            raise ValueError(f"{domain} is not partitioned — --slice not allowed")
        slice_to_date_range(slice_str, spec.format)

        # Re-normalize first so the cleaned CSV reflects any raw-input changes
        # the user made (e.g. dropped a fresh Inventory_Snapshot_YYYY_MM.csv).
        # Without this, _safe_replace_slice reads stale staged data and the
        # DELETE+INSERT round-trip looks like a no-op.
        _normalize_domain(domain, dry_run)

        if domain == CUSTOMER_DEMAND:
            _run_loader(_customer_demand_cmd(month=slice_str), dry_run)
            return "success"

        if dry_run:
            logger.info("DRY-RUN: would replace slice %s of %s", slice_str, domain)
            return "success"
        return _safe_replace_slice(domain, slice_str)

    # No slice — explicit single-file load (rarely useful; legacy code path).
    if file_arg is None:
        raise ValueError("mode=file requires either --slice or --file for this domain")
    if domain == CUSTOMER_DEMAND:
        _run_loader(_customer_demand_cmd(file=file_arg), dry_run)
        return "success"
    logger.warning(
        "load_dataset_postgres.py has no --file flag; ignoring --file %s "
        "and loading canonical CSV for %s", file_arg, domain,
    )
    _run_loader(_generic_load_cmd(domain, replace=False), dry_run)
    return "success"


def _safe_replace_slice(domain: str, slice_str: str) -> dict[str, Any]:
    """Replace a single slice (e.g. one month) in a partitioned-fact target.

    DELETE rows in the slice's date range, then INSERT only the matching rows
    from the cleaned CSV. Returns ``{status, inserted, updated, deleted}``.
    Updated is always 0 for slice replace (it's a wipe-and-reload of the slice).
    """
    import csv as csv_mod

    from common.core.domain_specs import get_spec

    spec = get_spec(domain)
    table = spec.table
    psp = get_partition(domain)
    if psp is None:
        raise ValueError(f"{domain} is not partitioned")
    start_d, end_d = slice_to_date_range(slice_str, psp.format)
    partition_col = psp.field
    csv_path = ROOT / "data" / spec.clean_file
    if not csv_path.exists():
        raise FileNotFoundError(f"cleaned CSV not found: {csv_path}")

    with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = %s AND table_schema = 'public' "
            "ORDER BY ordinal_position",
            (table,),
        )
        target_cols = {r[0]: r[1] for r in cur.fetchall()}
        with open(csv_path, encoding="utf-8") as f:
            csv_header = next(csv_mod.reader(f))
        common = [c for c in csv_header if c in target_cols]
        if partition_col not in csv_header:
            raise ValueError(
                f"partition column {partition_col!r} missing from {csv_path.name}"
            )

        ck_field = getattr(spec, "ck_field", None)
        bk_cols = _resolve_business_key(domain)
        ck_expr: str | None = None
        if ck_field and ck_field in target_cols and ck_field not in common:
            sep = (getattr(spec, "business_key_separator", "-") or "-").replace("'", "''")
            parts = [f'trim(s."{c}")' for c in bk_cols]
            ck_expr = parts[0] if len(parts) == 1 else f" || '{sep}' || ".join(parts)

        def cast_expr(col: str) -> str:
            t = target_cols[col]
            if t in ("integer", "bigint", "smallint",
                     "numeric", "double precision", "real",
                     "date", "timestamp without time zone",
                     "timestamp with time zone", "boolean"):
                return (
                    f'NULLIF(NULLIF(NULLIF(NULLIF(s."{col}", \'\'), \'null\'), '
                    f'\'none\'), \'NA\')::{t}'
                )
            return f's."{col}"'

        stg_cols = ", ".join(f'"{c}" TEXT' for c in csv_header)
        cur.execute(f"CREATE TEMP TABLE _slice_stg ({stg_cols})")
        with open(csv_path, encoding="utf-8") as f:
            with cur.copy(
                "COPY _slice_stg FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')"
            ) as cp:
                while chunk := f.read(8192):
                    cp.write(chunk)

        # Filter staging to just the slice's range
        cur.execute(
            f'CREATE TEMP TABLE _slice_filtered AS '
            f'SELECT * FROM _slice_stg s '
            f'WHERE s."{partition_col}"::date >= %s AND s."{partition_col}"::date < %s',
            (start_d, end_d),
        )

        if _is_pg_partitioned(cur, table):
            _ensure_monthly_partitions(cur, table, partition_col, "_slice_filtered")

        # DELETE existing slice rows
        cur.execute(
            f'DELETE FROM "{table}" WHERE "{partition_col}" >= %s AND "{partition_col}" < %s',
            (start_d, end_d),
        )
        deleted = cur.rowcount

        insert_cols = ([ck_field] if ck_expr else []) + common
        insert_cols_quoted = ", ".join(f'"{c}"' for c in insert_cols)
        select_pieces = ([ck_expr] if ck_expr else []) + [cast_expr(c) for c in common]
        select_exprs = ", ".join(select_pieces)
        cur.execute(
            f'INSERT INTO "{table}" ({insert_cols_quoted}) '
            f'SELECT {select_exprs} FROM _slice_filtered s'
        )
        inserted = cur.rowcount
        conn.commit()

    logger.info(
        "file: %s slice=%s replaced (deleted=%d, inserted=%d)",
        domain, slice_str, deleted, inserted,
    )
    return {
        "status": "success",
        "inserted": inserted,
        "updated": 0,
        "deleted": deleted,
    }


# ---------------------------------------------------------------------------
# Per-domain orchestration
# ---------------------------------------------------------------------------

def process_domain(
    domain: str, mode: str, slice_str: str | None, file_arg: str | None,
    dry_run: bool, cfg: dict | None,
) -> dict[str, Any]:
    """Run one domain end-to-end and return a result record."""
    started = _utc_iso()
    t0 = time.time()
    record: dict[str, Any] = {
        "domain": domain,
        "mode": mode,
        "slice": slice_str,
        "rows_loaded": None,
        "rows_inserted": None,
        "rows_updated": None,
        "rows_deleted": None,
        "duration_s": None,
        "status": "success",
        "error": None,
        "started_at": started,
        "completed_at": None,
    }

    try:
        if mode == "onetime":
            _do_onetime(domain, dry_run)
        elif mode == "delta":
            result = _do_delta(domain, dry_run)
            if isinstance(result, dict):
                # Safe-upsert path: counts come straight from RETURNING.
                record["status"] = result.get("status", "success")
                record["rows_inserted"] = result.get("inserted")
                record["rows_updated"] = result.get("updated")
                record["rows_deleted"] = result.get("deleted")
            else:
                record["status"] = result
        elif mode == "file":
            result = _do_file(domain, slice_str, file_arg, dry_run)
            if isinstance(result, dict):
                # Safe slice-replace path returns exact diff.
                record["status"] = result.get("status", "success")
                record["rows_inserted"] = result.get("inserted")
                record["rows_updated"] = result.get("updated")
                record["rows_deleted"] = result.get("deleted")
        else:
            raise ValueError(f"unknown mode {mode!r}")

        if record["status"] == "success" and not dry_run:
            # If safe-upsert provided exact counts, headline rows_loaded =
            # inserted + updated. Otherwise fall back to the loader's reported
            # batch row count (still useful for onetime / partitioned modes).
            ins = record["rows_inserted"]
            upd = record["rows_updated"]
            if ins is not None and upd is not None:
                record["rows_loaded"] = ins + upd
            else:
                record["rows_loaded"] = _last_row_count(domain)
            # The customer-demand loader owns refresh + exact profile lineage;
            # completing its audit batch before a wrapper refresh would expose
            # stale profile data as current.
            if domain != "customer_demand":
                _refresh_mvs(domain)
    except (subprocess.CalledProcessError, psycopg.Error,
            ValueError, FileNotFoundError) as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("domain=%s failed", domain)

    record["duration_s"] = round(time.time() - t0, 3)
    record["completed_at"] = _utc_iso()
    return record


def _last_row_count(domain: str) -> int | None:
    """Best-effort lookup of the most recent completed batch row_count_out."""
    try:
        with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT row_count_out FROM audit_load_batch "
                "WHERE domain = %s AND status = 'completed' "
                "ORDER BY completed_at DESC LIMIT 1",
                [domain],
            )
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else None
    except psycopg.Error as exc:
        logger.debug("row count lookup failed: %s", exc)
        return None


def _table_count(domain: str) -> int | None:
    """COUNT(*) of the domain's primary table — used to report NET change."""
    try:
        from common.core.domain_specs import get_spec
        table = get_spec(domain).table
    except (ImportError, KeyError, ValueError, AttributeError):
        return None
    try:
        with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
            cur.execute(f'SELECT COUNT(*) FROM "{table}"')
            row = cur.fetchone()
            return int(row[0]) if row else None
    except psycopg.Error as exc:
        logger.debug("table count lookup failed for %s: %s", domain, exc)
        return None


def _filter_orphan_fks(
    cur,
    table: str,
    stg: str = "_upsert_stg",
    available_columns: set[str] | None = None,
) -> int:
    """Delete rows from the staging table whose outgoing FKs don't resolve.

    Source CSVs sometimes contain rows referencing dim ids that aren't in the
    warehouse (data quality bugs upstream). Without filtering, the INSERT would
    raise ForeignKeyViolation. We skip those rows and log a count instead so
    delta loads complete cleanly. Returns total rows removed from staging.
    """
    cur.execute(
        """
        SELECT
            c.confrelid::regclass::text AS ref_table,
            array_agg(a.attname  ORDER BY k.pos)  AS local_cols,
            array_agg(af.attname ORDER BY fk.pos) AS ref_cols
        FROM pg_constraint c
        JOIN LATERAL unnest(c.conkey)  WITH ORDINALITY AS k(attnum, pos)  ON TRUE
        JOIN LATERAL unnest(c.confkey) WITH ORDINALITY AS fk(attnum, pos) ON fk.pos = k.pos
        JOIN pg_attribute a  ON a.attrelid  = c.conrelid  AND a.attnum  = k.attnum
        JOIN pg_attribute af ON af.attrelid = c.confrelid AND af.attnum = fk.attnum
        WHERE c.contype = 'f' AND c.conrelid = %s::regclass
        GROUP BY c.oid, c.confrelid
        """,
        (table,),
    )
    fks = cur.fetchall()
    total_removed = 0
    for ref_table, local_cols, ref_cols in fks:
        if available_columns is not None and any(
            local_col not in available_columns for local_col in local_cols
        ):
            logger.info(
                "skipping orphan-FK filter for %s -> %s: incoming staging data "
                "does not provide %s",
                table,
                ref_table,
                ", ".join(
                    local_col for local_col in local_cols
                    if local_col not in available_columns
                ),
            )
            continue
        join = " AND ".join(
        f's."{lc}" = r."{rc}"' for lc, rc in zip(local_cols, ref_cols, strict=True)
        )
        cur.execute(
            f'DELETE FROM {stg} s WHERE NOT EXISTS '
            f'(SELECT 1 FROM "{ref_table}" r WHERE {join})'
        )
        n = cur.rowcount
        if n:
            logger.warning(
                "skipped %d staging row(s) with unresolved FK -> %s (%s)",
                n, ref_table, ", ".join(local_cols),
            )
            total_removed += n
    return total_removed


def _resolve_conflict_target(cur, table: str, ck_field: str | None) -> list[str]:
    """Pick a unique-constraint column list suitable for ON CONFLICT.

    Prefers a unique index that contains ``ck_field`` (the natural row identity
    after computing the composite key). Falls back to the shortest non-PK
    unique index. Returns [] when none exists.
    """
    cur.execute(
        """
        SELECT i.relname,
               array_agg(a.attname ORDER BY array_position(ix.indkey::int[], a.attnum)) AS cols
        FROM pg_index ix
        JOIN pg_class i ON i.oid = ix.indexrelid
        JOIN pg_attribute a ON a.attrelid = ix.indrelid AND a.attnum = ANY(ix.indkey)
        WHERE ix.indrelid = %s::regclass
          AND ix.indisunique
          AND NOT ix.indisprimary
        GROUP BY i.relname
        """,
        (table,),
    )
    candidates = cur.fetchall()
    if not candidates:
        return []
    if ck_field:
        for _name, cols in candidates:
            if ck_field in cols:
                return list(cols)
    return list(min(candidates, key=lambda r: len(r[1]))[1])


# _is_pg_partitioned delegates to common/core/etl_helpers.is_pg_partitioned so
# the dispatcher and the bulk loader share one partitioned-check (US6).
_is_pg_partitioned = is_pg_partitioned


def _ensure_monthly_partitions(
    cur, table: str, partition_col: str, stg_alias: str = "_upsert_stg",
) -> int:
    """Pre-create any monthly partitions referenced by the staging table.

    Discovers the distinct months in staging, then delegates per-month creation
    to common/core/etl_helpers.ensure_monthly_partition (US6 convergence).
    Returns the count of partitions created.
    """
    cur.execute(
        f'SELECT DISTINCT date_trunc(\'month\', "{partition_col}"::date)::date '
        f'FROM {stg_alias} WHERE "{partition_col}" IS NOT NULL ORDER BY 1'
    )
    months = [r[0] for r in cur.fetchall()]
    created = 0
    for m in months:
        cur.execute(
            "SELECT 1 FROM pg_class WHERE relname = %s "
            "AND relnamespace = 'public'::regnamespace",
            (monthly_partition_name(table, m),),
        )
        if cur.fetchone():
            continue
        create_monthly_partition(cur, table, m)
        created += 1
    if created:
        logger.info("created %d new partition(s) on %s", created, table)
    return created


def _safe_upsert(domain: str) -> dict[str, int]:
    """Non-destructive upsert from cleaned CSV — never TRUNCATEs, never CASCADEs.

    Pipeline: COPY cleaned CSV -> TEMP staging -> INSERT ... ON CONFLICT DO
    UPDATE WHERE values differ. Returns exact insert/update/delete counts via
    RETURNING (xmax = 0) discrimination.

    Handles three target shapes:
      1. Ordinary dim/fact tables (sales, forecast, dim_item) — straight upsert.
      2. PG-partitioned tables (fact_inventory_snapshot) — pre-creates needed
         monthly partitions before the upsert.
      3. ON CONFLICT target auto-discovered from pg_index (preferring the
         unique constraint that contains the ck_field).

    Orphan deletion is only attempted for dim_* tables; for facts we treat
    delta as additive (removing fact rows by absence is rarely intended and
    expensive across multi-million-row tables).
    """
    import csv as csv_mod

    from common.core.domain_specs import get_spec

    spec = get_spec(domain)
    table = spec.table
    bk_cols = _resolve_business_key(domain)
    if not bk_cols:
        raise ValueError(f"no business key for {domain} — cannot upsert safely")
    csv_path = ROOT / "data" / spec.clean_file
    if not csv_path.exists():
        raise FileNotFoundError(f"cleaned CSV not found: {csv_path}")

    with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = %s AND table_schema = 'public' "
            "ORDER BY ordinal_position",
            (table,),
        )
        target_cols = {r[0]: r[1] for r in cur.fetchall()}
        if not target_cols:
            raise ValueError(f"target table {table} not found")

        with open(csv_path, encoding="utf-8") as f:
            csv_header = next(csv_mod.reader(f))
        common = [c for c in csv_header if c in target_cols]
        non_key = [c for c in common if c not in bk_cols]
        if not non_key:
            raise ValueError(f"no non-key columns in CSV for {domain}")

        # Some dim tables require a derived composite-key column (e.g. dim_item.item_ck)
        # that's NOT NULL and not in the CSV. Compute it from the business key.
        ck_field = getattr(spec, "ck_field", None)
        ck_expr: str | None = None
        if ck_field and ck_field in target_cols and ck_field not in common:
            sep = (getattr(spec, "business_key_separator", "-") or "-").replace("'", "''")
            parts = [f'trim(s."{c}")' for c in bk_cols]
            ck_expr = parts[0] if len(parts) == 1 else f" || '{sep}' || ".join(parts)

        def cast_expr(col: str) -> str:
            t = target_cols[col]
            # For strict types: '', 'null', 'none', 'NA' coerce to NULL + cast.
            # For TEXT: keep raw value (matches existing loader; 'null' is a
            # legitimate string in some columns like country).
            if t in ("integer", "bigint", "smallint",
                     "numeric", "double precision", "real",
                     "date", "timestamp without time zone",
                     "timestamp with time zone", "boolean"):
                return (
                    f'NULLIF(NULLIF(NULLIF(NULLIF(s."{col}", \'\'), \'null\'), '
                    f'\'none\'), \'NA\')::{t}'
                )
            return f's."{col}"'

        stg_cols = ", ".join(f'"{c}" TEXT' for c in csv_header)
        cur.execute(f"CREATE TEMP TABLE _upsert_stg ({stg_cols})")
        with open(csv_path, encoding="utf-8") as f:
            with cur.copy(
                "COPY _upsert_stg FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')"
            ) as cp:
                while chunk := f.read(8192):
                    cp.write(chunk)

        # Pre-filter staging to drop rows whose outgoing FKs don't resolve.
        # Surfaces upstream data-quality issues as warnings rather than
        # transaction-killing FK violations.
        _filter_orphan_fks(
            cur,
            table,
            "_upsert_stg",
            available_columns=set(csv_header),
        )

        # Auto-discover ON CONFLICT target from pg_index — handles single-col
        # (sales_ck), composite (forecast_ck+model_id), and partition-aware
        # (inventory_ck+snapshot_date) unique constraints uniformly.
        conflict_cols = _resolve_conflict_target(cur, table, ck_field)
        if not conflict_cols:
            raise ValueError(
                f"no usable unique constraint on {table} for ON CONFLICT — "
                f"add a UNIQUE index covering {ck_field or bk_cols}"
            )

        # Pre-create partitions for PG-partitioned targets so INSERT routing works.
        partitioned = _is_pg_partitioned(cur, table)
        if partitioned:
            partition_col = next(
                (c for c in conflict_cols if c in target_cols
                 and target_cols[c] in ("date", "timestamp without time zone",
                                        "timestamp with time zone")),
                None,
            )
            if partition_col is None and spec.date_fields:
                partition_col = next(iter(spec.date_fields))
            if partition_col and partition_col in csv_header:
                _ensure_monthly_partitions(cur, table, partition_col)

        # Insert columns = ck_field (if needed) + CSV common columns
        insert_cols = ([ck_field] if ck_expr else []) + common
        insert_cols_quoted = ", ".join(f'"{c}"' for c in insert_cols)
        # Alias every projected column so CREATE TABLE AS gives them stable
        # names (PG's auto-naming would collide on multiple NULLIF expressions).
        select_pieces = (
            ([f'{ck_expr} AS "{ck_field}"'] if ck_expr else [])
            + [f'{cast_expr(c)} AS "{c}"' for c in common]
        )
        select_exprs = ", ".join(select_pieces)
        update_assigns = [f'"{c}" = EXCLUDED."{c}"' for c in non_key]
        if "modified_ts" in target_cols and "modified_ts" not in non_key:
            update_assigns.append('"modified_ts" = NOW()')
        # Skip UPDATE when nothing actually changed (PG counts no-op updates
        # without this guard). DISTINCT FROM treats NULL == NULL.
        change_predicate = " OR ".join(
            f'"{table}"."{c}" IS DISTINCT FROM EXCLUDED."{c}"' for c in non_key
        )

        conflict_quoted = ", ".join(f'"{c}"' for c in conflict_cols)

        # Materialize the casted/projected source rows once, then pre-count
        # inserts and updates against the existing target. This avoids the
        # `RETURNING (xmax = 0)` trick — which Postgres rejects on partitioned
        # tables ("cannot retrieve a system column in this context") — and
        # works uniformly across ordinary and partitioned targets.
        # DISTINCT ON dedupes the source so the final INSERT can't trip
        # CardinalityViolation when the CSV has duplicate rows on the conflict
        # key (a known issue with customer_demand source feeds, ~0.3% dupes).
        cur.execute(
            f'CREATE TEMP TABLE _upsert_src AS '
            f'SELECT DISTINCT ON ({conflict_quoted}) * '
            f'FROM (SELECT {select_exprs} FROM _upsert_stg s) projected '
            f'ORDER BY {conflict_quoted}'
        )
        for col in conflict_cols:
            cur.execute(f'CREATE INDEX ON _upsert_src ("{col}")')

        conflict_join = " AND ".join(
            f't."{c}" = src."{c}"' for c in conflict_cols
        )
        cur.execute(
            f'SELECT COUNT(*) FROM _upsert_src src '
            f'WHERE NOT EXISTS (SELECT 1 FROM "{table}" t WHERE {conflict_join})'
        )
        ins = int(cur.fetchone()[0])

        non_key_diff = " OR ".join(
            f't."{c}" IS DISTINCT FROM src."{c}"' for c in non_key
        )
        cur.execute(
            f'SELECT COUNT(*) FROM _upsert_src src '
            f'JOIN "{table}" t ON {conflict_join} '
            f'WHERE {non_key_diff}'
        )
        upd = int(cur.fetchone()[0])

        # Now do the actual upsert (no RETURNING — counts already known).
        cur.execute(
            f'INSERT INTO "{table}" ({insert_cols_quoted}) '
            f'SELECT {insert_cols_quoted} FROM _upsert_src '
            f'ON CONFLICT ({conflict_quoted}) DO UPDATE '
            f'SET {", ".join(update_assigns)} '
            f'WHERE {change_predicate}'
        )

        # Orphan deletion: only for dim tables. Facts are treated as additive —
        # removing rows by absence is rarely intended and very expensive.
        deleted = 0
        if table.startswith("dim_"):
            deleted, skipped = _delete_orphans(cur, table, bk_cols)
            if skipped > 0:
                logger.info(
                    "delta: %s — kept %d orphan rows still referenced by FK constraints",
                    domain, skipped,
                )

        conn.commit()

        # Post-load maintenance: ANALYZE updates planner stats so subsequent
        # queries pick the right plan; cheap (~ms-s per table). Skipped when
        # nothing actually changed.
        if (ins + upd + deleted) > 0:
            _post_load_maintenance(table, reindex=_REINDEX_REQUESTED)

        return {"inserted": int(ins), "updated": int(upd), "deleted": int(deleted)}


# Module-level flag set from CLI --reindex; consumed by _safe_upsert.
_REINDEX_REQUESTED = False


def _post_load_maintenance(table: str, *, reindex: bool = False) -> None:
    """Run ANALYZE (and optional REINDEX) on the target after a successful load.

    ANALYZE refreshes planner statistics so query plans reflect the new data.
    REINDEX rebuilds indexes from scratch — useful after very large bulk
    upserts to defragment, but slow; off by default.
    """
    try:
        with psycopg.connect(**get_db_params(), autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(f'ANALYZE "{table}"')
            logger.info("ANALYZE %s", table)
            if reindex:
                # REINDEX TABLE locks the table briefly; users opt in via --reindex.
                cur.execute(f'REINDEX TABLE "{table}"')
                logger.info("REINDEX TABLE %s", table)
    except psycopg.Error as exc:
        logger.warning("post-load maintenance failed for %s: %s", table, exc)


def _delete_orphans(cur, table: str, bk_cols: list[str]) -> tuple[int, int]:
    """Delete rows from ``table`` whose business key is absent from ``_upsert_stg``,
    but skip any row still referenced by an enforced FK from another table.

    Returns ``(deleted, skipped_due_to_fk)``.
    """
    bk_join = " AND ".join(f's."{c}" = t."{c}"' for c in bk_cols)
    cur.execute(
        """
        SELECT
            conrelid::regclass::text AS ref_table,
            array_agg(a.attname  ORDER BY k.pos)  AS ref_cols,
            array_agg(af.attname ORDER BY fk.pos) AS dim_cols
        FROM pg_constraint c
        JOIN LATERAL unnest(c.conkey)  WITH ORDINALITY AS k(attnum, pos)  ON TRUE
        JOIN LATERAL unnest(c.confkey) WITH ORDINALITY AS fk(attnum, pos) ON fk.pos = k.pos
        JOIN pg_attribute a  ON a.attrelid  = c.conrelid  AND a.attnum  = k.attnum
        JOIN pg_attribute af ON af.attrelid = c.confrelid AND af.attnum = fk.attnum
        WHERE c.contype = 'f' AND c.confrelid = %s::regclass
        GROUP BY c.oid, conrelid
        """,
        (table,),
    )
    fk_refs = cur.fetchall()

    fk_protect_sql = ""
    for ref_table, ref_cols, dim_cols in fk_refs:
        join_clause = " AND ".join(
            f't."{dc}" = r."{rc}"' for dc, rc in zip(dim_cols, ref_cols, strict=True)
        )
        fk_protect_sql += (
            f' AND NOT EXISTS (SELECT 1 FROM "{ref_table}" r WHERE {join_clause})'
        )

    # Count would-be orphans BEFORE protection so we can report skipped.
    cur.execute(
        f'SELECT COUNT(*) FROM "{table}" t '
        f'WHERE NOT EXISTS (SELECT 1 FROM _upsert_stg s WHERE {bk_join})'
    )
    total_orphans = int(cur.fetchone()[0])

    if total_orphans == 0:
        return 0, 0

    cur.execute(
        f'DELETE FROM "{table}" t '
        f'WHERE NOT EXISTS (SELECT 1 FROM _upsert_stg s WHERE {bk_join})'
        f'{fk_protect_sql}'
    )
    deleted = cur.rowcount
    return deleted, max(0, total_orphans - deleted)


def _resolve_business_key(domain: str) -> list[str]:
    """Return the *business* key columns for a domain (e.g. ['item_id']).

    Surrogate primary keys (dim_item.item_sk) get regenerated on every
    TRUNCATE+INSERT, which makes a PK-based diff classify every row as
    deleted+inserted. The business key is what the user considers a unique
    row identity across reloads.
    """
    try:
        from common.core.domain_specs import get_spec
        spec = get_spec(domain)
    except (ImportError, KeyError, ValueError, AttributeError):
        return []
    if hasattr(spec, "key_fields"):
        try:
            return list(spec.key_fields())
        except (TypeError, AttributeError):  # spec.key_fields not callable/absent
            pass
    fields = getattr(spec, "business_key_fields", ()) or ()
    if fields:
        return list(fields)
    field = getattr(spec, "business_key_field", None)
    return [field] if field else []


def _run_with_diff(
    domain: str,
    do_load,  # callable that performs the actual load (subprocess etc.)
) -> tuple[int | None, int | None, int | None]:
    """Run ``do_load`` while capturing inserted/updated/deleted counts.

    Strategy: snapshot ``(pk_cols, md5(row::text))`` to a temp table BEFORE
    the load on a long-lived connection, then diff against the post-load
    state. Returns ``(inserted, updated, deleted)`` or ``(None, None, None)``
    when the diff cannot be computed (no PK, missing spec, DB error).
    """
    try:
        from common.core.domain_specs import get_spec
        table = get_spec(domain).table
    except (ImportError, KeyError, ValueError, AttributeError):
        do_load()
        return None, None, None
    pk_cols = _resolve_business_key(domain)
    if not pk_cols:
        logger.warning("no business key for %s — diff disabled, falling back to net count", domain)
        do_load()
        return None, None, None

    pk_select = ", ".join(f'"{c}"' for c in pk_cols)
    join_on = " AND ".join(f'a."{c}" = b."{c}"' for c in pk_cols)

    try:
        # Hold the connection open across the subprocess so the temp table
        # survives. The subprocess uses its own connection; ours is read-only
        # for this purpose so no lock contention.
        with psycopg.connect(**get_db_params()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'CREATE TEMP TABLE _diff_before AS '
                    f'SELECT {pk_select}, md5(t::text) AS row_hash FROM "{table}" t'
                )
                cur.execute(f'CREATE INDEX ON _diff_before ({pk_select})')
            conn.commit()
            do_load()
            with conn.cursor() as cur:
                cur.execute(
                    f'WITH after_state AS ('
                    f'  SELECT {pk_select}, md5(t::text) AS row_hash FROM "{table}" t'
                    f') '
                    f'SELECT '
                    f'  (SELECT COUNT(*) FROM after_state a '
                    f'    WHERE NOT EXISTS (SELECT 1 FROM _diff_before b WHERE {join_on})), '
                    f'  (SELECT COUNT(*) FROM after_state a '
                    f'    JOIN _diff_before b ON {join_on} WHERE a.row_hash <> b.row_hash), '
                    f'  (SELECT COUNT(*) FROM _diff_before b '
                    f'    WHERE NOT EXISTS (SELECT 1 FROM after_state a WHERE {join_on}))'
                )
                ins, upd, dele = cur.fetchone()
                return int(ins), int(upd), int(dele)
    except psycopg.Error as exc:
        logger.warning("diff computation failed for %s: %s", domain, exc)
        return None, None, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts.etl.load",
        description="Unified ETL load dispatcher (onetime | delta | file)",
    )
    parser.add_argument("--domain", required=True,
                        help="Domain name (e.g. sales, inventory, customer_demand) or 'all'")
    parser.add_argument("--mode", required=True, choices=ALLOWED_MODES,
                        help="onetime | delta | file")
    parser.add_argument("--slice", dest="slice_token", default=None,
                        help="Required for mode=file when domain is partitioned (e.g. 2026-03)")
    parser.add_argument("--file", dest="file_arg", default=None,
                        help="Optional CSV path (mode=file only); falls back to canonical path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan only — no SQL or subprocess execution")
    parser.add_argument("--reindex", action="store_true",
                        help="Run REINDEX TABLE after a successful upsert "
                             "(defragments indexes; slow — only for very large bulk loads)")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Cross-field validation that argparse choices/required cannot express."""
    if args.mode != "file" and args.slice_token:
        raise ValueError("--slice is only valid with --mode file")
    if args.mode != "file" and args.file_arg:
        raise ValueError("--file is only valid with --mode file")
    if args.mode == "file" and args.domain != "all" and is_partitioned(args.domain):
        if not args.slice_token and not args.file_arg:
            raise ValueError(
                f"--slice is required for mode=file with partitioned domain {args.domain!r}"
            )
        if args.slice_token:
            spec = get_partition(args.domain)
            if spec is None:
                raise ValueError(f"{args.domain} is not partitioned")
            # Raises ValueError on malformed input.
            slice_to_date_range(args.slice_token, spec.format)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2  # unreachable; parser.error exits

    # Set the module-level reindex flag once; consumed by _safe_upsert.
    global _REINDEX_REQUESTED
    _REINDEX_REQUESTED = bool(args.reindex)

    cfg = _load_etl_config()
    domains = _domain_order(cfg) if args.domain == "all" else [args.domain]

    if args.domain == "all" and args.mode == "file":
        # File mode + 'all' is supported only when no slice/file is given for
        # non-partitioned domains; partitioned ones will fail validation per-domain.
        logger.info("Running mode=file across all %d domains", len(domains))

    results: list[dict[str, Any]] = []
    any_failed = False
    for d in domains:
        result = process_domain(
            d, args.mode, args.slice_token, args.file_arg, args.dry_run, cfg,
        )
        _emit(result)
        results.append(result)
        if result["status"] == "failed":
            any_failed = True

    # Single-domain delta with skipped status -> exit 2 (per contract).
    if (
        args.domain != "all"
        and args.mode == "delta"
        and len(results) == 1
        and results[0]["status"] == "skipped"
    ):
        return 2

    return 1 if any_failed else 0


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    sys.exit(main())
