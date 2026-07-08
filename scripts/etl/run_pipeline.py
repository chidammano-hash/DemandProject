"""Unified data pipeline orchestrator with full-reload and incremental-refresh modes.

Usage:
    python scripts/etl/run_pipeline.py --mode full [--domains item,sales] [--parallel] [--dry-run]
    python scripts/etl/run_pipeline.py --mode refresh [--domains inventory] [--dry-run]
"""

import argparse
import logging
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.db import get_db_params
from common.core.domain_specs import get_spec
from common.core.etl_helpers import tables_written_for_domain  # noqa: E402
from common.core.mv_refresh import (  # noqa: E402
    mvs_for_tables,
    refresh_materialized_views,
    refresh_materialized_views_parallel,
)
from common.core.utils import load_config
from common.engines.medallion import file_hash
from common.services.perf_profiler import profiled_section

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_DOMAINS = ["item", "location", "customer", "time", "sku",
               "sales", "forecast", "inventory", "sourcing", "purchase_order"]

# Dimensions load first (wave 1); facts load after (wave 2).
# Sourcing is a dim table; forecast depends on sku existing.
_DIMENSION_DOMAINS = {"item", "location", "customer", "time", "sku", "sourcing"}


def _cfg() -> dict:
    """Load ETL pipeline config."""
    return load_config("etl_config.yaml")


def _elapsed(t0: float) -> str:
    """Human-readable elapsed time."""
    s = time.time() - t0
    if s < 60:
        return f"{s:.1f}s"
    return f"{s / 60:.1f}m"


# ---------------------------------------------------------------------------
# Change Detection
# ---------------------------------------------------------------------------

def detect_changes(domains: list[str], cur) -> dict[str, bool]:
    """Compare current clean CSV hashes vs last successful batch in audit_load_batch.

    Returns dict mapping domain name -> True if changed (needs reload).
    """
    data_dir = ROOT / "data"
    changes = {}
    for domain in domains:
        if domain == "inventory":
            continue  # handled separately
        spec = get_spec(domain)
        csv_path = data_dir / spec.clean_file
        if not csv_path.exists():
            changes[domain] = True
            continue
        current_hash = file_hash(csv_path)
        cur.execute(
            "SELECT source_hash FROM audit_load_batch "
            "WHERE domain = %s AND status = 'completed' "
            "ORDER BY completed_at DESC LIMIT 1",
            [domain],
        )
        row = cur.fetchone()
        changes[domain] = (row is None or row[0] != current_hash)
    return changes


def detect_inventory_changes(data_dir: Path, cur) -> list[Path]:
    """Return list of new/changed Inventory_Snapshot_*.csv files."""
    import glob as glob_mod

    all_files = sorted(
        Path(p) for p in glob_mod.glob(str(data_dir / "Inventory_Snapshot_*.csv"))
    )
    changed = []
    for f in all_files:
        h = file_hash(f)
        cur.execute(
            "SELECT source_hash FROM audit_load_batch "
            "WHERE domain = 'inventory' AND source_file = %s AND status = 'completed' "
            "ORDER BY completed_at DESC LIMIT 1",
            [f.name],
        )
        row = cur.fetchone()
        if row is None or row[0] != h:
            changed.append(f)
    return changed


def _month_range_from_filename(filename: str) -> tuple[str, str] | None:
    """Extract month start/end dates from Inventory_Snapshot_YYYY_MM.csv filename.

    Returns (first_day, first_day_of_next_month) or None if unparseable.
    """
    m = re.search(r"(\d{4})_(\d{2})", filename)
    if not m:
        return None
    year, month = int(m.group(1)), int(m.group(2))
    first_day = f"{year:04d}-{month:02d}-01"
    # First day of next month
    if month == 12:
        next_first = f"{year + 1:04d}-01-01"
    else:
        next_first = f"{year:04d}-{month + 1:02d}-01"
    return first_day, next_first


def build_incremental_delete(changed_files: list[Path]) -> str:
    """Build SQL WHERE clause to delete rows for changed inventory months."""
    conditions = []
    for f in changed_files:
        rng = _month_range_from_filename(f.name)
        if rng:
            conditions.append(
                f"(snapshot_date >= '{rng[0]}' AND snapshot_date < '{rng[1]}')"
            )
    if not conditions:
        return ""
    return " OR ".join(conditions)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_domain(domain: str, source_dir: Path) -> bool:
    """Normalize a single domain via subprocess. Returns True on success."""
    script = ROOT / "scripts" / "etl" / "normalize_dataset_csv.py"
    cmd = [sys.executable, str(script), "--dataset", domain, "--source-dir", str(source_dir)]
    logger.info("  Normalizing %s ...", domain)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("  FAILED: %s\n%s", domain, result.stderr)
        return False
    return True


def normalize_inventory(source_dir: Path, output: Path | None = None,
                        files: list[Path] | None = None) -> bool:
    """Normalize inventory snapshots. If files is given, only process those files."""
    script = ROOT / "scripts" / "etl" / "normalize_inventory_csv.py"
    cmd = [sys.executable, str(script), "--datafiles-dir", str(source_dir)]
    if output:
        cmd.extend(["--output", str(output)])
    logger.info("  Normalizing inventory (%s files) ...",
                len(files) if files else "all")

    if files:
        # For incremental: create a temp directory with symlinks to changed files only
        import tempfile
        with tempfile.TemporaryDirectory(prefix="inv_incr_") as tmpdir:
            for f in files:
                (Path(tmpdir) / f.name).symlink_to(f)
            cmd = [sys.executable, str(script), "--datafiles-dir", tmpdir]
            if output:
                cmd.extend(["--output", str(output)])
            result = subprocess.run(cmd, capture_output=True, text=True)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("  FAILED: inventory\n%s", result.stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_domain(domain: str, csv_path: Path,
                incremental_delete: str | None = None) -> dict:
    """Load a domain via direct CSV->table load."""
    from scripts.etl.load_dataset_postgres import load_domain as _load
    spec = get_spec(domain)
    return _load(spec, csv_path, incremental_delete=incremental_delete)


# customer_demand has its own normalizer + partitioned parallel loader, so it
# can't go through the generic normalize_domain/load_domain paths (US15).

def normalize_customer_demand(source_dir: Path) -> bool:
    """Normalize customer demand via its dedicated subprocess. True on success."""
    script = ROOT / "scripts" / "etl" / "normalize_customer_demand_csv.py"
    cmd = [sys.executable, str(script), "--source-dir", str(source_dir)]
    logger.info("  Normalizing customer_demand ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("  FAILED: customer_demand\n%s", result.stderr)
        return False
    return True


def load_customer_demand() -> dict:
    """Load customer demand via its dedicated loader (default UPSERT mode).

    UPSERT (no --replace) is the incremental-friendly path: changed rows are
    merged via ON CONFLICT, leaving untouched months intact.
    """
    script = ROOT / "scripts" / "etl" / "load_customer_demand_postgres.py"
    cmd = [sys.executable, str(script)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("  FAILED to load customer_demand\n%s", result.stderr)
        return {"domain": "customer_demand", "skipped": True}
    return {"domain": "customer_demand", "loaded": True}


# ---------------------------------------------------------------------------
# Materialized View Refresh
# ---------------------------------------------------------------------------

def get_mvs_for_domains(domains: list[str]) -> list[str]:
    """Dependent MVs for the tables the given domain loads write.

    Derived from the central dependency map (common/core/mv_refresh.py) via
    each domain's written tables — the per-domain lists that used to live in
    etl_config.yaml are gone.
    """
    tables: list[str] = []
    for domain in domains:
        try:
            tables.extend(tables_written_for_domain(domain))
        except ValueError:
            logger.warning("Unknown domain %r — no MVs derived for it", domain)
    return mvs_for_tables(tables)


def refresh_views(mv_list: list[str]) -> None:
    """Refresh materialized views sequentially (dedicated autocommit connection)."""
    refresh_materialized_views(mv_list)


def refresh_views_parallel(mv_list: list[str], max_workers: int = 3) -> None:
    """Refresh materialized views tier-parallel using separate connections."""
    refresh_materialized_views_parallel(mv_list, max_workers=max_workers)


def _load_domains_parallel(
    domains: list[str],
    data_dir: Path,
    max_workers: int,
) -> list[dict]:
    """Load multiple domains in parallel via ThreadPoolExecutor."""
    if not domains:
        return []
    results: list[dict] = []
    timings: dict[str, float] = {d: time.time() for d in domains}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for domain in domains:
            spec = get_spec(domain)
            csv_path = data_dir / spec.clean_file
            logger.info("  Submitting %s ...", domain)
            timings[domain] = time.time()
            futures[pool.submit(load_domain, domain, csv_path)] = domain
        for future in as_completed(futures):
            domain = futures[future]
            try:
                result = future.result()
                if result:
                    result["elapsed"] = _elapsed(timings[domain])
                    results.append(result)
                    logger.info("  Loaded %s (%s)", domain, result["elapsed"])
            except (psycopg.Error, OSError, ValueError) as exc:
                logger.error("  FAILED to load %s: %s", domain, exc)
                results.append({"domain": domain, "skipped": True})
    return results


# ---------------------------------------------------------------------------
# Pipeline Modes
# ---------------------------------------------------------------------------

def run_full(domains: list[str], source_dir: Path,
             parallel: bool = False, dry_run: bool = False) -> list[dict]:
    """Full reload: normalize all -> load all -> refresh all MVs."""
    results = []
    data_dir = ROOT / "data"
    cfg = _cfg()

    # Step 0: Run input cleanup
    cleanup_script = source_dir / "cleanup_input.py"
    if cleanup_script.exists():
        if dry_run:
            logger.info("[DRY-RUN] Would run %s", cleanup_script.name)
        else:
            logger.info("Running input cleanup (%s) ...", cleanup_script.name)
            subprocess.run([sys.executable, str(cleanup_script)], check=False)

    # Step 1: Normalize
    logger.info("=" * 60)
    logger.info("PHASE 1: Normalize (%d domains)", len(domains))
    logger.info("=" * 60)

    non_inv = [d for d in domains if d != "inventory"]
    has_inv = "inventory" in domains

    failed_domains: set[str] = set()

    with profiled_section("normalize"):
        if dry_run:
            for d in non_inv:
                logger.info("  [DRY-RUN] Would normalize: %s", d)
            if has_inv:
                logger.info("  [DRY-RUN] Would normalize: inventory (all files)")
        elif parallel and len(non_inv) > 1:
            par_cfg = cfg.get("parallel", {})
            norm_workers = par_cfg.get("normalize_workers",
                                       par_cfg.get("max_workers", 4))
            with ThreadPoolExecutor(max_workers=norm_workers) as pool:
                futures = {
                    pool.submit(normalize_domain, d, source_dir): d
                    for d in non_inv
                }
                for future in as_completed(futures):
                    d = futures[future]
                    if not future.result():
                        logger.error("Normalize failed for %s — will skip loading", d)
                        failed_domains.add(d)
            if has_inv:
                if not normalize_inventory(source_dir):
                    failed_domains.add("inventory")
        else:
            for d in non_inv:
                if not normalize_domain(d, source_dir):
                    failed_domains.add(d)
            if has_inv:
                if not normalize_inventory(source_dir):
                    failed_domains.add("inventory")

    if failed_domains:
        logger.warning("Skipping load for failed domains: %s",
                       ", ".join(sorted(failed_domains)))

    # Step 2: Load
    loadable = [d for d in domains if d not in failed_domains]
    logger.info("=" * 60)
    logger.info("PHASE 2: Load (%d domains)", len(loadable))
    logger.info("=" * 60)

    # Add skipped entries for failed domains
    for d in domains:
        if d in failed_domains:
            results.append({"domain": d, "skipped": True})

    with profiled_section("load_domains"):
        if dry_run:
            for domain in loadable:
                spec = get_spec(domain)
                csv_path = data_dir / spec.clean_file
                logger.info("  [DRY-RUN] Would load: %s from %s", domain, csv_path.name)
                results.append({"domain": domain, "dry_run": True})
        elif parallel and len(loadable) > 1:
            par_cfg = cfg.get("parallel", {})
            load_workers = par_cfg.get("load_workers",
                                       par_cfg.get("max_workers", 4))

            # Wave 1: dimensions (must complete before facts for FK/dependency safety)
            dims = [d for d in loadable if d in _DIMENSION_DOMAINS]
            facts = [d for d in loadable if d not in _DIMENSION_DOMAINS]

            if dims:
                logger.info("  Wave 1: dimensions (%s)", ", ".join(dims))
                results.extend(
                    _load_domains_parallel(dims, data_dir, load_workers))

            if facts:
                logger.info("  Wave 2: facts (%s)", ", ".join(facts))
                results.extend(
                    _load_domains_parallel(facts, data_dir, load_workers))
        else:
            # Sequential fallback
            for domain in loadable:
                spec = get_spec(domain)
                csv_path = data_dir / spec.clean_file
                logger.info("  Loading %s ...", domain)
                t0 = time.time()
                try:
                    result = load_domain(domain, csv_path)
                    if result:
                        result["elapsed"] = _elapsed(t0)
                        results.append(result)
                except (psycopg.Error, OSError, ValueError) as exc:
                    logger.error("  FAILED to load %s: %s", domain, exc)
                    results.append({"domain": domain, "skipped": True})

    # Step 3: Refresh MVs
    all_mvs = get_mvs_for_domains(domains)
    logger.info("=" * 60)
    logger.info("PHASE 3: Refresh materialized views (%d)", len(all_mvs))
    logger.info("=" * 60)

    with profiled_section("refresh_materialized_views"):
        if dry_run:
            for mv in all_mvs:
                logger.info("  [DRY-RUN] Would refresh: %s", mv)
        elif parallel and len(all_mvs) > 1:
            par_cfg = cfg.get("parallel", {})
            mv_workers = par_cfg.get("mv_refresh_workers",
                                     par_cfg.get("max_workers", 3))
            refresh_views_parallel(all_mvs, max_workers=mv_workers)
        else:
            refresh_views(all_mvs)

    return results


def run_refresh(domains: list[str], source_dir: Path,
                dry_run: bool = False) -> list[dict]:
    """Incremental refresh: detect changes -> normalize changed -> load changed."""
    results = []
    data_dir = ROOT / "data"

    db = get_db_params()
    with psycopg.connect(**db) as conn, conn.cursor() as cur:
        # Detect changes for non-inventory domains
        logger.info("=" * 60)
        logger.info("PHASE 1: Detecting changes")
        logger.info("=" * 60)

        with profiled_section("detect_changes"):
            non_inv = [d for d in domains if d != "inventory"]
            changes = detect_changes(non_inv, cur)
            changed_domains = [d for d, changed in changes.items() if changed]
            unchanged = [d for d, changed in changes.items() if not changed]

            # Detect inventory changes
            inv_changed_files: list[Path] = []
            if "inventory" in domains:
                inv_changed_files = detect_inventory_changes(source_dir, cur)
                if inv_changed_files:
                    changed_domains.append("inventory")
                else:
                    unchanged.append("inventory")

        for d in unchanged:
            logger.info("  %s: unchanged (skipping)", d)
        for d in changed_domains:
            if d == "inventory":
                logger.info("  inventory: %d file(s) changed", len(inv_changed_files))
                for f in inv_changed_files:
                    logger.info("    - %s", f.name)
            else:
                logger.info("  %s: changed", d)

        if not changed_domains:
            logger.info("No changes detected. Nothing to do.")
            return results

        # Normalize changed domains
        logger.info("=" * 60)
        logger.info("PHASE 2: Normalize changed domains (%d)", len(changed_domains))
        logger.info("=" * 60)

        with profiled_section("normalize_changed"):
            for domain in changed_domains:
                if domain == "inventory":
                    if dry_run:
                        logger.info("  [DRY-RUN] Would normalize inventory (%d files)",
                                    len(inv_changed_files))
                    else:
                        incr_output = data_dir / "inventory_incremental.csv"
                        normalize_inventory(source_dir, output=incr_output,
                                            files=inv_changed_files)
                elif domain == "customer_demand":
                    if dry_run:
                        logger.info("  [DRY-RUN] Would normalize: customer_demand")
                    else:
                        normalize_customer_demand(source_dir)
                else:
                    if dry_run:
                        logger.info("  [DRY-RUN] Would normalize: %s", domain)
                    else:
                        normalize_domain(domain, source_dir)

        # Load changed domains
        logger.info("=" * 60)
        logger.info("PHASE 3: Load changed domains (%d)", len(changed_domains))
        logger.info("=" * 60)

        with profiled_section("load_changed_domains"):
            for domain in changed_domains:
                if domain == "customer_demand":
                    if dry_run:
                        logger.info("  [DRY-RUN] Would load: customer_demand")
                        results.append({"domain": domain, "dry_run": True})
                        continue
                    logger.info("  Loading customer_demand ...")
                    t0 = time.time()
                    result = load_customer_demand()
                    result["elapsed"] = _elapsed(t0)
                    results.append(result)
                    continue

                spec = get_spec(domain)
                incr_delete = None

                if domain == "inventory":
                    csv_path = data_dir / "inventory_incremental.csv"
                    incr_delete = build_incremental_delete(inv_changed_files)
                else:
                    csv_path = data_dir / spec.clean_file

                if dry_run:
                    logger.info("  [DRY-RUN] Would load: %s", domain)
                    if incr_delete:
                        logger.info("    DELETE WHERE %s", incr_delete)
                    results.append({"domain": domain, "dry_run": True})
                    continue

                logger.info("  Loading %s ...", domain)
                t0 = time.time()
                try:
                    result = load_domain(domain, csv_path,
                                         incremental_delete=incr_delete)
                    if result:
                        result["elapsed"] = _elapsed(t0)
                        results.append(result)
                except (psycopg.Error, OSError, ValueError) as exc:
                    logger.error("  FAILED to load %s: %s", domain, exc)
                    results.append({"domain": domain, "skipped": True})

        # Refresh affected MVs
        mvs = get_mvs_for_domains(changed_domains)
        logger.info("=" * 60)
        logger.info("PHASE 4: Refresh affected views (%d)", len(mvs))
        logger.info("=" * 60)

        with profiled_section("refresh_materialized_views"):
            if dry_run:
                for mv in mvs:
                    logger.info("  [DRY-RUN] Would refresh: %s", mv)
            else:
                refresh_views(mvs)

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    """Print formatted summary table."""
    if not results:
        return

    print()
    print("=" * 50)
    print(f"{'Domain':<16} {'Rows In':>10} {'Loaded':>10} {'Time':>8}")
    print("-" * 50)
    for r in results:
        if r.get("dry_run"):
            print(f"{r['domain']:<16} {'(dry-run)':>10}")
            continue
        if r.get("skipped"):
            print(f"{r['domain']:<16} {'(skipped)':>10}")
            continue
        print(f"{r['domain']:<16} "
              f"{r.get('rows_in', 0):>10,} "
              f"{r.get('rows_loaded', 0):>10,} "
              f"{r.get('elapsed', ''):>8}")
    print("=" * 50)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified data pipeline orchestrator"
    )
    parser.add_argument(
        "--mode", required=True, choices=["full", "refresh"],
        help="full = wipe & reload everything; refresh = detect changes & load delta",
    )
    parser.add_argument(
        "--domains", default=None,
        help="Comma-separated domain list (default: all 10 domains)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--parallel", action="store_true",
                        help="Parallelize normalization, loading (two-wave), and MV refresh")
    parser.add_argument(
        "--data-dir", default=None,
        help="Source data directory (default: from etl_config.yaml)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = _cfg()
    domain_order = cfg.get("domain_order", ALL_DOMAINS)

    # Resolve domains
    if args.domains:
        requested = [d.strip() for d in args.domains.split(",")]
        # Preserve config ordering
        domains = [d for d in domain_order if d in requested]
    else:
        domains = list(domain_order)

    # Resolve source directory
    if args.data_dir:
        source_dir = Path(args.data_dir).resolve()
    else:
        source_dir = ROOT / cfg.get("source_data_dir", "data/input")

    logger.info("Pipeline mode: %s", args.mode)
    logger.info("Domains: %s", ", ".join(domains))
    logger.info("Source dir: %s", source_dir)
    if args.dry_run:
        logger.info("DRY-RUN mode: no changes will be made")
    print()

    t_total = time.time()

    if args.mode == "full":
        results = run_full(
            domains, source_dir,
            parallel=args.parallel,
            dry_run=args.dry_run,
        )
    else:
        results = run_refresh(
            domains, source_dir,
            dry_run=args.dry_run,
        )

    print_summary(results)
    logger.info("Total pipeline time: %s", _elapsed(t_total))


if __name__ == "__main__":
    main()
