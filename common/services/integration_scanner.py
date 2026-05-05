"""Smart change detector for ETL source files.

Scans ``data/input/`` for new or modified source files, compares each one's
SHA-256 against the latest completed batch in ``audit_load_batch``, and emits
both a per-domain change set and a sequential job chain in dim-first /
fact-last order. Degrades gracefully on DB error so the UI can still render.
"""
from __future__ import annotations

import glob
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypedDict

import psycopg

from common.core.domain_specs import get_spec
from common.engines.medallion import file_hash

logger = logging.getLogger(__name__)

KIND = Literal["dim", "fact"]
MODE = Literal["onetime", "delta", "file"]


class FileChange(TypedDict):
    path: str
    current_hash: str | None
    last_hash: str | None
    changed: bool


class DomainChange(TypedDict):
    domain: str
    kind: KIND
    changed: bool
    reason: str
    proposed_mode: MODE
    proposed_slice: str | None
    source_files: list[FileChange]


class ChainStep(TypedDict):
    step: int
    domain: str
    mode: MODE
    slice: str | None


class ScanResult(TypedDict):
    scanned_at: str
    changes: list[DomainChange]
    proposed_chain: list[ChainStep]


# Hardcoded ordering: dimensions first (masters), then facts (details).
# Within tier: independent dims first, derived dims after.
DOMAIN_ORDER: list[tuple[str, KIND]] = [
    ("time",     "dim"),
    ("item",     "dim"),
    ("location", "dim"),
    ("customer", "dim"),
    ("sku",      "dim"),
    ("sourcing", "dim"),
    ("sales",           "fact"),
    ("forecast",        "fact"),
    ("inventory",       "fact"),
    ("customer_demand", "fact"),
    ("purchase_order",  "fact"),
]

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "input"
_INVENTORY_MONTH_RE = re.compile(r"Inventory_Snapshot_(\d{4})_(\d{2})\.csv$")


def _resolve_source_files(domain: str) -> list[Path]:
    """Return list of source file paths for the domain. Empty if missing."""
    if domain == "inventory":
        return sorted(Path(p) for p in glob.glob(str(INPUT_DIR / "Inventory_Snapshot_*.csv")))
    if domain == "customer_demand":
        return sorted(Path(p) for p in glob.glob(str(INPUT_DIR / "*_customer_demand.csv")))
    if domain == "time":
        return []  # generated, not file-driven
    try:
        spec = get_spec(domain)
    except (KeyError, ValueError):
        logger.debug("no DomainSpec for %r", domain)
        return []
    src = (spec.source_file or "").strip()
    if not src or src.startswith("_generated"):
        return []
    path = INPUT_DIR / src
    return [path] if path.exists() else []


def _last_hash_for_file(pool, domain: str, basename: str) -> str | None:
    """Per-file last hash for inventory-style multi-file domains."""
    sql = (
        "SELECT source_hash FROM audit_load_batch "
        "WHERE domain = %s AND source_file = %s AND status = 'completed' "
        "ORDER BY completed_at DESC LIMIT 1"
    )
    try:
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, [domain, basename])
            row = cur.fetchone()
            return row[0] if row else None
    except psycopg.Error as exc:
        logger.warning("audit per-file lookup failed (%s/%s): %s", domain, basename, exc)
        return None


def _last_hash_for_domain(pool, domain: str) -> str | None:
    """Domain-level last hash for single-file domains."""
    sql = (
        "SELECT source_hash FROM audit_load_batch "
        "WHERE domain = %s AND status = 'completed' "
        "ORDER BY completed_at DESC LIMIT 1"
    )
    try:
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(sql, [domain])
            row = cur.fetchone()
            return row[0] if row else None
    except psycopg.Error as exc:
        logger.warning("audit domain lookup failed (%s): %s", domain, exc)
        return None


def _safe_file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        h = file_hash(path)
        return h or None
    except (OSError, IOError) as exc:
        logger.warning("file_hash failed for %s: %s", path, exc)
        return None


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _newest_changed_inventory_slice(changed_basenames: list[str]) -> str | None:
    """Pick the lexically-latest YYYY_MM slice from changed inventory files."""
    slices: list[str] = []
    for name in changed_basenames:
        m = _INVENTORY_MONTH_RE.search(name)
        if m:
            slices.append(f"{m.group(1)}_{m.group(2)}")
    return max(slices) if slices else None


def _is_multi_file_domain(domain: str) -> bool:
    """True for domains whose source_file is a glob (multiple raw files).

    These need per-file hash bookkeeping (one audit row per file). Currently:
    inventory (Inventory_Snapshot_*.csv) and customer_demand (*_customer_demand.csv).
    """
    if domain in ("inventory", "customer_demand"):
        return True
    try:
        spec = get_spec(domain)
    except (KeyError, ValueError):
        return False
    return "*" in (getattr(spec, "source_file", "") or "")


def _scan_domain(pool, domain: str, kind: KIND) -> DomainChange:
    files = _resolve_source_files(domain)
    file_changes: list[FileChange] = []

    if _is_multi_file_domain(domain):
        for fp in files:
            current = _safe_file_hash(fp)
            last = _last_hash_for_file(pool, domain, fp.name)
            file_changes.append(FileChange(
                path=_rel(fp), current_hash=current, last_hash=last,
                changed=(current is not None and current != last),
            ))
    else:
        last = _last_hash_for_domain(pool, domain) if files else None
        for fp in files:
            current = _safe_file_hash(fp)
            file_changes.append(FileChange(
                path=_rel(fp), current_hash=current, last_hash=last,
                changed=(current is not None and current != last),
            ))

    changed_files = [fc for fc in file_changes if fc["changed"]]
    any_changed = bool(changed_files)
    proposed_mode: MODE = "delta"
    proposed_slice: str | None = None
    reason = "no source files registered" if not files else "up to date"

    if any_changed:
        if domain == "inventory":
            changed_names = [Path(fc["path"]).name for fc in changed_files]
            if len(changed_files) == 1:
                slc = _newest_changed_inventory_slice(changed_names)
                if slc:
                    proposed_mode, proposed_slice = "file", slc
                    reason = f"1 changed snapshot ({slc})"
                else:
                    reason = "1 changed snapshot (unparseable name)"
            else:
                reason = f"{len(changed_files)} changed snapshots"
        else:
            reason = "hash mismatch" if any(fc["last_hash"] for fc in changed_files) else "new file"

    return DomainChange(
        domain=domain, kind=kind, changed=any_changed, reason=reason,
        proposed_mode=proposed_mode, proposed_slice=proposed_slice,
        source_files=file_changes,
    )


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def scan_input_dir(pool) -> ScanResult:
    """Walk ``data/input/``, hash source files, return change set + ordered chain.

    For each domain in ``DOMAIN_ORDER``: resolve source file(s), hash them,
    compare against ``audit_load_batch`` (per-file for inventory, domain-level
    otherwise), build a ``DomainChange``. Then assemble ``proposed_chain``
    from only the changed domains, preserving ``DOMAIN_ORDER``.

    Never raises -- DB / FS errors are logged and surface as best-effort
    "changed" entries so the UI can prompt the user to investigate.
    """
    changes: list[DomainChange] = []
    for domain, kind in DOMAIN_ORDER:
        try:
            changes.append(_scan_domain(pool, domain, kind))
        except (psycopg.Error, OSError) as exc:
            logger.exception("scan_domain failed for %s: %s", domain, exc)
            changes.append(DomainChange(
                domain=domain, kind=kind, changed=False,
                reason=f"scan error: {exc}", proposed_mode="delta",
                proposed_slice=None, source_files=[],
            ))

    chain: list[ChainStep] = []
    step = 1
    for ch in changes:
        if ch["changed"]:
            chain.append(ChainStep(
                step=step, domain=ch["domain"],
                mode=ch["proposed_mode"], slice=ch["proposed_slice"],
            ))
            step += 1

    return ScanResult(scanned_at=_utc_iso(), changes=changes, proposed_chain=chain)
