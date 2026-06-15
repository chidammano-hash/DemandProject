"""Normalize monthly inventory snapshot CSVs into a single clean CSV.

Each source file contains 7-14 million rows with columns:
    exec_date, item, loc, lead_time, tot_oh, tot_oh_oo, mtd_sls

Output CSV columns:
    item_id, loc, snapshot_date, lead_time_days,
    qty_on_hand, qty_on_hand_on_order, qty_on_order, mtd_sales

Usage:
    python scripts/normalize_inventory_csv.py
    python scripts/normalize_inventory_csv.py --datafiles-dir /path/to/csvs
    python scripts/normalize_inventory_csv.py --output data/staged/inventory_clean.csv
    python scripts/normalize_inventory_csv.py --workers 8
"""

import argparse
import csv
import glob
import logging
import shutil
import sys
import tempfile
from datetime import date
from multiprocessing import Pool, cpu_count
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.services.perf_profiler import profiled_section

logger = logging.getLogger(__name__)

# Null sentinel values — consistent with normalize_dataset_csv.py
NULL_SENTINELS = {"", "null", "none", "na"}

# Output column order
OUTPUT_COLUMNS = [
    "item_id",
    "loc",
    "snapshot_date",
    "lead_time_days",
    "qty_on_hand",
    "qty_on_hand_on_order",
    "qty_on_order",
    "mtd_sales",
]

# Source column index positions (by name)
SRC_EXEC_DATE = "exec_date"
SRC_ITEM = "item"
SRC_LOC = "loc"
SRC_LEAD_TIME = "lead_time"
SRC_TOT_OH = "tot_oh"
SRC_TOT_OH_OO = "tot_oh_oo"
SRC_MTD_SLS = "mtd_sls"


def is_null(value: str) -> bool:
    """Return True if value should be treated as NULL."""
    return value.strip().lower() in NULL_SENTINELS


def parse_float(value: str, default: float = 0.0) -> float:
    """Parse a string as float, returning default if empty/null/invalid."""
    s = value.strip()
    if is_null(s):
        return default
    try:
        return float(s)
    except ValueError:
        return default


def parse_iso_date(value: str) -> str:
    """Parse and validate an ISO date string (YYYY-MM-DD).

    Returns the ISO date string if valid, empty string otherwise.
    """
    s = value.strip()
    if is_null(s):
        return ""
    # Try ISO format directly (YYYY-MM-DD)
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        try:
            d = date.fromisoformat(s)
            return d.isoformat()
        except ValueError:
            pass
    # Try compact format (YYYYMMDD)
    if len(s) == 8 and s.isdigit():
        try:
            d = date(int(s[0:4]), int(s[4:6]), int(s[6:8]))
            return d.isoformat()
        except ValueError:
            pass
    return ""


def _normalize_file_to_path(args: tuple[str, str]) -> tuple[str, int]:
    """Process one snapshot CSV into a temp output file (no header).

    Accepts a tuple for compatibility with multiprocessing.Pool.map.
    Returns (source_filename, rows_written).
    """
    source_path = Path(args[0])
    output_path = Path(args[1])
    rows_written = 0

    with source_path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.reader(src)
        raw_headers = next(reader)
        file_header_idx = {h.strip().lower().replace("\r", ""): i for i, h in enumerate(raw_headers)}

        # Validate expected columns exist
        expected = [SRC_EXEC_DATE, SRC_ITEM, SRC_LOC, SRC_LEAD_TIME, SRC_TOT_OH, SRC_TOT_OH_OO, SRC_MTD_SLS]
        missing = [c for c in expected if c not in file_header_idx]
        if missing:
            logger.info(f"  WARNING: Missing columns {missing} in {source_path.name}, skipping file")
            # Write empty file so merge step doesn't fail
            output_path.write_text("")
            return source_path.name, 0

        idx_exec_date = file_header_idx[SRC_EXEC_DATE]
        idx_item = file_header_idx[SRC_ITEM]
        idx_loc = file_header_idx[SRC_LOC]
        idx_lead_time = file_header_idx[SRC_LEAD_TIME]
        idx_tot_oh = file_header_idx[SRC_TOT_OH]
        idx_tot_oh_oo = file_header_idx[SRC_TOT_OH_OO]
        idx_mtd_sls = file_header_idx[SRC_MTD_SLS]

        with output_path.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.writer(dst, lineterminator="\n")

            for row in reader:
                # Strip carriage returns from all fields
                row = [cell.replace("\r", "") for cell in row]

                # Guard against short rows
                if len(row) < len(expected):
                    continue

                # exec_date -> snapshot_date
                snapshot_date = parse_iso_date(row[idx_exec_date])
                if not snapshot_date:
                    continue

                # item -> item_id
                item_id = row[idx_item].strip()
                if is_null(item_id):
                    continue

                # loc -> loc
                loc = row[idx_loc].strip()
                if is_null(loc):
                    continue

                # Numeric fields
                lead_time_days = parse_float(row[idx_lead_time], default=0.0)
                qty_on_hand = parse_float(row[idx_tot_oh], default=0.0)
                qty_on_hand_on_order = parse_float(row[idx_tot_oh_oo], default=0.0)
                qty_on_order = qty_on_hand_on_order - qty_on_hand
                mtd_sales = parse_float(row[idx_mtd_sls], default=0.0)

                writer.writerow([
                    item_id,
                    loc,
                    snapshot_date,
                    lead_time_days,
                    qty_on_hand,
                    qty_on_hand_on_order,
                    qty_on_order,
                    mtd_sales,
                ])
                rows_written += 1

    return source_path.name, rows_written


def normalize_file(
    source_path: Path,
    writer: csv.writer,
    header_idx: dict[str, int] | None,
) -> tuple[int, dict[str, int] | None]:
    """Stream-process a single inventory snapshot CSV (sequential fallback).

    Returns (rows_written, header_idx) where header_idx is populated on
    first call and reused for subsequent files.
    """
    rows_written = 0

    with source_path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.reader(src)
        raw_headers = next(reader)
        file_header_idx = {h.strip().lower().replace("\r", ""): i for i, h in enumerate(raw_headers)}

        # Validate expected columns exist
        expected = [SRC_EXEC_DATE, SRC_ITEM, SRC_LOC, SRC_LEAD_TIME, SRC_TOT_OH, SRC_TOT_OH_OO, SRC_MTD_SLS]
        missing = [c for c in expected if c not in file_header_idx]
        if missing:
            logger.info(f"  WARNING: Missing columns {missing} in {source_path.name}, skipping file")
            return 0, header_idx

        idx_exec_date = file_header_idx[SRC_EXEC_DATE]
        idx_item = file_header_idx[SRC_ITEM]
        idx_loc = file_header_idx[SRC_LOC]
        idx_lead_time = file_header_idx[SRC_LEAD_TIME]
        idx_tot_oh = file_header_idx[SRC_TOT_OH]
        idx_tot_oh_oo = file_header_idx[SRC_TOT_OH_OO]
        idx_mtd_sls = file_header_idx[SRC_MTD_SLS]

        for row in reader:
            # Strip carriage returns from all fields
            row = [cell.replace("\r", "") for cell in row]

            # Guard against short rows
            if len(row) < len(expected):
                continue

            # exec_date -> snapshot_date
            snapshot_date = parse_iso_date(row[idx_exec_date])
            if not snapshot_date:
                continue

            # item -> item_id
            item_id = row[idx_item].strip()
            if is_null(item_id):
                continue

            # loc -> loc
            loc = row[idx_loc].strip()
            if is_null(loc):
                continue

            # Numeric fields
            lead_time_days = parse_float(row[idx_lead_time], default=0.0)
            qty_on_hand = parse_float(row[idx_tot_oh], default=0.0)
            qty_on_hand_on_order = parse_float(row[idx_tot_oh_oo], default=0.0)
            qty_on_order = qty_on_hand_on_order - qty_on_hand
            mtd_sales = parse_float(row[idx_mtd_sls], default=0.0)

            writer.writerow([
                item_id,
                loc,
                snapshot_date,
                lead_time_days,
                qty_on_hand,
                qty_on_hand_on_order,
                qty_on_order,
                mtd_sales,
            ])
            rows_written += 1

    return rows_written, file_header_idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize monthly inventory snapshot CSVs into a single clean CSV"
    )
    parser.add_argument(
        "--datafiles-dir",
        default=str(Path(__file__).resolve().parents[2] / "data" / "input"),
        help="Directory containing Inventory_Snapshot_*.csv files (default: data/input)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[2] / "data" / "staged" / "inventory_clean.csv"),
        help="Output path for merged clean CSV (default: data/staged/inventory_clean.csv)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(cpu_count() or 4, 8),
        help="Number of parallel workers (default: min(cpu_count, 8))",
    )
    args = parser.parse_args()

    datafiles_dir = Path(args.datafiles_dir).resolve()
    output_path = Path(args.output).resolve()
    workers = args.workers

    # Find all inventory snapshot files, sorted by name (chronological)
    pattern = str(datafiles_dir / "Inventory_Snapshot_*.csv")
    source_files = sorted(glob.glob(pattern))

    if not source_files:
        logger.info(f"No Inventory_Snapshot_*.csv files found in {datafiles_dir}")
        sys.exit(1)

    logger.info(f"Found {len(source_files)} inventory snapshot file(s) in {datafiles_dir}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Workers: {workers}")
    logger.info()

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if workers > 1 and len(source_files) > 1:
        # --- Parallel path: each file → temp CSV, then merge ---
        tmp_dir = Path(tempfile.mkdtemp(prefix="inv_normalize_"))
        try:
            # Build (source, temp_output) pairs
            work_items: list[tuple[str, str]] = []
            temp_files: list[Path] = []
            for filepath in source_files:
                src = Path(filepath)
                tmp = tmp_dir / f"{src.stem}.csv"
                work_items.append((str(src), str(tmp)))
                temp_files.append(tmp)

            effective_workers = min(workers, len(source_files))
            logger.info(f"  Normalizing {len(source_files)} files with {effective_workers} workers...")

            with profiled_section("normalize_inventory_parallel"):
                with Pool(processes=effective_workers) as pool:
                    results = pool.map(_normalize_file_to_path, work_items)

            # Report per-file results
            total_rows = 0
            for filename, rows in results:
                logger.info(f"  {filename}: {rows:,} rows")
                total_rows += rows

            # Merge: header + concatenate temp files in sorted order
            logger.info(f"\n  Merging {len(temp_files)} temp files...")
            with profiled_section("merge_inventory_csvs"):
                with output_path.open("wb") as dst:
                    dst.write((",".join(OUTPUT_COLUMNS) + "\n").encode("utf-8"))
                    for tmp in temp_files:
                        if tmp.stat().st_size > 0:
                            with tmp.open("rb") as src_f:
                                shutil.copyfileobj(src_f, dst, length=16 * 1024 * 1024)

        finally:
            # Cleanup temp directory
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        # --- Sequential path (single worker or single file) ---
        total_rows = 0
        header_idx: dict[str, int] | None = None

        with output_path.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.writer(dst, lineterminator="\n")
            writer.writerow(OUTPUT_COLUMNS)

            for filepath in source_files:
                source_path = Path(filepath)
                logger.info(f"  Processing {source_path.name} ...", end=" ", flush=True)

                with profiled_section(f"normalize_{source_path.stem}"):
                    rows_written, header_idx = normalize_file(source_path, writer, header_idx)
                total_rows += rows_written

                logger.info(f"{rows_written:,} rows")

    logger.info()
    logger.info(f"Total rows written: {total_rows:,}")
    logger.info(f"Output file: {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
