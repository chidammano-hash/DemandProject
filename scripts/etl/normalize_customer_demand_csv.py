#!/usr/bin/env python
"""Normalize customer demand CSVs into a single clean CSV.

Source columns: site, warehouse_no, item_no, customer_no, posting_prd, demand_cases, oos_cases
Target columns: item_id, customer_no, site, location_id, startdate, demand_qty, sales_qty, oos_qty

Location resolution: source 'site' column maps to site_id in dim_location (from locationdata.csv).
Multiple location_ids can share the same site_id -- pick primary_demand_location='Y', else
first alphabetically. warehouse_no is kept for reference but not used in the lookup.

Source files may be yearly (2024_customer_demand.csv) or monthly (202601_customer_demand.csv).

Usage:
    uv run python scripts/etl/normalize_customer_demand_csv.py
    uv run python scripts/etl/normalize_customer_demand_csv.py --datafiles-dir data/input
    uv run python scripts/etl/normalize_customer_demand_csv.py --output data/staged/customer_demand_clean.csv
"""

import argparse
import csv
import glob
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.services.perf_profiler import profiled_section

logger = logging.getLogger(__name__)

# Null sentinel values -- consistent with normalize_dataset_csv.py
NULL_SENTINELS = {"", "null", "none", "na", "n/a"}

# Output column order
OUTPUT_COLUMNS = [
    "item_id",
    "customer_no",
    "site",
    "location_id",
    "startdate",
    "demand_qty",
    "sales_qty",
    "oos_qty",
]

# Source column names (lowercase)
SRC_SITE = "site"
SRC_WAREHOUSE_NO = "warehouse_no"
SRC_ITEM_NO = "item_no"
SRC_CUSTOMER_NO = "customer_no"
SRC_POSTING_PRD = "posting_prd"
SRC_DEMAND_CASES = "demand_cases"
SRC_OOS_CASES = "oos_cases"

EXPECTED_COLS = [
    SRC_SITE,
    SRC_WAREHOUSE_NO,
    SRC_ITEM_NO,
    SRC_CUSTOMER_NO,
    SRC_POSTING_PRD,
    SRC_DEMAND_CASES,
    SRC_OOS_CASES,
]


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


def parse_posting_period(value: str) -> str:
    """Convert YYYYMM integer/string to YYYY-MM-01 date string.

    Returns empty string if the value cannot be parsed.
    """
    s = value.strip()
    if is_null(s):
        return ""
    # Remove any decimals (e.g., 202601.0)
    if "." in s:
        s = s.split(".")[0]
    if len(s) != 6 or not s.isdigit():
        return ""
    year = int(s[:4])
    month = int(s[4:6])
    if year < 2000 or year > 2099 or month < 1 or month > 12:
        return ""
    return f"{year:04d}-{month:02d}-01"


def build_location_map(location_file: Path) -> dict[str, str]:
    """Build site_id -> location_id mapping from location CSV.

    Prefers rows with primary_demand_location='Y'. If no primary exists
    for a site_id, picks the first location_id alphabetically.

    Returns dict keyed by site_id (as string).
    """
    # Collect all (location_id, is_primary) per site_id
    site_locations: dict[str, list[tuple[str, bool]]] = {}

    with location_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            site_id = row.get("site_id", "").strip()
            location_id = row.get("location_id", "").strip()
            primary = row.get("primary_demand_location", "").strip().upper()

            if not site_id or not location_id:
                continue

            is_primary = primary == "Y"
            if site_id not in site_locations:
                site_locations[site_id] = []
            site_locations[site_id].append((location_id, is_primary))

    # Resolve: prefer primary, else first alphabetically
    result: dict[str, str] = {}
    for site_id, locations in site_locations.items():
        primaries = [loc for loc, is_p in locations if is_p]
        if primaries:
            result[site_id] = sorted(primaries)[0]
        else:
            result[site_id] = sorted(loc for loc, _ in locations)[0]

    return result


def normalize_file(
    source_path: Path,
    writer: csv.writer,
    location_map: dict[str, str],
) -> tuple[int, int]:
    """Process a single customer demand CSV file, streaming rows to writer.

    Returns (rows_written, rows_skipped).
    """
    rows_written = 0
    rows_skipped = 0

    with source_path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.reader(src)
        raw_headers = next(reader)
        header_idx = {
            h.strip().lower().replace("\r", ""): i
            for i, h in enumerate(raw_headers)
        }

        missing = [c for c in EXPECTED_COLS if c not in header_idx]
        if missing:
            logger.warning("Missing columns %s in %s, skipping file", missing, source_path.name)
            return 0, 0

        idx_site = header_idx[SRC_SITE]
        idx_item = header_idx[SRC_ITEM_NO]
        idx_customer = header_idx[SRC_CUSTOMER_NO]
        idx_posting = header_idx[SRC_POSTING_PRD]
        idx_demand = header_idx[SRC_DEMAND_CASES]
        idx_oos = header_idx[SRC_OOS_CASES]

        for row in reader:
            row = [cell.replace("\r", "") for cell in row]

            if len(row) < len(EXPECTED_COLS):
                rows_skipped += 1
                continue

            item_id = row[idx_item].strip()
            if is_null(item_id):
                rows_skipped += 1
                continue
            if "." in item_id:
                item_id = item_id.split(".")[0]

            customer_no = row[idx_customer].strip()
            if is_null(customer_no):
                rows_skipped += 1
                continue
            if "." in customer_no:
                customer_no = customer_no.split(".")[0]

            site = row[idx_site].strip()
            if is_null(site):
                rows_skipped += 1
                continue
            if "." in site:
                site = site.split(".")[0]

            location_id = location_map.get(site)
            if not location_id:
                rows_skipped += 1
                continue

            startdate = parse_posting_period(row[idx_posting])
            if not startdate:
                rows_skipped += 1
                continue

            demand_qty = max(0.0, parse_float(row[idx_demand], default=0.0))
            oos_qty = max(0.0, parse_float(row[idx_oos], default=0.0))
            sales_qty = max(0.0, demand_qty - oos_qty)

            writer.writerow([
                item_id, customer_no, site, location_id, startdate,
                f"{demand_qty:.4f}", f"{sales_qty:.4f}", f"{oos_qty:.4f}",
            ])
            rows_written += 1

    return rows_written, rows_skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize customer demand CSVs into a single clean CSV",
    )
    parser.add_argument(
        "--datafiles-dir",
        default=str(ROOT / "data" / "input"),
        help="Directory containing *_customer_demand.csv files (default: data/input)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "staged" / "customer_demand_clean.csv"),
        help="Output path for clean CSV (default: data/staged/customer_demand_clean.csv)",
    )
    parser.add_argument(
        "--location-file",
        default=str(ROOT / "data" / "input" / "locationdata.csv"),
        help="Path to locationdata.csv for warehouse->location mapping",
    )
    args = parser.parse_args()

    datafiles_dir = Path(args.datafiles_dir).resolve()
    output_path = Path(args.output).resolve()
    location_file = Path(args.location_file).resolve()

    # --- Load location mapping ---
    if not location_file.exists():
        logger.error("Location file not found: %s", location_file)
        sys.exit(1)

    with profiled_section("build_location_map"):
        location_map = build_location_map(location_file)
    logger.info("Loaded %d site_id -> location_id mappings", len(location_map))

    # --- Discover source files ---
    pattern = str(datafiles_dir / "*_customer_demand.csv")
    source_files = sorted(glob.glob(pattern))

    if not source_files:
        logger.error(
            "No *_customer_demand.csv files found in %s", datafiles_dir
        )
        sys.exit(1)

    logger.info(
        "Found %d customer demand file(s) in %s",
        len(source_files),
        datafiles_dir,
    )
    logger.info("Output: %s", output_path)

    # --- Ensure output directory exists ---
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Stream-process files directly to output CSV ---
    total_written = 0
    total_skipped = 0

    with output_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.writer(dst, lineterminator="\n")
        writer.writerow(OUTPUT_COLUMNS)

        for filepath in source_files:
            source_path = Path(filepath)
            logger.info("Processing %s ...", source_path.name)

            with profiled_section(f"normalize_{source_path.stem}"):
                written, skipped = normalize_file(source_path, writer, location_map)

            total_written += written
            total_skipped += skipped
            logger.info(
                "  %s: %s rows written, %s skipped",
                source_path.name, f"{written:,}", f"{skipped:,}",
            )

    logger.info("Total rows written: %s", f"{total_written:,}")
    logger.info("Total rows skipped (no location match or bad data): %s", f"{total_skipped:,}")
    logger.info("Files processed: %d", len(source_files))
    logger.info("Output file: %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
