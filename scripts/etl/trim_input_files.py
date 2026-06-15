"""Trim all input files to keep only rows for specified locations and sites.

Filters CSV/TXT files in data/input/ to retain only rows matching
target location values (default: 1401-BULK) and/or target site values
(default: 1). Files with a location column are filtered by location;
files with only a site column are filtered by site.
Overwrites files in-place using atomic temp-file swap.

Usage:
    python scripts/etl/trim_input_files.py
    python scripts/etl/trim_input_files.py --dry-run
    python scripts/etl/trim_input_files.py --locations 1401-BULK 3201-COLU
    python scripts/etl/trim_input_files.py --sites 1 100
"""

import argparse
import csv
import logging
import os
import shutil
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "input"

DEFAULT_LOCATIONS = {"1401-BULK"}
DEFAULT_SITES = {"1"}

# Map of filename pattern -> (delimiter, location_column_name)
# Column names are case-sensitive as they appear in headers.
FILE_CONFIG: dict[str, tuple[str, str]] = {
    "Inventory_Snapshot": (",", "loc"),
    "locationdata.csv": (",", "location_id"),
    "purchase_orders.csv": (",", "loc"),
    "sourcing.csv": (",", "loc"),
    "dfu.txt": ("|", "LOC"),
    "dfu_lvl2_hist.txt": ("|", "LOC"),
    "dfu_stat_fcst.txt": ("|", "loc"),
}

# Map of filename pattern -> (delimiter, site_column_name)
# For files that have a site column instead of a location column.
SITE_CONFIG: dict[str, tuple[str, str]] = {
    "_customer_demand.csv": (",", "site"),
    "customerdata.csv": (",", "site"),
}

# Files in samples/ subdirectory
SAMPLE_CONFIG: dict[str, tuple[str, str]] = {
    "open_pos_sample.csv": (",", "loc"),
    "po_receipts_sample.csv": (",", "loc"),
}

# Files without a location or site column — left untouched
SKIP_FILES = {"itemdata.csv", "suppliers_sample.csv", "cleanup_input.py"}


def _resolve_config(filename: str) -> tuple[str, str, str] | None:
    """Return (delimiter, filter_column, filter_type) or None to skip.

    filter_type is 'loc' for location-based or 'site' for site-based filtering.
    """
    if filename in SKIP_FILES:
        return None
    # Check location-based config first
    for pattern, cfg in FILE_CONFIG.items():
        if filename.startswith(pattern) or filename == pattern:
            return (cfg[0], cfg[1], "loc")
    # Check site-based config
    for pattern, cfg in SITE_CONFIG.items():
        if filename.endswith(pattern) or filename == pattern:
            return (cfg[0], cfg[1], "site")
    return None


def _filter_file(
    filepath: Path,
    delimiter: str,
    filter_column: str,
    keep_values: set[str],
    dry_run: bool,
) -> tuple[int, int]:
    """Filter a single file in-place. Returns (original_rows, kept_rows)."""
    original_count = 0
    kept_count = 0

    fd, tmp_path = tempfile.mkstemp(
        dir=filepath.parent, suffix=filepath.suffix, prefix=".trim_"
    )
    try:
        with (
            open(filepath, "r", newline="", encoding="utf-8") as fin,
            os.fdopen(fd, "w", newline="", encoding="utf-8") as fout,
        ):
            reader = csv.DictReader(fin, delimiter=delimiter)
            if filter_column not in (reader.fieldnames or []):
                logger.warning(
                    "Column %r not found in %s (has: %s) — skipping",
                    filter_column,
                    filepath.name,
                    reader.fieldnames,
                )
                os.unlink(tmp_path)
                return 0, 0

            writer = csv.DictWriter(
                fout,
                fieldnames=reader.fieldnames,
                delimiter=delimiter,
                lineterminator="\n",
            )
            writer.writeheader()

            for row in reader:
                original_count += 1
                if row.get(filter_column, "").strip() in keep_values:
                    writer.writerow(row)
                    kept_count += 1

        if not dry_run:
            shutil.move(tmp_path, filepath)
        else:
            os.unlink(tmp_path)
    except (OSError, ValueError):
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return original_count, kept_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trim input files to target locations and sites",
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        default=sorted(DEFAULT_LOCATIONS),
        help="Location codes to keep (default: 1401-BULK)",
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=sorted(DEFAULT_SITES),
        help="Site codes to keep (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts without modifying files",
    )
    args = parser.parse_args()
    keep_locations = set(args.locations)
    keep_sites = set(args.sites)

    logger.info("Target locations: %s", sorted(keep_locations))
    logger.info("Target sites: %s", sorted(keep_sites))
    logger.info("Input directory: %s", INPUT_DIR)
    if args.dry_run:
        logger.info("DRY RUN — no files will be modified")

    total_original = 0
    total_kept = 0
    files_processed = 0

    # Process top-level files
    for entry in sorted(INPUT_DIR.iterdir()):
        if entry.is_dir() or entry.name.startswith("."):
            continue
        cfg = _resolve_config(entry.name)
        if cfg is None:
            logger.info("SKIP  %s (no location/site column)", entry.name)
            continue

        delimiter, filter_col, filter_type = cfg
        keep_values = keep_sites if filter_type == "site" else keep_locations
        logger.info(
            "FILTER %s (col=%s, type=%s, delim=%r)",
            entry.name, filter_col, filter_type, delimiter,
        )
        orig, kept = _filter_file(
            entry, delimiter, filter_col, keep_values, args.dry_run,
        )
        logger.info(
            "       %s: %s → %s rows (%.1f%%)",
            entry.name,
            f"{orig:,}",
            f"{kept:,}",
            (kept / orig * 100) if orig else 0,
        )
        total_original += orig
        total_kept += kept
        files_processed += 1

    # Process samples/ subdirectory
    samples_dir = INPUT_DIR / "samples"
    if samples_dir.is_dir():
        for entry in sorted(samples_dir.iterdir()):
            if entry.name in SKIP_FILES or entry.name.startswith("."):
                continue
            cfg = SAMPLE_CONFIG.get(entry.name)
            if cfg is None:
                logger.info("SKIP  samples/%s (no location column)", entry.name)
                continue

            delimiter, loc_col = cfg
            logger.info(
                "FILTER samples/%s (col=%s, delim=%r)", entry.name, loc_col, delimiter
            )
            orig, kept = _filter_file(
                entry, delimiter, loc_col, keep_locations, args.dry_run
            )
            logger.info(
                "       samples/%s: %s → %s rows (%.1f%%)",
                entry.name,
                f"{orig:,}",
                f"{kept:,}",
                (kept / orig * 100) if orig else 0,
            )
            total_original += orig
            total_kept += kept
            files_processed += 1

    logger.info("=" * 60)
    logger.info(
        "TOTAL: %d files processed, %s → %s rows (%.1f%%)",
        files_processed,
        f"{total_original:,}",
        f"{total_kept:,}",
        (total_kept / total_original * 100) if total_original else 0,
    )
    if args.dry_run:
        logger.info("DRY RUN complete — no files were modified")


if __name__ == "__main__":
    main()
