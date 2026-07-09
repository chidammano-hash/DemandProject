"""Pre-filter source files before normalization/loading.

- dfu.txt: remove rows where U_CLUSTER_ASSIGNMENT starts with L3_ and optionally
  keep only one LOC
- dfu_lvl2_hist.txt: keep only rows where U_LVL == 121
- dfu_stat_fcst.txt: keep only rows matching DFUs in dfu.txt + last 12 months
- Inventory_Snapshot_*.csv: keep only rows matching (item, loc) in dfu.txt and
  optionally keep only one LOC
"""

import argparse
import glob
import sys
from datetime import date
from pathlib import Path

DATA_DIR = Path(__file__).parent
DFU_FILE = DATA_DIR / "dfu.txt"
HIST_FILE = DATA_DIR / "dfu_lvl2_hist.txt"
FCST_FILE = DATA_DIR / "dfu_stat_fcst.txt"
INV_PATTERN = str(DATA_DIR / "Inventory_Snapshot_*.csv")


def filter_dfu(keep_loc: str | None = None):
    """Apply cluster and optional location filters to ``dfu.txt``."""
    lines = DFU_FILE.read_text().splitlines(keepends=True)
    header = lines[0]

    cols = header.strip().split("|")
    try:
        idx = cols.index("U_CLUSTER_ASSIGNMENT")
    except ValueError:
        print("ERROR: U_CLUSTER_ASSIGNMENT column not found in header")
        sys.exit(1)
    loc_idx = _find_col(cols, "LOC") if keep_loc is not None else -1

    kept = [header]
    removed_cluster = 0
    removed_loc = 0
    for line in lines[1:]:
        fields = line.strip().split("|")
        if len(fields) > idx and fields[idx].startswith("L3_"):
            removed_cluster += 1
            continue
        if keep_loc is not None and not (len(fields) > loc_idx and fields[loc_idx] == keep_loc):
            removed_loc += 1
            continue
        kept.append(line)

    DFU_FILE.write_text("".join(kept))
    msg = f"[dfu.txt] Removed {removed_cluster} rows with L3_* cluster assignments"
    if keep_loc is not None:
        msg += f", {removed_loc} rows (LOC != {keep_loc})"
    print(msg + ".")
    print(f"[dfu.txt] Remaining rows: {len(kept) - 1}")


def filter_hist(keep_loc: str | None = None):
    """Keep only rows where U_LVL == 121 in dfu_lvl2_hist.txt.

    When keep_loc is set, additionally keep only rows whose LOC equals it.
    """
    lines = HIST_FILE.read_text().splitlines(keepends=True)
    header = lines[0]

    cols = header.strip().split("|")
    try:
        idx = cols.index("U_LVL")
    except ValueError:
        print("ERROR: U_LVL column not found in header")
        sys.exit(1)
    loc_idx = _find_col(cols, "LOC") if keep_loc is not None else -1

    kept = [header]
    removed_lvl = 0
    removed_loc = 0
    for line in lines[1:]:
        fields = line.strip().split("|")
        if not (len(fields) > idx and fields[idx] == "121"):
            removed_lvl += 1
            continue
        if keep_loc is not None and not (len(fields) > loc_idx and fields[loc_idx] == keep_loc):
            removed_loc += 1
            continue
        kept.append(line)

    HIST_FILE.write_text("".join(kept))
    msg = f"[dfu_lvl2_hist.txt] Removed {removed_lvl} rows (U_LVL != 121)"
    if keep_loc is not None:
        msg += f", {removed_loc} rows (LOC != {keep_loc})"
    print(msg + ".")
    print(f"[dfu_lvl2_hist.txt] Remaining rows: {len(kept) - 1}")


def _find_col(header: list[str], name: str) -> int:
    """Find column index case-insensitively."""
    lower = name.lower()
    for i, col in enumerate(header):
        if col.strip().lower() == lower:
            return i
    raise ValueError(f"column '{name}' not found in header: {header[:10]}")


def _build_dfu_keys() -> set[str]:
    """Build set of valid DFU keys (dmdunit|dmdgroup|loc) from dfu.txt."""
    if not DFU_FILE.exists():
        return set()

    lines = DFU_FILE.read_text().splitlines()
    header = lines[0].strip().split("|")
    i_unit = _find_col(header, "DMDUNIT")
    i_group = _find_col(header, "DMDGROUP")
    i_loc = _find_col(header, "LOC")

    keys: set[str] = set()
    for line in lines[1:]:
        fields = line.strip().split("|")
        if len(fields) > max(i_unit, i_group, i_loc):
            keys.add(f"{fields[i_unit]}|{fields[i_group]}|{fields[i_loc]}")
    print(f"[dfu.txt] Built {len(keys):,} unique DFU keys")
    return keys


def _build_item_loc_keys() -> set[str]:
    """Build set of valid (item, loc) pairs from dfu.txt for inventory filtering."""
    if not DFU_FILE.exists():
        return set()

    lines = DFU_FILE.read_text().splitlines()
    header = lines[0].strip().split("|")
    i_unit = _find_col(header, "DMDUNIT")
    i_loc = _find_col(header, "LOC")

    keys: set[str] = set()
    for line in lines[1:]:
        fields = line.strip().split("|")
        if len(fields) > max(i_unit, i_loc):
            keys.add(f"{fields[i_unit]}|{fields[i_loc]}")
    return keys


def filter_inventory(keep_loc: str | None = None):
    """Filter inventory rows by valid DFU pairs and optional location."""
    inv_files = sorted(glob.glob(INV_PATTERN))
    if not inv_files:
        print("Skipping: no Inventory_Snapshot_*.csv files found")
        return

    item_loc_keys = _build_item_loc_keys()
    if not item_loc_keys:
        print("WARNING: no item/loc keys found — skipping inventory filter")
        return

    print(f"[inventory] Built {len(item_loc_keys):,} unique (item, loc) keys from dfu.txt")

    total_removed = 0
    total_kept = 0
    for fpath in inv_files:
        fname = Path(fpath).name
        lines = Path(fpath).read_text().splitlines(keepends=True)
        if not lines:
            continue

        header = lines[0]
        cols = header.strip().split(",")
        try:
            i_item = _find_col(cols, "item")
            i_loc = _find_col(cols, "loc")
        except ValueError:
            print(f"  [{fname}] WARNING: missing item/loc columns — skipping")
            continue

        kept = [header]
        removed_dfu = 0
        removed_loc = 0
        for line in lines[1:]:
            fields = line.strip().split(",")
            if len(fields) > max(i_item, i_loc):
                if keep_loc is not None and fields[i_loc] != keep_loc:
                    removed_loc += 1
                    continue
                key = f"{fields[i_item]}|{fields[i_loc]}"
                if key in item_loc_keys:
                    kept.append(line)
                else:
                    removed_dfu += 1
            else:
                removed_dfu += 1

        Path(fpath).write_text("".join(kept))
        removed = removed_dfu + removed_loc
        location_msg = f", removed {removed_loc:,} (loc != {keep_loc})" if keep_loc is not None else ""
        print(f"  [{fname}] Kept {len(kept) - 1:,}, removed {removed_dfu:,} invalid DFUs{location_msg}")
        total_removed += removed
        total_kept += len(kept) - 1

    print(f"[inventory] Total: kept {total_kept:,}, removed {total_removed:,} across {len(inv_files)} files")


def filter_forecast(keep_loc: str | None = None):
    """Keep only forecast rows matching DFUs in dfu.txt + last 12 months.

    When keep_loc is set, additionally keep only rows whose loc equals it.
    """
    dfu_keys = _build_dfu_keys()
    if not dfu_keys:
        print("WARNING: no DFU keys found — skipping forecast filter")
        return

    lines = FCST_FILE.read_text().splitlines(keepends=True)
    header = lines[0]
    cols = header.strip().split("|")
    i_unit = _find_col(cols, "dmdunit")
    i_group = _find_col(cols, "dmdgroup")
    i_loc = _find_col(cols, "loc")
    i_start = _find_col(cols, "startdate")

    # 12-month cutoff: first day of same month last year
    today = date.today()
    cutoff = date(today.year - 1, today.month, 1).isoformat()

    kept = [header]
    removed_dfu = 0
    removed_date = 0
    removed_loc = 0
    for line in lines[1:]:
        fields = line.strip().split("|")
        if len(fields) <= max(i_unit, i_group, i_loc, i_start):
            removed_dfu += 1
            continue

        if keep_loc is not None and fields[i_loc] != keep_loc:
            removed_loc += 1
            continue

        key = f"{fields[i_unit]}|{fields[i_group]}|{fields[i_loc]}"
        if key not in dfu_keys:
            removed_dfu += 1
            continue

        # startdate is YYYY-MM-DD — string compare works for ISO dates
        if fields[i_start] < cutoff:
            removed_date += 1
            continue

        kept.append(line)

    FCST_FILE.write_text("".join(kept))
    msg = (f"[dfu_stat_fcst.txt] Removed {removed_dfu:,} rows (no DFU match), "
           f"{removed_date:,} rows (before {cutoff})")
    if keep_loc is not None:
        msg += f", {removed_loc:,} rows (loc != {keep_loc})"
    print(msg + ".")
    print(f"[dfu_stat_fcst.txt] Remaining rows: {len(kept) - 1:,}")


ALL_FILTERS = ("dfu", "hist", "fcst", "inventory")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files",
        nargs="+",
        choices=ALL_FILTERS,
        default=list(ALL_FILTERS),
        help="Which filters to run (default: all).",
    )
    parser.add_argument(
        "--loc",
        default=None,
        help="If set, keep only rows with this LOC value in all four input file families.",
    )
    args = parser.parse_args()
    selected = set(args.files)

    if "dfu" in selected:
        if DFU_FILE.exists():
            filter_dfu(keep_loc=args.loc)
        else:
            print(f"Skipping: {DFU_FILE} not found")

    if "hist" in selected:
        if HIST_FILE.exists():
            filter_hist(keep_loc=args.loc)
        else:
            print(f"Skipping: {HIST_FILE} not found")

    if "fcst" in selected:
        if FCST_FILE.exists():
            filter_forecast(keep_loc=args.loc)
        else:
            print(f"Skipping: {FCST_FILE} not found")

    if "inventory" in selected:
        filter_inventory(keep_loc=args.loc)


if __name__ == "__main__":
    main()
