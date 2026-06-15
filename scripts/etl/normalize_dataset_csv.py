import argparse
import csv
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.domain_specs import DOMAIN_SPECS, get_spec
from common.core.etl_helpers import dfu_key_for_row, load_valid_dfu_keys
from common.services.perf_profiler import profiled_section

# Domains whose rows are filtered against dim_sku at normalize time (US8).
# Inventory is normalized by normalize_inventory_csv.py and keeps the load-time
# DFU net (its writer runs in parallel worker subprocesses).
_NORMALIZE_DFU_DOMAINS = {"sales", "forecast"}


def dedupe_headers(headers: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    out: list[str] = []
    for h in headers:
        key = h.strip()
        counts[key] = counts.get(key, 0) + 1
        out.append(key if counts[key] == 1 else f"{key}_{counts[key]}")
    return out


def month_end(d: date) -> date:
    if d.month == 12:
        return date(d.year, 12, 31)
    return date(d.year, d.month + 1, 1) - timedelta(days=1)


def quarter_start(d: date) -> date:
    q_month = ((d.month - 1) // 3) * 3 + 1
    return date(d.year, q_month, 1)


def quarter_end(d: date) -> date:
    qs = quarter_start(d)
    if qs.month == 10:
        return date(qs.year, 12, 31)
    return date(qs.year, qs.month + 3, 1) - timedelta(days=1)


def write_time_csv(target: Path, columns: list[str]) -> None:
    start = date(2020, 1, 1)
    end = date(2035, 12, 31)
    one_day = timedelta(days=1)

    with target.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=columns, lineterminator="\n")
        writer.writeheader()

        d = start
        while d <= end:
            iso_year, iso_week, _ = d.isocalendar()
            wk_start = d - timedelta(days=d.isoweekday() - 1)
            wk_end = wk_start + timedelta(days=6)
            m_end = month_end(d)
            q_num = ((d.month - 1) // 3) + 1
            q_start = quarter_start(d)
            q_end = quarter_end(d)
            y_start = date(d.year, 1, 1)
            y_end = date(d.year, 12, 31)

            writer.writerow(
                {
                    "date_key": d.isoformat(),
                    "day_name": d.strftime("%A"),
                    "day_of_week": d.isoweekday(),
                    "day_of_month": d.day,
                    "day_of_year": d.timetuple().tm_yday,
                    "iso_week_year": iso_year,
                    "iso_week": iso_week,
                    "week_start_date": wk_start.isoformat(),
                    "week_end_date": wk_end.isoformat(),
                    "month_number": d.month,
                    "month_name": d.strftime("%B"),
                    "month_start_date": date(d.year, d.month, 1).isoformat(),
                    "month_end_date": m_end.isoformat(),
                    "quarter_number": q_num,
                    "quarter_label": f"Q{q_num}",
                    "quarter_start_date": q_start.isoformat(),
                    "quarter_end_date": q_end.isoformat(),
                    "year_number": d.year,
                    "year_start_date": y_start.isoformat(),
                    "year_end_date": y_end.isoformat(),
                    "week_bucket": f"{iso_year}-W{iso_week:02d}",
                    "month_bucket": f"{d.year:04d}-{d.month:02d}",
                    "quarter_bucket": f"{d.year:04d}-Q{q_num}",
                    "year_bucket": f"{d.year:04d}",
                }
            )
            d += one_day


def to_iso_date_yyyymmdd(v: str, require_month_start: bool = False) -> str:
    s = (v or "").strip()
    if not s:
        return ""
    # Handle YYYY-MM-DD (ISO) format
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        try:
            parsed = date.fromisoformat(s)
            if require_month_start and parsed.day != 1:
                return ""
            return parsed.isoformat()
        except ValueError:
            return ""
    # Handle YYYYMMDD (compact) format
    if len(s) != 8 or not s.isdigit():
        return ""
    if require_month_start and s[6:8] != "01":
        return ""
    yyyy, mm, dd = int(s[0:4]), int(s[4:6]), int(s[6:8])
    try:
        parsed = date(yyyy, mm, dd)
    except ValueError:
        return ""
    return parsed.isoformat()


def to_iso_month_start(v: str) -> str:
    s = (v or "").strip()
    if not s:
        return ""
    if len(s) == 10:
        try:
            d = date.fromisoformat(s)
        except ValueError:
            return ""
        return d.isoformat() if d.day == 1 else ""
    return to_iso_date_yyyymmdd(s, require_month_start=True)


def month_diff(start_iso: str, fcst_iso: str) -> int | None:
    try:
        start = date.fromisoformat(start_iso)
        fcst = date.fromisoformat(fcst_iso)
    except ValueError:
        return None
    return (start.year - fcst.year) * 12 + (start.month - fcst.month)


def to_int_string(v: str) -> str:
    s = (v or "").strip()
    if not s:
        return ""
    try:
        return str(int(float(s)))
    except ValueError:
        return ""


def main() -> None:
    allowed = ", ".join(sorted(DOMAIN_SPECS))
    parser = argparse.ArgumentParser(description="Normalize source file for a given dataset (dimension or fact)")
    parser.add_argument("--dataset", required=True, help=allowed)
    parser.add_argument("--source-dir", default=None, help="Override source data directory")
    args = parser.parse_args()

    spec = get_spec(args.dataset)

    if args.source_dir:
        source = Path(args.source_dir).resolve() / spec.source_file
    else:
        source = Path(__file__).resolve().parents[2] / "data" / "input" / spec.source_file
    target = Path(__file__).resolve().parents[2] / "data" / spec.clean_file
    target.parent.mkdir(parents=True, exist_ok=True)

    if spec.name == "time":
        write_time_csv(target, spec.columns)
        print(f"Wrote normalized CSV for {spec.name}: {target}")
        return

    with profiled_section("read_source_csv"):
        with source.open("r", encoding="utf-8-sig", newline="") as src:
            reader = csv.reader(src, delimiter=spec.source_delimiter)
            raw_headers = next(reader)
            headers = dedupe_headers(raw_headers)
            header_idx = {h.strip().lower(): i for i, h in enumerate(headers)}

            source_keys = {c: spec.source_col_for(c).strip().lower() for c in spec.columns}
            present_cols = [c for c in spec.columns if source_keys[c] in header_idx]
            absent_cols = [c for c in spec.columns if source_keys[c] not in header_idx]

            source_rows = list(reader)

    # US8: when dim_sku is already populated (e.g. incremental refresh), drop
    # rows with no matching DFU here so the load step avoids a post-COPY DELETE
    # full-table scan. On a cold DB dim_sku is empty -> valid_keys is None and we
    # fall back to the load-time DFU filter (no rows dropped here).
    valid_keys = (
        load_valid_dfu_keys(spec.name)
        if spec.name in _NORMALIZE_DFU_DOMAINS
        else None
    )
    dfu_dropped = 0

    with profiled_section("normalize_and_write"):
        with target.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=spec.columns, lineterminator="\n")
            writer.writeheader()

            for row in source_rows:
                out = {
                    c: row[header_idx[source_keys[c]]] if header_idx[source_keys[c]] < len(row) else ""
                    for c in present_cols
                }
                for c in absent_cols:
                    out[c] = ""

                if spec.name == "sales":
                    if out.get("type", "").strip() != "1":
                        continue
                    out["startdate"] = to_iso_date_yyyymmdd(
                        out.get("startdate", ""),
                        require_month_start=True,
                    )
                    if not out["startdate"]:
                        continue
                    out["file_dt"] = to_iso_date_yyyymmdd(out.get("file_dt", ""))
                    out["type"] = out.get("type", "").strip()

                if spec.name == "forecast":
                    fcst_iso = to_iso_month_start(out.get("fcstdate", ""))
                    start_iso = to_iso_month_start(out.get("startdate", ""))
                    if not fcst_iso or not start_iso:
                        continue
                    lag = month_diff(start_iso, fcst_iso)
                    if lag is None or lag < 0 or lag > 4:
                        continue
                    out["fcstdate"] = fcst_iso
                    out["startdate"] = start_iso
                    out["lag"] = str(lag)
                    out["execution_lag"] = to_int_string(out.get("execution_lag", "")) or str(lag)
                    out["model_id"] = (out.get("model_id", "").strip() or "external")

                if spec.name == "sourcing":
                    src_cd = (out.get("source_cd", "") or "").strip()
                    parts = src_cd.split("-", 1)
                    out["supplier_id"] = parts[0] if parts else ""
                    out["plant_id"] = parts[1] if len(parts) > 1 else ""

                if spec.name == "purchase_order":
                    for dt_col in ("delivery_date", "original_delivery_date",
                                   "current_ship_date", "original_ship_date"):
                        out[dt_col] = to_iso_date_yyyymmdd(out.get(dt_col, ""))

                if valid_keys is not None and dfu_key_for_row(out, spec.name) not in valid_keys:
                    dfu_dropped += 1
                    continue

                writer.writerow(out)

    if dfu_dropped:
        print(f"DFU filter (normalize-time): dropped {dfu_dropped:,} {spec.name} "
              f"rows with no matching DFU in dim_sku")
    print(f"Wrote normalized CSV for {spec.name}: {target}")


if __name__ == "__main__":
    main()
