import csv
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3] / "datafiles" / "itemdata.csv"
TARGET = Path(__file__).resolve().parents[1] / "data" / "itemdata_clean.csv"
TARGET.parent.mkdir(parents=True, exist_ok=True)

REQUIRED_FIELDS = [
    "item_no",
    "item_desc",
    "item_status",
    "brand_name",
    "category",
    "class",
    "sub_class",
    "country",
    "scm_rtd_flag",
    "size",
    "case_weight",
    "cpl",
    "cpp",
    "lpp",
    "case_weight_uom",
    "bpc",
    "bottle_pack",
    "pack_case",
    "item_proof",
    "upc",
    "national_service_model",
    "supplier_no",
    "supplier_name",
    "item_is_deleted",
    "producer_name",
]


def dedupe_headers(headers: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    out: list[str] = []
    for h in headers:
        key = h.strip()
        counts[key] = counts.get(key, 0) + 1
        out.append(key if counts[key] == 1 else f"{key}_{counts[key]}")
    return out


with SOURCE.open("r", encoding="utf-8-sig", newline="") as src:
    reader = csv.reader(src)
    raw_headers = next(reader)
    headers = dedupe_headers(raw_headers)

    missing = [c for c in REQUIRED_FIELDS if c not in headers]
    if missing:
        raise ValueError(f"Missing required columns in source: {missing}")

    with TARGET.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=REQUIRED_FIELDS)
        writer.writeheader()

        for row in reader:
            rec = {headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))}
            writer.writerow({k: rec.get(k, "") for k in REQUIRED_FIELDS})

print(f"Wrote normalized CSV: {TARGET}")
