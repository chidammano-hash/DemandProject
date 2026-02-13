import csv
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "itemdata_clean.csv"

DB = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname": os.getenv("POSTGRES_DB", "demand_mvp"),
    "user": os.getenv("POSTGRES_USER", "demand"),
    "password": os.getenv("POSTGRES_PASSWORD", "demand"),
}


def to_int(v: str):
    v = (v or "").strip()
    return int(v) if v else None


def to_num(v: str):
    v = (v or "").strip()
    return float(v) if v else None


insert_sql = """
INSERT INTO dim_item (
  item_ck, item_no, item_desc, item_status, brand_name, category, class, sub_class,
  country, scm_rtd_flag, size, case_weight, cpl, cpp, lpp, case_weight_uom, bpc,
  bottle_pack, pack_case, item_proof, upc, national_service_model, supplier_no,
  supplier_name, item_is_deleted, producer_name
)
VALUES (
  %(item_no)s, %(item_no)s, %(item_desc)s, %(item_status)s, %(brand_name)s, %(category)s, %(class)s, %(sub_class)s,
  %(country)s, %(scm_rtd_flag)s, %(size)s, %(case_weight)s, %(cpl)s, %(cpp)s, %(lpp)s, %(case_weight_uom)s,
  %(bpc)s, %(bottle_pack)s, %(pack_case)s, %(item_proof)s, %(upc)s, %(national_service_model)s,
  %(supplier_no)s, %(supplier_name)s, %(item_is_deleted)s, %(producer_name)s
)
ON CONFLICT (item_ck) DO UPDATE SET
  item_desc = EXCLUDED.item_desc,
  item_status = EXCLUDED.item_status,
  brand_name = EXCLUDED.brand_name,
  category = EXCLUDED.category,
  class = EXCLUDED.class,
  sub_class = EXCLUDED.sub_class,
  country = EXCLUDED.country,
  scm_rtd_flag = EXCLUDED.scm_rtd_flag,
  size = EXCLUDED.size,
  case_weight = EXCLUDED.case_weight,
  cpl = EXCLUDED.cpl,
  cpp = EXCLUDED.cpp,
  lpp = EXCLUDED.lpp,
  case_weight_uom = EXCLUDED.case_weight_uom,
  bpc = EXCLUDED.bpc,
  bottle_pack = EXCLUDED.bottle_pack,
  pack_case = EXCLUDED.pack_case,
  item_proof = EXCLUDED.item_proof,
  upc = EXCLUDED.upc,
  national_service_model = EXCLUDED.national_service_model,
  supplier_no = EXCLUDED.supplier_no,
  supplier_name = EXCLUDED.supplier_name,
  item_is_deleted = EXCLUDED.item_is_deleted,
  producer_name = EXCLUDED.producer_name,
  modified_ts = NOW();
"""

with CSV_PATH.open("r", encoding="utf-8", newline="") as f, psycopg.connect(**DB) as conn:
    reader = csv.DictReader(f)
    with conn.cursor() as cur:
        for row in reader:
            row["case_weight"] = to_num(row.get("case_weight"))
            row["cpl"] = to_int(row.get("cpl"))
            row["cpp"] = to_int(row.get("cpp"))
            row["lpp"] = to_int(row.get("lpp"))
            row["bpc"] = to_int(row.get("bpc"))
            row["bottle_pack"] = to_int(row.get("bottle_pack"))
            row["pack_case"] = to_int(row.get("pack_case"))
            row["item_proof"] = to_num(row.get("item_proof"))
            cur.execute(insert_sql, row)
    conn.commit()

print("Loaded dim_item into Postgres")
