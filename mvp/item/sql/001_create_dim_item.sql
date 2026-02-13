CREATE TABLE IF NOT EXISTS dim_item (
  item_sk BIGSERIAL PRIMARY KEY,
  item_ck TEXT UNIQUE NOT NULL,
  item_no TEXT NOT NULL,
  item_desc TEXT NOT NULL,
  item_status TEXT NOT NULL,
  brand_name TEXT NOT NULL,
  category TEXT NOT NULL,
  class TEXT NOT NULL,
  sub_class TEXT NOT NULL,
  country TEXT NOT NULL,
  scm_rtd_flag TEXT,
  size TEXT,
  case_weight NUMERIC(12,4),
  cpl INTEGER,
  cpp INTEGER,
  lpp INTEGER,
  case_weight_uom TEXT,
  bpc INTEGER,
  bottle_pack INTEGER,
  pack_case INTEGER,
  item_proof NUMERIC(10,4),
  upc TEXT,
  national_service_model TEXT,
  supplier_no TEXT,
  supplier_name TEXT,
  item_is_deleted TEXT,
  producer_name TEXT,
  load_ts TIMESTAMPTZ DEFAULT NOW(),
  modified_ts TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dim_item_brand ON dim_item (brand_name);
CREATE INDEX IF NOT EXISTS idx_dim_item_category ON dim_item (category);
CREATE INDEX IF NOT EXISTS idx_dim_item_status ON dim_item (item_status);
