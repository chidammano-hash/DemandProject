CREATE TABLE IF NOT EXISTS dim_dfu (
  dfu_sk BIGSERIAL PRIMARY KEY,
  dfu_ck TEXT UNIQUE NOT NULL,
  dmdunit TEXT NOT NULL,
  dmdgroup TEXT,
  loc TEXT NOT NULL,
  brand TEXT,
  abc_vol TEXT,
  brand_desc TEXT,
  ded_div_sw INTEGER,
  execution_lag INTEGER,
  otc_status TEXT,
  premise TEXT,
  prod_subgrp_desc TEXT,
  region TEXT,
  service_lvl_grp TEXT,
  size TEXT,
  state_plan TEXT,
  supergroup TEXT,
  supplier_desc TEXT,
  total_lt INTEGER,
  vintage INTEGER,
  sales_div TEXT,
  purge_sw INTEGER,
  alcoh_pct NUMERIC(10,4),
  bot_type_desc TEXT,
  brand_size TEXT,
  cnty TEXT,
  dom_imp_opt TEXT,
  grape_vrty_desc TEXT,
  material TEXT,
  prod_cat_desc TEXT,
  producer_desc TEXT,
  proof NUMERIC(10,4),
  subclass_desc TEXT,
  prod_class_desc TEXT,
  file_dt TEXT,
  histstart TEXT,
  cluster_assignment TEXT,
  sop_ref TEXT,
  load_ts TIMESTAMPTZ DEFAULT NOW(),
  modified_ts TIMESTAMPTZ DEFAULT NOW()
);

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'dim_dfu' AND column_name = 'u_abc_vol'
  ) AND NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'dim_dfu' AND column_name = 'abc_vol'
  ) THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_abc_vol TO abc_vol;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_brand_desc')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'brand_desc') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_brand_desc TO brand_desc;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_ded_div_sw')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'ded_div_sw') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_ded_div_sw TO ded_div_sw;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_execution_lag')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'execution_lag') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_execution_lag TO execution_lag;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_otc_status')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'otc_status') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_otc_status TO otc_status;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_premise')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'premise') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_premise TO premise;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_prod_subgrp_desc')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'prod_subgrp_desc') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_prod_subgrp_desc TO prod_subgrp_desc;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_region')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'region') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_region TO region;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_service_lvl_grp')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'service_lvl_grp') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_service_lvl_grp TO service_lvl_grp;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_size')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'size') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_size TO size;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_state_plan')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'state_plan') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_state_plan TO state_plan;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_supergroup')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'supergroup') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_supergroup TO supergroup;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_supplier_desc')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'supplier_desc') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_supplier_desc TO supplier_desc;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_total_lt')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'total_lt') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_total_lt TO total_lt;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_vintage')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'vintage') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_vintage TO vintage;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_sales_div')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'sales_div') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_sales_div TO sales_div;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_purge_sw')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'purge_sw') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_purge_sw TO purge_sw;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_alcoh_pct')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'alcoh_pct') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_alcoh_pct TO alcoh_pct;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_bot_type_desc')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'bot_type_desc') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_bot_type_desc TO bot_type_desc;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_brand_size')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'brand_size') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_brand_size TO brand_size;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_cnty')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'cnty') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_cnty TO cnty;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_dom_imp_opt')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'dom_imp_opt') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_dom_imp_opt TO dom_imp_opt;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_grape_vrty_desc')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'grape_vrty_desc') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_grape_vrty_desc TO grape_vrty_desc;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_material')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'material') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_material TO material;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_prod_cat_desc')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'prod_cat_desc') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_prod_cat_desc TO prod_cat_desc;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_producer_desc')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'producer_desc') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_producer_desc TO producer_desc;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_proof')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'proof') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_proof TO proof;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_subclass_desc')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'subclass_desc') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_subclass_desc TO subclass_desc;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_prod_class_desc')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'prod_class_desc') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_prod_class_desc TO prod_class_desc;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_cluster_assignment')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'cluster_assignment') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_cluster_assignment TO cluster_assignment;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'u_sop_ref')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'dim_dfu' AND column_name = 'sop_ref') THEN
    ALTER TABLE dim_dfu RENAME COLUMN u_sop_ref TO sop_ref;
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_dim_dfu_dmdunit ON dim_dfu (dmdunit);
CREATE INDEX IF NOT EXISTS idx_dim_dfu_loc ON dim_dfu (loc);
CREATE INDEX IF NOT EXISTS idx_dim_dfu_brand ON dim_dfu (brand);
CREATE INDEX IF NOT EXISTS idx_dim_dfu_region ON dim_dfu (region);
CREATE INDEX IF NOT EXISTS idx_dim_dfu_cluster_assignment ON dim_dfu (cluster_assignment);
